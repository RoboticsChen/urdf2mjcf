"""Export per-body meshes from an MJCF file to OBJ/MTL pairs.

Each ``body`` element in the MJCF ``worldbody`` hierarchy is exported as an
OBJ file whose geometry aggregates all mesh-type ``geom`` descendants defined
directly under that body. Geoms are transformed into the body's frame, their
meshes are merged, and the resulting OBJ references an accompanying MTL file
describing the materials used by those geoms.

Example
-------
python mjcf2obj.py path/to/model.xml out_dir

This script follows the conventions used in ``urdf2mjcf.postprocess`` modules
for parsing MJCF files and relies on ``pymeshlab`` for mesh IO.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
	import pymeshlab
except ImportError as exc:  # pragma: no cover - dependency error path
	raise SystemExit("pymeshlab is required to run mjcf2obj.py") from exc


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing utilities


def _parse_floats(
	value: str | None,
	expected: int | None = None,
	*,
	default: Iterable[float] | None = None,
) -> np.ndarray:
	"""Parse a whitespace separated float sequence."""

	if value is None:
		if default is None:
			raise ValueError("Expected float sequence but value is None")
		elems = list(default)
	else:
		elems = [float(x) for x in value.strip().split()]

	if expected is not None and len(elems) != expected:
		raise ValueError(f"Expected {expected} floats, got {len(elems)} from '{value}'")

	return np.asarray(elems, dtype=float)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
	"""Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""

	if quat.shape[0] != 4:
		raise ValueError("Quaternion must have 4 components")

	w, x, y, z = quat
	n = w * w + x * x + y * y + z * z
	if n == 0.0:
		return np.eye(3)
	s = 2.0 / n
	wx, wy, wz = s * w * x, s * w * y, s * w * z
	xx, xy, xz = s * x * x, s * x * y, s * x * z
	yy, yz, zz = s * y * y, s * y * z, s * z * z

	return np.array(
		[
			[1.0 - (yy + zz), xy - wz, xz + wy],
			[xy + wz, 1.0 - (xx + zz), yz - wx],
			[xz - wy, yz + wx, 1.0 - (xx + yy)],
		]
	)


def _euler_to_matrix(euler: np.ndarray) -> np.ndarray:
	"""Convert extrinsic XYZ Euler angles (radians) to rotation matrix."""

	if euler.shape[0] != 3:
		raise ValueError("Euler angles must have 3 components")

	x, y, z = euler
	cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
	sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)

	# Rz * Ry * Rx (extrinsic XYZ)
	return np.array(
		[
			[cy * cz, -cy * sz, sy],
			[sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
			[-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
		]
	)


def _axisangle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
	"""Convert axis-angle (axis_x, axis_y, axis_z, angle) to rotation matrix."""

	if axis_angle.shape[0] != 4:
		raise ValueError("Axis-angle representation must have 4 components")

	axis = axis_angle[:3]
	angle = axis_angle[3]
	norm = np.linalg.norm(axis)
	if norm == 0.0:
		return np.eye(3)
	axis = axis / norm
	x, y, z = axis
	c = math.cos(angle)
	s = math.sin(angle)
	C = 1.0 - c

	return np.array(
		[
			[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
			[y * x * C + z * s, c + y * y * C, y * z * C - x * s],
			[z * x * C - y * s, z * y * C + x * s, c + z * z * C],
		]
	)


def _rotation_from_attributes(attrs: Dict[str, str]) -> np.ndarray:
	"""Determine rotation matrix from MJCF element attributes."""

	if "quat" in attrs:
		return _quat_to_matrix(_parse_floats(attrs.get("quat"), expected=4))
	if "euler" in attrs:
		return _euler_to_matrix(_parse_floats(attrs.get("euler"), expected=3))
	if "axisangle" in attrs:
		return _axisangle_to_matrix(_parse_floats(attrs.get("axisangle"), expected=4))
	return np.eye(3)


def _compose_transform(attrs: Dict[str, str]) -> np.ndarray:
	"""Create a 4x4 homogeneous transform from MJCF attributes."""

	rot = _rotation_from_attributes(attrs)
	pos = _parse_floats(attrs.get("pos"), expected=3, default=(0.0, 0.0, 0.0))

	transform = np.eye(4)
	transform[:3, :3] = rot
	transform[:3, 3] = pos
	return transform


# ---------------------------------------------------------------------------
# MJCF asset helpers


def _sanitize_name(name: str | None, fallback: str, used: set[str]) -> str:
	candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "") or fallback
	base = candidate
	suffix = 1
	while candidate in used:
		candidate = f"{base}_{suffix}"
		suffix += 1
	used.add(candidate)
	return candidate


def _gather_assets(
	root: ET.Element,
	mjcf_path: Path,
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
	"""Collect mesh, material, and texture assets from the MJCF tree."""

	compiler = root.find("compiler")
	meshdir_attr = compiler.attrib.get("meshdir", ".") if compiler is not None else "."
	meshdir = (mjcf_path.parent / meshdir_attr).resolve()

	asset_elem = root.find("asset")
	mesh_map: Dict[str, Path] = {}
	material_map: Dict[str, Dict[str, str]] = {}
	texture_map: Dict[str, Dict[str, str]] = {}

	if asset_elem is None:
		logger.warning("No <asset> section found in MJCF; mesh exports may fail")
		return mesh_map, material_map, texture_map

	for mesh in asset_elem.findall("mesh"):
		name = mesh.attrib.get("name")
		file_attr = mesh.attrib.get("file")
		if not name or not file_attr:
			continue
		mesh_map[name] = (meshdir / file_attr).resolve()

	for texture in asset_elem.findall("texture"):
		name = texture.attrib.get("name")
		file_attr = texture.attrib.get("file")
		if not name or not file_attr:
			continue
		texture_map[name] = {"file": str((meshdir / file_attr).resolve()), **texture.attrib}

	for material in asset_elem.findall("material"):
		name = material.attrib.get("name")
		if not name:
			continue
		material_map[name] = {**material.attrib}

	return mesh_map, material_map, texture_map


# ---------------------------------------------------------------------------
# OBJ/MTL export helpers


def _write_obj(
	obj_path: Path,
	mtl_filename: str,
	vertices: List[Tuple[float, float, float]],
	faces: List[Tuple[int, int, int]],
	face_materials: List[str],
) -> None:
	obj_path.parent.mkdir(parents=True, exist_ok=True)
	with obj_path.open("w", encoding="utf-8") as obj_file:
		obj_file.write("# Generated by mjcf2obj\n")
		obj_file.write(f"mtllib {mtl_filename}\n")
		obj_file.write(f"o {obj_path.stem}\n")

		for x, y, z in vertices:
			obj_file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

		current_material = None
		for material, face in zip(face_materials, faces):
			if material != current_material:
				obj_file.write(f"usemtl {material}\n")
				current_material = material
			obj_file.write(f"f {face[0]} {face[1]} {face[2]}\n")


def _write_mtl(
	mtl_path: Path,
	materials: Dict[str, Dict[str, str]],
) -> None:
	mtl_path.parent.mkdir(parents=True, exist_ok=True)
	with mtl_path.open("w", encoding="utf-8") as mtl_file:
		mtl_file.write("# Generated by mjcf2obj\n")
		for name, props in materials.items():
			rgba = _parse_floats(props.get("rgba"), expected=4, default=(0.7, 0.7, 0.7, 1.0))
			specular = float(props.get("specular", 0.0))
			shininess = float(props.get("shininess", 0.5))
			emission = float(props.get("emission", 0.0))

			mtl_file.write(f"newmtl {name}\n")
			mtl_file.write(f"Ka {rgba[0] * 0.1:.4f} {rgba[1] * 0.1:.4f} {rgba[2] * 0.1:.4f}\n")
			mtl_file.write(f"Kd {rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f}\n")
			mtl_file.write(f"Ks {specular:.4f} {specular:.4f} {specular:.4f}\n")
			mtl_file.write(f"Ns {min(max(shininess * 128.0, 1.0), 1000.0):.4f}\n")
			mtl_file.write(f"d {rgba[3]:.4f}\n")
			if emission > 0.0:
				mtl_file.write(f"Ke {emission:.4f} {emission:.4f} {emission:.4f}\n")

			texture_file = props.get("texture_file")
			if texture_file:
				mtl_file.write(f"map_Kd {texture_file}\n")

			mtl_file.write("\n")


# ---------------------------------------------------------------------------
# MJCF traversal and mesh aggregation


def _material_properties(
	asset_props: Dict[str, str],
	texture_assets: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
	props = {
		"rgba": asset_props.get("rgba", "0.8 0.8 0.8 1.0"),
		"specular": asset_props.get("specular", "0.0"),
		"shininess": asset_props.get("shininess", "0.4"),
		"emission": asset_props.get("emission", "0.0"),
	}
	texture_name = asset_props.get("texture")
	if texture_name and texture_name in texture_assets:
		props["texture_file"] = texture_assets[texture_name].get("file")
	return props


def _load_and_transform_mesh(
	mesh_path: Path,
	transform: np.ndarray,
	scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(str(mesh_path))

	vertices = np.asarray(ms.current_mesh().vertex_matrix(), dtype=float)
	faces = np.asarray(ms.current_mesh().face_matrix(), dtype=int)

	if vertices.size == 0 or faces.size == 0:
		return np.empty((0, 3)), np.empty((0, 3), dtype=int)

	if faces.shape[1] != 3:
		try:
			ms.apply_filter("meshing_triangulation_quad_dominant")
			faces = np.asarray(ms.current_mesh().face_matrix(), dtype=int)
		except RuntimeError:
			logger.warning("Failed to triangulate mesh '%s'; skipping", mesh_path)
			return np.empty((0, 3)), np.empty((0, 3), dtype=int)

	scaled_vertices = vertices * scale
	hom_vertices = np.hstack([scaled_vertices, np.ones((scaled_vertices.shape[0], 1))])
	transformed_vertices = (transform @ hom_vertices.T).T[:, :3]

	return transformed_vertices, faces


def _collect_body_geoms(
	body_elem: ET.Element,
	parent_transform: np.ndarray,
	mesh_assets: Dict[str, Path],
	material_assets: Dict[str, Dict[str, str]],
	texture_assets: Dict[str, Dict[str, str]],
	*,
	default_material_prefix: str,
	material_defs: Dict[str, Dict[str, str]],
	material_export_names: Dict[str, str],
) -> Tuple[str, List[Dict[str, object]]]:
	body_name = body_elem.attrib.get("name") or "unnamed_body"
	body_transform = parent_transform @ _compose_transform(body_elem.attrib)

	geoms: List[Dict[str, object]] = []

	for geom in body_elem.findall("geom"):
		geom_type = geom.attrib.get("type", "mesh")
		if geom_type != "mesh":
			continue

		mesh_name = geom.attrib.get("mesh")
		if not mesh_name:
			logger.warning("Mesh geom without 'mesh' attribute in body '%s'", body_name)
			continue

		mesh_path = mesh_assets.get(mesh_name)
		if mesh_path is None or not mesh_path.exists():
			logger.warning("Mesh '%s' in body '%s' not found at '%s'", mesh_name, body_name, mesh_path)
			continue

		geom_transform = body_transform @ _compose_transform(geom.attrib)

		scale_attr = geom.attrib.get("scale")
		if scale_attr:
			scale = _parse_floats(scale_attr)
			if scale.size == 1:
				scale = np.repeat(scale, 3)
		else:
			scale = np.ones(3)

		geom_name = geom.attrib.get("name") or mesh_name
		material_attr = geom.attrib.get("material")

		if material_attr and material_attr in material_assets:
			export_name = material_export_names.setdefault(
				material_attr,
				_sanitize_name(material_attr, material_attr, set(material_export_names.values())),
			)
			if export_name not in material_defs:
				material_defs[export_name] = _material_properties(material_assets[material_attr], texture_assets)
		else:
			fallback_name = f"{default_material_prefix}_{geom_name}_mat"
			export_name = _sanitize_name(fallback_name, fallback_name, set(material_defs.keys()))
			if export_name not in material_defs:
				material_defs[export_name] = {
					"rgba": geom.attrib.get("rgba", "0.8 0.8 0.8 1.0"),
					"specular": geom.attrib.get("specular", "0.0"),
					"shininess": geom.attrib.get("shininess", "0.4"),
				}

		geoms.append(
			{
				"geom_name": geom_name,
				"mesh_path": mesh_path,
				"transform": geom_transform,
				"scale": scale,
				"material": export_name,
			}
		)

	return body_name, geoms


def _export_body(
	body_elem: ET.Element,
	*,
	parent_transform: np.ndarray,
	output_dir: Path,
	mesh_assets: Dict[str, Path],
	material_assets: Dict[str, Dict[str, str]],
	texture_assets: Dict[str, Dict[str, str]],
	body_name_usage: set[str],
	material_export_names: Dict[str, str],
) -> int:
	body_transform = parent_transform @ _compose_transform(body_elem.attrib)
	default_material_prefix = (body_elem.attrib.get("name") or "body").replace(" ", "_")
	material_defs: Dict[str, Dict[str, str]] = {}

	body_name, geoms = _collect_body_geoms(
		body_elem,
		parent_transform=parent_transform,
		mesh_assets=mesh_assets,
		material_assets=material_assets,
		texture_assets=texture_assets,
		default_material_prefix=default_material_prefix,
		material_defs=material_defs,
		material_export_names=material_export_names,
	)

	exported = 0
	if geoms:
		safe_body_name = _sanitize_name(body_name, "body", body_name_usage)
		obj_path = output_dir / f"{safe_body_name}.obj"
		mtl_path = output_dir / f"{safe_body_name}.mtl"

		vertices: List[Tuple[float, float, float]] = []
		faces: List[Tuple[int, int, int]] = []
		face_materials: List[str] = []

		for geom in geoms:
			geom_vertices, geom_faces = _load_and_transform_mesh(
				geom["mesh_path"],
				transform=geom["transform"],
				scale=geom["scale"],
			)

			if geom_vertices.size == 0 or geom_faces.size == 0:
				logger.warning("Skipping empty mesh '%s' in body '%s'", geom["geom_name"], body_name)
				continue

			vertex_offset = len(vertices)
			vertices.extend(map(tuple, geom_vertices))

			for face in geom_faces:
				faces.append(tuple(int(idx) + 1 + vertex_offset for idx in face))
				face_materials.append(str(geom["material"]))

		if vertices and faces:
			_write_obj(obj_path, mtl_path.name, vertices, faces, face_materials)
			_write_mtl(mtl_path, material_defs)
			exported += 1
		else:
			logger.info("Body '%s' contains no valid mesh geometry; skipping OBJ export", body_name)

	for child in body_elem.findall("body"):
		exported += _export_body(
			child,
			parent_transform=body_transform,
			output_dir=output_dir,
			mesh_assets=mesh_assets,
			material_assets=material_assets,
			texture_assets=texture_assets,
			body_name_usage=body_name_usage,
			material_export_names=material_export_names,
		)

	return exported


def export_mjcf_bodies(mjcf_path: Path, output_dir: Path) -> None:
	tree = ET.parse(mjcf_path)
	root = tree.getroot()

	mesh_assets, material_assets, texture_assets = _gather_assets(root, mjcf_path)
	worldbody = root.find("worldbody")
	if worldbody is None:
		raise ValueError("MJCF file does not contain a <worldbody> element")

	body_name_usage: set[str] = set()
	material_export_names: Dict[str, str] = {}

	total_exported = 0
	for body_elem in worldbody.findall("body"):
		total_exported += _export_body(
			body_elem,
			parent_transform=np.eye(4),
			output_dir=output_dir,
			mesh_assets=mesh_assets,
			material_assets=material_assets,
			texture_assets=texture_assets,
			body_name_usage=body_name_usage,
			material_export_names=material_export_names,
		)

	logger.info("Exported %d body OBJ files to '%s'", total_exported, output_dir)


# ---------------------------------------------------------------------------
# CLI entry point


def _configure_logging(verbose: bool) -> None:
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export MJCF bodies to OBJ files")
	parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF (*.xml) file")
	parser.add_argument("output_dir", type=Path, help="Directory where OBJ/MTL files will be stored")
	parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
	return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
	args = parse_args(argv)
	_configure_logging(args.verbose)

	try:
		export_mjcf_bodies(args.mjcf_path.resolve(), args.output_dir.resolve())
	except Exception as exc:  # pragma: no cover - runtime error reporting
		logger.error("Failed to export MJCF bodies: %s", exc)
		return 1

	return 0


if __name__ == "__main__":  # pragma: no cover
	sys.exit(main())
