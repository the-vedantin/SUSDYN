"""Import a .STEP CAD solid (differential, engine, …) as a triangle mesh in the
Vahan world frame, for packaging / clearance checks against the suspension.

Vahan frame: X = lateral (outboard), Y = longitudinal (forward), Z = up, in mm.

Coordinate source (the two CAD tools disagree on which axis is up):
  - "onshape"    : Z-up, same as Vahan and the "Suspension Points" FeatureScript
                   -> no rotation.
  - "solidworks" : Y-up (this is why exporting to SolidWorks needs "Y axis up"
                   checked) -> rotate +90 deg about X so the part stands upright.
After the source rotation the part is nudged by an explicit offset (mm) and can
be flipped front/back, so the user aligns it against the car by eye — placement
is a real input, not a hidden guess.

STEP tessellation is done by OpenCASCADE via `cascadio` (STEP -> glTF), then
`trimesh` reads the glTF.  glTF is in metres; STEP is in mm, so vertices are
scaled x1000 back to mm.

Dependencies: cascadio, trimesh (pip).  Both optional — HAVE_STEP is False and
load_step_mesh raises a clear message if they are missing.
"""
from __future__ import annotations
import base64
import os
import tempfile
import numpy as np

try:
    import cascadio
    import trimesh
    HAVE_STEP = True
except Exception:            # pragma: no cover - import guard
    HAVE_STEP = False

try:                         # OpenCASCADE kernel — needed to WRITE a moved STEP
    from OCP.STEPControl import (STEPControl_Reader, STEPControl_Writer,
                                 STEPControl_StepModelType)
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.gp import gp_Trsf
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    HAVE_OCC = True
except Exception:            # pragma: no cover - import guard
    HAVE_OCC = False

# CAD source -> 3x3 matrix mapping CAD axes into the Vahan frame (X lat, Y fwd, Z up)
_ROT_X_90 = np.array([[1.0, 0.0, 0.0],      # x -> x
                      [0.0, 0.0, -1.0],     # y_up -> -z  (forward)
                      [0.0, 1.0, 0.0]])     # z -> y_up  -> Z
# Onshape's STEP export lands with Y and Z swapped relative to Vahan — verified
# on a real export (2026-08: the Onshape diff came in correct only as
# "SolidWorks + flip front/back", whose net effect is exactly this Y<->Z swap).
_SWAP_YZ = np.array([[1.0, 0.0, 0.0],       # x -> x
                     [0.0, 0.0, 1.0],       # z -> y (forward)
                     [0.0, 1.0, 0.0]])      # y -> z (up)
SOURCE_TRANSFORMS = {
    "onshape": _SWAP_YZ,                     # Y<->Z swap (matches real Onshape export)
    "solidworks": _ROT_X_90,                # Y-up -> Z-up
}
SOURCES = tuple(SOURCE_TRANSFORMS.keys())


def _mesh_from_glb(glb_path):
    """Return (vertices Nx3 in metres, faces Mx3) baked into world coords, or
    (None, None) if the file holds no solid bodies (e.g. a points/sketch export)."""
    loaded = trimesh.load(glb_path, force="scene")
    solids = [g for g in loaded.geometry.values()
              if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not solids:
        return None, None
    # bake the glTF scene-graph transforms into a single mesh (world coords)
    try:
        world = loaded.to_geometry()
    except Exception:
        world = trimesh.util.concatenate(
            [g.copy().apply_transform(loaded.graph.get(name)[0])
             for name, g in loaded.geometry.items()
             if isinstance(g, trimesh.Trimesh) and len(g.faces)])
    return np.asarray(world.vertices, float), np.asarray(world.faces, int)


def _placement_matrix(source, flip_forward):
    """The 3x3 that load_step_mesh applies to CAD axes (source rotation, then an
    optional forward flip) — so a STEP transformed by [M | offset] lands exactly
    where the imported mesh sits in Vahan."""
    M = SOURCE_TRANSFORMS[str(source).lower()].copy()
    if flip_forward:
        M = np.diag([1.0, -1.0, 1.0]) @ M
    return M


def export_moved_step(src_path, out_path, source="onshape",
                      offset_mm=(0.0, 0.0, 0.0), flip_forward=False):
    """Write a real .STEP of the part transformed to its Vahan position (source
    axis convention + offset), using the OpenCASCADE kernel — a true rigid-body
    move of the B-rep, NOT text surgery.  The output matches what Vahan shows.

    Returns the output path.  Raises RuntimeError if OCC is unavailable or the
    read/write fails."""
    if not HAVE_OCC:
        raise RuntimeError("STEP export needs the OpenCASCADE kernel "
                           "(pip install cadquery-ocp).")
    if not os.path.isfile(src_path):
        raise RuntimeError(f"original STEP not found: {src_path}\n"
                           "Export re-reads the source file; keep it in place.")
    M = _placement_matrix(source, flip_forward)
    o = [float(v) for v in offset_mm]          # OCC reads this STEP in mm
    reader = STEPControl_Reader()
    if reader.ReadFile(str(src_path)) != IFSelect_RetDone:
        raise RuntimeError(f"could not read STEP: {src_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    t = gp_Trsf()
    t.SetValues(M[0, 0], M[0, 1], M[0, 2], o[0],
                M[1, 0], M[1, 1], M[1, 2], o[1],
                M[2, 0], M[2, 1], M[2, 2], o[2])
    moved = BRepBuilderAPI_Transform(shape, t, True).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(moved, STEPControl_StepModelType.STEPControl_AsIs)
    if writer.Write(str(out_path)) != IFSelect_RetDone:
        raise RuntimeError(f"could not write STEP: {out_path}")
    return out_path


def transform_report(name, source, offset_mm, bounds_min_mm, bounds_max_mm):
    """Plain-language placement instructions for the powertrain team, in the car
    frame (X lateral, Y forward, Z up), millimetres."""
    o = [float(v) for v in offset_mm]
    cx = 0.5 * (bounds_min_mm[0] + bounds_max_mm[0])
    cy = 0.5 * (bounds_min_mm[1] + bounds_max_mm[1])
    cz = 0.5 * (bounds_min_mm[2] + bounds_max_mm[2])

    def mv(v, pos, neg):
        return f"{pos} {v:.2f} mm" if v >= 0 else f"{neg} {-v:.2f} mm"
    return (
        f"PART PLACEMENT — {name}\n"
        f"Car frame: X = lateral, Y = forward, Z = up.  Units: millimetres.\n"
        f"Imported from: {source} CAD.\n\n"
        f"Move applied from the imported origin:\n"
        f"  lateral (X):  {mv(o[0], 'right', 'left')}\n"
        f"  forward (Y):  {mv(o[1], 'forward', 'rearward')}\n"
        f"  up (Z):       {mv(o[2], 'up', 'down')}\n\n"
        f"Final part centre (car frame):  "
        f"({cx:.2f}, {cy:.2f}, {cz:.2f}) mm\n"
        f"Final bounding box:\n"
        f"  min ({bounds_min_mm[0]:.2f}, {bounds_min_mm[1]:.2f}, {bounds_min_mm[2]:.2f}) mm\n"
        f"  max ({bounds_max_mm[0]:.2f}, {bounds_max_mm[1]:.2f}, {bounds_max_mm[2]:.2f}) mm\n\n"
        f"The exported STEP is already positioned here; the numbers above are for "
        f"placing the part by hand in a CAD assembly if preferred.\n"
    )


def pack_mesh(verts_mm, faces):
    """Encode a mesh (vertices mm Nx3, faces Mx3) as compact base64 float32 /
    uint32 blobs, for storing inside a .vahan JSON without bloating it into
    number lists.  Returns a JSON-safe dict."""
    v = np.ascontiguousarray(np.asarray(verts_mm, np.float32))
    f = np.ascontiguousarray(np.asarray(faces, np.uint32))
    return {
        "n_vertices": int(v.shape[0]),
        "n_faces": int(f.shape[0]),
        "verts_b64": base64.b64encode(v.tobytes()).decode("ascii"),
        "faces_b64": base64.b64encode(f.tobytes()).decode("ascii"),
    }


def unpack_mesh(d):
    """Inverse of pack_mesh -> (verts_mm Nx3 float32, faces Mx3 uint32)."""
    v = np.frombuffer(base64.b64decode(d["verts_b64"]), np.float32).reshape(-1, 3)
    f = np.frombuffer(base64.b64decode(d["faces_b64"]), np.uint32).reshape(-1, 3)
    return v.copy(), f.copy()


def load_step_mesh(path, source="onshape", offset_mm=(0.0, 0.0, 0.0),
                   flip_forward=False, tol_linear_mm=0.3, tol_angular_deg=0.5):
    """Load a STEP file -> (vertices_mm Nx3 in the Vahan frame, faces Mx3, info).

    source : "onshape" (Z-up) or "solidworks" (Y-up).
    offset_mm : (x, y, z) translation applied after the source rotation, mm.
    flip_forward : negate the longitudinal (Y) axis, if the part came in facing
                   the wrong way.
    Raises RuntimeError with a plain message on any failure."""
    if not HAVE_STEP:
        raise RuntimeError("STEP import needs the 'cascadio' and 'trimesh' "
                           "packages (pip install cascadio trimesh).")
    if not os.path.isfile(path):
        raise RuntimeError(f"STEP file not found: {path}")
    src = str(source).lower()
    if src not in SOURCE_TRANSFORMS:
        raise RuntimeError(f"unknown CAD source '{source}'; "
                           f"expected one of {SOURCES}")

    glb = os.path.join(tempfile.gettempdir(),
                       f"_vahan_step_{abs(hash(path)) % 10**8}.glb")
    try:
        cascadio.step_to_glb(str(path), glb,
                             tol_linear=float(tol_linear_mm),
                             tol_angular=float(tol_angular_deg))
    except Exception as e:
        raise RuntimeError(f"could not tessellate STEP: {e}")

    v_m, faces = _mesh_from_glb(glb)
    try:
        os.remove(glb)
    except OSError:
        pass
    if v_m is None:
        raise RuntimeError("no solid bodies in this STEP (looks like a points / "
                           "sketch export, not a part) — export the diff/engine "
                           "as a solid.")

    v = v_m * 1000.0                                   # metres -> mm
    v = v @ SOURCE_TRANSFORMS[src].T                   # CAD axes -> Vahan axes
    if flip_forward:
        v[:, 1] = -v[:, 1]
    v = v + np.asarray(offset_mm, float)

    info = {
        "source": src,
        "n_vertices": int(len(v)),
        "n_faces": int(len(faces)),
        "bounds_min_mm": np.round(v.min(axis=0), 2).tolist(),
        "bounds_max_mm": np.round(v.max(axis=0), 2).tolist(),
        "size_mm": np.round(v.max(axis=0) - v.min(axis=0), 2).tolist(),
        "path": os.path.basename(str(path)),
    }
    return v, faces, info
