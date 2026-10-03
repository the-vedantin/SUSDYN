"""sw_link.py - parametric SolidWorks bridge for Vahan hardpoints.

USAGE (the whole flow):
  1. In Vahan: All Hardpoints dialog -> 'Export SW equations' -> vahan_hardpoints.txt
  2. In SolidWorks: open the target PART (make it the active document).
  3. Trial with 2 points first:
         py tools\\sw_link.py setup  --coords vahan_hardpoints.txt --limit 2
     Check the part: a 3D sketch "Vahan_Hardpoints" with 2 points, each with
     three distance dimensions driven by vh_* global variables.
     Delete the sketch (globals can stay), then run the full setup:
         py tools\\sw_link.py setup  --coords vahan_hardpoints.txt
  4. Every export afterwards (weekly):
         py tools\\sw_link.py update --coords vahan_hardpoints.txt
     Only the global variables are rewritten + a rebuild — the existing sketch
     points MOVE, so mates/planes/features referencing them survive.
  5. If anything fails: run the same command again (setup is idempotent per
     point) and paste the printed error block — it names the point, the API
     call, and the COM error text.

  --dry-run     parse + attach + print the plan, write NOTHING to the part
  --limit N     setup only the first N points (trial run)
  selftest      offline checks (parser / frame mapping / equation text), no SW

What setup builds (one time per part):
  * one global variable per coordinate:  "vh_<point>_<axis>"= <value>mm
  * one 3D sketch "Vahan_Hardpoints" with one point per hardpoint;
    each point carries three distance dimensions (to Right/Top/Front plane)
    whose equations are  "Dn@Vahan_Hardpoints" = abs("vh_...")  — DRIVEN.
  The sketch is renamed BEFORE the equations are written, so the equation
  references are born with the final name (a later feature rename would not
  be guaranteed to propagate into equation text).

Frame: the coords file is in the car frame (X lateral, Y forward, Z up, mm).
--swap-yz (default ON) maps to SolidWorks Y-up:
    SW x = car x (lateral)   -> dimension to Right Plane
    SW y = car z (up)        -> dimension to Top Plane
    SW z = car y (forward)   -> dimension to Front Plane
(SolidWorks default part planes: Front = XY, Top = XZ, Right = YZ.)
Pass --no-swap-yz to keep car axes = SW axes (car Z-up in SW Z).

SIGN CAVEAT (inherent to SolidWorks distance dimensions): a distance
dimension is unsigned, so the equations use abs() and each point keeps the
SIDE of the plane it was created on.  If a later update moves a coordinate
ACROSS zero, that point rebuilds mirrored on that axis — `update` detects
and warns about every such sign flip; fix by deleting the sketch and
re-running `setup` (downstream references to the deleted points must then be
re-attached — so keep hardpoints off the zero planes when possible).

Requires:  pip install pywin32     (Windows + running SolidWorks)
"""
from __future__ import annotations

import argparse
import re
import sys

MM = 0.001                      # SolidWorks API lengths are metres
SKETCH_NAME = "Vahan_Hardpoints"
PFX = "vh_"

# SolidWorks constants (hardcoded so no typelib import is needed)
swDocPART = 1                   # swDocumentTypes_e.swDocPART
swInputDimValOnCreate = 10      # swUserPreferenceToggle_e (SwConst typelib)
SEL_PLANE = "PLANE"             # SelectByID2 type strings
SEL_SKPOINT = "SKETCHPOINT"
TYPE_3DSKETCH = "3DProfileFeature"   # IFeature::GetTypeName2 for a 3D sketch
TYPE_REFPLANE = "RefPlane"

# The three principal planes and which SW coordinate each measures.
# (default English template names first; localized templates fall back to
# tree order — the first three RefPlane features are Front, Top, Right)
PLANE_AXIS = [                       # (candidate names, SW axis index)
    (("Right Plane", "Right"), 0),   # distance to Right Plane = |SW x|
    (("Top Plane", "Top"), 1),       # distance to Top Plane   = |SW y|
    (("Front Plane", "Front"), 2),   # distance to Front Plane = |SW z|
]


# ---------------------------------------------------------------- COM helpers
def fmt_com_error(e):
    """Human-readable text for a pywin32 com_error (or any exception)."""
    try:
        import pywintypes
        if isinstance(e, pywintypes.com_error):
            hr = e.args[0] if e.args else "?"
            msg = e.args[1] if len(e.args) > 1 else ""
            exc = e.args[2] if len(e.args) > 2 else None
            desc = exc[2] if exc and len(exc) > 2 and exc[2] else ""
            return f"COM error hr={hr:#010x} {msg!r} sw_says={desc!r}" \
                if isinstance(hr, int) else f"COM error {e.args!r}"
    except ImportError:
        pass
    return f"{type(e).__name__}: {e!r}"


def C(obj, name, *args):
    """Call a SolidWorks COM member no matter how pywin32 exposed it.

    pywin32 maps the SW typelib inconsistently between late binding (dynamic
    dispatch) and early binding (makepy/gen_py cache): 'GetCount' may be a
    plain property (return the int) or a method (return a bound method that
    MUST be called).  Bare `doc.EditRebuild3` silently does nothing under
    early binding.  This helper normalizes both worlds:
        C(doc, 'EditRebuild3')          # zero-arg method OR property
        C(eqm, 'Equation', 3)           # parameterized property get / method
    Raises RuntimeError with a readable message on COM failure.
    """
    try:
        attr = getattr(obj, name)
    except Exception as e:
        raise RuntimeError(f"{name}: attribute lookup failed — "
                           + fmt_com_error(e)) from e
    if callable(attr):
        try:
            return attr(*args)
        except Exception as e:
            raise RuntimeError(f"{name}{args!r} failed — "
                               + fmt_com_error(e)) from e
    if args:
        raise RuntimeError(f"{name} came back as a plain value but was "
                           f"called with args {args!r} (pywin32 binding "
                           "mismatch)")
    return attr


def put(obj, name, value):
    """Property put with a readable error (e.g. feature.Name = ...)."""
    try:
        setattr(obj, name, value)
    except Exception as e:
        raise RuntimeError(f"set {name}={value!r} failed — "
                           + fmt_com_error(e)) from e


# ------------------------------------------------------------------- parsing
LINE_PAT = re.compile(
    r'^"([A-Za-z0-9_]+)_([xyz])"\s*=\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
    r'\s*mm\s*$')


def read_coords(path):
    """Parse the Vahan SW-equations export: '"name_x"= 123.456mm' lines.
    Returns {point_name: [x_mm, y_mm, z_mm]} in file order."""
    vals = {}
    n_lines = 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            n_lines += 1
            m = LINE_PAT.match(line.strip())
            if m:
                vals.setdefault(m.group(1), [None, None, None])[
                    "xyz".index(m.group(2))] = float(m.group(3))
    pts = {k: v for k, v in vals.items() if None not in v}
    incomplete = sorted(set(vals) - set(pts))
    if incomplete:
        print(f"WARNING: {len(incomplete)} point(s) missing an axis, "
              f"skipped: {', '.join(incomplete[:8])}"
              + (" …" if len(incomplete) > 8 else ""))
    if not pts:
        raise SystemExit(f'no \'"name_x"= 1.234mm\' lines found in {path} '
                         f"({n_lines} lines read)")
    return pts


# -------------------------------------------------------------- frame mapping
def car_to_sw(p_mm, swap_yz):
    """Car frame mm (X lat, Y fwd, Z up) -> SolidWorks metres."""
    x, y, z = p_mm
    if swap_yz:
        return (x * MM, z * MM, y * MM)     # SW Y-up
    return (x * MM, y * MM, z * MM)


def sw_to_car(p_m, swap_yz):
    """Inverse of car_to_sw (used by selftest)."""
    x, y, z = p_m
    if swap_yz:
        return (x / MM, z / MM, y / MM)
    return (x / MM, y / MM, z / MM)


def axis_var_letter(sw_axis, swap_yz):
    """Which car-frame letter the given SW axis carries (for vh_* names)."""
    return ("xzy" if swap_yz else "xyz")[sw_axis]


def global_name(point, sw_axis, swap_yz):
    return f"{PFX}{point}_{axis_var_letter(sw_axis, swap_yz)}"


def global_eq(vname, val_mm):
    return f'"{vname}"= {val_mm:.4f}mm'


def dim_eq(dim_name, vname):
    # dimension reference must be Name@Sketch — NOT IDimension::FullName,
    # which is "D1@Sketch1@Part1.Part" (includes the model; equations use
    # the two-part form, cf. the Add2/Equation examples in the API help).
    return f'"{dim_name}@{SKETCH_NAME}" = abs ( "{vname}" )'


# ------------------------------------------------------------------ attaching
def attach():
    try:
        import win32com.client  # noqa: F401
    except ImportError:
        raise SystemExit("pywin32 missing — run:  pip install pywin32")
    import win32com.client
    try:
        app = win32com.client.GetActiveObject("SldWorks.Application")
    except Exception as e:
        raise SystemExit("could not attach to a RUNNING SolidWorks "
                         "(GetActiveObject('SldWorks.Application')) — start "
                         "SolidWorks and open the part first.\n  "
                         + fmt_com_error(e))
    doc = C(app, "ActiveDoc")
    if doc is None:
        raise SystemExit("attached to SolidWorks, but no document is active "
                         "— open (and focus) the target PART")
    title = "?"
    try:
        title = C(doc, "GetTitle")
    except RuntimeError:
        pass
    try:
        dtype = int(C(doc, "GetType"))
        if dtype != swDocPART:
            raise SystemExit(f"active document {title!r} is not a PART "
                             f"(GetType={dtype}; part={swDocPART})")
    except (TypeError, ValueError):
        print("note: could not read document type — continuing, but make "
              "sure the active document is the part")
    print(f"attached to SolidWorks, active document: {title}")
    return app, doc


# ------------------------------------------------------------------ equations
def snapshot_equations(eqm):
    """{lhs_name: (index, full_text)} for every existing equation."""
    out = {}
    n = int(C(eqm, "GetCount"))
    for i in range(n):
        txt = C(eqm, "Equation", i)
        m = re.match(r'^\s*"([^"]+)"\s*=', txt or "")
        if m:
            out[m.group(1)] = (i, txt)
    return out


def set_equation_text(eqm, idx, text):
    """Rewrite equation idx.  IEquationMgr::Equation is a parameterized
    get/put property; pywin32 needs one of three routes depending on binding:
      1. makepy early binding generates SetEquation(index, value)
      2. late binding: raw Invoke with DISPATCH_PROPERTYPUT
      3. last resort: Delete(idx) + Add2(idx, text, False) at the same index
    Raises RuntimeError (with all three failures) if none worked."""
    errors = []
    try:                                    # 1. makepy-generated putter
        eqm.SetEquation(idx, text)
        return
    except Exception as e:
        errors.append("SetEquation: " + fmt_com_error(e))
    try:                                    # 2. explicit property-put invoke
        import pythoncom
        ole = eqm._oleobj_
        try:
            dispid = ole.GetIDsOfNames("Equation")
        except TypeError:
            dispid = ole.GetIDsOfNames(0, "Equation")
        if isinstance(dispid, (tuple, list)):
            dispid = dispid[0]
        ole.Invoke(dispid, 0, pythoncom.DISPATCH_PROPERTYPUT, 0, idx, text)
        return
    except Exception as e:
        errors.append("Invoke(PROPERTYPUT): " + fmt_com_error(e))
    try:                                    # 3. delete + re-add at same index
        C(eqm, "Delete", idx)
        new_idx = int(C(eqm, "Add2", idx, text, False))
        if new_idx < 0:
            raise RuntimeError(f"Add2 returned {new_idx}")
        return
    except Exception as e:
        errors.append("Delete+Add2: " + fmt_com_error(e))
    raise RuntimeError("could not rewrite equation "
                       f"[{idx}] {text!r}:\n    " + "\n    ".join(errors))


def set_globals(doc, pts, swap_yz, dry_run=False):
    """Create/refresh one global variable per coordinate.
    Returns (eqm, n_failures)."""
    eqm = C(doc, "GetEquationMgr")
    existing = snapshot_equations(eqm)
    n_new = n_upd = n_same = n_fail = 0
    sign_flips = []
    for name, p in pts.items():
        for sw_axis in range(3):
            car_letter = axis_var_letter(sw_axis, swap_yz)
            val = p["xyz".index(car_letter)]
            vname = global_name(name, sw_axis, swap_yz)
            eq = global_eq(vname, val)
            if vname in existing:
                idx, old_txt = existing[vname]
                m = re.search(r"=\s*(-?[0-9.eE+]+)\s*mm", old_txt or "")
                if m and float(m.group(1)) * val < 0:
                    sign_flips.append(f"{vname}: {m.group(1)} -> {val:.4f}")
                if old_txt is not None and old_txt.strip() == eq:
                    n_same += 1
                    continue
                if dry_run:
                    n_upd += 1
                    continue
                try:
                    set_equation_text(eqm, idx, eq)
                    n_upd += 1
                except RuntimeError as e:
                    n_fail += 1
                    print(f"  !! update global {vname}: {e}")
            else:
                if dry_run:
                    n_new += 1
                    continue
                try:
                    idx = int(C(eqm, "Add2", -1, eq, False))
                except RuntimeError as e:
                    idx = -1
                    print(f"  !! add global {vname}: {e}")
                if idx < 0:
                    n_fail += 1
                    print(f"  !! could not add global {vname} "
                          "(Add2 returned -1; a same-named equation with a "
                          "different LHS format may already exist)")
                else:
                    n_new += 1
    verb = "would be" if dry_run else ""
    print(f"globals: {n_new} {verb} created, {n_upd} {verb} updated, "
          f"{n_same} unchanged, {n_fail} FAILED")
    if sign_flips:
        print(f"WARNING — {len(sign_flips)} coordinate(s) CROSSED ZERO since "
              "setup; their distance dimensions cannot flip sides "
              "(abs() semantics). Those points are now MIRRORED on that "
              "axis until you delete the sketch and re-run setup:")
        for s in sign_flips:
            print(f"    {s}")
    return eqm, n_fail


# -------------------------------------------------------------- feature tree
def iter_features(doc):
    f = C(doc, "FirstFeature")
    while f is not None:
        yield f
        f = C(f, "GetNextFeature")


def find_feature(doc, name):
    for f in iter_features(doc):
        if C(f, "Name") == name:
            return f
    return None


def resolve_planes(doc):
    """Return [(plane_feature_name, sw_axis), ...] for the three principal
    planes.  Tries standard English names first, then falls back to feature-
    tree order (first three RefPlane features are Front, Top, Right in every
    default template, localized or not)."""
    tree_planes = [C(f, "Name") for f in iter_features(doc)
                   if C(f, "GetTypeName2") == TYPE_REFPLANE][:3]
    out = []
    for cand_names, axis in PLANE_AXIS:
        hit = next((n for n in cand_names if n in tree_planes), None)
        out.append((hit, axis))
    if all(h for h, _ in out):
        return out
    if len(tree_planes) < 3:
        raise SystemExit("could not find 3 reference planes in the part "
                         f"(found: {tree_planes}) — is this a normal part "
                         "template?")
    # fall back to creation order: Front, Top, Right
    print(f"note: non-standard plane names {tree_planes} — assuming tree "
          "order = Front, Top, Right")
    front, top, right = tree_planes
    return [(right, 0), (top, 1), (front, 2)]


# ---------------------------------------------------------------------- setup
def select_point(doc, ext, pt, sw):
    """Select one 3D-sketch point (replace selection). Two routes:
    ISketchPoint::Select4(Append, SelectData) with SelectData=None, then
    SelectByID2 by LOCATION (empty name + coordinates) as fallback."""
    try:
        if C(pt, "Select4", False, None):
            return True
    except RuntimeError:
        pass
    return bool(C(ext, "SelectByID2", "", SEL_SKPOINT,
                  sw[0], sw[1], sw[2], False, 0, None, 0))


def setup_sketch(app, doc, eqm, pts, swap_yz, dry_run=False, limit=None):
    existing = find_feature(doc, SKETCH_NAME)
    if existing is not None:
        print(f'sketch "{SKETCH_NAME}" already exists — globals were '
              "refreshed, geometry untouched.\nDelete the sketch in the "
              "FeatureManager first if you want a full re-setup.")
        return
    work = list(pts.items())[:limit] if limit else list(pts.items())
    planes = resolve_planes(doc)
    if dry_run:
        print(f"DRY RUN: would create 3D sketch {SKETCH_NAME!r} with "
              f"{len(work)} point(s), {3 * len(work)} driven dimensions:")
        for name, p in work[:5]:
            sw = car_to_sw(p, swap_yz)
            dims = ", ".join(
                f"{pl}=|{sw[ax] / MM:.3f}mm|<-abs({global_name(name, ax, swap_yz)})"
                for pl, ax in planes)
            print(f"  {name}: SW({sw[0]/MM:.3f}, {sw[1]/MM:.3f}, "
                  f"{sw[2]/MM:.3f})mm  {dims}")
        if len(work) > 5:
            print(f"  … and {len(work) - 5} more")
        return

    ext = C(doc, "Extension")
    skm = C(doc, "SketchManager")
    selmgr = C(doc, "SelectionManager")

    # AddDimension2 pops a modal value-entry dialog unless this toggle is
    # off — that would hang an out-of-process script.
    old_input_dim = None
    try:
        old_input_dim = C(app, "GetUserPreferenceToggle", swInputDimValOnCreate)
        C(app, "SetUserPreferenceToggle", swInputDimValOnCreate, False)
    except RuntimeError as e:
        print(f"  !! could not disable 'Input dimension value' ({e}) — if "
              "the script hangs, uncheck Tools>Options>General>'Input "
              "dimension value' and re-run")

    made = 0
    dims_linked = 0
    eq_jobs = []            # (dim_name, vname) — equations added AFTER rename
    failed = []             # (point, step, message)
    try:
        C(skm, "Insert3DSketch", True)
        if C(skm, "ActiveSketch") is None:
            raise SystemExit("Insert3DSketch did not open a 3D sketch "
                             "(is a command/dialog active in SolidWorks?)")
        # direct-to-database mode: no grid snapping, no inferred relations
        try:
            put(skm, "AddToDB", True)
            put(skm, "DisplayWhenAdded", False)
        except RuntimeError as e:
            print(f"  note: AddToDB not set ({e}) — continuing")

        for name, p in work:
            sw = car_to_sw(p, swap_yz)
            try:
                pt = C(skm, "CreatePoint", sw[0], sw[1], sw[2])
            except RuntimeError as e:
                failed.append((name, "CreatePoint", str(e)))
                continue
            if pt is None:
                failed.append((name, "CreatePoint",
                               "returned None (no active sketch?)"))
                continue
            ok = 0
            for plane_name, sw_axis in planes:
                vname = global_name(name, sw_axis, swap_yz)
                step = f"dim to {plane_name} ({vname})"
                try:
                    C(doc, "ClearSelection2", True)
                    if not select_point(doc, ext, pt, sw):
                        raise RuntimeError("point selection failed "
                                           "(Select4 and SelectByID2)")
                    if not C(ext, "SelectByID2", plane_name, SEL_PLANE,
                             0, 0, 0, True, 0, None, 0):
                        raise RuntimeError(
                            f"plane {plane_name!r} selection failed")
                    n_sel = int(C(selmgr, "GetSelectedObjectCount2", -1))
                    if n_sel != 2:
                        raise RuntimeError(
                            f"expected 2 selected entities, got {n_sel}")
                    # place the dim text 10mm off the point along the
                    # measured axis so the three labels don't stack
                    txt = list(sw)
                    txt[sw_axis] += 0.010
                    dd = C(doc, "AddDimension2", txt[0], txt[1], txt[2])
                    if dd is None:
                        raise RuntimeError(
                            "AddDimension2 returned None (selection did not "
                            "define a point-to-plane distance)")
                    dim = C(dd, "GetDimension2", 0)
                    dim_name = C(dim, "Name")        # "D1", "D2", …
                    if not dim_name:
                        raise RuntimeError("dimension has no name")
                    eq_jobs.append((dim_name, vname))
                    ok += 1
                except RuntimeError as e:
                    failed.append((name, step, str(e)))
                except Exception as e:          # unexpected — keep going
                    failed.append((name, step, fmt_com_error(e)))
            if ok == 3:
                made += 1
                if made % 10 == 0:
                    print(f"  {made}/{len(work)} points dimensioned…")
            C(doc, "ClearSelection2", True)
    finally:
        try:
            put(skm, "AddToDB", False)
            put(skm, "DisplayWhenAdded", True)
        except RuntimeError:
            pass
        try:
            C(skm, "Insert3DSketch", True)      # close the sketch
        except RuntimeError as e:
            print(f"  !! closing the 3D sketch failed: {e}")
        if old_input_dim is not None:
            try:
                C(app, "SetUserPreferenceToggle",
                  swInputDimValOnCreate, old_input_dim)
            except RuntimeError:
                pass

    # ---- rename FIRST, then write the dimension equations with the final
    # name baked in.  (IDimension::FullName is Name@Sketch@Model — not usable
    # in equations — and rename-propagation into equation text is not
    # documented, so we never create an equation that mentions the temporary
    # "3DSketchN" name.)
    sketch_feat = C(doc, "FeatureByPositionReverse", 0)
    renamed = False
    if sketch_feat is not None:
        tname = ""
        try:
            tname = C(sketch_feat, "GetTypeName2")
        except RuntimeError:
            pass
        if tname == TYPE_3DSKETCH:
            try:
                put(sketch_feat, "Name", SKETCH_NAME)
                renamed = C(sketch_feat, "Name") == SKETCH_NAME
            except RuntimeError as e:
                print(f"  !! renaming the sketch failed: {e}")
        else:
            print(f"  !! newest feature is {tname!r}, not a 3D sketch — "
                  "not renaming")
    if not renamed:
        print(f"  !! sketch NOT renamed to {SKETCH_NAME!r} — NOT writing "
              "dimension equations (they would reference the wrong name).\n"
              "  Rename the new 3D sketch manually and re-run setup after "
              "deleting it, or investigate the error above.")
        eq_failed = list(eq_jobs)
        eq_jobs = []
    else:
        eq_failed = []

    n_eq = 0
    for dim_name, vname in eq_jobs:
        eq = dim_eq(dim_name, vname)
        try:
            idx = int(C(eqm, "Add2", -1, eq, False))
        except RuntimeError as e:
            idx = -1
            print(f"  !! equation {eq!r}: {e}")
        if idx < 0:
            eq_failed.append((dim_name, vname))
            print(f"  !! could not add equation {eq!r} (Add2 -> -1; note: "
                  "Add2 refuses a duplicate left-hand side)")
        else:
            n_eq += 1
            dims_linked += 1

    try:
        C(doc, "EditRebuild3")
    except RuntimeError as e:
        print(f"  !! rebuild failed: {e}")

    # ---------------- summary
    print("\n================ setup summary ================")
    print(f"points created and fully dimensioned : {made}/{len(work)}")
    print(f"dimensions linked to globals         : {dims_linked}/"
          f"{3 * len(work)}")
    print(f"sketch renamed to {SKETCH_NAME!r}    : {renamed}")
    if failed:
        print(f"\n{len(failed)} FAILURE(S):")
        for pname, step, why in failed:
            print(f"  !! {pname} | {step} | {why}")
    if eq_failed:
        print(f"\n{len(eq_failed)} dimension(s) left UNLINKED (no equation).")
    if failed or eq_failed:
        print("\nTo retry cleanly: delete the sketch "
              f"{SKETCH_NAME!r} in the FeatureManager (globals may stay) "
              "and run setup again. Paste the error lines above when "
              "reporting.")
    else:
        print("all points driven — verify one: change a vh_* global in "
              "Tools>Equations, rebuild, watch the point move.")


# -------------------------------------------------------------------- update
def update_check(eqm, pts, swap_yz):
    """After an update: warn about file points that have no global yet
    (they were added in Vahan after setup and have NO sketch point)."""
    existing = snapshot_equations(eqm)
    missing = sorted({name for name, p in pts.items()
                      for ax in range(3)
                      if global_name(name, ax, swap_yz) not in existing})
    if missing:
        print(f"note: {len(missing)} point(s) had no pre-existing globals "
              f"(new in Vahan since setup?): {', '.join(missing[:8])}"
              + (" …" if len(missing) > 8 else ""))
        print("      their globals are created now, but they have NO sketch "
              "point until you delete the sketch and re-run setup.")


# ------------------------------------------------------------------ selftest
def selftest():
    import math
    ok = True

    def check(what, cond):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + what)
        ok = ok and cond

    # parser
    import os, tempfile
    sample = ('﻿"uca_front_FL_x"= 283.530mm\n'
              '"uca_front_FL_y"= -127.000mm\n'
              '"uca_front_FL_z"= 239.094mm\n'
              '"broken_pt_x"= 1.000mm\n'          # missing y/z -> skipped
              'garbage line\n'
              '"neg_e_FL_x"= -1.5e2mm\n'
              '"neg_e_FL_y"= 0.000mm\n'
              '"neg_e_FL_z"= 12.5mm\n')
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(sample)
    try:
        pts = read_coords(tmp)
    finally:
        os.unlink(tmp)
    check("parser: 2 complete points", set(pts) == {"uca_front_FL", "neg_e_FL"})
    check("parser: values", pts["uca_front_FL"] == [283.530, -127.000, 239.094])
    check("parser: exponent float", pts["neg_e_FL"][0] == -150.0)

    # frame mapping round trip
    p = (283.530, -127.000, 239.094)
    for swap in (True, False):
        back = sw_to_car(car_to_sw(p, swap), swap)
        check(f"frame round-trip swap={swap}",
              all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(p, back)))
    sw = car_to_sw(p, True)
    check("swap maps car z -> SW y (Top Plane axis)",
          math.isclose(sw[1], 239.094 * MM))
    check("swap maps car y -> SW z (Front Plane axis)",
          math.isclose(sw[2], -127.000 * MM))
    check("axis letters under swap (SW x,y,z -> car x,z,y)",
          [axis_var_letter(a, True) for a in range(3)] == ["x", "z", "y"])
    check("axis letters no swap",
          [axis_var_letter(a, False) for a in range(3)] == ["x", "y", "z"])

    # equation text
    check("global equation text",
          global_eq("vh_uca_front_FL_x", 283.53) ==
          '"vh_uca_front_FL_x"= 283.5300mm')
    check("dimension equation text",
          dim_eq("D7", "vh_uca_front_FL_z") ==
          f'"D7@{SKETCH_NAME}" = abs ( "vh_uca_front_FL_z" )')

    # plane->axis table consistency: Right=x, Top=y, Front=z
    check("PLANE_AXIS covers axes 0,1,2 once",
          sorted(ax for _, ax in PLANE_AXIS) == [0, 1, 2])

    # LHS-name extraction (snapshot regex) matches what global_eq writes
    m = re.match(r'^\s*"([^"]+)"\s*=', global_eq("vh_a_x", 1.0))
    check("snapshot regex reads back the LHS name",
          bool(m) and m.group(1) == "vh_a_x")

    print("selftest:", "ALL OK" if ok else "FAILURES — fix before live run")
    return 0 if ok else 1


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="run 'selftest' for offline checks; see the module docstring "
               "for the full flow")
    ap.add_argument("cmd", choices=["setup", "update", "selftest"],
                    help="setup = one-time build of driven points; "
                         "update = rewrite globals + rebuild (points move); "
                         "selftest = offline checks, no SolidWorks needed")
    ap.add_argument("--coords",
                    help="Vahan 'Export SW equations' txt file "
                         "(required for setup/update)")
    ap.add_argument("--no-swap-yz", action="store_true",
                    help="keep car Z-up instead of mapping to SolidWorks "
                         "Y-up")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + attach + print the plan; write NOTHING")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="setup only the first N points (trial run)")
    args = ap.parse_args()

    if args.cmd == "selftest":
        sys.exit(selftest())
    if not args.coords:
        ap.error(f"{args.cmd} requires --coords")

    pts = read_coords(args.coords)
    print(f"read {len(pts)} points ({3 * len(pts)} values) from "
          f"{args.coords}")
    swap = not args.no_swap_yz
    app, doc = attach()

    if args.cmd == "update":
        eqm = C(doc, "GetEquationMgr")
        update_check(eqm, pts, swap)
        _, n_fail = set_globals(doc, pts, swap, dry_run=args.dry_run)
        if not args.dry_run:
            try:
                C(doc, "EditRebuild3")
                print("rebuilt — done." if n_fail == 0 else
                      f"rebuilt, but {n_fail} global(s) FAILED — see above.")
            except RuntimeError as e:
                print(f"!! rebuild failed: {e}")
        else:
            print("dry run — nothing written.")
        sys.exit(1 if n_fail else 0)

    # setup
    eqm, n_fail = set_globals(doc, pts, swap, dry_run=args.dry_run)
    setup_sketch(app, doc, eqm, pts, swap,
                 dry_run=args.dry_run, limit=args.limit)
    if args.dry_run:
        print("dry run — nothing written.")


if __name__ == "__main__":
    main()
