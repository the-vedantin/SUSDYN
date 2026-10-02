import numpy as np

from vahan.interference import (rocker_double_shear_polys,
                                rocker_local_clevis_polys, rocker_plate_gaps,
                                rocker_plate_style_for, arb_blade_envelope_radius,
                                rocker_pr_full_length_fork_for,
                                rocker_pr_full_length_fork_clear_gap_for,
                                rocker_pr_full_length_fork_jog_for,
                                arb_blade_polyline, arb_blade_dogleg_params_for,
                                arb_member_kwargs,
                                full_members, cross_corner_clashes, clashes)


def _points():
    return {
        "rocker_pivot": np.array([0.0, 0.0, 0.0]),
        "pushrod_inner": np.array([0.10, 0.0, 0.0]),
        "rocker_spring_pt": np.array([0.0, 0.10, 0.0]),
        "arb_drop_top": np.array([-0.08, 0.02, 0.0]),
    }


def test_double_shear_plates_share_declared_gap_and_arm_widths():
    polys = rocker_double_shear_polys(_points())
    assert len(polys) == 14
    centers = sorted(round(float(p[:, 2].mean()), 6) for p in polys)
    assert centers == [-0.015] * 7 + [0.015] * 7
    # Main arms are 38.1 mm wide; the ARB arm is 25.4 mm wide.
    arms = [polys[i] for i in (0, 1, 2, 7, 8, 9)]
    widths = sorted(round(float(np.linalg.norm(p[0] - p[3])), 4) for p in arms)
    assert widths == [0.0254, 0.0254, 0.0381, 0.0381, 0.0381, 0.0381]
    # Each plate has a 25.4 mm-radius pivot boss, two 19.05 mm main
    # pickup bosses and a 12.7 mm ARB boss, providing ligament past holes.
    for group in (polys[3:7], polys[10:14]):
        # Boss polygons circumscribe the declared circles: edge apothem is
        # the physical radius, rather than vertices inscribing/understating it.
        radii = sorted(round(float(np.linalg.norm(
            .5*(p[0, :2]+p[1, :2])-p[:, :2].mean(0))), 5) for p in group)
        assert radii == [0.0127, 0.01905, 0.01905, 0.0254]


def test_center_plane_link_has_real_clearance_and_offset_tube_hits_plate():
    pts = _points()
    centered = [{"name": "fixture link", "a": np.array([0.02, 0.0, 0.0]),
                 "b": np.array([0.08, 0.0, 0.0]), "r": 0.008001}]
    gap = dict(rocker_plate_gaps(pts, centered, style="double_shear_arms"))["fixture link"]
    assert 0.0039 < gap < 0.0041  # 12 mm inner face minus 8.001 mm body radius

    on_plate = [{"name": "fixture tube", "a": np.array([0.02, 0.0, 0.015]),
                 "b": np.array([0.08, 0.0, 0.015]), "r": 0.001}]
    hit = dict(rocker_plate_gaps(pts, on_plate, style="double_shear_arms"))["fixture tube"]
    assert hit < 0.0


def test_joint_spheres_and_torsion_are_checked_without_attachment_skip():
    pts = _points()
    joint = [{"name": "pushrod rod end", "a": pts["pushrod_inner"],
              "b": pts["pushrod_inner"], "r": 0.008001}]
    gap = dict(rocker_plate_gaps(pts, joint, style="double_shear_arms"))["pushrod rod end"]
    assert 0.0039 < gap < 0.0041

    torsion = [{"name": "ARB torsion bar", "a": np.array([-0.2, 0.0, 0.015]),
                "b": np.array([0.2, 0.0, 0.015]), "r": 0.004}]
    assert dict(rocker_plate_gaps(pts, torsion, style="double_shear_arms"))["ARB torsion bar"] < 0.0


def test_full_coilover_capsule_clears_70mm_gap_and_is_not_skipped():
    pts = _points()
    coil = [{"name": "coilover", "a": pts["rocker_spring_pt"],
             "b": np.array([0.0, 0.20, 0.0]), "r": 0.0315}]
    gap = dict(rocker_plate_gaps(
        pts, coil, style="double_shear_arms", clear_gap_m=0.070))["coilover"]
    assert 0.0034 < gap < 0.0036

    shifted = [{"name": "coilover", "a": pts["rocker_spring_pt"] + [0, 0, 0.038],
                "b": np.array([0.0, 0.20, 0.038]), "r": 0.0315}]
    hit = dict(rocker_plate_gaps(
        pts, shifted, style="double_shear_arms", clear_gap_m=0.070))["coilover"]
    assert hit < 0.0


def test_local_clevis_is_connected_and_clears_centered_pushrod_joint():
    pts = _points()
    polys = rocker_local_clevis_polys(pts)
    # Three centre arms, two centre bosses, two pushrod cheek arms/bosses and
    # bridge, plus a separate connected ARB tab/boss/bridge.
    assert len(polys) == 17
    for poly in polys:
        normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        signs = []
        for i in range(len(poly)):
            a = poly[(i + 1) % len(poly)] - poly[i]
            b = poly[(i + 2) % len(poly)] - poly[(i + 1) % len(poly)]
            signs.append(float(np.dot(np.cross(a, b), normal)))
        assert min(signs) >= -1e-12 or max(signs) <= 1e-12

    joint = [{"name": "pushrod rod end", "a": pts["pushrod_inner"],
              "b": pts["pushrod_inner"], "r": 0.008001}]
    gap = dict(rocker_plate_gaps(
        pts, joint, style="local_clevis_arms"))["pushrod rod end"]
    assert 0.0039 < gap < 0.0041

    # The transverse bridge spans the outer faces of both cheeks and crosses
    # the centre plane, making the local fork a connected modeled solid.
    bridge = polys[8]
    assert bridge[:, 2].min() <= -0.018
    assert bridge[:, 2].max() >= 0.018

    arb_joint = [{"name": "ARB rod end", "a": pts["arb_drop_top"],
                  "b": pts["arb_drop_top"], "r": 0.008001}]
    arb_gap = dict(rocker_plate_gaps(
        pts, arb_joint, style="local_clevis_arms"))["ARB rod end"]
    assert 0.0039 < arb_gap < 0.0041

    spring_joint = [{"name": "spring rod end", "a": pts["rocker_spring_pt"],
                     "b": pts["rocker_spring_pt"], "r": 0.008001}]
    spring_gap = dict(rocker_plate_gaps(
        pts, spring_joint, style="local_clevis_arms"))["spring rod end"]
    assert 0.0264 < spring_gap < 0.0266

    full_coil = [{"name": "coilover", "a": pts["rocker_spring_pt"],
                  "b": np.array([0.0, 0.20, 0.0]), "r": 0.0315}]
    coil_gap = dict(rocker_plate_gaps(
        pts, full_coil, style="local_clevis_arms"))["coilover"]
    assert 0.0029 < coil_gap < 0.0031


def test_local_clevis_does_not_hide_non_pushrod_clashes():
    pts = _points()
    fixture = [{"name": "fixture", "a": np.array([0.0, 0.02, 0.0]),
                "b": np.array([0.0, 0.08, 0.0]), "r": 0.001}]
    assert dict(rocker_plate_gaps(
        pts, fixture, style="local_clevis_arms"))["fixture"] < 0.0


def test_local_clevis_invalid_geometry_fails_closed():
    cases = []
    missing = _points(); missing.pop("arb_drop_top")
    cases.append(missing)
    collinear = _points(); collinear["rocker_spring_pt"] = np.array([0.05, 0.0, 0.0])
    cases.append(collinear)
    short_push = _points(); short_push["pushrod_inner"] = np.array([0.020, 0.0, 0.0])
    cases.append(short_push)
    short_arb = _points(); short_arb["arb_drop_top"] = np.array([-0.020, 0.0, 0.0])
    cases.append(short_arb)
    for pts in cases:
        try:
            rocker_plate_gaps(pts, [], style="local_clevis_arms")
        except ValueError as exc:
            assert "local_clevis_arms" in str(exc)
        else:
            raise AssertionError("invalid local clevis geometry passed without an error")


def test_rocker_style_resolves_per_axle_with_global_fallback():
    car = {"rocker_plate_style": "legacy",
           "front_rocker_plate_style": "local_clevis_arms"}
    assert rocker_plate_style_for(car, "FL") == "local_clevis_arms"
    assert rocker_plate_style_for(car, "FR") == "local_clevis_arms"
    assert rocker_plate_style_for(car, "RL") == "legacy"
    assert rocker_plate_style_for(car, "rear") == "legacy"
    car["rear_rocker_plate_style"] = "double_shear_arms"
    assert rocker_plate_style_for(car, "RR") == "double_shear_arms"
    assert rocker_plate_style_for({}, "front") == "legacy"


def test_full_length_pr_fork_is_opt_in_per_axle():
    car = {"rocker_pr_full_length_fork": False,
           "front_rocker_pr_full_length_fork": True}
    assert rocker_pr_full_length_fork_for(car, "FL") is True
    assert rocker_pr_full_length_fork_for(car, "rear") is False
    assert rocker_pr_full_length_fork_for({}, "front") is False
    widths = {"rocker_pr_full_length_fork_clear_gap_mm": 36.0,
              "front_rocker_pr_full_length_fork_clear_gap_mm": 44.0}
    assert np.isclose(rocker_pr_full_length_fork_clear_gap_for(widths, "FL"), 0.044)
    assert np.isclose(rocker_pr_full_length_fork_clear_gap_for(widths, "rear"), 0.036)
    assert np.isclose(rocker_pr_full_length_fork_clear_gap_for({}, "front"), 0.024)


def test_full_length_pr_fork_has_connected_side_loadpaths_and_clear_corridor():
    pts = _points()
    polys = rocker_local_clevis_polys(pts, pr_full_length_fork=True)
    # Both continuous cheek arms reach the pivot and pickup at z=+/-15 mm.
    long_cheeks = [p for p in polys if len(p) == 4 and
                   np.ptp(p[:, 0]) > 0.095 and
                   abs(abs(float(p[:, 2].mean())) - 0.015) < 1e-9]
    assert len(long_cheeks) == 2
    # A localized pivot sleeve/bridge spans both outer faces and overlaps the
    # central pivot boss; it does not extend down the PR branch.
    pivot_bridges = [p for p in polys if len(p) == 4 and
                     p[:, 2].min() <= -0.018 and p[:, 2].max() >= 0.018 and
                     np.max(np.linalg.norm(p[:, :2], axis=1)) < 0.03]
    assert len(pivot_bridges) == 1
    center_link = [{"name": "fixture link", "a": np.array([0.06, 0.0, 0.0]),
                    "b": np.array([0.08, 0.0, 0.0]), "r": 0.006}]
    gap = dict(rocker_plate_gaps(
        pts, center_link, style="local_clevis_arms",
        pr_full_length_fork=True))["fixture link"]
    assert 0.0059 < gap < 0.0061

    # Collision remains active against either real cheek; the corridor is not
    # a blanket exclusion for blades or other off-plane members.
    blade = [{"name": "fixture blade", "a": np.array([0.06, 0.0, -0.015]),
              "b": np.array([0.08, 0.0, -0.015]), "r": 0.0079375}]
    assert dict(rocker_plate_gaps(
        pts, blade, style="local_clevis_arms",
        pr_full_length_fork=True))["fixture blade"] < 0.0

    widened = rocker_local_clevis_polys(
        pts, pr_full_length_fork=True,
        pr_full_length_fork_clear_gap_m=0.050)
    wide_cheeks = sorted(abs(float(p[:, 2].mean())) for p in widened
                         if len(p) == 4 and np.ptp(p[:, 0]) > 0.095)
    assert np.allclose(wide_cheeks, [0.028, 0.028])


def test_full_length_pr_fork_negative_cheek_jog_is_connected_and_real_material():
    pts = _points(); jog = 0.001
    straight = rocker_local_clevis_polys(pts, pr_full_length_fork=True)
    routed = rocker_local_clevis_polys(
        pts, pr_full_length_fork=True,
        pr_full_length_fork_negative_cheek_jog_m=jog)
    # One straight negative cheek is replaced by five full-width connected
    # strips; no polygon is removed to create an invisible clearance hole.
    assert len(routed) == len(straight) + 4
    strips = routed[3:8]
    assert all(abs(np.linalg.norm(p[0]-p[3]) - 0.0381) < 2e-5 for p in strips)
    for a, b in zip(strips[:-1], strips[1:]):
        assert np.allclose(a[1:3], b[[0, 3]]) or np.allclose(a[[1, 2]], b[[0, 3]])
    blade = [{"name": "fixture blade", "a": np.array([0.076, 0.0, -0.0285]),
              "b": np.array([0.076, 0.0, -0.0285]), "r": 0.0079375}]
    old_gap = dict(rocker_plate_gaps(
        pts, blade, style="local_clevis_arms", pr_full_length_fork=True))["fixture blade"]
    new_gap = dict(rocker_plate_gaps(
        pts, blade, style="local_clevis_arms", pr_full_length_fork=True,
        pr_full_length_fork_negative_cheek_jog_m=jog))["fixture blade"]
    assert old_gap < 0.003 < new_gap
    car = {"front_rocker_pr_full_length_fork_negative_cheek_jog_mm": 1.0}
    assert np.isclose(rocker_pr_full_length_fork_jog_for(car, "FL")[0], 0.001)


def test_formed_fork_polygons_are_exact_world_x_mirrors():
    left = {k: v + np.array([0.2, 0.0, 0.0]) for k, v in _points().items()}
    mirror = np.diag([-1.0, 1.0, 1.0])
    right = {k: mirror@v for k, v in left.items()}
    kw = dict(pr_full_length_fork=True,
              pr_full_length_fork_negative_cheek_jog_m=0.001)
    lp = rocker_local_clevis_polys(left, **kw)
    rp = rocker_local_clevis_polys(right, **kw)
    assert len(lp) == len(rp)
    for a, b in zip(lp, rp):
        expected = np.asarray([mirror@v for v in a])
        assert all(np.min(np.linalg.norm(b-v, axis=1)) < 1e-12 for v in expected)


def test_fixed_push_arm_scallop_clears_nearby_arb_eye_without_thinning_whole_arm():
    pts = _points()
    pts["pushrod_inner"] = np.array([0.124366, 0.0, 0.0])
    pts["arb_drop_top"] = np.array([0.033078, -0.031966, 0.0])
    polys = rocker_local_clevis_polys(pts, spring_clevis_setback_m=0.049168)
    assert len(polys) == 19  # full-width before/after pieces plus local scallop
    eye = [{"name": "ARB rod end", "a": pts["arb_drop_top"],
            "b": pts["arb_drop_top"], "r": 0.008001}]
    gap = dict(rocker_plate_gaps(
        pts, eye, style="local_clevis_arms",
        spring_clevis_setback_m=0.049168))["ARB rod end"]
    assert gap >= 0.003 - 1e-12


def test_declared_rectangular_arb_blade_uses_circumscribed_shared_envelope():
    radius = arb_blade_envelope_radius(25.4, 1.3895849902182817)
    assert abs(radius - 0.5*np.hypot(25.4, 1.3895849902182817)/1000.0) < 1e-15
    pts = _points()
    pts["arb_arm_end_world"] = np.array([0.0, 0.08, 0.0])
    members = full_members(pts, {}, arb_pivot=np.array([0.0, 0.0, 0.0]),
                           arb_od_mm=12.7, arb_blade_w_mm=25.4,
                           arb_blade_t_mm=1.3895849902182817)
    blade = next(m for m in members if m["name"] == "ARB blade")
    assert np.allclose(blade["a"], [0.0, 0.0, 0.0])
    assert np.allclose(blade["b"], [0.0, 0.08, 0.0])
    assert blade["r"] == radius
    no_dims = full_members(pts, {}, arb_pivot=np.zeros(3), arb_od_mm=12.7)
    rigid = next(m for m in no_dims if m["name"] == "ARB blade")
    assert abs(rigid["r"] - 0.5*0.625*25.4/1000.0) < 1e-15
    assert rigid["section_model"] == "project-default circular 0.625 in OD"
    shared = {"FL": [{"name": "ARB torsion bar", "a": np.array([.2, 0., 0.]),
                       "b": np.array([-.2, 0., 0.]), "r": .00635}],
              "FR": [{"name": "ARB blade", "a": np.array([-.2, 0., 0.]),
                       "b": np.array([-.2, .08, 0.]), "r": radius}]}
    assert cross_corner_clashes(shared) == []


def test_arb_blade_dogleg_is_rigid_mirrored_and_collision_visible():
    pts = _points(); pts["arb_arm_end_world"] = np.array([0.08, 0.06, 0.01])
    pivot = np.array([0.08, -0.02, -0.01])
    path = arb_blade_polyline(pts, pivot, 0.016, 0.43, 0.94)
    assert len(path) == 4 and np.allclose(path[0], pivot)
    assert np.allclose(path[-1], pts["arb_arm_end_world"])
    # Rocker motion cannot morph an ARB arm whose endpoints are held fixed.
    moved_rocker = dict(pts)
    moved_rocker["pushrod_inner"] = pts["pushrod_inner"] + [0.02, -0.03, 0.04]
    moved_rocker["rocker_spring_pt"] = pts["rocker_spring_pt"] + [-0.01, 0.02, -0.03]
    assert np.allclose(arb_blade_polyline(moved_rocker, pivot, 0.016, 0.43, 0.94), path)
    # Rotation about the actual X torsion axis rotates the complete rigid arm.
    ang = 0.37; R = np.array([[1.0, 0.0, 0.0],
                              [0.0, np.cos(ang), -np.sin(ang)],
                              [0.0, np.sin(ang), np.cos(ang)]])
    rotated = {k: R@v for k, v in pts.items()}
    path_r = arb_blade_polyline(rotated, R@pivot, 0.016, 0.43, 0.94)
    assert np.allclose(path_r, [R@v for v in path])
    lengths = lambda q: [np.linalg.norm(b-a) for a,b in zip(q[:-1],q[1:])]
    assert np.allclose(lengths(path_r), lengths(path))
    mirror = np.diag([-1.0, 1.0, 1.0])
    mirrored = {k: mirror@v for k, v in pts.items()}
    path_m = arb_blade_polyline(mirrored, mirror@pivot, 0.016, 0.43, 0.94)
    assert np.allclose(path_m, [mirror@v for v in path])
    members = full_members(pts, {}, arb_pivot=pivot, arb_od_mm=12.7,
                           arb_blade_dogleg_side_offset_mm=16.0)
    blades = [m for m in members if m["name"] == "ARB blade"]
    assert len(blades) == 3
    ae_eye = next(m for m in members if m["name"] == "ARB arm-end rod end")
    assert np.allclose(ae_eye["a"], pts["arb_arm_end_world"])
    assert ae_eye["r"] == 0.315*25.4/1000.0
    hit_point = 0.5*(path[1]+path[2])
    obstacle = {"name": "obstacle", "a": hit_point, "b": hit_point, "r": 0.001}
    assert clashes(blades + [obstacle], margin_mm=3.0)
    assert arb_blade_dogleg_params_for({}, "front") == (0.0, 0.0, 0.43, 0.94)
    car = {"front_arb_blade_dogleg_side_offset_mm": 16.0,
           "front_arb_blade_dogleg_axial_offset_mm": 11.0}
    assert arb_blade_dogleg_params_for(car, "FL")[:2] == (16.0, 11.0)
    axial_path = arb_blade_polyline(pts, pivot, 0.016, 0.43, 0.94, 0.011)
    axial_rotated = arb_blade_polyline(rotated, R@pivot, 0.016, 0.43, 0.94, 0.011)
    assert np.allclose(axial_rotated, [R@v for v in axial_path])


def test_configured_arb_member_kwargs_emit_every_routed_physical_segment():
    pts = _points(); pts["arb_arm_end_world"] = np.array([0.08, 0.06, 0.01])
    car = {"front_arb_blade_dogleg_side_offset_mm": -39.0,
           "front_arb_blade_dogleg_axial_offset_mm": -35.8,
           "front_arb_blade_dogleg_start_fraction": 0.05,
           "front_arb_blade_dogleg_end_fraction": 0.56}
    kw = arb_member_kwargs(car, "FL", 0.0, 0.0)
    members = full_members(pts, car, arb_pivot=np.array([0.08, -0.02, -0.01]),
                           arb_od_mm=12.7, **kw)
    blades = [m for m in members if m["name"] == "ARB blade"]
    assert len(blades) == 3
    assert np.allclose(blades[0]["b"], blades[1]["a"])
    assert np.allclose(blades[1]["b"], blades[2]["a"])
    assert not np.allclose(blades[0]["b"],
                           .95*blades[0]["a"] + .05*blades[2]["b"])


def test_arb_arm_end_rod_end_is_checked_against_unrelated_pr_plate():
    pts = _points(); pts["arb_arm_end_world"] = np.array([0.06, 0.0, -0.015])
    members = full_members(pts, {}, arb_pivot=np.array([0.08, -0.02, -0.01]),
                           arb_od_mm=12.7)
    eye = [m for m in members if m["name"] == "ARB arm-end rod end"]
    assert len(eye) == 1
    gap = dict(rocker_plate_gaps(
        pts, eye, style="local_clevis_arms",
        pr_full_length_fork=True))["ARB arm-end rod end"]
    assert gap < 0.0
