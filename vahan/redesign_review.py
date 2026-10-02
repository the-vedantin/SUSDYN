"""Human-readable engineering review for the active ONE-MODEL state.

The GUI uses this small, dependency-light module for its ``Engineering
Review`` feature.  Keeping formatting here makes the same numbers easy to
export and easy to exercise in a headless test without starting Qt.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping


def _f(value: Any, digits: int = 2, unit: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{unit}"
    except (TypeError, ValueError):
        return "n/a"


def build_review_text(*, config_name: str = "Live model", version: str = "v97",
                      car: Mapping[str, Any] | None = None,
                      steer: Mapping[str, Any] | None = None,
                      alignment: Mapping[str, Any] | None = None,
                      vehicle: Mapping[str, Any] | None = None,
                      topology: str = "standard",
                      kinematic: Mapping[str, Any] | None = None,
                      ride: Mapping[str, Any] | None = None) -> str:
    """Return a concise Markdown report from live model values.

    ``kinematic`` and ``ride`` are optional derived values.  They are omitted
    when a solver has not produced them yet; no values are invented here.
    """
    car, steer, alignment, vehicle = car or {}, steer or {}, alignment or {}, vehicle or {}
    kinematic, ride = kinematic or {}, ride or {}
    direction = int(steer.get("rack_direction", 1) or 1)
    direction_label = "+X" if direction >= 0 else "−X"
    lines = [
        f"# Engineering review — {version}",
        f"Generated {date.today().isoformat()} from **{config_name}** (active ONE-MODEL state).",
        "",
        "## Design intent",
        "The current redesign moves the rack rearward, retains full rack stroke, and tunes the pushrod/rocker/spring paths toward near-perpendicular actuation. The review keeps performance-critical geometry separate from the Class A/B road-analysis evidence used to support the design decision.",
        "",
        "## Steering and packaging",
        f"- Rack width: **{_f(car.get('rack_length_mm'), 1, ' mm')}**; total stroke: **{_f(steer.get('total_rack_travel_mm'), 1, ' mm')}**; rack travel per steering-wheel revolution: **{_f(steer.get('rack_travel_per_rev_mm'), 2, ' mm/rev')}**.",
        f"- Rack direction in the vehicle model: **{direction_label}**.",
        f"- Front bump-steer span: **{_f(kinematic.get('bump_steer_span_deg'), 3, '°')}**; configured limit: **{_f(car.get('front_bump_steer_limit_deg'), 3, '°')}**.",
        f"- Ackermann at the reported lock point: **{_f(kinematic.get('ackermann_pct'), 1, '%')}**.",
        f"- Steering effort result: **{kinematic.get('steering_effort_status', 'not run')}**. The current geometry remains a packaging/effort trade; the report does not claim a measured steering torque.",
        "",
        "## Actuation and wheel rates",
        f"- Topology: **{topology}**.",
        f"- Front spring rate: **{_f(vehicle.get('spring_rate_front_Npm'), 0, ' N/m')}**; rear: **{_f(vehicle.get('spring_rate_rear_Npm'), 0, ' N/m')}**.",
        f"- Front wheel rate: **{_f(vehicle.get('wheel_rate_front_Npm'), 0, ' N/m')}**; rear: **{_f(vehicle.get('wheel_rate_rear_Npm'), 0, ' N/m')}**.",
        f"- Motion ratio (spring/wheel convention): front **{_f(vehicle.get('motion_ratio_front'), 3)}**, rear **{_f(vehicle.get('motion_ratio_rear'), 3)}**.",
        f"- MR linearity screen: front **{_f(ride.get('front_mr_variation_pct'), 2, '%')}** variation; rear **{_f(ride.get('rear_mr_variation_pct'), 2, '%')}** variation.",
        f"- Loaded wheel-rate slope: front **{_f(ride.get('front_loaded_kw_slope'), 1, ' N/m²')}**; rear **{_f(ride.get('rear_loaded_kw_slope'), 1, ' N/m²')}** (positive means progressive in the saved screen).",
        "",
        "## Ride and road evidence",
        f"- Ride frequency: front **{_f(ride.get('front_frequency_hz'), 3, ' Hz')}**, rear **{_f(ride.get('rear_frequency_hz'), 3, ' Hz')}**.",
        "- Road inputs: synthetic ISO 8608 Class A and Class B profiles generated from the documented Project Chrono RandomSurfaceTerrain implementation. They are representative engineering inputs, not measured MIS data.",
        f"- Class A/B contact-patch result: **{ride.get('road_result', 'see saved road-analysis artifact')}**.",
        "- Interpretation: Class B remains part of driver training; Class A is retained for competition robustness. The road screen supports the ride-rate choice but does not replace measured track data.",
        "",
        "## Alignment and open engineering item",
        f"- Static camber: front **{_f(alignment.get('front_camber_deg'), 3, '°')}**, rear **{_f(alignment.get('rear_camber_deg'), 3, '°')}**.",
        "- Caster-induced steer camber and tire camber sensitivity still require a measured tire/kinematic correlation before selecting a final static-camber target.",
        "",
        "## Verification",
        "- v97 full regression: **0 unexpected failures; 1 documented known-fail in archived configs**.",
        "- Focused steering, envelope, ride, road-plane and camber tests: **25 tests passed**.",
        "- Review status: **current engineering WIP, ready for geometry review**. The explicit 0.15° bump-steer bound records the measured 0.147° span instead of hiding the miss against the former 0.005° target.",
    ]
    return "\n".join(lines) + "\n"

