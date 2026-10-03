"""
vahan/topology.py — Suspension topology configuration.

Defines how each axle's spring/damper actuation, anti-roll device, and
spring-spring decoupling are arranged.  The topology is decided once
(at project creation or load) and feeds the solver, the hardpoint set,
the 3D renderer, and the dynamics calculations.

Two axles, each independently configurable:

    DamperActuation         How wheel travel reaches the spring/damper.
        DIRECT              Damper bolts straight between chassis and arm/upright.
                            No rocker.  Spring length = damper length.
        PUSHROD             Wheel → pushrod → chassis-mounted bellcrank rocker
                            → spring/damper (compressive rod).
        PULLROD             Wheel → pullrod → chassis-mounted low rocker
                            → spring/damper (tensile rod, low CG of spring mass).

    DamperMount             Where the rod (or direct damper) is anchored on
                            the moving side.  Only used when actuation ≠ DIRECT
                            (or when DIRECT with a non-default attach).
        UCA                 On the upper control arm.
        LCA                 On the lower control arm.
        UPRIGHT             Directly on the upright (no arm load path).

    ARBType                 How roll stiffness is added.
        BELLCRANK           Standard U-bar.  Drop link from each rocker to a
                            short arm on the torsion tube.  Tube spans the car.
        CONTROL_ARM         Drop link bolts directly to the LCA at some point.
                            No rocker stage; simpler but less leverage tuning.
        TBAR                Centre-pivoted torsion bar.  Pivot is along chassis
                            X (lateral), torsion axis is perpendicular.  Drop
                            links go to the rockers.  Used when pushrod/damper
                            plane is skewed so a U-bar can't route cleanly.
        NONE                No anti-roll device — roll stiffness from springs
                            only (rare; only on cars with a heave/roll split).

    SpringConfig            How the per-axle springs are arranged.
        CORNER              One spring/damper per corner (standard).
        HEAVE_TBAR          Corner springs + a "third element" coilover whose
                            chassis end mounts to the T-bar's pivot bracket.
                            The T-bar pivots (translates) in heave because both
                            drop links push the same way, compressing the 3rd
                            spring.  In roll the T-bar torsions without
                            translating, so the 3rd spring sees zero load —
                            heave stiffness is decoupled from roll stiffness.
                            Requires arb_type == TBAR.
        DECOUPLED           Twin rockers per axle: one "flat" rocker driven by
                            both pushrods sums motion (acts in heave), one
                            "angled" rocker differences motion (acts in roll).
                            Two separate springs.  Corner springs are removed
                            entirely.  Heave and roll stiffness are fully
                            independent.

The default topology (`SuspensionTopology.standard()`) matches the existing
Vahan default and gives the current behaviour byte-for-byte:

    Front: PUSHROD on UCA, BELLCRANK ARB, CORNER springs.
    Rear:  PUSHROD on LCA, BELLCRANK ARB, CORNER springs.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum


class DamperActuation(Enum):
    DIRECT  = 'direct'
    PUSHROD = 'pushrod'
    PULLROD = 'pullrod'


class DamperMount(Enum):
    UCA     = 'uca'
    LCA     = 'lca'
    UPRIGHT = 'upright'


class ARBType(Enum):
    BELLCRANK   = 'bellcrank'
    CONTROL_ARM = 'control_arm'
    TBAR        = 'tbar'
    NONE        = 'none'


class SpringConfig(Enum):
    CORNER     = 'corner'
    HEAVE_TBAR = 'heave_tbar'   # 3rd-element coilover on T-bar pivot mount
    DECOUPLED  = 'decoupled'    # twin rockers (heave + roll), no corner springs


@dataclass
class AxleTopology:
    """Topology choice for a single axle (front OR rear)."""
    damper_actuation: DamperActuation = DamperActuation.PUSHROD
    damper_mount:     DamperMount     = DamperMount.UCA
    arb_type:         ARBType         = ARBType.BELLCRANK
    spring_config:    SpringConfig    = SpringConfig.CORNER

    # ── Self-checks ─────────────────────────────────────────────────────
    def validate(self) -> list[str]:
        """Return human-readable warnings about this axle's config."""
        warnings: list[str] = []
        if (self.spring_config == SpringConfig.HEAVE_TBAR
                and self.arb_type != ARBType.TBAR):
            warnings.append(
                'HEAVE_TBAR spring config requires arb_type == TBAR '
                '(the heave spring mounts on the T-bar pivot bracket).')
        if (self.spring_config == SpringConfig.DECOUPLED
                and self.damper_actuation == DamperActuation.DIRECT):
            warnings.append(
                'DECOUPLED springs need rocker-based actuation '
                '(twin rockers per axle).  Use PUSHROD or PULLROD.')
        if (self.damper_actuation == DamperActuation.DIRECT
                and self.arb_type == ARBType.BELLCRANK):
            warnings.append(
                'Bellcrank ARB needs a damper rocker for the drop link. '
                'With DIRECT damper, use a control-arm ARB, T-bar, or no ARB.')
        return warnings

    # ── Serialization ───────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            'damper_actuation': self.damper_actuation.value,
            'damper_mount':     self.damper_mount.value,
            'arb_type':         self.arb_type.value,
            'spring_config':    self.spring_config.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AxleTopology':
        return cls(
            damper_actuation=DamperActuation(d.get('damper_actuation',
                                                   'pushrod')),
            damper_mount=DamperMount(d.get('damper_mount', 'uca')),
            arb_type=ARBType(d.get('arb_type', 'bellcrank')),
            spring_config=SpringConfig(d.get('spring_config', 'corner')),
        )

    # ── Convenience predicates ──────────────────────────────────────────
    @property
    def has_rocker(self) -> bool:
        return self.damper_actuation in (DamperActuation.PUSHROD,
                                         DamperActuation.PULLROD)

    @property
    def needs_arb_hardpoints(self) -> bool:
        return self.arb_type != ARBType.NONE

    @property
    def has_heave_element(self) -> bool:
        """True if a separate heave spring exists (HEAVE_TBAR or DECOUPLED)."""
        return self.spring_config in (SpringConfig.HEAVE_TBAR,
                                      SpringConfig.DECOUPLED)

    @property
    def has_corner_springs(self) -> bool:
        """True iff per-corner springs are part of the spring set."""
        return self.spring_config != SpringConfig.DECOUPLED


@dataclass
class SuspensionTopology:
    """Whole-vehicle topology = one AxleTopology per axle."""
    front: AxleTopology = field(default_factory=AxleTopology)
    rear:  AxleTopology = field(default_factory=AxleTopology)

    @classmethod
    def standard(cls) -> 'SuspensionTopology':
        """Vahan default: pushrod on UCA front / LCA rear, bellcrank ARBs, corner springs."""
        return cls(
            front=AxleTopology(
                damper_actuation=DamperActuation.PUSHROD,
                damper_mount=DamperMount.UCA,
                arb_type=ARBType.BELLCRANK,
                spring_config=SpringConfig.CORNER,
            ),
            rear=AxleTopology(
                damper_actuation=DamperActuation.PUSHROD,
                damper_mount=DamperMount.LCA,
                arb_type=ARBType.BELLCRANK,
                spring_config=SpringConfig.CORNER,
            ),
        )

    # ── Self-checks ─────────────────────────────────────────────────────
    def validate(self) -> dict[str, list[str]]:
        return {
            'front': self.front.validate(),
            'rear':  self.rear.validate(),
        }

    def is_valid(self) -> bool:
        v = self.validate()
        return not v['front'] and not v['rear']

    # ── Serialization ───────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            'front': self.front.to_dict(),
            'rear':  self.rear.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SuspensionTopology':
        if not d:
            return cls.standard()
        return cls(
            front=AxleTopology.from_dict(d.get('front', {})),
            rear=AxleTopology.from_dict(d.get('rear', {})),
        )

    # ── Pretty-print ────────────────────────────────────────────────────
    def describe(self) -> str:
        """Human-readable one-paragraph description."""
        def axle_str(t: AxleTopology, label: str) -> str:
            damper = t.damper_actuation.value
            if t.damper_actuation != DamperActuation.DIRECT:
                damper += f' on {t.damper_mount.value}'
            arb = (f'{t.arb_type.value} ARB' if t.arb_type != ARBType.NONE
                   else 'no ARB')
            springs = t.spring_config.value
            return f'{label}: {damper}, {arb}, {springs} springs'
        return ' · '.join([axle_str(self.front, 'F'),
                           axle_str(self.rear,  'R')])


# ─────────────────────────────────────────────────────────────────────────────
#  Hardpoint sets — which point names are required for a given topology
# ─────────────────────────────────────────────────────────────────────────────
#
# These are reference lists used by the GUI + 3D viewer to decide which
# hardpoints to show/edit/render.  The kinematic solver consumes these
# via the SuspensionConstraints subclass for that topology.
#
# Note: the *common* set (wishbone + tie rod + wheel center) is shared
# across every topology, so the diff is just the spring / damper /
# rocker / ARB block.

_COMMON_HPS = (
    'uca_front', 'uca_rear', 'uca_outer',
    'lca_front', 'lca_rear', 'lca_outer',
    'tie_rod_inner', 'tie_rod_outer',
    'wheel_center',
)

def required_hardpoints(t: AxleTopology) -> tuple[str, ...]:
    """All hardpoint keys this topology needs (excluding ARB drop link)."""
    hps: list[str] = list(_COMMON_HPS)

    # Spring / damper / rocker block
    if t.damper_actuation == DamperActuation.DIRECT:
        hps += ['damper_chassis_pt', 'damper_outer_pt']
    else:   # pushrod or pullrod
        hps += [
            'pushrod_outer', 'pushrod_inner',
            'rocker_pivot', 'rocker_spring_pt', 'rocker_axis_pt',
            'spring_chassis_pt',
        ]
        if t.spring_config == SpringConfig.DECOUPLED:
            # Twin-rocker decoupled: second rocker for the other motion mode.
            # The "primary" rocker is the heave one; this is the roll rocker.
            hps += [
                'roll_rocker_pivot', 'roll_rocker_spring_pt',
                'roll_rocker_axis_pt', 'roll_spring_chassis_pt',
            ]
        if t.spring_config == SpringConfig.HEAVE_TBAR:
            # 3rd element heave spring lives on the T-bar pivot bracket
            # — one end fixed to chassis, other to the floating T-bar mount.
            hps += [
                'heave_spring_chassis_pt', 'heave_spring_tbar_pt',
            ]
    return tuple(hps)


def required_arb_hardpoints(t: AxleTopology) -> tuple[str, ...]:
    """ARB / T-bar hardpoints for one side (mirrored to the other)."""
    if t.arb_type == ARBType.BELLCRANK:
        return ('arb_drop_top', 'arb_arm_end', 'arb_pivot')
    if t.arb_type == ARBType.CONTROL_ARM:
        # Drop link from ARB lever to a point on the LCA
        return ('arb_drop_top', 'arb_lca_attach', 'arb_arm_end', 'arb_pivot')
    if t.arb_type == ARBType.TBAR:
        # Centre-pivoted torsion bar: pivot (chassis-fixed), arm ends, drop
        # link top on the rocker (same name as bellcrank for consistency).
        return ('arb_drop_top', 'arb_arm_end', 'arb_pivot')
    return tuple()
