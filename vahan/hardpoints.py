"""
Hardpoint definitions for double wishbone + pushrod/rocker suspension.

All coordinates are in the chassis frame (meters):
    X  ->  lateral (outboard = +X for the corner being modelled)
    Y  ->  longitudinal (rearward = +Y: front axle Y = 0, rear axle at +wheelbase)
    Z  ->  up

Origin convention: vehicle centerline (X=0), front axle (Y=0), ground (Z=0).
For the left-front corner outboard is +X.  Mirror X to get the right side.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DoubleWishboneHardpoints:
    """
    All hardpoints that define a double wishbone corner with pushrod/rocker.

    Inboard pivot pairs (uca_front/rear, lca_front/rear) define the rotation
    axis of each control arm.  The solver enforces constant distances from each
    outboard point to both inboard pivots, so the arm sweeps the correct arc.

    The upright is treated as a rigid body whose position is tracked through
    four points: uca_outer, lca_outer, tie_rod_outer, and wheel_center.
    Any additional upright-fixed point (e.g. pushrod_outer when body='upright')
    is expressed in the upright local frame at design and transformed at each step.
    """

    # Upper Control Arm
    uca_front:   np.ndarray   # inboard front chassis pivot
    uca_rear:    np.ndarray   # inboard rear chassis pivot
    uca_outer:   np.ndarray   # outboard balljoint (top of upright)

    # Lower Control Arm
    lca_front:   np.ndarray   # inboard front chassis pivot
    lca_rear:    np.ndarray   # inboard rear chassis pivot
    lca_outer:   np.ndarray   # outboard balljoint (bottom of upright)

    # Steering
    tie_rod_inner: np.ndarray  # chassis / rack-end pickup (chassis-fixed for kinematics)
    tie_rod_outer: np.ndarray  # upright pickup (moves with upright)

    # Wheel
    wheel_center: np.ndarray   # hub / wheel-center (moves with upright)

    # ── Pushrod / Rocker (used by PUSHROD / PULLROD actuation) ───────────
    # All optional — None when the active topology is DIRECT damper.
    pushrod_outer:     np.ndarray = field(default=None)
    pushrod_inner:     np.ndarray = field(default=None)
    rocker_pivot:      np.ndarray = field(default=None)
    rocker_spring_pt:  np.ndarray = field(default=None)
    spring_chassis_pt: np.ndarray = field(default=None)
    # Rocker rotation axis (optional — defaults to a Y-parallel axis through
    # rocker_pivot if the user doesn't supply an explicit second point).
    rocker_axis_pt:    np.ndarray = field(default=None)

    # ── Direct damper (used by DIRECT actuation) ─────────────────────────
    # damper_chassis_pt = top mount on chassis (fixed)
    # damper_outer_pt   = bottom mount on a moving body (UCA / LCA / upright)
    damper_chassis_pt: np.ndarray = field(default=None)
    damper_outer_pt:   np.ndarray = field(default=None)

    def __post_init__(self):
        """Cast every populated field to a float64 numpy array on construction."""
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, np.asarray(val, dtype=float))
        # Auto-compute rocker_axis_pt only if we have a rocker_pivot
        if self.rocker_axis_pt is None and self.rocker_pivot is not None:
            self.rocker_axis_pt = self.rocker_pivot + np.array([0., 0.0254, 0.])

    @classmethod
    def from_dict(cls, d: dict) -> "DoubleWishboneHardpoints":
        """Convenience constructor from a plain dict of lists/arrays."""
        return cls(**{k: np.array(v, dtype=float) for k, v in d.items()})

    def mirror_x(self) -> "DoubleWishboneHardpoints":
        """
        Return a mirrored copy for the opposite side of the car.

        Axis convention: X=lateral, Y=longitudinal, Z=up.
        Negating X produces the mirror-image corner (left <-> right).
        Optional fields (rocker / damper, depending on topology) are
        carried through as-None or mirrored when populated.
        """
        def flip(v):
            if v is None:
                return None
            w = v.copy()
            w[0] = -w[0]
            return w

        return DoubleWishboneHardpoints(
            uca_front=flip(self.uca_front),
            uca_rear=flip(self.uca_rear),
            uca_outer=flip(self.uca_outer),
            lca_front=flip(self.lca_front),
            lca_rear=flip(self.lca_rear),
            lca_outer=flip(self.lca_outer),
            tie_rod_inner=flip(self.tie_rod_inner),
            tie_rod_outer=flip(self.tie_rod_outer),
            wheel_center=flip(self.wheel_center),
            pushrod_outer=flip(self.pushrod_outer),
            pushrod_inner=flip(self.pushrod_inner),
            rocker_pivot=flip(self.rocker_pivot),
            rocker_spring_pt=flip(self.rocker_spring_pt),
            spring_chassis_pt=flip(self.spring_chassis_pt),
            rocker_axis_pt=flip(self.rocker_axis_pt),
            damper_chassis_pt=flip(self.damper_chassis_pt),
            damper_outer_pt=flip(self.damper_outer_pt),
        )

    # keep old name as alias for compatibility
    def mirror_y(self) -> "DoubleWishboneHardpoints":
        return self.mirror_x()
