"""
vahan/heave_tbar.py -- HEAVE-T-BAR twin-rocker actuation solver (reference).

ONE MODEL: this computes the ACTIVE spring's motion for a heave-T-bar car,
driven by the SAME pushrod_outer the corner wishbone solve produces, so the
kinematic graph, the dynamics, and the 3D all read one model.

Mechanism (from the user's reference geometry, verified by prototype):
  wheel travel -> pushrod -> ROCKER rotates about (rocker_pivot, rocker_axis)
  -> the rocker-resident drop-link foot (dl_nsm) moves
  -> drop link drives the T-BAR:
       * HEAVE (both wheels same): the whole bar pivots about its LATERAL axis
         (tbar_pivot -> tbar_axis, ~+X) -> the junction + 3rd-spring attach
         (3rd_arb) rise -> the 3rd (heave) spring (3rd_arb -> 3rd_chassis)
         compresses.  No twist.
       * ROLL (wheels opposite): the arms rotate in OPPOSITE senses about the
         SHAFT axis (tbar_pivot -> tbar_junc, ~vertical) -> the bar TWISTS by
         the angle that the torsion stiffness reacts.  3rd spring ~unchanged.

Required geometry keys (per axle, LEFT-side / central; right mirrors in X):
    rocker_pivot_left, rocker_axis_pt_left, pushrod_inner_left,
    tbar_drop_foot_left   (drop-link foot on the rocker),
    tbar_arm_tip_left     (drop-link top = T-bar arm tip),
    tbar_pivot, tbar_axis, tbar_junc,
    heave_spring_tbar_pt  (3rd-spring attach on the bar),
    heave_spring_chassis_pt.
plus the design pushrod_outer_left (wheel side) to freeze the pushrod length.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rot(p: np.ndarray, pivot: np.ndarray, axis: np.ndarray, th: float) -> np.ndarray:
    """Rodrigues rotation of point p about (pivot, unit axis) by angle th."""
    a = _norm(axis)
    r = p - pivot
    c, s = np.cos(th), np.sin(th)
    return pivot + c * r + s * np.cross(a, r) + (1.0 - c) * float(np.dot(a, r)) * a


@dataclass
class HeaveTBarState:
    bar_heave_angle: float     # bar pivot about lateral axis (rad), heave mode
    twist_angle: float         # arm twist about shaft axis (rad), roll mode
    heave_spring_length: float # 3rd-spring length (m)
    heave_spring_delta: float  # free - length (m); + = compression
    arb_attach_world: np.ndarray   # 3rd-spring bar attach after heave pivot


class HeaveTBarRockerSolver:
    """Mode-decoupled heave-T-bar solver.  Construct once per axle (freezes the
    rocker axis, bar axes, pushrod + drop-link lengths and the 3rd-spring free
    length at design pose, so deltas read 0 at design)."""

    def __init__(self, geo: dict, pushrod_outer_left_design: np.ndarray):
        g = {k: np.asarray(v, float).copy() for k, v in geo.items()
             if v is not None}
        self._g = g
        self._rock_pivot = g['rocker_pivot_left']
        self._rock_axis = _norm(g['rocker_axis_pt_left'] - g['rocker_pivot_left'])
        self._pin0 = g['pushrod_inner_left']
        self._foot0 = g['tbar_drop_foot_left']
        self._tip0 = g['tbar_arm_tip_left']
        self._tbar_pivot = g['tbar_pivot']
        self._heave_axis = _norm(g['tbar_axis'] - g['tbar_pivot'])     # lateral
        self._twist_axis = _norm(g['tbar_junc'] - g['tbar_pivot'])     # shaft
        # OPTIONAL 3rd (heave) spring.  Present -> HEAVE_TBAR car.  Absent ->
        # plain T-bar ARB car: the SAME mechanism without the 3rd element
        # (user-confirmed: "t bar is just heave t bar without 3rd element").
        self._arb0 = g.get('heave_spring_tbar_pt')
        self._chassis = g.get('heave_spring_chassis_pt')
        self.has_third_spring = (self._arb0 is not None
                                 and self._chassis is not None)
        # OPTIONAL corner coilover (spring+damper on the bellcrank's far end):
        # rocker-side attach rotates with the bellcrank; chassis end is fixed.
        self._coil_rocker0 = g.get('coil_rocker_left')
        self._coil_chassis = g.get('coil_chassis')

        self._L_push = float(np.linalg.norm(
            self._pin0 - np.asarray(pushrod_outer_left_design, float)))
        self._L_drop = float(np.linalg.norm(self._tip0 - self._foot0))
        self._L3_free = (float(np.linalg.norm(self._arb0 - self._chassis))
                         if self.has_third_spring else 0.0)

    # ── chain pieces ─────────────────────────────────────────────────────
    def _rocker_angle(self, po_world: np.ndarray) -> float:
        """Rocker rotation about its (possibly SKEWED 3-D) axis that keeps the
        rigid pushrod length to the live pushrod_outer.  Newton on |pin-po|²."""
        th = 0.0
        for _ in range(60):
            pin = _rot(self._pin0, self._rock_pivot, self._rock_axis, th)
            d = pin - po_world
            r = float(d @ d) - self._L_push ** 2
            if abs(r) < 1e-16:
                break
            dpin = np.cross(self._rock_axis, pin - self._rock_pivot)
            drdt = 2.0 * float(d @ dpin)
            if abs(drdt) < 1e-12:
                th += 0.01
                continue
            th = float(np.clip(th - r / drdt, -np.pi / 2, np.pi / 2))
        return th

    def _rocker_foot(self, po_world: np.ndarray) -> np.ndarray:
        """Drop-link foot world after the rocker rotates to take up the travel."""
        th = self._rocker_angle(po_world)
        return _rot(self._foot0, self._rock_pivot, self._rock_axis, th)

    def pose(self, po_world: np.ndarray) -> dict:
        """Full HEAVE-mode posed linkage for 3-D rendering: every node's world
        position at the given live pushrod_outer.  This is THE one model the 3D
        view must draw — same solve the graph + dynamics use.  The rocker turns
        in its skewed plane, the drop link drives the bar, and the bar pivots
        about its lateral axis carrying the junction + 3rd-spring attach."""
        po = np.asarray(po_world, float)
        th = self._rocker_angle(po)
        pin = _rot(self._pin0, self._rock_pivot, self._rock_axis, th)
        foot = _rot(self._foot0, self._rock_pivot, self._rock_axis, th)
        phi = self._bar_angle(foot, self._heave_axis)
        tip = _rot(self._tip0, self._tbar_pivot, self._heave_axis, phi)
        junc = _rot(self._g['tbar_junc'], self._tbar_pivot, self._heave_axis, phi)
        out = {
            'rocker_pivot':  self._rock_pivot,
            'rocker_axis_pt': self._g['rocker_axis_pt_left'],
            'pushrod_outer': po,
            'pushrod_inner': pin,
            'drop_foot':     foot,
            'arm_tip':       tip,
            'tbar_pivot':    self._tbar_pivot,
            'tbar_axis_pt':  self._g['tbar_axis'],
            'tbar_junc':     junc,
            'rocker_angle':  th,
            'bar_heave_angle': phi,
        }
        if self.has_third_spring:
            out['arb_attach'] = _rot(self._arb0, self._tbar_pivot,
                                     self._heave_axis, phi)
            out['chassis'] = self._chassis
        return out

    def _bar_angle(self, foot_world: np.ndarray, axis: np.ndarray) -> float:
        """Bar/arm angle about `axis` that preserves the drop-link length."""
        th = 0.0
        for _ in range(60):
            tip = _rot(self._tip0, self._tbar_pivot, axis, th)
            d = tip - foot_world
            r = float(d @ d) - self._L_drop ** 2
            if abs(r) < 1e-16:
                break
            dtip = np.cross(axis, tip - self._tbar_pivot)
            drdt = 2.0 * float(d @ dtip)
            if abs(drdt) < 1e-12:
                th += 0.01
                continue
            th = float(np.clip(th - r / drdt, -np.pi / 2, np.pi / 2))
        return th

    def pose_axle(self, po_left_world: np.ndarray,
                  po_right_world: np.ndarray) -> dict:
        """Both-sides T-bar pose as ONE rigid lever on ONE torsion bar.

        The lever (left_tip — junction — right_tip) is rigid and meets the
        torsion bar at the junction.  The bar PIVOTS about its base lateral
        axis by phi (heave) and the lever TWISTS about the pivoted shaft axis
        by psi (roll); the right tip twists -psi.  We solve (phi, psi) from the
        two drop-link length constraints (2-DOF Newton), so:
          * there is ONE shared junction (not two half-bars), and
          * the heave-spring attach moves with phi ONLY -> pure roll leaves it
            put (no spurious heave-spring actuation).
        Returns every node needed to draw the single bar + both sides."""
        def _mir(p):
            q = np.asarray(p, float).copy(); q[0] *= -1.0; return q

        poL = np.asarray(po_left_world, float)
        poR = np.asarray(po_right_world, float)
        piv = self._tbar_pivot
        hax = self._heave_axis
        rp, ra = self._rock_pivot, self._rock_axis

        # Each rocker's angle, and its drop foot + pushrod inner (right = mirror).
        thL = self._rocker_angle(poL)
        thR = self._rocker_angle(_mir(poR))
        footL = _rot(self._foot0, rp, ra, thL)
        footR_L = _rot(self._foot0, rp, ra, thR)   # right foot expressed in the LEFT frame
        footR = _mir(footR_L)
        pinL  = _rot(self._pin0, rp, ra, thL)
        pinR  = _mir(_rot(self._pin0, rp, ra, thR))

        # Each side's bar-pivot angle (1-DOF, the proven _bar_angle) — the angle
        # the lever arm must take so its drop link stays rigid to that foot.
        phiL = self._bar_angle(footL,  hax)
        phiR = self._bar_angle(footR_L, hax)
        # HEAVE = the SYMMETRIC part (mean) -> ONE shared junction + heave spring
        # that move with heave only.  In pure roll phiL = -phiR so phi_heave = 0
        # and the junction / spring stay put.  The tips pivot by their OWN side's
        # angle, so the single lever twists about the fixed junction in roll.
        phi_heave = 0.5 * (phiL + phiR)
        junc = _rot(self._g['tbar_junc'], piv, hax, phi_heave)
        left_tip  = _rot(self._tip0, piv, hax, phiL)
        right_tip = _mir(_rot(self._tip0, piv, hax, phiR))

        out = {
            'pivot_angle': phi_heave, 'roll_angle': 0.5 * (phiL - phiR),
            'tbar_pivot': piv, 'junc': junc,
            'left_tip': left_tip, 'right_tip': right_tip,
            'drop_foot_L': footL, 'drop_foot_R': footR,
            'pushrod_inner_L': pinL, 'pushrod_inner_R': pinR,
            'rocker_pivot_L': rp, 'rocker_pivot_R': _mir(rp),
            'pushrod_outer_L': poL, 'pushrod_outer_R': poR,
        }
        # 3rd (heave) spring only when the car has one (HEAVE_TBAR; a plain
        # T-bar ARB is the same mechanism without it).
        if self.has_third_spring:
            out['arb_attach'] = _rot(self._arb0, piv, hax, phi_heave)
            out['chassis'] = self._chassis
        # Corner coilover: rocker end rotates with the bellcrank, chassis end
        # fixed.  Right side is the X-mirror.  Only when supplied.
        if self._coil_rocker0 is not None and self._coil_chassis is not None:
            out['coil_rocker_L'] = _rot(self._coil_rocker0, rp, ra, thL)
            out['coil_rocker_R'] = _mir(_rot(self._coil_rocker0, rp, ra, thR))
            out['coil_chassis_L'] = self._coil_chassis
            out['coil_chassis_R'] = _mir(self._coil_chassis)
            out['coil_length_L'] = float(np.linalg.norm(
                out['coil_rocker_L'] - out['coil_chassis_L']))
            out['coil_length_R'] = float(np.linalg.norm(
                out['coil_rocker_R'] - out['coil_chassis_R']))
        return out

    # ── coil (corner coilover) helpers ───────────────────────────────────
    def coil_length(self, pushrod_outer_left_world: np.ndarray) -> float:
        """LEFT corner-coilover length at a live pushrod_outer (m).  The coil
        sits on the bellcrank, so its length tracks the rocker angle."""
        if self._coil_rocker0 is None or self._coil_chassis is None:
            return float('nan')
        th = self._rocker_angle(np.asarray(pushrod_outer_left_world, float))
        ck = _rot(self._coil_rocker0, self._rock_pivot, self._rock_axis, th)
        return float(np.linalg.norm(ck - self._coil_chassis))

    # ── public: heave + roll ─────────────────────────────────────────────
    def heave(self, pushrod_outer_left_world: np.ndarray) -> HeaveTBarState:
        """Both wheels heave together (symmetric): bar pivots about the lateral
        axis, 3rd spring compresses.  Without a 3rd spring (plain T-bar ARB)
        the spring length is NaN — callers must not read it for that car."""
        foot = self._rocker_foot(np.asarray(pushrod_outer_left_world, float))
        phi = self._bar_angle(foot, self._heave_axis)
        if not self.has_third_spring:
            return HeaveTBarState(bar_heave_angle=phi, twist_angle=0.0,
                                  heave_spring_length=float('nan'),
                                  heave_spring_delta=float('nan'),
                                  arb_attach_world=np.full(3, np.nan))
        arb = _rot(self._arb0, self._tbar_pivot, self._heave_axis, phi)
        L = float(np.linalg.norm(arb - self._chassis))
        return HeaveTBarState(bar_heave_angle=phi, twist_angle=0.0,
                              heave_spring_length=L,
                              heave_spring_delta=self._L3_free - L,
                              arb_attach_world=arb)

    def twist(self, pushrod_outer_left_world: np.ndarray) -> float:
        """Antisymmetric (roll): the left arm twists about the shaft axis.  The
        bar twist = 2*theta_arm for a symmetric pair; we return the single-arm
        angle which is what the torsion stiffness reacts per side."""
        foot = self._rocker_foot(np.asarray(pushrod_outer_left_world, float))
        return self._bar_angle(foot, self._twist_axis)
