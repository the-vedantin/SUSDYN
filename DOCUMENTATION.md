# Vahan — Suspension Simulation & Optimization Software

## Technical Documentation

**Version:** 1.0 (Kinematics Complete)
**Date:** April 2026
**Platform:** Python 3.12+ / PyQt6 / NumPy / SciPy

---

## 1. Project Overview

Vahan is a ground-up suspension design tool covering the full pipeline from kinematic geometry through inverse optimization. It targets double-wishbone suspensions with pushrod/rocker spring actuation — the standard layout for FSAE and production performance cars.

### What It Does Today

1. **Forward Kinematics** — Given 14 hardpoint coordinates, solve the full suspension constraint system at any wheel travel position. Computes 30+ metrics (camber, toe, caster, roll centre, anti-dive, motion ratio, etc.).

2. **Inverse Kinematics** — Given target metric curves (e.g., "I want camber to go from 0 deg at static to -2 deg at full bump"), find the hardpoint positions that produce them. Uses a priority-ordered staged solver that exploits geometric orthogonality between metrics.

3. **Interactive GUI** — Real-time 3D visualization, live metric graphs, hardpoint editing, motion sweeps (heave/roll/pitch/steer), and full inverse solve integration with collision detection.

### Architecture

```
vahan/                         GUI (PyQt6)
  hardpoints.py                  main_window.py
  solver.py        <------->     panels.py
  kinematics.py                  view3d.py (VisPy/OpenGL)
  metrics_catalog.py
  analysis.py
  optimizer.py
```

The `vahan/` package is a pure computation library with zero GUI dependencies. The `gui/` layer wraps it in a desktop application. Either can be used independently.

---

## 2. Coordinate System

```
        Z (up)
        |
        |
        +------ X (lateral, outboard positive for left corner)
       /
      Y (longitudinal, forward positive)
```

- **Origin:** Vehicle centreline (X=0), front axle line (Y=0), ground (Z=0)
- **Units:** Metres internally. GUI displays millimetres.
- **Corners:** FL (left-front, modelled), FR (X-mirrored), RL (left-rear, absolute Y coords), RR (X-mirrored of RL)

### Mapping from Excel (OptimumK Export)

The team's OptimumK export uses inches with X=longitudinal, Y=lateral, Z=vertical. Vahan swaps X/Y and converts to metres:

| OptimumK (in) | Vahan (m)          | Example: CHAS_LowFor |
|---------------|--------------------|-----------------------|
| X = 4.625     | Y = 4.625 * 0.0254 = 0.11748 m | lca_front Y |
| Y = 8.5       | X = 8.5 * 0.0254 = 0.21590 m   | lca_front X |
| Z = 4.75      | Z = 4.75 * 0.0254 = 0.12065 m  | lca_front Z |

---

## 3. Hardpoint Geometry

### 3.1 The 14 Hardpoints

Each suspension corner is defined by 14 points in 3D space (42 scalar coordinates):

**Control Arms (6 points)**
| Point | Description | Role |
|-------|-------------|------|
| `uca_front` | UCA front inboard pivot | Fixed to chassis |
| `uca_rear` | UCA rear inboard pivot | Fixed to chassis |
| `uca_outer` | UCA outer ball joint | Moves with wheel |
| `lca_front` | LCA front inboard pivot | Fixed to chassis |
| `lca_rear` | LCA rear inboard pivot | Fixed to chassis |
| `lca_outer` | LCA lower ball joint | Moves with wheel |

**Steering (2 points)**
| Point | Description | Role |
|-------|-------------|------|
| `tie_rod_inner` | Rack end / chassis pickup | Fixed (heave) or translates (steer) |
| `tie_rod_outer` | Upright steer arm pickup | Moves with wheel |

**Wheel (1 point)**
| Point | Description | Role |
|-------|-------------|------|
| `wheel_center` | Hub centre | Moves with wheel (driven by travel) |

**Pushrod/Rocker (5 points)**
| Point | Description | Role |
|-------|-------------|------|
| `pushrod_outer` | Pushrod attachment on arm/upright | Moves with arm |
| `pushrod_inner` | Pushrod attachment on rocker | Moves with rocker |
| `rocker_pivot` | Rocker chassis pivot | Fixed to chassis |
| `rocker_spring_pt` | Rocker end touching spring/damper | Moves with rocker |
| `spring_chassis_pt` | Chassis spring/damper mount | Fixed to chassis |

### 3.2 Default Values (2020 Car)

Derived from the team's OptimumK exports:

**Front (FL corner, metres):**
```
uca_front:       [0.26353, -0.12700,  0.26353]
uca_rear:        [0.23243,  0.12700,  0.24877]
uca_outer:       [0.48260,  0.00912,  0.28598]
lca_front:       [0.21590, -0.11748,  0.12065]
lca_rear:        [0.21590,  0.12342,  0.12700]
lca_outer:       [0.53340, -0.00318,  0.11913]
tie_rod_inner:   [0.21908, -0.06985,  0.15199]
tie_rod_outer:   [0.54293, -0.07303,  0.17145]
wheel_center:    [0.55880,  0.00000,  0.20320]
pushrod_outer:   [0.43815, -0.00318,  0.31953]
pushrod_inner:   [0.25740, -0.00318,  0.64683]
rocker_pivot:    [0.21293, -0.00318,  0.62230]
rocker_spring_pt:[0.20749, -0.00318,  0.67919]
spring_chassis:  [0.01588, -0.00318,  0.66091]
```

**From Excel:**
- Half track: 23 in (584.2 mm)
- Tire diameter: 16 in (406.4 mm) -> radius 203.2 mm
- Front spring rate: 43.78 lb/in (7.67 N/mm)
- Front ARB stiffness: 29.77 (lb/in at wheel or in-lb/deg — needs clarification)
- Steering ratio: 4.71:1
- Pushrod attached to: Lower A-Arm (front), Upright (rear)

---

## 4. Forward Kinematic Solver

### 4.1 The Constraint System

The suspension has **1 degree of freedom** — vertical wheel travel. Given a travel value, the solver finds the positions of 4 moving points (UCA outer, LCA outer, tie rod outer, wheel centre) by solving 12 simultaneous constraint equations.

**The 12 constraints** are all rigid-link length equations:

```
|uca_outer - uca_front|^2  = L1^2     (UCA front arm)
|uca_outer - uca_rear|^2   = L2^2     (UCA rear arm)
|lca_outer - lca_front|^2  = L3^2     (LCA front arm)
|lca_outer - lca_rear|^2   = L4^2     (LCA rear arm)
|lca_outer - uca_outer|^2  = L5^2     (upright: BJ separation)
|tr_outer  - uca_outer|^2  = L6^2     (upright: tie rod to UCA BJ)
|tr_outer  - lca_outer|^2  = L7^2     (upright: tie rod to LCA BJ)
|tr_outer  - tr_inner|^2   = L8^2     (tie rod length)
|wc - uca_outer|^2         = L9^2     (upright: WC to UCA BJ)
|wc - lca_outer|^2         = L10^2    (upright: WC to LCA BJ)
|wc - tr_outer|^2          = L11^2    (upright: WC to tie rod)
wc_z                       = wc0_z + travel   (drive constraint)
```

All link lengths L1...L11 are computed once from the design-position hardpoints and held constant. The 12th equation is the drive constraint that imposes the wheel travel.

### 4.2 Newton-Raphson Solver

The system is solved with full Newton-Raphson iteration:

```
x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)
```

Where:
- `x` = 12-vector [uca_outer(3), lca_outer(3), tr_outer(3), wheel_center(3)]
- `F(x)` = 12-vector of constraint residuals
- `J(x)` = 12x12 analytical Jacobian (no finite differences)

**Key features:**
- **Analytical Jacobian** — Each row of J is the gradient of one constraint with respect to x. For distance constraints `|a-b|^2 = L^2`, the partial derivatives are `2*(a_i - b_i)` or `-2*(a_i - b_i)`.
- **Warm-start** — The solution from the previous travel step is used as the initial guess for the next step. This gives convergence in 3-5 iterations (vs 15+ from a cold start).
- **Two-pass sweep** — Sweeps outward from the centre (design position) in both directions, so the warm-start chain is always continuous.
- **Convergence:** Tolerance 1e-10 on residual norm, typically converges in 3-5 iterations.

### 4.3 Rocker Solver

After the main 12-DOF solve, a separate 1-DOF Newton-Raphson solves the rocker angle. The pushrod outer point moves with the arm; the rocker must rotate to keep the pushrod length constant.

**Branch resolution:** The rocker equation has two solutions (rocker can flip). Vahan uses spring-length continuity — the correct branch is the one where the spring length changes smoothly from the previous travel step.

### 4.4 Motion Modes

| Mode | What `travel` means | How it works |
|------|---------------------|--------------|
| **Heave** | Vertical wheel displacement (mm) | Direct: `wc_z = wc0_z + travel` |
| **Roll** | Same as heave per corner | Left corner bumps, right droops (or vice versa) |
| **Pitch** | Same as heave per corner | Front bumps, rear droops (or vice versa) |
| **Steer** | Steering wheel angle (deg) | Rack translates tie_rod_inner in X by `angle * rack_mm_per_rev / 360` |

---

## 5. Kinematic Metrics

### 5.1 Computed from SolvedState

After solving, `KinematicMetrics` computes everything from the 3D point positions:

**Wheel Alignment Angles:**
- **Camber** — Angle of wheel plane vs vertical in front view (XZ). Negative = top leans inboard.
- **Toe** — Steering angle in top view (XY). Positive = toe-in.
- **Caster** — Kingpin axis tilt in side view (YZ). Positive = rearward tilt.
- **KPI** — Kingpin inclination in front view (XZ). Positive = top leans inboard.

**Steering Geometry:**
- **Scrub Radius** — Lateral distance from kingpin ground intercept to contact patch centre (mm).
- **Mechanical Trail** — Longitudinal distance from kingpin ground intercept to contact patch (mm).

**Roll Centre:**
- **IC (front view)** — Intersection of UCA and LCA lines projected into the front-view (XZ) plane.
- **RC Height** — Where the line from contact patch through IC crosses the vehicle centreline. Controls lateral load transfer distribution.

**Anti-Geometry (requires vehicle params):**
- **Anti-Dive %** — Percentage of braking pitch resisted by suspension geometry (front axle).
- **Anti-Squat %** — Percentage of acceleration squat resisted (rear axle).
- **Anti-Lift %** — Percentage of braking lift resisted (rear axle).
- Computed from side-view instant centre position relative to CG height and wheelbase.

**Rocker/Spring:**
- **Motion Ratio** — `d(spring_length) / d(wheel_travel)`. Dimensionless. Relates wheel rate to spring rate.
- **Spring Length** — Current spring/damper length (m).
- **Rocker Angle** — Rocker rotation from design position (deg).

### 5.2 Metrics Catalog

All 30+ metrics are registered in `metrics_catalog.py` with:
- Key (e.g., `'camber'`)
- Display label (e.g., `'Camber Angle'`)
- Unit (e.g., `'deg'`)
- Category (Angles, Lengths, Geometry, Anti, Ratios, etc.)
- Evaluation function

This catalog drives the GUI's metric picker, graph selector, and values table automatically.

---

## 6. Inverse Kinematics Solver

### 6.1 Problem Statement

**Given:** A target metric curve (e.g., camber = 0 deg at -30mm travel, linearly decreasing to -2 deg at +30mm travel), plus constraints on other metrics (locks).

**Find:** Hardpoint positions that produce that curve.

### 6.2 Formulation

The IK solver wraps the forward solver in a least-squares optimization:

```
minimize  sum_i  w_i * ||predicted_i(x) - target_i||^2  +  regularisation
```

Where:
- `x` = flat vector of selected hardpoint coordinates (design variables)
- `predicted_i(x)` = forward sweep at 21 travel points, extracting metric i
- `target_i` = desired curve for metric i
- `w_i` = importance weight

**Residual components:**
1. **Target error** — `sqrt(w) * (predicted - target)` per travel point
2. **Tolerance dead-band** — For lock constraints: zero penalty inside +/-tolerance
3. **Regularisation** — Small penalty for moving far from the starting hardpoints
4. **Collision avoidance** — Smooth ramp penalty starting 1mm before tube contact

### 6.3 Auto-Balanced Weights

When solving with lock constraints (e.g., "change camber but keep toe, anti-dive, RC constant"), the weights are automatically balanced:

```
primary_weight = n_locks * 10.0    (primary target dominates)
lock_weight    = 1.0               (locks are soft constraints)
lock_tolerance = 5.0               (dead-band in metric units)
```

This prevents N lock constraints from drowning out the single primary target.

### 6.4 Orthogonal Variable Groups

**Key insight:** Different suspension metrics are controlled by geometrically independent hardpoint subsets. Solving each metric with only its relevant variables prevents cross-contamination.

| Group | Metric | Variables | Why Orthogonal |
|-------|--------|-----------|----------------|
| 1 | Motion ratio | Pushrod/rocker X, Z | Completely independent — different mechanism |
| 2 | Toe / bump steer | Tie rod Y, Z | Steering linkage only — near-zero cross-talk |
| 3 | Anti-dive/squat/lift | UCA/LCA inboard **Y only** | Side-view pivot tilt. Doesn't touch front-view (Z/X) |
| 4 | Camber | UCA/LCA outer **Z**, inboard **Z** and **X** | Front-view IC geometry. Doesn't touch side-view (Y) |
| 5 | RC height | Same as camber | Coupled with camber through front-view IC |
| 6 | Caster / trail | Outer BJs **Y** | Kingpin fore-aft tilt. Minor effect on everything else |

### 6.5 Staged Solving Strategy

The `staged` method solves metrics sequentially in priority order:

```
1. STAGE: motion_ratio  (pushrod/rocker vars only)
2. STAGE: toe           (tie rod vars only)
3. STAGE: anti_dive     (inboard Y vars only)
4. STAGE: camber        (front-view Z/X vars only)
   ...
5. FINAL POLISH: all variables + all targets (warm-started from staged result)
```

**How it works:**
- Each stage creates a sub-solver with only that metric's orthogonal variables
- Solves locally (single LM from current position)
- Updates the working hardpoints
- Next stage starts from the updated geometry
- The final polish refines everything together, starting from the staged result

**Result:** Same solution quality as multi-start hybrid, but **5-6x faster** because the staged warm-start lands in the correct basin immediately.

### 6.6 Available Methods

| Method | Description | Speed | When to Use |
|--------|-------------|-------|-------------|
| `staged` | Orthogonal decomposition + polish | Fast (1-2 LM runs) | **Default.** Best for most problems |
| `hybrid` | 5 random LM starts, keep best | Slow (5 LM runs) | Fallback if staged misses |
| `local` | Single LM from current position | Fastest | Good warm-start available |
| `global` | Differential Evolution | Very slow | Desperate — large search space |

### 6.7 Collision Detection

Every solution is checked for physical feasibility — no two suspension tubes may overlap:

**Members checked:** UCA front/rear arms, LCA front/rear arms, tie rod, pushrod, spring/damper.

**Algorithm:** Exact 3D minimum distance between all non-connected line segment pairs. If `distance < (radius_A + radius_B)`, the solution has a collision.

**Integration:**
- **Penalty in optimizer** — Smooth ramp starting 1mm before contact (weight 2000). Steers the LM away from collisions during solving.
- **Post-solve check** — Hard collision detection on the final result. Reported in UI.
- **Explore filter** — Colliding solutions are rejected from the solution picker.

**Configurable tube diameters** (default FSAE values):
- UCA/LCA arms: 25.4 mm (1 in)
- Tie rod / pushrod: 19.0 mm (3/4 in)
- Spring/damper: 50.8 mm (2 in)

### 6.8 Explore (Find Solutions)

When the primary solve can't meet the target within bounds, the user can search wider:

1. Initial solve gives a warm-start x vector
2. Explore tries 4 bound levels: 2x, 4x, 7x, 10x the base bound
3. Each level uses warm-start LM from the initial solution
4. Runs in parallel (ThreadPoolExecutor — avoids Windows multiprocessing spawn overhead)
5. Colliding solutions are filtered out
6. Remaining solutions presented in a picker dialog sorted by cost

---

## 7. GUI

### 7.1 Layout

```
+-------------------+----------------------------+--------------------+
|   Left Sidebar    |      Centre 3D View        |   Right Panel      |
|                   |                            |                    |
| Motion Control    |   VisPy/OpenGL rendering   | Matplotlib Graphs  |
| Car Parameters    |   of suspension geometry   | (metric curves)    |
| Front Hardpoints  |                            |                    |
| Rear Hardpoints   |   + NavCube (orientation)  | Values Table       |
| Metric Picker     |                            | (current metrics)  |
| Steering          |                            |                    |
| Inverse Kin.      |                            |                    |
+-------------------+----------------------------+--------------------+
```

### 7.2 Panels (Left Sidebar)

All panels are collapsible sections. Key panels:

- **MotionPanel** — Heave/roll/pitch/steer mode, travel range, slider, damper stroke/sag
- **CarParamsPanel** — Wheelbase, CG height, track width, brake/drive bias
- **HardpointPanel** (x2) — Edit all 14 hardpoints per corner, load/save
- **GraphPickerPanel** — Select which metrics to plot
- **InverseKinematicsPanel** — Full IK solver UI: target metric, range, locks, method, tube ODs, solve/explore buttons, results table

### 7.3 3D View

GPU-accelerated rendering via VisPy:
- Control arms, upright, tie rod, pushrod, rocker, spring rendered as coloured lines
- Rocker angle arc indicator
- NavCube overlay (click faces to snap to front/side/top views)
- Mouse: right-drag orbit, middle-drag pan, scroll zoom

### 7.4 Grayscale Theme

The entire UI uses a grayscale colour scheme (dark background, grey text/borders). Only the 3D points and graph lines use colour, keeping the focus on the engineering data.

### 7.5 Scroll-Wheel Protection

All spinboxes and combo boxes ignore the scroll wheel unless they have keyboard focus. This prevents accidental value changes when scrolling the sidebar.

---

## 8. Vehicle Data (from SUS Calculations Google Sheet)

### 8.1 Vehicle Parameters (2026 Car)

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Total mass (car + driver) | m | 290.35 | kg | CG sheet |
| Car mass | — | 222.35 | kg | CG sheet |
| Driver mass | — | 68 | kg | CG sheet |
| Total weight | W | 2848.33 | N | CG sheet |
| Sprung mass | Ms | 223.8 | kg | Springs sheet |
| Unsprung mass (total) | Mu | 66.55 | kg | Springs sheet |
| Unsprung mass (front axle) | — | 26.5 | kg | Load transfer sheet |
| Unsprung mass (rear axle) | — | 40.05 | kg | Load transfer sheet |
| Wheelbase | L | 1530 | mm | Vehicle params |
| CG height | hm | 260.63 | mm | CG sheet (calculated) |
| CG distance from front axle | lm | 841.18 | mm | CG sheet (calculated) |
| Front track width | Tf | 1221.8 | mm | Vehicle params |
| Rear track width | Tr | 1200 | mm | Vehicle params |
| Wheel radius | — | 203.2 | mm | Vehicle params |
| F:R weight distribution | — | 45% / 55% | — | CG sheet |

### 8.2 Springs, Dampers & Rates

| Parameter | Front | Rear | Unit | Notes |
|-----------|-------|------|------|-------|
| Spring rate | 22.0 | 22.0 | N/mm | Pitch stiffness sheet |
| Motion ratio (wheel/spring) | 0.97 | 0.82 | — | Pitch stiffness sheet |
| Wheel centre rate (Kw) | 17.5–25.8 | 14.8–27.5 | N/mm | Varies by calc method |
| Ride rate (Kr) | 15.8–22.2 | 15.5–23.5 | N/mm | Series with tire |
| Tire spring rate (Ks) | 159.1 | 159.1 | N/mm | Hoosier @ 14 PSI, ~500 lb |
| Damper stroke | 50 | 50 | mm | Springs sheet |
| Sprung natural frequency (fs) | 1.56 | — | Hz | No downforce |
| Unsprung natural frequency (fu) | 8.41 | — | Hz | — |

### 8.3 Damping

| Parameter | Front | Rear | Unit |
|-----------|-------|------|------|
| Damping ratio (zeta) | 0.70 | 0.70 | — |
| Damping coefficient (C) | 1957.5 | 2225.2 | N/m/s |
| Critical damping (Cc) | 9299.8 | 9299.8 | N/m/s |
| Low-freq bump slope | 1305.0 | 1483.4 | N/m/s |
| Low-freq rebound slope | 2936.2 | 3337.8 | N/m/s |
| High-freq bump slope | 652.5 | 741.7 | N/m/s |
| High-freq rebound slope | 1468.1 | 1668.9 | N/m/s |
| Jounce ratio | 0.625 | 0.643 | — |
| Rebound ratio | 1.407 | 0.720 | — |
| Damper type | Ohlins TTX25 (twin-tube) | — | — |

### 8.4 Anti-Roll Bars

| Parameter | Front | Rear | Unit |
|-----------|-------|------|------|
| ARB outer diameter (D) | 12.7 | 12.7 | mm |
| ARB inner diameter (d) | 9.75 | 9.75 | mm |
| Arm length (A) | 84.3–90 | 104.8 | mm |
| Half-length (L) | 249.4 | 286.3 | mm |
| Shear modulus (G) | 79.3 | 79.3 | GPa |
| ARB motion ratio | 2.4–2.5 | 2.92–3.0 | — |
| ARB link stiffness | 50.12 | 30.73 | N/mm |
| ARB wheel rate per wheel | 16.8–20.9 | 6.5–10.5 | N/mm |

### 8.5 Roll & Pitch Stiffness

| Parameter | Value | Unit |
|-----------|-------|------|
| Front roll stiffness (spring only) | 451.5 | Nm/deg |
| Rear roll stiffness (spring only) | 304.2 | Nm/deg |
| Total roll stiffness (spring only) | 755.7 | Nm/deg |
| Front roll stiffness (spring + ARB) | 691.6 | Nm/deg |
| Rear roll stiffness (spring + ARB) | 451.8 | Nm/deg |
| Total roll stiffness (spring + ARB) | 1702.6 | Nm/deg |
| Front roll stiffness distribution | 59.7% | — |
| Body roll at 1g | 0.8 | deg |
| Pitch stiffness | 739.9 | Nm/deg |
| Roll centre height (front) | 64.36 | mm |
| Roll centre height (rear) | 85.12 | mm |
| Roll axis to sprung CG (ha) | 237.2 | mm |

### 8.6 Anti-Geometry & Cornering

| Parameter | Value | Unit |
|-----------|-------|------|
| Anti-dive | 18.94 | % |
| Anti-lift | 29.13 | % |
| Anti-squat | 52.98 | % |
| Max lateral g (no downforce) | 1.5 | g |
| Max lateral g (with downforce) | 1.96 | g |
| Friction coefficient | 1.5 | — |
| Total downforce (max speed) | 875.4 | N |
| Aero balance (F:R) | 40/60 | % |

### 8.7 What's Available vs Still Needed for Dynamics

| Parameter | Status | Source |
|-----------|--------|--------|
| Spring rates (F/R) | HAVE | SUS Calculations |
| Motion ratios (F/R) | HAVE | SUS Calculations + kinematics |
| Damping coefficients (asymmetric) | HAVE | SUS Calculations |
| ARB stiffness (F/R) | HAVE | SUS Calculations |
| Sprung / unsprung mass | HAVE | SUS Calculations |
| CG location (3D) | HAVE | SUS Calculations |
| Tire vertical rate | HAVE | 159.1 N/mm (Hoosier) |
| Roll / pitch stiffness | HAVE | SUS Calculations |
| Roll centre heights | HAVE | SUS Calculations + kinematics |
| Anti-geometry % | HAVE | SUS Calculations + kinematics |
| Damper dyno curves (full) | PARTIAL | Have slopes, need full Ohlins TTX25 curves |
| Roll/pitch inertia | NEED | Estimate from CAD or test |
| Tire lateral force model | NEED | TTC data or Pacejka fits |

---

## 9. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `vahan/hardpoints.py` | ~80 | Hardpoint dataclass + mirror ops |
| `vahan/solver.py` | ~350 | 12-DOF Newton-Raphson constraint solver |
| `vahan/kinematics.py` | ~150 | Metric computation from solved state |
| `vahan/analysis.py` | ~80 | High-level sweep interface |
| `vahan/metrics_catalog.py` | ~260 | 30+ metric definitions |
| `vahan/optimizer.py` | ~850 | IK solver, orthogonal groups, collision detection |
| `gui/main_window.py` | ~1700 | Main window, steering model, all wiring |
| `gui/panels.py` | ~1150 | 11 sidebar panels |
| `gui/view3d.py` | ~500 | VisPy 3D rendering + NavCube |
| `app.py` | ~10 | Entry point |

---

## 10. What's Next: Dynamics

The kinematics engine computes how the suspension geometry changes with travel. The next phase adds **forces and accelerations** — how the car responds to road inputs, cornering loads, and driver inputs over time.

This requires coupling the kinematic model with:
- Spring/damper force models (force as a function of displacement and velocity)
- Tire force models (vertical load, lateral force, slip angle)
- Rigid body dynamics (sprung mass pitch/roll/heave equations of motion)
- Road surface inputs (bump profiles, random roughness)

The kinematic solver becomes the "geometry engine" inside a larger time-stepping simulation loop.
