# Vahan

Suspension simulation and optimization software for double-wishbone suspensions with pushrod/rocker actuation. Built for FSAE but applicable to any double-wishbone geometry.

![Main Window](screenshots/main_window.png)

## Features

### Forward Kinematics
- Full constraint-based solver (12 simultaneous equations, Newton-Raphson) for double-wishbone + pushrod/rocker
- 30+ kinematic metrics: camber, toe, caster, KPI, roll centre height, anti-dive/squat/lift, motion ratio, scrub radius, mechanical trail, and more
- Heave, roll, pitch, and steer sweep modes
- Four-corner solving with X-mirroring (FL/FR/RL/RR)

### Inverse Kinematics (Optimization)
- Define target curves (e.g. "camber from 0 deg at static to -2 deg at full bump") and let the solver find the hardpoints
- **Staged solver** decomposes the problem using orthogonal variable groups for ~5x speedup over brute-force hybrid DE+LM
- **Widened search** explores solution space at increasing perturbation levels to find alternative geometries
- Collision detection with configurable tube outer diameters — rejects solutions where suspension members intersect

### Steady-State Dynamics
- Load transfer (elastic + geometric + unsprung) with iterative roll convergence
- Degressive tire model with load sensitivity (`C_alpha ~ (Fz/Fz_ref)^n`, n < 1)
- Understeer gradient from back-calculated slip angles
- Friction circle tire utilization per corner
- Roll angle, pitch angle, LLTD

### Sensitivity & Optimization
- Central finite-difference sensitivity of any dynamic output to any vehicle parameter
- Practical step sizes (e.g. 1 mm spring preload, 1 N/mm spring rate)
- Recommendation engine: which parameter changes achieve a target understeer/roll/pitch delta

### Component Loads
- 6x6 static equilibrium on upright free body for all member axial forces
- Ball joint resultant forces decomposed into V (up+) and H (fwd+)
- Bearing loads at inner/outer bearings (V and H) via moment equilibrium
- Brake caliper mounting bolt forces (upper/lower, V and H) with direct shear + torque couple
- Separate front/rear brake parameters (pad mu, piston area, pad radius, bolt spacing)
- Brake system: torque, caliper clamp, line pressure

### 3D Visualization
- Interactive OpenGL viewport (VisPy) with full-car wireframe rendering
- Colour-coded suspension members (UCA, LCA, tie rod, pushrod, rocker, spring/damper)
- Roll centre and roll axis overlays
- Click-to-select hardpoint editing

### Kinematic Curves
- Live metric plots across wheel travel for all four corners
- Configurable graph selection from the full metrics catalog

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.12+, NumPy, SciPy, Matplotlib, PyQt6

Optional for 3D rendering: VisPy (`pip install vispy`)

## Usage

```bash
python app.py
```

The GUI opens with default FSAE hardpoints loaded. From there you can:

1. **Adjust hardpoints** — Edit coordinates directly in the side panels (mm)
2. **Run sweeps** — Choose heave/roll/pitch/steer and set travel range
3. **View metrics** — Select which kinematic metrics to plot
4. **Inverse solve** — Set target curves, select which hardpoints to adjust, and hit Solve
5. **Explore** — Run widened search to find alternative solutions across the design space
6. **Save/Load** — File menu for JSON geometry files

## Project Structure

```
vahan/                  Core computation library (no GUI dependencies)
  hardpoints.py           Hardpoint data class (14 points x 3 coords)
  solver.py               Forward kinematic constraint solver
  kinematics.py           Metric computation from solved states
  metrics_catalog.py      Registry of 30+ metrics with metadata
  analysis.py             Sweep orchestration (heave/roll/pitch/steer)
  optimizer.py            Inverse kinematics solver + collision detection
  tire_model.py           Linear tire model with load sensitivity
  dynamics.py             Steady-state vehicle dynamics solver
  loads.py                Component force calculator (members, bearings, brakes)

gui/                    PyQt6 desktop application
  main_window.py          Main window, panel wiring, sweep + dynamics logic
  panels.py               All sidebar panels (motion, IK, dynamics, loads, etc.)
  view3d.py               VisPy/OpenGL 3D viewport

app.py                  Entry point
DOCUMENTATION.md        Full technical documentation
```

## Technical Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed coverage of:
- Coordinate system and hardpoint definitions
- Constraint equations and Newton-Raphson solver
- All 30+ kinematic metrics with formulas
- Inverse solver architecture (staged decomposition, orthogonal groups)
- Collision detection algorithm
- Vehicle data (2026 FSAE car parameters)

## Collaborators

<!-- Add collaborator names/handles here -->

## License

Personal project by [@the-vedantin](https://github.com/the-vedantin).
