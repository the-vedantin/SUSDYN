# Vahan / 2027 — Roadmap & Future Projects

Quick notes, captured 2026-08-05. Rough backlog, not scheduled.

---

## Near-term (do next)

1. **Update + refine the binder** — bring the design binder current (latest geometry = v62, Ackermann −27.7% justification, etc.).
2. **Update + refine the Vahan git push** — clean up and push the current Vahan state (lots of uncommitted work: solver fixes, `ackermann_report.py`, UI de-yellow, screenshots).
3. **Data usage — validate/improve the sim against the 2026 car** — the data comes from the **2026 car**, so it will **not** translate 100% to the 2027 design. The value is **correlation**: compare 2026-car data ↔ the sim of the **2026 car**, and use the gap to **improve Vahan itself** (the software/model), not the 2027 design directly. (Ties to Future #3 sensor ingest + #7 curve-fit agent.)

---

## Future projects

### 1. Vahan → Onshape export: linkage thickness + diff position
Extend the Onshape FeatureScript export so it carries **linkage tube thickness** and **differential position**, not just the curves/points. (Onshape FS update.)

### 2. STEP import into Vahan
Import **STEP files** into Vahan for **collision detection / interference checks** and similar — bring real CAD solids into the model instead of the capsule approximations.

### 3. Data utilization (sensor ingest)
Ingest and use real car data:
- **Now:** linear potentiometers (damper travel), IMU, throttle.
- **Later:** steering angle, wheel speed, wheel temperature, tyre internal pressure.
Feed it back into the model (validation, correlation, tuning).

### 4. Chassis optimization *(tentative — MASSIVE)*
Simulate **chassis load under the various suspension load cases** to optimize **member placement and thickness**.
- **Hard constraint:** must comply with **FSAE rules during the optimization** (not just after).
- Big undertaking; flagged as its own major effort.

### 5. Simplify the software for potential OEM use
Replace the lengthy Python install with a **single `.exe`** that **works out of the box** — while still leaving the consumer **free to modify the app** to their needs.

### 6. Onshape Application — drop-in part-studio + pre-built assembly
A part-studio reference for **all geometry AND assembly development**.
- Linkages-as-curves are **pre-assembled in an Onshape assembly** with **solved mates and correct travel limits already set up**.
- The user just **drags and fastens components to the curves** — no need to build the assembly or its mates.
- The assembly **exists beforehand**, straight from the part studio → big time saver.

### 7. Data normalization + curve-fitting agent (for OEM)
An agent that takes **noisy data of any kind, from any sensor**, **curve-fits it**, and **applies it to the model** automatically. Aimed at the OEM product.

### 8. Security / IP / data-privacy research (pre-close-source)
Once OEM mods are made, the product **may go closed-source** → need **heavy security, intellectual-property, and data-privacy research** before that step.
