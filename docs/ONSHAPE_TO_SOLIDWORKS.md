# Suspension Hardpoints — Onshape → SolidWorks (STEP)

**Audience: SolidWorks users pulling the car's suspension geometry out of this Onshape document.**

## What this document is

This Onshape document (**Suspoints**) holds the 2027 car's suspension **hardpoints** as a
live FeatureScript called **"Suspension Points"**. It builds all four corners — upper/lower
control-arm points, tie-rod / toe-link, pushrod, rocker, ARB, and wheel centers — as points,
then connects them with **curves** (the linkage centerlines). The wheels are solid bodies.

Current geometry: **v63** (see the version list — "Versions and history" on the left toolbar).

The two tabs at the bottom of the window:
- **ME AND MY GANG HATE DASSAULT** — the Part Studio that contains the geometry. Export from here.
- **CAD Imports** — scratch tab for pulling other CAD in.

## Why you need this recipe

Two gotchas, both already solved below:

1. **Onshape can't export sketch _points_ through STEP.** Only the **curves** survive. That's
   fine — you build your solids to the curves, not the points.
2. **SolidWorks drops the curves by default.** On import it will only show the wheel solids
   unless you turn on one option. That option is in Step 2 below.

---

## Step 1 — Export from Onshape

1. At the **bottom** of the Onshape window, **right-click the Part Studio tab**
   ("ME AND MY GANG HATE DASSAULT").
2. Choose **Export**.
3. **Format = STEP** (`.step`).
4. Check **"Y axis up"** — so the car comes in upright in SolidWorks instead of rotated.
5. Export. You get a file like `2027_V63_Hardpoints_Step_Export.step`.

> Only the curves + wheel solids are in the file. The construction points are not — expected.

## Step 2 — Import into SolidWorks

1. **Open** the `.step` file in SolidWorks.
2. In the import prompt, choose **"Import as graphical body"** (check that box).
3. Click **Options**, and turn on **"Import free curves and points as sketch"**.
4. **OK.**

You should now see the four wheels **plus the linkage curves as sketch geometry**. Build your
control arms, tie rods, pushrods, etc. to those curves.

> **If you only see the wheels and no linkage curves, you skipped Step 2.3** — re-open with
> "Import free curves and points as sketch" turned on.

---

## Units (read this)

- The Onshape **DOCUMENT** is set to **inches**.
- The FeatureScript "Suspension Points" defines its coordinates in **millimeters** — each value
  carries its own mm unit, so the geometry resolves correctly no matter the document units.
- The STEP export therefore carries true size. If SolidWorks asks on import, confirm units so
  nothing gets scaled.

## Notes

- Every time the geometry changes in Onshape, a new **version** is created (named `v6x — …`).
  Re-export from the newest version so you're not building to stale points.
- The hardpoints originate from the Vahan suspension model (single source of truth); this
  Onshape doc is downstream of it.
