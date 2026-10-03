SUSPENSION HARDPOINTS - Onshape to SolidWorks (STEP)
=====================================================

Audience: SolidWorks users pulling the car's suspension geometry out of
this Onshape document.


WHAT THIS DOCUMENT IS
---------------------
This Onshape document (Suspoints) holds the 2027 car's suspension HARDPOINTS
as a live FeatureScript called "Suspension Points". It builds all four corners
- upper/lower control-arm points, tie-rod / toe-link, pushrod, rocker, ARB,
and wheel centers - as points, then connects them with CURVES (the linkage
centerlines). The wheels are solid bodies.

Current geometry: v63 (see "Versions and history" on the left toolbar).

The tab that holds the suspension points is:
    "ME AND MY GANG HATE DASSAULT"   <-- export from here.


WHY YOU NEED THIS RECIPE
------------------------
Two gotchas, both already solved below:

1. Onshape can't export sketch POINTS through STEP. Only the CURVES survive.
   That's fine - you build your solids to the curves, not the points.

2. SolidWorks drops the curves by default. On import it will only show the
   wheel solids unless you turn on one option (Step 2.3 below).


STEP 1 - EXPORT FROM ONSHAPE
----------------------------
1. At the BOTTOM of the Onshape window, RIGHT-CLICK the Part Studio tab
   ("ME AND MY GANG HATE DASSAULT").
2. Choose Export.
3. Format = STEP  (.step)
4. Check "Y axis up"  - so the car comes in upright in SolidWorks, not rotated.
5. Export. You get a file like  2027_V63_Hardpoints_Step_Export.step

   (Only the curves + wheel solids are in the file. The construction points
   are not - that's expected.)


STEP 2 - IMPORT INTO SOLIDWORKS
-------------------------------
1. OPEN the .step file in SolidWorks.
2. In the import prompt, choose "Import as graphical body" (check that box).
3. Click Options, and turn on "Import free curves and points as sketch".
4. OK.

You should now see the four wheels PLUS the linkage curves as sketch geometry.
Build your control arms, tie rods, pushrods, etc. to those curves.

   If you only see the wheels and no linkage curves, you skipped Step 2.3 -
   re-open with "Import free curves and points as sketch" turned on.


UNITS (read this)
-----------------
- The Onshape DOCUMENT is set to INCHES.
- The FeatureScript "Suspension Points" defines its coordinates in MILLIMETERS - each
  value carries its own mm unit, so the geometry resolves correctly no matter the
  document units.
- The STEP export therefore carries true size. If SolidWorks asks on import, confirm
  units so nothing gets scaled.


NOTES
-----
- Every time the geometry changes in Onshape, a new VERSION is created
  (named "v6x - ..."). Re-export from the newest version so you're not
  building to stale points.
- The hardpoints originate from the Vahan suspension model (single source of
  truth); this Onshape doc is downstream of it.
