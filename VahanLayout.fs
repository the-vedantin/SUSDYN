// VahanLayout.fs  —  Blank labeled suspension workspace for Onshape
//
// Creates one editable sketch per kinematic group with all hardpoints
// pre-named and placed at rough FSAE placeholder positions.
//
// Workflow:
//   1. Add this feature to a Part Studio (no inputs needed beyond corner label).
//   2. Open each sketch → drag/constrain points to your design positions.
//   3. To reorient a sketch to a real arm plane: right-click the sketch in the
//      feature tree → Edit → change the sketch plane to any face or datum.
//   4. When done, copy coordinates back to Vahan (VahanExport.fs, coming soon).
//
// Groups:  uca | lca | rocker | steering | arb | misc
//
// Axis convention (matches Vahan): X = lateral (outboard +), Y = long (fwd +), Z = up (+)
// Sketch coordinates on the shared front-view plane: u = X lateral, v = Z vertical
//
// NOTE: Replace FeatureScript version numbers with whatever your Feature
// Studio auto-generates when you create a new FS file.

FeatureScript 2931;
import(path : "onshape/std/common.fs", version : "2931.0");

// ── Corner label ──────────────────────────────────────────────────────────────
// Only used so VahanExport.fs can identify which corner these points belong to.

export enum VahanCorner
{
    annotation { "Name" : "Front Left  (FL)" } FL,
    annotation { "Name" : "Front Right (FR)" } FR,
    annotation { "Name" : "Rear Left   (RL)" } RL,
    annotation { "Name" : "Rear Right  (RR)" } RR
}

// ── Feature ───────────────────────────────────────────────────────────────────

annotation { "Feature Type Name" : "Vahan Layout" }
export const vahanLayout = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Corner" }
        definition.corner is VahanCorner;
    }
    {
        // ── Shared workspace plane ─────────────────────────────────────────────
        // All six sketches start on this XZ front-view plane.
        //   normal = -Y  (looking rearward = standard suspension front view)
        //   xAxis  = +X  (lateral outboard)
        //   v-axis = cross(-Y, X) = +Z  (vertical up)
        //
        // To reorient any sketch: right-click it → Edit → change the sketch plane.
        // The workspace plane itself is just an initial scaffold; change it freely.

        var wsPln = plane(
            vector(0, 0, 0) * meter,   // origin at world origin
            vector(0, -1, 0),           // normal = -Y (front view)
            vector(1,  0, 0)            // x-axis = +X (lateral outboard)
        );
        opPlane(context, id + "ws", { "plane" : wsPln, "size" : 0.6 * meter });
        var wsQ = qCreatedBy(id + "ws", EntityType.FACE);

        // Placeholder positions are roughly proportional to a typical FSAE car (mm).
        // All u = X (lateral), v = Z (vertical).  Drag every point to your geometry.
        // front/rear arm points are spread in X so they are individually selectable;
        // their true Y (fore-aft) separation is defined via constraints after placement.

        // ── 1. UCA sketch ──────────────────────────────────────────────────────
        try silent
        {
            var sk = newSketch(context, id + "uca", { "sketchPlane" : wsQ });
            skPoint(sk, "uca_front", { "position" : vector(185, 263) * millimeter });
            skPoint(sk, "uca_rear",  { "position" : vector(225, 248) * millimeter });
            skPoint(sk, "uca_outer", { "position" : vector(483, 286) * millimeter });
            skSolve(sk);
        }

        // ── 2. LCA sketch ──────────────────────────────────────────────────────
        try silent
        {
            var sk = newSketch(context, id + "lca", { "sketchPlane" : wsQ });
            skPoint(sk, "lca_front", { "position" : vector(195, 124) * millimeter });
            skPoint(sk, "lca_rear",  { "position" : vector(235, 120) * millimeter });
            skPoint(sk, "lca_outer", { "position" : vector(533, 119) * millimeter });
            skSolve(sk);
        }

        // ── 3. Rocker sketch ────────────────────────────────────────────────────
        // Reorient this sketch to the rocker's actual swing plane once you know
        // the pivot axis direction.
        try silent
        {
            var sk = newSketch(context, id + "rocker", { "sketchPlane" : wsQ });
            skPoint(sk, "rocker_pivot",      { "position" : vector(213, 622) * millimeter });
            skPoint(sk, "rocker_axis_pt",    { "position" : vector(213, 647) * millimeter });
            skPoint(sk, "rocker_spring_pt",  { "position" : vector(207, 679) * millimeter });
            skPoint(sk, "pushrod_inner",     { "position" : vector(257, 647) * millimeter });
            skPoint(sk, "spring_chassis_pt", { "position" : vector( 16, 661) * millimeter });
            skSolve(sk);
        }

        // ── 4. Steering sketch ──────────────────────────────────────────────────
        try silent
        {
            var sk = newSketch(context, id + "steering", { "sketchPlane" : wsQ });
            skPoint(sk, "tie_rod_inner", { "position" : vector(219, 152) * millimeter });
            skPoint(sk, "tie_rod_outer", { "position" : vector(543, 171) * millimeter });
            skSolve(sk);
        }

        // ── 5. ARB sketch ───────────────────────────────────────────────────────
        try silent
        {
            var sk = newSketch(context, id + "arb", { "sketchPlane" : wsQ });
            skPoint(sk, "arb_pivot",    { "position" : vector(195, 558) * millimeter });
            skPoint(sk, "arb_arm_end",  { "position" : vector(238, 558) * millimeter });
            skPoint(sk, "arb_drop_top", { "position" : vector(238, 621) * millimeter });
            skSolve(sk);
        }

        // ── 6. Misc sketch ──────────────────────────────────────────────────────
        try silent
        {
            var sk = newSketch(context, id + "misc", { "sketchPlane" : wsQ });
            skPoint(sk, "wheel_center",  { "position" : vector(559, 203) * millimeter });
            skPoint(sk, "pushrod_outer", { "position" : vector(438, 320) * millimeter });
            skSolve(sk);
        }
    }
);
