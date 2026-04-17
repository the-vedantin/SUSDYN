// Vahan Hardpoints -- Suspension Point Cloud for Onshape
//
// Paste hardpoint data from Vahan "Copy for Onshape" button,
// or adjust individual coordinates manually (clear Paste Data first).
//
// Coordinate system: X = lateral, Y = longitudinal, Z = up (same as Vahan)
//
// NOTE: Replace the FeatureScript version numbers with whatever your
// Feature Studio auto-generates when you create a new one.

FeatureScript 2931;
import(path : "onshape/std/common.fs", version : "2931.0");

// Coordinate bounds: covers any FSAE car
export const HP_BOUNDS = {
    (millimeter) : [-3000, 0, 3000]
} as LengthBoundSpec;

const HP_LIST = [
    "uca_front", "uca_rear", "uca_outer",
    "lca_front", "lca_rear", "lca_outer",
    "tie_rod_inner", "tie_rod_outer",
    "wheel_center",
    "pushrod_outer", "pushrod_inner",
    "rocker_pivot", "rocker_spring_pt", "spring_chassis_pt", "rocker_axis_pt",
    "arb_drop_top", "arb_arm_end", "arb_pivot"
];

const CORNERS = ["FL", "FR", "RL", "RR"];

// ---- String helpers -------------------------------------------------------

function splitByChar(text is string, delim is string) returns array
{
    var chars = splitIntoCharacters(text);
    var result = [];
    var current = "";
    for (var ch in chars)
    {
        if (ch == delim)
        {
            result = append(result, current);
            current = "";
        }
        else if (ch != "\r")
        {
            current = current ~ ch;
        }
    }
    result = append(result, current);
    return result;
}

// ---- Paste data parser ----------------------------------------------------
//
// Format (single line, pipe-separated records, comma-separated fields):
//   name,FL_X,FL_Y,FL_Z,FR_X,FR_Y,FR_Z,RL_X,RL_Y,RL_Z,RR_X,RR_Y,RR_Z|name,...
//
// Returns: map of { "uca_front" : { "FL" : { x, y, z }, "FR" : ..., ... }, ... }

function parsePasteData(text is string) returns map
{
    var result = {};
    var records = splitByChar(text, "|");
    for (var record in records)
    {
        if (length(record) == 0)
            continue;
        var fields = splitByChar(record, ",");
        if (size(fields) < 13)
            continue;
        var name = fields[0];
        if (name == "" || name == "Point")
            continue;
        var corners = {};
        for (var ci = 0; ci < 4; ci += 1)
        {
            var corner = CORNERS[ci];
            var xi = 1 + ci * 3;
            corners[corner] = {
                "x" : stringToNumber(fields[xi])     * millimeter,
                "y" : stringToNumber(fields[xi + 1]) * millimeter,
                "z" : stringToNumber(fields[xi + 2]) * millimeter
            };
        }
        result[name] = corners;
    }
    return result;
}

// ---- Feature definition ---------------------------------------------------

annotation { "Feature Type Name" : "Vahan Hardpoints",
             "Editing Logic Function" : "vahanHPEditLogic" }
export const vahanHardpoints = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Paste Data (from Vahan Copy for Onshape)", "MaxLength" : 10000 }
        definition.pasteData is string;

        annotation { "Name" : "Show FL", "Default" : true }
        definition.showFL is boolean;
        annotation { "Name" : "Show FR", "Default" : true }
        definition.showFR is boolean;
        annotation { "Name" : "Show RL", "Default" : true }
        definition.showRL is boolean;
        annotation { "Name" : "Show RR", "Default" : true }
        definition.showRR is boolean;

        annotation { "Group Name" : "FL - Front Left", "Collapsed By Default" : true }
        {
            annotation { "Name" : "uca front X" }
            isLength(definition.FL_uca_front_x, HP_BOUNDS);
            annotation { "Name" : "uca front Y" }
            isLength(definition.FL_uca_front_y, HP_BOUNDS);
            annotation { "Name" : "uca front Z" }
            isLength(definition.FL_uca_front_z, HP_BOUNDS);
            annotation { "Name" : "uca rear X" }
            isLength(definition.FL_uca_rear_x, HP_BOUNDS);
            annotation { "Name" : "uca rear Y" }
            isLength(definition.FL_uca_rear_y, HP_BOUNDS);
            annotation { "Name" : "uca rear Z" }
            isLength(definition.FL_uca_rear_z, HP_BOUNDS);
            annotation { "Name" : "uca outer X" }
            isLength(definition.FL_uca_outer_x, HP_BOUNDS);
            annotation { "Name" : "uca outer Y" }
            isLength(definition.FL_uca_outer_y, HP_BOUNDS);
            annotation { "Name" : "uca outer Z" }
            isLength(definition.FL_uca_outer_z, HP_BOUNDS);
            annotation { "Name" : "lca front X" }
            isLength(definition.FL_lca_front_x, HP_BOUNDS);
            annotation { "Name" : "lca front Y" }
            isLength(definition.FL_lca_front_y, HP_BOUNDS);
            annotation { "Name" : "lca front Z" }
            isLength(definition.FL_lca_front_z, HP_BOUNDS);
            annotation { "Name" : "lca rear X" }
            isLength(definition.FL_lca_rear_x, HP_BOUNDS);
            annotation { "Name" : "lca rear Y" }
            isLength(definition.FL_lca_rear_y, HP_BOUNDS);
            annotation { "Name" : "lca rear Z" }
            isLength(definition.FL_lca_rear_z, HP_BOUNDS);
            annotation { "Name" : "lca outer X" }
            isLength(definition.FL_lca_outer_x, HP_BOUNDS);
            annotation { "Name" : "lca outer Y" }
            isLength(definition.FL_lca_outer_y, HP_BOUNDS);
            annotation { "Name" : "lca outer Z" }
            isLength(definition.FL_lca_outer_z, HP_BOUNDS);
            annotation { "Name" : "tie rod inner X" }
            isLength(definition.FL_tie_rod_inner_x, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Y" }
            isLength(definition.FL_tie_rod_inner_y, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Z" }
            isLength(definition.FL_tie_rod_inner_z, HP_BOUNDS);
            annotation { "Name" : "tie rod outer X" }
            isLength(definition.FL_tie_rod_outer_x, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Y" }
            isLength(definition.FL_tie_rod_outer_y, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Z" }
            isLength(definition.FL_tie_rod_outer_z, HP_BOUNDS);
            annotation { "Name" : "wheel center X" }
            isLength(definition.FL_wheel_center_x, HP_BOUNDS);
            annotation { "Name" : "wheel center Y" }
            isLength(definition.FL_wheel_center_y, HP_BOUNDS);
            annotation { "Name" : "wheel center Z" }
            isLength(definition.FL_wheel_center_z, HP_BOUNDS);
            annotation { "Name" : "pushrod outer X" }
            isLength(definition.FL_pushrod_outer_x, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Y" }
            isLength(definition.FL_pushrod_outer_y, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Z" }
            isLength(definition.FL_pushrod_outer_z, HP_BOUNDS);
            annotation { "Name" : "pushrod inner X" }
            isLength(definition.FL_pushrod_inner_x, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Y" }
            isLength(definition.FL_pushrod_inner_y, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Z" }
            isLength(definition.FL_pushrod_inner_z, HP_BOUNDS);
            annotation { "Name" : "rocker pivot X" }
            isLength(definition.FL_rocker_pivot_x, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Y" }
            isLength(definition.FL_rocker_pivot_y, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Z" }
            isLength(definition.FL_rocker_pivot_z, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt X" }
            isLength(definition.FL_rocker_spring_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Y" }
            isLength(definition.FL_rocker_spring_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Z" }
            isLength(definition.FL_rocker_spring_pt_z, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt X" }
            isLength(definition.FL_spring_chassis_pt_x, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Y" }
            isLength(definition.FL_spring_chassis_pt_y, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Z" }
            isLength(definition.FL_spring_chassis_pt_z, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt X" }
            isLength(definition.FL_rocker_axis_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Y" }
            isLength(definition.FL_rocker_axis_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Z" }
            isLength(definition.FL_rocker_axis_pt_z, HP_BOUNDS);
            annotation { "Name" : "arb drop top X" }
            isLength(definition.FL_arb_drop_top_x, HP_BOUNDS);
            annotation { "Name" : "arb drop top Y" }
            isLength(definition.FL_arb_drop_top_y, HP_BOUNDS);
            annotation { "Name" : "arb drop top Z" }
            isLength(definition.FL_arb_drop_top_z, HP_BOUNDS);
            annotation { "Name" : "arb arm end X" }
            isLength(definition.FL_arb_arm_end_x, HP_BOUNDS);
            annotation { "Name" : "arb arm end Y" }
            isLength(definition.FL_arb_arm_end_y, HP_BOUNDS);
            annotation { "Name" : "arb arm end Z" }
            isLength(definition.FL_arb_arm_end_z, HP_BOUNDS);
            annotation { "Name" : "arb pivot X" }
            isLength(definition.FL_arb_pivot_x, HP_BOUNDS);
            annotation { "Name" : "arb pivot Y" }
            isLength(definition.FL_arb_pivot_y, HP_BOUNDS);
            annotation { "Name" : "arb pivot Z" }
            isLength(definition.FL_arb_pivot_z, HP_BOUNDS);
        }

        annotation { "Group Name" : "FR - Front Right", "Collapsed By Default" : true }
        {
            annotation { "Name" : "uca front X" }
            isLength(definition.FR_uca_front_x, HP_BOUNDS);
            annotation { "Name" : "uca front Y" }
            isLength(definition.FR_uca_front_y, HP_BOUNDS);
            annotation { "Name" : "uca front Z" }
            isLength(definition.FR_uca_front_z, HP_BOUNDS);
            annotation { "Name" : "uca rear X" }
            isLength(definition.FR_uca_rear_x, HP_BOUNDS);
            annotation { "Name" : "uca rear Y" }
            isLength(definition.FR_uca_rear_y, HP_BOUNDS);
            annotation { "Name" : "uca rear Z" }
            isLength(definition.FR_uca_rear_z, HP_BOUNDS);
            annotation { "Name" : "uca outer X" }
            isLength(definition.FR_uca_outer_x, HP_BOUNDS);
            annotation { "Name" : "uca outer Y" }
            isLength(definition.FR_uca_outer_y, HP_BOUNDS);
            annotation { "Name" : "uca outer Z" }
            isLength(definition.FR_uca_outer_z, HP_BOUNDS);
            annotation { "Name" : "lca front X" }
            isLength(definition.FR_lca_front_x, HP_BOUNDS);
            annotation { "Name" : "lca front Y" }
            isLength(definition.FR_lca_front_y, HP_BOUNDS);
            annotation { "Name" : "lca front Z" }
            isLength(definition.FR_lca_front_z, HP_BOUNDS);
            annotation { "Name" : "lca rear X" }
            isLength(definition.FR_lca_rear_x, HP_BOUNDS);
            annotation { "Name" : "lca rear Y" }
            isLength(definition.FR_lca_rear_y, HP_BOUNDS);
            annotation { "Name" : "lca rear Z" }
            isLength(definition.FR_lca_rear_z, HP_BOUNDS);
            annotation { "Name" : "lca outer X" }
            isLength(definition.FR_lca_outer_x, HP_BOUNDS);
            annotation { "Name" : "lca outer Y" }
            isLength(definition.FR_lca_outer_y, HP_BOUNDS);
            annotation { "Name" : "lca outer Z" }
            isLength(definition.FR_lca_outer_z, HP_BOUNDS);
            annotation { "Name" : "tie rod inner X" }
            isLength(definition.FR_tie_rod_inner_x, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Y" }
            isLength(definition.FR_tie_rod_inner_y, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Z" }
            isLength(definition.FR_tie_rod_inner_z, HP_BOUNDS);
            annotation { "Name" : "tie rod outer X" }
            isLength(definition.FR_tie_rod_outer_x, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Y" }
            isLength(definition.FR_tie_rod_outer_y, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Z" }
            isLength(definition.FR_tie_rod_outer_z, HP_BOUNDS);
            annotation { "Name" : "wheel center X" }
            isLength(definition.FR_wheel_center_x, HP_BOUNDS);
            annotation { "Name" : "wheel center Y" }
            isLength(definition.FR_wheel_center_y, HP_BOUNDS);
            annotation { "Name" : "wheel center Z" }
            isLength(definition.FR_wheel_center_z, HP_BOUNDS);
            annotation { "Name" : "pushrod outer X" }
            isLength(definition.FR_pushrod_outer_x, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Y" }
            isLength(definition.FR_pushrod_outer_y, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Z" }
            isLength(definition.FR_pushrod_outer_z, HP_BOUNDS);
            annotation { "Name" : "pushrod inner X" }
            isLength(definition.FR_pushrod_inner_x, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Y" }
            isLength(definition.FR_pushrod_inner_y, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Z" }
            isLength(definition.FR_pushrod_inner_z, HP_BOUNDS);
            annotation { "Name" : "rocker pivot X" }
            isLength(definition.FR_rocker_pivot_x, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Y" }
            isLength(definition.FR_rocker_pivot_y, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Z" }
            isLength(definition.FR_rocker_pivot_z, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt X" }
            isLength(definition.FR_rocker_spring_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Y" }
            isLength(definition.FR_rocker_spring_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Z" }
            isLength(definition.FR_rocker_spring_pt_z, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt X" }
            isLength(definition.FR_spring_chassis_pt_x, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Y" }
            isLength(definition.FR_spring_chassis_pt_y, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Z" }
            isLength(definition.FR_spring_chassis_pt_z, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt X" }
            isLength(definition.FR_rocker_axis_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Y" }
            isLength(definition.FR_rocker_axis_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Z" }
            isLength(definition.FR_rocker_axis_pt_z, HP_BOUNDS);
            annotation { "Name" : "arb drop top X" }
            isLength(definition.FR_arb_drop_top_x, HP_BOUNDS);
            annotation { "Name" : "arb drop top Y" }
            isLength(definition.FR_arb_drop_top_y, HP_BOUNDS);
            annotation { "Name" : "arb drop top Z" }
            isLength(definition.FR_arb_drop_top_z, HP_BOUNDS);
            annotation { "Name" : "arb arm end X" }
            isLength(definition.FR_arb_arm_end_x, HP_BOUNDS);
            annotation { "Name" : "arb arm end Y" }
            isLength(definition.FR_arb_arm_end_y, HP_BOUNDS);
            annotation { "Name" : "arb arm end Z" }
            isLength(definition.FR_arb_arm_end_z, HP_BOUNDS);
            annotation { "Name" : "arb pivot X" }
            isLength(definition.FR_arb_pivot_x, HP_BOUNDS);
            annotation { "Name" : "arb pivot Y" }
            isLength(definition.FR_arb_pivot_y, HP_BOUNDS);
            annotation { "Name" : "arb pivot Z" }
            isLength(definition.FR_arb_pivot_z, HP_BOUNDS);
        }

        annotation { "Group Name" : "RL - Rear Left", "Collapsed By Default" : true }
        {
            annotation { "Name" : "uca front X" }
            isLength(definition.RL_uca_front_x, HP_BOUNDS);
            annotation { "Name" : "uca front Y" }
            isLength(definition.RL_uca_front_y, HP_BOUNDS);
            annotation { "Name" : "uca front Z" }
            isLength(definition.RL_uca_front_z, HP_BOUNDS);
            annotation { "Name" : "uca rear X" }
            isLength(definition.RL_uca_rear_x, HP_BOUNDS);
            annotation { "Name" : "uca rear Y" }
            isLength(definition.RL_uca_rear_y, HP_BOUNDS);
            annotation { "Name" : "uca rear Z" }
            isLength(definition.RL_uca_rear_z, HP_BOUNDS);
            annotation { "Name" : "uca outer X" }
            isLength(definition.RL_uca_outer_x, HP_BOUNDS);
            annotation { "Name" : "uca outer Y" }
            isLength(definition.RL_uca_outer_y, HP_BOUNDS);
            annotation { "Name" : "uca outer Z" }
            isLength(definition.RL_uca_outer_z, HP_BOUNDS);
            annotation { "Name" : "lca front X" }
            isLength(definition.RL_lca_front_x, HP_BOUNDS);
            annotation { "Name" : "lca front Y" }
            isLength(definition.RL_lca_front_y, HP_BOUNDS);
            annotation { "Name" : "lca front Z" }
            isLength(definition.RL_lca_front_z, HP_BOUNDS);
            annotation { "Name" : "lca rear X" }
            isLength(definition.RL_lca_rear_x, HP_BOUNDS);
            annotation { "Name" : "lca rear Y" }
            isLength(definition.RL_lca_rear_y, HP_BOUNDS);
            annotation { "Name" : "lca rear Z" }
            isLength(definition.RL_lca_rear_z, HP_BOUNDS);
            annotation { "Name" : "lca outer X" }
            isLength(definition.RL_lca_outer_x, HP_BOUNDS);
            annotation { "Name" : "lca outer Y" }
            isLength(definition.RL_lca_outer_y, HP_BOUNDS);
            annotation { "Name" : "lca outer Z" }
            isLength(definition.RL_lca_outer_z, HP_BOUNDS);
            annotation { "Name" : "tie rod inner X" }
            isLength(definition.RL_tie_rod_inner_x, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Y" }
            isLength(definition.RL_tie_rod_inner_y, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Z" }
            isLength(definition.RL_tie_rod_inner_z, HP_BOUNDS);
            annotation { "Name" : "tie rod outer X" }
            isLength(definition.RL_tie_rod_outer_x, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Y" }
            isLength(definition.RL_tie_rod_outer_y, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Z" }
            isLength(definition.RL_tie_rod_outer_z, HP_BOUNDS);
            annotation { "Name" : "wheel center X" }
            isLength(definition.RL_wheel_center_x, HP_BOUNDS);
            annotation { "Name" : "wheel center Y" }
            isLength(definition.RL_wheel_center_y, HP_BOUNDS);
            annotation { "Name" : "wheel center Z" }
            isLength(definition.RL_wheel_center_z, HP_BOUNDS);
            annotation { "Name" : "pushrod outer X" }
            isLength(definition.RL_pushrod_outer_x, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Y" }
            isLength(definition.RL_pushrod_outer_y, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Z" }
            isLength(definition.RL_pushrod_outer_z, HP_BOUNDS);
            annotation { "Name" : "pushrod inner X" }
            isLength(definition.RL_pushrod_inner_x, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Y" }
            isLength(definition.RL_pushrod_inner_y, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Z" }
            isLength(definition.RL_pushrod_inner_z, HP_BOUNDS);
            annotation { "Name" : "rocker pivot X" }
            isLength(definition.RL_rocker_pivot_x, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Y" }
            isLength(definition.RL_rocker_pivot_y, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Z" }
            isLength(definition.RL_rocker_pivot_z, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt X" }
            isLength(definition.RL_rocker_spring_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Y" }
            isLength(definition.RL_rocker_spring_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Z" }
            isLength(definition.RL_rocker_spring_pt_z, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt X" }
            isLength(definition.RL_spring_chassis_pt_x, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Y" }
            isLength(definition.RL_spring_chassis_pt_y, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Z" }
            isLength(definition.RL_spring_chassis_pt_z, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt X" }
            isLength(definition.RL_rocker_axis_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Y" }
            isLength(definition.RL_rocker_axis_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Z" }
            isLength(definition.RL_rocker_axis_pt_z, HP_BOUNDS);
            annotation { "Name" : "arb drop top X" }
            isLength(definition.RL_arb_drop_top_x, HP_BOUNDS);
            annotation { "Name" : "arb drop top Y" }
            isLength(definition.RL_arb_drop_top_y, HP_BOUNDS);
            annotation { "Name" : "arb drop top Z" }
            isLength(definition.RL_arb_drop_top_z, HP_BOUNDS);
            annotation { "Name" : "arb arm end X" }
            isLength(definition.RL_arb_arm_end_x, HP_BOUNDS);
            annotation { "Name" : "arb arm end Y" }
            isLength(definition.RL_arb_arm_end_y, HP_BOUNDS);
            annotation { "Name" : "arb arm end Z" }
            isLength(definition.RL_arb_arm_end_z, HP_BOUNDS);
            annotation { "Name" : "arb pivot X" }
            isLength(definition.RL_arb_pivot_x, HP_BOUNDS);
            annotation { "Name" : "arb pivot Y" }
            isLength(definition.RL_arb_pivot_y, HP_BOUNDS);
            annotation { "Name" : "arb pivot Z" }
            isLength(definition.RL_arb_pivot_z, HP_BOUNDS);
        }

        annotation { "Group Name" : "RR - Rear Right", "Collapsed By Default" : true }
        {
            annotation { "Name" : "uca front X" }
            isLength(definition.RR_uca_front_x, HP_BOUNDS);
            annotation { "Name" : "uca front Y" }
            isLength(definition.RR_uca_front_y, HP_BOUNDS);
            annotation { "Name" : "uca front Z" }
            isLength(definition.RR_uca_front_z, HP_BOUNDS);
            annotation { "Name" : "uca rear X" }
            isLength(definition.RR_uca_rear_x, HP_BOUNDS);
            annotation { "Name" : "uca rear Y" }
            isLength(definition.RR_uca_rear_y, HP_BOUNDS);
            annotation { "Name" : "uca rear Z" }
            isLength(definition.RR_uca_rear_z, HP_BOUNDS);
            annotation { "Name" : "uca outer X" }
            isLength(definition.RR_uca_outer_x, HP_BOUNDS);
            annotation { "Name" : "uca outer Y" }
            isLength(definition.RR_uca_outer_y, HP_BOUNDS);
            annotation { "Name" : "uca outer Z" }
            isLength(definition.RR_uca_outer_z, HP_BOUNDS);
            annotation { "Name" : "lca front X" }
            isLength(definition.RR_lca_front_x, HP_BOUNDS);
            annotation { "Name" : "lca front Y" }
            isLength(definition.RR_lca_front_y, HP_BOUNDS);
            annotation { "Name" : "lca front Z" }
            isLength(definition.RR_lca_front_z, HP_BOUNDS);
            annotation { "Name" : "lca rear X" }
            isLength(definition.RR_lca_rear_x, HP_BOUNDS);
            annotation { "Name" : "lca rear Y" }
            isLength(definition.RR_lca_rear_y, HP_BOUNDS);
            annotation { "Name" : "lca rear Z" }
            isLength(definition.RR_lca_rear_z, HP_BOUNDS);
            annotation { "Name" : "lca outer X" }
            isLength(definition.RR_lca_outer_x, HP_BOUNDS);
            annotation { "Name" : "lca outer Y" }
            isLength(definition.RR_lca_outer_y, HP_BOUNDS);
            annotation { "Name" : "lca outer Z" }
            isLength(definition.RR_lca_outer_z, HP_BOUNDS);
            annotation { "Name" : "tie rod inner X" }
            isLength(definition.RR_tie_rod_inner_x, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Y" }
            isLength(definition.RR_tie_rod_inner_y, HP_BOUNDS);
            annotation { "Name" : "tie rod inner Z" }
            isLength(definition.RR_tie_rod_inner_z, HP_BOUNDS);
            annotation { "Name" : "tie rod outer X" }
            isLength(definition.RR_tie_rod_outer_x, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Y" }
            isLength(definition.RR_tie_rod_outer_y, HP_BOUNDS);
            annotation { "Name" : "tie rod outer Z" }
            isLength(definition.RR_tie_rod_outer_z, HP_BOUNDS);
            annotation { "Name" : "wheel center X" }
            isLength(definition.RR_wheel_center_x, HP_BOUNDS);
            annotation { "Name" : "wheel center Y" }
            isLength(definition.RR_wheel_center_y, HP_BOUNDS);
            annotation { "Name" : "wheel center Z" }
            isLength(definition.RR_wheel_center_z, HP_BOUNDS);
            annotation { "Name" : "pushrod outer X" }
            isLength(definition.RR_pushrod_outer_x, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Y" }
            isLength(definition.RR_pushrod_outer_y, HP_BOUNDS);
            annotation { "Name" : "pushrod outer Z" }
            isLength(definition.RR_pushrod_outer_z, HP_BOUNDS);
            annotation { "Name" : "pushrod inner X" }
            isLength(definition.RR_pushrod_inner_x, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Y" }
            isLength(definition.RR_pushrod_inner_y, HP_BOUNDS);
            annotation { "Name" : "pushrod inner Z" }
            isLength(definition.RR_pushrod_inner_z, HP_BOUNDS);
            annotation { "Name" : "rocker pivot X" }
            isLength(definition.RR_rocker_pivot_x, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Y" }
            isLength(definition.RR_rocker_pivot_y, HP_BOUNDS);
            annotation { "Name" : "rocker pivot Z" }
            isLength(definition.RR_rocker_pivot_z, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt X" }
            isLength(definition.RR_rocker_spring_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Y" }
            isLength(definition.RR_rocker_spring_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker spring pt Z" }
            isLength(definition.RR_rocker_spring_pt_z, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt X" }
            isLength(definition.RR_spring_chassis_pt_x, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Y" }
            isLength(definition.RR_spring_chassis_pt_y, HP_BOUNDS);
            annotation { "Name" : "spring chassis pt Z" }
            isLength(definition.RR_spring_chassis_pt_z, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt X" }
            isLength(definition.RR_rocker_axis_pt_x, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Y" }
            isLength(definition.RR_rocker_axis_pt_y, HP_BOUNDS);
            annotation { "Name" : "rocker axis pt Z" }
            isLength(definition.RR_rocker_axis_pt_z, HP_BOUNDS);
            annotation { "Name" : "arb drop top X" }
            isLength(definition.RR_arb_drop_top_x, HP_BOUNDS);
            annotation { "Name" : "arb drop top Y" }
            isLength(definition.RR_arb_drop_top_y, HP_BOUNDS);
            annotation { "Name" : "arb drop top Z" }
            isLength(definition.RR_arb_drop_top_z, HP_BOUNDS);
            annotation { "Name" : "arb arm end X" }
            isLength(definition.RR_arb_arm_end_x, HP_BOUNDS);
            annotation { "Name" : "arb arm end Y" }
            isLength(definition.RR_arb_arm_end_y, HP_BOUNDS);
            annotation { "Name" : "arb arm end Z" }
            isLength(definition.RR_arb_arm_end_z, HP_BOUNDS);
            annotation { "Name" : "arb pivot X" }
            isLength(definition.RR_arb_pivot_x, HP_BOUNDS);
            annotation { "Name" : "arb pivot Y" }
            isLength(definition.RR_arb_pivot_y, HP_BOUNDS);
            annotation { "Name" : "arb pivot Z" }
            isLength(definition.RR_arb_pivot_z, HP_BOUNDS);
        }
    }
    {
        // ------------------------------------------------------------------
        // SOURCE SELECTION
        // If paste data is present, parse it directly here in the feature
        // body so points are created regardless of whether editing logic fired.
        // If paste data is empty, fall back to individual coordinate params.
        // ------------------------------------------------------------------
        var hasPaste = (definition.pasteData != undefined && definition.pasteData != "");
        var parsed = {};
        if (hasPaste)
        {
            parsed = parsePasteData(definition.pasteData);
        }

        for (var ci = 0; ci < size(CORNERS); ci += 1)
        {
            var corner = CORNERS[ci];
            if (!definition["show" ~ corner])
                continue;

            for (var hi = 0; hi < size(HP_LIST); hi += 1)
            {
                var hp = HP_LIST[hi];
                var prefix = corner ~ "_" ~ hp;

                var x = 0 * millimeter;
                var y = 0 * millimeter;
                var z = 0 * millimeter;

                if (hasPaste)
                {
                    // Read from parsed paste data
                    var pt = try silent(parsed[hp][corner]);
                    if (pt == undefined)
                        continue;
                    x = pt.x;
                    y = pt.y;
                    z = pt.z;
                }
                else
                {
                    // Manual mode: read individual coordinate params
                    x = definition[prefix ~ "_x"];
                    y = definition[prefix ~ "_y"];
                    z = definition[prefix ~ "_z"];

                    // Skip params that were never set (all zero)
                    if (abs(x / millimeter) < 0.001 &&
                        abs(y / millimeter) < 0.001 &&
                        abs(z / millimeter) < 0.001)
                        continue;
                }

                opPoint(context, id + prefix, {
                    "point" : vector(x, y, z)
                });

                try silent
                {
                    setProperty(context, {
                        "entities"     : qCreatedBy(id + prefix, EntityType.BODY),
                        "propertyType" : PropertyType.NAME,
                        "value"        : corner ~ " " ~ hp
                    });
                }
            }
        }
    });

// ---- Editing logic --------------------------------------------------------
// Populates individual coordinate fields from paste data so the user can
// see and manually tweak values. Points are created directly from parsed
// data in the feature body above, so this is purely for UI feedback.

export function vahanHPEditLogic(context is Context, id is Id,
    oldDefinition is map, definition is map,
    isCreating is boolean, specifiedParameters is map) returns map
{
    if (definition.pasteData != undefined &&
        definition.pasteData != "" &&
        definition.pasteData != oldDefinition.pasteData)
    {
        var parsed = parsePasteData(definition.pasteData);
        for (var corner in CORNERS)
        {
            for (var hp in HP_LIST)
            {
                var pt = try silent(parsed[hp][corner]);
                if (pt == undefined)
                    continue;
                var prefix = corner ~ "_" ~ hp;
                if (specifiedParameters[prefix ~ "_x"] != true)
                    definition[prefix ~ "_x"] = pt.x;
                if (specifiedParameters[prefix ~ "_y"] != true)
                    definition[prefix ~ "_y"] = pt.y;
                if (specifiedParameters[prefix ~ "_z"] != true)
                    definition[prefix ~ "_z"] = pt.z;
            }
        }
    }
    return definition;
}
