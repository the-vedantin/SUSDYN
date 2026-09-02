# Rule 10 — Ground contact

## The rule
At design ride height, each axle's tire bottom sits on the ground plane (Z = 0).
The tire bottom is the wheel-centre height minus the tire radius, and it must be
zero to within tolerance.

## Why
The hardpoints are authored with the tires touching the ground at design. If a
config floats the car above the ground, everything measured from ride height
(roll-centre height, anti-squat, aero attitude, camber under load) is computed at
the wrong height and is silently wrong. A CAD export of a floating car is a
build error the moment it reaches the shop.

## The trap this rule closes
Applying suspension sag once baked the sag shift into the hardpoints without
re-grounding the car — the rear sat 14.6 mm high from v41 onward. Apply-Sag now
re-grounds, a warning fires on loading a floating config, and the net gates it.

## How Vahan holds it
- Apply-Sag re-grounds the car after baking sag.
- The regression net's "ground contact" gate checks the design config and the
  experiment configs sit on Z = 0 within tolerance, and lists the archived
  pre-fix configs that still float as a documented known-fail.
- A config saved from a pre-fix base inherits the float; ground-check any such
  config before relying on it.

## Tolerance
Tire bottom within 3 mm of the ground plane. Front around 1.5 mm and rear near
0.0 mm are the current design-config values.
