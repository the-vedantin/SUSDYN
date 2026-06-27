"""
gui/section_info.py — "Immense detail" descriptions for every GUI section.

Each entry is rich HTML rendered in a scrollable QTextBrowser when the user
clicks the ⓘ button on a section header (see CollapsibleSection.set_info).

The goal is that every box explains, in engineering terms:
  * WHAT the section controls / shows,
  * the PHYSICAL MODEL behind it (and its idealisations / limits),
  * how each field feeds the solver,
  * units + sign conventions,
  * gotchas.

Colour palette is colour-blind-safe (blue / yellow / white / red — no
orange, green, or purple), matching the user's vision profile.
    H_TITLE  blue    #4FC3F7   section title
    H_SUB    yellow  #FFD54F   sub-headings
    K        blue    #4FC3F7   field / key names
    WARN     red     #EF5350   warnings / limits
    body     white   #e0e0e0   (default text colour)

Axis convention everywhere: X = lateral (outboard +), Y = longitudinal
(forward = -Y), Z = up; SI metres internally.
"""

# ── shared style tokens ───────────────────────────────────────────────────────
_TITLE = 'color:#4FC3F7;'
_SUB   = 'color:#FFD54F;'
_K     = 'color:#4FC3F7;font-weight:bold;'
_WARN  = 'color:#EF5350;font-weight:bold;'


def _h(title: str) -> str:
    return f'<h3 style="{_TITLE}">{title}</h3>'


def _sub(text: str) -> str:
    return f'<h4 style="{_SUB}">{text}</h4>'


# ══════════════════════════════════════════════════════════════════════════════
#  STEERING — the "linear reduction" question lives here
# ══════════════════════════════════════════════════════════════════════════════

STEERING = _h('Steering Parameters (Front)') + f"""
<p>Converts <b>driver hand-wheel rotation</b> into <b>road-wheel steer
angle</b> through two stages.  The dynamics, transient, and skidpad solvers
all work internally in <i>road-wheel angle</i> (what the bicycle model needs);
this panel owns the mapping back to what the driver actually turns.</p>

<p style="{_SUB}">The conversion chain</p>
<pre style="color:#e0e0e0;">
 hand_wheel_deg
   x (rack_travel_per_rev_mm / 360)      &larr; STAGE 1: rack reduction (this panel)
   = rack_travel_mm
   x (d&delta;/d(rack)) from kinematics       &larr; STAGE 2: linkage geometry (your hardpoints)
   = road_wheel_angle
</pre>

{_sub('Stage 1 &mdash; the "linear reduction" (rack &amp; pinion)')}
<p><span style="{_K}">Rack travel/rev</span> (mm/rev) is the reduction you asked
about.  It is modelled as an <b>idealised rigid rack-and-pinion</b>:</p>
<table cellspacing="4">
<tr><td style="{_K}">Constant ratio</td>
<td>mm/rev does <b>not</b> change with pinion angle. One pinion of fixed pitch
radius r gives rack&nbsp;travel&nbsp;=&nbsp;2&pi;r per revolution, the same at
centre and at lock. The ratio is a straight line (hence "linear").</td></tr>
<tr><td style="{_K}">Bidirectional</td>
<td>Pushes and pulls identically &mdash; the gear teeth carry load both ways.</td></tr>
<tr><td style="{_K}">Infinitely stiff</td>
<td>No compliance, no backlash, no lash take-up. Hand-wheel angle maps to rack
position with zero lag.</td></tr>
</table>

<p style="{_WARN}">What this is NOT &mdash; a spool / cable (capstan) drive</p>
<p>You raised exactly the right distinction. A cable-and-spool reduction at the
<i>same nominal ratio</i> behaves differently, and Vahan does <b>not</b> model
any of it:</p>
<table cellspacing="4">
<tr><td style="{_K}">Progressive radius</td>
<td>As cable winds onto the drum, the effective radius grows with each layer, so
the ratio <i>changes with position</i> (rising-rate steering). The rack model is
flat by construction.</td></tr>
<tr><td style="{_K}">Tension-only</td>
<td>A string pulls but cannot push; a real spool drive needs two opposing cables
for two-way actuation. The rack model is inherently two-way.</td></tr>
<tr><td style="{_K}">Cable stretch</td>
<td>Cable elasticity adds a soft spring in series &rarr; lag and hysteresis under
load. The rack model has none.</td></tr>
</table>
<p>So if your real mechanism were a spool-and-cable, treating it as this
constant mm/rev rack is a first-order approximation that will miss the
position-varying ratio and the compliance. For a true rack &amp; pinion the model
is exact.</p>

{_sub('Stage 2 &mdash; where the nonlinearity actually lives')}
<p>Vahan captures steering nonlinearity in the <b>rack &rarr; road-wheel</b> step,
not the pinion step. <code>SteeringGeometry.from_probe</code> solves your real
tie-rod / steer-arm kinematics across the whole rack stroke and builds a dense
lookup &mdash; so bump-steer, Ackermann, and the falling gain near lock are all
reproduced from <i>your</i> hardpoints. (A pure-linear fallback,
<code>from_linear_ratio</code>, is used only when no corner solver is available,
e.g. headless init.)</p>

{_sub('Fields')}
<table cellspacing="4">
<tr><td style="{_K}">Rack travel/rev</td>
<td>mm of rack per full turn of the hand-wheel. Lower = faster steering (fewer
turns lock-to-lock). This is Stage-1 above.</td></tr>
<tr><td style="{_K}">Total rack travel</td>
<td>Full mechanical rack stroke (mm). Half of it (&plusmn;) is the physical limit;
the readout converts it to max hand-wheel angle and turns lock-to-lock.</td></tr>
</table>
<table cellspacing="4">
<tr><td style="{_K}">Max hand-wheel</td>
<td>(total/2) / per_rev &times; 360. The driver-side steering lock implied by your
rack numbers; also bounds physical rack saturation in the transient solver.</td></tr>
</table>
<p style="color:#888;">Steer is driven from <b>Motion &gt; Steer</b>, not a slider
here. Changing rack mm/rev genuinely changes simulation output.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  MOTION / DAMPER — the OTHER reduction (motion ratio) is here
# ══════════════════════════════════════════════════════════════════════════════

MOTION = _h('Motion &amp; Damper') + f"""
<p>Drives the suspension through a <b>travel sweep</b> (heave / roll / pitch /
steer) and owns the damper-limit + static-sag bookkeeping. Everything the 3D
view and the metric plots show is the solver evaluated at the position set here.</p>

{_sub('Motion modes')}
<table cellspacing="4">
<tr><td style="{_K}">Heave</td><td>Both wheels move together (vertical, mm).
Pure ride travel.</td></tr>
<tr><td style="{_K}">Roll</td><td>Wheels move in opposition; the slider is
chassis roll handled as equal-and-opposite corner travel.</td></tr>
<tr><td style="{_K}">Pitch</td><td>Front/rear opposition (dive / squat).</td></tr>
<tr><td style="{_K}">Steer</td><td>Sweeps rack travel; converts the slider (deg)
through the Steering panel's reduction.</td></tr>
</table>

{_sub('Motion ratio &mdash; the reduction from wheel to damper')}
<p>The pushrod / rocker reduction is the <b>motion ratio</b>
MR&nbsp;=&nbsp;&delta;<sub>shock</sub>&nbsp;/&nbsp;&delta;<sub>wheel</sub>.
Unlike the steering rack, this is <b>not</b> assumed linear:</p>
<table cellspacing="4">
<tr><td style="{_K}">Geometry-faithful</td>
<td>MR is measured numerically by central difference about the operating travel
&mdash; MR(&tau;) = [&#8467;<sub>spring</sub>(&tau;+&Delta;) &minus;
&#8467;<sub>spring</sub>(&tau;&minus;&Delta;)] / 2&Delta;, &Delta; = 1&nbsp;mm
&mdash; from the <i>actual</i> rocker, pushrod, and spring positions your
hardpoints define.</td></tr>
<tr><td style="{_K}">Position-varying</td>
<td>It changes through travel (rising/falling rate), exactly the behaviour a
constant ratio would miss. Wheel rate
K<sub>w</sub>&nbsp;=&nbsp;K<sub>spring</sub>&nbsp;&times;&nbsp;MR&sup2;, so MR
curvature is what makes your wheel rate progressive.</td></tr>
</table>
<p>So Vahan carries two different reduction philosophies on purpose: the
steering rack is an idealised constant (Stage 1 in the Steering box), while the
damper motion ratio is the true, nonlinear, geometry-derived ratio.</p>

{_sub('Damper limits &amp; sag (read-only sag)')}
<table cellspacing="4">
<tr><td style="{_K}">Stroke</td><td>Total damper stroke, shock-frame mm.</td></tr>
<tr><td style="{_K}">Preload F / R</td><td>Spring preload per axle (mm of shock).</td></tr>
<tr><td style="{_K}">Fully ext.</td>
<td>Eye-to-eye length at zero compression. 0 disables the shift diagnostic; when
nonzero, the app computes how far the CAD damper sits from physics-static ride
height.</td></tr>
</table>
<p>Static sag is <b>computed</b> from spring rate, sprung mass, MR, and preload
&mdash; never typed in. <span style="{_WARN}">Apply to HPs</span> is the only thing
that moves hardpoints; changing damper numbers alone never shifts geometry.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  CAR PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

CAR_PARAMS = _h('Car Parameters') + f"""
<p>Chassis-level geometry, tyre envelope, and mass centre. These numbers scale
the hardpoints and feed every dynamics calculation. They are the frame your
corner geometry lives inside.</p>

{_sub('Geometry')}
<table cellspacing="4">
<tr><td style="{_K}">Axle spacing</td>
<td>CAD distance between the front and rear suspension subassemblies. Changing it
<b>physically shifts the rear hardpoints</b> in Y. This is a geometry/packaging
number.</td></tr>
<tr><td style="{_K}">Wheelbase</td>
<td>Contact-patch-to-contact-patch distance front&harr;rear. This is the
<b>dynamics</b> number: it sets longitudinal load transfer, pitch, and the
turn/Ackermann geometry. <span style="{_WARN}">Not the same as axle spacing</span>
&mdash; axle spacing moves CAD points, wheelbase is what the vehicle model uses.</td></tr>
<tr><td style="{_K}">Track width F / R</td>
<td>Lateral contact-patch separation per axle. Drives lateral load transfer and
roll geometry. By default only the outboard pickups move when you change it; tick
<i>track pushes inboard</i> to translate the whole corner subassembly instead
(keeps arm length/angle constant).</td></tr>
<tr><td style="{_K}">Wheel offset F / R</td>
<td>How far the wheel centre sits laterally past the outboard pickups (mm).</td></tr>
</table>

{_sub('Tyre &amp; mass')}
<table cellspacing="4">
<tr><td style="{_K}">Tyre OD / Rim dia / width</td>
<td>Tyre envelope (mm). OD sets the loaded radius used for speed and longitudinal
force; the tyre carcass also adds vertical stiffness <i>in series</i> with the
wheel rate (ride-rate softening).</td></tr>
<tr><td style="{_K}">CG X / Y / Z</td>
<td>Centre of gravity: X lateral, Y longitudinal (sets static front/rear weight
split), Z height. <b>CG height</b> is the dominant lever for load transfer in
both roll and pitch.</td></tr>
<tr><td style="{_K}">Front brake bias</td>
<td>Front share of total brake torque (%). Feeds the brake/longitudinal models.</td></tr>
</table>

{_sub('Hardware widths (render + clearance)')}
<table cellspacing="4">
<tr><td style="{_K}">Rack length</td><td>Sets the inboard rack endpoints.</td></tr>
<tr><td style="{_K}">Spring OD / Damper OD</td>
<td>Coil and body diameters &mdash; drive the 3D model and clearance checks, not the
physics.</td></tr>
<tr><td style="{_K}">Show ground grid</td><td>Toggles the reference floor in the 3D view.</td></tr>
</table>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HARDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

HARDPOINTS = _h('Hardpoints (editable table)') + f"""
<p>The <b>geometric source of truth</b> for the whole program. Every joint of the
suspension is an XYZ coordinate here; the kinematic solver builds its link-length
constraints from these points, and the 3D view, loads, and dynamics all derive
from them. Edit a cell and the solver re-runs.</p>

{_sub('Axis convention (all values mm)')}
<table cellspacing="4">
<tr><td style="{_K}">X</td><td>Lateral &mdash; outboard positive.</td></tr>
<tr><td style="{_K}">Y</td><td>Longitudinal &mdash; forward is &minus;Y.</td></tr>
<tr><td style="{_K}">Z</td><td>Vertical &mdash; up positive.</td></tr>
</table>

{_sub('Row categories (colour-coded)')}
<p>The table shows up to four blocks; empty ones are hidden. The colour tells you
the category &mdash; here it is in words so you never have to rely on the hue:</p>
<table cellspacing="4">
<tr><td style="{_K}">Corner (blue / red)</td>
<td>Wishbones, tie rod, pushrod, rocker. <b>Blue = chassis (inboard)</b> pickups
that bolt to the frame; <b>red = outboard</b> points that define the upright.</td></tr>
<tr><td style="{_K}">ARB (amber)</td>
<td>Anti-roll device hardware (bellcrank / T-bar / control-arm).</td></tr>
<tr><td style="{_K}">Heave (green)</td>
<td>Third-element heave bracket &mdash; HEAVE_TBAR topology only.</td></tr>
<tr><td style="{_K}">Decoupled (purple)</td>
<td>Twin-bellcrank cradle + cross-car damper attaches &mdash; DECOUPLED topology only.</td></tr>
</table>

<p style="{_WARN}">Inboard vs outboard under track change</p>
<p>Inboard (chassis) points stay bolted to the frame when you change track width,
so the control arms get longer/steeper &mdash; unless <i>track pushes inboard</i> is
on (Car Parameters), which translates the whole corner. Outboard points are
rigidly the upright.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE VALUES
# ══════════════════════════════════════════════════════════════════════════════

VALUES = _h('Live Values') + f"""
<p><b>Show Live Values</b> opens a table of every kinematic metric for all four
corners (FL / FR / RL / RR), evaluated by the real solver at the <i>current</i>
Motion position. Move the Motion slider and the numbers track it live.</p>

{_sub('What it shows')}
<table cellspacing="4">
<tr><td style="{_K}">All catalogue metrics</td>
<td>Camber, toe, caster, trail, roll-centre height, anti-geometry, motion ratio,
&hellip; &mdash; the same quantities you can plot, but as a live numeric snapshot.</td></tr>
<tr><td style="{_K}">Max bump travel</td>
<td>Compression remaining at the wheel = (stroke &minus; sag) / MR.</td></tr>
<tr><td style="{_K}">Max droop travel</td>
<td>Extension remaining at the wheel = sag / MR.</td></tr>
</table>
<p>This is a pure read-out &mdash; it changes nothing. It reflects the solver pose, so
it is the fastest way to confirm the 3D view and the numbers agree.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH SELECTION
# ══════════════════════════════════════════════════════════════════════════════

GRAPH_PICKER = _h('Graph Selection') + f"""
<p>Controls the metric-vs-travel plot: which metrics are drawn and which corners
appear. Pure display selection &mdash; no physics.</p>

{_sub('Controls')}
<table cellspacing="4">
<tr><td style="{_K}">Corners</td>
<td>FL / FR / RL / RR check-boxes &mdash; show or hide each corner's curve.</td></tr>
<tr><td style="{_K}">Metric list</td>
<td>Grouped by category (kinematic, geometry, anti-, &hellip;). Tick any to add its
curve. Starts on a sensible default set.</td></tr>
<tr><td style="{_K}">All / None</td>
<td>Bulk select or clear the metric list.</td></tr>
</table>
<p>The travel axis itself is set by the Motion panel's range and mode.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

ALIGNMENT = _h('Alignment (Static)') + f"""
<p>Sets the car's <b>static</b> toe and camber by moving two specific hardpoints
per axle to hit your targets. <b>Apply Alignment</b> runs a Newton solve at the
static (zero-travel) position.</p>

{_sub('What moves')}
<table cellspacing="4">
<tr><td style="{_K}">Toe</td>
<td>Achieved by adjusting <b>tie-rod inner X</b> (rack-end position) until the
static toe matches your target.</td></tr>
<tr><td style="{_K}">Camber</td>
<td>Achieved by adjusting the <b>UCA pivot X</b> until static camber matches.</td></tr>
</table>

{_sub('Fields')}
<table cellspacing="4">
<tr><td style="{_K}">Front / Rear toe (deg)</td>
<td>Target static toe per axle.</td></tr>
<tr><td style="{_K}">Front / Rear camber (deg)</td>
<td>Target static camber per axle. Negative = top of the tyre leaning inboard.</td></tr>
</table>
<p style="{_WARN}">Scope</p>
<p>This is a targeted two-point set, not a full inverse-kinematics fit &mdash; it only
touches tie-rod-inner and the UCA pivot. For shaping a metric <i>across travel</i>
(bump-steer, camber curve), use Inverse Kinematics.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  INVERSE KINEMATICS
# ══════════════════════════════════════════════════════════════════════════════

INVERSE_KINEMATICS = _h('Inverse Kinematics') + f"""
<p>The reverse of normal analysis: instead of "given hardpoints, what is the
camber curve?", it asks <b>"given a target curve, where should the hardpoints
be?"</b> &mdash; and moves them to fit.</p>

{_sub('How it works')}
<p>A forward evaluator sweeps suspension travel and computes your chosen metric
from the live geometry; an optimiser (differential-evolution + Levenberg&ndash;
Marquardt hybrid) nudges the points you allow until the computed curve matches
the target, staying inside a per-point bound box. It works on <b>every
topology</b> &mdash; geometry metrics come purely from the wishbone + tie-rod solve,
so direct-acting and decoupled corners fit just as well as bell-crank ones.</p>

{_sub('You choose')}
<table cellspacing="4">
<tr><td style="{_K}">Metric</td>
<td>Camber, bump-steer (toe), anti-dive/squat/lift, Ackermann, RC height, caster,
trail, motion ratio, ARB motion ratio.</td></tr>
<tr><td style="{_K}">Target @Min &rarr; @Max</td>
<td>The metric value at min travel (droop) and max travel (bump). Constant =
both equal.</td></tr>
<tr><td style="{_K}">Curve</td>
<td>Shape between the endpoints: <b>Linear</b>, <b>Progressive</b> (τ^p &mdash;
rises late/deep in bump), <b>Digressive</b> (rises early then plateaus), or
<b>Exponential</b>. <b>Curvature</b> sets how sharp. This lets you target a
<b>NONLINEAR motion ratio</b> &mdash; a rising-rate MR whose wheel rate (∝ MR²)
stiffens under compression.</td></tr>
<tr><td style="{_K}">Hardpoints + axes</td>
<td>Which inboard/outboard points may move, and on which of X / Y / Z.</td></tr>
<tr><td style="{_K}">Bound</td>
<td>How far (mm) each point may travel from its current spot &mdash; the search box.</td></tr>
<tr><td style="{_K}">Motion</td>
<td>heave / roll / pitch / steer &mdash; the sweep the metric is measured over.</td></tr>
<tr><td style="{_K}">Method</td>
<td><b>local</b> = one least-squares from the current point (fast, repeatable);
<b>staged / hybrid / global</b> add differential-evolution exploration for tougher,
multi-modal fits.</td></tr>
</table>
<p>If several distinct geometries hit the target, a picker lists them with their
max error and total point movement so you can choose the least-disruptive one.</p>

{_sub('Nonlinear MR to hold ride height under aero')}
<p>A rising-rate MR can resist aero squat: as downforce compresses the
suspension, a higher MR stiffens the wheel rate and limits the heave. The
Laptime page's <b>Ride height</b> graph models this through the real MR(travel)
curve, and its <b>cap</b> readout gives the MR you need at compression
(<i>target_hi</i> for a progressive curve).</p>
<p style="{_WARN}">Positioning matters:</p>
<p>The rate rise must land <b>at the aero-compression stroke</b> (typically the
first 10&ndash;20 mm of bump), not deep in travel. A gently progressive curve
over the full travel barely helps ride height &mdash; the MR has hardly risen by
the time the aero load is reacted. Concentrate the rise near the operating
point (use a sharper Curvature, or set @Max close to the aero-heave travel).
A single rocker has limited freedom to hit a strong nonlinearity, so check the
residual on the graph &mdash; it shows how close the geometry actually got.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS  (palette-safe rewrite of the older orange help)
# ══════════════════════════════════════════════════════════════════════════════

DYNAMICS = _h('Dynamics (steady-state)') + f"""
<p>Solves the car's <b>steady-state equilibrium</b> at a chosen lateral and
longitudinal g, then reports per-corner loads, grip utilisation, roll/pitch, and
balance. The detailed per-field "?" reference is also on this panel.</p>

{_sub('The solve loop')}
<p>It iterates to self-consistency (2&ndash;3 passes):</p>
<pre style="color:#e0e0e0;">
 roll angle  &rarr;  per-corner travel  &rarr;  kinematic solve (RC migration, camber)
            &rarr;  load transfer  &rarr;  updated roll  &rarr; (repeat to convergence)
</pre>
<p><b>Sweep</b> runs the solve across many g-levels and plots every output &mdash;
lateral for an understeer/oversteer balance diagram, longitudinal for braking/
accel load shift.</p>

{_sub('Key inputs')}
<table cellspacing="4">
<tr><td style="{_K}">Masses</td>
<td>Total, sprung, and unsprung F/R. Sprung = supported by the springs; unsprung
acts at wheel-centre height.</td></tr>
<tr><td style="{_K}">Spring / tyre / ARB rate</td>
<td>Spring rate is at the spring (lbf/in); wheel rate = spring &times; MR&sup2; (MR read
from geometry). Tyre rate is in series. ARB rate is the equivalent single-wheel
roll rate.</td></tr>
<tr><td style="{_K}">g, power, RPM, ratios, radius</td>
<td>Operating point. Turn radius + speed (from RPM &times; ratio &times; tyre radius)
auto-derive lateral g = v&sup2;/(R&middot;g).</td></tr>
<tr><td style="{_K}">Drivetrain</td>
<td>RWD / FWD / AWD &mdash; which axle makes traction force.</td></tr>
</table>
<p>Motion ratio, RC heights, and camber gains are <b>auto-sourced from your live
geometry</b> &mdash; not typed here.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  AERO LOAD TARGETS
# ══════════════════════════════════════════════════════════════════════════════

AERO = _h('Aero Load Targets') + f"""
<p>Answers: "how much extra vertical load (downforce) does each corner need to
reach a target tyre utilisation at this operating point?" It is a
<b>load-target calculator</b>, not a CFD model.</p>

{_sub('Inputs')}
<table cellspacing="4">
<tr><td style="{_K}">Lateral g / Long. g</td><td>The operating point to evaluate.</td></tr>
<tr><td style="{_K}">Target util</td>
<td>Desired tyre utilisation (used grip / available grip), 0&ndash;1. 0.80 = leave 20%
margin.</td></tr>
</table>

{_sub('Outputs')}
<table cellspacing="4">
<tr><td style="{_K}">Addl Fz / corner</td>
<td>Extra vertical load each tyre needs to be pulled up to the target utilisation.</td></tr>
<tr><td style="{_K}">Front / rear axle need, total</td>
<td>Summed deficits and the implied <b>rear aero bias %</b> &mdash; a starting point for
sizing wings / splitter balance.</td></tr>
</table>
<p><b>Sweep</b> repeats across g to show how the downforce requirement grows with
cornering speed.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT LOADS
# ══════════════════════════════════════════════════════════════════════════════

LOADS = _h('Component Loads') + f"""
<p>Turns the dynamic corner loads into <b>member and fastener forces</b> &mdash; what
you size tubes, rod-ends, bearings, and bolts against.</p>

{_sub('Model')}
<p>At the chosen condition it takes a free body of the corner/upright and solves
static equilibrium through the suspension links. The link set is
<b>topology-aware</b>:</p>
<table cellspacing="4">
<tr><td style="{_K}">5-link base</td>
<td>Two upper + two lower wishbone legs + tie rod carry the corner.</td></tr>
<tr><td style="{_K}">+ load path</td>
<td>Pushrod / pullrod for rocker cars; a <b>direct damper line as a 6th member</b>
for direct-acting cars; the <b>live twin-bellcrank cradle solver</b> for decoupled
cars.</td></tr>
</table>

{_sub('Outputs')}
<table cellspacing="4">
<tr><td style="{_K}">Member forces</td><td>Tension/compression in each tube + the rod/damper line.</td></tr>
<tr><td style="{_K}">Bearing &amp; caliper-bolt forces</td>
<td>With vertical/horizontal decomposition for bracket design.</td></tr>
</table>
<p>Front and rear brake parameters and upright geometry are entered separately so
braking forces enter the corner free body correctly.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  BRAKE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

BRAKE = _h('Brake Calculator') + f"""
<p>Sizes the hydraulic brake system and checks thermal margin for all four
corners at an operating point.</p>

{_sub('Chain modelled')}
<pre style="color:#e0e0e0;">
 pedal force x pedal ratio        &rarr; master-cylinder force
   / MC bore area                 &rarr; line pressure
   x caliper piston area          &rarr; clamp force
   x 2 x pad mu x effective radius &rarr; brake torque
   / tyre radius                  &rarr; brake force  &rarr; deceleration
</pre>

{_sub('Inputs')}
<table cellspacing="4">
<tr><td style="{_K}">Pedal ratio / Bias</td>
<td>Lever mechanical advantage; front share of total force (bias bar).</td></tr>
<tr><td style="{_K}">MC bore F / R</td>
<td>Master-cylinder bore diameters (5/8&Prime; = 15.87 mm). Bore ratio + bias set the
front/rear pressure split.</td></tr>
<tr><td style="{_K}">Pad mu, piston area, pad radius</td>
<td>Caliper/pad geometry that turns pressure into torque.</td></tr>
<tr><td style="{_K}">Rotor thermal</td>
<td>Rotor mass/dimensions &rarr; single-stop temperature rise from the kinetic energy
dumped.</td></tr>
</table>

{_sub('Outputs')}
<p>Line pressure, brake torque, clamp force, the <b>pedal force to lock</b> each
tyre (given its vertical load), and rotor &Delta;T per event.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

DYNAMICS_OPT = _h('Dynamics Optimizer') + f"""
<p>A <b>sensitivity analyser</b>: pick a dynamics metric and a target, and it
ranks every change &mdash; parameter tweaks <i>and</i> kinematic moves &mdash; that pushes
you toward it, with the side effects each one causes.</p>

{_sub('How it works')}
<p>At the operating point it perturbs each candidate input, re-runs the dynamics
solver, and measures d(metric)/d(input). Results are ranked by effectiveness so
you see the cheapest lever first, and the collateral effect on other metrics is
listed so you do not fix balance and wreck ride.</p>

{_sub('Inputs')}
<table cellspacing="4">
<tr><td style="{_K}">Analyze at (g lat / lon)</td><td>Operating point for the sensitivity sweep.</td></tr>
<tr><td style="{_K}">Turn radius</td><td>Used for the ideal-Ackermann reference.</td></tr>
<tr><td style="{_K}">Target metric + value</td><td>What to improve and the goal.</td></tr>
</table>
<p>Think of it as "what should I change next, and what will it cost me elsewhere?"</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SKIDPAD / TRANSIENT
# ══════════════════════════════════════════════════════════════════════════════

SKIDPAD = _h('Skidpad / Transient') + f"""
<p>The <b>time-domain</b> solver (the Dynamics panel is steady-state). It
integrates the vehicle through a manoeuvre &mdash; skidpad, step-steer, ramp-steer
&mdash; so you see build-up, overshoot, and settling, not just the end state.</p>

{_sub('Solve mode (closed-loop skidpad)')}
<p>On a fixed-radius path, speed and lateral-g are linked by v = &radic;(a<sub>y</sub>
&middot; g &middot; R), so you pin one and the other follows:</p>
<table cellspacing="4">
<tr><td style="{_K}">Target speed</td><td>You set mph; it reports the resulting lateral-g.</td></tr>
<tr><td style="{_K}">Target lateral-g</td><td>You set g; it derives the speed.</td></tr>
</table>

{_sub('Inputs')}
<table cellspacing="4">
<tr><td style="{_K}">Peak steer, ramp/sim duration, dt</td>
<td>Manoeuvre shape and integration time-step.</td></tr>
<tr><td style="{_K}">Damper bump / rebound F &amp; R</td>
<td>Damper coefficients (N&middot;s/m <i>at the shock</i>). They are combined with the
motion ratio (MR&sup2;) and track width into the roll-damping the model uses &mdash;
nothing downstream reads a raw roll-damping number.</td></tr>
<tr><td style="{_K}">Steer lag &tau;</td><td>First-order delay on steering input.</td></tr>
</table>
<p>Yaw/roll inertias (Izz, Ixx) and Ackermann % are auto-filled from geometry and
shown in the read-out.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

VEHICLE_CONSTANTS = _h('Vehicle Constants') + f"""
<p><b>Show Computed Constants</b> opens a read-only popup of everything the model
<i>derives</i> from your inputs &mdash; no inputs are shown, only outputs, with the
intermediate steps so each result traces back to its source.</p>

{_sub('What it lists')}
<table cellspacing="4">
<tr><td style="{_K}">Wheel rates</td><td>Spring rate &times; MR&sup2; per axle (and ride rate with the tyre in series).</td></tr>
<tr><td style="{_K}">Roll stiffness + LLTD</td>
<td>Front/rear roll rates (spring + ARB contributions) and the lateral
load-transfer distribution they imply.</td></tr>
<tr><td style="{_K}">Ride frequencies</td><td>Undamped natural frequency per axle &mdash; the classic ride-tuning number.</td></tr>
</table>
<p>Because every intermediate is shown, this popup is built to be
<b>cross-checked against a hand spreadsheet</b> &mdash; if a number looks wrong you can
see exactly which input drove it.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS & VALIDATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

ANALYSIS_PLOTS = _h('Analysis &amp; Validation Plots') + f"""
<p>A library of higher-level validation charts. Each button runs the <i>real</i>
solver across a sweep and renders a matplotlib figure you can copy or save &mdash;
these are the plots you put in a design report.</p>

{_sub('Available plots')}
<table cellspacing="4">
<tr><td style="{_K}">Brake-torque capacity</td><td>Achievable decel vs line pressure / pad &amp; rotor spec.</td></tr>
<tr><td style="{_K}">Ride frequency</td><td>Front/rear ride frequencies from rates + masses.</td></tr>
<tr><td style="{_K}">RC vs roll</td><td>Roll-centre migration through chassis roll.</td></tr>
<tr><td style="{_K}">Steering torque</td><td>Kingpin/​rack torque demand vs steer.</td></tr>
<tr><td style="{_K}">Ackermann demand / Fz&middot;Fy / rack-zero</td>
<td>Steering geometry vs the ideal-Ackermann target, including the rack position
that zeroes Ackermann error.</td></tr>
<tr><td style="{_K}">MMD</td><td>Milliken Moment Diagram &mdash; yaw moment vs lateral force for balance/stability.</td></tr>
<tr><td style="{_K}">Wheel-rate linearity</td><td>How constant the wheel rate stays across travel (MR&sup2; curvature).</td></tr>
<tr><td style="{_K}">Lateral load transfer</td><td>LLT distribution vs g.</td></tr>
</table>
<p>Each plot has its own small input block right above its button.</p>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECT EDIT MODE
# ══════════════════════════════════════════════════════════════════════════════

DIRECT_EDIT = _h('Direct Edit Mode') + f"""
<p>Keyboard-driven hardpoint editing straight in the 3D view &mdash; the fast way to
feel out a geometry change without typing coordinates.</p>

{_sub('Nudging a point')}
<table cellspacing="4">
<tr><td style="{_K}">Enable</td><td>The 3D viewer captures the keyboard while this is on.</td></tr>
<tr><td style="{_K}">Corner + Hardpoint</td><td>Pick FL/FR/RL/RR and the point by name.</td></tr>
<tr><td style="{_K}">W A S D</td><td>Move in the X&ndash;Y (ground) plane.</td></tr>
<tr><td style="{_K}">Q E</td><td>Move down/up the Z axis.</td></tr>
<tr><td style="{_K}">1&ndash;6</td><td>Step size: 0.1 / 0.5 / 1 / 2 / 5 / 10 mm.</td></tr>
<tr><td style="{_K}">Ctrl+Z</td><td>Undo the last nudge.</td></tr>
<tr><td style="{_K}">Mirror F&harr;R</td><td>Apply the same delta to the matching point on the other axle.</td></tr>
</table>
<p>Every nudge re-runs the solver and redraws &mdash; the geometry, metrics, and 3D
pose stay in lock-step.</p>

{_sub('Damper-actuation plane tools')}
<table cellspacing="4">
<tr><td style="{_K}">Tilt actuation plane</td>
<td>Rotate the <b>entire</b> actuation plane (pushrod + rocker + spring + ARB
drop-link) about an axis through a chosen <b>pivot hardpoint</b>, so the rocker
assembly stays coplanar while you re-orient it. You pick the pivot, the axis, and
the angle.</td></tr>
<tr><td style="{_K}">Snap axis to normal</td>
<td>Forces <code>rocker_axis_pt</code> so the pin axis is exactly normal to the
rocker plane (pivot + spring-point + pushrod-inner) &mdash; kills any skewed-pullrod
out-of-plane error in one click.</td></tr>
</table>
"""


DIFFERENTIAL = _h('Differential (Drexler FSAE LSD)') + f"""
<p>The differential splits drive (and engine-overrun) torque between the two
driven wheels. In a corner that split is asymmetric, which makes a
<b>yaw moment</b> &mdash; so the diff directly sets how the car
<b>over/understeers going into and out of corners</b>. The locking %% is the
tuning knob.</p>

{_sub('The three effects')}
<table cellspacing="4">
<tr><td style="{_K}">On power (corner exit)</td>
<td>The clutch biases drive toward the slower <b>inner</b> wheel, so the axle's
force asymmetry <b>resists the turn &rarr; understeer</b>. More power-lock = more
exit understeer (and more traction off the corner).</td></tr>
<tr><td style="{_K}">On coast (corner entry)</td>
<td>Off-throttle, the clutch biases overrun to the inner wheel, which
<b>resists the rear rotating &rarr; entry stability</b>. More coast-lock = less
lift-off rotation; less coast-lock = a pointier, rotation-happy entry.</td></tr>
<tr><td style="{_K}">Preload</td>
<td>The clutch is pre-clamped, so a baseline of that stabilising effect is
present <b>at all times</b>, even at neutral throttle.</td></tr>
</table>

{_sub('Inputs')}
<table cellspacing="4">
<tr><td style="{_K}">Type</td>
<td><b>Open</b> = no locking (free inner/outer; least understeer, least
traction). <b>Spool</b> = fully locked (max exit understeer + traction, draggy
on entry). <b>Salisbury</b> = the Drexler ramp/clutch LSD (tunable).</td></tr>
<tr><td style="{_K}">Ramp (pwr/coast)</td>
<td>You set the <b>ramp angles</b> &mdash; exactly how you adjust the real diff
(you swap ramps, you don't dial a %%). The first angle is the power side, the
second the coast side. Smaller angle = more aggressive = more lock.</td></tr>
<tr><td style="{_K}">Preload</td>
<td>Clutch pre-clamp torque (Drexler: 25&ndash;35 Nm).</td></tr>
</table>

{_sub('Drexler ramp &rarr; locking %% (from the 2017 manual)')}
<table cellspacing="4">
<tr><td style="{_K}">30&deg;</td><td>~88%%</td>
<td style="{_K}">40&deg;</td><td>~60%%</td>
<td style="{_K}">45&deg;</td><td>~51%%</td>
<td style="{_K}">50&deg;</td><td>~42%%</td>
<td style="{_K}">60&deg;</td><td>~29%%</td></tr>
</table>
<p>The 3 selectable options (power/coast): <b>40/50</b> (basic),
<b>45/60</b> (softest), <b>30/45</b> (most aggressive).</p>

{_sub('How to tune it')}
<table cellspacing="4">
<tr><td>Pushing wide on corner exit?</td><td>Soften the <b>power</b> ramp
(bigger angle, less lock) &rarr; option 45/60.</td></tr>
<tr><td>Snappy / loose on turn-in?</td><td>Add <b>coast</b> lock (smaller coast
angle) or more <b>preload</b> &rarr; calms entry rotation.</td></tr>
<tr><td>Want max rotation for a tight course?</td><td>Less coast lock / less
preload &rarr; pointier entry.</td></tr>
</table>

{_sub('Model + sign convention')}
<p>The clutch biases the driven-axle torque, making an inter-wheel force bias
ΔFx = lock%% &middot; drive-force (capped by the INNER wheel's grip from the
solve) and a yaw moment Mz = ΔFx &middot; track/2. That moment is reacted by a
front/rear lateral-force couple ΔFy = Mz / wheelbase, which shifts each axle's
slip angle &mdash; so it feeds <b>directly into the understeer gradient</b>.
<b>Positive = understeer (exit) / stabilising (entry).</b></p>
<p>This is the single model: the diff now moves the <b>understeer gradient</b>
on the Dynamics page, the lap detail pass, AND the <b>Diff yaw moment</b> graph
channel &mdash; they all derive from the same solve. The readout shows the
actual understeer-gradient shift (deg) vs an open diff, on exit and on entry.</p>

<p style="{_WARN}">Engine braking:</p>
<p>The coast ramp only acts on overrun torque through the diff, so it needs an
<b>Engine braking</b> input (crank Nm, closed throttle). Set it from your dyno
or feel &mdash; the brakes act at the wheels (outside the diff) and don't work
the coast ramp. The quasi-steady lap-time <i>number</i> is balance-agnostic
(driver assumed at the grip limit), so use the understeer-gradient readout and
the yaw channel to tune balance, not the lap clock.</p>
"""


# Registry so panels can fetch by key and a smoke test can confirm coverage.
SECTION_INFO = {
    'differential':       DIFFERENTIAL,
    'motion':             MOTION,
    'steering':           STEERING,
    'car_params':         CAR_PARAMS,
    'hardpoints':         HARDPOINTS,
    'values':             VALUES,
    'graph_picker':       GRAPH_PICKER,
    'alignment':          ALIGNMENT,
    'inverse_kinematics': INVERSE_KINEMATICS,
    'dynamics':           DYNAMICS,
    'aero':               AERO,
    'loads':              LOADS,
    'brake':              BRAKE,
    'dynamics_opt':       DYNAMICS_OPT,
    'skidpad':            SKIDPAD,
    'vehicle_constants':  VEHICLE_CONSTANTS,
    'analysis_plots':     ANALYSIS_PLOTS,
    'direct_edit':        DIRECT_EDIT,
}
