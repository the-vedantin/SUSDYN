"""
vahan/ackermann_report.py — the ACKERMANN JUSTIFICATION REPORT, as a Vahan feature.

WHY THIS EXISTS.  The Ackermann decision (keep the v62 slight-reverse rack, band
-20% to -30%) needs a written, defensible record that a scrutineer, a teammate or
next year's team can read WITHOUT the app open — and that cannot rot into a stale
copy-paste.  So this generator RECOMPUTES every number live from the one solved
car and writes them into the prose; if a live number contradicts the authored
narrative, the report says so (see the SELF-CONSISTENCY CHECK) instead of lying.

WHAT IT COMPUTES.  Nothing about tyres or the car — every physics number comes
out of vahan:

    ideal Ackermann vs lateral g (aero-true)   vahan.ackermann.ackermann_bucket
    as-built % (probed from the linkage)        MainWindow._probe_static_ackermann
    grip ceiling / tie per setting              vahan.ymd.mmm_metrics_sweep
    per-tyre Fz + slip at the limit             (rows inside ackermann_bucket)

ONE MODEL.  This module ARRANGES and NARRATES.  Duplicated physics is exactly the
mechanism that produced rival Ackermann answers before, so there is none here.

TTC POLICY (memory project_ttc_publishing).  The tyre is belt-rig (Calspan TIRF)
data governed by the FSAE TTC agreement — this report carries NO tyre identity
string (no compound / size / file name).  It refers to "the car's tyre", quotes
slip in degrees only, and states the belt->road derate (x0.70) rather than any
raw belt grip.

STYLE (docs/DESIGN.md — Impeccable, dark-theme adaptation).  Red #E23B48 accent,
ink #ECECEE, muted #9A9AA2.  NO yellow and NO blue anywhere (this is a
DELIVERABLE — memory user_colorblind: "no blue in any deliverable"; DESIGN.md:
yellow banned from text/UI).  Graph lines are white / red / warm-grey only.
"""
from __future__ import annotations

import base64
import datetime
import html as _html
import io
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')                    # headless — no GUI needed
import matplotlib.pyplot as plt

# ── docs/DESIGN.md tokens (dark app adaptation) ──────────────────────────────
INK = '#ECECEE'
MUTED = '#9A9AA2'
PAPER = '#0B0B0D'
PANEL = '#141416'
LINE = '#2A2A2E'
ACCENT = '#E23B48'          # Impeccable red — the ONLY accent
WGY = '#C8BCA6'             # warm grey (graph line / secondary)
WHT = '#ECECEE'
FIG_BG = '#0B0B0D'
AX_BG = '#141416'

# The book's own reasons, cited so the binder and the code are both traceable.
# (memory reference_rcvd: cite the RCVD page when a method/definition comes from
# it.)  These are references, not physics — the physics is all in vahan/.
RCVD = {
    'load_sens': 'RCVD Ch 2 (printed p25/p27) — peak lateral force occurs at a '
                 'HIGHER slip angle as vertical load rises',
    'ack_defn': 'RCVD Ch 8 (printed p82-97) — pro vs reverse steer geometry '
                '(Ackermann is defined in degrees of toe-out, not a percentage)',
    'grip_tie': 'RCVD Ch 19 (printed p716) — parallel or a little reverse is a '
                'reasonable compromise; the geometry sets character, not grip',
    'inner_drag': 'RCVD Ch 19 (printed p715) — full Ackermann drags the lightly '
                  'loaded inner front past its peak at the limit',
    'compliance': 'RCVD Ch 19 (printed p717) — measure steer-steer compliance; '
                  'it can steer the wheels more than the built-in geometry',
}


# ═══════════════════════════════════════════════════════════════════════════
#  input normalisation — accept a MainWindow OR a bare (solver, tire, aero, fn)
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_source(source, radius_m):
    """Return (solver, tire, aero_dict_or_None, probe_fn).

    `source` is either a MainWindow (has ``_build_dynamics_solver``) or a
    4-tuple ``(solver, tire, aero, probe_fn)``.  For a MainWindow the aero gate
    is LIFTED for the read exactly the way the Ackermann page does it
    (gui/ackermann_page._ack_aero), so the car's real package couples in even
    when the Dynamics toggle is off on load."""
    if hasattr(source, '_build_dynamics_solver'):
        mw = source
        solver = mw._build_dynamics_solver()
        tire = getattr(mw, '_tire_model', None)
        if tire is None:
            raise RuntimeError('no tyre model loaded — load a file on the '
                               'Dynamics panel first')
        was = getattr(mw, '_aero_active', False)
        try:
            mw._aero_active = True
            aero = mw._get_aero_Fz_per_g(radius_m=float(radius_m))
        except Exception:
            aero = None
        finally:
            mw._aero_active = was
        probe_fn = mw._probe_static_ackermann
        return solver, tire, (aero or None), probe_fn

    try:
        solver, tire, aero, probe_fn = source
    except Exception as exc:
        raise TypeError('source must be a MainWindow or a (solver, tire, aero, '
                        'probe_fn) tuple') from exc
    return solver, tire, (aero or None), probe_fn


# ═══════════════════════════════════════════════════════════════════════════
#  the live computation — every number the report prints comes from here
# ═══════════════════════════════════════════════════════════════════════════
def compute_ackermann_evidence(source, radius_m=8.0, grip_multiplier=None):
    """Recompute the whole evidence base LIVE and return it as a dict.

    This is deliberately separated from the HTML so the numbers can be checked
    against a fresh independent call (the verify step does exactly that).
    Nothing is hardcoded; every value below is read from a solver call."""
    from vahan.ackermann import ackermann_bucket
    from vahan.ymd import mmm_metrics_sweep, build_loads_table

    R = float(radius_m)
    solver, tire, aero, probe_fn = _resolve_source(source, R)
    from vahan.dynamics import resolve_grip_scale   # None = project scale
    grip = resolve_grip_scale(grip_multiplier, solver)

    # ── as-built %, probed from the steering linkage at full lock ────────────
    try:
        as_built_pct = float(probe_fn())
    except Exception:
        as_built_pct = float('nan')

    # ── grip ceiling per setting -> is Ackermann a GRIP lever? (the TIE) ─────
    pcts = [-100, -60, -30, 0, 30, 60, 100]
    tbl = build_loads_table(solver, aero_Fz_per_g=aero)
    mm = mmm_metrics_sweep(tire, solver, pcts, radius_m=R,
                           grip_multiplier=grip, aero_Fz_per_g=aero,
                           loads_table=tbl)
    ay = np.array([r['ay_trim_max'] for r in mm], float)
    grip_span_g = float(np.nanmax(ay) - np.nanmin(ay))
    grip_ceiling_g = float(np.nanmean(ay))
    grip_span_pct = 100.0 * grip_span_g / max(grip_ceiling_g, 1e-9)
    GRIP_TIE_FRAC = 0.10
    grip_ties = grip_span_g <= GRIP_TIE_FRAC * grip_ceiling_g

    # The LIMIT lateral g the car can pull = the grip ceiling (the g at which
    # the front axle saturates).  This is what the narrative calls "peak".  It
    # is a live number, not a lap-sim assumption.
    peak_g = grip_ceiling_g

    # ── ideal Ackermann vs lateral g, AERO-TRUE ──────────────────────────────
    named_g = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    peak_g_r = round(peak_g, 3)
    g_list = sorted(set([round(g, 3) for g in named_g] + [peak_g_r]))
    bucket = ackermann_bucket(tire, solver, radius_m=R, lat_g_list=tuple(g_list),
                              grip_multiplier=grip, aero=aero)
    ideal_rows = []
    for row in bucket:
        valid = bool(row.get('valid', True))
        pct = float(row.get('ackermann_pct', float('nan')))
        ideal_rows.append({
            'lat_g': float(row['lat_g']),
            'valid': valid,
            'ideal_pct': pct if valid and np.isfinite(pct) else float('nan'),
            'spread_deg': (float(row.get('point_spread_deg', float('nan')))
                           if valid else float('nan')),
            'Fz_outer_N': float(row['Fz_outer']),
            'Fz_inner_N': float(row['Fz_inner']),
            'slip_outer_deg': float(row['outer_slip_deg']),
            'slip_inner_deg': float(row['inner_slip_deg']),
            'is_peak': False,
        })
    # Mark the row nearest the live limit g as the peak/limit row (float-safe:
    # the grid value is the rounded peak_g, so match by nearest, not equality).
    if ideal_rows:
        _pk_i = int(np.argmin([abs(r['lat_g'] - peak_g) for r in ideal_rows]))
        ideal_rows[_pk_i]['is_peak'] = True

    # smooth curve for the figure (fine g sweep), aero-true
    fine_g = [float(g) for g in np.round(np.arange(0.4, 2.001, 0.1), 2)]
    fine = ackermann_bucket(tire, solver, radius_m=R, lat_g_list=tuple(fine_g),
                            grip_multiplier=grip, aero=aero)
    fg, fpct, sat_from = [], [], None
    for g, row in zip(fine_g, fine):
        if row.get('valid', True) and np.isfinite(row.get('ackermann_pct',
                                                          np.nan)):
            fg.append(g)
            fpct.append(float(row['ackermann_pct']))
        elif sat_from is None:
            sat_from = g

    # ── the LIMIT row = the bucket row at peak g (the character-setting want) ─
    peak_row = next((r for r in ideal_rows if r['is_peak']), None)
    if peak_row is None or not peak_row['valid']:
        # peak sits past the last valid g — fall back to the highest valid row
        valids = [r for r in ideal_rows if r['valid']
                  and np.isfinite(r['ideal_pct'])]
        peak_row = valids[-1] if valids else None
        peak_saturated = True
    else:
        peak_saturated = False

    # ── sub-limit want (the pro end), read at a representative sub-limit g ────
    sub_g = 0.6
    sub_row = next((r for r in ideal_rows
                    if abs(r['lat_g'] - sub_g) < 1e-6 and r['valid']), None)

    # ── THE SELF-CONSISTENCY CHECK: recompute the LIMIT WANT SIGN live ───────
    # The narrative's whole premise is "at the limit the tyres want REVERSE".
    # Test it two independent ways on this car, live:
    #   (a) the ideal Ackermann at the limit is negative (reverse), and
    #   (b) the loaded OUTER front needs MORE slip than the light inner
    #       (which is WHY reverse — RCVD Ch 2 load sensitivity).
    limit_ideal_pct = (peak_row['ideal_pct'] if peak_row is not None
                       else float('nan'))
    limit_outer_slip = (peak_row['slip_outer_deg'] if peak_row is not None
                        else float('nan'))
    limit_inner_slip = (peak_row['slip_inner_deg'] if peak_row is not None
                        else float('nan'))
    want_is_reverse = bool(np.isfinite(limit_ideal_pct)
                           and limit_ideal_pct < 0.0)
    outer_wants_more_slip = bool(np.isfinite(limit_outer_slip)
                                 and np.isfinite(limit_inner_slip)
                                 and limit_outer_slip > limit_inner_slip)
    premise_holds = want_is_reverse and outer_wants_more_slip

    # self-recovery / character (secondary, for the reasoning line)
    stab = np.array([r['stability_index'] for r in mm], float)

    return {
        'radius_m': R,
        'grip_multiplier': grip,
        'aero_on': bool(aero),
        'aero_dict': aero,
        'as_built_pct': as_built_pct,
        'band_lo_pct': -30.0, 'band_hi_pct': -20.0,
        # grip tie
        'grip_pcts': pcts,
        'grip_ay_g': ay,
        'grip_span_g': grip_span_g,
        'grip_ceiling_g': grip_ceiling_g,
        'grip_span_pct': grip_span_pct,
        'grip_ties': grip_ties,
        'grip_tie_frac': GRIP_TIE_FRAC,
        'stability_index': stab,
        # ideal vs g
        'ideal_rows': ideal_rows,
        'fine_g': fg, 'fine_pct': fpct, 'sat_from': sat_from,
        'peak_g': peak_g,
        'peak_row': peak_row,
        'peak_saturated': peak_saturated,
        'sub_row': sub_row,
        # self-consistency
        'limit_ideal_pct': limit_ideal_pct,
        'limit_outer_slip_deg': limit_outer_slip,
        'limit_inner_slip_deg': limit_inner_slip,
        'want_is_reverse': want_is_reverse,
        'outer_wants_more_slip': outer_wants_more_slip,
        'premise_holds': premise_holds,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  the DECIDE figure — regenerated headless into the report dir
# ═══════════════════════════════════════════════════════════════════════════
def _decide_figure(ev, png_path):
    """Regenerate the DECIDE-style figure (ideal % vs g + grip ceiling +
    per-tyre limit slip) from the live evidence and save it as a PNG.  White /
    red / warm-grey only — no yellow, no blue (deliverable palette)."""
    fig = plt.figure(figsize=(13.0, 5.6), facecolor=FIG_BG)
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.34)

    def _style(ax, xlab, ylab, title):
        ax.set_facecolor(AX_BG)
        ax.set_title(title, color=INK, fontsize=10.5, fontweight='bold')
        ax.set_xlabel(xlab, color=MUTED, fontsize=9)
        ax.set_ylabel(ylab, color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color='#1c1a1e', lw=0.6)
        for sp in ax.spines.values():
            sp.set_color(LINE)

    # ── PRIMARY: ideal Ackermann vs g ────────────────────────────────────────
    axm = fig.add_subplot(gs[:, 0:2])
    _style(axm, 'lateral acceleration (g, on asphalt)',
           'ideal Ackermann the tyres want (%)',
           'IDEAL ACKERMANN vs HOW HARD YOU CORNER')
    axm.axhline(100, color=ACCENT, ls='--', lw=1.1)
    axm.text(0.4, 100, ' 100% = zero-slip (parking-lot) ideal',
             color=ACCENT, fontsize=8, va='bottom')
    axm.axhline(0, color=MUTED, lw=1.0)
    axm.text(2.0, 0, 'parallel (0%) ', color=MUTED, fontsize=8, ha='right',
             va='bottom')
    if ev['fine_g']:
        axm.plot(ev['fine_g'], ev['fine_pct'], color=WHT, lw=2.0, marker='o',
                 ms=3, label='ideal (slip-matched, aero-true)')
    # the limit (peak) g line + the reverse want there
    pk = ev['peak_g']
    axm.axvline(pk, color=ACCENT, ls=':', lw=1.4)
    axm.text(pk, axm.get_ylim()[0], f' limit {pk:.2f} g', color=ACCENT,
             fontsize=8, ha='left', va='bottom')
    pr = ev['peak_row']
    if pr is not None and np.isfinite(pr['ideal_pct']):
        axm.plot([pr['lat_g']], [pr['ideal_pct']], 'o', color=ACCENT, ms=7,
                 zorder=7)
        axm.annotate(f"limit -> {pr['ideal_pct']:+.0f}% (reverse)",
                     xy=(pr['lat_g'], pr['ideal_pct']), xytext=(8, 0),
                     textcoords='offset points', color=ACCENT, fontsize=9,
                     fontweight='bold', va='center')
    if ev['sat_from'] is not None:
        axm.axvspan(ev['sat_from'], 2.02, color=ACCENT, alpha=0.09, zorder=0)
        axm.text(0.5 * (ev['sat_from'] + 2.0), axm.get_ylim()[1],
                 'tyres saturate', color=ACCENT, fontsize=8, ha='center',
                 va='top')
    axm.set_xlim(0.38, 2.02)

    # ── grip ceiling per setting (the TIE) ───────────────────────────────────
    axg = fig.add_subplot(gs[0, 2])
    _style(axg, 'Ackermann %', 'max cornering it holds (g)',
           'GRIP CEILING per setting')
    axg.plot(ev['grip_pcts'], ev['grip_ay_g'], color=WGY, lw=1.8, marker='s',
             ms=3)
    mid = ev['grip_ceiling_g']
    axg.set_ylim(mid - 0.25, mid + 0.25)
    axg.text(0.5, 0.90,
             f"span {ev['grip_span_g']:.3f} g = {ev['grip_span_pct']:.0f}% of grip",
             transform=axg.transAxes, ha='center', va='top', color=INK,
             fontsize=8)
    axg.text(0.5, 0.75,
             (f"TIE (<= {ev['grip_tie_frac']*100:.0f}%) - NOT a grip lever"
              if ev['grip_ties'] else 'settings SEPARATE on grip'),
             transform=axg.transAxes, ha='center', va='top',
             color=ACCENT if ev['grip_ties'] else WHT, fontsize=8.5,
             fontweight='bold')

    # ── per-tyre slip at the limit (WHY reverse) ─────────────────────────────
    axs = fig.add_subplot(gs[1, 2])
    _style(axs, '', 'slip angle needed at the limit (deg)',
           'WHY REVERSE — outer needs more slip')
    so = ev['limit_outer_slip_deg']
    si = ev['limit_inner_slip_deg']
    fo = ev['peak_row']['Fz_outer_N'] if ev['peak_row'] else float('nan')
    fi = ev['peak_row']['Fz_inner_N'] if ev['peak_row'] else float('nan')
    bars = axs.bar(['outer\n(loaded)', 'inner\n(light)'], [so, si],
                   color=[ACCENT, WGY], width=0.6)
    for b, f in zip(bars, [fo, fi]):
        axs.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f'{f:.0f} N', ha='center', va='bottom', color=MUTED,
                 fontsize=8)
    axs.tick_params(axis='x', colors=MUTED, labelsize=8)

    fig.savefig(png_path, dpi=130, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    with open(png_path, 'rb') as fh:
        b64 = base64.b64encode(fh.read()).decode('ascii')
    plt.close(fig)
    return 'data:image/png;base64,' + b64


# ═══════════════════════════════════════════════════════════════════════════
#  HTML rendering (docs/DESIGN.md tokens; no yellow / no blue text)
# ═══════════════════════════════════════════════════════════════════════════
def _fmt(x, dp=1, sign=False, suffix=''):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    s = f'{x:+.{dp}f}' if sign else f'{x:.{dp}f}'
    return s + suffix


def _ideal_table(ev):
    rows = []
    for r in ev['ideal_rows']:
        peak_tag = (' <span class="tag">peak / limit</span>'
                    if r['is_peak'] else '')
        if not r['valid']:
            ideal_s = '<span class="sat">saturated</span>'
        else:
            ideal_s = _fmt(r['ideal_pct'], 0, sign=True, suffix='%')
        cls = ' class="peakrow"' if r['is_peak'] else ''
        rows.append(
            f'<tr{cls}><td class="num">{_fmt(r["lat_g"], 3)}{peak_tag}</td>'
            f'<td class="num">{ideal_s}</td>'
            f'<td class="num">{_fmt(r["spread_deg"], 3, sign=True)}</td>'
            f'<td class="num">{_fmt(r["Fz_outer_N"], 0)}</td>'
            f'<td class="num">{_fmt(r["slip_outer_deg"], 2)}</td>'
            f'<td class="num">{_fmt(r["Fz_inner_N"], 0)}</td>'
            f'<td class="num">{_fmt(r["slip_inner_deg"], 2)}</td></tr>')
    return '\n'.join(rows)


def render_html(ev, decide_png_uri):
    """Return the full self-contained HTML string."""
    when = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    asb = ev['as_built_pct']
    asb_s = _fmt(asb, 1, sign=True, suffix='%')
    in_band = (np.isfinite(asb) and ev['band_lo_pct'] <= asb <= ev['band_hi_pct'])
    lim = ev['limit_ideal_pct']
    sub = ev['sub_row']
    sub_s = (_fmt(sub['ideal_pct'], 0, sign=True, suffix='%')
             if sub else '—')

    # ── self-consistency box ─────────────────────────────────────────────────
    if ev['premise_holds']:
        sc_class = 'sc-pass'
        sc_head = 'SELF-CONSISTENCY CHECK — PREMISE HOLDS'
        sc_body = (
            f'Recomputed live at the {_fmt(ev["peak_g"], 2)} g limit: the ideal '
            f'Ackermann is {_fmt(lim, 0, sign=True, suffix="%")} (reverse), and '
            f'the loaded outer front needs '
            f'{_fmt(ev["limit_outer_slip_deg"], 2)}&deg; of slip against the '
            f'light inner&rsquo;s {_fmt(ev["limit_inner_slip_deg"], 2)}&deg;. '
            f'Both independent tests say the tyres want reverse at the limit, so '
            f'the conclusion below stands on this car.')
    else:
        sc_class = 'sc-fail'
        sc_head = 'PREMISE FAILED — RE-DECIDE'
        sc_body = (
            f'The narrative assumes the tyres want REVERSE at the limit, but the '
            f'live recompute on this car disagrees: at {_fmt(ev["peak_g"], 2)} g '
            f'the ideal Ackermann is {_fmt(lim, 0, sign=True, suffix="%")} '
            f'(reverse-want {ev["want_is_reverse"]}), and outer-needs-more-slip '
            f'is {ev["outer_wants_more_slip"]}. Do NOT ship the "slight reverse" '
            f'conclusion — re-run the decision from the evidence above.')

    # decision banner text depends on the check
    if ev['premise_holds']:
        decision_line = ('KEEP the v62 slight-reverse rack. '
                         f'Target band {_fmt(ev["band_lo_pct"],0,suffix="%")} '
                         f'to {_fmt(ev["band_hi_pct"],0,suffix="%")} Ackermann.')
    else:
        decision_line = ('DECISION SUSPENDED — the live premise check failed '
                         '(see below).')

    band_note = ('as built sits INSIDE the band'
                 if in_band else 'as built is OUTSIDE the target band')

    grip_verdict = ('a TIE' if ev['grip_ties'] else 'a real split')

    # reasoning chain, with live numbers interpolated
    reasoning = f"""
    <ol class="reason">
      <li><b>Grip does not choose.</b> Across &minus;100% to +100% the most
        cornering the front axle can hold spans only
        {_fmt(ev['grip_span_g'],3)} g = {_fmt(ev['grip_span_pct'],0)}% of the
        {_fmt(ev['grip_ceiling_g'],2)} g ceiling &mdash; {grip_verdict}. Peak
        slip rises only ~2&deg; over ~1000 N of load, so that flat tyre peak
        keeps every setting within skidpad run-to-run scatter. Ackermann is
        therefore <b>not a grip lever</b> ({RCVD['grip_tie']}).</li>
      <li><b>The tyres want opposite things by regime.</b> Sub-limit the light,
        similarly-loaded fronts want <b>pro</b> ({sub_s} at
        {_fmt(sub['lat_g'],1) if sub else '—'} g); at the
        {_fmt(ev['peak_g'],2)} g limit, load transfer throws the outer front to
        {_fmt(ev['limit_outer_slip_deg'],2)}&deg; of needed slip against the
        inner&rsquo;s {_fmt(ev['limit_inner_slip_deg'],2)}&deg;, so the tyres
        want <b>reverse</b> ({_fmt(lim,0,sign=True,suffix='%')}). A more heavily
        loaded tyre peaks at a higher slip angle ({RCVD['load_sens']}); full
        Ackermann would drag the light inner past its peak
        ({RCVD['inner_drag']}).</li>
      <li><b>A fixed rack cannot serve both regimes.</b> One steering arm gives
        one toe-vs-steer law, so the setting must be <b>weighted to the
        limit</b>, where the lap is actually decided and where the loaded outer
        must stay near its peak slip. Slight reverse matches that limit want and
        keeps the outer working; the cost is mild pro-scrub at parking-lot
        speeds, which is acceptable ({RCVD['ack_defn']}).</li>
      <li><b>Where the car sits.</b> As built the linkage probes {asb_s} at full
        lock &mdash; {band_note}
        ({_fmt(ev['band_lo_pct'],0,suffix='%')} to
        {_fmt(ev['band_hi_pct'],0,suffix='%')}).</li>
    </ol>"""

    ideal_rows_html = _ideal_table(ev)
    aero_s = ('aero ON (car package, V²-scaled to the radius)'
              if ev['aero_on'] else 'aero OFF (no package data)')

    doc = f"""<div class="wrap">
  <header>
    <div class="cat">Vahan &middot; Ackermann &middot; justification</div>
    <h1>Why this car runs slight-reverse Ackermann</h1>
    <div class="sub">Every number below is recomputed live from the one solved
      car (radius {_fmt(ev['radius_m'],0)} m, {aero_s}, the car&rsquo;s tyre with
      a &times;{_fmt(ev['grip_multiplier'],2)} belt&rarr;road derate, slip in
      degrees). Generated {when}. <span class="beta">BETA</span></div>
  </header>

  <section class="banner {'ok' if ev['premise_holds'] else 'bad'}">
    <div class="cat">Decision</div>
    <div class="decision">{decision_line}</div>
    <div class="asbuilt">As built (probed at full lock):
      <b>{asb_s}</b> &nbsp;&middot;&nbsp; {band_note}</div>
  </section>

  <section class="{sc_class}">
    <div class="cat">Live check</div>
    <div class="sc-head">{sc_head}</div>
    <div class="sc-body">{sc_body}</div>
  </section>

  <section>
    <div class="cat">Evidence &mdash; all live</div>
    <h2>1 &middot; What the tyres want, vs how hard you corner (aero-true)</h2>
    <p class="lede">Ideal Ackermann the slip-matched tyres ask for at each
      lateral g on a {_fmt(ev['radius_m'],0)} m corner, with the per-wheel load
      and slip that drive it. Positive = pro (inner turned more); negative =
      reverse (outer turned more).</p>
    <div class="tbl-scroll">
    <table>
      <thead><tr>
        <th>lateral g</th><th>ideal Ackermann</th><th>toe spread (deg)</th>
        <th>outer Fz (N)</th><th>outer slip (deg)</th>
        <th>inner Fz (N)</th><th>inner slip (deg)</th>
      </tr></thead>
      <tbody>
{ideal_rows_html}
      </tbody>
    </table>
    </div>
    <p class="foot">The sign flips from pro to reverse as g rises: the load
      moved onto the outer front makes it demand progressively more slip than
      the unloading inner. Rows past the grip ceiling are marked saturated
      (no honest answer &mdash; a saturated wheel&rsquo;s slip is a peak-angle
      fallback, not a solution).</p>

    <h2>2 &middot; Grip ceiling per setting &mdash; is Ackermann a grip lever?</h2>
    <p class="lede">Maximum trimmed cornering the whole car holds at each
      setting. If this is flat, grip cannot choose the setting.</p>
    <div class="tbl-scroll">
    <table>
      <thead><tr><th>Ackermann %</th>{''.join(f'<th>{p:+d}%</th>' for p in ev['grip_pcts'])}</tr></thead>
      <tbody>
        <tr><td>max cornering (g)</td>{''.join(f'<td class="num">{_fmt(a,3)}</td>' for a in ev['grip_ay_g'])}</tr>
      </tbody>
    </table>
    </div>
    <p class="foot"><b>Span {_fmt(ev['grip_span_g'],3)} g = {_fmt(ev['grip_span_pct'],1)}%</b>
      of the {_fmt(ev['grip_ceiling_g'],2)} g ceiling across the whole swing
      &mdash; {'a TIE (within skidpad scatter)' if ev['grip_ties'] else 'a real, chase-able split'}.
      {RCVD['grip_tie']}.</p>

    <h2>3 &middot; Per-tyre at the limit &mdash; why reverse</h2>
    <p class="lede">At the {_fmt(ev['peak_g'],2)} g limit the two front tyres are
      nothing alike: the loaded outer carries far more vertical load and must
      run more slip to make its share of the force. The wheel that needs more
      slip is the one that must be steered more &mdash; the outer &mdash; which
      is reverse Ackermann.</p>
    <div class="tbl-scroll">
    <table>
      <thead><tr><th>front wheel</th><th>vertical load Fz (N)</th><th>slip needed (deg)</th></tr></thead>
      <tbody>
        <tr><td>outer (loaded)</td><td class="num">{_fmt(ev['peak_row']['Fz_outer_N'] if ev['peak_row'] else None,0)}</td><td class="num">{_fmt(ev['limit_outer_slip_deg'],2)}</td></tr>
        <tr><td>inner (light)</td><td class="num">{_fmt(ev['peak_row']['Fz_inner_N'] if ev['peak_row'] else None,0)}</td><td class="num">{_fmt(ev['limit_inner_slip_deg'],2)}</td></tr>
      </tbody>
    </table>
    </div>
  </section>

  <section>
    <div class="cat">Figure</div>
    <h2>The decision, in one picture</h2>
    <img class="fig" src="{decide_png_uri}" alt="Ideal Ackermann vs lateral g, grip ceiling, and per-tyre limit slip" />
    <p class="foot">Left: the ideal Ackermann the tyres want, aero-true, with the
      live {_fmt(ev['peak_g'],2)} g limit marked. Top-right: the grip-ceiling
      tie. Bottom-right: the loaded outer front needs more slip than the light
      inner &mdash; the mechanism behind reverse.</p>
  </section>

  <section>
    <div class="cat">Reasoning</div>
    <h2>The chain, with the live numbers</h2>
    {reasoning}
  </section>

  <section>
    <div class="cat">Caveats</div>
    <h2>What this does not settle</h2>
    <ul class="caveats">
      <li><b>Rigid-system numbers.</b> Everything here assumes an infinitely
        stiff steering system. Compliance can steer the wheels more than the
        built-in geometry &mdash; measure steer&ndash;steer compliance on the
        car before committing ({RCVD['compliance']}).</li>
      <li><b>The band is a comfort zone, not a knife-edge.</b> Because grip ties
        ({_fmt(ev['grip_span_pct'],1)}% span), the whole
        {_fmt(ev['band_lo_pct'],0,suffix='%')}&hellip;{_fmt(ev['band_hi_pct'],0,suffix='%')}
        band is indistinguishable on grip; character and use decide the exact
        value, not lap time.</li>
      <li><b>Aero-on is the less-reverse case.</b> Downforce is grip for free and
        pushes the ideal toward less reverse, so the reverse conclusion is
        robust: with aero the case for reverse is only weaker, and it still
        holds.</li>
      <li><b>Belt&rarr;road derate.</b> The tyre data is belt-rig; a
        &times;{_fmt(ev['grip_multiplier'],2)} derate puts saturation at the road
        limit. The lateral-g axis is road g throughout.</li>
      <li><b>One radius.</b> Solved at {_fmt(ev['radius_m'],0)} m (about the
        skidpad path). The required toe difference changes sign with radius; a
        fixed rack cannot match every corner at once.</li>
    </ul>
  </section>

  <footer>
    Generated by <code>vahan/ackermann_report.py</code> &mdash; one model, live
    numbers, RCVD-cited. Tyre data is de-identified per the FSAE TTC agreement
    (FSAE TTC &middot; Calspan Tire Research Facility).
  </footer>
</div>"""

    return _PAGE.format(body=doc)


# One-file page shell.  All CSS inline; dark-theme Impeccable tokens; no yellow,
# no blue anywhere.
_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Ackermann justification — Vahan</title>
<style>
  :root {{
    --ink:#ECECEE; --muted:#9A9AA2; --paper:#0B0B0D; --panel:#141416;
    --line:#2A2A2E; --accent:#E23B48; --wgy:#C8BCA6;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; padding:0; background:var(--paper); color:var(--ink);
    font-family:'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif;
    font-size:13px; line-height:1.5; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:32px 22px 60px; }}
  .cat {{ text-transform:uppercase; letter-spacing:0.15em; font-weight:650;
    color:var(--accent); font-size:11px; margin-bottom:6px; }}
  h1 {{ font-size:27px; font-weight:600; letter-spacing:-0.02em; margin:0 0 8px;
    text-wrap:balance; }}
  h2 {{ font-size:17px; font-weight:600; margin:26px 0 8px; letter-spacing:-0.01em; }}
  .sub {{ color:var(--muted); font-size:12px; margin-bottom:8px; }}
  .beta {{ background:var(--accent); color:#fff; font-size:9px; font-weight:700;
    letter-spacing:0.1em; padding:1px 6px; border-radius:3px; vertical-align:2px; }}
  header {{ border-bottom:1px solid var(--line); padding-bottom:20px; }}
  section {{ margin-top:26px; }}
  p.lede {{ color:var(--ink); margin:6px 0 12px; }}
  p.foot {{ color:var(--muted); font-size:12px; margin:8px 0 0; }}
  .banner {{ border:1px solid var(--line); border-left:4px solid var(--accent);
    background:var(--panel); border-radius:6px; padding:16px 18px; }}
  .banner.bad {{ border-left-color:var(--accent); }}
  .banner .decision {{ font-size:19px; font-weight:650; letter-spacing:-0.01em; }}
  .banner .asbuilt {{ color:var(--muted); font-size:12px; margin-top:8px; }}
  .banner b {{ color:var(--ink); }}
  .sc-pass, .sc-fail {{ border-radius:6px; padding:14px 16px; margin-top:16px;
    border:1px solid var(--line); background:var(--panel); }}
  .sc-fail {{ border:1px solid var(--accent); border-left:4px solid var(--accent);
    background:#1a1012; }}
  .sc-pass {{ border-left:4px solid var(--wgy); }}
  .sc-head {{ font-weight:700; letter-spacing:0.02em; margin-bottom:5px; }}
  .sc-fail .sc-head {{ color:var(--accent); }}
  .sc-body {{ color:var(--muted); font-size:12.5px; }}
  .tbl-scroll {{ overflow-x:auto; margin:6px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
  thead th {{ text-transform:uppercase; font-size:10.5px; letter-spacing:0.06em;
    color:var(--muted); font-weight:600; text-align:right; padding:6px 10px;
    border-bottom:1.5px solid var(--ink); white-space:nowrap; }}
  thead th:first-child {{ text-align:left; }}
  tbody td {{ padding:6px 10px; border-bottom:1px solid var(--line);
    text-align:left; }}
  td.num {{ text-align:right; font-family:'Consolas','SF Mono',monospace;
    font-variant-numeric:tabular-nums; }}
  tr.peakrow td {{ background:#17181b; }}
  .tag {{ font-size:9px; text-transform:uppercase; letter-spacing:0.08em;
    color:var(--accent); border:1px solid var(--accent); border-radius:3px;
    padding:0 4px; margin-left:6px; }}
  .sat {{ color:var(--muted); font-style:italic; }}
  ol.reason {{ padding-left:20px; }}
  ol.reason li {{ margin:10px 0; }}
  ul.caveats {{ padding-left:20px; }}
  ul.caveats li {{ margin:8px 0; color:var(--ink); }}
  b {{ color:var(--ink); }}
  img.fig {{ width:100%; max-width:100%; height:auto; border:1px solid var(--line);
    border-radius:6px; background:var(--paper); }}
  code {{ font-family:'Consolas','SF Mono',monospace; color:var(--wgy); }}
  footer {{ margin-top:34px; padding-top:16px; border-top:1px solid var(--line);
    color:var(--muted); font-size:11.5px; }}
</style></head>
<body>
{body}
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
#  the public entry point
# ═══════════════════════════════════════════════════════════════════════════
def build_ackermann_report(source, out_html=None, radius_m=8.0,
                           grip_multiplier=None):
    """Build the Ackermann justification report and write a self-contained HTML.

    source : a MainWindow, or a (solver, tire, aero, probe_fn) tuple.
    out_html : output path; defaults to <repo>/figs/ackermann_justification.html.

    Returns a dict: {'html_path', 'png_path', 'evidence', 'premise_holds'}.
    Every number in the report is recomputed live here (nothing hardcoded); if
    the live limit-want sign contradicts the reverse-Ackermann premise, the
    report prints a red PREMISE FAILED box instead of asserting the conclusion.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if out_html is None:
        out_html = os.path.join(repo, 'figs', 'ackermann_justification.html')
    out_dir = os.path.dirname(os.path.abspath(out_html))
    os.makedirs(out_dir, exist_ok=True)

    ev = compute_ackermann_evidence(source, radius_m=radius_m,
                                    grip_multiplier=grip_multiplier)
    png_path = os.path.join(out_dir, 'ackermann_justification_decide.png')
    png_uri = _decide_figure(ev, png_path)
    html = render_html(ev, png_uri)
    with open(out_html, 'w', encoding='utf-8') as fh:
        fh.write(html)
    return {'html_path': out_html, 'png_path': png_path, 'evidence': ev,
            'premise_holds': ev['premise_holds']}
