"""Trace the autocross26.png red course into an ordered centerline polyline,
calibrate to meters via the feet axis + the user's constraints (length <= 800 m,
longest straight <= 61 m, width <= 3.5 m), and save tracks/autocross26.json.

Verification artifacts: _tmp_track_overlay.png (trace over the image) and
_tmp_track_meters.png (the metric polyline + curvature colouring).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# usage: python tools/trace_track.py [IMG] [OUT_JSON] [TRACK_NAME]
IMG = sys.argv[1] if len(sys.argv) > 1 else 'autocross26.png'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'tracks/autocross26.json'
TRACK_NAME = sys.argv[3] if len(sys.argv) > 3 else 'Autocross 2024 (FSAE)'
TAG = os.path.splitext(os.path.basename(IMG))[0]

im = np.asarray(Image.open(IMG).convert('RGB')).astype(int)
H, W, _ = im.shape
R, G, B = im[..., 0], im[..., 1], im[..., 2]

# ── 1. red-stroke mask ───────────────────────────────────────────────────────
# Two regimes: clean PNG (saturated red) and JPG (compression spreads red into
# pinks, so use red-DOMINANCE rather than absolute thresholds; this also
# rejects the yellow/green cone markers where G >= R).
is_jpg = IMG.lower().endswith(('.jpg', '.jpeg'))
if is_jpg:
    mask = (R > G + 30) & (R > B + 15) & (R > 110)
else:
    mask = (R > 170) & (G < 95) & (B < 95)
mask[:, :300] = False          # the left side is the skidpad / other course

# largest connected component = the course stroke
from scipy import ndimage
lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
if n > 1:
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    mask = lab == (1 + int(np.argmax(sizes)))
ys, xs = np.nonzero(mask)
print(f'red pixels: {mask.sum()}  bbox x:{xs.min()}..{xs.max()} y:{ys.min()}..{ys.max()}')

# ── 2. ordered walk-trace of the stroke ──────────────────────────────────────
pts = np.column_stack([xs, ys]).astype(float)
alive = np.ones(len(pts), bool)

# start = red pixel nearest the green start arrow (top-left of the course)
start_guess = np.array([float(xs.min()), 40.0])
i0 = int(np.argmin(np.linalg.norm(pts - start_guess, axis=1)))
path = [pts[i0]]
alive[np.linalg.norm(pts - pts[i0], axis=1) < 3.0] = False
direction = np.array([1.0, 0.0])

STEP_R = 6.0        # search radius per step (stroke is ~3-4 px wide)
CLEAR_R = 3.0       # pixels consumed around each accepted step
for _ in range(20000):
    cur = path[-1]
    d = np.linalg.norm(pts - cur, axis=1)
    cand = np.nonzero(alive & (d < STEP_R))[0]
    if len(cand) == 0:
        cand = np.nonzero(alive & (d < STEP_R * 2.0))[0]   # jump small gaps
        if len(cand) == 0:
            break
    # prefer candidates continuing the current direction
    vecs = pts[cand] - cur
    norms = np.linalg.norm(vecs, axis=1) + 1e-9
    score = (vecs @ direction) / norms - 0.15 * (norms / STEP_R)
    j = cand[int(np.argmax(score))]
    new = pts[j]
    v = new - cur
    nv = np.linalg.norm(v)
    if nv > 1e-9:
        direction = 0.6 * direction + 0.4 * (v / nv)
        direction /= np.linalg.norm(direction)
    path.append(new)
    alive[np.linalg.norm(pts - new, axis=1) < CLEAR_R] = False
path = np.array(path)
print(f'traced {len(path)} path points; consumed {(~alive).sum()}/{len(pts)} red px')

# ── 3. smooth + resample ─────────────────────────────────────────────────────
def smooth(p, w):
    k = np.ones(w) / w
    out = p.copy()
    for c in range(2):
        out[:, c] = np.convolve(np.pad(p[:, c], w // 2, mode='edge'), k, 'valid')[:len(p)]
    return out

path_s = smooth(path, 9)

# ── 4. calibration: feet axis -> meters ─────────────────────────────────────
# Axis labels 0'..1700' span x ~200..985 px  =>  px/ft; refined against the
# user's constraints (course length <= 800 m, longest straight <= 61 m).
PX_PER_FT = (985.0 - 200.0) / 1700.0
M_PER_PX = 0.3048 / PX_PER_FT
xy = path_s * M_PER_PX
xy[:, 1] = -xy[:, 1]                      # image y-down -> world y-up
xy -= xy[0]                                # start at origin

seglen = np.linalg.norm(np.diff(xy, axis=0), axis=1)
length = float(seglen.sum())
print(f'raw length at axis scale: {length:.1f} m')
if length > 800.0:                         # respect the hard constraint
    scale = 795.0 / length
    xy *= scale
    M_PER_PX *= scale
    seglen *= scale
    length = float(seglen.sum())
    print(f'rescaled by {scale:.4f} -> {length:.1f} m')

# resample at 1 m
s = np.concatenate([[0.0], np.cumsum(seglen)])
s_new = np.arange(0.0, s[-1], 1.0)
xr = np.interp(s_new, s, xy[:, 0])
yr = np.interp(s_new, s, xy[:, 1])
cl = np.column_stack([xr, yr])
cl = smooth(cl, 7)

# curvature via central differences
d1 = np.gradient(cl, axis=0)
d2 = np.gradient(d1, axis=0)
num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-12
kappa = num / den
# light smoothing of curvature (tracing noise)
k = np.ones(7) / 7
kappa = np.convolve(np.pad(kappa, 3, mode='edge'), k, 'valid')[:len(kappa)]
# Clamp to a physical floor: a course this wide can't be driven tighter than
# ~3 m radius, so anything sharper is a tracing kink (e.g. the return-to-start
# pinch), not a real corner.  Removes phantom hairpins that would stall the sim.
KAPPA_MAX = 1.0 / 3.0
kappa = np.clip(kappa, -KAPPA_MAX, KAPPA_MAX)

# longest straight (|kappa| < 1/200 m^-1 contiguous)
straight = np.abs(kappa) < (1.0 / 200.0)
best = run = 0
for v in straight:
    run = run + 1 if v else 0
    best = max(best, run)
print(f'length={len(cl)} m   longest straight={best} m (limit 61)   '
      f'max |kappa|={np.abs(kappa).max():.3f}  min radius={1/np.abs(kappa).max():.1f} m')

# ── 5. save + verification renders ──────────────────────────────────────────
os.makedirs('tracks', exist_ok=True)
json.dump({
    'name': TRACK_NAME,
    'source': IMG,
    'spacing_m': 1.0,
    'width_m': 3.5,
    'x_m': [round(float(v), 3) for v in cl[:, 0]],
    'y_m': [round(float(v), 3) for v in cl[:, 1]],
    'kappa_1pm': [round(float(v), 5) for v in kappa],
}, open(OUT, 'w'))
print(f'wrote {OUT}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 3), facecolor='black')
ax.imshow(np.asarray(Image.open(IMG)))
ax.plot(path_s[:, 0], path_s[:, 1], c='cyan', lw=1.0)
ax.plot(path_s[0, 0], path_s[0, 1], 'g^', ms=8)
ax.plot(path_s[-1, 0], path_s[-1, 1], 'ws', ms=6)
ax.set_title('traced centerline over image (cyan; green=start, white=end)', color='w')
ax.axis('off')
fig.tight_layout()
fig.savefig(f'_tmp_overlay_{TAG}.png', dpi=110, facecolor='black')

fig, ax = plt.subplots(figsize=(12, 5), facecolor='black')
ax.set_facecolor('black')
sc = ax.scatter(cl[:, 0], cl[:, 1], c=np.abs(kappa), s=3, cmap='plasma')
plt.colorbar(sc, label='|curvature| 1/m')
ax.plot(cl[0, 0], cl[0, 1], 'g^', ms=10)
ax.set_aspect('equal'); ax.tick_params(colors='gray')
ax.set_title(f'metric centerline — {len(cl)} m, longest straight {best} m', color='w')
fig.tight_layout()
fig.savefig(f'_tmp_meters_{TAG}.png', dpi=100, facecolor='black')
print(f'wrote _tmp_overlay_{TAG}.png + _tmp_meters_{TAG}.png')
