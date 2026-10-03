"""Convert AutoX_2026.OLTra (OptimumLap track) to tracks/autox_2026.json.

OLTra layout (reverse-engineered): header strings, then three float64 arrays
of N=685 entries at fixed offsets — section LENGTHS (m), corner RADII (m,
UNSIGNED; 0 or huge = straight), and a mostly-zero auxiliary array.

Physics (lap time / speed caps) uses the OLTra |curvature| EXACTLY.
Turn DIRECTION is not stored, so for drawing the map we transfer signs from
the previously traced image centerline (tracks/autocross26.json) by
arc-length matching — display-only, clearly noted in the JSON.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = 'AutoX_2026.OLTra'
N, O_LEN = 685, 306
O_RAD = O_LEN + N * 8 + 4

raw = open(SRC, 'rb').read()
L = np.frombuffer(raw, dtype='<f8', count=N, offset=O_LEN).copy()
R = np.frombuffer(raw, dtype='<f8', count=N, offset=O_RAD).copy()

keep = L > 1e-6                      # trailing zero-length padding
L, R = L[keep], R[keep]
s_end = np.cumsum(L)
s_mid = s_end - L / 2.0
total = float(s_end[-1])
print(f'sections: {len(L)}  total length: {total:.1f} m')

# curvature magnitude: straight when R == 0 or absurdly large
kappa_mag = np.where((R > 0.05) & (R < 1500.0), 1.0 / np.maximum(R, 0.05), 0.0)
rmin = R[(R > 0.05) & (R < 1500.0)].min()
print(f'min radius in data: {rmin:.2f} m   stations with R<3m: '
      f'{int(((R > 0.05) & (R < 3.0)).sum())}')

# ── resample to 1 m stations ────────────────────────────────────────────────
s_new = np.arange(0.0, total, 1.0)
k_new = np.interp(s_new, s_mid, kappa_mag)

# ── map GEOMETRY from the image trace (true course shape), rescaled to the
#    OLTra length; CURVATURE stays exact from the OLTra sections.  Integrating
#    heading from unsigned OLTra radii + transferred signs collapses into a
#    tangle (direction isn't stored in the file), so the drawn map uses the
#    image shape while every physics number comes from the OLTra data.
trace_path = 'tracks/autocross26.json'
if not os.path.isfile(trace_path):
    raise SystemExit('need tracks/autocross26.json (image trace) for the '
                     'map shape — run tools/trace_track.py first')
tr = json.load(open(trace_path))
xt = np.asarray(tr['x_m'], float)
yt = np.asarray(tr['y_m'], float)
seg = np.linalg.norm(np.diff(np.column_stack([xt, yt]), axis=0), axis=1)
s_tr = np.concatenate([[0.0], np.cumsum(seg)])
geo_scale = total / float(s_tr[-1])
xt, yt, s_tr = xt * geo_scale, yt * geo_scale, s_tr * geo_scale
x = np.interp(s_new, s_tr, xt)
y = np.interp(s_new, s_tr, yt)
# signed curvature column: OLTra magnitude, sign from the (rescaled) trace
k_tr = np.asarray(tr['kappa_1pm'], float) / geo_scale
k_sign_src = np.interp(s_new, s_tr[:len(k_tr)], k_tr)
w = 5
k_sm = np.convolve(np.pad(k_sign_src, w // 2, mode='edge'),
                   np.ones(w) / w, 'valid')[:len(k_sign_src)]
kappa = k_new * np.where(k_sm >= 0, 1.0, -1.0)
print(f'map shape from image trace (rescaled x{geo_scale:.3f}); '
      f'|kappa| exact from OLTra')

os.makedirs('tracks', exist_ok=True)
json.dump({
    'name': 'AutoX 2026 (OptimumLap)',
    'source': SRC,
    'note': ('curvature magnitudes are EXACT from the OLTra section data '
             '(lap-time physics); turn directions for the drawn map are '
             'transferred from the image trace (display only).'),
    'spacing_m': 1.0,
    'width_m': 3.5,
    'x_m': [round(float(v), 3) for v in x],
    'y_m': [round(float(v), 3) for v in y],
    'kappa_1pm': [round(float(v), 5) for v in kappa],
}, open('tracks/autox_2026.json', 'w'))
print(f'wrote tracks/autox_2026.json  ({len(s_new)} stations)')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(13, 8), facecolor='black')
axs[0].set_facecolor('black')
sc = axs[0].scatter(x, y, c=np.abs(kappa), s=4, cmap='plasma')
plt.colorbar(sc, ax=axs[0], label='|kappa| 1/m')
axs[0].plot(x[0], y[0], 'g^', ms=10)
axs[0].set_aspect('equal'); axs[0].tick_params(colors='gray')
axs[0].set_title(f'AutoX 2026 from OLTra — {total:.0f} m', color='w')
axs[1].set_facecolor('black')
axs[1].plot(s_new, kappa, c='#FFD600', lw=0.9)
axs[1].set_title('signed curvature vs s', color='w')
axs[1].tick_params(colors='gray')
fig.tight_layout(); fig.savefig('_tmp_autox26_oltra.png', dpi=95,
                                facecolor='black')
print('wrote _tmp_autox26_oltra.png')
