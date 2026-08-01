"""
vahan/tire_model.py — Tire force model from test data

Loads tire test data from .mat (TTC), .csv, or .xlsx files
and processes raw data into smooth lookup surfaces for lateral
force Fy(slip_angle, Fz, camber) and aligning moment Mz(SA, Fz, camber).

Uses binned median + cubic spline interpolation on a regular grid.
No Pacejka fitting required — works directly from measured data.

Supported formats:
    .mat  — TTC (Tire Test Consortium) MATLAB format
    .csv  — Comma-separated with required column headers
    .xlsx — Excel spreadsheet with required column headers

Required columns for CSV/XLSX (case-insensitive, flexible naming):
    SA or slip_angle     — Slip angle (deg)
    FZ or normal_load    — Normal load (N, positive = compression)
    FY or lateral_force  — Lateral force (N)
    MZ or aligning_moment — Aligning moment (Nm)  [optional, zeros if missing]
    IA or camber or inclination — Camber / inclination angle (deg)
    V  or velocity       — Velocity (kph)          [optional, no filter if missing]
    P  or pressure       — Pressure (kPa)           [optional]
    MX or overturning    — Overturning moment (Nm)  [optional]
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d


# ─────────────────────────────────────────────────────────────────────────────
#  Raw data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTCData:
    """Raw TTC test data from a single .mat file."""
    tire_id: str
    test_id: str
    slip_angle_deg: np.ndarray    # SA
    normal_load_N: np.ndarray     # |FZ| (positive, TTC convention negated)
    lateral_force_N: np.ndarray   # FY
    aligning_moment_Nm: np.ndarray  # MZ
    overturning_moment_Nm: np.ndarray  # MX
    camber_deg: np.ndarray        # IA (inclination angle)
    pressure_kPa: np.ndarray      # P
    velocity_kph: np.ndarray      # V


def load_ttc_mat(path: str) -> TTCData:
    """
    Load a TTC .mat file.

    TTC convention: FZ is negative (compression = downward).
    We store |FZ| so all downstream code uses positive normal loads.
    """
    from scipy.io import loadmat
    d = loadmat(path)
    return TTCData(
        tire_id=str(d['tireid'][0]),
        test_id=str(d['testid'][0]),
        slip_angle_deg=d['SA'].ravel(),
        normal_load_N=np.abs(d['FZ'].ravel()),
        lateral_force_N=d['FY'].ravel(),
        aligning_moment_Nm=d['MZ'].ravel(),
        overturning_moment_Nm=d['MX'].ravel(),
        camber_deg=d['IA'].ravel(),
        pressure_kPa=d['P'].ravel(),
        velocity_kph=d['V'].ravel(),
    )


# Column name aliases (maps various common names → canonical name)
_COL_ALIASES = {
    'sa': 'SA', 'slip_angle': 'SA', 'slip angle': 'SA', 'slipdeg': 'SA',
    'fz': 'FZ', 'normal_load': 'FZ', 'normal load': 'FZ', 'vertical_load': 'FZ',
    'fy': 'FY', 'lateral_force': 'FY', 'lateral force': 'FY', 'lat_force': 'FY',
    'mz': 'MZ', 'aligning_moment': 'MZ', 'aligning moment': 'MZ',
    'mx': 'MX', 'overturning': 'MX', 'overturning_moment': 'MX',
    'ia': 'IA', 'camber': 'IA', 'inclination': 'IA', 'inclination_angle': 'IA',
    'p': 'P', 'pressure': 'P',
    'v': 'V', 'velocity': 'V', 'speed': 'V',
}


def _resolve_columns(df_columns: list) -> dict:
    """Map DataFrame column names to canonical names (SA, FZ, FY, etc.)."""
    mapping = {}
    for col in df_columns:
        key = col.strip().lower().replace(' ', '_')
        if key in _COL_ALIASES:
            mapping[col] = _COL_ALIASES[key]
        elif col.upper() in ('SA', 'FZ', 'FY', 'MZ', 'MX', 'IA', 'P', 'V'):
            mapping[col] = col.upper()
    return mapping


def load_csv(path: str) -> TTCData:
    """
    Load tire data from a CSV file.

    Required columns: SA (or slip_angle), FZ (or normal_load), FY (or lateral_force), IA (or camber)
    Optional columns: MZ, MX, V, P

    FZ should be positive = compression. If all FZ values are negative,
    they are automatically negated (assumes TTC convention).
    """
    import pandas as pd
    df = pd.read_csv(path)
    return _dataframe_to_ttc(df, path)


def load_xlsx(path: str) -> TTCData:
    """Load tire data from an Excel file. Same column requirements as CSV."""
    import pandas as pd
    df = pd.read_excel(path)
    return _dataframe_to_ttc(df, path)


def _dataframe_to_ttc(df, path: str) -> TTCData:
    """Convert a pandas DataFrame to TTCData."""
    col_map = _resolve_columns(list(df.columns))
    # Reverse: canonical → original column name
    rev = {}
    for orig, canon in col_map.items():
        if canon not in rev:
            rev[canon] = orig

    # Required columns
    for req in ('SA', 'FZ', 'FY', 'IA'):
        if req not in rev:
            found = list(col_map.values())
            raise ValueError(
                f"Missing required column '{req}'. Found: {found}. "
                f"Need: SA (slip angle), FZ (normal load), FY (lateral force), IA (camber).")

    sa = df[rev['SA']].values.astype(float)
    fz = df[rev['FZ']].values.astype(float)
    fy = df[rev['FY']].values.astype(float)
    ia = df[rev['IA']].values.astype(float)

    # Auto-detect TTC FZ sign convention (all negative → negate)
    if np.median(fz) < 0:
        fz = np.abs(fz)

    mz = df[rev['MZ']].values.astype(float) if 'MZ' in rev else np.zeros_like(sa)
    mx = df[rev['MX']].values.astype(float) if 'MX' in rev else np.zeros_like(sa)
    v  = df[rev['V']].values.astype(float) if 'V' in rev else np.full_like(sa, 50.0)
    p  = df[rev['P']].values.astype(float) if 'P' in rev else np.full_like(sa, 80.0)

    import os
    name = os.path.splitext(os.path.basename(path))[0]
    return TTCData(
        tire_id=name,
        test_id=name,
        slip_angle_deg=sa,
        normal_load_N=fz,
        lateral_force_N=fy,
        aligning_moment_Nm=mz,
        overturning_moment_Nm=mx,
        camber_deg=ia,
        pressure_kPa=p,
        velocity_kph=v,
    )


def load_tire_data(path: str) -> TTCData:
    """
    Auto-detect file format and load tire data.
    Supports .mat (TTC), .csv, .xlsx/.xls
    """
    ext = path.lower().rsplit('.', 1)[-1] if '.' in path else ''
    if ext == 'mat':
        return load_ttc_mat(path)
    elif ext == 'csv':
        return load_csv(path)
    elif ext in ('xlsx', 'xls'):
        return load_xlsx(path)
    else:
        raise ValueError(f"Unsupported file format '.{ext}'. Use .mat, .csv, or .xlsx")


# ─────────────────────────────────────────────────────────────────────────────
#  Processed tire model (grid interpolation)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_peak_sa(sa_axis, fy_abs):
    """Slip angle at peak Fy, from a Magic-Formula fit rather than argmax.

    The Fy(alpha) curve is flat near its maximum, so argmax returns whichever
    sample noise happened to lift highest — scatter, not signal.  Fitting
    D*sin(C*atan(B*alpha)) uses the whole curve; its peak is analytic at
    alpha = (pi / 2C) / B.  Falls back to argmax only if the fit fails.
    """
    sa = np.asarray(sa_axis, float)
    fy = np.asarray(fy_abs, float)
    m = np.isfinite(sa) & np.isfinite(fy) & (sa >= 0)
    if m.sum() < 4:
        return float(sa[int(np.argmax(fy))]) if len(sa) else 0.0
    x, y = sa[m], fy[m]
    try:
        from scipy.optimize import curve_fit
        f = lambda a, B, C, D: D * np.sin(C * np.arctan(B * a))
        p, _ = curve_fit(f, x, y, p0=[0.25, 1.5, max(y.max(), 1.0)], maxfev=20000)
        B, C, D = p
        if B > 1e-6 and C > 1e-6:
            a_pk = (np.pi / (2.0 * C)) / B
            # WARNING, measured 2026-07-27: on FSAE slick data this UNDERSTATES
            # the peak by ~4 deg (4.1-5.9 vs a real 8.5-12.5).  D*sin(C*atan(Ba))
            # is forced to fall away after its peak, but the real curve SATURATES
            # and stays flat to 16 deg, so the fit drags the peak early.  A
            # 4-parameter fit with the E falloff term is unstable on this data
            # (it runs to the sweep bound at light load).  Do not treat this as
            # a trustworthy peak location — see slip_angle_at_peak_Fy.
            if np.isfinite(a_pk) and x.min() <= a_pk <= x.max() * 1.5:
                return float(a_pk)
    except Exception:
        pass
    return float(x[int(np.argmax(y))])


def _fit_magic_formula(sa_deg, fz_N, fy_N):
    """Fit Pacejka MF5.2 pure-slip lateral to ALL raw samples at once.

    WHY GLOBAL AND NOT PER-LOAD-BIN.  Fitting each load bin separately (what
    `_fit_peak_sa` does) lets a noisy bin run away on its own: measured
    2026-07-27, argmax peak slip on the binned grid read 8.5 deg at 616 N,
    12.5 deg at 752 N and 9.0 deg at 888 N — pure scatter, r = -0.58 against
    load, and three different estimators disagreed by 3.3 deg about which way
    the trend even ran.  Ackermann is designed around exactly that trend (does
    the loaded outer tyre want MORE slip than the light inner one?), so the
    binned grid cannot answer the question it is being asked.

    In the Magic Formula the load dependence lives in the COEFFICIENTS, so a
    single fit across every load forces one smooth, monotone family of curves
    and a light-load bin physically cannot diverge from its neighbours:

        dfz = (Fz - Fz0)/Fz0
        C   = pCy1
        D   = (pDy1 + pDy2*dfz) * Fz          peak force (mu falls with load)
        K   = pKy1*Fz0*sin(2*atan(Fz/(pKy2*Fz0)))   cornering stiffness at 0
        B   = K/(C*D)
        E   = pEy1 + pEy2*dfz                 capped at 1 for stability
        Fy  = D*sin(C*atan(B*a - E*(B*a - atan(B*a))))

    The E term is what the earlier 3-parameter D*sin(C*atan(Ba)) fit lacked;
    without it the curve is forced to fall away after its peak while this tyre
    actually SATURATES, which dragged the fitted peak ~4 deg early.  It is
    stable here only because the load dependence is shared.

    Returns (params dict, r2, peak_slip_fn) or (None, nan, None) on failure.
    """
    sa = np.asarray(sa_deg, float)
    fz = np.asarray(fz_N, float)
    fy = np.asarray(fy_N, float)
    m = np.isfinite(sa) & np.isfinite(fz) & np.isfinite(fy) & (fz > 20.0)
    sa, fz, fy = sa[m], fz[m], fy[m]
    if len(sa) < 200:
        return None, float('nan'), None
    a = np.radians(sa)
    y = -fy                      # SAE -> positive force for positive slip
    fz0 = float(np.median(fz))

    def mf(X, pCy1, pDy1, pDy2, pKy1, pKy2, pEy1, pEy2):
        al, Fz = X
        dfz = (Fz - fz0) / fz0
        C = pCy1
        D = np.maximum((pDy1 + pDy2 * dfz) * Fz, 1.0)
        K = pKy1 * fz0 * np.sin(2.0 * np.arctan(Fz / (pKy2 * fz0)))
        B = K / (C * D)
        E = np.minimum(pEy1 + pEy2 * dfz, 1.0)
        Ba = B * al
        return D * np.sin(C * np.arctan(Ba - E * (Ba - np.arctan(Ba))))

    try:
        from scipy.optimize import curve_fit
        p0 = [1.5, 2.5, -0.2, 20.0, 1.5, 0.0, 0.0]
        lo = [0.5, 0.5, -3.0, 1.0, 0.1, -5.0, -5.0]
        hi = [3.0, 6.0, 3.0, 200.0, 20.0, 1.0, 5.0]
        p, _ = curve_fit(mf, (a, fz), y, p0=p0, bounds=(lo, hi), maxfev=60000)
    except Exception:
        return None, float('nan'), None
    pred = mf((a, fz), *p)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    keys = ('pCy1', 'pDy1', 'pDy2', 'pKy1', 'pKy2', 'pEy1', 'pEy2')
    params = dict(zip(keys, [float(v) for v in p]))
    params['Fz0'] = fz0

    def curve(alpha_deg, Fz):
        return mf((np.radians(np.asarray(alpha_deg, float)),
                   np.asarray(Fz, float)), *p)

    def peak_slip(Fz):
        """Slip angle (deg) at the fitted peak — smooth by construction."""
        grid = np.linspace(0.0, 25.0, 1001)
        Fz = np.atleast_1d(np.asarray(Fz, float))
        out = []
        for f in Fz:
            v = curve(grid, np.full_like(grid, f))
            out.append(float(grid[int(np.argmax(v))]))
        return np.array(out) if len(out) > 1 else out[0]

    return params, r2, (curve, peak_slip)


class TireModel:
    """
    Processed tire model from TTC data.

    Builds 3D regular grids for Fy and Mz as functions of
    (slip_angle, normal_load, camber).  Lookup via trilinear
    interpolation with clamping at grid bounds.

    Typical usage:
        model = TireModel.from_mat_file('your_tire_data.mat')
        Fy = model.Fy(slip_angle_deg=3.0, Fz_N=800, camber_deg=2.0)
    """

    def __init__(self, ttc: TTCData,
                 sa_bin_deg: float = 0.5,
                 fz_n_bins: int = 8,
                 warmup_pts: int = 1500,
                 pressure_psi: float = None):
        """
        Process raw TTC data into interpolation grids.

        Parameters
        ----------
        ttc : TTCData
            Raw data from load_ttc_mat().
        sa_bin_deg : float
            Slip angle bin width in degrees (default 0.5).
        fz_n_bins : int
            Number of normal load bins (default 8).  Only used as a fallback —
            the load axis is normally taken from the test's own setpoints.
        warmup_pts : int
            Number of initial warmup samples to discard (default 1500).
        pressure_psi : float or None
            RUN THE TYRE AT ONE PRESSURE.  A TTC cornering run sweeps several
            (this Round 9 file holds 8/10/12/14 psi), and blending them into one
            grid is not a tyre — it is an average of four different tyres.  It
            matters more than it sounds: on this R20 the 8 psi sweep passes the
            peak inside the rig's +/-12 deg of slip, while the 14 psi sweep
            never reaches it at any load, so the blend smears a real peak
            against a censored one.  None keeps every sample (the old
            behaviour) and sets `pressure_blended = True` so callers can say so.
        """
        self.tire_id = ttc.tire_id
        self.test_id = ttc.test_id

        # ── Pressure selection ───────────────────────────────────────────
        p_raw = np.asarray(ttc.pressure_kPa, float)
        _pos = p_raw[p_raw > 0]
        # what the file actually contains, rounded to the nearest psi
        self.available_pressures_psi = sorted({
            float(np.round(v * 0.145038)) for v in np.unique(np.round(_pos))
        }) if _pos.size else []
        # collapse near-duplicates from ramp samples: keep the well-populated ones
        if _pos.size:
            _psi_all = _pos * 0.145038
            _keep = []
            for _p in self.available_pressures_psi:
                if np.sum(np.abs(_psi_all - _p) < 0.5) >= max(200, 0.02 * _psi_all.size):
                    _keep.append(_p)
            if _keep:
                self.available_pressures_psi = _keep
        # BLENDING IS BANNED (user order 2026-07-30: "NEVER use blended
        # data").  A TTC cornering run sweeps several pressures; averaging
        # them is not a tyre.  There is no blend mode any more — a caller
        # that does not pick a pressure gets a refusal, not an average.
        if pressure_psi is None:
            raise ValueError(
                'no tyre pressure selected — blending all pressures is '
                f'forbidden; this file holds {sorted(self.available_pressures_psi)} psi, '
                'pick one')
        self.pressure_requested_psi = float(pressure_psi)
        self.pressure_blended = False

        mask_p = np.ones(len(p_raw), dtype=bool)
        if pressure_psi is not None and _pos.size:
            tol_kPa = 0.5 / 0.145038                 # +/- 0.5 psi
            target_kPa = float(pressure_psi) / 0.145038
            mask_p = np.abs(p_raw - target_kPa) <= tol_kPa
            if mask_p.sum() < 500:
                raise ValueError(
                    f'tyre file has no usable data at {pressure_psi:.0f} psi '
                    f'({int(mask_p.sum())} samples); it holds '
                    f'{self.available_pressures_psi} psi')

        self.pressure_kPa_mean = (float(np.median(p_raw[mask_p & (p_raw > 0)]))
                                  if np.any(mask_p & (p_raw > 0)) else 0.0)
        self.pressure_psi = self.pressure_kPa_mean * 0.145038

        # Discard warmup
        n = len(ttc.slip_angle_deg)
        keep = mask_p.copy()
        if warmup_pts < n:
            keep[:warmup_pts] = False
        sl = keep

        sa = ttc.slip_angle_deg[sl]
        fz = ttc.normal_load_N[sl]
        fy = ttc.lateral_force_N[sl]
        mz = ttc.aligning_moment_Nm[sl]
        ia = ttc.camber_deg[sl]

        # Filter to positive velocity (exclude stationary data)
        mask_v = ttc.velocity_kph[sl] > 5.0
        sa, fz, fy, mz, ia = sa[mask_v], fz[mask_v], fy[mask_v], mz[mask_v], ia[mask_v]

        # ── Determine grid axes ──────────────────────────────────────────
        # Camber levels (round to nearest integer).  KEEP ONLY levels with
        # real data coverage: TTC tests sweep a few discrete inclination
        # angles (e.g. 0/2/4 deg), and the transition samples between them
        # round into PHANTOM integer levels (+-1, +-3, -2 ...) holding only
        # a sliver of points.  Those levels' (sa, fz) cells fail the
        # >=5-sample cut, get gap-filled as 0.0, and the camber-axis
        # interpolation then averages real rows against zero rows —
        # peak_mu(fz, 0.45deg) came out non-monotone garbage (mu 1.57@400N
        # rising to 2.39@939N) which corrupted utilization and the
        # understeer gradient in SteadyStateSolver.  A level must hold
        # >=1% of samples (and >=200 absolute) to be treated as a tested
        # inclination; stray-sample levels are dropped so interpolation
        # only ever spans measured rows.
        ia_rounded = np.round(ia).astype(int)
        _levels, _counts = np.unique(ia_rounded, return_counts=True)
        # A held camber setpoint owns a large share of the samples; a ramp
        # between setpoints rounds into a thin sliver.  The cut must be
        # RELATIVE to the biggest level, not an absolute count: the absolute
        # `max(200, 1%)` rule was defeated the moment the pressure filter
        # shrank the sample base — at 12 psi the IA=1/IA=3 transition slivers
        # cleared 200 samples, came back as phantom rows gap-filled to ~zero,
        # and every camber interpolation ran through them (Fy at +6 deg,
        # 222 N read +15 N at IA=1 between healthy -506 at IA=0 and -494 at
        # IA=2 — the light-wheel Ackermann inversion 'saturated' on that).
        # Same disease as GOTCHA 12, new entry hole.
        _min_count = max(200, int(0.20 * _counts.max()))
        _good = _counts >= _min_count
        if _good.sum() >= 2:      # interpolator needs >=2 rows; else keep all
            _levels = _levels[_good]
        self._camber_levels = np.sort(_levels).astype(float)

        # Slip angle axis
        sa_min = np.floor(sa.min())
        sa_max = np.ceil(sa.max())
        self._sa_axis = np.arange(sa_min, sa_max + sa_bin_deg, sa_bin_deg)

        # ── Normal load axis: THE TEST'S OWN LOAD SETPOINTS ──────────────
        # This used to be `np.linspace(percentile(fz,2), percentile(fz,98), 8)`,
        # which has nothing to do with how the tyre was actually tested.  The
        # rig HOLDS load at a handful of setpoints (this Round 9 run: 222, 444,
        # 667, 890, 1112 N = 50/100/150/200/250 lbf) and sweeps slip at each.
        # Evenly spaced bins put THREE of the eight bin centres in gaps where
        # no setpoint exists, so those bins could only be filled from samples
        # taken while the rig was RAMPING BETWEEN LOADS — and then a fifth of
        # each such curve was fabricated by the NaN gap-fill below.  That is
        # where the "peak at 12.5 deg for the 752 N bin" came from, sitting
        # between real neighbours at 8.5 and 9.0 deg.  Measured 2026-07-27:
        # the 752 N bin held 357 real high-slip samples vs ~1960 in a setpoint
        # bin, 19% of its curve interpolated.
        # Find the setpoints instead: they are the modes of the load histogram,
        # because the rig spends nearly all its time holding, not ramping.
        fz_lo, fz_hi = np.percentile(fz, [2, 98])
        _cnt, _edge = np.histogram(fz[(fz >= fz_lo) & (fz <= fz_hi)], bins=120)
        _mid = 0.5 * (_edge[:-1] + _edge[1:])
        _pk = [i for i in range(1, len(_cnt) - 1)
               if _cnt[i] >= _cnt[i - 1] and _cnt[i] >= _cnt[i + 1]
               and _cnt[i] > 0.25 * _cnt.max()]
        _sp = []
        for i in _pk:                      # merge peaks that are the same hold
            if _sp and _mid[i] - _sp[-1] < 0.06 * max(fz_hi, 1.0):
                continue
            _sp.append(float(_mid[i]))
        if len(_sp) >= 3:
            # snap each setpoint to the mean of the samples it actually owns
            _sp = np.array(sorted(_sp))
            _half = np.diff(_sp).min() * 0.5
            self._fz_axis = np.array([
                float(np.mean(fz[np.abs(fz - s) < _half])) if
                np.any(np.abs(fz - s) < _half) else float(s) for s in _sp])
            self.fz_binning = 'test setpoints'
        else:
            self._fz_axis = np.linspace(fz_lo, fz_hi, fz_n_bins)
            self.fz_binning = 'evenly spaced (no setpoints found)'
        # per-bin half width, since the setpoints are NOT evenly spaced
        if len(self._fz_axis) > 1:
            _d = np.diff(self._fz_axis)
            _hw = np.concatenate([[_d[0]], _d]) * 0.6
        else:
            _hw = np.array([max(fz_hi - fz_lo, 1.0) * 0.5])
        fz_half_width = _hw

        # ── Build grids ──────────────────────────────────────────────────
        n_sa = len(self._sa_axis)
        n_fz = len(self._fz_axis)
        n_ia = len(self._camber_levels)

        fy_grid = np.full((n_sa, n_fz, n_ia), np.nan)
        mz_grid = np.full((n_sa, n_fz, n_ia), np.nan)

        for k, ia_val in enumerate(self._camber_levels):
            mask_ia = ia_rounded == int(ia_val)
            for j, fz_val in enumerate(self._fz_axis):
                mask_fz = mask_ia & (np.abs(fz - fz_val) < fz_half_width[j])
                if mask_fz.sum() < 5:
                    continue
                sa_sub = sa[mask_fz]
                fy_sub = fy[mask_fz]
                mz_sub = mz[mask_fz]

                # Bin by slip angle and take median
                for i, sa_val in enumerate(self._sa_axis):
                    mask_sa = np.abs(sa_sub - sa_val) < sa_bin_deg * 0.6
                    if mask_sa.sum() >= 2:
                        fy_grid[i, j, k] = np.median(fy_sub[mask_sa])
                        mz_grid[i, j, k] = np.median(mz_sub[mask_sa])

        # ── Fill NaN gaps via interpolation along SA axis ────────────────
        for k in range(n_ia):
            for j in range(n_fz):
                fy_col = fy_grid[:, j, k]
                mz_col = mz_grid[:, j, k]
                valid = ~np.isnan(fy_col)
                if valid.sum() >= 3:
                    fy_grid[:, j, k] = np.interp(
                        self._sa_axis, self._sa_axis[valid], fy_col[valid])
                    mz_grid[:, j, k] = np.interp(
                        self._sa_axis, self._sa_axis[valid], mz_col[valid])
                elif valid.sum() > 0:
                    fy_grid[:, j, k] = np.nanmean(fy_col)
                    mz_grid[:, j, k] = np.nanmean(mz_col)
                else:
                    fy_grid[:, j, k] = 0.0
                    mz_grid[:, j, k] = 0.0

        # Smooth lightly along SA to reduce test noise
        for k in range(n_ia):
            for j in range(n_fz):
                fy_grid[:, j, k] = uniform_filter1d(fy_grid[:, j, k], size=3)
                mz_grid[:, j, k] = uniform_filter1d(mz_grid[:, j, k], size=3)

        # ── Remove the zero-slip force offset ────────────────────────────
        # The binned data does not pass through the origin: at zero slip and
        # zero camber this tyre reported up to -71 N (mu 0.21 at the 340 N bin),
        # scattering in BOTH directions across load bins, so it is plysteer /
        # conicity / belt-direction bias plus bin scatter, not physics — a race
        # tyre rolling straight and upright makes no side force.  Left in, it
        # put the force zero-crossing at -0.19 deg of slip instead of 0, which
        # is a 142 N step exactly where the lightly-loaded inner front tyre
        # lives, and it showed up as a notch in the Ackermann sweep.
        # Subtract the offset measured at ZERO CAMBER from every camber slice:
        # that pins the upright curve through the origin while leaving the
        # slice-to-slice difference — the real camber thrust — untouched.
        i0 = int(np.argmin(np.abs(self._camber_levels)))
        iz = int(np.argmin(np.abs(self._sa_axis)))
        if abs(self._sa_axis[iz]) < 1e-6:
            self.fy_zero_offset_N = fy_grid[iz, :, i0].copy()
            for k in range(n_ia):
                fy_grid[:, :, k] -= self.fy_zero_offset_N[None, :]
        else:
            self.fy_zero_offset_N = np.zeros(n_fz)

        # ── PHYSICS CLAMP: no force growth past the peak-slip line ─────
        # The rig's light-load curves never decline inside the +/-12 deg
        # sweep: the raw 222 N points climb to ~556 N by 8 deg and hold flat
        # to 13 (measured 2026-07-30, 12 psi, IA~0), so the grid kept paying
        # a few extra newtons for absurd inner-wheel slip — the Ackermann
        # force optimizer "preferred" >100% settings by 0.01-0.17% of total
        # force, which is crumbs, but an argmax eats crumbs.  A tyre cannot
        # keep GAINING force past its peak slip angle, and the peak moves to
        # HIGHER slip as load rises (RCVD p25/p715) — a light tyre peaks
        # early, it does not creep forever.  Anchor the peak-slip line on
        # the one load bin whose peak is resolved inside the sweep (same
        # rule as _calibrate_peak_slip), then walk every (load, camber)
        # column outward and forbid |Fy| from rising beyond that line:
        # running minimum, so any MEASURED decline passes through untouched
        # and only the unphysical creep is removed.  Applied to the grid
        # BEFORE the interpolators are built so Fy, peak_Fy, peak_mu and
        # slip_angle_for_Fy all inherit the same shape — one model.
        _slope = self.PEAK_SLIP_SLOPE_DEFAULT
        _pos = np.where(self._sa_axis >= 0.0)[0]
        _anchor = None
        for j in range(len(self._fz_axis) - 1, -1, -1):
            _col = np.abs(fy_grid[_pos, j, i0])
            _k = int(np.argmax(_col))
            if 2 < _k < len(_pos) - 3:
                _anchor = (float(self._fz_axis[j]),
                           abs(float(self._sa_axis[_pos[_k]])))
                break
        # MEASURE the slope instead of assuming it, from PNEUMATIC TRAIL
        # COLLAPSE.  When the contact patch fully slides, the self-centring
        # torque dies — and unlike the force peak, that collapse happens
        # INSIDE the rig's sweep at every load (measured 2026-07-30 at
        # 12 psi: trail <25% of its own peak by 6.8 deg at 222 N rising to
        # 8.8 deg at 1110 N; the 8 psi Fy peaks, which ARE resolvable,
        # confirm the same rising trend).  Fit a line through the per-load
        # collapse angles for the SLOPE; keep the LEVEL pinned to the one
        # load bin whose force peak is genuinely measured, by shifting the
        # line to pass through that anchor.  Falls back to the RCVD input
        # slope when fewer than 3 load bins resolve a collapse.
        _fit = []
        _sa_pos = self._sa_axis[_pos]
        for j in range(len(self._fz_axis)):
            _fyc = np.abs(fy_grid[_pos, j, i0])
            _mzc = np.abs(mz_grid[_pos, j, i0])
            _okm = (_sa_pos > 0.5) & (_fyc > 30.0)
            if _okm.sum() < 8:
                continue
            _tr = _mzc[_okm] / _fyc[_okm]
            _sao = _sa_pos[_okm]
            _kk = int(np.argmax(_tr))
            _beyond = _sao[_kk:][_tr[_kk:] < 0.25 * _tr[_kk]]
            if _beyond.size:
                _fit.append((float(self._fz_axis[j]), float(_beyond[0])))
        self.peak_slip_source = None
        if len(_fit) >= 3:
            _fx = np.array([f for f, _ in _fit]) / 1000.0
            _fy_ = np.array([a for _, a in _fit])
            _slope_meas = float(np.polyfit(_fx, _fy_, 1)[0])
            if 0.0 < _slope_meas < 6.0:
                _slope = _slope_meas
                self.peak_slip_source = (
                    f'slope {_slope:.2f} deg/kN measured from pneumatic-trail '
                    f'collapse ({len(_fit)} load bins); level anchored to the '
                    f'measured force peak')
        self.fy_grid_peak_clamped = _anchor is not None
        if _anchor is not None:
            _f0, _a0 = _anchor
            # publish the SAME line the clamp uses, so peak_slip_angle and
            # every consumer agree with the grid's shape
            self.peak_slip_anchor_N, self.peak_slip_anchor_deg = _f0, _a0
            self.peak_slip_slope_deg_per_kN = _slope
            self.peak_slip_measured = True
            _neg = np.where(self._sa_axis <= 0.0)[0][::-1]
            for j, _fzv in enumerate(self._fz_axis):
                _sap = max(_a0 + _slope * (float(_fzv) - _f0) / 1000.0, 0.5)
                for k in range(fy_grid.shape[2]):
                    for _side in (_pos, _neg):
                        _sa_side = np.abs(self._sa_axis[_side])
                        _beyond = _side[_sa_side > _sap]
                        _within = _side[_sa_side <= _sap]
                        if _beyond.size == 0 or _within.size == 0:
                            continue
                        _cap = abs(fy_grid[_within[-1], j, k])
                        _mag = np.minimum.accumulate(
                            np.concatenate(([_cap],
                                            np.abs(fy_grid[_beyond, j, k]))))[1:]
                        fy_grid[_beyond, j, k] = (
                            np.sign(fy_grid[_beyond, j, k]) * _mag)

        # ── Build interpolators ──────────────────────────────────────────
        self._fy_interp = RegularGridInterpolator(
            (self._sa_axis, self._fz_axis, self._camber_levels),
            fy_grid, method='linear', bounds_error=False, fill_value=None)

        self._mz_interp = RegularGridInterpolator(
            (self._sa_axis, self._fz_axis, self._camber_levels),
            mz_grid, method='linear', bounds_error=False, fill_value=None)

        # ── Precompute peak Fy for utilization calc ──────────────────────
        # For each (Fz, camber), find max |Fy| across all slip angles
        self._peak_fy_grid = np.max(np.abs(fy_grid), axis=0)  # (n_fz, n_ia)
        # DO NOT smooth this along the LOAD axis.  It used to, to suppress the
        # per-Fz scatter, and that was defensible while the load bins were
        # arbitrary percentile slices full of ramp-between-loads samples.  With
        # the bins now sitting on the test's real setpoints the scatter is
        # already gone, and the moving average does active harm AT THE ENDS,
        # where it drags each edge bin toward its only neighbour.  Measured
        # 2026-07-27 on the R20: it inflated the lightest bin (222 N) by
        # +37.6% because its neighbour at 443 N makes roughly twice the force,
        # and cut the heaviest (1110 N) by 4.5%.  peak_Fy feeds peak_mu, which
        # feeds utilization, the grip limit and the lap sim, so a 38% error at
        # the lightly loaded inner wheel propagated into every balance number.
        # An edge-biased filter is worse than the noise it removes.
        self.peak_fy_smoothed = False
        self._peak_fy_interp = RegularGridInterpolator(
            (self._fz_axis, self._camber_levels),
            self._peak_fy_grid, method='linear',
            bounds_error=False, fill_value=None)

        # ── Out-of-data behaviour ────────────────────────────────────────
        # RegularGridInterpolator was left to extrapolate, which on this grid
        # means HOLDING THE EDGE VALUE FLAT.  That reported a tyre at 3000 N
        # making the same 2605 N as one at 1160 N (mu 0.87), and the same force
        # at 45 deg of slip as at 13 deg.  Both flatter the car exactly where it
        # operates: the outer tyre passes the load ceiling at about 1.0 g.
        # Replace the clamp with the MEASURED TRENDS continued outward — no
        # invented curve shapes — and record every excursion so it is visible.
        self._fz_lo, self._fz_hi = float(self._fz_axis[0]), float(self._fz_axis[-1])
        self._sa_hi = float(np.max(np.abs(self._sa_axis)))
        # mu vs Fz slope over the UPPER HALF of the measured loads (the lower
        # loads carry most of the scatter and would drag the slope)
        _mu = self._peak_fy_grid[:, 0] / np.maximum(self._fz_axis, 1.0)
        _up = self._fz_axis >= np.median(self._fz_axis)
        if _up.sum() >= 2:
            self._mu_slope = float(np.polyfit(self._fz_axis[_up], _mu[_up], 1)[0])
        else:
            self._mu_slope = 0.0
        self._mu_edge = float(_mu[-1])
        # trailing slope of Fy vs slip at the top of the measured sweep, per deg
        _iu = np.argsort(self._sa_axis)
        _sa_s = self._sa_axis[_iu]
        _fy_s = np.abs(self._peak_fy_grid[-1, 0]) if False else None
        _tail = np.abs(fy_grid[_iu, -1, 0])
        if len(_sa_s) >= 3 and (_sa_s[-1] - _sa_s[-3]) > 1e-6:
            self._fy_tail_slope = float((_tail[-1] - _tail[-3]) / (_sa_s[-1] - _sa_s[-3]))
        else:
            self._fy_tail_slope = 0.0
        self._fy_tail_slope = min(self._fy_tail_slope, 0.0)   # never rising
        # ── BELOW the lowest measured load ───────────────────────────────
        # The grid clamps here too, and that direction is worse than the top
        # end: it handed a tyre carrying 24 N the full 727 N of the 209 N test
        # point (mu 30), and a tyre at literally zero load 727 N of grip.  The
        # inner front sits below the 209 N floor for 45% of cornering time, so
        # this is not an edge case, it is where the Ackermann answer lives.
        # Same treatment as the ceiling: continue the MEASURED mu-vs-Fz trend,
        # here fitted over the LOWER half of the loads, as Fy = mu(Fz)*Fz.  That
        # form goes to zero at zero load on its own — no invented cap needed —
        # while keeping the load sensitivity the data actually shows (mu rises
        # as load falls).
        _lo = self._fz_axis <= np.median(self._fz_axis)
        if _lo.sum() >= 2:
            _p = np.polyfit(self._fz_axis[_lo], _mu[_lo], 1)
            self._mu_lo_slope, self._mu_lo_int = float(_p[0]), float(_p[1])
        else:
            self._mu_lo_slope, self._mu_lo_int = 0.0, float(_mu[0])
        self._mu_lo_edge = float(_mu[0])
        self.out_of_range = {'fz_max': 0.0, 'fz_min': 0.0, 'sa_max': 0.0, 'n': 0}
        # This block used to sit inline here and was orphaned behind an earlier
        # `return`, so _peak_sa_interp was never built and slip_angle_at_peak_Fy
        # raised AttributeError.  Called explicitly now.
        self._build_peak_sa(fy_grid)

        # ── Pacejka fit on the RAW zero-camber samples ───────────────────
        # The binned grid cannot resolve how peak slip angle moves with load
        # (see _fit_magic_formula), and that trend is the whole Ackermann
        # design lever.  Fit the raw data instead.  Kept ALONGSIDE the grid,
        # not replacing it: the grid stays the default lookup so this does not
        # silently move every number in the tool, while anything that needs a
        # smooth peak asks the fit and can report its R^2.
        _m0 = np.abs(ia) <= 0.5
        self.mf_params, self.mf_r2, _mf = _fit_magic_formula(
            sa[_m0], fz[_m0], fy[_m0])
        self._mf_curve, self._mf_peak_slip = _mf if _mf else (None, None)
        # peak-slip-vs-load calibration (needs peak_Fy + cornering_stiffness,
        # so it must come after every grid and extrapolation is in place)
        self._calibrate_peak_slip()

    def _mu_below_floor(self, fz):
        """Measured mu trend continued below the lowest tested load.

        Anchored to the mu actually measured AT the floor rather than to the
        fit's own intercept, so mu is continuous across the floor instead of
        stepping 0.15 at the handover.
        """
        return np.maximum(
            self._mu_lo_edge + self._mu_lo_slope * (fz - self._fz_lo), 0.05)

    def _note_excursion(self, sa=None, fz=None):
        """Record how far outside the measured data a query went."""
        if fz is not None:
            fza = np.asarray(fz, float)
            over = float(np.max(fza)) - self._fz_hi
            if over > 0:
                self.out_of_range['fz_max'] = max(self.out_of_range['fz_max'], over)
                self.out_of_range['n'] += 1
            under = self._fz_lo - float(np.min(fza))
            if under > 0:
                self.out_of_range['fz_min'] = max(self.out_of_range['fz_min'], under)
                self.out_of_range['n'] += 1
        if sa is not None:
            over = float(np.max(np.abs(np.asarray(sa, float)))) - self._sa_hi
            if over > 0:
                self.out_of_range['sa_max'] = max(self.out_of_range['sa_max'], over)
                self.out_of_range['n'] += 1

    def data_range_report(self) -> str:
        """Human-readable note on how far anything has queried outside the data."""
        o = self.out_of_range
        if not o['n']:
            return (f'all queries inside measured data '
                    f'(Fz {self._fz_lo:.0f}-{self._fz_hi:.0f} N, |SA| <= {self._sa_hi:.0f} deg)')
        bits = []
        if o['fz_max'] > 0:
            bits.append(f'Fz up to {self._fz_hi + o["fz_max"]:.0f} N '
                        f'({100*o["fz_max"]/self._fz_hi:.0f}% over the {self._fz_hi:.0f} N ceiling)')
        if o['fz_min'] > 0:
            bits.append(f'Fz down to {max(self._fz_lo - o["fz_min"], 0.0):.0f} N '
                        f'(under the {self._fz_lo:.0f} N floor)')
        if o['sa_max'] > 0:
            bits.append(f'|SA| up to {self._sa_hi + o["sa_max"]:.1f} deg '
                        f'(data ends at {self._sa_hi:.0f})')
        return 'EXTRAPOLATED beyond measured data: ' + '; '.join(bits)

    def _build_peak_sa(self, fy_grid):
        # ── Precompute the SLIP ANGLE at peak Fy ─────────────────────────
        # NOT by argmax.  Past about 6 deg these curves are flat, so the
        # location of the maximum is set by noise: on the R20 data argmax
        # scatters 7.5-13 deg with R2 = 0.07 against load AND SLOPES THE WRONG
        # WAY.  Every consumer that wanted a peak slip angle was doing its own
        # argmax and inheriting that, which is why the Ackermann demand curve
        # pointed backwards.
        # Instead fit a Magic-Formula shape D*sin(C*atan(B*a)) through the whole
        # curve and take its analytic peak a_pk = (pi/2C)/B.  Using every point
        # rather than the single highest one recovers R2 = 0.83, rising with
        # load, which is the physically correct direction.
        self._peak_sa_grid = np.zeros_like(self._peak_fy_grid)
        for j in range(len(self._fz_axis)):
            for k in range(len(self._camber_levels)):
                self._peak_sa_grid[j, k] = _fit_peak_sa(
                    self._sa_axis, np.abs(fy_grid[:, j, k]))
        if self._peak_sa_grid.shape[0] >= 3:
            for k in range(self._peak_sa_grid.shape[1]):
                self._peak_sa_grid[:, k] = uniform_filter1d(
                    self._peak_sa_grid[:, k], size=3)
        self._peak_sa_interp = RegularGridInterpolator(
            (self._fz_axis, self._camber_levels),
            self._peak_sa_grid, method='linear',
            bounds_error=False, fill_value=None)

    @classmethod
    def from_mat_file(cls, path: str, **kwargs) -> "TireModel":
        """Load .mat and process in one call."""
        return cls(load_ttc_mat(path), **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "TireModel":
        """Load from any supported format (.mat, .csv, .xlsx) and process."""
        return cls(load_tire_data(path), **kwargs)

    # ── Public lookup API ────────────────────────────────────────────────

    def Fy(self, slip_angle_deg, Fz_N, camber_deg=0.0):
        """
        Lateral force (N).

        Accepts scalar or array inputs (broadcast together).
        Clamps to grid bounds for out-of-range queries.
        """
        sa = np.atleast_1d(np.asarray(slip_angle_deg, float))
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        sa_b, fz_b = np.broadcast_arrays(sa, fz)
        self._note_excursion(sa=sa_b, fz=fz_b)
        pts = self._make_pts(slip_angle_deg, Fz_N, camber_deg)
        out = np.atleast_1d(self._fy_interp(pts))

        # Past the measured slip sweep the grid holds flat; continue the curve's
        # OWN trailing slope instead so over-slipping costs force, as it must.
        over_sa = np.abs(sa_b).ravel() - self._sa_hi
        m = over_sa > 0
        if np.any(m) and self._fy_tail_slope < 0:
            out.ravel()[m] = out.ravel()[m] + np.sign(
                out.ravel()[m]) * self._fy_tail_slope * over_sa[m]

        # Above the top measured load the grid also holds flat, which INVENTS
        # grip.  Continue the measured mu-vs-Fz decay instead.
        over_fz = fz_b.ravel() - self._fz_hi
        m2 = over_fz > 0
        if np.any(m2):
            mu_ext = self._mu_edge + self._mu_slope * over_fz[m2]
            mu_ext = np.maximum(mu_ext, 0.05)
            scale = np.clip(mu_ext * fz_b.ravel()[m2]
                            / np.maximum(self._mu_edge * self._fz_hi, 1e-9), 0.0, 10.0)
            out.ravel()[m2] = out.ravel()[m2] * scale

        # Below the lowest measured load the grid held flat as well, so an
        # unloaded tyre still made the 209 N test point's force.  Scale by the
        # measured mu trend times the ACTUAL load, which vanishes at zero load.
        under = self._fz_lo - fz_b.ravel()
        m3 = under > 0
        if np.any(m3):
            fzu = np.maximum(fz_b.ravel()[m3], 0.0)
            scale = self._mu_below_floor(fzu) * fzu \
                / np.maximum(self._mu_lo_edge * self._fz_lo, 1e-9)
            out.ravel()[m3] = out.ravel()[m3] * np.clip(scale, 0.0, 1.0)
        return out.squeeze()

    def Mz(self, slip_angle_deg, Fz_N, camber_deg=0.0):
        """Aligning moment (Nm)."""
        pts = self._make_pts(slip_angle_deg, Fz_N, camber_deg)
        return self._mz_interp(pts).squeeze()

    def peak_Fy(self, Fz_N, camber_deg=0.0):
        """Maximum |Fy| across all slip angles at given (Fz, camber)."""
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        ia = np.atleast_1d(np.asarray(camber_deg, float))
        fz, ia = np.broadcast_arrays(fz, ia)
        pts = np.column_stack([
            np.clip(fz, self._fz_axis[0], self._fz_axis[-1]),
            np.clip(ia, self._camber_levels[0], self._camber_levels[-1]),
        ])
        out = np.atleast_1d(self._peak_fy_interp(pts))
        self._note_excursion(fz=fz)
        over = np.asarray(fz, float).ravel() - self._fz_hi
        m = over > 0
        if np.any(m):
            mu_ext = np.maximum(self._mu_edge + self._mu_slope * over[m], 0.05)
            out.ravel()[m] = mu_ext * np.asarray(fz, float).ravel()[m]
        under = self._fz_lo - np.asarray(fz, float).ravel()
        m2 = under > 0
        if np.any(m2):
            fzu = np.maximum(np.asarray(fz, float).ravel()[m2], 0.0)
            out.ravel()[m2] = np.sign(out.ravel()[m2] + 1e-30) \
                * self._mu_below_floor(fzu) * fzu
        return out.squeeze()

    def slip_angle_at_peak_Fy(self, Fz_N, camber_deg=0.0):
        """Slip angle (deg) near peak |Fy|.  TREAT AS INDICATIVE ONLY.

        NOT reliable for Ackermann demand.  Measured on FSAE slick data
        2026-07-27: Fy(alpha) SATURATES rather than peaking — it is flat from
        roughly 6 deg out to the end of the sweep — so "the slip angle at peak"
        is not a well-posed quantity for this tyre.  Every estimator tried gave
        R^2 <= 0.33 against vertical load, and the implied inner-minus-outer
        demand FLIPS SIGN with the choice of estimator:
            argmax  +3.02 deg  |  99% of max  +0.31  |  95%  +0.21  |  90%  -0.11
        Ackermann percentage therefore CANNOT be set from this data.  Choose it
        on packaging, low-speed manoeuvring and tyre-wear grounds and validate on
        track.  Kept only as a rough indicator; do not build a conclusion on it.
        """
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        ia = np.atleast_1d(np.asarray(camber_deg, float))
        fz, ia = np.broadcast_arrays(fz, ia)
        pts = np.column_stack([
            np.clip(fz, self._fz_axis[0], self._fz_axis[-1]),
            np.clip(ia, self._camber_levels[0], self._camber_levels[-1]),
        ])
        return self._peak_sa_interp(pts).squeeze()

    # How fast the slip angle at peak lateral force rises with vertical load,
    # in deg per 1000 N.  AN INPUT, NOT A MEASUREMENT — see peak_slip_angle.
    PEAK_SLIP_SLOPE_DEFAULT = 2.0

    def peak_slip_angle(self, Fz_N, camber_deg=0.0):
        """Slip angle (deg) at peak Fy vs load — the quantity Ackermann is
        designed around.

        THIS DATASET CANNOT MEASURE THE TREND.  That is the honest finding, and
        the reason this is a straight line with an INPUT slope rather than
        something derived from the curves:

        * The rig sweeps only +/-12 deg. At most pressures the light-load
          curves are still climbing when the sweep ends, so there is no peak in
          the data to find. `argmax` then returns the last sample and reports
          the light tyre peaking LATE, inverting the trend.
        * Every estimator disagrees, in sign as well as magnitude. Measured
          2026-07-27 on this R20: argmax -5.0, 99%-of-peak -3.5, 95% -1.4,
          the Pacejka fit -12.9 (its E term pinned at its cap below 800 N),
          and D/K +3.4 deg/1000 N.
        * D/K was tried here and REFUTED. For a_pk = k*D/K to hold, one k must
          fit every load; the implied k actually falls monotonically with load
          by a factor of 2.16 at 12 psi (and 2.08 / 1.73 at 10 / 14 psi), so
          the proportionality is false for this tyre. Worse, the answer became
          a choice of anchor bin: anchoring on each of the five equally-valid
          bins gave +6.80 / +5.86 / +4.59 / +3.15 / +3.30 deg/1000 N.

        So: the LEVEL is anchored to the one load bin the rig genuinely carries
        past its peak (a measurement), and the SLOPE is an input the user owns.
        The default is +2.0 deg/1000 N, which is NOT from this file — it comes
        from RCVD (printed p.25: peak lateral force occurs at a higher slip
        angle as load increases; p.715 says the same from the light-load end)
        and from two censoring-free estimates on the raw data that bracket
        +1.2 to +2.4 deg/1000 N. Set `peak_slip_slope_deg_per_kN` to your own
        value; set it negative if you want to test the opposite belief.
        """
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        a0 = float(self.peak_slip_anchor_deg)
        f0 = float(self.peak_slip_anchor_N or 0.0)
        s = float(self.peak_slip_slope_deg_per_kN)
        out = np.maximum(a0 + s * (fz.ravel() - f0) / 1000.0, 0.5)
        return float(out[0]) if out.size == 1 else out

    def _calibrate_peak_slip(self):
        """Measure the LEVEL at the one load bin that resolves its own peak.

        Only the level is measured here.  The slope is an input — the data
        cannot support one (see peak_slip_angle).
        """
        if getattr(self, 'peak_slip_source', None):
            # slope already MEASURED (pneumatic-trail collapse) and the grid
            # clamped with it at build time — recalibrating here on the
            # clamped grid would just echo the clamp back as a measurement.
            return
        self.peak_slip_slope_deg_per_kN = self.PEAK_SLIP_SLOPE_DEFAULT
        sa = np.linspace(0.0, float(self._sa_axis[-1]), 200)
        best = None
        for j in range(len(self._fz_axis) - 1, -1, -1):
            f = float(self._fz_axis[j])
            v = np.array([abs(float(self.Fy(a, f, 0.0))) for a in sa])
            k = int(np.argmax(v))
            # a real peak means the maximum is INSIDE the sweep, not at its end
            if k < len(sa) - 3:
                best = (f, float(sa[k]))
                break
        if best is None:
            # No load bin resolves its peak at all: there is nothing to anchor
            # to.  Say so rather than inventing a level.
            self.peak_slip_anchor_N = None
            self.peak_slip_anchor_deg = float('nan')
            self.peak_slip_measured = False
            return
        self.peak_slip_anchor_N, self.peak_slip_anchor_deg = best
        self.peak_slip_measured = True

    def report_dict(self):
        """Everything a report needs to document this tyre model — derived
        numbers only, computed HERE so binder and GUI can never disagree.

        DE-IDENTIFIED BY CONSTRUCTION: no tyre make/compound/size string leaves
        this function.  The repo is public and the TTC agreement allows derived
        numbers with attribution, not identification (memory:
        project_ttc_publishing).  Do not add tire_id to this dict.
        """
        rows = []
        for f in self._fz_axis:
            f = float(f)
            pk = abs(float(self.peak_Fy(f, 0.0)))
            ca = abs(float(self.cornering_stiffness(f, 0.0)))
            rows.append({
                'fz_N': round(f), 'peak_fy_N': round(pk),
                'mu': round(pk / f, 3),
                'cornering_stiffness_N_per_deg': round(ca, 1),
                'peak_slip_deg': round(float(self.peak_slip_angle(f)), 2),
            })
        off = np.asarray(getattr(self, 'fy_zero_offset_N', [0.0]), float)
        return {
            'source': ('FSAE Tire Test Consortium data, measured on the '
                       'Calspan TIRF belt rig. De-identified per the TTC '
                       'agreement.'),
            'pressure_psi': round(float(self.pressure_psi), 2),
            'pressure_blended': bool(self.pressure_blended),
            'pressures_available_psi': list(self.available_pressures_psi),
            'fz_binning': str(getattr(self, 'fz_binning', '?')),
            'fz_setpoints_N': [round(float(v)) for v in self._fz_axis],
            'sa_sweep_deg': [float(self._sa_axis[0]), float(self._sa_axis[-1])],
            'camber_levels_deg': [float(v) for v in self._camber_levels],
            'per_load': rows,
            'zero_slip_offset_removed_max_N': round(float(np.max(np.abs(off))), 1),
            'mf_fit_r2': round(float(self.mf_r2), 4) if self.mf_r2 == self.mf_r2
                         else None,
            'peak_slip_anchor_N': (round(float(self.peak_slip_anchor_N))
                                   if self.peak_slip_anchor_N else None),
            'peak_slip_anchor_deg': (round(float(self.peak_slip_anchor_deg), 2)
                                     if self.peak_slip_anchor_deg ==
                                     self.peak_slip_anchor_deg else None),
            'peak_slip_slope_deg_per_kN':
                float(self.peak_slip_slope_deg_per_kN),
            'peak_slip_slope_is_assumption': True,
            'limits': [
                f'Slip sweep ends at ±{abs(float(self._sa_axis[-1])):.0f}°; at '
                f'light loads the force is still rising there, so light-load '
                f'peak locations are not measured — the peak-slip TREND is an '
                f'input, only its level (at the anchor load) is measured.',
                f'Above {float(self._fz_hi):.0f} N and below '
                f'{float(self._fz_lo):.0f} N every value is extrapolated by '
                f'continuing the measured mu-vs-load trend (no invented curve '
                f'shapes); force goes to zero at zero load.',
                'Belt-rig grip exceeds asphalt; a 0.65-0.75 derate applies to '
                'any absolute grip number.',
            ],
        }

    def mf_Fy(self, slip_angle_deg, Fz_N):
        """Lateral force from the PACEJKA FIT (SAE-signed, like Fy()).

        Smooth by construction, so unlike the binned grid it has a single
        well-defined peak whose location moves smoothly with load.  Use this
        wherever a peak or a slope matters (Ackermann, pair analysis, MMD);
        the grid stays the default lookup everywhere else.
        """
        if self._mf_curve is None:
            raise RuntimeError('Magic Formula fit unavailable for this tyre')
        return -self._mf_curve(slip_angle_deg, Fz_N)

    def mf_peak_slip_angle(self, Fz_N):
        """Slip angle (deg) at peak Fy from the fit — smooth against load."""
        if self._mf_peak_slip is None:
            raise RuntimeError('Magic Formula fit unavailable for this tyre')
        return self._mf_peak_slip(Fz_N)

    def cornering_stiffness(self, Fz_N, camber_deg=0.0):
        """
        C_alpha = dFy/d(alpha) at alpha=0 (N/deg).

        Central difference over ±0.5 deg.
        """
        da = 0.5
        fy_pos = self.Fy(da, Fz_N, camber_deg)
        fy_neg = self.Fy(-da, Fz_N, camber_deg)
        return (fy_pos - fy_neg) / (2 * da)

    def peak_mu(self, Fz_N, camber_deg=0.0):
        """Peak friction coefficient: mu = peak_Fy / Fz.

        LOAD SENSITIVITY: mu falls as Fz rises (real, measured tire physics).
        Above the tested Fz range we EXTEND that decline using the last
        measured segment's slope.  Two physical guards, NOT optimistic ones:

          * absolute floor 0.3 — warm rubber keeps a sliding-friction grip
            of ~0.3 even hopelessly overloaded; mu never goes to zero.
          * grip-FORCE ceiling — total grip Fy = mu*Fz may NOT exceed the
            peak force the tire ever made.  Without this, a constant-mu floor
            makes Fy climb again with load ("more load -> more grip"), the
            exact error load sensitivity exists to remove.  With it, grip
            force saturates and then declines as load keeps rising.
        """
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        fz_max = float(self._fz_axis[-1])
        peak = self.peak_Fy(fz, camber_deg)
        mu = np.where(fz > 1.0, peak / np.maximum(fz, 1.0), 0.0)
        over = fz > fz_max
        if np.any(over):
            # MARGINAL-SLOPE extrapolation (2026-07-12, replaces the hard
            # grip-FORCE ceiling).  History of this guard, because it keeps
            # oscillating: v1 floored mu at a constant -> grip force RE-ROSE
            # with load ("more load -> more grip", unphysical).  v2 capped
            # grip force at the max measured Fy -> ZERO marginal grip beyond
            # the data ceiling, so added load (aero!) bought nothing and the
            # aero-required solver slammed into its cap (the 6 kN plateau).
            # Both extremes are wrong: measured load sensitivity means
            # DIMINISHING marginal grip, not zero and not constant mu.
            # v3: continue Fy_peak(Fz) beyond the ceiling at the MEASURED
            # marginal slope of the top ~40% of the load grid (robust secant
            # on the smoothed peak grid), held constant.  mu then decays
            # hyperbolically toward that slope — load sensitivity preserved,
            # no zero-cliff, no runaway.  Floor mu at 0.3 (sliding rubber).
            fy_at_max = float(self.peak_Fy(fz_max, camber_deg))
            i_lo = max(0, int(0.6 * (len(self._fz_axis) - 1)))
            f_lo = float(self._fz_axis[i_lo])
            fy_lo = float(self.peak_Fy(f_lo, camber_deg))
            m_top = (fy_at_max - fy_lo) / max(fz_max - f_lo, 1e-6)
            m_top = float(np.clip(m_top, 0.0, fy_at_max / fz_max))
            fy_ext = fy_at_max + m_top * (fz - fz_max)
            mu_ext = np.maximum(fy_ext / np.maximum(fz, 1.0), 0.3)
            mu = np.where(over, mu_ext, mu)
        return mu.squeeze()

    def slip_angle_for_Fy(self, Fy_target_N, Fz_N, camber_deg=0.0):
        """
        Inverse lookup: find the slip angle that produces a given |Fy|.

        Searches SA range using abs(Fy) to handle TTC sign convention
        (positive SA → negative Fy). Returns the slip angle (deg) on
        the rising portion of the |Fy| vs |SA| curve.
        If Fy_target exceeds peak Fy, returns the SA at peak.
        """
        target = abs(float(Fy_target_N))
        fz = float(Fz_N)
        ia = float(camber_deg)
        self.last_lookup_saturated = False
        if target < 1.0 or fz < 1.0:
            return 0.0
        sa_hi = float(self._sa_axis[-1])
        # Which sign of slip raises |Fy| (TTC sign convention varies by file)
        sa_sign = 1.0 if abs(float(self.Fy(2.0, fz, ia))) >= \
            abs(float(self.Fy(-2.0, fz, ia))) else -1.0

        # REACHABILITY MUST BE TESTED AGAINST THE PEAK, NOT THE END OF THE
        # SWEEP.  This used to compare against |Fy| at +/-13 deg, the last
        # measured slip angle.  On this tyre |Fy| PEAKS near 9.9 deg and then
        # FALLS to 13, so every target between the two was wrongly declared
        # impossible — and the fallback then returned whichever point of a
        # 30-sample linspace happened to be highest.  linspace(0, 13, 30)[22]
        # is 9.862 deg, which is where the solver's mystery "9.86" came from:
        # an array index, not an angle the tyre had anything to do with.
        scan = np.linspace(0.0, sa_hi, 200)
        fy_scan = np.array([abs(float(self.Fy(sa_sign * s, fz, ia)))
                            for s in scan])
        k_pk = int(np.argmax(fy_scan))
        fy_pk = float(fy_scan[k_pk])
        if target >= fy_pk:
            # Genuinely unreachable — report the peak angle AND say so, so a
            # caller can tell a saturated tyre from a solved one.
            self.last_lookup_saturated = True
            return float(scan[k_pk])

        # Bisect on the RISING branch only (0 .. peak), which is monotone.
        # Searching the whole sweep could land on the falling side and return a
        # larger angle for the same force.
        lo, hi = 0.0, float(scan[k_pk])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if abs(float(self.Fy(sa_sign * mid, fz, ia))) < target:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < 1e-4:
                break
        return 0.5 * (lo + hi)

    # ── Grid info ────────────────────────────────────────────────────────

    @property
    def sa_range(self):
        return (self._sa_axis[0], self._sa_axis[-1])

    @property
    def fz_range(self):
        return (self._fz_axis[0], self._fz_axis[-1])

    @property
    def camber_levels(self):
        return self._camber_levels.copy()

    @property
    def load_levels(self):
        return self._fz_axis.copy()

    # ── Internals ────────────────────────────────────────────────────────

    def _make_pts(self, sa, fz, ia):
        sa = np.atleast_1d(np.asarray(sa, float))
        fz = np.atleast_1d(np.asarray(fz, float))
        ia = np.atleast_1d(np.asarray(ia, float))
        sa, fz, ia = np.broadcast_arrays(sa, fz, ia)
        return np.column_stack([
            np.clip(sa, self._sa_axis[0], self._sa_axis[-1]),
            np.clip(fz, self._fz_axis[0], self._fz_axis[-1]),
            np.clip(ia, self._camber_levels[0], self._camber_levels[-1]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  Linear fallback (no TTC data needed)
# ─────────────────────────────────────────────────────────────────────────────

class LinearTireModel:
    """
    Simple parametric tire model used when no TTC .mat data is loaded.

    Per RCVD audit (chapters 2.4, 2.5, 14.2) the parametric model needs
    to capture THREE load-sensitivities and camber thrust:

    1. Cornering stiffness vs Fz:
           C_alpha(Fz) = C_alpha_ref * (Fz / Fz_ref) ** load_sensitivity_Ca
       load_sensitivity_Ca < 1.0  ->  degressive (realistic, RCVD p27-29).

    2. Peak friction coefficient vs Fz (RCVD section 2.4, fig 2.10):
           mu(Fz)     = mu_ref     * (Fz / Fz_ref) ** (load_sensitivity_mu - 1)
       load_sensitivity_mu < 1.0  ->  peak mu falls with load.  Typical
       FSAE Hoosier ~0.85.  Previously this was a constant -- understeer
       gradient errors up to 15 % at limit.

    3. Camber thrust (RCVD section 2.5, fig 2.24-2.26):
           Fy_camber = C_gamma * sin(camber_rad)   (adds to slip-angle Fy)
       For small camber angles sin(gamma) ~= gamma so the relationship
       is approximately linear in degrees: C_gamma ~ 0.1 * C_alpha per
       RCVD's typical numbers.  Default disabled (C_gamma_N_per_deg=0)
       so old behaviour is preserved unless the user opts in.

    Fy_total = -C_alpha(Fz) * slip_angle  +  C_gamma * sin(camber_rad),
    saturated symmetrically at mu(Fz) * Fz.
    """

    def __init__(self, C_alpha_N_per_deg: float = 200.0,
                 Fz_ref_N: float = 700.0,
                 mu: float = 1.5,
                 load_sensitivity: float = 0.8,
                 load_sensitivity_mu: float = 0.9,
                 C_gamma_N_per_deg: float = 0.0):
        self._Ca = C_alpha_N_per_deg
        self._Fz_ref = Fz_ref_N
        self._mu_ref = mu
        self._ls_Ca = load_sensitivity
        self._ls_mu = load_sensitivity_mu
        self._Cg = C_gamma_N_per_deg
        self.tire_id = (f'Linear (Ca={C_alpha_N_per_deg}, mu={mu}, '
                        f'ls_Ca={load_sensitivity}, ls_mu={load_sensitivity_mu}, '
                        f'Cg={C_gamma_N_per_deg})')

    # ─── Backward-compat: callers reading ._mu still get the reference ──
    @property
    def _mu(self):
        return self._mu_ref

    def _mu_of_Fz(self, fz_arr: np.ndarray) -> np.ndarray:
        """RCVD load-sensitive peak mu: mu(Fz) = mu_ref * (Fz/Fz_ref)^(ls_mu - 1).

        Floored at mu_ref / 4 to avoid pathological zero-mu at very low Fz
        (which can happen for an inside wheel near liftoff -- numerically
        the formula goes to infinity there, which is wrong).
        """
        ratio = np.maximum(fz_arr / self._Fz_ref, 1e-3)
        mu = self._mu_ref * ratio ** (self._ls_mu - 1.0)
        return np.minimum(np.maximum(mu, self._mu_ref / 4.0),
                          self._mu_ref * 2.0)

    def Fy(self, slip_angle_deg, Fz_N, camber_deg=0.0):
        sa = np.atleast_1d(np.asarray(slip_angle_deg, float))
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        gamma = np.atleast_1d(np.asarray(camber_deg, float))
        fz_safe = np.maximum(fz, 0.0)
        ca = self._Ca * (fz_safe / self._Fz_ref) ** self._ls_Ca
        # Slip-angle component
        fy_slip = -ca * sa
        # Camber thrust component (RCVD section 2.5).  sin(gamma_rad) ~= gamma
        # for small angles; using sin keeps it exact at large camber too.
        fy_camber = self._Cg * np.sin(np.deg2rad(gamma))
        fy = fy_slip + fy_camber
        # Saturate at load-sensitive peak
        mu = self._mu_of_Fz(fz_safe)
        fy_max = mu * fz_safe
        return np.clip(fy, -fy_max, fy_max).squeeze()

    def Mz(self, slip_angle_deg, Fz_N, camber_deg=0.0):
        return np.zeros_like(np.atleast_1d(np.asarray(slip_angle_deg, float))).squeeze()

    def peak_Fy(self, Fz_N, camber_deg=0.0):
        fz = np.atleast_1d(np.maximum(np.asarray(Fz_N, float), 0.0))
        return (self._mu_of_Fz(fz) * fz).squeeze()

    def cornering_stiffness(self, Fz_N, camber_deg=0.0):
        fz = np.atleast_1d(np.maximum(np.asarray(Fz_N, float), 0.0))
        return (self._Ca * (fz / self._Fz_ref) ** self._ls_Ca).squeeze()

    def peak_mu(self, Fz_N, camber_deg=0.0):
        """Now load-sensitive per RCVD section 2.4 (previously constant)."""
        fz = np.atleast_1d(np.maximum(np.asarray(Fz_N, float), 0.0))
        return self._mu_of_Fz(fz).squeeze()

    def slip_angle_for_Fy(self, Fy_target_N, Fz_N, camber_deg=0.0):
        fz = max(float(Fz_N), 0.0)
        if fz < 1.0:
            return 0.0
        ca = self._Ca * (fz / self._Fz_ref) ** self._ls_Ca
        if ca < 0.1:
            return 0.0
        # Subtract camber-thrust contribution from the target before
        # solving the slip-angle component -- otherwise we under-predict
        # required slip angle when there's camber.
        fy_camber = self._Cg * float(np.sin(np.deg2rad(camber_deg)))
        fy_residual = float(Fy_target_N) - fy_camber
        sa = abs(fy_residual) / ca
        mu = float(self._mu_of_Fz(np.array([fz]))[0])
        return min(sa, abs(fy_residual) / (mu * fz + 1e-9) + 2.0)
