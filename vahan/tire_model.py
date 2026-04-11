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

class TireModel:
    """
    Processed tire model from TTC data.

    Builds 3D regular grids for Fy and Mz as functions of
    (slip_angle, normal_load, camber).  Lookup via trilinear
    interpolation with clamping at grid bounds.

    Typical usage:
        model = TireModel.from_mat_file('B2356run5.mat')
        Fy = model.Fy(slip_angle_deg=3.0, Fz_N=800, camber_deg=2.0)
    """

    def __init__(self, ttc: TTCData,
                 sa_bin_deg: float = 0.5,
                 fz_n_bins: int = 8,
                 warmup_pts: int = 1500):
        """
        Process raw TTC data into interpolation grids.

        Parameters
        ----------
        ttc : TTCData
            Raw data from load_ttc_mat().
        sa_bin_deg : float
            Slip angle bin width in degrees (default 0.5).
        fz_n_bins : int
            Number of normal load bins (default 8).
        warmup_pts : int
            Number of initial warmup samples to discard (default 1500).
        """
        self.tire_id = ttc.tire_id

        # Discard warmup
        n = len(ttc.slip_angle_deg)
        if warmup_pts < n:
            sl = slice(warmup_pts, None)
        else:
            sl = slice(None)

        sa = ttc.slip_angle_deg[sl]
        fz = ttc.normal_load_N[sl]
        fy = ttc.lateral_force_N[sl]
        mz = ttc.aligning_moment_Nm[sl]
        ia = ttc.camber_deg[sl]

        # Filter to positive velocity (exclude stationary data)
        mask_v = ttc.velocity_kph[sl] > 5.0
        sa, fz, fy, mz, ia = sa[mask_v], fz[mask_v], fy[mask_v], mz[mask_v], ia[mask_v]

        # ── Determine grid axes ──────────────────────────────────────────
        # Camber levels (round to nearest integer)
        ia_rounded = np.round(ia).astype(int)
        self._camber_levels = np.sort(np.unique(ia_rounded)).astype(float)

        # Slip angle axis
        sa_min = np.floor(sa.min())
        sa_max = np.ceil(sa.max())
        self._sa_axis = np.arange(sa_min, sa_max + sa_bin_deg, sa_bin_deg)

        # Normal load axis (evenly spaced bins covering the data range)
        fz_lo, fz_hi = np.percentile(fz, [2, 98])
        self._fz_axis = np.linspace(fz_lo, fz_hi, fz_n_bins)
        fz_bin_width = self._fz_axis[1] - self._fz_axis[0]

        # ── Build grids ──────────────────────────────────────────────────
        n_sa = len(self._sa_axis)
        n_fz = len(self._fz_axis)
        n_ia = len(self._camber_levels)

        fy_grid = np.full((n_sa, n_fz, n_ia), np.nan)
        mz_grid = np.full((n_sa, n_fz, n_ia), np.nan)

        for k, ia_val in enumerate(self._camber_levels):
            mask_ia = ia_rounded == int(ia_val)
            for j, fz_val in enumerate(self._fz_axis):
                mask_fz = mask_ia & (np.abs(fz - fz_val) < fz_bin_width * 0.6)
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
        self._peak_fy_interp = RegularGridInterpolator(
            (self._fz_axis, self._camber_levels),
            self._peak_fy_grid, method='linear',
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
        pts = self._make_pts(slip_angle_deg, Fz_N, camber_deg)
        return self._fy_interp(pts).squeeze()

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
        return self._peak_fy_interp(pts).squeeze()

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
        """Peak friction coefficient: mu = peak_Fy / Fz."""
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        peak = self.peak_Fy(fz, camber_deg)
        return np.where(fz > 1.0, peak / fz, 0.0).squeeze()

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
        if target < 1.0 or fz < 1.0:
            return 0.0
        # Bisection on |SA| from 0 to SA_max, comparing abs(Fy)
        sa_hi = float(self._sa_axis[-1])
        sa_lo = 0.1  # avoid exactly 0 where Fy=0
        # Check if target is even reachable
        fy_at_max = abs(float(self.Fy(sa_hi, fz, ia)))
        if target >= fy_at_max:
            # Also check negative SA side
            fy_at_neg = abs(float(self.Fy(-sa_hi, fz, ia)))
            if target >= max(fy_at_max, fy_at_neg):
                # Find SA at peak |Fy|
                best_sa, best_fy = 0.0, 0.0
                for sa_test in np.linspace(0, sa_hi, 30):
                    fy_pos = abs(float(self.Fy(sa_test, fz, ia)))
                    fy_neg = abs(float(self.Fy(-sa_test, fz, ia)))
                    fy_best = max(fy_pos, fy_neg)
                    if fy_best > best_fy:
                        best_fy = fy_best
                        best_sa = sa_test
                return best_sa
        # Determine which sign of SA gives increasing |Fy|
        # TTC convention: usually negative SA → positive Fy (or vice versa)
        fy_pos = abs(float(self.Fy(2.0, fz, ia)))
        fy_neg = abs(float(self.Fy(-2.0, fz, ia)))
        sa_sign = 1.0 if fy_pos >= fy_neg else -1.0
        sa_lo, sa_hi_search = 0.1, sa_hi
        for _ in range(40):
            sa_mid = (sa_lo + sa_hi_search) / 2
            fy_mid = abs(float(self.Fy(sa_sign * sa_mid, fz, ia)))
            if fy_mid < target:
                sa_lo = sa_mid
            else:
                sa_hi_search = sa_mid
            if (sa_hi_search - sa_lo) < 0.01:
                break
        return (sa_lo + sa_hi_search) / 2

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
    Simple tire model for use without TTC data.

    Includes load sensitivity (degressivity) so that cornering stiffness
    does not scale linearly with Fz — heavier loaded tires are less
    efficient, which is what creates understeer/oversteer tendencies.

    C_alpha(Fz) = C_alpha_ref * (Fz / Fz_ref)^load_sensitivity
    where load_sensitivity < 1.0 means degressive (realistic).
    Typical FSAE tire: 0.7-0.85.

    Fy = -C_alpha(Fz) * slip_angle, saturated at mu * Fz.
    """

    def __init__(self, C_alpha_N_per_deg: float = 200.0,
                 Fz_ref_N: float = 700.0,
                 mu: float = 1.5,
                 load_sensitivity: float = 0.8):
        self._Ca = C_alpha_N_per_deg
        self._Fz_ref = Fz_ref_N
        self._mu = mu
        self._ls = load_sensitivity
        self.tire_id = f'Linear (Ca={C_alpha_N_per_deg}, mu={mu}, ls={load_sensitivity})'

    def Fy(self, slip_angle_deg, Fz_N, camber_deg=0.0):
        sa = np.atleast_1d(np.asarray(slip_angle_deg, float))
        fz = np.atleast_1d(np.asarray(Fz_N, float))
        fz_safe = np.maximum(fz, 0.0)
        ca = self._Ca * (fz_safe / self._Fz_ref) ** self._ls
        fy_linear = -ca * sa
        fy_max = self._mu * fz_safe
        return np.clip(fy_linear, -fy_max, fy_max).squeeze()

    def Mz(self, slip_angle_deg, Fz_N, camber_deg=0.0):
        return np.zeros_like(np.atleast_1d(np.asarray(slip_angle_deg, float))).squeeze()

    def peak_Fy(self, Fz_N, camber_deg=0.0):
        return self._mu * np.atleast_1d(np.maximum(np.asarray(Fz_N, float), 0.0)).squeeze()

    def cornering_stiffness(self, Fz_N, camber_deg=0.0):
        fz = np.atleast_1d(np.maximum(np.asarray(Fz_N, float), 0.0))
        return (self._Ca * (fz / self._Fz_ref) ** self._ls).squeeze()

    def peak_mu(self, Fz_N, camber_deg=0.0):
        return self._mu

    def slip_angle_for_Fy(self, Fy_target_N, Fz_N, camber_deg=0.0):
        fz = max(float(Fz_N), 0.0)
        if fz < 1.0:
            return 0.0
        ca = self._Ca * (fz / self._Fz_ref) ** self._ls
        if ca < 0.1:
            return 0.0
        sa = abs(float(Fy_target_N)) / ca
        return min(sa, abs(float(Fy_target_N)) / (self._mu * fz + 1e-9) + 2.0)
