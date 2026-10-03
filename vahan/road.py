"""Band-limited synthetic road inputs; spatial frequency is cycles/metre.

ISO 8608 A/B class-centre spectra are roughness scenarios, not measurements
of any named site. Sources and downloaded reference material are recorded in
DESIGN_2027/binder_run/road_models_research_2026-09-07.md (private design evidence).
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RoadSpectrum:
    road_class: str
    low_cycles_per_m: float = 0.011
    high_cycles_per_m: float = 2.83
    waviness: float = 2.0

    def __post_init__(self):
        if self.road_class not in ('A', 'B'):
            raise ValueError('Only requested classes A and B are supported')
        vals = [self.low_cycles_per_m, self.high_cycles_per_m, self.waviness]
        if not np.all(np.isfinite(vals)) or not 0 < vals[0] < vals[1] or vals[2] <= 0:
            raise ValueError('Require positive finite band limits and waviness')

    def psd(self, n):
        """One-sided displacement PSD in m^3, referenced to 0.1 cycles/m."""
        n = np.asarray(n, dtype=float)
        if np.any(~np.isfinite(n)) or np.any(n < 0):
            raise ValueError('Spatial frequencies must be finite and nonnegative')
        result = np.zeros_like(n)
        mask = (n >= self.low_cycles_per_m) & (n <= self.high_cycles_per_m)
        g0 = {'A': 16e-6, 'B': 64e-6}[self.road_class]
        result[mask] = g0 * (n[mask] / 0.1) ** (-self.waviness)
        return result

    def temporal_psd(self, hz, speed_mps):
        """S_z(f)=G_z(f/v)/v, preserving displacement variance."""
        if not np.isfinite(speed_mps) or speed_mps <= 0:
            raise ValueError('Speed must be positive and finite')
        return self.psd(np.asarray(hz) / speed_mps) / speed_mps


def synthesize_profile(spectrum, *, length_m, spacing_m, seed):
    """Return periodic spatial samples with exact one-sided Fourier-bin power.

    Random phases, no random amplitude; each realization has the specified
    discrete PSD. Excludes Nyquist. Do not concatenate repeats as independent
    roads; use separate seeds/longer records for robustness.
    """
    if not np.all(np.isfinite([length_m, spacing_m])) or min(length_m, spacing_m) <= 0:
        raise ValueError('Length and spacing must be positive and finite')
    count = int(round(length_m / spacing_m))
    if count < 4 or not np.isclose(count * spacing_m, length_m, rtol=0,
                                  atol=16*np.finfo(float).eps*length_m):
        raise ValueError('Length must be an integer multiple of spacing')
    if spectrum.high_cycles_per_m >= 0.5 / spacing_m:
        raise ValueError('Road band must be strictly below spatial Nyquist')
    if 1 / length_m > spectrum.low_cycles_per_m:
        raise ValueError('Record too short to resolve low-frequency band')
    n = np.fft.rfftfreq(count, spacing_m)
    g = spectrum.psd(n)
    phase = np.random.default_rng(seed).uniform(0, 2*np.pi, len(n))
    # irfft contributes 2*abs(X)/N for each positive-frequency cosine.
    coefficients = count * np.sqrt(g / (2 * length_m)) * np.exp(1j * phase)
    coefficients[0] = 0
    if count % 2 == 0:
        coefficients[-1] = 0
    return np.arange(count) * spacing_m, np.fft.irfft(coefficients, n=count)


def delayed_profile(samples, spacing, delay):
    """Periodic band-limited delay: y(x) = z(x - delay), exact in the Fourier
    domain for a periodic record (spacing and delay in the same unit, metres
    for a road in distance or seconds for a time record). Same-path rear
    wheels see the front profile delayed by the wheelbase."""
    z = np.asarray(samples, dtype=float)
    if z.ndim != 1 or len(z) < 4 or not np.all(np.isfinite(z)):
        raise ValueError('finite 1-D record with at least four samples required')
    if not np.all(np.isfinite([spacing, delay])) or spacing <= 0:
        raise ValueError('spacing must be positive; delay finite')
    n = len(z)
    coefficients = np.fft.rfft(z) * np.exp(-2j*np.pi*np.fft.rfftfreq(n, spacing)*delay)
    if n % 2 == 0:
        coefficients[-1] = coefficients[-1].real   # Nyquist stays representable
    return np.fft.irfft(coefficients, n=n)


def corner_cross_psd(spectrum, hz, speed_mps, wheelbase_m, *, track_correlation):
    """FL,FR,RL,RR road cross-PSD (m²/Hz), with rear axle transport delay.

    track_correlation is real cross-spectral correlation, not its square
    (magnitude-squared coherence). 0 and 1 bracket independent/identical
    left/right tracks. They are assumptions, not measured cross-track data.
    Convention S_ij=E[Z_i*conj(Z_j)], rear Z=front Z*exp(-i*omega*L/v).
    """
    if not np.isfinite(wheelbase_m) or wheelbase_m <= 0:
        raise ValueError('Wheelbase must be positive and finite')
    if not np.isfinite(track_correlation) or not 0 <= track_correlation <= 1:
        raise ValueError('Track correlation must lie in [0,1]')
    f = np.atleast_1d(np.asarray(hz, dtype=float))
    density = spectrum.temporal_psd(f, speed_mps)
    delay = np.exp(-2j*np.pi*f*wheelbase_m/speed_mps)
    transport = np.zeros((len(f), 4, 2), dtype=complex)
    transport[:, 0, 0] = transport[:, 1, 1] = 1
    transport[:, 2, 0] = transport[:, 3, 1] = delay
    track = np.array([[1., track_correlation], [track_correlation, 1.]])
    return density[:, None, None] * (transport @ track @ transport.conj().transpose(0, 2, 1))
