from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm

import os, sys

os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads to avoid oversubscription
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads to avoid oversubscription
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Limit NumExpr threads to avoid oversubscription
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS threads to avoid oversubscription

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, skew, pearsonr
from scipy.signal import find_peaks

import maria
from maria.instrument import Band
from maria.spectrum import AtmosphericSpectrum

import simple_ccat

# ============================================================
# User settings
# ============================================================

ccat_band = "280"  # "850" or "350"

Run = True # whether to run the TOD analysis or just load existing TOD

eta = 0.5 # optical efficiency for this estimate

Polarized = False # whether to include polarization in the simulation

# 0.36, 0.67, & 1.28 are Q1, Q2, and Q3 zenith PMV values for Chajnantor

EL_LIMITS = (65, 75)  # degrees

START_TIME = "2022-02-10T17:00:00"

#scan pattern options: "lissajous", "raster", "back_and_forth", "daisy", "double_circle", "stare"

SCAN_PATTERN = "daisy"

ELEV_LABEL = "65-75"

SPEED = 0.1

eta = 0.5

# "2022-02-10T22:45:00" for around 75 degrees ?
#"2022-02-10T20:30:00" for around 60 degrees 
#"2022-02-10T18:55:00" for roughly 45 degrees
#"2022-02-10T18:30:00" for roughly 40 degrees
#"2022-02-10T17:00:00" for roughly 30 degrees


 

TOTAL_DURATION_S = 900  # seconds 15 mins
SIM_DURATION_S = 900  # seconds
CHUNK_NUMBER = 0


# RUN_SPEED_GRID = True
# COMBINE_EXISTING_CSVS = False
# SPEED_CSV_DIR = Path(f"outputs/OrionA_{SCAN_PATTERN}_speed_tests/speed_csv")
# COMBINED_PLOT_DIR = Path(f"outputs/OrionA_{SCAN_PATTERN}_speed_tests/combined_plots")
# SPEED_CSV_DIR.mkdir(parents=True, exist_ok=True)
# COMBINED_PLOT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE_HZ = 10

PWV_MM = 0.67

# Atmospheric power tests
# Run the script once for each of the three PWV values below.
# Each run will save its own results and update the combined PWV plots.
PWV_TEST_VALUES = [0.36, 0.67, 1.28]
ELEVATION_BIN_WIDTH_DEG = 1.0
AIRMass_BIN_WIDTH = 0.01
ATMOSPHERE_TEST_ROOT = Path(
    f"outputs/atmospheric_power_tests/{ccat_band}GHz/{SCAN_PATTERN}"
)
ATMOSPHERE_TEST_ROOT.mkdir(parents=True, exist_ok=True)

if ccat_band == "850":
    NU_HZ = 850e9
    NU_GHZ = NU_HZ / 1e9 #GHz
    bandwidth_hz = 97e9  # GHz bandwidth for 850 GHz band

    BAND_LABEL = "850"
    selected_band = "850" #make sure these match

elif ccat_band == "350":
    NU_HZ = 350e9
    NU_GHZ = NU_HZ / 1e9 #GHz
    bandwidth_hz = 35e9  # GHz bandwidth for 350 GHz band

    BAND_LABEL = "350"
    selected_band = "350" #make sure these match

elif ccat_band == "280":
    NU_HZ = 280e9
    NU_GHZ = NU_HZ / 1e9 #GHz
    bandwidth_hz = 60e9  # GHz bandwidth for 280 GHz band

    BAND_LABEL = "280"
    selected_band = "280" #make sure these match
# # Expected values for 850 GHz band
# Q_r = 15000
# R_0 = -1e7
# P_0 = 120e-12

# DETECTORS_TO_PLOT = [0, 50, 309, 339, 472, 604, 843]
DETECTORS_TO_PLOT = [604] #these are the detectors that are in the center of the array and should be the most stable

pwv_tag = f"{PWV_MM:.2f}".replace(".", "p")

run_prefix = (
    f"OrionA_{SCAN_PATTERN.lower()}_{ELEV_LABEL}_speed_{SPEED:.1f}_"
    f"PWV_{pwv_tag}mm_small_map"
    .replace(".", "p")
)

TOD_OUTDIR = Path(f"outputs/{run_prefix}_tods") #Im going to have to change this for each run I am interested in
fits_path = TOD_OUTDIR / f"{run_prefix}_dim_reduced_tods.fits"

OUTDIR = Path(f"outputs/delta_f_analysis/{SCAN_PATTERN}/900_run_{run_prefix}_{BAND_LABEL}GHz_power_deltaf_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load TOD
# ============================================================

if Run == True:
    simple_ccat.tod_analysis(
        PREFIX=run_prefix,
        tod_diagnostics=False,
        maps=False,
        save_all_plots=True,
        run_mode="fits",
        atm_plot=False,
        temp_mode="inst",
        ccat_band=selected_band,
        map_type="BM",
        pwv_mm=PWV_MM,
        start_time=START_TIME,
        total_duration_s=TOTAL_DURATION_S,
        sim_duration_s=SIM_DURATION_S,
        sample_rate_hz=SAMPLE_RATE_HZ,
        scan_pattern=SCAN_PATTERN,
        el_limits=EL_LIMITS,
        speed=SPEED,
    )
else:
    print(f"Skipping TOD generation; loading existing TOD from: {fits_path}")

site = maria.get_site("cerro_chajnantor", altitude=5600)


if ccat_band == "850":
    band = Band(
        name="m2/f093",
        center=NU_HZ,
        width=bandwidth_hz,
        efficiency=eta,
        NET_CMB=13e-6,
        knee=1.0,
        gain_error=5e-2,
    )

elif ccat_band == "350":
    band = Band(
        name="m2/f093",
        center=NU_HZ,
        width=bandwidth_hz,
        efficiency=eta,
        NET_CMB=48e-6,
        knee=1.0,
        gain_error=5e-2,
    )

elif ccat_band == "280":
    f280 = Band(
        name="m2/f093",
        center=NU_HZ,
        width=bandwidth_hz,
        efficiency= eta,
        NET_CMB=13e-6,
        knee=1.0,
        gain_error=5e-2,
    )
    band = f280

if not fits_path.exists():
    raise FileNotFoundError(f"Missing TOD file: {fits_path}")

tod = maria.tod.load(fits_path, site=site, bands=[band])

print(f"\nLoaded TOD: {fits_path}")
print(f"TOD shape: {tod.shape}")


# ============================================================
# Load detector power
# ============================================================

P_pW = tod.to("pW").signal
P_pW = np.asarray(P_pW, dtype=np.float64)

# ============================================================
# Prepare detector-resolved Az/El matrices for animations
# ============================================================

az_raw = np.asarray(tod.az, dtype=np.float64)
el_raw = np.asarray(tod.el, dtype=np.float64)

print("Raw tod.az shape:", az_raw.shape)
print("Raw tod.el shape:", el_raw.shape)
print("Power shape:", P_pW.shape)

# Remove only singleton dimensions, if any.
az_raw = np.squeeze(az_raw)
el_raw = np.squeeze(el_raw)

print("Squeezed tod.az shape:", az_raw.shape)
print("Squeezed tod.el shape:", el_raw.shape)

# ------------------------------------------------------------
# Match the expected detector-by-time orientation
# ------------------------------------------------------------

if az_raw.shape == P_pW.shape:
    az_matrix = az_raw

elif az_raw.T.shape == P_pW.shape:
    az_matrix = az_raw.T

else:
    raise ValueError(
        "Could not match tod.az to detector power shape. "
        f"tod.az shape after squeeze: {az_raw.shape}; "
        f"P_pW shape: {P_pW.shape}"
    )


if el_raw.shape == P_pW.shape:
    el_matrix = el_raw

elif el_raw.T.shape == P_pW.shape:
    el_matrix = el_raw.T

else:
    raise ValueError(
        "Could not match tod.el to detector power shape. "
        f"tod.el shape after squeeze: {el_raw.shape}; "
        f"P_pW shape: {P_pW.shape}"
    )

# ------------------------------------------------------------
# MARIA Az/El coordinates are normally stored in radians.
# Convert to degrees.
# ------------------------------------------------------------

az_deg_matrix_raw = np.rad2deg(az_matrix)
el_deg_matrix = np.rad2deg(el_matrix)

az_deg_matrix = np.mod(az_deg_matrix_raw, 360.0)

time_sec_full = (
    np.arange(P_pW.shape[1], dtype=np.float64)
    / SAMPLE_RATE_HZ
)

print("Animation Az shape:", az_deg_matrix.shape)
print("Animation El shape:", el_deg_matrix.shape)
print("Animation power shape:", P_pW.shape)
print("Animation time shape:", time_sec_full.shape)
print(
    "Az range:",
    np.nanmin(az_deg_matrix),
    np.nanmax(az_deg_matrix),
)
print(
    "El range:",
    np.nanmin(el_deg_matrix),
    np.nanmax(el_deg_matrix),
)


# ============================================================
# Responsivity model
# ============================================================
if ccat_band == "850":
    Q_r = 15000
    R_0 = -1e7
    P_0 = 120e-12
elif ccat_band == "350":
    Q_r = 40000 # Quality factor taken from Bayguchi thesis
    P_0 = 957e-18 # idk but do not question the mighty jordan wheeler
    R_0 = -2.448e9 #avg responsivity in W^-1 from Jordan Wheeler

elif ccat_band == "280":
    Q_r = 40000 # Quality factor taken from Bayguchi thesis
    P_0 = 957e-18 # idk but do not question the mighty jordan wheeler
    R_0 = -2.448e9 #avg responsivity in W^-1 from Jordan Wheeler

def R(P_W):
    return R_0 / np.sqrt(1 + P_W / P_0)

print("Power shape:", P_pW.shape)
print(f"Using {BAND_LABEL} GHz constants:")
print(f"Q_r = {Q_r:.6g}")
print(f"R_0 = {R_0:.6g} W^-1")
print(f"P_0 = {P_0:.6g} W")

def delta_f_over_fwhm(P_track_pW):
    """
    Calculate Delta f / FWHM relative to a fixed readout tone
    selected at the median detector loading.
    """
    P_track_pW = np.asarray(
        P_track_pW,
        dtype=np.float64,
    )

    valid = np.isfinite(P_track_pW)

    P_track_pW = P_track_pW[valid]
    P_track_W = P_track_pW * 1e-12

    if P_track_W.size == 0:
        return np.array([], dtype=np.float64)

    # Fixed-tone operating point.
    P_ref_W = np.nanmedian(P_track_W)

    # Responsivity evaluated at the fixed operating point.
    R_ref = R(P_ref_W)

    delta_P_W = P_track_W - P_ref_W

    return Q_r * R_ref * delta_P_W


# ============================================================
# Helpers
# ============================================================

def summarize(name, values, units):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    stats = {
        "quantity": name,
        "units": units,
        "mean": np.nanmean(values),
        "std": np.nanstd(values),
        "min": np.nanmin(values),
        "max": np.nanmax(values),
        "ptp": np.nanmax(values) - np.nanmin(values),
        "median": np.nanmedian(values),
        "p05": np.nanpercentile(values, 5),
        "p16": np.nanpercentile(values, 16),
        "p84": np.nanpercentile(values, 84),
        "p95": np.nanpercentile(values, 95),
        "skewness": skew(values),
    }

    print(f"\n{name} [{units}]")
    print("-" * 50)
    print(f"Mean:     {stats['mean']:.6g}")
    print(f"Std:      {stats['std']:.6g}")
    print(f"Median:   {stats['median']:.6g}")
    print(f"Min/max:  {stats['min']:.6g}, {stats['max']:.6g}")
    print(f"PTP:      {stats['ptp']:.6g}")
    print(f"5/95%:    {stats['p05']:.6g}, {stats['p95']:.6g}")
    print(f"16/84%:   {stats['p16']:.6g}, {stats['p84']:.6g}")
    print(f"Skewness: {stats['skewness']:.6g}")

    return stats


def make_hist(values, xlabel, title, filename, bins=50, density=False):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    plt.figure(figsize=(8, 6))

    plt.hist(
        values,
        bins=bins,
        edgecolor="black",
        alpha=0.8,
        density=density,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Probability density" if density else "Number of samples")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def standardize(values):
    """Return finite values standardized to zero mean and unit variance."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    sigma = np.nanstd(values)
    if len(values) == 0 or not np.isfinite(sigma) or sigma == 0:
        return np.array([], dtype=np.float64)

    return (values - np.nanmean(values)) / sigma


def contiguous_event_starts(mask):
    """Return indices at which contiguous True regions begin."""
    mask = np.asarray(mask, dtype=bool)
    selected = np.flatnonzero(mask)

    if len(selected) == 0:
        return np.array([], dtype=int)

    return selected[np.r_[True, np.diff(selected) > 1]]


def candidate_spike_bins(values, bins=100, max_candidates=3):
    """Find narrow histogram peaks above a smoothed local background.

    These are diagnostic candidates, not formal statistically significant peaks.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < 10:
        return np.array([], dtype=int), np.array([]), np.array([]), np.array([])

    counts, edges = np.histogram(values, bins=bins)

    # A short moving-average baseline highlights bins that are narrow relative
    # to the surrounding distribution.
    kernel = np.ones(7, dtype=np.float64) / 7.0
    baseline = np.convolve(counts.astype(float), kernel, mode="same")
    excess = counts - baseline

    # Require both positive excess and modest prominence to avoid returning
    # arbitrary bins from a smooth tail.
    min_prominence = max(2.0, 0.02 * np.nanmax(counts))
    peaks, properties = find_peaks(excess, prominence=min_prominence)

    if len(peaks) == 0:
        return np.array([], dtype=int), counts, edges, excess

    order = np.argsort(properties["prominences"])[::-1]
    peaks = peaks[order[:max_candidates]]

    return peaks, counts, edges, excess

def robust_sigma_mad(values):
    """estimate of std using MAD"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan
    
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))

    return 1.4826 * mad  # Scale factor for Gaussian distribution


def rolling_median(values, window_samples):
    """Centered rolling-median smoothing that preserves array length"""
    values = np.asarray(values, dtype=np.float64)

    window_samples = max (1, int(window_samples))

    if window_samples % 2 == 0:
        window_samples += 1  # Ensure odd window size for centering

    return (
        pd.Series(values).rolling(window=window_samples, center = True, min_periods = 1).median().to_numpy(dtype = np.float64)
    )

def remove_short_true_regions(mask, min_samples):
    """Remove contiguous True regions shorter than min_samples."""
    mask = np.asarray(mask, dtype=bool)
    cleaned = np.zeros_like(mask, dtype=bool)

    if len(mask) == 0:
        return cleaned
    
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(int))

    starts = np.flatnonzero(changes ==1)
    stops = np.flatnonzero(changes == -1)
    
    for start, stop in zip(starts, stops):
        if stop - start >= min_samples:
            cleaned[start:stop] = True
    return cleaned

def mask_regions(mask):
    """Return inclusive start and stop indicies of contiguous True regions in a boolean mask."""

    mask = np.asarray(mask, dtype=bool)

    if len(mask) == 0:
        return []
    
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(int))

    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    return [ 
        (start, stop-1)
        for start, stop in zip(starts, stops)
    ]


def analyse_frequency_plateaus(
    time_sec,
    frequency_track,
    sample_rate_hz,
    smooth_window_s=0.7,
    min_plateau_duration_s=0.5,
    plateau_mad_factor=0.75,
    spike_mad_factor=6.0,
):
    """
    Smooth a frequency-shift track, calculate its time derivative,
    and identify sustained low-derivative intervals.

    Parameters
    ----------
    time_sec : array
        Time coordinate in seconds.

    frequency_track : array
        Frequency-shift track. In this script this is Delta f / FWHM.

    sample_rate_hz : float
        TOD sample rate.

    smooth_window_s : float
        Width of rolling-median smoothing window in seconds.

    min_plateau_duration_s : float
        Minimum duration required for a low-derivative region to be
        considered a plateau.

    plateau_mad_factor : float
        Controls the low-derivative plateau threshold.

    spike_mad_factor : float
        Controls the high-derivative transition threshold.
    """
    time_sec = np.asarray(time_sec, dtype=np.float64)
    frequency_track = np.asarray(
        frequency_track,
        dtype=np.float64,
    )

    if len(time_sec) != len(frequency_track):
        raise ValueError(
            "time_sec and frequency_track must have the same length."
        )

    finite = (
        np.isfinite(time_sec)
        & np.isfinite(frequency_track)
    )

    time_sec = time_sec[finite]
    frequency_track = frequency_track[finite]

    if len(time_sec) < 3:
        raise ValueError(
            "At least three finite samples are required."
        )

    # Convert smoothing duration to an odd number of samples.
    smooth_samples = max(
        1,
        int(round(smooth_window_s * sample_rate_hz)),
    )

    frequency_smoothed = rolling_median(
        frequency_track,
        smooth_samples,
    )

    # Units are FWHM per second because frequency_track is Delta f/FWHM.
    df_dt = np.gradient(
        frequency_smoothed,
        time_sec,
    )

    abs_df_dt = np.abs(df_dt)

    derivative_baseline = np.nanmedian(abs_df_dt)
    derivative_sigma = robust_sigma_mad(abs_df_dt)

    if (
        not np.isfinite(derivative_sigma)
        or derivative_sigma == 0
    ):
        derivative_sigma = np.nanstd(abs_df_dt)

    if (
        not np.isfinite(derivative_sigma)
        or derivative_sigma == 0
    ):
        derivative_sigma = np.finfo(float).eps

    plateau_threshold = (
        derivative_baseline
        + plateau_mad_factor * derivative_sigma
    )

    raw_plateau_mask = (
        abs_df_dt <= plateau_threshold
    )

    min_plateau_samples = max(
        1,
        int(round(
            min_plateau_duration_s * sample_rate_hz
        )),
    )

    plateau_mask = remove_short_true_regions(
        raw_plateau_mask,
        min_plateau_samples,
    )

    # Separately identify unusually rapid transitions.
    derivative_center = np.nanmedian(df_dt)
    spike_sigma = robust_sigma_mad(df_dt)

    if (
        not np.isfinite(spike_sigma)
        or spike_sigma == 0
    ):
        spike_sigma = np.nanstd(df_dt)

    if (
        not np.isfinite(spike_sigma)
        or spike_sigma == 0
    ):
        spike_sigma = np.finfo(float).eps

    spike_threshold = (
        spike_mad_factor * spike_sigma
    )

    derivative_spike_mask = (
        np.abs(df_dt - derivative_center)
        >= spike_threshold
    )

    plateau_rows = []

    for plateau_number, (start_idx, stop_idx) in enumerate(
        mask_regions(plateau_mask),
        start=1,
    ):
        region = slice(start_idx, stop_idx + 1)

        plateau_rows.append({
            "plateau_number": plateau_number,
            "start_index": start_idx,
            "stop_index": stop_idx,
            "start_time_s": time_sec[start_idx],
            "stop_time_s": time_sec[stop_idx],
            "duration_s": (
                time_sec[stop_idx]
                - time_sec[start_idx]
            ),
            "mean_delta_f_over_fwhm": np.nanmean(
                frequency_smoothed[region]
            ),
            "std_delta_f_over_fwhm": np.nanstd(
                frequency_smoothed[region]
            ),
            "frequency_range": (
                np.nanmax(frequency_smoothed[region])
                - np.nanmin(frequency_smoothed[region])
            ),
            "mean_abs_df_dt": np.nanmean(
                abs_df_dt[region]
            ),
            "max_abs_df_dt": np.nanmax(
                abs_df_dt[region]
            ),
        })

    return {
        "time_sec": time_sec,
        "frequency_raw": frequency_track,
        "frequency_smoothed": frequency_smoothed,
        "df_dt": df_dt,
        "plateau_mask": plateau_mask,
        "derivative_spike_mask": derivative_spike_mask,
        "plateau_threshold": plateau_threshold,
        "spike_threshold": spike_threshold,
        "plateau_rows": plateau_rows,
    }
    
def make_array_feature_diagnostic(
    delta_matrix,
    time_sec,
    lower,
    upper,
    feature_rank,
    outdir,
    run_prefix,
    band_label,
    scan_pattern,
    elev_label,
    smooth_window_s=2.0,
):
    """
    Compare an all-detector histogram feature with the array-common
    frequency track and its time derivative.

    Parameters
    ----------
    delta_matrix : ndarray
        Detector-by-time matrix of Delta f / FWHM.

    time_sec : ndarray
        Time coordinate corresponding to the columns of delta_matrix.

    lower, upper : float
        Lower and upper edges of the candidate histogram feature.

    feature_rank : int
        Candidate-feature number used in plot labels and filenames.

    smooth_window_s : float
        Width of the rolling-median smoothing window.
    """
    delta_matrix = np.asarray(delta_matrix, dtype=np.float64)
    time_sec = np.asarray(time_sec, dtype=np.float64)

    if delta_matrix.ndim != 2:
        raise ValueError("delta_matrix must be detector by time.")

    if delta_matrix.shape[1] != len(time_sec):
        raise ValueError(
            "delta_matrix time axis does not match time_sec."
        )

    # --------------------------------------------------------
    # 1. Array-common frequency-shift track
    # --------------------------------------------------------

    common_delta = np.nanmedian(delta_matrix, axis=0)

    common_valid = (
        np.isfinite(common_delta)
        & np.isfinite(time_sec)
    )

    diagnostic_time = time_sec[common_valid]
    common_delta = common_delta[common_valid]

    if len(common_delta) < 3:
        print(
            f"Feature {feature_rank}: insufficient finite samples."
        )
        return None

    # Rolling-median smoothing.
    dt_median = np.nanmedian(np.diff(diagnostic_time))

    if not np.isfinite(dt_median) or dt_median <= 0:
        raise ValueError("Could not determine a valid time spacing.")

    smooth_samples = max(
        1,
        int(round(smooth_window_s / dt_median)),
    )

    if smooth_samples % 2 == 0:
        smooth_samples += 1

    common_smoothed = (
        pd.Series(common_delta)
        .rolling(
            window=smooth_samples,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy(dtype=np.float64)
    )

    # --------------------------------------------------------
    # 2. Time derivative
    # --------------------------------------------------------

    common_df_dt = np.gradient(
        common_smoothed,
        diagnostic_time,
    )

    abs_df_dt = np.abs(common_df_dt)

    # --------------------------------------------------------
    # 3. Fraction of valid detectors in the feature at each time
    # --------------------------------------------------------

    full_valid = np.isfinite(delta_matrix)

    in_feature = (
        full_valid
        & (delta_matrix >= lower)
        & (delta_matrix < upper)
    )

    n_valid_detectors = np.sum(full_valid, axis=0)
    n_feature_detectors = np.sum(in_feature, axis=0)

    feature_occupancy = np.divide(
        n_feature_detectors,
        n_valid_detectors,
        out=np.full(
            delta_matrix.shape[1],
            np.nan,
            dtype=np.float64,
        ),
        where=n_valid_detectors > 0,
    )

    feature_occupancy = feature_occupancy[common_valid]

    # --------------------------------------------------------
    # 4. Define high-occupancy periods for visual comparison
    # --------------------------------------------------------

    peak_occupancy = np.nanmax(feature_occupancy)

    if np.isfinite(peak_occupancy) and peak_occupancy > 0:
        high_occupancy_threshold = 0.5 * peak_occupancy

        high_occupancy_mask = (
            feature_occupancy >= high_occupancy_threshold
        )
    else:
        high_occupancy_threshold = np.nan
        high_occupancy_mask = np.zeros(
            len(feature_occupancy),
            dtype=bool,
        )

    # --------------------------------------------------------
    # 5. Numerical comparison
    # --------------------------------------------------------

    baseline_median_abs_df_dt = np.nanmedian(abs_df_dt)

    if np.any(high_occupancy_mask):
        feature_median_abs_df_dt = np.nanmedian(
            abs_df_dt[high_occupancy_mask]
        )

        feature_mean_abs_df_dt = np.nanmean(
            abs_df_dt[high_occupancy_mask]
        )

        derivative_ratio = (
            feature_median_abs_df_dt
            / baseline_median_abs_df_dt
            if baseline_median_abs_df_dt > 0
            else np.nan
        )

        high_occupancy_duration_s = np.sum(
            np.diff(
                np.r_[
                    diagnostic_time,
                    diagnostic_time[-1] + dt_median,
                ]
            )[high_occupancy_mask]
        )
    else:
        feature_median_abs_df_dt = np.nan
        feature_mean_abs_df_dt = np.nan
        derivative_ratio = np.nan
        high_occupancy_duration_s = 0.0

    print(
        f"\nArray feature {feature_rank}: "
        f"[{lower:.6g}, {upper:.6g})"
    )
    print(
        f"  Peak detector occupancy: "
        f"{peak_occupancy:.2%}"
    )
    print(
        f"  High-occupancy threshold: "
        f"{high_occupancy_threshold:.2%}"
    )
    print(
        f"  High-occupancy duration: "
        f"{high_occupancy_duration_s:.2f} s"
    )
    print(
        f"  Median |df/dt| overall: "
        f"{baseline_median_abs_df_dt:.6g} FWHM/s"
    )
    print(
        f"  Median |df/dt| during high occupancy: "
        f"{feature_median_abs_df_dt:.6g} FWHM/s"
    )
    print(
        f"  High-occupancy / overall derivative ratio: "
        f"{derivative_ratio:.4g}"
    )

    # --------------------------------------------------------
    # 6. Three-panel diagnostic plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [2.0, 1.2, 1.2],
        },
    )

    frequency_axis = axes[0]
    derivative_axis = axes[1]
    occupancy_axis = axes[2]

    # Frequency track
    frequency_axis.plot(
        diagnostic_time,
        common_delta,
        linewidth=0.5,
        alpha=0.45,
        label="Raw array median",
    )

    frequency_axis.plot(
        diagnostic_time,
        common_smoothed,
        linewidth=1.2,
        label="Smoothed array median",
    )

    frequency_axis.axhspan(
        lower,
        upper,
        alpha=0.2,
        label="Candidate histogram feature",
    )

    if np.any(high_occupancy_mask):
        frequency_axis.fill_between(
            diagnostic_time,
            0,
            1,
            where=high_occupancy_mask,
            alpha=0.08,
            transform=frequency_axis.get_xaxis_transform(),
            label="High array occupancy",
        )

    frequency_axis.set_ylabel(
        r"Array-median $\Delta f/\mathrm{FWHM}$"
    )
    frequency_axis.grid(alpha=0.3)
    frequency_axis.legend(loc="best")


    # Absolute derivative
    derivative_axis.plot(
        diagnostic_time,
        abs_df_dt,
        linewidth=0.7,
        label=r"$|d(\Delta f/\mathrm{FWHM})/dt|$",
    )

    if np.any(high_occupancy_mask):
        derivative_axis.fill_between(
            diagnostic_time,
            0,
            1,
            where=high_occupancy_mask,
            alpha=0.08,
            transform=derivative_axis.get_xaxis_transform(),
        )

        derivative_axis.scatter(
            diagnostic_time[high_occupancy_mask],
            abs_df_dt[high_occupancy_mask],
            s=8,
            label="High feature occupancy",
            zorder=4,
        )

    derivative_axis.set_ylabel(
        r"$|d(\Delta f/\mathrm{FWHM})/dt|$"
        "\n"
        r"$(\mathrm{s}^{-1})$"
    )
    derivative_axis.grid(alpha=0.3)
    derivative_axis.legend(loc="best")


    # Detector occupancy
    occupancy_axis.plot(
        diagnostic_time,
        feature_occupancy,
        linewidth=0.9,
    )

    if np.isfinite(high_occupancy_threshold):
        occupancy_axis.axhline(
            high_occupancy_threshold,
            linestyle="--",
            linewidth=1,
            label="50% of peak occupancy",
        )

    occupancy_axis.set_xlabel("Time (s)")
    occupancy_axis.set_ylabel(
        "Fraction of detectors\nin feature"
    )
    occupancy_axis.grid(alpha=0.3)
    occupancy_axis.legend(loc="best")

# ============================================================

    plt.tight_layout()

    output_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        f"array_feature_{feature_rank}_"
        "frequency_derivative_occupancy.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved diagnostic: {output_path}")

    return {
        "feature_rank": feature_rank,
        "feature_lower": lower,
        "feature_upper": upper,
        "peak_detector_occupancy": peak_occupancy,
        "high_occupancy_threshold": high_occupancy_threshold,
        "high_occupancy_duration_s": high_occupancy_duration_s,
        "overall_median_abs_df_dt": baseline_median_abs_df_dt,
        "feature_median_abs_df_dt": feature_median_abs_df_dt,
        "feature_mean_abs_df_dt": feature_mean_abs_df_dt,
        "derivative_ratio": derivative_ratio,
    }


def make_feature_geometry_diagnostic(
    tod,
    delta_matrix,
    time_sec,
    lower,
    upper,
    feature_rank,
    outdir,
    run_prefix,
    band_label,
    scan_pattern,
    elev_label,
    occupancy_fraction_of_peak=0.5,
):
    """
    Test whether a pooled Delta f/FWHM histogram feature corresponds
    to a preferred telescope position or observing geometry.

    The function:
      1. computes the fraction of detectors in the feature at each time,
      2. marks times with occupancy above a fraction of the peak,
      3. compares those times with Az/El, RA/Dec, elevation, and airmass.

    Parameters
    ----------
    tod
        Loaded MARIA TOD object.

    delta_matrix : ndarray
        Detector-by-time Delta f/FWHM matrix.

    time_sec : ndarray
        Time coordinate corresponding to the columns of delta_matrix.

    lower, upper : float
        Candidate feature bounds in Delta f/FWHM.

    feature_rank : int
        Feature number used in titles and filenames.

    occupancy_fraction_of_peak : float
        High-occupancy threshold expressed as a fraction of the peak
        detector occupancy.
    """
    delta_matrix = np.asarray(delta_matrix, dtype=np.float64)
    time_sec = np.asarray(time_sec, dtype=np.float64)

    if delta_matrix.ndim != 2:
        raise ValueError(
            "delta_matrix must have shape detector-by-time."
        )

    if delta_matrix.shape[1] != len(time_sec):
        raise ValueError(
            "delta_matrix time axis does not match time_sec."
        )

    # ========================================================
    # Detector occupancy in this Delta f/FWHM feature
    # ========================================================

    valid_delta = np.isfinite(delta_matrix)

    in_feature = (
        valid_delta
        & (delta_matrix >= lower)
        & (delta_matrix < upper)
    )

    n_valid = np.sum(valid_delta, axis=0)
    n_in_feature = np.sum(in_feature, axis=0)

    occupancy = np.divide(
        n_in_feature,
        n_valid,
        out=np.full(
            delta_matrix.shape[1],
            np.nan,
            dtype=np.float64,
        ),
        where=n_valid > 0,
    )

    peak_occupancy = np.nanmax(occupancy)

    if not np.isfinite(peak_occupancy) or peak_occupancy <= 0:
        print(
            f"Feature {feature_rank}: no finite detector occupancy."
        )
        return None

    occupancy_threshold = (
        occupancy_fraction_of_peak * peak_occupancy
    )

    high_occupancy_mask = (
        np.isfinite(occupancy)
        & (occupancy >= occupancy_threshold)
    )

    if not np.any(high_occupancy_mask):
        print(
            f"Feature {feature_rank}: no high-occupancy samples."
        )
        return None

    # ========================================================
    # Helper to collapse a TOD coordinate to one value per time
    # ========================================================

    def common_coordinate(values, coordinate_name):
        """
        Return one representative coordinate per time sample.

        MARIA coordinates may be:
          - one-dimensional: time
          - two-dimensional: detector by time
        """
        values = np.asarray(values, dtype=np.float64)

        if values.ndim == 1:
            common = values

        elif values.ndim == 2:
            if values.shape[1] == len(time_sec):
                common = np.nanmedian(values, axis=0)

            elif values.shape[0] == len(time_sec):
                common = np.nanmedian(values, axis=1)

            else:
                raise ValueError(
                    f"Could not match {coordinate_name} shape "
                    f"{values.shape} to {len(time_sec)} time samples."
                )

        else:
            raise ValueError(
                f"{coordinate_name} has unsupported shape "
                f"{values.shape}."
            )

        if len(common) != len(time_sec):
            raise ValueError(
                f"{coordinate_name} has {len(common)} samples, "
                f"but time_sec has {len(time_sec)}."
            )

        return common

    # ========================================================
    # Extract representative telescope coordinates
    # ========================================================

    az = common_coordinate(tod.az, "azimuth")
    el = common_coordinate(tod.el, "elevation")
    ra = common_coordinate(tod.ra, "right ascension")
    dec = common_coordinate(tod.dec, "declination")

    # MARIA normally stores angular coordinates in radians.
    # Convert only when values appear to be in radians.
    def angular_values_to_degrees(values, coordinate_name):
        values = np.asarray(values, dtype=np.float64)
        finite_values = values[np.isfinite(values)]

        if len(finite_values) == 0:
            return values

        max_abs = np.nanmax(np.abs(finite_values))

        if coordinate_name in {"azimuth", "right ascension"}:
            probably_radians = max_abs <= 2 * np.pi + 0.1
        else:
            probably_radians = max_abs <= np.pi + 0.1

        if probably_radians:
            return np.rad2deg(values)

        return values

    az_deg = angular_values_to_degrees(
        az,
        "azimuth",
    )

    el_deg = angular_values_to_degrees(
        el,
        "elevation",
    )

    ra_deg = angular_values_to_degrees(
        ra,
        "right ascension",
    )

    dec_deg = angular_values_to_degrees(
        dec,
        "declination",
    )

    # Put azimuth into 0–360 degrees for easier interpretation.
    az_deg = np.mod(az_deg, 360.0)

    # Approximate plane-parallel airmass.
    # This is sufficient for comparing 65–75 degree observations.
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        airmass = 1.0 / np.sin(
            np.deg2rad(el_deg)
        )

    airmass[
        ~np.isfinite(airmass)
        | (el_deg <= 0)
    ] = np.nan

    # Require finite geometry and occupancy values.
    geometry_valid = (
        np.isfinite(time_sec)
        & np.isfinite(occupancy)
        & np.isfinite(az_deg)
        & np.isfinite(el_deg)
        & np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(airmass)
    )

    high_geometry_mask = (
        high_occupancy_mask
        & geometry_valid
    )

    if not np.any(high_geometry_mask):
        print(
            f"Feature {feature_rank}: high-occupancy samples "
            "have no valid geometry."
        )
        return None

    # ========================================================
    # Numerical summaries
    # ========================================================

    selected_time = time_sec[high_geometry_mask]
    selected_az = az_deg[high_geometry_mask]
    selected_el = el_deg[high_geometry_mask]
    selected_ra = ra_deg[high_geometry_mask]
    selected_dec = dec_deg[high_geometry_mask]
    selected_airmass = airmass[high_geometry_mask]
    selected_occupancy = occupancy[high_geometry_mask]

    summary = {
        "feature_rank": feature_rank,
        "feature_lower": lower,
        "feature_upper": upper,
        "peak_detector_occupancy": peak_occupancy,
        "occupancy_threshold": occupancy_threshold,
        "n_high_occupancy_samples": int(
            np.sum(high_geometry_mask)
        ),
        "mean_azimuth_deg": np.nanmean(selected_az),
        "std_azimuth_deg": np.nanstd(selected_az),
        "azimuth_range_deg": (
            np.nanmax(selected_az)
            - np.nanmin(selected_az)
        ),
        "mean_elevation_deg": np.nanmean(selected_el),
        "std_elevation_deg": np.nanstd(selected_el),
        "elevation_range_deg": (
            np.nanmax(selected_el)
            - np.nanmin(selected_el)
        ),
        "mean_ra_deg": np.nanmean(selected_ra),
        "std_ra_deg": np.nanstd(selected_ra),
        "mean_dec_deg": np.nanmean(selected_dec),
        "std_dec_deg": np.nanstd(selected_dec),
        "mean_airmass": np.nanmean(selected_airmass),
        "std_airmass": np.nanstd(selected_airmass),
        "airmass_range": (
            np.nanmax(selected_airmass)
            - np.nanmin(selected_airmass)
        ),
    }

    print(
        f"\nFeature {feature_rank} geometry:"
    )
    print(
        f"  Delta f/FWHM range: "
        f"[{lower:.6g}, {upper:.6g})"
    )
    print(
        f"  Peak occupancy: "
        f"{peak_occupancy:.2%}"
    )
    print(
        f"  Selected occupancy threshold: "
        f"{occupancy_threshold:.2%}"
    )
    print(
        f"  High-occupancy samples: "
        f"{summary['n_high_occupancy_samples']}"
    )
    print(
        f"  Elevation: "
        f"{summary['mean_elevation_deg']:.4f} "
        f"+/- {summary['std_elevation_deg']:.4f} deg"
    )
    print(
        f"  Elevation range: "
        f"{summary['elevation_range_deg']:.4f} deg"
    )
    print(
        f"  Azimuth range: "
        f"{summary['azimuth_range_deg']:.4f} deg"
    )
    print(
        f"  Airmass: "
        f"{summary['mean_airmass']:.6f} "
        f"+/- {summary['std_airmass']:.6f}"
    )

    # ========================================================
    # Plot 1: Azimuth–elevation geometry
    # ========================================================

    fig, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.plot(
        az_deg[geometry_valid],
        el_deg[geometry_valid],
        linewidth=0.5,
        alpha=0.35,
        label="Full scan path",
        zorder=1,
    )

    scatter = axis.scatter(
        selected_az,
        selected_el,
        c=selected_occupancy,
        s=20,
        zorder=3,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=axis,
    )

    colorbar.set_label(
        "Fraction of detectors in feature"
    )

    axis.set_xlabel("Array-median azimuth (deg)")
    axis.set_ylabel("Array-median elevation (deg)")

    axis.set_title(
        f"Feature {feature_rank}: Azimuth–Elevation Locations\n"
        f"{band_label} GHz, {scan_pattern}, Elev={elev_label}, "
        f"Feature={lower:.4g} to {upper:.4g}"
    )

    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()

    azel_output = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        f"array_feature_{feature_rank}_"
        "azimuth_elevation_geometry.png"
    )

    fig.savefig(
        azel_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Plot 2: RA–Dec geometry
    # ========================================================

    fig, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.plot(
        ra_deg[geometry_valid],
        dec_deg[geometry_valid],
        linewidth=0.5,
        alpha=0.35,
        label="Full scan path",
        zorder=1,
    )

    scatter = axis.scatter(
        selected_ra,
        selected_dec,
        c=selected_occupancy,
        s=20,
        zorder=3,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=axis,
    )

    colorbar.set_label(
        "Fraction of detectors in feature"
    )

    axis.set_xlabel("Array-median right ascension (deg)")
    axis.set_ylabel("Array-median declination (deg)")

    axis.set_title(
        f"Feature {feature_rank}: RA–Dec Locations\n"
        f"{band_label} GHz, {scan_pattern}, Elev={elev_label}"
    )

    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()

    radec_output = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        f"array_feature_{feature_rank}_"
        "radec_geometry.png"
    )

    fig.savefig(
        radec_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Plot 3: time, elevation, airmass, and occupancy
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
    )

    elevation_axis = axes[0]
    airmass_axis = axes[1]
    occupancy_axis = axes[2]

    # Elevation
    elevation_axis.plot(
        time_sec[geometry_valid],
        el_deg[geometry_valid],
        linewidth=0.8,
        label="Array-median elevation",
        zorder=2,
    )

    elevation_axis.scatter(
        selected_time,
        selected_el,
        c=selected_occupancy,
        s=15,
        label="High feature occupancy",
        zorder=3,
    )

    elevation_axis.set_ylabel("Elevation (deg)")
    elevation_axis.grid(alpha=0.3)
    elevation_axis.legend(loc="best")

    # Airmass
    airmass_axis.plot(
        time_sec[geometry_valid],
        airmass[geometry_valid],
        linewidth=0.8,
        label="Approximate airmass",
        zorder=2,
    )

    airmass_axis.scatter(
        selected_time,
        selected_airmass,
        c=selected_occupancy,
        s=15,
        label="High feature occupancy",
        zorder=3,
    )

    airmass_axis.set_ylabel("Airmass")
    airmass_axis.grid(alpha=0.3)
    airmass_axis.legend(loc="best")

    # Occupancy
    occupancy_axis.plot(
        time_sec,
        occupancy,
        linewidth=0.8,
        label="Detector occupancy",
    )

    occupancy_axis.axhline(
        occupancy_threshold,
        linestyle="--",
        linewidth=1,
        label=(
            f"{occupancy_fraction_of_peak:.0%} "
            "of peak occupancy"
        ),
    )

    occupancy_axis.set_xlabel("Time (s)")
    occupancy_axis.set_ylabel(
        "Fraction of detectors\nin feature"
    )

    occupancy_axis.grid(alpha=0.3)
    occupancy_axis.legend(loc="best")

    fig.suptitle(
        f"Feature {feature_rank}: Geometry versus Time\n"
        f"{band_label} GHz, {scan_pattern}, Elev={elev_label}, "
        f"Feature={lower:.4g} to {upper:.4g}"
    )

    fig.tight_layout()

    time_geometry_output = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        f"array_feature_{feature_rank}_"
        "elevation_airmass_occupancy_vs_time.png"
    )

    fig.savefig(
        time_geometry_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Save selected geometry samples
    # ========================================================

    selected_geometry_df = pd.DataFrame({
        "time_s": selected_time,
        "feature_occupancy": selected_occupancy,
        "azimuth_deg": selected_az,
        "elevation_deg": selected_el,
        "ra_deg": selected_ra,
        "dec_deg": selected_dec,
        "airmass": selected_airmass,
    })

    selected_geometry_csv = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        f"array_feature_{feature_rank}_"
        "high_occupancy_geometry_samples.csv"
    )

    selected_geometry_df.to_csv(
        selected_geometry_csv,
        index=False,
    )

    print(f"  Saved Az/El plot: {azel_output}")
    print(f"  Saved RA/Dec plot: {radec_output}")
    print(
        f"  Saved geometry time plot: "
        f"{time_geometry_output}"
    )
    print(
        f"  Saved selected samples: "
        f"{selected_geometry_csv}"
    )

    return summary


def animate_detector_azel_power(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    power_pW: np.ndarray,
    time_s: np.ndarray,
    output_path: Path,
    *,
    colour_mode: str = "absolute",
    frame_step: int = 10,
    fps: int = 20,
    marker_size: float = 14.0,
    title: str = "Detector Power in Azimuth–Elevation",
    fixed_az_limits: Optional[tuple[float, float]] = None,
    fixed_el_limits: Optional[tuple[float, float]] = None,
) -> None:
    """
    Animate detector locations in azimuth/elevation, coloured by power.

    Expected input shape:
        az_deg:   (n_detectors, n_times)
        el_deg:   (n_detectors, n_times)
        power_pW: (n_detectors, n_times)
        time_s:   (n_times,)

    Parameters
    ----------
    colour_mode
        The mode of colouring the detectors:
        - "absolute": Use the absolute detector power.
        - "detector_median_subtracted": Subtract the median power of each detector.
        - "frame_median_subtracted": Subtract the median power of each frame.
    frame_step
        Use every `frame_step`-th time sample as an animation frame.
    fps
        Frames per second in the saved animation.
    fixed_az_limits, fixed_el_limits
        Optional fixed plotting limits. If omitted, limits are computed
        from the entire observation and remain fixed for every frame.
    """

    az_deg = np.asarray(az_deg, dtype=float)
    el_deg = np.asarray(el_deg, dtype=float)
    power_pW = np.asarray(power_pW, dtype=float)
    time_s = np.asarray(time_s, dtype=float)

    if az_deg.ndim != 2 or el_deg.ndim != 2 or power_pW.ndim != 2:
        raise ValueError(
            "az_deg, el_deg, and power_pW must all be 2D arrays with "
            "shape (n_detectors, n_times)."
        )

    if az_deg.shape != el_deg.shape or az_deg.shape != power_pW.shape:
        raise ValueError(
            "az_deg, el_deg, and power_pW must have identical shapes. "
            f"Received {az_deg.shape}, {el_deg.shape}, and {power_pW.shape}."
        )

    n_detectors, n_times = power_pW.shape

    if time_s.ndim != 1 or len(time_s) != n_times:
        raise ValueError(
            f"time_s must have shape ({n_times},), but has {time_s.shape}."
        )

    if frame_step < 1:
        raise ValueError("frame_step must be at least 1.")

    # ------------------------------------------------------------
    # Select the quantity represented by colour
    # ------------------------------------------------------------

    finite_position = np.isfinite(az_deg) & np.isfinite(el_deg)

    valid_colour_modes = {
        "absolute",
        "detector_median_subtracted",
        "frame_median_subtracted",
    }

    if colour_mode not in valid_colour_modes:
        raise ValueError(
            f"colour_mode must be one of {sorted(valid_colour_modes)}. "
            f"Received {colour_mode!r}."
        )


    if colour_mode == "absolute":
        # Original detector power.
        colour_values = power_pW
        colour_label = "Detector power (pW)"
        mode_label = "absolute power"

        finite_colour = colour_values[np.isfinite(colour_values)]

        if finite_colour.size == 0:
            raise ValueError("No finite detector power values found.")

        colour_low, colour_high = np.nanpercentile(
            finite_colour,
            [1.0, 99.0],
        )

        if colour_high <= colour_low:
            colour_low = np.nanmin(finite_colour)
            colour_high = np.nanmax(finite_colour)

        norm = Normalize(
            vmin=colour_low,
            vmax=colour_high,
        )
        cmap = "viridis"


    elif colour_mode == "detector_median_subtracted":
        # Remove the time median of each detector:
        #
        # P_i(t) - median_t[P_i(t)]
        detector_medians = np.nanmedian(
            power_pW,
            axis=1,
            keepdims=True,
        )

        colour_values = power_pW - detector_medians

        colour_label = (
            "Detector-median-subtracted power (pW)"
        )
        mode_label = "detector-median-subtracted power"

        finite_colour = colour_values[np.isfinite(colour_values)]

        if finite_colour.size == 0:
            raise ValueError(
                "No finite detector-median-subtracted values found."
            )

        colour_limit = np.nanpercentile(
            np.abs(finite_colour),
            99.0,
        )

        if not np.isfinite(colour_limit) or colour_limit <= 0:
            colour_limit = np.nanmax(np.abs(finite_colour))

        norm = TwoSlopeNorm(
            vmin=-colour_limit,
            vcenter=0.0,
            vmax=colour_limit,
        )
        cmap = "coolwarm"


    elif colour_mode == "frame_median_subtracted":
        # Step 1: Remove the time median of each detector.
        detector_medians = np.nanmedian(
            power_pW,
            axis=1,
            keepdims=True,
        )

        detector_relative_power = (
            power_pW - detector_medians
        )

        # Step 2: For every time sample, remove the median residual
        # across all detectors.
        #
        # [P_i(t) - median_t(P_i)]
        #       - median_i[P_i(t) - median_t(P_i)]
        frame_medians = np.nanmedian(
            detector_relative_power,
            axis=0,
            keepdims=True,
        )

        colour_values = (
            detector_relative_power - frame_medians
        )

        colour_label = (
            "Detector- and frame-median-subtracted power (pW)"
        )
        mode_label = (
            "detector- and frame-median-subtracted power"
        )

        finite_colour = colour_values[np.isfinite(colour_values)]

        if finite_colour.size == 0:
            raise ValueError(
                "No finite frame-median-subtracted values found."
            )

        # A slightly tighter percentile may be useful here because
        # this mode is intended to expose small residual variations.
        colour_limit = np.nanpercentile(
            np.abs(finite_colour),
            99.0,
        )

        if not np.isfinite(colour_limit) or colour_limit <= 0:
            colour_limit = np.nanmax(np.abs(finite_colour))

        norm = TwoSlopeNorm(
            vmin=-colour_limit,
            vcenter=0.0,
            vmax=colour_limit,
        )
        cmap = "coolwarm"

    if not np.any(finite_position):
        raise ValueError("No finite detector azimuth/elevation positions found.")

    if fixed_az_limits is None:
        az_min = np.nanpercentile(az_deg[finite_position], 0.1)
        az_max = np.nanpercentile(az_deg[finite_position], 99.9)
        az_padding = max(0.02, 0.03 * (az_max - az_min))
        fixed_az_limits = (
            az_min - az_padding,
            az_max + az_padding,
        )

    if fixed_el_limits is None:
        el_min = np.nanpercentile(el_deg[finite_position], 0.1)
        el_max = np.nanpercentile(el_deg[finite_position], 99.9)
        el_padding = max(0.02, 0.03 * (el_max - el_min))
        fixed_el_limits = (
            el_min - el_padding,
            el_max + el_padding,
        )

    frame_indices = np.arange(0, n_times, frame_step)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))

    initial_index = frame_indices[0]
    initial_valid = (
        np.isfinite(az_deg[:, initial_index])
        & np.isfinite(el_deg[:, initial_index])
        & np.isfinite(colour_values[:, initial_index])
    )

    scatter = ax.scatter(
        az_deg[initial_valid, initial_index],
        el_deg[initial_valid, initial_index],
        c=colour_values[initial_valid, initial_index],
        s=marker_size,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )

    ax.set_xlim(*fixed_az_limits)
    ax.set_ylim(*fixed_el_limits)
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.grid(alpha=0.25)

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "none",
        },
    )


    ax.set_title(f"{title}\nColoured by {mode_label}")

    colourbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colourbar.set_label(colour_label)

    def update(frame_number: int):
        sample_index = frame_indices[frame_number]

        valid = (
            np.isfinite(az_deg[:, sample_index])
            & np.isfinite(el_deg[:, sample_index])
            & np.isfinite(colour_values[:, sample_index])
        )

        positions = np.column_stack(
            (
                az_deg[valid, sample_index],
                el_deg[valid, sample_index],
            )
        )

        scatter.set_offsets(positions)
        scatter.set_array(colour_values[valid, sample_index])

        time_text.set_text(
            f"Time: {time_s[sample_index]:.1f} s\n"
            f"Detectors shown: {np.count_nonzero(valid)}/{n_detectors}"
        )

        return scatter, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".mp4":
        try:
            writer = FFMpegWriter(
                fps=fps,
                bitrate=3000,
                metadata={"artist": "Matplotlib"},
            )
            animation.save(output_path, writer=writer, dpi=150)

        except FileNotFoundError as error:
            plt.close(fig)
            raise RuntimeError(
                "FFmpeg was not found. Install FFmpeg or save the "
                "animation with a .gif extension."
            ) from error

    elif suffix == ".gif":
        animation.save(
            output_path,
            writer=PillowWriter(fps=fps),
            dpi=120,
        )

    else:
        plt.close(fig)
        raise ValueError(
            "output_path must end in either '.mp4' or '.gif'."
        )

    plt.close(fig)

    print(f"Saved detector animation to: {output_path}")

# ============================================================
# Atmospheric power test helpers
# ============================================================

def safe_correlation(x, y):
    """Return a Pearson correlation coefficient using finite samples only."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(valid) < 3:
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
        return np.nan

    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def bin_metric(x, y, bin_width):
    """
    Bin y as a function of x and return the median and 16th/84th percentiles.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return pd.DataFrame()

    lower_edge = np.floor(np.nanmin(x) / bin_width) * bin_width
    upper_edge = np.ceil(np.nanmax(x) / bin_width) * bin_width

    edges = np.arange(
        lower_edge,
        upper_edge + bin_width,
        bin_width,
    )

    rows = []

    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (x >= left) & (x < right)

        if np.count_nonzero(in_bin) == 0:
            continue

        values = y[in_bin]

        rows.append({
            "bin_left": left,
            "bin_right": right,
            "bin_centre": 0.5 * (left + right),
            "median": np.nanmedian(values),
            "p16": np.nanpercentile(values, 16),
            "p84": np.nanpercentile(values, 84),
            "n_samples": np.count_nonzero(np.isfinite(values)),
        })

    return pd.DataFrame(rows)


def calculate_atmospheric_power_metrics(
    power_pW,
    elevation_deg_matrix,
):
    """
    Calculate common-mode and small-scale detector power quantities.

    Returns one value per time sample for:
      - raw array-median power,
      - detector-median-subtracted common power,
      - small-scale detector scatter,
      - elevation,
      - airmass.
    """
    power_pW = np.asarray(power_pW, dtype=np.float64)
    elevation_deg_matrix = np.asarray(
        elevation_deg_matrix,
        dtype=np.float64,
    )

    if power_pW.ndim != 2:
        raise ValueError(
            "power_pW must have shape (n_detectors, n_times)."
        )

    if elevation_deg_matrix.shape != power_pW.shape:
        raise ValueError(
            "elevation_deg_matrix must have the same shape as power_pW."
        )

    # Representative elevation of the array at each time sample.
    elevation_deg = np.nanmedian(
        elevation_deg_matrix,
        axis=0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        airmass = 1.0 / np.sin(np.deg2rad(elevation_deg))

    airmass[
        (~np.isfinite(airmass))
        | (elevation_deg <= 0)
    ] = np.nan

    # Absolute/common atmospheric loading.
    raw_array_median_pW = np.nanmedian(
        power_pW,
        axis=0,
    )

    # Remove each detector's median over the whole observation.
    detector_medians = np.nanmedian(
        power_pW,
        axis=1,
        keepdims=True,
    )

    detector_median_subtracted = (
        power_pW - detector_medians
    )

    # This is the common temporal signal remaining after static
    # detector-to-detector offsets are removed.
    median_subtracted_common_pW = np.nanmedian(
        detector_median_subtracted,
        axis=0,
    )

    # Remove the instantaneous common signal to isolate the
    # detector-to-detector residual structure.
    small_scale_residual_pW = (
        detector_median_subtracted
        - median_subtracted_common_pW[np.newaxis, :]
    )

    small_scale_std_pW = np.nanstd(
        small_scale_residual_pW,
        axis=0,
    )

    small_scale_mad_pW = np.full(
        power_pW.shape[1],
        np.nan,
        dtype=np.float64,
    )

    for time_index in range(power_pW.shape[1]):
        small_scale_mad_pW[time_index] = robust_sigma_mad(
            small_scale_residual_pW[:, time_index]
        )

    return {
        "elevation_deg": elevation_deg,
        "airmass": airmass,
        "raw_array_median_pW": raw_array_median_pW,
        "detector_median_subtracted_pW": detector_median_subtracted,
        "median_subtracted_common_pW": median_subtracted_common_pW,
        "small_scale_residual_pW": small_scale_residual_pW,
        "small_scale_std_pW": small_scale_std_pW,
        "small_scale_mad_pW": small_scale_mad_pW,
    }


def plot_binned_relationship(
    x,
    y,
    x_label,
    y_label,
    title,
    output_path,
    bin_width,
):
    """Make a scatter plot with a binned median and percentile range."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)

    binned = bin_metric(
        x=x,
        y=y,
        bin_width=bin_width,
    )

    fig, axis = plt.subplots(figsize=(8, 6))

    axis.scatter(
        x[valid],
        y[valid],
        s=5,
        alpha=0.15,
        rasterized=True,
        label="Time samples",
    )

    if not binned.empty:
        axis.plot(
            binned["bin_centre"],
            binned["median"],
            marker="o",
            linewidth=1.5,
            label="Binned median",
        )

        axis.fill_between(
            binned["bin_centre"],
            binned["p16"],
            binned["p84"],
            alpha=0.2,
            label="16th-84th percentile",
        )

    correlation = safe_correlation(x, y)

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(
        f"{title}\nPearson r = {correlation:.3f}"
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return binned, correlation


def update_combined_pwv_plots(
    comparison_directory,
    band_label,
    scan_pattern,
):
    """
    Rebuild combined PWV plots from all run summaries currently available.
    """
    comparison_directory = Path(comparison_directory)

    summary_files = sorted(
        comparison_directory.glob(
            "*_atmospheric_power_summary.csv"
        )
    )

    if len(summary_files) == 0:
        return

    summary_frames = []

    for summary_file in summary_files:
        try:
            summary_frames.append(pd.read_csv(summary_file))
        except Exception as error:
            print(
                f"Could not read {summary_file}: {error}"
            )

    if len(summary_frames) == 0:
        return

    combined = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values("pwv_mm")
        .drop_duplicates(
            subset=["pwv_mm", "elevation_label"],
            keep="last",
        )
    )

    combined_csv = comparison_directory / (
        f"{band_label}GHz_{scan_pattern}_"
        "combined_atmospheric_power_summary.csv"
    )

    combined.to_csv(
        combined_csv,
        index=False,
    )

    # Absolute loading versus PWV.
    fig, axis = plt.subplots(figsize=(8, 6))

    axis.plot(
        combined["pwv_mm"],
        combined["median_raw_array_power_pW"],
        marker="o",
    )

    axis.set_xlabel("PWV (mm)")
    axis.set_ylabel("Median array power (pW)")
    axis.set_title(
        f"Absolute Detector Loading versus PWV\n"
        f"{band_label} GHz, {scan_pattern}"
    )
    axis.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        comparison_directory
        / (
            f"{band_label}GHz_{scan_pattern}_"
            "median_array_power_vs_pwv.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Small-scale residual amplitude versus PWV.
    fig, axis = plt.subplots(figsize=(8, 6))

    axis.plot(
        combined["pwv_mm"],
        combined["median_small_scale_std_pW"],
        marker="o",
        label="Standard deviation",
    )

    axis.plot(
        combined["pwv_mm"],
        combined["median_small_scale_mad_pW"],
        marker="s",
        label="Robust MAD estimate",
    )

    axis.set_xlabel("PWV (mm)")
    axis.set_ylabel("Median small-scale scatter (pW)")
    axis.set_title(
        f"Small-Scale Detector Fluctuations versus PWV\n"
        f"{band_label} GHz, {scan_pattern}"
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()

    fig.savefig(
        comparison_directory
        / (
            f"{band_label}GHz_{scan_pattern}_"
            "small_scale_scatter_vs_pwv.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Updated combined atmospheric comparison: {combined_csv}"
    )

def get_tau0_from_maria_transmission(
    *,
    band,
    pwv_mm,
    reference_elevation_deg=60.0,
    site_altitude_m=5600.0,
):
    """
    Derive zenith optical depth from MARIA transmission at the
    frequency sample nearest the observing-band centre.

    MARIA returns line-of-sight transmission:

        transmission = exp(-tau_0 / sin(elevation))

    Therefore:

        tau_0 = -log(transmission) * sin(elevation)
    """

    atmosphere = AtmosphericSpectrum(
        region="chajnantor",
        altitude=site_altitude_m,
    )

    frequencies_hz = np.asarray(
        band.nu.Hz,
        dtype=np.float64,
    )

    band_center_hz = float(band.center.Hz)

    center_index = np.argmin(
        np.abs(frequencies_hz - band_center_hz)
    )

    reference_elevation_rad = np.deg2rad(
        reference_elevation_deg
    )

    transmission = atmosphere.transmission(
        nu=frequencies_hz,
        elevation=float(reference_elevation_rad),
        pwv=float(pwv_mm),
    )

    transmission = np.asarray(
        transmission,
        dtype=np.float64,
    )

    transmission_center = transmission[center_index]

    if (
        not np.isfinite(transmission_center)
        or transmission_center <= 0
        or transmission_center > 1
    ):
        raise ValueError(
            "Invalid MARIA transmission at the band centre: "
            f"{transmission_center}"
        )

    tau_0 = (
        -np.log(transmission_center)
        * np.sin(reference_elevation_rad)
    )

    actual_center_ghz = (
        frequencies_hz[center_index] / 1e9
    )

    print("\nMARIA transmission-derived opacity")
    print("-" * 50)
    print(f"Requested band centre: {band_center_hz / 1e9:.3f} GHz")
    print(f"Nearest MARIA frequency: {actual_center_ghz:.3f} GHz")
    print(f"PWV: {pwv_mm:.3f} mm")
    print(
        f"Reference elevation: "
        f"{reference_elevation_deg:.2f} deg"
    )
    print(
        f"Band-centre transmission: "
        f"{transmission_center:.6g}"
    )
    print(f"Derived tau_0: {tau_0:.6g}")

    return {
        "tau_0": float(tau_0),
        "transmission_center": float(transmission_center),
        "frequency_center_ghz": float(actual_center_ghz),
        "reference_elevation_deg": float(
            reference_elevation_deg
        ),
    }

def run_atmospheric_power_tests(
    power_pW,
    elevation_deg_matrix,
    time_sec,
    pwv_mm,
    outdir,
    comparison_directory,
    run_prefix,
    band_label,
    scan_pattern,
    elevation_label,
    elevation_bin_width_deg=1.0,
    airmass_bin_width=0.01,
):
    """
    Run the elevation, airmass, and PWV power tests for one TOD.
    """
    outdir = Path(outdir)
    comparison_directory = Path(comparison_directory)

    atmosphere_outdir = outdir / "atmospheric_power_tests"
    atmosphere_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = calculate_atmospheric_power_metrics(
        power_pW=power_pW,
        elevation_deg_matrix=elevation_deg_matrix,
    )

    if len(time_sec) != len(metrics["elevation_deg"]):
        raise ValueError(
            "time_sec does not match the number of power samples."
        )

    # Save a per-time-sample table for later checks.
    time_series_df = pd.DataFrame({
        "time_s": time_sec,
        "pwv_mm": pwv_mm,
        "elevation_deg": metrics["elevation_deg"],
        "airmass": metrics["airmass"],
        "raw_array_median_power_pW": (
            metrics["raw_array_median_pW"]
        ),
        "median_subtracted_common_power_pW": (
            metrics["median_subtracted_common_pW"]
        ),
        "small_scale_std_pW": (
            metrics["small_scale_std_pW"]
        ),
        "small_scale_mad_pW": (
            metrics["small_scale_mad_pW"]
        ),
    })

    time_series_csv = atmosphere_outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "atmospheric_power_time_series.csv"
    )

    time_series_df.to_csv(
        time_series_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Time comparison
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 12),
        sharex=True,
    )

    axes[0].plot(
        time_sec,
        metrics["elevation_deg"],
        linewidth=0.8,
    )
    axes[0].set_ylabel("Elevation (deg)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        time_sec,
        metrics["median_subtracted_common_pW"],
        linewidth=0.7,
    )
    axes[1].set_ylabel(
        "Median-subtracted\ncommon power (pW)"
    )
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        time_sec,
        metrics["small_scale_std_pW"],
        linewidth=0.7,
        label="Standard deviation",
    )
    axes[2].plot(
        time_sec,
        metrics["small_scale_mad_pW"],
        linewidth=0.7,
        alpha=0.8,
        label="MAD estimate",
    )
    axes[2].set_ylabel(
        "Small-scale\nscatter (pW)"
    )
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(
        time_sec,
        metrics["airmass"],
        linewidth=0.8,
    )
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Airmass")
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        f"Atmospheric Power Quantities versus Time\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm, "
        f"{scan_pattern}, Elev={elevation_label}"
    )

    fig.tight_layout()

    fig.savefig(
        atmosphere_outdir
        / (
            f"{run_prefix}_{band_label}GHz_"
            "atmospheric_power_quantities_vs_time.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Elevation correlations
    # --------------------------------------------------------
    elevation_tests = {
        "raw_array_power": (
            metrics["raw_array_median_pW"],
            "Array-median power (pW)",
            "Raw Array Power versus Elevation",
        ),
        "median_subtracted_common_power": (
            metrics["median_subtracted_common_pW"],
            "Median-subtracted common power (pW)",
            "Median-Subtracted Common Power versus Elevation",
        ),
        "small_scale_std": (
            metrics["small_scale_std_pW"],
            "Small-scale standard deviation (pW)",
            "Small-Scale Power Scatter versus Elevation",
        ),
        "small_scale_mad": (
            metrics["small_scale_mad_pW"],
            "Small-scale MAD estimate (pW)",
            "Robust Small-Scale Power Scatter versus Elevation",
        ),
    }

    correlation_results = {}
    binned_elevation_frames = []

    for test_name, (
        y_values,
        y_label,
        title,
    ) in elevation_tests.items():

        binned, correlation = plot_binned_relationship(
            x=metrics["elevation_deg"],
            y=y_values,
            x_label="Array-median elevation (deg)",
            y_label=y_label,
            title=(
                f"{title}\n"
                f"{band_label} GHz, PWV={pwv_mm:.2f} mm"
            ),
            output_path=(
                atmosphere_outdir
                / (
                    f"{run_prefix}_{band_label}GHz_"
                    f"{test_name}_vs_elevation.png"
                )
            ),
            bin_width=elevation_bin_width_deg,
        )

        correlation_results[
            f"{test_name}_elevation_r"
        ] = correlation

        if not binned.empty:
            binned["metric"] = test_name
            binned["pwv_mm"] = pwv_mm
            binned["x_quantity"] = "elevation_deg"
            binned_elevation_frames.append(binned)

    # --------------------------------------------------------
    # Airmass correlations
    # --------------------------------------------------------
    airmass_tests = {
        "raw_array_power": (
            metrics["raw_array_median_pW"],
            "Array-median power (pW)",
            "Raw Array Power versus Airmass",
        ),
        "median_subtracted_common_power": (
            metrics["median_subtracted_common_pW"],
            "Median-subtracted common power (pW)",
            "Median-Subtracted Common Power versus Airmass",
        ),
        "small_scale_std": (
            metrics["small_scale_std_pW"],
            "Small-scale standard deviation (pW)",
            "Small-Scale Power Scatter versus Airmass",
        ),
        "small_scale_mad": (
            metrics["small_scale_mad_pW"],
            "Small-scale MAD estimate (pW)",
            "Robust Small-Scale Power Scatter versus Airmass",
        ),
    }

    binned_airmass_frames = []

    for test_name, (
        y_values,
        y_label,
        title,
    ) in airmass_tests.items():

        binned, correlation = plot_binned_relationship(
            x=metrics["airmass"],
            y=y_values,
            x_label="Approximate airmass",
            y_label=y_label,
            title=(
                f"{title}\n"
                f"{band_label} GHz, PWV={pwv_mm:.2f} mm"
            ),
            output_path=(
                atmosphere_outdir
                / (
                    f"{run_prefix}_{band_label}GHz_"
                    f"{test_name}_vs_airmass.png"
                )
            ),
            bin_width=airmass_bin_width,
        )

        correlation_results[
            f"{test_name}_airmass_r"
        ] = correlation

        if not binned.empty:
            binned["metric"] = test_name
            binned["pwv_mm"] = pwv_mm
            binned["x_quantity"] = "airmass"
            binned_airmass_frames.append(binned)

    if len(binned_elevation_frames) > 0:
        elevation_binned_df = pd.concat(
            binned_elevation_frames,
            ignore_index=True,
        )

        elevation_binned_df.to_csv(
            atmosphere_outdir
            / (
                f"{run_prefix}_{band_label}GHz_"
                "power_metrics_binned_by_elevation.csv"
            ),
            index=False,
        )

        # Save a copy in the shared comparison directory.
        elevation_binned_df.to_csv(
            comparison_directory
            / (
                f"{run_prefix}_{band_label}GHz_"
                "power_metrics_binned_by_elevation.csv"
            ),
            index=False,
        )

    if len(binned_airmass_frames) > 0:
        airmass_binned_df = pd.concat(
            binned_airmass_frames,
            ignore_index=True,
        )

        airmass_binned_df.to_csv(
            atmosphere_outdir
            / (
                f"{run_prefix}_{band_label}GHz_"
                "power_metrics_binned_by_airmass.csv"
            ),
            index=False,
        )

    summary_row = {
        "run_prefix": run_prefix,
        "band_ghz": band_label,
        "scan_pattern": scan_pattern,
        "elevation_label": elevation_label,
        "pwv_mm": pwv_mm,
        "median_elevation_deg": np.nanmedian(
            metrics["elevation_deg"]
        ),
        "median_airmass": np.nanmedian(
            metrics["airmass"]
        ),
        "median_raw_array_power_pW": np.nanmedian(
            metrics["raw_array_median_pW"]
        ),
        "median_small_scale_std_pW": np.nanmedian(
            metrics["small_scale_std_pW"]
        ),
        "median_small_scale_mad_pW": np.nanmedian(
            metrics["small_scale_mad_pW"]
        ),
        "p16_small_scale_std_pW": np.nanpercentile(
            metrics["small_scale_std_pW"],
            16,
        ),
        "p84_small_scale_std_pW": np.nanpercentile(
            metrics["small_scale_std_pW"],
            84,
        ),
    }

    summary_row.update(correlation_results)

    summary_df = pd.DataFrame([summary_row])

    summary_csv = atmosphere_outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "atmospheric_power_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    shared_summary_csv = comparison_directory / (
        f"{run_prefix}_{band_label}GHz_"
        "atmospheric_power_summary.csv"
    )

    summary_df.to_csv(
        shared_summary_csv,
        index=False,
    )

    update_combined_pwv_plots(
        comparison_directory=comparison_directory,
        band_label=band_label,
        scan_pattern=scan_pattern,
    )

    print(
        f"Saved atmospheric power tests to: {atmosphere_outdir}"
    )

    return metrics, summary_df




def get_mkid_parameters(band_label):
    """
    Return the MKID response parameters used for one Prime-Cam band.
    """

    band_label = str(band_label)

    if band_label == "280":
        return {
            "Q_r": 40000.0,
            "R_0": -2.448e9,
            "P_0": 957e-18,
        }

    if band_label == "350":
        return {
            "Q_r": 40000.0,
            "R_0": -2.448e9,
            "P_0": 957e-18,
        }

    if band_label == "850":
        return {
            "Q_r": 15000.0,
            "R_0": -1.0e7,
            "P_0": 120e-12,
        }

    raise ValueError(
        f"Unsupported Prime-Cam band: {band_label}"
    )

def calculate_linewidth_feasibility_metrics(
    power_pW,
    band_label,
    linewidth_limit=0.10,
):
    """
    Calculate fixed-tone MKID frequency excursions for a full
    detector-by-time power matrix.

    The fixed readout tone for each detector is assumed to correspond
    to that detector's median optical loading over the observation.

    Parameters
    ----------
    power_pW : ndarray
        Detector optical power with shape
        (n_detectors, n_times), in pW.

    band_label : str
        Prime-Cam band: "280", "350", or "850".

    linewidth_limit : float
        Adopted acceptable absolute Delta f / FWHM excursion.

    Returns
    -------
    metrics : dict
        One-row summary metrics for the simulation.

    delta_f_fwhm : ndarray
        Detector-by-time normalized frequency-shift matrix.
    """

    power_pW = np.asarray(
        power_pW,
        dtype=np.float64,
    )

    if power_pW.ndim != 2:
        raise ValueError(
            "power_pW must have shape "
            "(n_detectors, n_times)."
        )

    parameters = get_mkid_parameters(
        band_label
    )

    Q_r_local = parameters["Q_r"]
    R_0_local = parameters["R_0"]
    P_0_local = parameters["P_0"]

    power_W = power_pW * 1e-12

    # One fixed operating point for each detector.
    median_power_W = np.nanmedian(
        power_W,
        axis=1,
        keepdims=True,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        responsivity_at_reference = (
            R_0_local
            / np.sqrt(
                1.0
                + median_power_W / P_0_local
            )
        )

    delta_power_W = (
        power_W - median_power_W
    )

    delta_f_fwhm = (
        Q_r_local
        * responsivity_at_reference
        * delta_power_W
    )

    finite = np.isfinite(
        delta_f_fwhm
    )

    absolute_shift = np.abs(
        delta_f_fwhm[finite]
    )

    if absolute_shift.size == 0:
        raise ValueError(
            "No finite Delta f / FWHM values were calculated."
        )

    within_limit = (
        absolute_shift <= linewidth_limit
    )

    # Fraction within the limit for each individual detector.
    per_detector_fraction = np.full(
        power_pW.shape[0],
        np.nan,
        dtype=np.float64,
    )

    per_detector_max_abs = np.full(
        power_pW.shape[0],
        np.nan,
        dtype=np.float64,
    )

    per_detector_p95_abs = np.full(
        power_pW.shape[0],
        np.nan,
        dtype=np.float64,
    )

    for detector_index in range(
        power_pW.shape[0]
    ):
        detector_values = np.abs(
            delta_f_fwhm[
                detector_index,
                np.isfinite(
                    delta_f_fwhm[
                        detector_index
                    ]
                ),
            ]
        )

        if detector_values.size == 0:
            continue

        per_detector_fraction[
            detector_index
        ] = np.mean(
            detector_values <= linewidth_limit
        )

        per_detector_max_abs[
            detector_index
        ] = np.nanmax(
            detector_values
        )

        per_detector_p95_abs[
            detector_index
        ] = np.nanpercentile(
            detector_values,
            95,
        )

    metrics = {
        "linewidth_limit": float(
            linewidth_limit
        ),

        # All detector-time samples pooled together.
        "fraction_within_linewidth_limit": float(
            np.mean(within_limit)
        ),
        "p50_abs_delta_f_over_fwhm": float(
            np.nanpercentile(
                absolute_shift,
                50,
            )
        ),
        "p95_abs_delta_f_over_fwhm": float(
            np.nanpercentile(
                absolute_shift,
                95,
            )
        ),
        "p99_abs_delta_f_over_fwhm": float(
            np.nanpercentile(
                absolute_shift,
                99,
            )
        ),
        "max_abs_delta_f_over_fwhm": float(
            np.nanmax(
                absolute_shift
            )
        ),

        # Distribution across detectors.
        "median_detector_fraction_within_limit": float(
            np.nanmedian(
                per_detector_fraction
            )
        ),
        "minimum_detector_fraction_within_limit": float(
            np.nanmin(
                per_detector_fraction
            )
        ),
        "fraction_detectors_fully_within_limit": float(
            np.nanmean(
                per_detector_fraction >= 1.0
            )
        ),
        "median_detector_p95_abs_delta_f_over_fwhm": float(
            np.nanmedian(
                per_detector_p95_abs
            )
        ),
        "maximum_detector_p95_abs_delta_f_over_fwhm": float(
            np.nanmax(
                per_detector_p95_abs
            )
        ),
        "median_detector_max_abs_delta_f_over_fwhm": float(
            np.nanmedian(
                per_detector_max_abs
            )
        ),

        # Record the model parameters in the output.
        "mkid_Q_r": float(
            Q_r_local
        ),
        "mkid_R_0_W_inverse": float(
            R_0_local
        ),
        "mkid_P_0_W": float(
            P_0_local
        ),
    }

    return metrics, delta_f_fwhm

def match_coordinate_to_power_shape(
    coordinate,
    power_shape,
    coordinate_name,
):
    """
    Convert a MARIA coordinate array to detector-by-time shape.
    """

    coordinate = np.squeeze(
        np.asarray(
            coordinate,
            dtype=np.float64,
        )
    )

    if coordinate.shape == power_shape:
        return coordinate

    if coordinate.T.shape == power_shape:
        return coordinate.T

    raise ValueError(
        f"Could not match {coordinate_name} shape "
        f"{coordinate.shape} to power shape "
        f"{power_shape}."
    )
def calculate_sky_coverage_metrics(
    tod,
    power_shape,
    map_size_deg,
    pixel_size_deg=0.02,
    minimum_hits_for_coverage=1,
    minimum_hits_for_revisit=2,
):
    """
    Calculate focal-plane sky coverage using every detector position.

    Coverage is evaluated within a fixed square map of width
    map_size_deg centred on the median array pointing.

    Returns
    -------
    metrics : dict
        Coverage fraction, revisit fraction, and hit-count uniformity.

    hit_map : ndarray
        Two-dimensional map of sample counts.

    x_edges, y_edges : ndarray
        Histogram edges in tangent-plane degrees.
    """

    ra_matrix = match_coordinate_to_power_shape(
        tod.ra,
        power_shape,
        "right ascension",
    )

    dec_matrix = match_coordinate_to_power_shape(
        tod.dec,
        power_shape,
        "declination",
    )

    # MARIA normally stores RA and Dec in radians.
    ra_rad = np.asarray(
        ra_matrix,
        dtype=np.float64,
    )

    dec_rad = np.asarray(
        dec_matrix,
        dtype=np.float64,
    )

    finite_geometry = (
        np.isfinite(ra_rad)
        & np.isfinite(dec_rad)
    )

    if not np.any(finite_geometry):
        raise ValueError(
            "No finite RA/Dec coordinates were found."
        )

    # Representative map centre.
    ra_centre_rad = np.nanmedian(
        ra_rad[finite_geometry]
    )

    dec_centre_rad = np.nanmedian(
        dec_rad[finite_geometry]
    )

    # Wrap the RA difference onto [-pi, pi].
    delta_ra_rad = np.angle(
        np.exp(
            1j
            * (
                ra_rad
                - ra_centre_rad
            )
        )
    )

    # Small-angle tangent-plane coordinates.
    x_deg = np.rad2deg(
        delta_ra_rad
        * np.cos(dec_centre_rad)
    )

    y_deg = np.rad2deg(
        dec_rad
        - dec_centre_rad
    )

    half_width_deg = (
        0.5 * float(map_size_deg)
    )

    n_bins = max(
        1,
        int(
            np.ceil(
                map_size_deg
                / pixel_size_deg
            )
        ),
    )

    histogram_range = [
        [
            -half_width_deg,
            half_width_deg,
        ],
        [
            -half_width_deg,
            half_width_deg,
        ],
    ]

    valid = (
        np.isfinite(x_deg)
        & np.isfinite(y_deg)
        & (x_deg >= -half_width_deg)
        & (x_deg <= half_width_deg)
        & (y_deg >= -half_width_deg)
        & (y_deg <= half_width_deg)
    )

    hit_map, x_edges, y_edges = np.histogram2d(
        x_deg[valid],
        y_deg[valid],
        bins=(n_bins, n_bins),
        range=histogram_range,
    )

    covered_mask = (
        hit_map
        >= minimum_hits_for_coverage
    )

    revisit_mask = (
        hit_map
        >= minimum_hits_for_revisit
    )

    total_pixels = hit_map.size
    covered_hits = hit_map[
        covered_mask
    ]

    coverage_fraction = (
        np.count_nonzero(
            covered_mask
        )
        / total_pixels
    )

    revisit_fraction = (
        np.count_nonzero(
            revisit_mask
        )
        / total_pixels
    )

    if (
        covered_hits.size > 0
        and np.nanmean(
            covered_hits
        ) > 0
    ):
        mean_hits_per_covered_pixel = float(
            np.nanmean(
                covered_hits
            )
        )

        hit_count_cv = float(
            np.nanstd(
                covered_hits
            )
            / mean_hits_per_covered_pixel
        )
    else:
        mean_hits_per_covered_pixel = np.nan
        hit_count_cv = np.nan

    metrics = {
        "coverage_map_size_deg": float(
            map_size_deg
        ),
        "coverage_pixel_size_deg": float(
            pixel_size_deg
        ),
        "coverage_n_pixels": int(
            total_pixels
        ),
        "coverage_n_covered_pixels": int(
            np.count_nonzero(
                covered_mask
            )
        ),
        "coverage_fraction": float(
            coverage_fraction
        ),
        "revisit_fraction": float(
            revisit_fraction
        ),
        "hit_count_cv": float(
            hit_count_cv
        ),
        "mean_hits_per_covered_pixel": float(
            mean_hits_per_covered_pixel
        ),
    }

    return (
        metrics,
        hit_map,
        x_edges,
        y_edges,
    )


def save_coverage_hit_map(
    hit_map,
    x_edges,
    y_edges,
    output_path,
    title,
):
    """
    Save a focal-plane hit-count map.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        hit_map.T,
        origin="lower",
        aspect="equal",
        extent=[
            x_edges[0],
            x_edges[-1],
            y_edges[0],
            y_edges[-1],
        ],
    )

    colorbar = fig.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Detector samples per pixel"
    )

    axis.set_xlabel(
        "Projected RA offset (deg)"
    )

    axis.set_ylabel(
        "Declination offset (deg)"
    )

    axis.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_band_for_sweep(
    band_label,
    efficiency=0.5,
):
    """
    Construct the MARIA Band object for one Prime-Cam band.
    """

    band_label = str(band_label)

    if band_label == "280":
        centre_hz = 280e9
        width_hz = 60e9
        net_cmb = 13e-6

    elif band_label == "350":
        centre_hz = 350e9
        width_hz = 35e9
        net_cmb = 48e-6

    elif band_label == "850":
        centre_hz = 850e9
        width_hz = 97e9
        net_cmb = 13e-6

    else:
        raise ValueError(
            f"Unsupported band: {band_label}"
        )

    return Band(
        name="m2/f093",
        center=centre_hz,
        width=width_hz,
        efficiency=efficiency,
        NET_CMB=net_cmb,
        knee=1.0,
        gain_error=5e-2,
    )


def analyse_power_against_optical_depth(
    power_pW,
    elevation_deg_matrix,
    time_sec,
    pwv_mm,
    tau_0,
    outdir,
    run_prefix,
    band_label,
):
    """
    Compare detector power with the elevation-dependent atmospheric
    loading predicted using a transmission-derived zenith opacity.

    Parameters
    ----------
    power_pW : ndarray
        Detector power with shape (n_detectors, n_times), in pW.

    elevation_deg_matrix : ndarray
        Detector elevation with shape (n_detectors, n_times),
        in degrees.

    time_sec : ndarray
        Time coordinate with shape (n_times,), in seconds.

    pwv_mm : float
        PWV used in the MARIA simulation, in mm.

    tau_0 : float
        Zenith optical depth derived directly from MARIA transmission.

    outdir : Path
        Directory in which plots and CSV files are saved.

    run_prefix : str
        Prefix used for output filenames.

    band_label : str
        Band label, such as "850".

    Returns
    -------
    summary : dict
        Summary statistics from the atmospheric-model comparison.
    """

    power_pW = np.asarray(
        power_pW,
        dtype=np.float64,
    )

    elevation_deg_matrix = np.asarray(
        elevation_deg_matrix,
        dtype=np.float64,
    )

    time_sec = np.asarray(
        time_sec,
        dtype=np.float64,
    )

    tau_0 = float(tau_0)
    pwv_mm = float(pwv_mm)

    # --------------------------------------------------------
    # Basic input checks
    # --------------------------------------------------------

    if power_pW.ndim != 2:
        raise ValueError(
            "power_pW must have shape "
            "(n_detectors, n_times)."
        )

    if elevation_deg_matrix.shape != power_pW.shape:
        raise ValueError(
            "elevation_deg_matrix and power_pW must have "
            "the same detector-by-time shape. "
            f"Received {elevation_deg_matrix.shape} and "
            f"{power_pW.shape}."
        )

    if time_sec.ndim != 1:
        raise ValueError(
            "time_sec must be one-dimensional."
        )

    if power_pW.shape[1] != len(time_sec):
        raise ValueError(
            "The time axis of power_pW does not match time_sec."
        )

    if not np.isfinite(tau_0) or tau_0 <= 0:
        raise ValueError(
            f"tau_0 must be finite and positive. Received {tau_0}."
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Construct array-common power quantities
    # --------------------------------------------------------

    # Raw common loading across the detector array.
    raw_array_power_pW = np.nanmedian(
        power_pW,
        axis=0,
    )

    # Remove each detector's median over the observation.
    detector_median_pW = np.nanmedian(
        power_pW,
        axis=1,
        keepdims=True,
    )

    detector_median_subtracted_pW = (
        power_pW - detector_median_pW
    )

    # Common time-varying signal remaining after detector
    # baselines have been removed.
    median_subtracted_common_pW = np.nanmedian(
        detector_median_subtracted_pW,
        axis=0,
    )

    # Remove the instantaneous common signal to obtain
    # small-scale detector-to-detector residuals.
    small_scale_residual_pW = (
        detector_median_subtracted_pW
        - median_subtracted_common_pW[np.newaxis, :]
    )

    # Standard and robust measures of small-scale scatter.
    small_scale_std_pW = np.nanstd(
        small_scale_residual_pW,
        axis=0,
    )

    residual_median_pW = np.nanmedian(
        small_scale_residual_pW,
        axis=0,
        keepdims=True,
    )

    small_scale_mad_pW = (
        1.4826
        * np.nanmedian(
            np.abs(
                small_scale_residual_pW
                - residual_median_pW
            ),
            axis=0,
        )
    )

    # --------------------------------------------------------
    # Construct one representative elevation per time
    # --------------------------------------------------------

    array_elevation_deg = np.nanmedian(
        elevation_deg_matrix,
        axis=0,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        airmass = (
            1.0
            / np.sin(
                np.deg2rad(array_elevation_deg)
            )
        )

    invalid_geometry = (
        ~np.isfinite(airmass)
        | (array_elevation_deg <= 0)
        | (array_elevation_deg > 90)
    )

    airmass[invalid_geometry] = np.nan

    # --------------------------------------------------------
    # Optical-depth model
    # --------------------------------------------------------

    # Line-of-sight optical depth:
    #
    # tau_los = tau_0 * airmass
    tau_line_of_sight = (
        tau_0 * airmass
    )

    # Atmospheric emission term relative to the physical
    # temperature of the atmosphere:
    #
    # T_atm / T_0 = 1 - exp(-tau_los)
    atmospheric_emission_fraction = (
        1.0
        - np.exp(-tau_line_of_sight)
    )

    # --------------------------------------------------------
    # Select finite samples
    # --------------------------------------------------------

    valid = (
        np.isfinite(time_sec)
        & np.isfinite(array_elevation_deg)
        & np.isfinite(airmass)
        & np.isfinite(tau_line_of_sight)
        & np.isfinite(atmospheric_emission_fraction)
        & np.isfinite(raw_array_power_pW)
        & np.isfinite(median_subtracted_common_pW)
        & np.isfinite(small_scale_std_pW)
        & np.isfinite(small_scale_mad_pW)
    )

    if np.count_nonzero(valid) < 3:
        raise ValueError(
            "Fewer than three finite time samples remain "
            "for the optical-depth analysis."
        )

    time_valid = time_sec[valid]

    elevation_valid = (
        array_elevation_deg[valid]
    )

    airmass_valid = airmass[valid]

    tau_valid = (
        tau_line_of_sight[valid]
    )

    emission_valid = (
        atmospheric_emission_fraction[valid]
    )

    raw_power_valid = (
        raw_array_power_pW[valid]
    )

    common_power_valid = (
        median_subtracted_common_pW[valid]
    )

    small_scale_std_valid = (
        small_scale_std_pW[valid]
    )

    small_scale_mad_valid = (
        small_scale_mad_pW[valid]
    )

    # --------------------------------------------------------
    # Fit the atmospheric model to measured power
    # --------------------------------------------------------
    #
    # The emission fraction is dimensionless, while measured
    # power is in pW and contains an arbitrary baseline.
    #
    # Therefore fit:
    #
    # P_model = scale * emission_fraction + offset
    # --------------------------------------------------------

    model_scale, model_offset = np.polyfit(
        emission_valid,
        raw_power_valid,
        deg=1,
    )

    predicted_raw_power_pW = (
        model_scale * emission_valid
        + model_offset
    )

    raw_model_residual_pW = (
        raw_power_valid
        - predicted_raw_power_pW
    )

    # Centre the predicted track before comparing it with the
    # detector-median-subtracted common power.
    predicted_common_power_pW = (
        predicted_raw_power_pW
        - np.nanmedian(predicted_raw_power_pW)
    )

    common_model_residual_pW = (
        common_power_valid
        - predicted_common_power_pW
    )

    # --------------------------------------------------------
    # Statistical comparisons
    # --------------------------------------------------------

    raw_model_r, raw_model_p = pearsonr(
        raw_power_valid,
        predicted_raw_power_pW,
    )

    common_model_r, common_model_p = pearsonr(
        common_power_valid,
        predicted_common_power_pW,
    )

    tau_raw_r, tau_raw_p = pearsonr(
        tau_valid,
        raw_power_valid,
    )

    tau_common_r, tau_common_p = pearsonr(
        tau_valid,
        common_power_valid,
    )

    tau_small_std_r, tau_small_std_p = pearsonr(
        tau_valid,
        small_scale_std_valid,
    )

    tau_small_mad_r, tau_small_mad_p = pearsonr(
        tau_valid,
        small_scale_mad_valid,
    )

    time_small_std_r, time_small_std_p = pearsonr(
        time_valid,
        small_scale_std_valid,
    )

    time_small_mad_r, time_small_mad_p = pearsonr(
        time_valid,
        small_scale_mad_valid,
    )

    residual_elevation_r, residual_elevation_p = pearsonr(
        elevation_valid,
        common_model_residual_pW,
    )

    residual_airmass_r, residual_airmass_p = pearsonr(
        airmass_valid,
        common_model_residual_pW,
    )

    # Coefficient of determination for raw-power model.
    sum_squared_residuals = np.nansum(
        raw_model_residual_pW**2
    )

    sum_squared_total = np.nansum(
        (
            raw_power_valid
            - np.nanmean(raw_power_valid)
        ) ** 2
    )

    if sum_squared_total > 0:
        raw_model_r_squared = (
            1.0
            - sum_squared_residuals
            / sum_squared_total
        )
    else:
        raw_model_r_squared = np.nan

    # Fraction of raw temporal variance left after subtracting
    # the fitted elevation-dependent opacity model.
    raw_variance = np.nanvar(
        raw_power_valid
    )

    residual_variance = np.nanvar(
        raw_model_residual_pW
    )

    if raw_variance > 0:
        residual_variance_fraction = (
            residual_variance
            / raw_variance
        )
    else:
        residual_variance_fraction = np.nan

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\nAtmospheric optical-depth power analysis")
    print("-" * 60)

    print(f"Band: {band_label} GHz")
    print(f"PWV: {pwv_mm:.3f} mm")
    print(
        "Transmission-derived zenith opacity: "
        f"tau_0={tau_0:.6g}"
    )

    print(
        "Elevation range: "
        f"{np.nanmin(elevation_valid):.4f} to "
        f"{np.nanmax(elevation_valid):.4f} deg"
    )

    print(
        "Airmass range: "
        f"{np.nanmin(airmass_valid):.6f} to "
        f"{np.nanmax(airmass_valid):.6f}"
    )

    print(
        "Line-of-sight optical-depth range: "
        f"{np.nanmin(tau_valid):.6g} to "
        f"{np.nanmax(tau_valid):.6g}"
    )

    print(
        "Raw power versus fitted opacity model: "
        f"r={raw_model_r:.4f}, "
        f"R^2={raw_model_r_squared:.4f}"
    )

    print(
        "Median-subtracted common power versus model: "
        f"r={common_model_r:.4f}"
    )

    print(
        "Small-scale standard deviation versus tau_los: "
        f"r={tau_small_std_r:.4f}"
    )

    print(
        "Small-scale MAD versus tau_los: "
        f"r={tau_small_mad_r:.4f}"
    )

    print(
        "Raw model-residual standard deviation: "
        f"{np.nanstd(raw_model_residual_pW):.6g} pW"
    )

    print(
        "Fraction of raw temporal variance remaining: "
        f"{residual_variance_fraction:.4f}"
    )

    print(
        "Residual common power versus elevation: "
        f"r={residual_elevation_r:.4f}"
    )
    print(
        "Small-scale standard deviation versus time: "
        f"r={time_small_std_r:.4f}"
    )

    print(
        "Small-scale MAD versus tau_los:"
        f"r={tau_small_mad_r:.4f}"
    )
    print(
    "Small-scale standard deviation versus time: "
    f"r={time_small_std_r:.4f}"
    )

    print(
        "Small-scale MAD versus time: "
        f"r={time_small_mad_r:.4f}"
    )

    # ========================================================
    # Figure 1: atmospheric model quantities versus time
    # ========================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 12),
        sharex=True,
    )

    axes[0].plot(
        time_valid,
        elevation_valid,
        linewidth=0.8,
    )

    axes[0].set_ylabel("Elevation (deg)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        time_valid,
        tau_valid,
        linewidth=0.8,
    )

    axes[1].set_ylabel(
        r"Line-of-sight $\tau$"
    )

    axes[1].grid(alpha=0.3)

    axes[2].plot(
        time_valid,
        raw_power_valid,
        linewidth=0.8,
        label="Observed array-median power",
    )

    axes[2].plot(
        time_valid,
        predicted_raw_power_pW,
        linewidth=1.4,
        label="Fitted opacity prediction",
    )

    axes[2].set_ylabel("Power (pW)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(
        time_valid,
        raw_model_residual_pW,
        linewidth=0.8,
    )

    axes[3].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel(
        "Model residual (pW)"
    )

    axes[3].grid(alpha=0.3)

    fig.suptitle(
        "Detector Power and Transmission-Derived "
        "Optical-Depth Model\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm, "
        fr"$\tau_0={tau_0:.4f}$"
    )

    fig.tight_layout()

    model_time_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "transmission_opacity_model_vs_time.png"
    )

    fig.savefig(
        model_time_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Figure 2: raw power versus line-of-sight optical depth
    # ========================================================

    fig, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.scatter(
        tau_valid,
        raw_power_valid,
        s=5,
        alpha=0.25,
        rasterized=True,
        label="Time samples",
    )

    sort_indices = np.argsort(
        tau_valid
    )

    axis.plot(
        tau_valid[sort_indices],
        predicted_raw_power_pW[sort_indices],
        linewidth=2,
        label="Fitted opacity model",
    )

    axis.set_xlabel(
        r"Line-of-sight optical depth "
        r"$\tau_{\mathrm{los}}$"
    )

    axis.set_ylabel(
        "Array-median power (pW)"
    )

    axis.set_title(
        "Raw Detector Power versus Optical Depth\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm\n"
        f"Pearson r={tau_raw_r:.3f}, "
        f"$R^2$={raw_model_r_squared:.3f}"
    )

    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    fig.tight_layout()

    raw_tau_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "raw_array_power_vs_tau_los.png"
    )

    fig.savefig(
        raw_tau_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Figure 3: common median-subtracted power versus tau
    # ========================================================

    fig, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.scatter(
        tau_valid,
        common_power_valid,
        s=5,
        alpha=0.25,
        rasterized=True,
        label="Time samples",
    )

    axis.plot(
        tau_valid[sort_indices],
        predicted_common_power_pW[sort_indices],
        linewidth=2,
        label="Centred opacity prediction",
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_xlabel(
        r"Line-of-sight optical depth "
        r"$\tau_{\mathrm{los}}$"
    )

    axis.set_ylabel(
        "Median-subtracted common power (pW)"
    )

    axis.set_title(
        "Median-Subtracted Common Power versus Optical Depth\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm\n"
        f"Pearson r={tau_common_r:.3f}"
    )

    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    fig.tight_layout()

    common_tau_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "median_subtracted_common_power_vs_tau_los.png"
    )

    fig.savefig(
        common_tau_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Figure 4: small-scale scatter versus tau
    # ========================================================

    fig, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.scatter(
        tau_valid,
        small_scale_std_valid,
        s=5,
        alpha=0.25,
        rasterized=True,
        label="Standard deviation",
    )

    axis.scatter(
        tau_valid,
        small_scale_mad_valid,
        s=5,
        alpha=0.25,
        rasterized=True,
        label="MAD estimate",
    )

    axis.set_xlabel(
        r"Line-of-sight optical depth "
        r"$\tau_{\mathrm{los}}$"
    )

    axis.set_ylabel(
        "Small-scale power scatter (pW)"
    )

    axis.set_title(
        "Small-Scale Power Scatter versus Optical Depth\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm\n"
        f"STD r={tau_small_std_r:.3f}, "
        f"MAD r={tau_small_mad_r:.3f}"
    )

    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    fig.tight_layout()

    small_scale_tau_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "small_scale_scatter_vs_tau_los.png"
    )

    fig.savefig(
        small_scale_tau_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    # ========================================================
    # Figure 5: small-scale scatter versus time
    # ========================================================
    fig, axes = plt.subplots(
        3,
        1,
        figsize = (13,10),
        sharex=True,
    )

    axes[0].plot(
        time_valid,
        tau_valid,
        linewidth=0.8,
    )

    axes[0].set_ylabel(
        r"Line-of-sight $\tau$"
    )

    axes[0].set_title(
        "Line-of-Sight Optical Depth versus Time\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm"
    )

    axes[0].grid(alpha=0.3)

    axes[1].plot(
        time_valid,
        small_scale_std_valid,
        linewidth=0.7,
        label="Standard deviation",
    )

    axes[1].plot(
        time_valid,
        small_scale_mad_valid,
        linewidth=0.7,
        alpha=0.8,
        label="MAD estimate",
    )

    axes[1].set_ylabel(
        "Small-scale\nscatter (pW)"
    )

    axes[1].set_title(
        "Small-Scale Power Scatter versus Time\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm"
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")


    axes[2].plot(
        time_valid,
        common_model_residual_pW,
        linewidth=0.8,
    )

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel(
        "Residual common\npower (pW)"
    )

    axes[2].set_title(
        "Residual versus Time\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm"
    )

    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "Temporal Evolution of Atmospheric Power Structure\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm\n"
        f"STD–time r={time_small_std_r:.3f}, "
        f"MAD–time r={time_small_mad_r:.3f}"
    )

    fig.tight_layout()

    small_scale_time_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "small_scale_scatter_and_residuals_vs_time.png"
    )

    fig.savefig(small_scale_time_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    # ========================================================
    # Figure 6: residual common power
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
    )

    axes[0].plot(
        time_valid,
        common_model_residual_pW,
        linewidth=0.8,
    )

    axes[0].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(
        "Residual common\npower (pW)"
    )

    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        elevation_valid,
        common_model_residual_pW,
        s=5,
        alpha=0.25,
        rasterized=True,
    )

    axes[1].set_xlabel(
        "Array-median elevation (deg)"
    )

    axes[1].set_ylabel(
        "Residual common\npower (pW)"
    )

    axes[1].set_title(
        "Residual versus Elevation: "
        f"r={residual_elevation_r:.3f}"
    )

    axes[1].grid(alpha=0.3)

    axes[2].scatter(
        airmass_valid,
        common_model_residual_pW,
        s=5,
        alpha=0.25,
        rasterized=True,
    )

    axes[2].set_xlabel(
        "Approximate airmass"
    )

    axes[2].set_ylabel(
        "Residual common\npower (pW)"
    )

    axes[2].set_title(
        "Residual versus Airmass: "
        f"r={residual_airmass_r:.3f}"
    )

    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "Power Remaining after Removing "
        "the Optical-Depth Prediction\n"
        f"{band_label} GHz, PWV={pwv_mm:.2f} mm"
    )

    fig.tight_layout()

    residual_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "transmission_opacity_model_residuals.png"
    )

    fig.savefig(
        residual_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Save time-dependent quantities
    # ========================================================

    result_df = pd.DataFrame({
        "time_s": time_valid,
        "elevation_deg": elevation_valid,
        "airmass": airmass_valid,
        "pwv_mm": pwv_mm,
        "tau_0": tau_0,
        "tau_line_of_sight": tau_valid,
        "atmospheric_emission_fraction": emission_valid,
        "raw_array_median_power_pW": raw_power_valid,
        "predicted_raw_power_pW": predicted_raw_power_pW,
        "raw_model_residual_pW": raw_model_residual_pW,
        "median_subtracted_common_power_pW": (
            common_power_valid
        ),
        "predicted_common_power_pW": (
            predicted_common_power_pW
        ),
        "common_model_residual_pW": (
            common_model_residual_pW
        ),
        "small_scale_std_pW": small_scale_std_valid,
        "small_scale_mad_pW": small_scale_mad_valid,
    })

    result_csv_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "transmission_opacity_power_samples.csv"
    )

    result_df.to_csv(
        result_csv_path,
        index=False,
    )

    # ========================================================
    # Save one-row summary
    # ========================================================

    summary = {
        "band_ghz": band_label,
        "pwv_mm": pwv_mm,
        "tau_0_transmission": tau_0,
        "median_elevation_deg": np.nanmedian(
            elevation_valid
        ),
        "median_airmass": np.nanmedian(
            airmass_valid
        ),
        "tau_line_of_sight_min": np.nanmin(
            tau_valid
        ),
        "tau_line_of_sight_median": np.nanmedian(
            tau_valid
        ),
        "tau_line_of_sight_max": np.nanmax(
            tau_valid
        ),
        "model_scale_pW": model_scale,
        "model_offset_pW": model_offset,
        "raw_power_model_pearson_r": raw_model_r,
        "raw_power_model_p_value": raw_model_p,
        "raw_power_model_r_squared": raw_model_r_squared,
        "common_power_model_pearson_r": common_model_r,
        "common_power_model_p_value": common_model_p,
        "tau_raw_power_pearson_r": tau_raw_r,
        "tau_raw_power_p_value": tau_raw_p,
        "tau_common_power_pearson_r": tau_common_r,
        "tau_common_power_p_value": tau_common_p,
        "tau_small_scale_std_pearson_r": tau_small_std_r,
        "tau_small_scale_std_p_value": tau_small_std_p,
        "tau_small_scale_mad_pearson_r": tau_small_mad_r,
        "tau_small_scale_mad_p_value": tau_small_mad_p,

        "time_small_scale_std_pearson_r": (
            time_small_std_r
        ),
        "time_small_scale_std_p_value": (
            time_small_std_p
        ),
        "time_small_scale_mad_pearson_r": (
            time_small_mad_r
        ),
        "time_small_scale_mad_p_value": (
            time_small_mad_p
        ),
        "raw_model_residual_std_pW": np.nanstd(
            raw_model_residual_pW
        ),
        "common_model_residual_std_pW": np.nanstd(
            common_model_residual_pW
        ),
        "residual_variance_fraction": (
            residual_variance_fraction
        ),
        "residual_elevation_pearson_r": (
            residual_elevation_r
        ),
        "residual_elevation_p_value": (
            residual_elevation_p
        ),
        "residual_airmass_pearson_r": (
            residual_airmass_r
        ),
        "residual_airmass_p_value": (
            residual_airmass_p
        ),
        "median_small_scale_std_pW": np.nanmedian(
            small_scale_std_valid
        ),
        "median_small_scale_mad_pW": np.nanmedian(
            small_scale_mad_valid
        ),
    }

    summary_csv_path = outdir / (
        f"{run_prefix}_{band_label}GHz_"
        "transmission_opacity_power_summary.csv"
    )

    pd.DataFrame([summary]).to_csv(
        summary_csv_path,
        index=False,
    )

    print(f"\nSaved opacity-model time plot: {model_time_path}")
    print(f"Saved raw power vs tau plot: {raw_tau_path}")
    print(f"Saved common power vs tau plot: {common_tau_path}")
    print(
        "Saved small-scale scatter vs tau plot: "
        f"{small_scale_tau_path}"
    )
    print(
        "Saved small-scale scatter and residuals vs time plot: "
        f"{small_scale_time_path}"
    )
    print(f"Saved opacity-model residual plot: {residual_path}")
    print(f"Saved time-sample CSV: {result_csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")

    return summary

# ============================================================
# All-detector power statistics
# ============================================================

all_power_samples = P_pW[np.isfinite(P_pW)]

stats_rows = []

stats_rows.append(
    summarize(
        f"All-detector power, {BAND_LABEL} GHz",
        all_power_samples,
        "pW",
    )
)

make_hist(
    all_power_samples,
    "Detector power (pW)",
    (
        f"All-Detector Power Distribution\n"
        f"{BAND_LABEL} GHz, PWV={PWV_MM:.2f} mm, Elev={ELEV_LABEL}"
    ),
    f"{run_prefix}_{BAND_LABEL}GHz_all_detector_power_histogram.png",
)


# ============================================================
# Per-detector power and delta f / FWHM statistics
# ============================================================

detector_rows = []

all_delta_tracks = []
half_ptp_delta_by_detector = []
ptp_delta_by_detector = []
mean_power_by_detector = []
sigma_power_by_detector = []
frac_width_by_detector = []
standardized_power_tracks = []
standardized_delta_tracks = []

# Retain a detector-by-time matrix so that detector populations producing
# narrow features can be identified after the loop.
delta_matrix = np.full(P_pW.shape, np.nan, dtype=np.float64)

time_sec_full = (
    np.arange(P_pW.shape[1], dtype=np.float64) / SAMPLE_RATE_HZ
)

n_detectors = P_pW.shape[0]

for det_idx in range(n_detectors):

    P_track = np.asarray(P_pW[det_idx, :], dtype=np.float64)
    P_track = P_track[np.isfinite(P_track)]

    if len(P_track) == 0:
        continue

    mu, sigma = norm.fit(P_track)
    frac_width = sigma / mu if mu != 0 else np.nan

    delta_track = delta_f_over_fwhm(P_track)

    # The valid samples preserve their original order because boolean indexing
    # only removes non-finite entries. Save them back into the full matrix.
    valid_full = np.isfinite(P_pW[det_idx, :])
    delta_matrix[det_idx, valid_full] = delta_track

    power_standardized = standardize(P_track)
    delta_standardized = standardize(delta_track)

    if len(power_standardized) > 0:
        standardized_power_tracks.append(power_standardized)
    if len(delta_standardized) > 0:
        standardized_delta_tracks.append(delta_standardized)

    delta_ptp = np.nanmax(delta_track) - np.nanmin(delta_track)
    delta_half_ptp = np.abs(delta_ptp / 2)

    mean_power_by_detector.append(mu)
    sigma_power_by_detector.append(sigma)
    frac_width_by_detector.append(frac_width)
    ptp_delta_by_detector.append(delta_ptp)
    half_ptp_delta_by_detector.append(delta_half_ptp)
    all_delta_tracks.append(delta_track)

    detector_rows.append({
        "detector": det_idx,
        "power_mean_pW": mu,
        "power_sigma_pW": sigma,
        "power_frac_width": frac_width,
        "delta_f_over_fwhm_mean": np.nanmean(delta_track),
        "delta_f_over_fwhm_std": np.nanstd(delta_track),
        "delta_f_over_fwhm_min": np.nanmin(delta_track),
        "delta_f_over_fwhm_max": np.nanmax(delta_track),
        "delta_f_over_fwhm_ptp": delta_ptp,
        "delta_f_over_fwhm_half_ptp": delta_half_ptp,
    })


detector_df = pd.DataFrame(detector_rows)
detector_csv_path = OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_detector_power_deltaf_stats.csv"
detector_df.to_csv(detector_csv_path, index=False)

print(f"\nSaved detector statistics CSV to: {detector_csv_path}")


# ============================================================
#Array-common frequency-shift track
# ============================================================

# common_delta_track = np.nanmedian(delta_matrix, axis=0)
# common_valid = np.isfinite(common_delta_track)

# common_time = time_sec_full

# common_delta = common_delta_track[common_valid]
# common_time = common_time[common_valid]

# common_result = analyse_frequency_plateaus(
#     time_sec=common_time,
#     frequency_track=common_delta_track[common_valid],
#     sample_rate_hz=SAMPLE_RATE_HZ,
#     smooth_window_s=2.0,
#     min_plateau_duration_s=2.0,
#     plateau_mad_factor=0.75,
#     spike_mad_factor=6.0,
# )

# dfdt = common_result["df_dt"]

# plateau_mask = common_result["plateau_mask"]

# frequency = common_result["frequency_smoothed"]


# ============================================================
# Distribution histograms across detectors
# ============================================================

make_hist(
    mean_power_by_detector,
    "Mean detector power (pW)",
    f"Distribution of Mean Detector Power\n{BAND_LABEL} GHz",
    f"{run_prefix}_{BAND_LABEL}GHz_detector_mean_power_histogram.png",
)

make_hist(
    sigma_power_by_detector,
    "Gaussian sigma of detector power (pW)",
    f"Distribution of Detector Power Sigma\n{BAND_LABEL} GHz",
    f"{run_prefix}_{BAND_LABEL}GHz_detector_power_sigma_histogram.png",
)

make_hist(
    frac_width_by_detector,
    r"Fractional width $\sigma / \mu$",
    f"Distribution of Detector Power Fractional Width\n{BAND_LABEL} GHz",
    f"{run_prefix}_{BAND_LABEL}GHz_detector_fractional_width_histogram.png",
)

make_hist(
    ptp_delta_by_detector,
    r"Peak-to-peak $\Delta f / \mathrm{FWHM}$",
    f"Distribution of Peak-to-Peak Frequency Shifts\n{BAND_LABEL} GHz",
    f"{run_prefix}_{BAND_LABEL}GHz_detector_deltaf_fwhm_ptp_histogram.png",
)

make_hist(
    half_ptp_delta_by_detector,
    r"Half peak-to-peak $\Delta f / \mathrm{FWHM}$",
    f"Distribution of Half Peak-to-Peak Frequency Shifts\n{BAND_LABEL} GHz",
    f"{run_prefix}_{BAND_LABEL}GHz_detector_deltaf_fwhm_half_ptp_histogram.png",
)


# ============================================================
# All-sample delta f / FWHM histogram
# ============================================================

all_delta = np.concatenate(all_delta_tracks)

# ============================================================
# Array-wide frequency-feature diagnostic
# ============================================================

array_candidate_bins, array_counts, array_edges, array_excess = (
    candidate_spike_bins(
        all_delta,
        bins=100,
        max_candidates=3,
    )
)

array_feature_rows = []
array_geometry_rows = []

for feature_rank, bin_idx in enumerate(
    array_candidate_bins,
    start=1,
):
    lower = array_edges[bin_idx]
    upper = array_edges[bin_idx + 1]

    result = make_array_feature_diagnostic(
        delta_matrix=delta_matrix,
        time_sec=time_sec_full,
        lower=lower,
        upper=upper,
        feature_rank=feature_rank,
        outdir=OUTDIR,
        run_prefix=run_prefix,
        band_label=BAND_LABEL,
        scan_pattern=SCAN_PATTERN,
        elev_label=ELEV_LABEL,
        smooth_window_s=2.0,
    )

    if result is not None:
        array_feature_rows.append(result)

    geometry_result = make_feature_geometry_diagnostic(
        tod=tod,
        delta_matrix=delta_matrix,
        time_sec=time_sec_full,
        lower=lower,
        upper=upper,
        feature_rank=feature_rank,
        outdir=OUTDIR,
        run_prefix=run_prefix,
        band_label=BAND_LABEL,
        scan_pattern=SCAN_PATTERN,
        elev_label=ELEV_LABEL,
        occupancy_fraction_of_peak=0.5
    )

    if geometry_result is not None:
        array_geometry_rows.append(geometry_result)


if len(array_feature_rows) > 0:
    array_feature_df = pd.DataFrame(
        array_feature_rows
    )

    array_feature_csv = OUTDIR / (
        f"{run_prefix}_{BAND_LABEL}GHz_"
        "array_feature_diagnostic_summary.csv"
    )

    array_feature_df.to_csv(
        array_feature_csv,
        index=False,
    )

    print(
        f"\nSaved array-feature summary to: "
        f"{array_feature_csv}"
    )

if len(array_geometry_rows) > 0:
    array_geometry_df = pd.DataFrame(
        array_geometry_rows
    )

    array_geometry_csv = OUTDIR / (
        f"{run_prefix}_{BAND_LABEL}GHz_"
        "array_feature_geometry_summary.csv"
    )

    array_geometry_df.to_csv(
        array_geometry_csv,
        index=False,
    )

    print(
        f"\nSaved array-feature geometry summary to: "
        f"{array_geometry_csv}"
    )

stats_rows.append(
    summarize(
        f"All-detector Delta f / FWHM, {BAND_LABEL} GHz",
        all_delta,
        "dimensionless",
    )
)

make_hist(
    all_delta,
    r"$\Delta f / \mathrm{FWHM}$",
    (
        f"All-Detector Estimated Fractional Frequency Shift\n"
        f"{BAND_LABEL} GHz, PWV={PWV_MM:.2f} mm, Elev={ELEV_LABEL}"
    ),
    f"{run_prefix}_{BAND_LABEL}GHz_all_detector_deltaf_fwhm_histogram.png",
)


# ============================================================
# Diagnostic 1: standardized pooled distributions
# ============================================================

all_power_standardized = np.concatenate(standardized_power_tracks)
all_delta_standardized = np.concatenate(standardized_delta_tracks)

common_standardized_bins = np.linspace(-6, 6, 121)

plt.figure(figsize=(8, 6))
plt.hist(
    all_power_standardized,
    bins=common_standardized_bins,
    density=True,
    histtype="step",
    linewidth=1.8,
    label="Per-detector standardized power",
)
plt.hist(
    -all_delta_standardized,
    bins=common_standardized_bins,
    density=True,
    histtype="step",
    linewidth=1.2,
    linestyle="--",
    label=r"Reflected standardized $\Delta f/\mathrm{FWHM}$",
)
plt.xlabel("Standardized detector-relative sample")
plt.ylabel("Probability density")
plt.title(
    "Power–Frequency-Shift Shape Comparison\n"
    f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_standardized_power_delta_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# Diagnostic 2: detector index versus delta f / FWHM density
# ============================================================

finite_delta = delta_matrix[np.isfinite(delta_matrix)]
if len(finite_delta) > 0:
    delta_low, delta_high = np.nanpercentile(finite_delta, [0.5, 99.5])
    delta_edges = np.linspace(delta_low, delta_high, 121)

    detector_indices = np.repeat(
        np.arange(n_detectors),
        delta_matrix.shape[1],
    )
    delta_flat = delta_matrix.ravel()
    finite_flat = np.isfinite(delta_flat)

    plt.figure(figsize=(10, 7))
    plt.hist2d(
        detector_indices[finite_flat],
        delta_flat[finite_flat],
        bins=[min(120, n_detectors), delta_edges],
    )
    plt.xlabel("Detector index")
    plt.ylabel(r"$\Delta f / \mathrm{FWHM}$")
    plt.title(
        "Detector Contributions to Frequency-Shift Distribution\n"
        f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
    )
    plt.colorbar(label="Number of samples")
    plt.tight_layout()
    plt.savefig(
        OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_detector_deltaf_density.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ============================================================
# Diagnostic 3: Time vs detector index coloured with detector Power
# ============================================================

time_sec = np.arange(P_pW.shape[1]) / SAMPLE_RATE_HZ

plt.figure(figsize=(14, 7))

extent = [
    time_sec[0],
    time_sec[-1],
    0,
    n_detectors - 1,
]

plt.imshow(
    P_pW,
    origin="lower",
    aspect="auto",
    extent=extent,
    interpolation="nearest",
    cmap="coolwarm",
)

plt.xlabel("Time (s)")
plt.ylabel("Detector index")
plt.title(
    "Detector Power vs Time\n"
    f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
)

cbar = plt.colorbar()
cbar.set_label("Detector Power (pW)")

plt.tight_layout()
plt.savefig(
    OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_detector_power_vs_time.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# Diagnostic 4: Time vs detector index coloured with detector Power subtracting median
# ============================================================

time_sec = np.arange(P_pW.shape[1]) / SAMPLE_RATE_HZ

P_relative = P_pW - np.nanmedian(P_pW, axis=1, keepdims=True)

color_limit = np.nanstd(P_relative, axis=1, keepdims=True).max()

plt.figure(figsize=(14, 7))

extent = [
    time_sec[0],
    time_sec[-1],
    0,
    n_detectors - 1,
]

plt.imshow(
    P_relative,
    origin="lower",
    aspect="auto",
    extent=extent,
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-color_limit,
    vmax=color_limit
)

plt.xlabel("Time (s)")
plt.ylabel("Detector index")
plt.title(
    "Median Subtracted Detector Power vs Time\n"
    f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
)

cbar = plt.colorbar()
cbar.set_label("Detector Power (pW)")

plt.tight_layout()
plt.savefig(
    OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_median_subtracted_detector_power_vs_time.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# Diagnostic 4.5: Time vs detector index coloured with detector Power subtracting median
# ============================================================

time_sec = np.arange(P_pW.shape[1]) / SAMPLE_RATE_HZ

P_detector_sub = P_pW - np.nanmedian(P_pW, axis=1, keepdims=True)

P_small_scale = P_detector_sub - np.nanmedian(P_detector_sub, axis=0, keepdims=True)

color_limit = float(
    np.nanpercentile(np.abs(P_small_scale), 99.5)
)

plt.figure(figsize=(14, 7))

extent = [
    time_sec[0],
    time_sec[-1],
    0,
    n_detectors - 1,
]

plt.imshow(
    P_small_scale,
    origin="lower",
    aspect="auto",
    extent=extent,
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-color_limit,
    vmax=color_limit
)

plt.xlabel("Time (s)")
plt.ylabel("Detector index")
plt.title(
    "Small Scale Median Subtracted Detector Power vs Time\n"
    f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
)

cbar = plt.colorbar()
cbar.set_label("Detector Power (pW)")

plt.tight_layout()
plt.savefig(
    OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_small_scale_median_subtracted_detector_power_vs_time.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# Detector index vs detector elevation coloured by
# median-subtracted detector power
# ============================================================

n_detectors, n_times = P_pW.shape

# Each detector's elevation at every time sample.
elevation_deg_matrix = np.asarray(
    el_deg_matrix,
    dtype=np.float64,
)

if elevation_deg_matrix.shape != P_pW.shape:
    raise ValueError(
        "el_deg_matrix and P_pW must have the same shape. "
        f"Received {elevation_deg_matrix.shape} and {P_pW.shape}."
    )

# Detector index for every detector-time sample.
detector_grid = np.broadcast_to(
    np.arange(n_detectors)[:, np.newaxis],
    P_pW.shape,
)

# ------------------------------------------------------------
# Plot 1: Detector-median-subtracted power
# ------------------------------------------------------------

P_colour = (
    P_pW
    - np.nanmedian(P_pW, axis=1, keepdims=True)
)

x = elevation_deg_matrix.ravel()
y = detector_grid.ravel()
colour = P_colour.ravel()

valid = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(colour)
)

x_valid = x[valid]
y_valid = y[valid]
colour_valid = colour[valid]

color_limit = float(
    np.nanpercentile(np.abs(colour_valid), 99.5)
)

if not np.isfinite(color_limit) or color_limit <= 0:
    color_limit = 1.0

plt.figure(figsize=(12, 7))

scatter = plt.scatter(
    x_valid,
    y_valid,
    c=colour_valid,
    cmap="coolwarm",
    s=1.0,
    marker=".",
    linewidths=0,
    rasterized=True,
    vmin=-color_limit,
    vmax=color_limit,
)

plt.xlabel("Detector Elevation (deg)")
plt.ylabel("Detector Index")
plt.title(
    "Median-Subtracted Detector Power vs Elevation\n"
    f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
)

cbar = plt.colorbar(scatter)
cbar.set_label("Median-Subtracted Detector Power (pW)")

plt.tight_layout()

plt.savefig(
    OUTDIR
    / (
        f"{run_prefix}_{BAND_LABEL}GHz_"
        "median_subtracted_detector_power_vs_elevation.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Detector index vs detector elevation coloured by
# small-scale detector power residual
# ============================================================

# Step 1: Remove each detector's median over time.
P_detector_sub = (
    P_pW
    - np.nanmedian(P_pW, axis=1, keepdims=True)
)

# Step 2: Remove the instantaneous array median.
P_small_scale = (
    P_detector_sub
    - np.nanmedian(
        P_detector_sub,
        axis=0,
        keepdims=True,
    )
)

x = elevation_deg_matrix.ravel()
y = detector_grid.ravel()
colour = P_small_scale.ravel()

valid = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(colour)
)

x_valid = x[valid]
y_valid = y[valid]
colour_valid = colour[valid]

color_limit = float(
    np.nanpercentile(np.abs(colour_valid), 99.5)
)

if not np.isfinite(color_limit) or color_limit <= 0:
    color_limit = 1.0

plt.figure(figsize=(12, 7))

scatter = plt.scatter(
    x_valid,
    y_valid,
    c=colour_valid,
    cmap="coolwarm",
    s=1.0,
    marker=".",
    linewidths=0,
    rasterized=True,
    vmin=-color_limit,
    vmax=color_limit,
)

plt.xlabel("Detector Elevation (deg)")
plt.ylabel("Detector Index")
plt.title(
    "Small-Scale Detector Power Residual vs Elevation\n"
    f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
)

cbar = plt.colorbar(scatter)
cbar.set_label(
    "Detector- and Frame-Median-Subtracted Power (pW)"
)

plt.tight_layout()

plt.savefig(
    OUTDIR
    / (
        f"{run_prefix}_{BAND_LABEL}GHz_"
        "small_scale_detector_power_vs_elevation.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ============================================================
# Diagnostic 5: Time vs detector index coloured with detector Power instantaneous subtracting median
# ============================================================

time_sec = np.arange(P_pW.shape[1]) / SAMPLE_RATE_HZ

P_relative = P_pW - np.nanmedian(P_pW, axis=0, keepdims=True)

color_limit = np.nanstd(P_relative, axis=1, keepdims=True).max()

plt.figure(figsize=(14, 7))

extent = [
    time_sec[0],
    time_sec[-1],
    0,
    n_detectors - 1,
]

plt.imshow(
    P_relative,
    origin="lower",
    aspect="auto",
    extent=extent,
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-color_limit,
    vmax=color_limit
)

plt.xlabel("Time (s)")
plt.ylabel("Detector index")
plt.title(
    "Instantaneous Median Subtracted Detector Power vs Time\n"
    f"{BAND_LABEL} GHz, {SCAN_PATTERN}, {ELEV_LABEL}"
)

cbar = plt.colorbar()
cbar.set_label("Detector Power (pW)")

plt.tight_layout()
plt.savefig(
    OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_inst_median_subtracted_detector_power_vs_time.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ============================================================
# Atmospheric loading, elevation, airmass, and PWV tests
# ============================================================

atmospheric_metrics, atmospheric_summary = (
    run_atmospheric_power_tests(
        power_pW=P_pW,
        elevation_deg_matrix=el_deg_matrix,
        time_sec=time_sec_full,
        pwv_mm=PWV_MM,
        outdir=OUTDIR,
        comparison_directory=ATMOSPHERE_TEST_ROOT,
        run_prefix=run_prefix,
        band_label=BAND_LABEL,
        scan_pattern=SCAN_PATTERN,
        elevation_label=ELEV_LABEL,
        elevation_bin_width_deg=ELEVATION_BIN_WIDTH_DEG,
        airmass_bin_width=AIRMass_BIN_WIDTH,
    )
)

# ============================================================
# Transmission-derived optical-depth analysis
# ============================================================


array_elevation_deg = np.nanmedian(el_deg_matrix, axis=0)

reference_elevation_deg = float(np.nanmedian(array_elevation_deg))

maria_opacity = get_tau0_from_maria_transmission(
    band=band,
    pwv_mm=PWV_MM,
    reference_elevation_deg=reference_elevation_deg,
    site_altitude_m= 5600.0,
)

tau_0_maria = maria_opacity["tau_0"]

optical_depth_summary = analyse_power_against_optical_depth(
    power_pW=P_pW,
    elevation_deg_matrix=el_deg_matrix,
    time_sec=time_sec_full,
    pwv_mm=PWV_MM,
    tau_0=tau_0_maria,
    outdir=OUTDIR,
    run_prefix=run_prefix,
    band_label=BAND_LABEL,
)

# ============================================================
# Individual detector plots
# ============================================================

time_sec_full = np.arange(P_pW.shape[1]) / SAMPLE_RATE_HZ

for det_idx in DETECTORS_TO_PLOT:

    if det_idx >= P_pW.shape[0]:
        print(f"Detector {det_idx} does not exist; skipping.")
        continue

    P_track = np.asarray(P_pW[det_idx, :], dtype=np.float64)
    valid = np.isfinite(P_track)

    P_track = P_track[valid]
    time_sec = time_sec_full[valid]

    delta_track = delta_f_over_fwhm(P_track)

    # ========================================================
    # Direct shape check for this detector
    # ========================================================


    power_standardized = standardize(P_track)
    delta_standardized = standardize(delta_track)

    if len(power_standardized) > 0 and len(delta_standardized) > 0:
        comparison_bins = np.linspace(-6, 6, 121)

        plt.figure(figsize=(8, 6))
        plt.hist(
            power_standardized,
            bins=comparison_bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Standardized power",
        )
        plt.hist(
            -delta_standardized,
            bins=comparison_bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            linestyle="--",
            label=r"Reflected standardized $\Delta f/\mathrm{FWHM}$",
        )
        plt.xlabel("Standardized value")
        plt.ylabel("Probability density")
        plt.title(
            f"Detector {det_idx}: Power–Frequency-Shift Shape Check\n"
            f"{BAND_LABEL} GHz, {SCAN_PATTERN}"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            OUTDIR / (
                f"{run_prefix}_{BAND_LABEL}GHz_detector_{det_idx}_"
                "power_delta_shape_comparison.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    make_hist(
        P_track,
        "Detector power (pW)",
        (
            f"Detector {det_idx} Power Distribution\n"
            f"{BAND_LABEL} GHz, PWV={PWV_MM:.2f} mm"
        ),
        f"{run_prefix}_{BAND_LABEL}GHz_detector_{det_idx}_power_histogram.png",
    )

    make_hist(
        delta_track,
        r"$\Delta f / \mathrm{FWHM}$",
        (
            f"Detector {det_idx} Estimated Fractional Frequency Shift\n"
            f"{BAND_LABEL} GHz, PWV={PWV_MM:.2f} mm"
        ),
        f"{run_prefix}_{BAND_LABEL}GHz_detector_{det_idx}_deltaf_fwhm_histogram.png",
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        time_sec,
        delta_track,
        lw=0.5,
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel(r"$\Delta f / \mathrm{FWHM}$")
    plt.title(
        f"Detector {det_idx} Delta f / FWHM vs Time\n"
        f"{BAND_LABEL} GHz, PWV={PWV_MM:.2f} mm"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_detector_{det_idx}_deltaf_fwhm_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Candidate spike locations and recurrence diagnostics
    # ========================================================
    candidate_bins, counts, edges, excess = candidate_spike_bins(
        delta_track,
        bins=100,
        max_candidates=3,
    )

    for rank, bin_idx in enumerate(candidate_bins, start=1):
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]
        spike_mask = (delta_track >= lower) & (delta_track < upper)

        event_start_indices = contiguous_event_starts(spike_mask)
        event_times = time_sec[event_start_indices]
        event_separations = np.diff(event_times)

        print(
            f"Detector {det_idx}, candidate spike {rank}: "
            f"[{lower:.6g}, {upper:.6g}), "
            f"{spike_mask.sum()} samples in {len(event_times)} events"
        )

        plt.figure(figsize=(11, 5))
        plt.plot(time_sec, delta_track, linewidth=0.5, label="All samples")
        plt.scatter(
            time_sec[spike_mask],
            delta_track[spike_mask],
            s=8,
            label=f"Candidate spike {rank}",
        )
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\Delta f / \mathrm{FWHM}$")
        plt.title(
            f"Detector {det_idx}: Candidate Histogram Feature {rank}\n"
            f"{lower:.4g} to {upper:.4g}"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            OUTDIR / (
                f"{run_prefix}_{BAND_LABEL}GHz_detector_{det_idx}_"
                f"candidate_spike_{rank}_vs_time.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        if len(event_separations) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(event_separations, bins=50, edgecolor="black")
            plt.xlabel("Time between feature events (s)")
            plt.ylabel("Number of event pairs")
            plt.title(
                f"Detector {det_idx}: Recurrence of Candidate Feature {rank}\n"
                f"Median interval = {np.nanmedian(event_separations):.3f} s"
            )
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                OUTDIR / (
                    f"{run_prefix}_{BAND_LABEL}GHz_detector_{det_idx}_"
                    f"candidate_spike_{rank}_event_separations.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


# ============================================================
# Save summary statistics
# ============================================================

summary_df = pd.DataFrame(stats_rows)
summary_csv_path = OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_summary_stats.csv"
summary_df.to_csv(summary_csv_path, index=False)

print(f"\nSaved summary statistics CSV to: {summary_csv_path}")
print(f"Saved plots to: {OUTDIR}")


animation_dir = OUTDIR / "detector_azel_animations"

os.makedirs(animation_dir, exist_ok=True)

power_matrix = tod.to("pW").signal          # (Ndet, Ntime)

# el_deg_matrix = np.rad2deg(tod.el)          # (Ndet, Ntime)
# az_deg_matrix = np.cos(el_deg_matrix) * np.rad2deg(tod.az)          # (Ndet, Ntime)

# raise SystemExit("Animation Generation Disabled")
animate_detector_azel_power(
    az_deg=az_deg_matrix,
    el_deg=el_deg_matrix,
    power_pW=P_pW,
    time_s=time_sec_full,
    output_path=(
        animation_dir
        / f"{BAND_LABEL}GHz_detector_azel_power_animation.mp4"
    ),
    colour_mode="absolute",
    frame_step=10,
    fps=20,
    marker_size=10.0,
    title=(
        f"Prime-Cam Detector Loading in Az-El\n"
        f"{BAND_LABEL} GHz, {SCAN_PATTERN}, Elev={ELEV_LABEL}"
    ),
)

animate_detector_azel_power(
    az_deg=az_deg_matrix,
    el_deg=el_deg_matrix,
    power_pW=P_pW,
    time_s=time_sec_full,
    output_path=(
        animation_dir
        / f"{run_prefix}_{BAND_LABEL}GHz_azel_detector_median_subtracted_power.mp4"
    ),
    colour_mode="detector_median_subtracted",
    frame_step=10,
    fps=20,
    marker_size=18,
    title=(
        f"Prime-Cam Large-Scale Loading Changes in Az–El\n"
        f"{BAND_LABEL} GHz, {SCAN_PATTERN}, Elev={ELEV_LABEL}"
    ),
)

animate_detector_azel_power(
    az_deg=az_deg_matrix,
    el_deg=el_deg_matrix,
    power_pW=P_pW,
    time_s=time_sec_full,
    output_path=(
        animation_dir
        / f"{run_prefix}_{BAND_LABEL}GHz_azel_small_scale_residuals.mp4"
    ),
    colour_mode="frame_median_subtracted",
    frame_step=10,
    fps=20,
    marker_size=18,
    title=(
        f"Prime-Cam Small-Scale Loading Residuals in Az–El\n"
        f"{BAND_LABEL} GHz, {SCAN_PATTERN}, Elev={ELEV_LABEL}"
    ),
)


# ============================================================
# Full atmospheric parameter sweep
# ============================================================

BANDS_TO_TEST = ["280", "350", "850"]
PWVS_TO_TEST = [0.36, 0.67, 1.28]

ELEVATION_RANGES_TO_TEST = [
    (45, 55),
    (55, 65),
    (65, 75),
]

SCAN_PATTERNS_TO_TEST = [
    "lissajous",
    "raster",
    "back_and_forth",
    "daisy",
    "double_circle",
    "pong",
]

SWEEP_SPEED = 0.1
SWEEP_DURATION_S = 900
SWEEP_SAMPLE_RATE_HZ = 10


LINEWIDTH_LIMIT = 0.10

# Physical width of the requested target map.
# Change this when running the 0.5, 1.5, or 3.0 degree maps.
SWEEP_MAP_SIZE_DEG = 0.5

# Angular size of one coverage-map pixel.
# This should remain fixed when comparing scan patterns.
COVERAGE_PIXEL_SIZE_DEG = 0.02

MIN_HITS_FOR_COVERAGE = 1
MIN_HITS_FOR_REVISIT = 2


SWEEP_ROOT = Path(
    "outputs/atmospheric_parameter_sweep"
)

SWEEP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

COMBINED_CSV_PATH = (
    SWEEP_ROOT
    / "all_band_pwv_elevation_summary.csv"
)

all_results = []

site = maria.get_site(
    "cerro_chajnantor",
    altitude=5600,
)

for scan in SCAN_PATTERNS_TO_TEST:

    for band_label in BANDS_TO_TEST:

        maria_band = make_band_for_sweep(
            band_label=band_label,
            efficiency=eta,
        )

        for pwv_mm in PWVS_TO_TEST:

            for elev_limits in ELEVATION_RANGES_TO_TEST:

                elev_label = (
                    f"{elev_limits[0]}-"
                    f"{elev_limits[1]}"
                )

                pwv_tag = (
                    f"{pwv_mm:.2f}"
                    .replace(".", "p")
                )

                speed_tag = (
                    f"{SWEEP_SPEED:.1f}"
                    .replace(".", "p")
                )

                run_prefix = (
                    f"OrionA_{scan}_"
                    f"{elev_label}_"
                    f"speed_{speed_tag}_"
                    f"PWV_{pwv_tag}mm_"
                    f"small_map"
                )

                print("\n" + "=" * 70)
                print(
                    "Running atmospheric sweep:"
                )
                print(
                    f"Band={band_label} GHz, "
                    f"PWV={pwv_mm:.2f} mm, "
                    f"Elevation={elev_label}, "
                    f"Scan={scan}"
                )
                print("=" * 70)

                # --------------------------------------------
                # Paths for this individual simulation
                # --------------------------------------------

                tod_outdir = Path(
                    f"outputs/{run_prefix}_tods"
                )

                fits_path = (
                    tod_outdir
                    / (
                        f"{run_prefix}_"
                        "dim_reduced_tods.fits"
                    )
                )

                run_outdir = (
                    SWEEP_ROOT
                    / f"{band_label}GHz"
                    / scan
                    / (
                        f"PWV_{pwv_tag}mm_"
                        f"elev_{elev_label}"
                    )
                )

                run_outdir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                comparison_directory = (
                    SWEEP_ROOT
                    / f"{band_label}GHz"
                    / scan
                    / "combined_comparisons"
                )

                comparison_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # --------------------------------------------
                # Run MARIA simulation
                # --------------------------------------------

                simple_ccat.tod_analysis(
                    PREFIX=run_prefix,
                    tod_diagnostics=False,
                    maps=False,
                    save_all_plots=True,
                    run_mode="fits",
                    atm_plot=False,
                    temp_mode="inst",
                    ccat_band=band_label,
                    map_type="BM",
                    pwv_mm=pwv_mm,
                    start_time=START_TIME,
                    total_duration_s=(
                        SWEEP_DURATION_S
                    ),
                    sim_duration_s=(
                        SWEEP_DURATION_S
                    ),
                    sample_rate_hz=(
                        SWEEP_SAMPLE_RATE_HZ
                    ),
                    scan_pattern=scan,
                    el_limits=elev_limits,
                    speed=SWEEP_SPEED,
                )

                if not fits_path.exists():
                    raise FileNotFoundError(
                        "The expected TOD file was "
                        "not created:\n"
                        f"{fits_path}"
                    )

                # --------------------------------------------
                # Load the TOD produced by simple_ccat
                # --------------------------------------------

                tod = maria.tod.load(
                    fits_path,
                    site=site,
                    bands=[maria_band],
                )

                power_pW = np.asarray(
                    tod.to("pW").signal,
                    dtype=np.float64,
                )

                elevation_raw = np.squeeze(
                    np.asarray(
                        tod.el,
                        dtype=np.float64,
                    )
                )

                # Match the detector-by-time power shape.
                if elevation_raw.shape == power_pW.shape:
                    elevation_matrix = elevation_raw

                elif (
                    elevation_raw.T.shape
                    == power_pW.shape
                ):
                    elevation_matrix = elevation_raw.T

                else:
                    raise ValueError(
                        "Could not match the elevation "
                        "array to the power array. "
                        f"Elevation shape: "
                        f"{elevation_raw.shape}; "
                        f"power shape: "
                        f"{power_pW.shape}"
                    )

                elevation_deg_matrix = np.rad2deg(
                    elevation_matrix
                )

                time_sec = (
                    np.arange(
                        power_pW.shape[1],
                        dtype=np.float64,
                    )
                    / SWEEP_SAMPLE_RATE_HZ
                )

                # --------------------------------------------
                # Fixed-tone detector-linewidth metrics
                # --------------------------------------------

                (
                    linewidth_metrics,
                    delta_f_fwhm_matrix,
                ) = calculate_linewidth_feasibility_metrics(
                    power_pW=power_pW,
                    band_label=band_label,
                    linewidth_limit=LINEWIDTH_LIMIT,
                )

                print("\nDetector-linewidth suitability")
                print("-" * 50)
                print(
                    "Fraction of all samples within limit: "
                    f"{linewidth_metrics['fraction_within_linewidth_limit']:.4%}"
                )
                print(
                    "95th percentile absolute excursion: "
                    f"{linewidth_metrics['p95_abs_delta_f_over_fwhm']:.6g}"
                )
                print(
                    "Maximum absolute excursion: "
                    f"{linewidth_metrics['max_abs_delta_f_over_fwhm']:.6g}"
                )
                print(
                    "Fraction of detectors fully within limit: "
                    f"{linewidth_metrics['fraction_detectors_fully_within_limit']:.4%}"
                )


                # --------------------------------------------
                # Focal-plane coverage metrics
                # --------------------------------------------

                (
                    coverage_metrics,
                    coverage_hit_map,
                    coverage_x_edges,
                    coverage_y_edges,
                ) = calculate_sky_coverage_metrics(
                    tod=tod,
                    power_shape=power_pW.shape,
                    map_size_deg=SWEEP_MAP_SIZE_DEG,
                    pixel_size_deg=COVERAGE_PIXEL_SIZE_DEG,
                    minimum_hits_for_coverage=(
                        MIN_HITS_FOR_COVERAGE
                    ),
                    minimum_hits_for_revisit=(
                        MIN_HITS_FOR_REVISIT
                    ),
                )

                coverage_plot_path = (
                    run_outdir
                    / (
                        f"{run_prefix}_{band_label}GHz_"
                        "focal_plane_coverage_hit_map.png"
                    )
                )

                save_coverage_hit_map(
                    hit_map=coverage_hit_map,
                    x_edges=coverage_x_edges,
                    y_edges=coverage_y_edges,
                    output_path=coverage_plot_path,
                    title=(
                        "Prime-Cam Focal-Plane Coverage\n"
                        f"{band_label} GHz, {scan}, "
                        f"Elevation={elev_label}, "
                        f"Duration={SWEEP_DURATION_S} s"
                    ),
                )

                print("\nCoverage efficiency")
                print("-" * 50)
                print(
                    "Coverage fraction: "
                    f"{coverage_metrics['coverage_fraction']:.4%}"
                )
                print(
                    "Revisit fraction: "
                    f"{coverage_metrics['revisit_fraction']:.4%}"
                )
                print(
                    "Hit-count coefficient of variation: "
                    f"{coverage_metrics['hit_count_cv']:.6g}"
                )

                # --------------------------------------------
                # Elevation, airmass, and small-scale analysis
                # --------------------------------------------

                (
                    atmospheric_metrics,
                    atmospheric_summary_df,
                ) = run_atmospheric_power_tests(
                    power_pW=power_pW,
                    elevation_deg_matrix=(
                        elevation_deg_matrix
                    ),
                    time_sec=time_sec,
                    pwv_mm=pwv_mm,
                    outdir=run_outdir,
                    comparison_directory=(
                        comparison_directory
                    ),
                    run_prefix=run_prefix,
                    band_label=band_label,
                    scan_pattern=scan,
                    elevation_label=elev_label,
                    elevation_bin_width_deg=(
                        ELEVATION_BIN_WIDTH_DEG
                    ),
                    airmass_bin_width=(
                        AIRMass_BIN_WIDTH
                    ),
                )

                # --------------------------------------------
                # Transmission-derived tau_0
                # --------------------------------------------

                array_elevation_deg = np.nanmedian(
                    elevation_deg_matrix,
                    axis=0,
                )

                reference_elevation_deg = float(
                    np.nanmedian(
                        array_elevation_deg
                    )
                )

                maria_opacity = (
                    get_tau0_from_maria_transmission(
                        band=maria_band,
                        pwv_mm=pwv_mm,
                        reference_elevation_deg=(
                            reference_elevation_deg
                        ),
                        site_altitude_m=5600.0,
                    )
                )

                # --------------------------------------------
                # Optical-depth analysis
                # --------------------------------------------

                optical_summary = (
                    analyse_power_against_optical_depth(
                        power_pW=power_pW,
                        elevation_deg_matrix=(
                            elevation_deg_matrix
                        ),
                        time_sec=time_sec,
                        pwv_mm=pwv_mm,
                        tau_0=(
                            maria_opacity["tau_0"]
                        ),
                        outdir=run_outdir,
                        run_prefix=run_prefix,
                        band_label=band_label,
                    )
                )

                # --------------------------------------------
                # Combine all information into one row
                # --------------------------------------------

                atmospheric_row = (
                    atmospheric_summary_df
                    .iloc[0]
                    .to_dict()
                )

                combined_row = {
                    "run_prefix": run_prefix,
                    "band_ghz": band_label,
                    "pwv_mm": pwv_mm,
                    "scan_pattern": scan,
                    "elevation_min_deg": elev_limits[0],
                    "elevation_max_deg": elev_limits[1],
                    "mean_elevation_deg": float(
                        np.nanmean(
                            array_elevation_deg
                        )
                    ),
                    "elevation_label": elev_label,

                    "input_speed_deg_s": SWEEP_SPEED,
                    "map_size_deg": SWEEP_MAP_SIZE_DEG,
                    "duration_s": SWEEP_DURATION_S,
                    "sample_rate_hz": SWEEP_SAMPLE_RATE_HZ,
                    "tod_path": str(fits_path),
                }

                combined_row.update(linewidth_metrics)

                combined_row.update(coverage_metrics)

                # Add every value returned by the
                # atmospheric-power analysis.
                for key, value in (
                    atmospheric_row.items()
                ):
                    combined_row[
                        f"atmospheric_{key}"
                    ] = value

                # Add every value returned by the
                # optical-depth analysis.
                for key, value in (
                    optical_summary.items()
                ):
                    combined_row[
                        f"optical_{key}"
                    ] = value

                # Add values returned directly by the
                # MARIA transmission helper.
                for key, value in (
                    maria_opacity.items()
                ):
                    combined_row[
                        f"transmission_{key}"
                    ] = value

                all_results.append(
                    combined_row
                )

                # --------------------------------------------
                # Save a checkpoint after every run
                # --------------------------------------------

                combined_summary_df = pd.DataFrame(
                    all_results
                )

                combined_summary_df.to_csv(
                    COMBINED_CSV_PATH,
                    index=False,
                )

                print(
                    "Updated combined CSV: "
                    f"{COMBINED_CSV_PATH}"
                )
                print(
                    "Completed rows: "
                    f"{len(all_results)}"
                )

# ============================================================
# Final combined parameter-sweep summary
# ============================================================

combined_summary_df = pd.DataFrame(
    all_results
)

combined_summary_df = (
    combined_summary_df
    .sort_values(
        by=[
            "band_ghz",
            "pwv_mm",
            "elevation_min_deg",
            "scan_pattern",
        ]
    )
    .reset_index(drop=True)
)

combined_summary_df.to_csv(
    COMBINED_CSV_PATH,
    index=False,
)

print("\nAtmospheric parameter sweep complete.")
print(
    f"Number of completed simulations: "
    f"{len(combined_summary_df)}"
)
print(
    f"Combined summary saved to: "
    f"{COMBINED_CSV_PATH}"
)