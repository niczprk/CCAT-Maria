from pathlib import Path
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

from scipy.stats import norm, skew
from scipy.signal import find_peaks

import maria
from maria.instrument import Band

import simple_ccat

# ============================================================
# User settings
# ============================================================

ccat_band = "850"  # "850" or "350"

Run = True # whether to run the TOD analysis or just load existing TOD

eta = 0.5 # optical efficiency for this estimate

Polarized = False # whether to include polarization in the simulation

# 0.36, 0.67, & 1.28 are Q1, Q2, and Q3 zenith PMV values for Chajnantor

EL_LIMITS = (65, 75)  # degrees

START_TIME = "2022-02-10T17:00:00"

#scan pattern options: "lissajous", "raster", "back_and_forth", "daisy", "double_circle", "stare"

SCAN_PATTERN = "double_circle"

ELEV_LABEL = "65-75"

SPEED = 0.1

eta = 0.5

# "2022-02-10T22:45:00" for around 75 degrees ?
#"2022-02-10T20:30:00" for around 60 degrees 
#"2022-02-10T18:55:00" for roughly 45 degrees
#"2022-02-10T18:30:00" for roughly 40 degrees
#"2022-02-10T17:00:00" for roughly 30 degrees


 

TOTAL_DURATION_S = 900  # seconds
SIM_DURATION_S = 900  # seconds
CHUNK_NUMBER = 0


# RUN_SPEED_GRID = True
# COMBINE_EXISTING_CSVS = False
# SPEED_CSV_DIR = Path(f"outputs/OrionA_{SCAN_PATTERN}_speed_tests/speed_csv")
# COMBINED_PLOT_DIR = Path(f"outputs/OrionA_{SCAN_PATTERN}_speed_tests/combined_plots")
# SPEED_CSV_DIR.mkdir(parents=True, exist_ok=True)
# COMBINED_PLOT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE_HZ = 10

PWV_MM = 0.36

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

run_prefix = (
    f"OrionA_{SCAN_PATTERN.lower()}_{ELEV_LABEL}_speed_{SPEED:.1f}_small_map"
    .replace(".", "p")
)

TOD_OUTDIR = Path(f"outputs/{run_prefix}_tods") #Im going to have to change this for each run I am interested in
fits_path = TOD_OUTDIR / f"{run_prefix}_dim_reduced_tods.fits"

OUTDIR = Path(f"outputs/delta_f_analysis/{SCAN_PATTERN}/{run_prefix}_{BAND_LABEL}GHz_power_deltaf_analysis")
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
    P_track_pW = np.asarray(P_track_pW, dtype=np.float64)
    valid = np.isfinite(P_track_pW)

    P_track_pW = P_track_pW[valid]
    P_track_W = P_track_pW * 1e-12

    P_ref_W = np.nanmean(P_track_W)

    return Q_r * R(P_ref_W) * (P_track_W - P_ref_W)


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

    # Shade times at which many detectors occupy the feature.
    if np.any(high_occupancy_mask):
        frequency_axis.fill_between(
            diagnostic_time,
            frequency_axis.get_ylim()[0],
            frequency_axis.get_ylim()[1],
            where=high_occupancy_mask,
            alpha=0.08,
            transform=frequency_axis.get_xaxis_transform(),
            label="High array occupancy",
        )

    frequency_axis.set_ylabel(
        r"Array-median $\Delta f/\mathrm{FWHM}$"
    )

    frequency_axis.set_title(
        f"Array-Wide Histogram Feature {feature_rank} Diagnostic\n"
        f"{band_label} GHz, {scan_pattern}, Elev={elev_label}, "
        f"Feature={lower:.4g} to {upper:.4g}"
    )

    frequency_axis.grid(alpha=0.3)
    frequency_axis.legend(loc="best")

    # Derivative
    derivative_axis.plot(
        diagnostic_time,
        common_df_dt,
        linewidth=0.7,
    )

    derivative_axis.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    if np.any(high_occupancy_mask):
        derivative_axis.scatter(
            diagnostic_time[high_occupancy_mask],
            common_df_dt[high_occupancy_mask],
            s=8,
            label="High feature occupancy",
            zorder=4,
        )

    derivative_axis.set_ylabel(
        r"$d(\Delta f/\mathrm{FWHM})/dt$"
        "\n"
        r"$(\mathrm{s}^{-1})$"
    )

    derivative_axis.grid(alpha=0.3)

    if np.any(high_occupancy_mask):
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

    occupancy_axis.set_ylim(
        bottom=0,
        top=max(
            1.05 * peak_occupancy
            if np.isfinite(peak_occupancy)
            else 1.0,
            0.05,
        ),
    )

    occupancy_axis.grid(alpha=0.3)
    occupancy_axis.legend(loc="best")

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