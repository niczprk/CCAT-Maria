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

import maria
from maria.instrument import Band

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

SCAN_PATTERN = "daisy"

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

DETECTORS_TO_PLOT = [0, 50, 309, 339, 472, 604, 843]

run_prefix = (
    f"OrionA_{SCAN_PATTERN.lower()}_{ELEV_LABEL}_speed_{SPEED:.1f}"
    .replace(".", "p")
)

TOD_OUTDIR = Path(f"outputs/{run_prefix}_tods") #Im going to have to change this for each run I am interested in
fits_path = TOD_OUTDIR / f"{run_prefix}_dim_reduced_tods.fits"

OUTDIR = Path(f"outputs/{run_prefix}_{BAND_LABEL}GHz_power_deltaf_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load TOD
# ============================================================

if Run == True:
    simple_ccat.tod_analysis(
        PREFIX=run_prefix,
        tod_diagnostics=False,
        maps=False,
        save_all_plots=False,
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


def make_hist(values, xlabel, title, filename, bins=50):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    plt.figure(figsize=(8, 6))

    plt.hist(
        values,
        bins=bins,
        edgecolor="black",
        alpha=0.8,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


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

n_detectors = P_pW.shape[0]

for det_idx in range(n_detectors):

    P_track = np.asarray(P_pW[det_idx, :], dtype=np.float64)
    P_track = P_track[np.isfinite(P_track)]

    if len(P_track) == 0:
        continue

    mu, sigma = norm.fit(P_track)
    frac_width = sigma / mu if mu != 0 else np.nan

    delta_track = delta_f_over_fwhm(P_track)

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


# ============================================================
# Save summary statistics
# ============================================================

summary_df = pd.DataFrame(stats_rows)
summary_csv_path = OUTDIR / f"{run_prefix}_{BAND_LABEL}GHz_summary_stats.csv"
summary_df.to_csv(summary_csv_path, index=False)

print(f"\nSaved summary statistics CSV to: {summary_csv_path}")
print(f"Saved plots to: {OUTDIR}")