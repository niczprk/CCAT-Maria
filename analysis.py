from pathlib import Path

import simple_ccat
from maria.tod import TOD

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt


selected_band = "280" #make sure these match

NU_HZ = 280e9  # Hz
bandwidth_hz = 60e9  # GHz bandwidth for 850 GHz band
eta = 0.5 # optical efficiency for this estimate

Polarized = False # whether to include polarization in the simulation

NU_GHZ = NU_HZ / 1e9 #GHz

PWV_MM = 0.36  #  mm, precip water vapour this only affects the main if pwv is None

# 0.36, 0.67, & 1.28 are Q1, Q2, and Q3 zenith PMV values for Chajnantor

EL_LIMITS = (68, 85)  # degrees

T_0 = 278.868 #K, atmospheric ground temp



START_TIME = "2022-02-10T23:05:00"

# "2022-02-10T22:45:00" for around 75 degrees ?
#"2022-02-10T20:30:00" for around 60 degrees 
#"2022-02-10T18:55:00" for roughly 45 degrees
#"2022-02-10T18:30:00" for roughly 40 degrees
#"2022-02-10T17:00:00" for roughly 30 degrees
 

TOTAL_DURATION_S = 1800  # seconds
SIM_DURATION_S = 1800  # seconds
SAMPLE_RATE_HZ = 50  # Hz
SCAN_PATTERN = "daisy"
CHUNK_NUMBER = 0

PREFIX = "OrionA_tod_test"
ANALYSIS_OUTDIR = Path(f"outputs/{PREFIX}_analysis_outputs")
TOD_OUTDIR = Path(f"outputs/{PREFIX}_tod_files")




# simple_ccat.tod_analysis(
#     tod_diagnostics=False,
#     maps = False,
#     save_all_plots = True,
#     run_mode = "fits",
#     atm_plot = True,
#     temp_mode = "inst",
#     ccat_band = "280",
#     map_type = "BM",
#     pwv_mm = PWV_MM,
#     start_time = START_TIME,
#     total_duration_s = TOTAL_DURATION_S,
#     sim_duration_s = SIM_DURATION_S,
#     sample_rate_hz = SAMPLE_RATE_HZ,
# )

if __name__ == "__main__":


    tod = TOD.from_fits(TOD_OUTDIR / f"{PREFIX}_dim_expanded_tods.fits", format = "Mustang-2")

    print(tod)
    print(tod.shape)
    print(tod.fields)

    print("signal type:", type(tod.signal))
    print("signal shape:", tod.signal.shape)

    print("ra shape:", np.shape(tod.ra))
    print("dec shape:", np.shape(tod.dec))
    print("el shape:", np.shape(tod.el))
    print("az shape:", np.shape(tod.az))
    print("time shape:", np.shape(tod.time))

    P_det = tod.to("pW").signal
    P = np.asarray(P_det, dtype=np.float64)  # Convert to numpy array for calculations


    print("P shape:", P.shape)

    P_mean = np.nanmean(P, axis=1).ravel()

    time_idx = int(1*SAMPLE_RATE_HZ)  # Index for 1 second into the TOD

    ra_det = np.asarray(tod.ra[:, time_idx], dtype=np.float64)
    dec_det = np.asarray(tod.dec[:, time_idx], dtype=np.float64)

    plt.figure(figsize=(8,6))

    sc = plt.scatter(
        ra_det,
        dec_det,
        c=P_mean,
        cmap="viridis",
        s=50,
        edgecolor="k",
        alpha=0.7
    )
    plt.title(
        f"Detector Locations Colour-Coded by Mean Power\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")

    cbar = plt.colorbar(sc)
    cbar.set_label("Mean Detector Power (pW)")

    plt.axis("equal")
    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_locations_mean_power_PWV{PWV_MM:.2f}.png"
    )

    plt.close("all")
    plt.gca().invert_xaxis()  # Invert RA axis for astronomical convention
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")

    P_std = np.nanstd(P, axis=1).ravel()

    plt.figure(figsize=(8,6))

    sc = plt.scatter(
        ra_det,
        dec_det,
        c=P_std,
        cmap="coolwarm",
        s=50,
        edgecolor="k",
        alpha=0.7
    )
    plt.title(
        f"Detector Locations Colour-Coded by Standard Deviation of Power\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")

    cbar = plt.colorbar(sc)
    cbar.set_label("Standard Deviation of Detector Power (pW)")
    plt.axis("equal")
    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_locations_std_power_PWV{PWV_MM:.2f}.png"
    )

    plt.close("all")
    plt.gca().invert_xaxis()  # Invert RA axis for astronomical convention
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")


    det_idx = 50

    ra_track = np.asarray(tod.ra[det_idx, :], dtype=np.float64)
    dec_track = np.asarray(tod.dec[det_idx, :], dtype=np.float64)
    P_track = np.asarray(P[det_idx, :], dtype=np.float64)

    plt.figure(figsize=(10,6))

    sc = plt.scatter(
        ra_track,
        dec_track,
        c=P_track,
        cmap="viridis",
        s=2,
        alpha=0.7
    )

    

    plt.gca().invert_xaxis()  # Invert RA axis for astronomical convention
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")
    plt.title(
        f"Detector {det_idx} Track Colour-Coded by Power\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("Detector Power (pW)")

    plt.axis("equal")
    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_track_power_PWV{PWV_MM:.2f}.png"
    )
    plt.close("all")

    from scipy.stats import norm, skew

    # Calculate statistics for the power distribution

    P_track_flat = P_track[np.isfinite(P_track)]  # Flatten and remove NaN values

    mu, sigma = norm.fit(P_track_flat)

    frac_width = sigma / mu 
    frac_width_percent = frac_width * 100
    median = np.median(P_track_flat)
    p5, p16, p84, p95 = np.percentile(P_track_flat, [5, 16, 84, 95])
    mi_val = np.min(P_track_flat)
    ma_val = np.max(P_track_flat)
    ptp_val = np.abs(ma_val - mi_val)
    skewness = skew(P_track_flat)

    print(f"\nDetector {det_idx} power statistics")
    print("--------------------------------")
    print(f"Gaussian mean:          {mu:.6g} pW")
    print(f"Gaussian sigma:         {sigma:.6g} pW")
    print(f"Fractional width:       {frac_width:.6g}")
    print(f"Fractional width:       {frac_width_percent:.3f}%")
    print(f"Median:                 {median:.6g} pW")
    print(f"Min / Max:              {mi_val:.6g}, {ma_val:.6g} pW")
    print(f"Peak-to-peak:           {ptp_val:.6g} pW")
    print(f"5th / 95th percentile:  {p5:.6g}, {p95:.6g} pW")
    print(f"16th / 84th percentile: {p16:.6g}, {p84:.6g} pW")
    print(f"Skewness:               {skewness:.6g}")

    plt.figure(figsize=(8,6))

    counts, bins, _ = plt.hist(
        P_track_flat,
        bins = 50,
        alpha = 0.7,
        edgecolor="k", 
        label = f"Detector {det_idx} Power Distribution"
    )
    
    x = np.linspace(min(P_track_flat), max(P_track_flat), 1000)
    bin_width = bins[1] - bins[0]
    gaussian_counts = norm.pdf(x, mu, sigma) * len(P_track_flat) * bin_width
    plt.plot(
        x,
        gaussian_counts,
        color="red",
        lw=2,
        label=(
            f"Gaussian Fit\n"
            rf"$\mu$={mu:.3g} pW" "\n"
            rf"$\sigma$={sigma:.3g} pW"
        )
    )

    plt.xlabel("Detector Power (pW)")
    plt.ylabel("Number of Samples")
    plt.title(
        f"Power Distribution for Detector {det_idx}\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}, Skewness={skewness:.2f}"
    )
    plt.grid(True)
    plt.legend()

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_power_histogram_PWV{PWV_MM:.2f}.png"
    )

    plt.close("all")

    P_track_std = np.std(P_track_flat[det_idx, :])

    plt.figure(figsize=(8,6))
    plt.hist(
        P_track_flat,
        bins=50,
        alpha=0.7,
        edgecolor="k",
        label=f"Detector {det_idx} Power Distribution\n$\sigma$={P_track_std:.3g} pW"
    )
    plt.xlabel("Detector Power (pW)")
    plt.ylabel("Number of Samples")
    plt.title(
        f"Power Distribution for Detector {det_idx}\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)
    plt.legend()
    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_power_histogram_std_PWV{PWV_MM:.2f}.png"
    ) 

    plt.close("all")
    # tod.to("pW").plot()
    # simple_ccat.savefig(ANALYSIS_OUTDIR / f"{PREFIX}_tod_plot.png", f"{PREFIX}_tod_plot.png", dpi=300)
    # plt.close("all")

    # P_det = tod.to("pW").signal
    # P = np.asarray(P_det, dtype=np.float64)  # Convert to numpy array for calculations


    # print("P shape:", P.shape)

    # P_mean = np.nanmean(P, axis=1).ravel()
    # P_std  = np.nanstd(P, axis=1).ravel()
    # P_ptp  = (np.nanmax(P, axis=1) - np.nanmin(P, axis=1)).ravel()

    # plt.figure(figsize=(8,6))
    # plt.hist(P_mean[np.isfinite(P_mean)], bins=30, alpha=0.7, density=False)
    # plt.xlabel("Mean Detector Power (pW)")
    # plt.ylabel("Number of Detectors")
    # plt.title(f"Distribution of Mean Direct Detector Power (PWV={PWV_MM:.2f} mm $\\eta$={eta:.2f})")
    # plt.grid(True)
    # simple_ccat.savefig(ANALYSIS_OUTDIR, f"{PREFIX}_detector_direct_power_eta_{eta:.2f}_histogram_PWV{PWV_MM:.2f}.png")






    signal = tod.signal.compute()
    ra = tod.ra
    dec = tod.dec
    time = tod.time
    el = tod.el
    az = tod.az

    raise SystemExit("Test Complete")