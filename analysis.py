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

EL_LIMITS = (45, 55)  # degrees

T_0 = 278.868 #K, atmospheric ground temp

Q_r = 40000 # Quality factor taken from Bayguchi thesis

P_0 = 957e-18 # idk but do not question the mighty jordan wheeler

R_0 = -2.448e9 #avg responsivity in W^-1 from Jordan Wheeler

START_TIME = "2022-02-10T17:00:00"

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

PREFIX = "OrionA_45_tod"
ANALYSIS_OUTDIR = Path(f"outputs/{PREFIX}_analysis_outputs")
TOD_OUTDIR = Path(f"outputs/{PREFIX}_tod_files")




# simple_ccat.tod_analysis(
#     prfx=PREFIX,
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
#     scan_pattern = SCAN_PATTERN,
#     el_limits=EL_LIMITS,
# )

if __name__ == "__main__":


    tod = TOD.from_fits(TOD_OUTDIR / f"{PREFIX}_dim_reduced_tods.fits", format = "CCAT")

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

    print("RELOADED TOD")
    print("el min/max:", np.nanmin(np.rad2deg(tod.el)), np.nanmax(np.rad2deg(tod.el)))
    print("az min/max:", np.nanmin(np.rad2deg(tod.az)), np.nanmax(np.rad2deg(tod.az)))

    P_det = tod.to("pW").signal
    P = np.asarray(P_det, dtype=np.float64)  # Convert to numpy array for calculations


    print("P shape:", P.shape)

    P_mean = np.nanmean(P, axis=1).ravel()

    time_idx = int(1*SAMPLE_RATE_HZ)  # Index for 1 second into the TOD

    ra_det = np.asarray(tod.ra[:, time_idx], dtype=np.float64)
    dec_det = np.asarray(tod.dec[:, time_idx], dtype=np.float64)

    ra_det_deg = np.rad2deg(ra_det)
    dec_det_deg = np.rad2deg(dec_det)

    from scipy.spatial.distance import cdist

    coords = np.column_stack((ra_det_deg, dec_det_deg))
    distance_matrix = cdist(coords, coords)
    np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distance

    nearest_idx = np.argmin(distance_matrix, axis=1)
    nearest_dist = np.min(distance_matrix, axis=1)

    power_diff = np.abs(P_mean - P_mean[nearest_idx])

    best_idx = np.nanargmax(power_diff)

    det1 = best_idx
    det2 = nearest_idx[best_idx]

    print("\nNeighbouring detector pair")
    print("--------------------------")
    print(f"Detector 1: {det1}")
    print(f"Detector 2: {det2}")
    print(f"Separation: {nearest_dist[best_idx]:.6g} deg")
    print(f"Detector {det1} mean power: {P_mean[det1]:.6g} pW")
    print(f"Detector {det2} mean power: {P_mean[det2]:.6g} pW")
    print(f"Difference: {power_diff[best_idx]:.6g} pW")

    plt.figure(figsize=(8,6))

    sc = plt.scatter(
        ra_det_deg,
        dec_det_deg,
        c=P_mean,
        cmap="viridis",
        s=50,
        edgecolor="k",
        alpha=0.7
    )

    plt.scatter(
        ra_det_deg[[det1, det2]],
        dec_det_deg[[det1, det2]],
        s=250,
        facecolors="none",
        edgecolors="red",
        linewidths=1.0,
        zorder=5,
        label="Selected neighbouring detectors"
    )

    for det in [det1, det2]:
        plt.text(
            ra_det_deg[det],
            dec_det_deg[det],
            f" {det}",
            color="red",
            fontsize=10,
            weight="bold",
            zorder=6
        )

    plt.title(
        f"Detector Locations Colour-Coded by Mean Power\n"
        f"Highlighted pair: {det1} and {det2}, PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.xlabel("RA (degrees)")
    plt.ylabel("Dec (degrees)")

    cbar = plt.colorbar(sc)
    cbar.set_label("Mean Detector Power (pW)")

    plt.gca().invert_xaxis()
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_locations_mean_power_highlight_pair_PWV{PWV_MM:.2f}.png"
    )

    plt.close("all")

    # P_std = np.nanstd(P, axis=1).ravel()

    # plt.figure(figsize=(8,6))

    # sc = plt.scatter(
    #     ra_det,
    #     dec_det,
    #     c=P_std,
    #     cmap="coolwarm",
    #     s=50,
    #     edgecolor="k",
    #     alpha=0.7
    # )
    # plt.title(
    #     f"Detector Locations Colour-Coded by Standard Deviation of Power\n"
    #     f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    # )
    # plt.xlabel("RA (degrees)")
    # plt.ylabel("Dec (degrees)")

    # cbar = plt.colorbar(sc)
    # cbar.set_label("Standard Deviation of Detector Power (pW)")
    # plt.axis("equal")
    # plt.grid(True)

    # simple_ccat.savefig(
    #     ANALYSIS_OUTDIR,
    #     f"{PREFIX}_detector_locations_std_power_PWV{PWV_MM:.2f}.png"
    # )

    # plt.close("all")
    # plt.gca().invert_xaxis()  # Invert RA axis for astronomical convention
    # plt.xlabel("RA (degrees)")
    # plt.ylabel("Dec (degrees)")


    # det_idx = det1
    for det_idx in [det1, det2]:
        ra_track = np.asarray(tod.ra[det_idx, :], dtype=np.float64)
        dec_track = np.asarray(tod.dec[det_idx, :], dtype=np.float64)
        P_track = np.asarray(P[det_idx, :], dtype=np.float64)

        ra_track_deg = np.rad2deg(ra_track)
        dec_track_deg = np.rad2deg(dec_track)

        plt.figure(figsize=(10,6))

        sc = plt.scatter(
            ra_track_deg,
            dec_track_deg,
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

        time_sec = np.arange(len(P_track)) / SAMPLE_RATE_HZ

        plt.figure(figsize=(10,6))

        plt.plot(
            time_sec,
            P_track,
            lw=0.5,
            color="black"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Detector Power (pW)")

        plt.title(
            f"Detector {det_idx} Power vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_power_vs_time_PWV{PWV_MM:.2f}.png"
        )

        # ============================================================
        # Plot: Individual detector Az/El track colour-coded by power
        # ============================================================

        az_track = np.asarray(tod.az[det_idx, :], dtype=np.float64)
        el_track = np.asarray(tod.el[det_idx, :], dtype=np.float64)

        az_track_deg = np.rad2deg(az_track)
        el_track_deg = np.rad2deg(el_track)

        plt.figure(figsize=(10, 6))

        sc = plt.scatter(
            az_track_deg,
            el_track_deg,
            c=P_track,
            cmap="viridis",
            s=2,
            alpha=0.7
        )

        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("Elevation (degrees)")
        plt.title(
            f"Detector {det_idx} Az/El Track Colour-Coded by Power\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        cbar = plt.colorbar(sc)
        cbar.set_label("Detector Power (pW)")

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_azel_track_power_PWV{PWV_MM:.2f}.png"
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

    P_track_det1 = np.asarray(P[det1, :], dtype=np.float64)
    P_track_det2 = np.asarray(P[det2, :], dtype=np.float64)

    time_sec = np.arange(len(P_track_det1)) / SAMPLE_RATE_HZ

    power_diff = P_track_det1 - P_track_det2
    power_abs_diff = np.abs(power_diff)

    # Avoid divide-by-zero issues
    power_ratio = np.full_like(P_track_det1, np.nan)
    valid = np.isfinite(P_track_det1) & np.isfinite(P_track_det2) & (P_track_det2 != 0)
    power_ratio[valid] = P_track_det1[valid] / P_track_det2[valid]

    plt.figure(figsize=(10,6))
    plt.plot(time_sec, power_abs_diff, lw=0.5, color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Difference in Detector Power (pW)")
    plt.title(
        f"Power Difference Between Detectors {det1} and {det2} vs Time\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det1}_{det2}_power_difference_vs_time_PWV{PWV_MM:.2f}.png"
    )
    plt.close("all")


    plt.figure(figsize=(10,6))
    plt.plot(time_sec, power_ratio, lw=0.5, color="black")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Detector Power Ratio {det1}/{det2}")
    plt.title(
        f"Detector Power Ratio {det1}/{det2} vs Time\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det1}_{det2}_power_ratio_vs_time_PWV{PWV_MM:.2f}.png"
    )
    plt.close("all")

    print(f"\nDetector {det1} mean/std: {np.nanmean(P_track_det1):.6g}, {np.nanstd(P_track_det1):.6g} pW")
    print(f"Detector {det2} mean/std: {np.nanmean(P_track_det2):.6g}, {np.nanstd(P_track_det2):.6g} pW")
    print(f"Mean absolute difference: {np.nanmean(power_abs_diff):.6g} pW")
    print(f"Median ratio {det1}/{det2}: {np.nanmedian(power_ratio):.6g}")

    power_diff_centered = power_diff - np.nanmean(power_diff)

    plt.figure(figsize=(8,6))
    plt.plot(time_sec, power_diff_centered, lw=0.5, color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean-subtracted Power Difference (pW)")
    plt.title(
        f"Mean-subtracted Power Difference: Detectors {det1} - {det2}\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det1}_{det2}_mean_subtracted_power_difference_PWV{PWV_MM:.2f}.png"
    )
    plt.close("all")

    scale_factor = np.nanmedian(P_track_det1/P_track_det2)
    residual = P_track_det1 - scale_factor * P_track_det2

    plt.figure(figsize=(10,6))
    plt.plot(time_sec, residual, lw=0.5, color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Residual After Scaling (pW)")
    plt.title(
        f"Residual After Scaling Detector {det2} by {scale_factor:.3g}\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)


    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det1}_{det2}_residual_after_scaling_PWV{PWV_MM:.2f}.png"
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

    
    detector_indices = []
    mean_list = []
    sigma_list = []
    two_delta_list = []
    fractional_width_list = []
    two_delta_over_mean_list = []
    delta_f_over_fwhm_list = []

    n_detectors = P.shape[0]

    for det_idx in range(n_detectors):
        P_track = np.asarray(P[det_idx, :], dtype=np.float64)
        P_track_flat = P_track[np.isfinite(P_track)]

        P_track_mean = np.nanmean(P_track_flat)

        P_track_W = P_track_flat * 1e-12  # Convert pW to W

        P_track_W_mean = np.nanmean(P_track_W)



        if len(P_track_flat) == 0:
            detector_indices.append(det_idx)
            mean_list.append(np.nan)
            sigma_list.append(np.nan)
            two_delta_list.append(np.nan)
            fractional_width_list.append(np.nan)
            two_delta_over_mean_list.append(np.nan)
            delta_f_over_fwhm_list.append(np.nan)
            continue

        mu, sigma = norm.fit(P_track_flat)

        # two_delta = np.max(P_track_flat) - np.min(P_track_flat)
        fractional_width = sigma / mu if mu != 0 else np.nan

        delta_f_over_fwhm = (Q_r * R_0 / (np.sqrt(1+P_track_W_mean / P_0)) * P_track_W_mean) * fractional_width 

        detector_indices.append(det_idx)
        mean_list.append(mu)
        sigma_list.append(sigma)
        fractional_width_list.append(fractional_width)
        delta_f_over_fwhm_list.append(np.abs(delta_f_over_fwhm))

        # for det_idx in range(n_detectors):
        #     P_track = np.asarray(P[det_idx, :], dtype=np.float64)
        #     P_track_flat = P_track[np.isfinite(P_track)]

        #     P_track_W = P_track_flat * 1e-12  # Convert pW to W
        #     delta_f_over_fwhm_by_detector_list.append(fractional_width_list[det_idx] * Q_r * R_0 / (np.sqrt(1+ P_track_W/ P_0)) *P_track_flat)


    def make_hist(data, xlabel, title, filename):
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data)]

        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=30, alpha=0.7, density=False, edgecolor="k")
        plt.xlabel(xlabel)
        plt.ylabel("Number of Detectors")
        plt.title(f"{title}\nPWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}")
        plt.grid(True)

        simple_ccat.savefig(ANALYSIS_OUTDIR, filename)
        plt.close("all")


    make_hist(
        mean_list,
        "Gaussian Mean of Detector Power (pW)",
        "Distribution of Gaussian Means for Detectors",
        f"{PREFIX}_detector_gaussian_mean_histogram_PWV{PWV_MM:.2f}.png"
    )

    make_hist(
        sigma_list,
        "Gaussian Sigma of Detector Power (pW)",
        "Distribution of Gaussian Sigmas for Detectors",
        f"{PREFIX}_detector_gaussian_sigma_histogram_PWV{PWV_MM:.2f}.png"
    )

    # make_hist(
    #     two_delta_list,
    #     r"$2\Delta = 2\sigma$ (pW)",
    #     r"Distribution of $2\Delta$ for Detectors",
    #     f"{PREFIX}_detector_gaussian_two_delta_histogram_PWV{PWV_MM:.2f}.png"
    # )

    make_hist(
        fractional_width_list,
        r"Fractional Width $\sigma / \mu$",
        r"Distribution of Fractional Widths for Detectors",
        f"{PREFIX}_detector_gaussian_fractional_width_histogram_PWV{PWV_MM:.2f}.png"
    )

    # make_hist(
    #     delta_f_over_fwhm_by_detector_list,
    #     r"Estimated $\Delta f / \mathrm{FWHM}$",
    #     r"Distribution of Estimated $\Delta f / \mathrm{FWHM}$ for Detectors",
    #     f"{PREFIX}_detector_gaussian_delta_f_over_fwhm_histogram_PWV{PWV_MM:.2f}.png"
    # )

    # make_hist(
    #     two_delta_over_mean_list,
    #     r"$2\Delta / \mu$",
    #     r"Distribution of $2\Delta / \mu$ for Detectors",
    #     f"{PREFIX}_detector_gaussian_two_delta_over_mean_histogram_PWV{PWV_MM:.2f}.png"
    # )

    make_hist(
        delta_f_over_fwhm_list,
        r"Estimated $\Delta f / \mathrm{FWHM}$",
        r"Distribution of Estimated $\Delta f / \mathrm{FWHM}$ for Detectors",
        f"{PREFIX}_detector_gaussian_delta_f_over_fwhm_histogram_PWV{PWV_MM:.2f}.png"
    )


    P_mean_array = np.array(mean_list)
    P_ref = np.nanmean(P_mean_array)

    print(f"\nReference power level (median of Gaussian means): {P_ref:.6g} pW")
    P_mean_W = P_mean_array * 1e-12  # Convert pW to W
    P_ref_W = P_ref * 1e-12  # Convert pW to W
    def R(P):
        return R_0 / (np.sqrt(1+ P/ P_0))
        
    delta_f_over_fwhm_by_detector = []

    for det_idx in range(P.shape[0]):
        P_track = np.asarray(P[det_idx, :], dtype=np.float64)
        P_track_W = P_track * 1e-12  # Convert pW to W
        P_track_flat_W = P_track_W[np.isfinite(P_track_W)]

        delta_track =  Q_r * R(P_track_W) * (P_track_W - P_ref_W)
        delta_f_over_fwhm_by_detector.append(delta_track)

    det_309 = 309
    det_339 = 339
    delta_track = delta_f_over_fwhm_by_detector[det_339]

    time_sec = np.arange(len(delta_track)) / SAMPLE_RATE_HZ

    plt.figure(figsize=(10,6))
    plt.plot(time_sec, delta_track, lw=0.5, color="black")

    plt.axhline(0, color="red", lw=1, ls="--")

    plt.xlabel("Time (s)")
    plt.ylabel(r"$\Delta f / \mathrm{FWHM}$")
    plt.title(
        f"Estimated Fractional Frequency Shift for Detector {det_339}\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_339}_fractional_frequency_shift_PWV{PWV_MM:.2f}.png"
    )
    plt.close("all")

    delta_mean_list = []

    for delta_track in delta_f_over_fwhm_by_detector:
        delta_mean = np.nanmean(delta_track)
        delta_mean_list.append(delta_mean)

    plt.figure(figsize=(8,6))
    plt.hist(delta_mean_list, bins=30, alpha=0.7, edgecolor="k")
    plt.xlabel(r"Mean $\Delta f / \mathrm{FWHM}$")
    plt.ylabel("Number of Detectors")
    plt.title(
        f"Distribution of Mean Fractional Frequency Shifts for Detectors\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)
    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_mean_fractional_frequency_shift_histogram_PWV{PWV_MM:.2f}.png"
    )

    plt.close("all")

    for det_idx in [0, 50, 309, 339, 472, 843]:

        det_idx = det_idx 

        P_track_pW = np.asarray(P[det_idx, :], dtype=np.float64)
        valid = np.isfinite(P_track_pW)

        P_track_pW = P_track_pW[valid]
        P_track_W = P_track_pW * 1e-12

        P_ref_W = np.nanmean(P_track_W)
        # or better if skewed:
        # P_ref_W = np.nanmedian(P_track_W)

        def R(P_W):
            return R_0 / np.sqrt(1 + P_W / P_0)

        delta_f_over_fwhm = Q_r * R(P_ref_W) * (P_track_W - P_ref_W)

        plt.figure(figsize=(10,6))

        plt.hist(delta_f_over_fwhm, bins=30, alpha=0.7, edgecolor="k")

        plt.axvline(0, color="red", lw=1, ls="--")

        plt.xlabel(r"$\Delta f / \mathrm{FWHM}$")
        plt.ylabel("Number of Samples")
        plt.title(
            f"Distribution of Estimated Fractional Frequency Shifts for Detector {det_idx}\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_fractional_frequency_shift_histogram_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")

        two_delta_f_over_fwhm_list = []
        half_delta_f_over_fwhm_list = []
        def R(P_W):
            return R_0 / np.sqrt(1 + P_W / P_0)

        for det_idx in range(P.shape[0]):
            P_track_pW = np.asarray(P[det_idx, :], dtype=np.float64)
            valid = np.isfinite(P_track_pW)

            P_track_pW = P_track_pW[valid]
            P_track_W = P_track_pW * 1e-12

            P_ref_W = np.nanmean(P_track_W) #mean power for individual reference

            delta_f_over_fwhm = Q_r * R(P_ref_W) * (P_track_W - P_ref_W)

            two_delta_f_over_fwhm = np.max(delta_f_over_fwhm) - np.min(delta_f_over_fwhm)
            half_delta_f_over_fwhm = np.abs(two_delta_f_over_fwhm / 2)
            two_delta_f_over_fwhm_list.append(two_delta_f_over_fwhm)
            half_delta_f_over_fwhm_list.append(half_delta_f_over_fwhm)

    make_hist(
        two_delta_f_over_fwhm_list,
        r"$2\Delta f / \mathrm{FWHM}$",
        r"Distribution of Peak-to-Peak Fractional Frequency Shifts for Detectors",
        f"{PREFIX}_detector_two_delta_f_over_fwhm_histogram_PWV{PWV_MM:.2f}.png"
    )

    make_hist(
        half_delta_f_over_fwhm_list,
        r"$\Delta f / \mathrm{FWHM}$",
        r"Distribution of Half Peak-to-Peak Fractional Frequency Shifts for Detectors",
        f"{PREFIX}_detector_half_delta_f_over_fwhm_histogram_PWV{PWV_MM:.2f}.png"
    )

# ============================================================
# Plot: Individual detector Az/El track colour-coded by delta f / FWHM
# Uses each detector's own mean power as the reference
# ============================================================

for det_idx in [0, 50, 309, 339, 472, 843]:\

    time_sec_full = np.arange(P.shape[1]) / SAMPLE_RATE_HZ

    time_sec = time_sec_full[valid]

    P_track_pW = np.asarray(P[det_idx, :], dtype=np.float64)
    az_track = np.asarray(tod.az[det_idx, :], dtype=np.float64)
    el_track = np.asarray(tod.el[det_idx, :], dtype=np.float64)

    valid = (
        np.isfinite(P_track_pW)
        & np.isfinite(az_track)
        & np.isfinite(el_track)
    )

    P_track_pW = P_track_pW[valid]
    P_track_W = P_track_pW * 1e-12

    az_track_deg = np.rad2deg(az_track[valid])
    el_track_deg = np.rad2deg(el_track[valid])

    P_ref_W = np.nanmean(P_track_W)

    delta_f_over_fwhm = Q_r * R(P_ref_W) * (P_track_W - P_ref_W)

    plt.figure(figsize=(10, 6))

    sc = plt.scatter(
        az_track_deg,
        el_track_deg,
        c=delta_f_over_fwhm,
        cmap="viridis",
        s=2,
        alpha=0.7
    )

    plt.axhline(np.nanmean(el_track_deg), color="red", lw=1, ls="--")

    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Elevation (degrees)")
    plt.title(
        f"Detector {det_idx} Az/El Track Colour-Coded by Delta f / FWHM\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )

    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\Delta f / \mathrm{FWHM}$")

    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_azel_track_delta_f_over_fwhm_PWV{PWV_MM:.2f}.png"
    )

    plt.figure(figsize=(10, 6))

    sc = plt.scatter(
        az_track_deg,
        el_track_deg,
        c=time_sec,
        cmap="viridis",
        s=2,
        alpha=0.7
    )

    plt.axhline(np.nanmean(el_track_deg), color="red", lw=1, ls="--")

    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Elevation (degrees)")
    plt.title(
        f"Detector {det_idx} Az/El Track Colour-Coded by Time\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )

    cbar = plt.colorbar(sc)
    cbar.set_label(r"Time (s)")

    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_azel_track_time_PWV{PWV_MM:.2f}.png"
    )

    P_track_pW = np.asarray(P[det_idx, :], dtype=np.float64)
    P_track_pW = P_track_pW[valid]
    P_track_W = P_track_pW * 1e-12

    def R(P):
        return R_0 / (np.sqrt(1+ P/ P_0))
    
    Queens_Park_R = Q_r * R(P_track_W) * P_track_W
    
    plt.figure(figsize=(10, 6))
    plt.scatter(
        P_track_W,
        Queens_Park_R,
        s=2,
        alpha=0.7,
        color="black"
    )

    plt.xlabel("Detector Power (W)")
    plt.ylabel(r"Queens Park Rangers (Hz/W)")
    plt.title(
        f"Detector {det_idx} Power vs Queens Park Rangers\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )

    plt.grid(True)
    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_power_vs_queens_park_rangers_PWV{PWV_MM:.2f}.png"
    )
    plt.close("all")

    # ============================================================
    # Plot: Individual detector delta f / FWHM vs time
    # Uses each detector's own mean power as the reference
    # ============================================================

    time_sec_full = np.arange(P.shape[1]) / SAMPLE_RATE_HZ

    time_sec = time_sec_full[valid]

    plt.figure(figsize=(10, 6))

    plt.plot(
        time_sec,
        delta_f_over_fwhm,
        lw=0.5,
        color="black"
    )

    plt.axhline(0, color="red", lw=1, ls="--")

    plt.xlabel("Time (s)")
    plt.ylabel(r"$\Delta f / \mathrm{FWHM}$")
    plt.title(
        f"Detector {det_idx} Delta f / FWHM vs Time\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )

    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_delta_f_over_fwhm_vs_time_PWV{PWV_MM:.2f}.png"
    )

    plt.close("all")


    # ============================================================
    # Plot: Individual detector delta f / FWHM vs elevation
    # Uses each detector's own mean power as the reference
    # ============================================================

    el_track = np.asarray(tod.el[det_idx, :], dtype=np.float64)
    el_track_deg = np.rad2deg(el_track[valid])

    time_sec_full = np.arange(P.shape[1]) / SAMPLE_RATE_HZ

    time_sec = time_sec_full[valid]

    plt.figure(figsize=(10, 6))

    plt.plot(
        el_track_deg,
        delta_f_over_fwhm,
        lw=0.5,
        color="black"
    )

    plt.axhline(0, color="red", lw=1, ls="--")

    plt.xlabel("Elevation (degrees)")
    plt.ylabel(r"$\Delta f / \mathrm{FWHM}$")
    plt.title(
        f"Detector {det_idx} Delta f / FWHM vs Elevation\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )

    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_delta_f_over_fwhm_vs_elevation_PWV{PWV_MM:.2f}.png"
    )


    plt.figure(figsize=(10, 6))

    plt.plot(
        el_track_deg,
        P_track_pW,
        lw=0.5,
        color="black"
    )

    plt.axhline(0, color="red", lw=1, ls="--")

    plt.xlabel("Elevation (degrees)")
    plt.ylabel(r"Detector Power (W)")
    plt.title(
        f"Detector {det_idx} Power vs Elevation\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )

    plt.grid(True)

    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_detector_{det_idx}_power_vs_elevation_PWV{PWV_MM:.2f}.png"
    )


    plt.close("all")




    signal = tod.signal.compute()
    ra = tod.ra
    dec = tod.dec
    time = tod.time
    el = tod.el
    az = tod.az

raise SystemExit("Test Complete")