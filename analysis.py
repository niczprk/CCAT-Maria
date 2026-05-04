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




simple_ccat.tod_analysis(
    maps = False,
    save_all_plots = False,
    run_mode = "fits",
    atm_plot = True,
    temp_mode = "inst",
    ccat_band = "280",
    map_type = "BM",
    pwv_mm = PWV_MM,
    start_time = START_TIME,
    total_duration_s = TOTAL_DURATION_S,
    sim_duration_s = SIM_DURATION_S,
    sample_rate_hz = SAMPLE_RATE_HZ,
)

if __name__ == "__main__":


    tod = TOD.from_fits(TOD_OUTDIR / f"{PREFIX}_tods.fits", format = "Mustang-2")

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

    tod.to("pW").plot()
    simple_ccat.savefig(ANALYSIS_OUTDIR / f"{PREFIX}_tod_plot.png", f"{PREFIX}_tod_plot.png", dpi=300)
    plt.close("all")

    P_det = tod.to("pW").signal
    P = np.asarray(P_det, dtype=np.float64)  # Convert to numpy array for calculations


    print("P shape:", P.shape)

    P_mean = np.nanmean(P, axis=1).ravel()
    P_std  = np.nanstd(P, axis=1).ravel()
    P_ptp  = (np.nanmax(P, axis=1) - np.nanmin(P, axis=1)).ravel()

    plt.figure(figsize=(8,6))
    plt.hist(P_mean[np.isfinite(P_mean)], bins=30, alpha=0.7, density=False)
    plt.xlabel("Mean Detector Power (pW)")
    plt.ylabel("Number of Detectors")
    plt.title(f"Distribution of Mean Direct Detector Power (PWV={PWV_MM:.2f} mm $\\eta$={eta:.2f})")
    plt.grid(True)
    simple_ccat.savefig(ANALYSIS_OUTDIR, f"{PREFIX}_detector_direct_power_eta_{eta:.2f}_histogram_PWV{PWV_MM:.2f}.png")






    signal = tod.signal.compute()
    ra = tod.ra
    dec = tod.dec
    time = tod.time
    el = tod.el
    az = tod.az

    raise SystemExit("Test Complete")