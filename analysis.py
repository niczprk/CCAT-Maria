# =============================================================================
# MARIA / CCAT Motion and Detector Analysis Script
# =============================================================================
# Organization-only version.
#
# This file keeps the original code intact and mainly adds clearer section
# headings so it is easier to navigate. I have not intentionally removed,
# rewritten, or refactored the working code blocks.
# =============================================================================

# ========================================================================
# Imports and external dependencies
# ========================================================================

from pathlib import Path
from turtle import color

import simple_ccat
from maria import tod
from maria.instrument import Band
import maria

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt


# ========================================================================
# User-controlled configuration and physical constants
# ========================================================================

selected_band = "280" #make sure these match

NU_HZ = 280e9  # Hz
bandwidth_hz = 60e9  # GHz bandwidth for 850 GHz band
eta = 0.5 # optical efficiency for this estimate

Polarized = False # whether to include polarization in the simulation

NU_GHZ = NU_HZ / 1e9 #GHz

PWV_MM = 0.36  #  mm, precip water vapour this only affects the main if pwv is None

# 0.36, 0.67, & 1.28 are Q1, Q2, and Q3 zenith PMV values for Chajnantor

EL_LIMITS = (65, 75)  # degrees

ELEVATION_RANGES = [
    (35, 45),
    (45, 55),
    (55, 65),
    (65, 75)
]
ELEV_LABEL = f"{EL_LIMITS[0]}-{EL_LIMITS[1]}"



SPEED  = 0.2 # deg/s, this is a guess for now but should be in the right ballpark for a daisy scan at 30-40 deg elevation with a 3 degree radius

T_0 = 278.868 #K, atmospheric ground temp

Q_r = 40000 # Quality factor taken from Bayguchi thesis

P_0 = 957e-18 # idk but do not question the mighty jordan wheeler

R_0 = -2.448e9 #avg responsivity in W^-1 from Jordan Wheeler

START_TIME = "2022-02-10T17:00:00"

eta = 0.5

# "2022-02-10T22:45:00" for around 75 degrees ?
#"2022-02-10T20:30:00" for around 60 degrees 
#"2022-02-10T18:55:00" for roughly 45 degrees
#"2022-02-10T18:30:00" for roughly 40 degrees
#"2022-02-10T17:00:00" for roughly 30 degrees
 

TOTAL_DURATION_S = 1800  # seconds
SIM_DURATION_S = 1800  # seconds
SAMPLE_RATE_HZ = 20  # Hz
SCAN_PATTERN = "back_and_forth"
CHUNK_NUMBER = 0

PREFIX = f"OrionA_{ELEV_LABEL}_vel_el"
ANALYSIS_OUTDIR = Path(f"outputs/{PREFIX}_lissa_analysis_outputs")
TOD_OUTDIR = Path(f"outputs/{PREFIX}_lissa_tod")

RUN_SPEED_GRID = True
COMBINE_EXISTING_CSVS = False
SPEED_CSV_DIR = Path(f"outputs/OrionA_{SCAN_PATTERN}_speed_tests/speed_csv")
COMBINED_PLOT_DIR = Path(f"outputs/OrionA_{SCAN_PATTERN}_speed_tests/combined_plots")
SPEED_CSV_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_PLOT_DIR.mkdir(parents=True, exist_ok=True)

print(dir((maria.tod)))




# ========================================================================
# Main execution block
# ========================================================================

if __name__ == "__main__":

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------

    # csv_dir = Path("outputs/OrionA_speed_tests/speed_csv")

    # csv_files = [
    #     csv_dir / "maria_speed_elevation_motion_limits_35-45.csv",
    #     csv_dir / "maria_speed_elevation_motion_limits_45-55.csv",
    #     csv_dir / "maria_speed_elevation_motion_limits_55-65.csv",
    #     csv_dir / "maria_speed_elevation_motion_limits_65-75.csv",
    # ]

    # combined_csv_path = csv_dir / "maria_speed_elevation_motion_limits_combined.csv"

    # plot_outdir = Path("outputs/OrionA_speed_tests/combined_plots")
    # plot_outdir.mkdir(parents=True, exist_ok=True)

    # # -------------------------------------------------
    # # Combine CSVs
    # # -------------------------------------------------

    # dfs = []

    # for csv_file in csv_files:
    #     df = pd.read_csv(csv_file)

    #     # In case elevation_label is missing or inconsistent
    #     if "elevation_label" not in df.columns:
    #         label = csv_file.stem.split("_")[-1]
    #         df["elevation_label"] = label

    #     dfs.append(df)

    # combined_df = pd.concat(dfs, ignore_index=True)

    # combined_df.to_csv(combined_csv_path, index=False)

    # print(f"Saved combined CSV to: {combined_csv_path}")
    # print(f"Total rows: {len(combined_df)}")


    # # -------------------------------------------------
    # # Overlaid plots
    # # -------------------------------------------------

    # quantities = {
    #     "max_az_velocity_deg_s": {
    #         "label": "Maximum AZ Velocity",
    #         "unit": "deg/s",
    #         "limit": 3.0,
    #         "filename": "combined_max_az_velocity_vs_input_speed.png",
    #     },
    #     "max_el_velocity_deg_s": {
    #         "label": "Maximum EL Velocity",
    #         "unit": "deg/s",
    #         "limit": 1.5,
    #         "filename": "combined_max_el_velocity_vs_input_speed.png",
    #     },
    #     "max_az_acceleration_deg_s2": {
    #         "label": "Maximum AZ Acceleration",
    #         "unit": "deg/s²",
    #         "limit": 6.0,
    #         "filename": "combined_max_az_acceleration_vs_input_speed.png",
    #     },
    #     "max_el_acceleration_deg_s2": {
    #         "label": "Maximum EL Acceleration",
    #         "unit": "deg/s²",
    #         "limit": 1.5,
    #         "filename": "combined_max_el_acceleration_vs_input_speed.png",
    #     },
    # }

    # for column, info in quantities.items():

    #     plt.figure(figsize=(8, 6))

    #     for elevation_label, group in combined_df.groupby("elevation_label"):
    #         group = group.sort_values("input_speed_deg_s")

    #         plt.plot(
    #             group["input_speed_deg_s"],
    #             group[column],
    #             marker="o",
    #             linewidth=2,
    #             alpha=0.8,
    #             label=elevation_label,
    #         )

    #     plt.axhline(
    #         info["limit"],
    #         color="black",
    #         linestyle="--",
    #         linewidth=1.5,
    #         label="Motion limit",
    #     )

    #     plt.xlabel("Maria input speed (deg/s)")
    #     plt.ylabel(f"{info['label']} ({info['unit']})")
    #     plt.title(f"{info['label']} vs Maria Input Speed")
    #     plt.grid(True, alpha=0.3)
    #     plt.legend(title="Elevation range")

    #     plt.savefig(
    #         plot_outdir / info["filename"],
    #         dpi=300,
    #         bbox_inches="tight",
    #     )

    #     plt.close()

    # print(f"Saved combined plots to: {plot_outdir}") 

    # raise SystemExit("all done.")

    # simple_ccat.tod_analysis(
    # PREFIX=PREFIX,
    # tod_diagnostics=False,
    # maps = False,
    # save_all_plots = True,
    # run_mode = "fits",
    # atm_plot = True,
    # temp_mode = "inst",
    # ccat_band = "280",
    # map_type = "BM",
    # pwv_mm = PWV_MM,
    # start_time = START_TIME,
    # total_duration_s = TOTAL_DURATION_S,
    # sim_duration_s = SIM_DURATION_S,
    # sample_rate_hz = SAMPLE_RATE_HZ,
    # scan_pattern = SCAN_PATTERN,
    # el_limits = EL_LIMITS,
    # speed = SPEED
    # )

    # ========================================================================
    # Part 3: run/load TODs and evaluate motion limits over speed grid
    # ========================================================================

    # -------------------------------------------------
    # Variable Speed Analysis
    # -------------------------------------------------
    # speed_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    speed_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8] #changed for raster scan to avoid hitting the 1.0 deg/s limit


    # start_time_dict = {
    #     "30deg": "2022-02-10T17:00:00",
    #     "40deg": "2022-02-10T18:30:00",
    #     "45deg": "2022-02-10T18:55:00",
    #     "60deg": "2022-02-10T20:30:00",
    #     "75deg": "2022-02-10T22:45:00",
    # }

    motion_results = []

    site = maria.get_site("cerro_chajnantor", altitude=5600)

    band = Band(
        name="m2/f093",
        center=280e9,
        width=60e9,
        efficiency=eta,
        NET_CMB=13e-6,
        knee=1.0,
        gain_error=5e-2,
    )

    # for elev_label, start_time in start_time_dict.items():

    # ========================================================================
    # Loop over MARIA input scan speeds
    # ========================================================================
    for EL_LIMITS in ELEVATION_RANGES:

        ELEV_LABEL = f"{EL_LIMITS[0]}-{EL_LIMITS[1]}"

        print(f"\nStarting analysis for elevation range: {ELEV_LABEL} degrees")
        print("===============================================")
        PREFIX = f"OrionA_{ELEV_LABEL}_{SCAN_PATTERN.lower()}"
        ANALYSIS_OUTDIR = Path(f"outputs/{PREFIX}_analysis_outputs")
        ANALYSIS_OUTDIR.mkdir(parents=True, exist_ok=True)

        motion_results = []  # Reset motion results for this elevation range

        for spd in speed_list:

            run_prefix = f"OrionA_{SCAN_PATTERN.lower()}_{ELEV_LABEL}_speed_{spd:.1f}".replace(".", "p")

            run_analysis_outdir = Path(f"outputs/{run_prefix}_analysis_outputs")
            run_tod_outdir = Path(f"outputs/{run_prefix}_tods")

            print("\nRunning simulation")
            print("------------------")
            print("Elevation label:", ELEV_LABEL)
            print("Start time:", START_TIME)
            print("Speed:", spd)
            print("Prefix:", run_prefix)

            # ========================================================================
            # Generate/load TOD for this speed
            # ========================================================================

            simple_ccat.tod_analysis(
                PREFIX=run_prefix,
                tod_diagnostics=False,
                maps=False,
                save_all_plots=False,
                run_mode="fits",
                atm_plot=False,
                temp_mode="inst",
                ccat_band="280",
                map_type="BM",
                pwv_mm=PWV_MM,
                start_time=START_TIME,
                total_duration_s=TOTAL_DURATION_S,
                sim_duration_s=SIM_DURATION_S,
                sample_rate_hz=SAMPLE_RATE_HZ,
                scan_pattern=SCAN_PATTERN,
                el_limits=EL_LIMITS,
                speed=spd,
            )

            fits_path = run_tod_outdir / f"{run_prefix}_dim_reduced_tods.fits"

            if not fits_path.exists():
                print(f"Missing FITS file for {run_prefix}")
                continue

            # ========================================================================
            # Load FITS TOD and initialize max motion metrics
            # ========================================================================

            tod = maria.tod.load(fits_path, site=site, bands=[band])

            global_max_az_vel = 0.0
            global_max_el_vel = 0.0
            global_max_az_acc = 0.0
            global_max_el_acc = 0.0
            # global_max_az_jerk = 0.0
            # global_max_el_jerk = 0.0

            mean_el_list = []

            # ========================================================================
            # Evaluate detector trajectory derivatives
            # ========================================================================

            for det_idx in [604]:

                az_deg_track = np.rad2deg(tod.az[det_idx, :])
                el_deg_track = np.rad2deg(tod.el[det_idx, :])

                ra_deg_track = np.rad2deg(tod.ra[det_idx, :])
                dec_deg_track = np.rad2deg(tod.dec[det_idx, :])

                time = np.arange(len(az_deg_track)) / SAMPLE_RATE_HZ
                dt = 1 / SAMPLE_RATE_HZ

                valid = np.isfinite(az_deg_track) & np.isfinite(el_deg_track) & np.isfinite(ra_deg_track) & np.isfinite(dec_deg_track)
                az_deg_track = az_deg_track[valid]
                el_deg_track = el_deg_track[valid]
                ra_deg_track = ra_deg_track[valid]
                dec_deg_track = dec_deg_track[valid]
                time = time[valid]

                if len(az_deg_track) < 3:
                    continue

                mean_el_list.append(np.nanmean(el_deg_track))

                # Projected angular velocity on sky

                velocity_ra = (
                    np.cos(np.radians(dec_deg_track[1:-1]))
                    * (ra_deg_track[2:] - ra_deg_track[:-2])
                    / (2 * dt)
                )

                velocity_dec = (
                    (dec_deg_track[2:] - dec_deg_track[:-2])
                    / (2 * dt)
                )

                acceleration_ra = (
                    np.cos(np.radians(dec_deg_track[1:-1]))
                    * (ra_deg_track[2:] - 2 * ra_deg_track[1:-1] + ra_deg_track[:-2])
                    / (dt ** 2)
                )

                acceleration_dec = (
                    (dec_deg_track[2:] - 2 * dec_deg_track[1:-1] + dec_deg_track[:-2])
                    / (dt ** 2)
                )

                projected_velocity_az = (
                    np.cos(np.radians(el_deg_track[1:-1]))
                    * (az_deg_track[2:] - az_deg_track[:-2])
                    / (2 * dt)
                )

                velocity_el = (
                    (el_deg_track[2:] - el_deg_track[:-2])
                    / (2 * dt)
                )

                projected_acceleration_az = (
                    np.cos(np.radians(el_deg_track[1:-1]))
                    * (az_deg_track[2:] - 2 * az_deg_track[1:-1] + az_deg_track[:-2])
                    / (dt ** 2)
                )

                acceleration_el = (
                    (el_deg_track[2:] - 2 * el_deg_track[1:-1] + el_deg_track[:-2])
                    / (dt ** 2)
                )

                radec_speed_total = np.sqrt(velocity_ra**2 + velocity_dec**2)

                radec_velocity_sum = velocity_ra + velocity_dec

                azel_projected_velocity_sum = projected_velocity_az + velocity_el

                projected_speed_total = np.sqrt(projected_velocity_az**2 + velocity_el**2)


                # jerk_el = (
                #     (el_deg_track[4:] - 2 * el_deg_track[3:-1] + 2 * el_deg_track[1:-3] - el_deg_track[:-4])
                #     / (2*dt**3)
                # )

                # Actual azimuth motor motion, without cos(el)
                motor_velocity_az = (
                    (az_deg_track[2:] - az_deg_track[:-2])
                    / (2 * dt)
                )

                motor_acceleration_az = (
                    (az_deg_track[2:] - 2 * az_deg_track[1:-1] + az_deg_track[:-2])
                    / (dt ** 2)
                )

                motor_velocity_el = velocity_el

                motor_acceleration_el = acceleration_el

                motor_velocity_sum = motor_velocity_az + motor_velocity_el

                motor_speed_total = np.sqrt(motor_velocity_az**2 + motor_velocity_el**2)

                time_mid = time[1:-1]

                # -------------------------------------------------
                # RA/Dec speed magnitude vs Maria input speed
                # -------------------------------------------------

                plt.figure(figsize=(10, 6))

                plt.plot(
                    time_mid,
                    radec_speed_total,
                    lw=0.8,
                    color="blue",
                    label=r"$\sqrt{v_\mathrm{RA}^2 + v_\mathrm{Dec}^2}$"
                )

                plt.axhline(
                    spd,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="Maria input speed"
                )

                plt.xlabel("Time (s)")
                plt.ylabel("Speed magnitude (deg/s)")
                plt.title(
                    f"RA/Dec Speed Magnitude vs Maria Input Speed\n"
                    f"{SCAN_PATTERN}, Detector {det_idx}, Speed={spd:.2f} deg/s, Elev={ELEV_LABEL}"
                )

                plt.grid(True)
                plt.legend()

                simple_ccat.savefig(
                    ANALYSIS_OUTDIR,
                    f"{run_prefix}_detector_{det_idx}_radec_speed_magnitude_vs_input_speed.png"
                )

                plt.close("all")


                # -------------------------------------------------
                # Projected Az/El speed magnitude vs Maria input speed
                # -------------------------------------------------

                plt.figure(figsize=(10, 6))

                plt.plot(
                    time_mid,
                    projected_speed_total,
                    lw=0.8,
                    color="green",
                    label=r"$\sqrt{v_\mathrm{Az,proj}^2 + v_\mathrm{El}^2}$"
                )

                plt.axhline(
                    spd,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="Maria input speed"
                )

                plt.xlabel("Time (s)")
                plt.ylabel("Speed magnitude (deg/s)")
                plt.title(
                    f"Projected Az/El Speed Magnitude vs Maria Input Speed\n"
                    f"{SCAN_PATTERN}, Detector {det_idx}, Speed={spd:.2f} deg/s, Elev={ELEV_LABEL}"
                )

                plt.grid(True)
                plt.legend()

                simple_ccat.savefig(
                    ANALYSIS_OUTDIR,
                    f"{run_prefix}_detector_{det_idx}_projected_azel_speed_magnitude_vs_input_speed.png"
                )

                plt.close("all")


                # -------------------------------------------------
                # Motor Az/El speed magnitude vs Maria input speed
                # -------------------------------------------------

                plt.figure(figsize=(10, 6))

                plt.plot(
                    time_mid,
                    motor_speed_total,
                    lw=0.8,
                    color="purple",
                    label=r"$\sqrt{v_\mathrm{Az,motor}^2 + v_\mathrm{El,motor}^2}$"
                )

                plt.axhline(
                    spd,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="Maria input speed"
                )

                plt.xlabel("Time (s)")
                plt.ylabel("Speed magnitude (deg/s)")
                plt.title(
                    f"Motor Speed Magnitude vs Maria Input Speed\n"
                    f"{SCAN_PATTERN}, Detector {det_idx}, Speed={spd:.2f} deg/s, Elev={ELEV_LABEL}"
                )

                plt.grid(True)
                plt.legend()

                simple_ccat.savefig(
                    ANALYSIS_OUTDIR,
                    f"{run_prefix}_detector_{det_idx}_motor_speed_magnitude_vs_input_speed.png"
                )

                plt.close("all")


                # motor_jerk_az = (
                #     ((az_deg_track[4:] - 2*az_deg_track[3:-1] + 2*az_deg_track[1:-3] - az_deg_track[:-4])
                #     / (2 * dt**3) )
                # )

                global_max_az_vel = max(
                    global_max_az_vel,
                    np.nanmax(np.abs(motor_velocity_az))
                )

                global_max_el_vel = max(
                    global_max_el_vel,
                    np.nanmax(np.abs(velocity_el))
                )

                global_max_az_acc = max(
                    global_max_az_acc,
                    np.nanmax(np.abs(motor_acceleration_az))
                )

                global_max_el_acc = max(
                    global_max_el_acc,
                    np.nanmax(np.abs(acceleration_el))
                )

                # global_max_az_jerk = max(
                #     global_max_az_jerk,
                #     np.nanmax(np.abs(motor_jerk_az))
                # )

                # gloabl_max_el_jerk = max(
                #     global_max_el_jerk,
                #     np.nanmax(np.abs(jerk_el))
                # )

            mean_elevation = np.nanmean(mean_el_list)

            az_speed_limit = 3.0
            el_speed_limit = 1.5
            az_acc_limit = 6.0
            el_acc_limit = 1.5

            passes_limits = (
                global_max_az_vel < az_speed_limit
                and global_max_el_vel < el_speed_limit
                and global_max_az_acc < az_acc_limit
                and global_max_el_acc < el_acc_limit
            )

            # ========================================================================
            # Store motion summary for this speed
            # ========================================================================

            motion_results.append({
                "elevation_label": ELEV_LABEL,
                "start_time": START_TIME,
                "mean_elevation_deg": mean_elevation,
                "input_speed_deg_s": spd,

                "max_projected_az_velocity_deg_s": np.nanmax(np.abs(projected_velocity_az)),
                "max_el_velocity_deg_s": global_max_el_vel,
                "max_projected_az_acceleration_deg_s2": np.nanmax(np.abs(projected_acceleration_az)),
                "max_el_acceleration_deg_s2": global_max_el_acc,

                "max_motor_az_velocity_deg_s": global_max_az_vel,
                "max_motor_el_velocity_deg_s": global_max_el_vel,
                "max_motor_az_acceleration_deg_s2": global_max_az_acc,
                "max_motor_el_acceleration_deg_s2": global_max_el_acc,

                "max_radec_speed_deg_s": np.nanmax(np.abs(radec_speed_total)),
                "max_projected_azel_speed_deg_s": np.nanmax(np.abs(projected_speed_total)),
                "max_motor_speed_deg_s": np.nanmax(np.abs(motor_speed_total)),

                "passes_limits": passes_limits,
            })

            print("\nMotion summary")
            print("--------------")
            print(f"Mean elevation: {mean_elevation:.2f} deg")
            print(f"Input speed: {spd:.2f} deg/s")
            print(f"Max AZ velocity: {global_max_az_vel:.3f} deg/s")
            print(f"Max EL velocity: {global_max_el_vel:.3f} deg/s")
            print(f"Max AZ acceleration: {global_max_az_acc:.3f} deg/s^2")
            print(f"Max EL acceleration: {global_max_el_acc:.3f} deg/s^2")
            print(f"Passes limits: {passes_limits}")


    # ========================================================================
    # Save single-elevation speed analysis table
    # ========================================================================

            import csv 
            import pandas as pd

            if len(motion_results) == 0:
                        print(f"No motion results for elevation range {ELEV_LABEL}")
                        raise SystemExit("No results to save, exiting.")

            csv_path = SPEED_CSV_DIR / f"maria_{SCAN_PATTERN.lower()}_speed_elevation_motion_limits_{ELEV_LABEL}.csv"

            df = pd.DataFrame(motion_results)
            df.to_csv(csv_path, index=False)

            print(f"\nSaved {SCAN_PATTERN} motion results to {csv_path}")

    csv_files = sorted(
        SPEED_CSV_DIR.glob(f"maria_{SCAN_PATTERN.lower()}_speed_elevation_motion_limits_*.csv")
    )

    dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if "elevation_label" not in df.columns:
            label = csv_file.stem.split("_")[-1]
            df["elevation_label"] = label

        dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError("No CSV files found for combined analysis")
    
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_csv_path = SPEED_CSV_DIR / f"maria_{SCAN_PATTERN.lower()}_speed_elevation_motion_limits_combined.csv"
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"Saved combined CSV to: {combined_csv_path}")

    quantities = {
            "max_motor_az_velocity_deg_s": {
                "label": "Maximum Motor AZ Velocity",
                "unit": "deg/s",
                "limit": 3.0,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_motor_az_velocity_vs_input_speed.png",
            },
            "max_motor_el_velocity_deg_s": {
                "label": "Maximum Motor EL Velocity",
                "unit": "deg/s",
                "limit": 1.5,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_motor_el_velocity_vs_input_speed.png",
            },
            "max_motor_az_acceleration_deg_s2": {
                "label": "Maximum Motor AZ Acceleration",
                "unit": "deg/s²",
                "limit": 6.0,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_motor_az_acceleration_vs_input_speed.png",
            },
            "max_motor_el_acceleration_deg_s2": {
                "label": "Maximum Motor EL Acceleration",
                "unit": "deg/s²",
                "limit": 1.5,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_motor_el_acceleration_vs_input_speed.png",
            },
            "max_projected_az_velocity_deg_s": {
                "label": "Maximum Projected AZ Velocity",
                "unit": "deg/s",
                "limit": None,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_projected_az_velocity_vs_input_speed.png",
            },
            "max_projected_az_acceleration_deg_s2": {
                "label": "Maximum Projected AZ Acceleration",
                "unit": "deg/s²",
                "limit": None,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_projected_az_acceleration_vs_input_speed.png",
            },
            "max_radec_speed_deg_s": {
                "label": "Maximum RA/Dec Speed Magnitude",
                "unit": "deg/s",
                "limit": None,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_radec_speed_magnitude_vs_input_speed.png",
            },
            "max_projected_azel_speed_deg_s": {
                "label": "Maximum Projected Az/El Speed Magnitude",
                "unit": "deg/s",
                "limit": None,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_projected_azel_speed_magnitude_vs_input_speed.png",
            },
            "max_motor_speed_deg_s": {
                "label": "Maximum Motor Speed Magnitude",
                "unit": "deg/s",
                "limit": None,
                "filename": f"{SCAN_PATTERN.lower()}_combined_max_motor_speed_magnitude_vs_input_speed.png",
            },
    }

    for column, info in quantities.items():

            plt.figure(figsize=(8, 6))

            for elevation_label, group in combined_df.groupby("elevation_label"):
                group = group.sort_values("input_speed_deg_s")

                plt.plot(
                    group["input_speed_deg_s"],
                    group[column],
                    marker="o",
                    linewidth=2,
                    alpha=0.8,
                    label=elevation_label,
                )

            if info["limit"] is not None:
                plt.axhline(
                    info["limit"],
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label="Motion limit",
                )

            plt.xlabel("Maria input speed (deg/s)")
            plt.ylabel(f"{info['label']} ({info['unit']})")
            plt.title(f"{info['label']} vs Maria Input Speed")
            plt.grid(True, alpha=0.3)
            plt.legend(title="Elevation range")

            plt.savefig(
                COMBINED_PLOT_DIR / info["filename"],
                dpi=300,
                bbox_inches="tight",
            )

            plt.close("all")

    print(f"Saved combined {SCAN_PATTERN} plots to: {COMBINED_PLOT_DIR}")
    
    raise SystemExit("Finished motion limit analysis, exiting before TOD plotting")

    # ========================================================================
    # Single-elevation diagnostic plots
    # ========================================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["input_speed_deg_s"],
        df["max_el_acceleration_deg_s2"],
        marker="o",
    )

    plt.axhline(
        1.5,
        color="black",
        linestyle="--",
        label="CCAT max EL accel"
    )

    plt.xlabel("Maria input speed (deg/s)")
    plt.ylabel("Maximum EL acceleration (deg/s²)")
    plt.title(
        f"Maximum EL Acceleration vs Maria Input Speed\n{ELEV_LABEL}"
    )

    plt.grid(True)
    plt.legend()
    plt.savefig(
        ANALYSIS_OUTDIR / f"{PREFIX}_el_acceleration_vs_input_speed.png"
    )
    plt.close("all")

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["input_speed_deg_s"],
        df["max_az_acceleration_deg_s2"],
        marker="o",
    )

    plt.axhline(
        6.0,
        color="black",
        linestyle="--",
        label="CCAT max AZ accel"
    )

    plt.xlabel("Maria input speed (deg/s)")
    plt.ylabel("Maximum AZ acceleration (deg/s²)")
    plt.title(
        f"Maximum AZ Acceleration vs Maria Input Speed\n{ELEV_LABEL}"
    )

    plt.grid(True)
    plt.legend()
    plt.savefig(
        ANALYSIS_OUTDIR / f"{PREFIX}_az_acceleration_vs_input_speed.png"
    )
    plt.close("all")

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["input_speed_deg_s"],
        df["max_az_velocity_deg_s"],
        marker="o",
    )

    plt.axhline(
        3.0,
        color="red",
        linestyle="--",
        label="CCAT max AZ vel"
    )

    plt.xlabel("Maria input speed (deg/s)")
    plt.ylabel("Maximum AZ velocity (deg/s)")
    plt.title(
        f"Maximum AZ Velocity vs Maria Input Speed\n{ELEV_LABEL}"
    )

    plt.grid(True)
    plt.legend()
    plt.savefig(
        ANALYSIS_OUTDIR / f"{PREFIX}_az_velocity_vs_input_speed.png"
    )
    plt.close("all")    

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["input_speed_deg_s"],
        df["max_el_velocity_deg_s"],
        marker="o",
    )

    plt.axhline(
        1.5,
        color="red",
        linestyle="--",
        label="CCAT max EL vel"
    )

    plt.xlabel("Maria input speed (deg/s)")
    plt.ylabel("Maximum EL velocity (deg/s)")
    plt.title(
        f"Maximum EL Velocity vs Maria Input Speed\n{ELEV_LABEL}"
    )

    plt.grid(True)
    plt.legend()
    plt.savefig(
        ANALYSIS_OUTDIR / f"{PREFIX}_el_velocity_vs_input_speed.png"
    )
    plt.close("all")

    # ========================================================================
    # Second combined CSV block from original script
    # ========================================================================

    csv_path_35 = Path("outputs/OrionA_speed_tests/speed_csv/maria_speed_elevation_motion_limits_35-45")

    csv_path_45 = Path("outputs/OrionA_speed_tests/speed_csv/maria_speed_elevation_motion_limits_45-55")
    
    csv_path_55 = Path("outputs/OrionA_speed_tests/speed_csv/maria_speed_elevation_motion_limits_55-65")

    csv_path_65 = Path("outputs/OrionA_speed_tests/speed_csv/maria_speed_elevation_motion_limits_65-75")

    dfs = [
        pd.read_csv(csv_path_35),
        pd.read_csv(csv_path_45),
        pd.read_csv(csv_path_55),
        pd.read_csv(csv_path_65)
    ]

    combined_df = pd.concat(dfs, ignore_index=True)

    output_path = Path("outputs/OrionA_speed_tests/speed_csv/maria_speed_elevation_motion_limits_combined.csv")

    print(f"Saved cmobined CSV to: {output_path}")
    print(f"Total rows: {len(combined_df)}")

    combined_df.to_csv(output_path, index=False)
    raise SystemExit("Finished motion limit analysis, exiting before TOD plotting")


    # ========================================================================
    # Part 4: detailed TOD checks and detector trajectory plots
    # ========================================================================

    fits_path = TOD_OUTDIR / f"{PREFIX}_dim_reduced_tods.fits"

    print("\nChecking FITS output")
    print("--------------------")
    print("Expected path:", fits_path)
    print("Exists:", fits_path.exists())

    if TOD_OUTDIR.exists():
        print("Files in directory:")
        for f in TOD_OUTDIR.iterdir():
            print("  ", f)
    else:
        print("TOD directory does not exist")

        if not fits_path.exists():
            print("FITS file does not exist. Check if the simulation ran successfully and saved the output.")
            raise FileNotFoundError(f"Expected FITS file not found at {fits_path}")
        

    site = maria.get_site("cerro_chajnantor", altitude=5600)

    band = Band(
        name="m2/f093",
        center = 280e9,
        width = 60e9,
        efficiency = eta,
        NET_CMB = 13e-6,
        knee = 1.0,
        gain_error = 5e-2
    )


    # tod = TOD.from_fits(TOD_OUTDIR / f"{PREFIX}_dim_reduced_tods.fits", format = "CCAT")

    tod = maria.tod.load(TOD_OUTDIR / f"{PREFIX}_dim_reduced_tods.fits", site = site, bands = [band])
    

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


    # ========================================================================
    # Detailed RA/Dec and Az/El velocity/acceleration plots
    # ========================================================================

    #---------------------------------------------------------
    #--- Velocity and Acceleration Detector Tracking Plots ---
    #---------------------------------------------------------

    for det_idx in [604]:

        ra_deg_track = np.rad2deg(tod.ra[det_idx, :])
        dec_deg_track = np.rad2deg(tod.dec[det_idx, :])

        az_deg_track = np.rad2deg(tod.az[det_idx, :])
        el_deg_track = np.rad2deg(tod.el[det_idx, :])

        time = np.arange(len(ra_deg_track)) / SAMPLE_RATE_HZ

        dt = 1 / SAMPLE_RATE_HZ\
        
        valid = np.isfinite(ra_deg_track) & np.isfinite(dec_deg_track)

        ra_deg_track = ra_deg_track[valid]
        dec_deg_track = dec_deg_track[valid]

        az_deg_track = az_deg_track[valid]
        el_deg_track = el_deg_track[valid]

        time = time[valid]

        velocity_ra = np.cos(np.radians(dec_deg_track[1:-1])) * (ra_deg_track[2:] - ra_deg_track[:-2]) / (2 * dt)
        velocity_dec = (dec_deg_track[2:] - dec_deg_track[:-2]) / (2 * dt)

        acceleration_ra = np.cos(np.radians(dec_deg_track[1:-1])) * (ra_deg_track[2:] - 2 * ra_deg_track[1:-1] + ra_deg_track[:-2]) / (dt ** 2)
        acceleration_dec = (dec_deg_track[2:] - 2 * dec_deg_track[1:-1] + dec_deg_track[:-2]) / (dt ** 2)

        velocity_az = np.cos(np.radians(el_deg_track[1:-1])) * (az_deg_track[2:] - az_deg_track[:-2]) / (2 * dt)
        velocity_el = (el_deg_track[2:] - el_deg_track[:-2]) / (2 * dt)

        acceleration_az = np.cos(np.radians(el_deg_track[1:-1])) * (az_deg_track[2:] - 2 * az_deg_track[1:-1] + az_deg_track[:-2]) / (dt ** 2)
        acceleration_el = (el_deg_track[2:] - 2 * el_deg_track[1:-1] + el_deg_track[:-2]) / (dt ** 2)

        magnitude_velocity_ra_dec = np.sqrt(velocity_ra**2 + velocity_dec**2)
        magnitude_velocity_az_el = np.sqrt(velocity_az**2 + velocity_el**2)

        magnitude_acceleration_ra_dec = np.sqrt(acceleration_ra**2 + acceleration_dec**2)
        magnitude_acceleration_az_el = np.sqrt(acceleration_az**2 + acceleration_el**2)

        motor_velocity_az = ((az_deg_track[2:] - az_deg_track[:-2]) / (2 * dt))

        motor_velocity_el = velocity_el

        motor_velocity_total = np.sqrt(motor_velocity_az**2 + motor_velocity_el**2)


        mag_ratio_vel = magnitude_velocity_az_el / magnitude_velocity_ra_dec

        mag_ratio_acc = magnitude_acceleration_az_el / magnitude_acceleration_ra_dec



        

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], velocity_ra, lw=0.5, color="blue", label="RA Velocity")
        plt.plot(time[1:-1], velocity_dec, lw=0.5, color="orange", label="Dec Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], velocity_ra, lw=0.5, color="green", label="RA Velocity")
        plt.plot(time[1:-1], velocity_dec, lw=0.5, color="blue", label="Dec Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Zoomed In Detector {det_idx} RA/Dec Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_ra_dec_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], acceleration_ra, lw=0.5, color="blue", label="RA Acceleration")
        plt.plot(time[1:-1], acceleration_dec, lw=0.5, color="orange", label="Dec Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Detector {det_idx} Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], acceleration_ra, lw=0.5, color="green", label="RA Acceleration")
        plt.plot(time[1:-1], acceleration_dec, lw=0.5, color="blue", label="Dec Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Zoomed In Detector {det_idx} RA/Dec Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_ra_dec_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], velocity_az, lw=0.5, color="green", label="Az Velocity")
        plt.plot(time[1:-1], velocity_el, lw=0.5, color="red", label="El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Az/El Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_azel_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], velocity_az, lw=0.5, color="green", label="Az Velocity")
        plt.plot(time[1:-1], velocity_el, lw=0.5, color="red", label="El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Zoomed In Detector {det_idx} Az/El Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_azel_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], acceleration_az, lw=0.5, color="green", label="Az Acceleration")
        plt.plot(time[1:-1], acceleration_el, lw=0.5, color="red", label="El Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Detector {det_idx} Az/El Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
             f"{PREFIX}_detector_{det_idx}_azel_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")
    
        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], acceleration_az, lw=0.5, color="green", label="Az Acceleration")
        plt.plot(time[1:-1], acceleration_el, lw=0.5, color="red", label="El Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Zoomed In Detector {det_idx} Az/El Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_azel_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], magnitude_velocity_ra_dec, lw=0.5, color="blue", label="RA/Dec Velocity")
        plt.plot(time[1:-1], magnitude_velocity_az_el, lw=0.5, color="green", label="Az/El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Velocity Magnitude Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
             f"{PREFIX}_detector_{det_idx}_velocity_magnitude_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], magnitude_velocity_ra_dec, lw=0.5, color="blue", label="RA/Dec Velocity")
        plt.plot(time[1:-1], magnitude_velocity_az_el, lw=0.5, color="green", label="Az/El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Zoomed In Detector {det_idx} RA/Dec Velocity Magnitude vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_ra_dec_velocity_magnitude_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], mag_ratio_vel, lw=0.5, color="purple", label="Az/El Velocity / RA/Dec Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity Ratio")
        plt.title(
            f"Detector {det_idx} Velocity Ratio Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_velocity_ratio_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], mag_ratio_vel, lw=0.5, color="purple", label="Az/El Velocity / RA/Dec Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity Ratio")
        plt.title(
            f"Zoomed In Detector {det_idx} Velocity Ratio Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_velocity_ratio_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], mag_ratio_acc, lw=0.5, color="purple", label="Az/El Acceleration / RA/Dec Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration Ratio")
        plt.title(
            f"Detector {det_idx} Acceleration Ratio Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_acceleration_ratio_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], mag_ratio_acc, lw=0.5, color="purple", label="Az/El Acceleration / RA/Dec Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration Ratio")
        plt.title(
            f"Zoomed In Detector {det_idx} Acceleration Ratio Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_acceleration_ratio_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], motor_velocity_az, lw=0.5, color="green", label="Motor Az Velocity")
        plt.plot(time[1:-1], motor_velocity_el, lw=0.5, color="red", label="Motor El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Motor Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_motor_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], motor_velocity_az, lw=0.5, color="green", label="Motor Az Velocity")
        plt.plot(time[1:-1], motor_velocity_el, lw=0.5, color="red", label="Motor El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Zoomed In Detector {det_idx} Motor Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_motor_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], magnitude_acceleration_ra_dec, lw=0.5, color="blue", label="RA/Dec Acceleration")
        plt.plot(time[1:-1], magnitude_acceleration_az_el, lw=0.5, color="green", label="Az/El Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Detector {det_idx} Acceleration Magnitude Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
             f"{PREFIX}_detector_{det_idx}_acceleration_magnitude_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], magnitude_acceleration_ra_dec, lw=0.5, color="blue", label="RA/Dec Acceleration")
        plt.plot(time[1:-1], magnitude_acceleration_az_el, lw=0.5, color="green", label="Az/El Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Zoomed In Detector {det_idx} Acceleration Magnitude Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.xlim(100, 200)  # Zoom in on the 100-200 second range
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_zoomed_detector_{det_idx}_acceleration_magnitude_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")
        # -------------------------------------------------
        # RA Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_ra, lw=0.5, color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("RA Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} RA Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_ra_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")




        # -------------------------------------------------
        # Dec Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_dec, lw=0.5, color="orange")

        plt.xlabel("Time (s)")
        plt.ylabel("Dec Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} Dec Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_dec_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # RA Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_ra, lw=0.5, color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("RA Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} RA Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_ra_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Dec Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_dec, lw=0.5, color="orange")

        plt.xlabel("Time (s)")
        plt.ylabel("Dec Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} Dec Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_dec_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Projected Az Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_az, lw=0.5, color="green")

        plt.xlabel("Time (s)")
        plt.ylabel("Projected Az Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} Projected Az Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_projected_az_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Elevation Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_el, lw=0.5, color="red")

        plt.xlabel("Time (s)")
        plt.ylabel("Elevation Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} Elevation Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_el_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Projected Az Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_az, lw=0.5, color="green")

        plt.xlabel("Time (s)")
        plt.ylabel("Projected Az Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} Projected Az Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_projected_az_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Elevation Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_el, lw=0.5, color="red")

        plt.xlabel("Time (s)")
        plt.ylabel("Elevation Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} Elevation Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_el_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # RA/Dec Velocity Magnitude
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], magnitude_velocity_ra_dec, lw=0.5, color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("RA/Dec Velocity Magnitude (degrees/s)")

        plt.title(
            f"Detector {det_idx} RA/Dec Velocity Magnitude vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_radec_velocity_magnitude_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Az/El Velocity Magnitude
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], magnitude_velocity_az_el, lw=0.5, color="green")

        plt.xlabel("Time (s)")
        plt.ylabel("Az/El Velocity Magnitude (degrees/s)")

        plt.title(
            f"Detector {det_idx} Az/El Velocity Magnitude vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_azel_velocity_magnitude_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")




    # ========================================================================
    # Part 5: detector power distribution and spatial power comparisons
    # ========================================================================

    P_det = tod.to("pW").signal
    P = np.asarray(P_det, dtype=np.float64)  # Convert to numpy array for calculations

    reloaded_power_list = []

    for det_idx in range(P.shape[0]):

        P_track = np.asarray(P[det_idx, :], dtype=np.float64)
        P_track_flat = P_track[np.isfinite(P_track)]

        if P_track_flat.size == 0:
            continue

        reloaded_power_list.extend(P_track_flat[::50])  # Append all valid power values to the list

    reloaded_power_array = np.asarray(reloaded_power_list)

    from scipy.stats import norm, skew

    mu = np.mean(reloaded_power_array)
    sigma = np.std(reloaded_power_array)


    print(f"Fitted Gaussian parameters: mu={mu:.2f} pW, sigma={sigma:.2f} pW")

    plt.figure(figsize=(8,6))

    counts, bins, _ = plt.hist(
        reloaded_power_array,
        bins=40,
        density=True,
        alpha=0.7,
    )

    # Gaussian fit
    x = np.linspace(bins.min(), bins.max(), 1000)
    gaussian = norm.pdf(x, mu, sigma)

    plt.plot(x, gaussian, linewidth=2)

    plt.axvline(mu, linestyle="--", linewidth=2, label="Mean")
    plt.axvline(mu + sigma, linestyle=":", linewidth=2, label=r"$+1\sigma$")
    plt.axvline(mu - sigma, linestyle=":", linewidth=2, label=r"$-1\sigma$")

    plt.xlabel("Detector Power (pW)")
    plt.ylabel("Normalized Counts")

    plt.title(
        "Redistribution of Detector Power Across All Detectors\n"
        f"Mean={mu:.4e} pW, σ={sigma:.4e} pW"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    simple_ccat.savefig(ANALYSIS_OUTDIR, f"{PREFIX}_reloaded_detector_power_distribution_PWV{PWV_MM:.2f}.png")  
    plt.close("all")



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
    # ========================================================================
    # Detailed power plots for selected neighbouring detector pair
    # ========================================================================

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

    
    # ========================================================================
    # Part 6: detector-by-detector power statistics and frequency response
    # ========================================================================

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


    # ========================================================================
    # Reusable histogram helper for detector distributions
    # ========================================================================

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
        f"{PREFIX}_bogus_detector_gaussian_delta_f_over_fwhm_histogram_PWV{PWV_MM:.2f}.png"
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

    # delta_mean_list = []

    # for delta_track in delta_f_over_fwhm_by_detector:
    #     delta_mean = np.nanmean(delta_track)
    #     delta_mean_list.append(delta_mean)

    # plt.figure(figsize=(8,6))
    # plt.hist(delta_mean_list, bins=30, alpha=0.7, edgecolor="k")
    # plt.xlabel(r"Mean $\Delta f / \mathrm{FWHM}$")
    # plt.ylabel("Number of Detectors")
    # plt.title(
    #     f"Distribution of Mean Fractional Frequency Shifts for Detectors\n"
    #     f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    # )
    # plt.grid(True)
    # simple_ccat.savefig(
    #     ANALYSIS_OUTDIR,
    #     f"{PREFIX}_detector_mean_fractional_frequency_shift_histogram_PWV{PWV_MM:.2f}.png"
    # )

    # plt.close("all")

    # ========================================================================
    # Individual detector fractional-frequency-shift histograms
    # ========================================================================

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

    # ========================================================================
    # All-detector fractional-frequency-shift distributions
    # ========================================================================

    two_delta_f_over_fwhm_list = []
    half_delta_f_over_fwhm_list = []
    delta_f_over_fwhm_by_detector_list = []
    def R(P_W):
        return R_0 / np.sqrt(1 + P_W / P_0)

    for det_idx in range(P.shape[0]):
        P_track_pW = np.asarray(P[det_idx, :], dtype=np.float64)
        valid = np.isfinite(P_track_pW)

        P_track_pW = P_track_pW[valid]
        P_track_W = P_track_pW * 1e-12

        P_ref_W = np.nanmean(P_track_W) #mean power for individual reference

        delta_f_over_fwhm = Q_r * R(P_ref_W) * (P_track_W - P_ref_W)


        delta_f_over_fwhm_by_detector_list.append(delta_f_over_fwhm)
        two_delta_f_over_fwhm = np.max(delta_f_over_fwhm) - np.min(delta_f_over_fwhm)
        half_delta_f_over_fwhm = np.abs(two_delta_f_over_fwhm / 2)
        two_delta_f_over_fwhm_list.append(two_delta_f_over_fwhm)
        half_delta_f_over_fwhm_list.append(half_delta_f_over_fwhm)

    delta_f_over_fwhm_all = np.concatenate(delta_f_over_fwhm_by_detector_list)


    plt.figure(figsize=(8,6))
    plt.hist(delta_f_over_fwhm_all, bins=50, alpha=0.7, edgecolor="k")
    plt.xlabel(r"Estimated $\Delta f / \mathrm{FWHM}$")
    plt.ylabel("Number of Samples")
    plt.title(
        f"Distribution of Estimated Fractional Frequency Shifts for All Detectors\n"
        f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
    )
    plt.grid(True)
    simple_ccat.savefig(
        ANALYSIS_OUTDIR,
        f"{PREFIX}_all_detectors_fractional_frequency_shift_histogram_PWV{PWV_MM:.2f}.png"
    )
    plt.close("all")
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

# ========================================================================
# Part 7: Az/El tracks colour-coded by frequency response
# ========================================================================

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

    # ========================================================================
    # Individual detector time/elevation response plots
    # ========================================================================

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

    # ========================================================================
    # Part 8: velocity/acceleration plots with detector power overlay
    # ========================================================================

        #---------------------------------------------------------
    #--- Velocity and Acceleration Detector Tracking Plots ---
    #---------------------------------------------------------

    for det_idx in [604]:

        P_track_pW = np.asarray(P[det_idx, :], dtype=np.float64)

        ra_deg_track = np.rad2deg(tod.ra[det_idx, :])
        dec_deg_track = np.rad2deg(tod.dec[det_idx, :])

        az_deg_track = np.rad2deg(tod.az[det_idx, :])
        el_deg_track = np.rad2deg(tod.el[det_idx, :])

        time = np.arange(len(ra_deg_track)) / SAMPLE_RATE_HZ

        dt = 1 / SAMPLE_RATE_HZ\
        
        valid = (
            np.isfinite(P_track_pW)
            & np.isfinite(ra_deg_track)
            & np.isfinite(dec_deg_track)
            & np.isfinite(az_deg_track)
            & np.isfinite(el_deg_track)
        )

        P_track_pW = P_track_pW[valid]

        ra_deg_track = ra_deg_track[valid]
        dec_deg_track = dec_deg_track[valid]

        az_deg_track = az_deg_track[valid]
        el_deg_track = el_deg_track[valid]

        time = time[valid]

        velocity_ra = np.cos(np.radians(dec_deg_track[1:-1])) * (ra_deg_track[2:] - ra_deg_track[:-2]) / (2 * dt)
        velocity_dec = (dec_deg_track[2:] - dec_deg_track[:-2]) / (2 * dt)

        acceleration_ra = np.cos(np.radians(dec_deg_track[1:-1])) * (ra_deg_track[2:] - 2 * ra_deg_track[1:-1] + ra_deg_track[:-2]) / (dt ** 2)
        acceleration_dec = (dec_deg_track[2:] - 2 * dec_deg_track[1:-1] + dec_deg_track[:-2]) / (dt ** 2)

        velocity_az = np.cos(np.radians(el_deg_track[1:-1])) * (az_deg_track[2:] - az_deg_track[:-2]) / (2 * dt)
        velocity_el = (el_deg_track[2:] - el_deg_track[:-2]) / (2 * dt)

        acceleration_az = np.cos(np.radians(el_deg_track[1:-1])) * (az_deg_track[2:] - 2 * az_deg_track[1:-1] + az_deg_track[:-2]) / (dt ** 2)
        acceleration_el = (el_deg_track[2:] - 2 * el_deg_track[1:-1] + el_deg_track[:-2]) / (dt ** 2)

        magnitude_velocity_ra_dec = np.sqrt(velocity_ra**2 + velocity_dec**2)
        magnitude_velocity_az_el = np.sqrt(velocity_az**2 + velocity_el**2)

        magnitude_acceleration_ra_dec = np.sqrt(acceleration_ra**2 + acceleration_dec**2)
        magnitude_acceleration_az_el = np.sqrt(acceleration_az**2 + acceleration_el**2)

        motor_velocity_az = ((az_deg_track[2:] - az_deg_track[:-2]) / (2 * dt))

        motor_velocity_el = velocity_el

        motor_velocity_total = np.sqrt(motor_velocity_az**2 + motor_velocity_el**2)

        P_mid_pW = P_track_pW[1:-1]


        

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], velocity_ra, lw=0.5, color="blue", label="RA Velocity")
        plt.plot(time[1:-1], velocity_dec, lw=0.5, color="orange", label="Dec Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], acceleration_ra, lw=0.5, color="blue", label="RA Acceleration")
        plt.plot(time[1:-1], acceleration_dec, lw=0.5, color="orange", label="Dec Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Detector {det_idx} Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], velocity_az, lw=0.5, color="green", label="Az Velocity")
        plt.plot(time[1:-1], velocity_el, lw=0.5, color="red", label="El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Az/El Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_azel_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], acceleration_az, lw=0.5, color="green", label="Az Acceleration")
        plt.plot(time[1:-1], acceleration_el, lw=0.5, color="red", label="El Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Detector {det_idx} Az/El Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
             f"{PREFIX}_detector_{det_idx}_azel_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], magnitude_velocity_ra_dec, lw=0.5, color="blue", label="RA/Dec Velocity")
        plt.plot(time[1:-1], magnitude_velocity_az_el, lw=0.5, color="green", label="Az/El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Velocity Magnitude Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
             f"{PREFIX}_detector_{det_idx}_velocity_magnitude_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], motor_velocity_az, lw=0.5, color="green", label="Motor Az Velocity")
        plt.plot(time[1:-1], motor_velocity_el, lw=0.5, color="red", label="Motor El Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (degrees/s)")
        plt.title(
            f"Detector {det_idx} Motor Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_motor_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(time[1:-1], magnitude_acceleration_ra_dec, lw=0.5, color="blue", label="RA/Dec Acceleration")
        plt.plot(time[1:-1], magnitude_acceleration_az_el, lw=0.5, color="green", label="Az/El Acceleration")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (degrees/s^2)")
        plt.title(
            f"Detector {det_idx} Acceleration Magnitude Comparison\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )
        plt.grid(True)
        plt.legend()
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
             f"{PREFIX}_detector_{det_idx}_acceleration_magnitude_comparison_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        # -------------------------------------------------
        # RA Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_ra, lw=0.5, color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("RA Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} RA Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_ra_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Dec Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_dec, lw=0.5, color="orange")

        plt.xlabel("Time (s)")
        plt.ylabel("Dec Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} Dec Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_dec_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # RA Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_ra, lw=0.5, color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("RA Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} RA Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_ra_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Dec Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_dec, lw=0.5, color="orange")

        plt.xlabel("Time (s)")
        plt.ylabel("Dec Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} Dec Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_dec_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Projected Az Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_az, lw=0.5, color="green")

        plt.xlabel("Time (s)")
        plt.ylabel("Projected Az Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} Projected Az Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_projected_az_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Elevation Velocity
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], velocity_el, lw=0.5, color="red")

        plt.xlabel("Time (s)")
        plt.ylabel("Elevation Velocity (degrees/s)")

        plt.title(
            f"Detector {det_idx} Elevation Velocity vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_el_velocity_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Projected Az Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_az, lw=0.5, color="green")

        plt.xlabel("Time (s)")
        plt.ylabel("Projected Az Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} Projected Az Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_projected_az_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Elevation Acceleration
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], acceleration_el, lw=0.5, color="red")

        plt.xlabel("Time (s)")
        plt.ylabel("Elevation Acceleration (degrees/s$^2$)")

        plt.title(
            f"Detector {det_idx} Elevation Acceleration vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_el_acceleration_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # RA/Dec Velocity Magnitude
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], magnitude_velocity_ra_dec, lw=0.5, color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("RA/Dec Velocity Magnitude (degrees/s)")

        plt.title(
            f"Detector {det_idx} RA/Dec Velocity Magnitude vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_radec_velocity_magnitude_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")


        # -------------------------------------------------
        # Az/El Velocity Magnitude
        # -------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(time[1:-1], magnitude_velocity_az_el, lw=0.5, color="green")

        plt.xlabel("Time (s)")
        plt.ylabel("Az/El Velocity Magnitude (degrees/s)")

        plt.title(
            f"Detector {det_idx} Az/El Velocity Magnitude vs Time\n"
            f"PWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}"
        )

        plt.grid(True)

        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_azel_velocity_magnitude_vs_time_PWV{PWV_MM:.2f}.png"
        )

        plt.close("all")



        plt.figure(figsize=(10, 6))
        plt.plot(velocity_el, P_mid_pW, lw=0.5, color="black")
        plt.xlabel("Detector Power (pW)")
        plt.ylabel("Velocity (deg/s)")
        plt.title(f"Detector {det_idx} Velocity vs Power\nPWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}")
        plt.grid(True)
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_velocity_vs_power_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(velocity_az, P_mid_pW, lw=0.5, color="black")
        plt.xlabel("Detector Power (pW)")
        plt.ylabel("Velocity (deg/s)")
        plt.title(f"Detector {det_idx} Velocity vs Power\nPWV={PWV_MM:.2f} mm, $\\eta$={eta:.2f}")
        plt.grid(True)
        simple_ccat.savefig(
            ANALYSIS_OUTDIR,
            f"{PREFIX}_detector_{det_idx}_velocity_az_vs_power_PWV{PWV_MM:.2f}.png"
        )
        plt.close("all")



    signal = tod.signal.compute()
    ra = tod.ra
    dec = tod.dec
    time = tod.time
    el = tod.el
    az = tod.az

raise SystemExit("Test Complete")