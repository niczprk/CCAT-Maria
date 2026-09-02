"""
observing_strategies.py

Simple observing-strategy analysis using ONE combined CSV produced by
module_analysis.py.

The combined CSV is expected to contain:
    - observing configuration
    - detector linewidth metrics
    - coverage metrics
    - telescope motor motion metrics

The analysis then applies three criteria:

    detector pass
    + mechanical pass
    + coverage pass
    = overall observing suitability
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import maria
from maria.instrument import Band

from matplotlib.lines import Line2D
plot_all = False
RUN_DAISY_PERCENTILE_ANALYSIS = False
plot_multi_realization_p95 = True

# ============================================================
# USER SETTINGS
# ============================================================

# SUMMARY_CSV = Path(
#     "outputs/atmospheric_parameter_sweep/"
#     "all_band_pwv_elevation_summary_3.csv"
# )
SUMMARY_CSV = Path(
    "outputs/atmospheric_parameter_sweep/"
    "daisy_900s_five_realization_summary.csv"
)

OUTPUT_DIR = Path(
    "outputs/observing_strategy_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SUITABILITY LIMITS
# ============================================================

# Detector stability criterion.
LINEWIDTH_LIMIT = 0.75  # |Δf/FWHM| absolute difference in either direction from readout tone

# FYST mechanical limits.
AZ_VELOCITY_LIMIT = 3.0       # deg/s
EL_VELOCITY_LIMIT = 1.5       # deg/s

AZ_ACCELERATION_LIMIT = 6.0   # deg/s^2
EL_ACCELERATION_LIMIT = 1.5   # deg/s^2

# Initial coverage criterion.
MIN_COVERAGE_FRACTION = 0.90


# ============================================================
# LOAD COMBINED SUMMARY
# ============================================================

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(
        f"Combined summary CSV not found:\n{SUMMARY_CSV}"
    )

df = pd.read_csv(
    SUMMARY_CSV
)

print(
    f"Loaded observing-strategy summary: "
    f"{len(df)} rows"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    # Observing configuration
    "band_ghz",
    "pwv_mm",
    "scan_pattern",
    "elevation_min_deg",
    "elevation_max_deg",
    "mean_elevation_deg",
    "input_speed_deg_s",
    "map_size_deg",
    "duration_s",
    "realization",
    "p50_abs_delta_f_over_fwhm",
    "p95_abs_delta_f_over_fwhm",

    # Coverage
    "coverage_fraction",

    # Telescope mechanics
    "max_motor_az_velocity_deg_s",
    "max_motor_el_velocity_deg_s",
    "max_motor_az_acceleration_deg_s2",
    "max_motor_el_acceleration_deg_s2",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        "Combined summary CSV is missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CLEAN SCAN-PATTERN NAMES
# ============================================================

df["scan_pattern"] = (
    df["scan_pattern"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)


# ============================================================
# SUMMARIZE FIVE REALIZATIONS
# ============================================================

REALIZATION_GROUP_COLUMNS = [
    "band_ghz",
    "pwv_mm",
    "scan_pattern",
    "elevation_min_deg",
    "elevation_max_deg",
    "input_speed_deg_s",
    "map_size_deg",
    "duration_s",
]


def summarize_realization_group(group):

    p50_values = (
        group["p50_abs_delta_f_over_fwhm"]
        .dropna()
        .to_numpy()
    )

    p95_values = (
        group["p95_abs_delta_f_over_fwhm"]
        .dropna()
        .to_numpy()
    )

    return pd.Series({
        "n_realizations": len(group),

        # Median response across the 5 simulations
        "p50_realization_median": np.nanmedian(
            p50_values
        ),

        "p95_realization_median": np.nanmedian(
            p95_values
        ),

        # Full run-to-run range
        "p50_realization_min": np.nanmin(
            p50_values
        ),

        "p50_realization_max": np.nanmax(
            p50_values
        ),

        "p95_realization_min": np.nanmin(
            p95_values
        ),

        "p95_realization_max": np.nanmax(
            p95_values
        ),
    })


realization_summary_df = (
    df
    .groupby(
        REALIZATION_GROUP_COLUMNS
    )
    .apply(
        summarize_realization_group
    )
    .reset_index()
)

# ------------------------------------------------------------
# Check that all configurations have five realizations
# ------------------------------------------------------------

incomplete = realization_summary_df[
    realization_summary_df["n_realizations"]
    != 5
]

if not incomplete.empty:

    print(
        "\nWARNING: Some configurations do not "
        "contain five realizations:"
    )

    print(
        incomplete[
            REALIZATION_GROUP_COLUMNS
            + ["n_realizations"]
        ].to_string(index=False)
    )

REALIZATION_SUMMARY_CSV = (
    OUTPUT_DIR
    / "daisy_900s_five_realization_statistics.csv"
)

realization_summary_df.to_csv(
    REALIZATION_SUMMARY_CSV,
    index=False,
)

print(
    "\nSaved five-realization statistical summary:\n"
    f"{REALIZATION_SUMMARY_CSV}"
)


# ============================================================
# DETECTOR PASS / FAIL
# ============================================================

df["detector_pass"] = (
    df["p95_abs_delta_f_over_fwhm"]
    <= LINEWIDTH_LIMIT
)


# ============================================================
# COVERAGE PASS / FAIL
# ============================================================

df["coverage_pass"] = (
    df["coverage_fraction"]
    >= MIN_COVERAGE_FRACTION
)


# ============================================================
# MECHANICAL PASS / FAIL
# ============================================================

df["az_velocity_pass"] = (
    df["max_motor_az_velocity_deg_s"]
    <= AZ_VELOCITY_LIMIT
)

df["el_velocity_pass"] = (
    df["max_motor_el_velocity_deg_s"]
    <= EL_VELOCITY_LIMIT
)

df["az_acceleration_pass"] = (
    df["max_motor_az_acceleration_deg_s2"]
    <= AZ_ACCELERATION_LIMIT
)

df["el_acceleration_pass"] = (
    df["max_motor_el_acceleration_deg_s2"]
    <= EL_ACCELERATION_LIMIT
)

df["mechanical_pass"] = (
    df["az_velocity_pass"]
    & df["el_velocity_pass"]
    & df["az_acceleration_pass"]
    & df["el_acceleration_pass"]
)


# ============================================================
# OVERALL SUITABILITY
# ============================================================

df["overall_pass"] = (
    df["detector_pass"]
    & df["coverage_pass"]
    & df["mechanical_pass"]
)


def classify_run(row):
    """
    Give each run a simple human-readable classification.
    """

    failed = []

    if not row["detector_pass"]:
        failed.append("detector")

    if not row["coverage_pass"]:
        failed.append("coverage")

    if not row["mechanical_pass"]:
        failed.append("mechanical")

    if len(failed) == 0:
        return "Suitable"

    if len(failed) == 1:
        return f"{failed[0].capitalize()} limited"

    return "Multiple limits"


df["suitability_class"] = df.apply(
    classify_run,
    axis=1,
)


# ============================================================
# SORT RESULTS
# ============================================================

sort_columns = [
    "band_ghz",
    "scan_pattern",
    "duration_s",
    "pwv_mm",
    "elevation_min_deg",
]

df = (
    df
    .sort_values(sort_columns)
    .reset_index(drop=True)
)


# ============================================================
# SAVE FULL FEASIBILITY TABLE
# ============================================================

full_output_csv = (
    OUTPUT_DIR
    / "observing_strategy_feasibility.csv"
)

df.to_csv(
    full_output_csv,
    index=False,
)

print(
    f"\nSaved feasibility table:\n"
    f"{full_output_csv}"
)


# ============================================================
# SAVE COMPACT REPORT TABLE
# ============================================================

report_columns = [
    "band_ghz",
    "pwv_mm",
    "scan_pattern",
    "elevation_min_deg",
    "elevation_max_deg",
    "mean_elevation_deg",
    "duration_s",
    "input_speed_deg_s",
    "map_size_deg",

    "p95_abs_delta_f_over_fwhm",
    "coverage_fraction",

    "max_motor_az_velocity_deg_s",
    "max_motor_el_velocity_deg_s",
    "max_motor_az_acceleration_deg_s2",
    "max_motor_el_acceleration_deg_s2",

    "detector_pass",
    "coverage_pass",
    "mechanical_pass",
    "overall_pass",
    "suitability_class",
]

report_table = df[
    report_columns
].copy()

report_output_csv = (
    OUTPUT_DIR
    / "observing_strategy_report_table.csv"
)

report_table.to_csv(
    report_output_csv,
    index=False,
)

print(
    f"Saved compact report table:\n"
    f"{report_output_csv}"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nSuitability summary")
print("-" * 60)

print(
    df["suitability_class"]
    .value_counts()
    .to_string()
)

print("\nSuitable runs:")

suitable = df[
    df["overall_pass"]
]

if len(suitable) == 0:

    print(
        "No runs passed all three criteria."
    )

else:

    print(
        suitable[
            [
                "band_ghz",
                "pwv_mm",
                "scan_pattern",
                "elevation_min_deg",
                "elevation_max_deg",
                "duration_s",
                "p95_abs_delta_f_over_fwhm",
                "coverage_fraction",
                "max_motor_az_velocity_deg_s",
                "max_motor_el_velocity_deg_s",
                "max_motor_az_acceleration_deg_s2",
                "max_motor_el_acceleration_deg_s2",
            ]
        ]
        .to_string(
            index=False
        )
    )

if plot_all:
    # ============================================================
    # PLOT 1:
    # DETECTOR STABILITY VS PWV
    # ============================================================

    for (
        band,
        scan_pattern,
        duration,
    ), group in df.groupby(
        [
            "band_ghz",
            "scan_pattern",
            "duration_s",
        ]
    ):

        fig, axis = plt.subplots(
            figsize=(8, 6)
        )

        for (
            elev_min,
            elev_max,
        ), elevation_group in group.groupby(
            [
                "elevation_min_deg",
                "elevation_max_deg",
            ]
        ):

            elevation_group = (
                elevation_group
                .sort_values(
                    "pwv_mm"
                )
            )

            axis.plot(
                elevation_group[
                    "pwv_mm"
                ],
                elevation_group[
                    "p95_abs_delta_f_over_fwhm"
                ],
                marker="o",
                label=(
                    f"{elev_min:g}-"
                    f"{elev_max:g} deg"
                ),
            )

        axis.axhline(
            LINEWIDTH_LIMIT,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Linewidth limit "
                f"({LINEWIDTH_LIMIT:.2f})"
            ),
        )

        axis.set_xlabel(
            "PWV (mm)"
        )

        axis.set_ylabel(
            r"95th percentile "
            r"$|\Delta f/\mathrm{FWHM}|$"
        )

        axis.set_title(
            "Detector Stability versus PWV\n"
            f"{band:g} GHz, "
            f"{scan_pattern.replace('_', ' ').title()}, "
            f"{duration:g} s"
        )

        axis.grid(
            alpha=0.3
        )

        axis.legend(
            loc="best"
        )

        fig.tight_layout()

        filename = (
            f"{band:g}GHz_"
            f"{scan_pattern}_"
            f"{duration:g}s_"
            "detector_stability_vs_pwv.png"
        )

        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


    # ============================================================
    # PLOT 2:
    # COVERAGE BY SCAN PATTERN
    # ============================================================

    coverage_summary = (
        df
        .groupby(
            [
                "scan_pattern",
                "map_size_deg",
                "duration_s",
            ],
            as_index=False,
        )[
            "coverage_fraction"
        ]
        .median()
    )

    for (
        map_size,
        duration,
    ), group in coverage_summary.groupby(
        [
            "map_size_deg",
            "duration_s",
        ]
    ):

        fig, axis = plt.subplots(
            figsize=(9, 6)
        )

        axis.bar(
            group[
                "scan_pattern"
            ],
            group[
                "coverage_fraction"
            ],
        )

        axis.axhline(
            MIN_COVERAGE_FRACTION,
            linestyle="--",
            linewidth=1.5,
            label=(
                "Minimum coverage "
                f"({MIN_COVERAGE_FRACTION:.0%})"
            ),
        )

        axis.set_xlabel(
            "Scan pattern"
        )

        axis.set_ylabel(
            "Coverage fraction"
        )

        axis.set_title(
            "Median Coverage Efficiency\n"
            f"Map={map_size:g} deg, "
            f"Duration={duration:g} s"
        )

        axis.tick_params(
            axis="x",
            rotation=30,
        )

        axis.grid(
            axis="y",
            alpha=0.3,
        )

        axis.legend(
            loc="best"
        )

        fig.tight_layout()

        filename = (
            f"map_{map_size:g}deg_"
            f"{duration:g}s_"
            "coverage_by_scan_pattern.png"
        )

        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


    # ============================================================
    # PLOT 3:
    # PWV-ELEVATION SUITABILITY
    # ============================================================

    for (
        band,
        scan_pattern,
        duration,
    ), group in df.groupby(
        [
            "band_ghz",
            "scan_pattern",
            "duration_s",
        ]
    ):

        plot_group = (
            group
            .groupby(
                [
                    "pwv_mm",
                    "mean_elevation_deg",
                ],
                as_index=False,
            )[
                "overall_pass"
            ]
            .all()
        )

        fig, axis = plt.subplots(
            figsize=(8, 6)
        )

        pass_group = plot_group[
            plot_group[
                "overall_pass"
            ]
        ]

        fail_group = plot_group[
            ~plot_group[
                "overall_pass"
            ]
        ]

        if len(
            pass_group
        ) > 0:

            axis.scatter(
                pass_group[
                    "pwv_mm"
                ],
                pass_group[
                    "mean_elevation_deg"
                ],
                marker="o",
                s=100,
                label="Suitable",
            )

        if len(
            fail_group
        ) > 0:

            axis.scatter(
                fail_group[
                    "pwv_mm"
                ],
                fail_group[
                    "mean_elevation_deg"
                ],
                marker="x",
                s=100,
                label="Not suitable",
            )

        axis.set_xlabel(
            "PWV (mm)"
        )

        axis.set_ylabel(
            "Mean elevation (deg)"
        )

        axis.set_title(
            "Time-Domain Observing Suitability\n"
            f"{band:g} GHz, "
            f"{scan_pattern.replace('_', ' ').title()}, "
            f"{duration:g} s"
        )

        axis.grid(
            alpha=0.3
        )

        axis.legend(
            loc="best"
        )

        fig.tight_layout()

        filename = (
            f"{band:g}GHz_"
            f"{scan_pattern}_"
            f"{duration:g}s_"
            "suitability_vs_pwv_elevation.png"
        )

        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


    print(
        "\nObserving-strategy analysis complete."
    )

# ============================================================
# PLOT 4:
# CROSS-BAND P95 FREQUENCY-RESPONSE SUMMARY
# ============================================================
#
# This plot compares detector frequency-response excursions
# across the 280, 350, and 850 GHz bands while holding scan
# pattern and observation duration fixed.
#
# Each panel corresponds to one observing band.
# PWV is varied along the x-axis, while separate curves show
# the different observing elevation ranges.
# ============================================================

P95_SUMMARY_SCAN_PATTERN = "daisy"
P95_SUMMARY_DURATION = 900

# Select only the representative observing configuration.
p95_summary_df = df[
    (df["scan_pattern"] == P95_SUMMARY_SCAN_PATTERN)
    & (
        df["duration_s"]
        == P95_SUMMARY_DURATION
    )
].copy()


# ------------------------------------------------------------
# Check that data were found
# ------------------------------------------------------------

if p95_summary_df.empty:
    raise ValueError(
        "No data found for the requested P95 summary "
        f"configuration:\n"
        f"Scan pattern = {P95_SUMMARY_SCAN_PATTERN}\n"
        f"Duration = {P95_SUMMARY_DURATION} s"
    )


print(
    "\nP95 cross-band summary data:"
)

print(
    p95_summary_df[
        [
            "band_ghz",
            "pwv_mm",
            "duration_s",
            "elevation_min_deg",
            "elevation_max_deg",
            "p95_abs_delta_f_over_fwhm",
        ]
    ]
    .sort_values(
        [
            "band_ghz",
            "duration_s",
            "elevation_min_deg",
            "pwv_mm",
        ]
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# Bands and elevation ranges
# ------------------------------------------------------------

bands = [
    280,
    350,
    850,
]

elevation_ranges = (
    p95_summary_df[
        [
            "elevation_min_deg",
            "elevation_max_deg",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "elevation_min_deg"
    )
)


# ------------------------------------------------------------
# Make three-panel figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(24, 7.5),
    sharex=True,
    sharey=True,
)

for axis, band in zip(axes, bands):

    band_df = p95_summary_df[
        p95_summary_df["band_ghz"] == band
    ]
    for _, elevation_row in elevation_ranges.iterrows():

        elev_min = elevation_row[
            "elevation_min_deg"
        ]

        elev_max = elevation_row[
            "elevation_max_deg"
        ]

        elevation_df = band_df[
            (
                band_df[
                    "elevation_min_deg"
                ] == elev_min
            )
            & (
                band_df[
                    "elevation_max_deg"
                ] == elev_max
            )
        ].copy()

        elevation_df = (
            elevation_df
            .sort_values(
                "pwv_mm"
            )
        )

        if elevation_df.empty:
            continue

        pwv = elevation_df[
            "pwv_mm"
        ].to_numpy()

        p50 = elevation_df[
            "p50_abs_delta_f_over_fwhm"
        ].to_numpy()

        p95 = elevation_df[
            "p95_abs_delta_f_over_fwhm"
        ].to_numpy()

        # --------------------------------------------------------
        # P50 = characteristic response
        # --------------------------------------------------------

        line = axis.plot(
            pwv,
            p50,
            marker="o",
            markersize = 5,
            linewidth=5,
            linestyle="-",
            label=(
                f"{elev_min:g}-"
                f"{elev_max:g}°"
            ),
        )[0]

        # --------------------------------------------------------
        # P95 = upper-tail response
        #
        # Use the SAME colour as the corresponding elevation
        # curve, but a dashed line.
        # --------------------------------------------------------

        axis.plot(
            pwv,
            p95,
            linewidth=3,
            linestyle="--",
            color=line.get_color(),
        )

    axis.set_title(
        f"{band} GHz",
        fontsize=18,
    )

    axis.grid(
        alpha=0.3,
    )

    axis.tick_params(
        labelsize=15,
    )


# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

axes[0].set_ylabel(
    r"$P_{95}\left(|\Delta f/\mathrm{FWHM}|\right)$",
    fontsize=13,
)

fig.supxlabel(
    "PWV (mm)",
    fontsize=13,
)


# ------------------------------------------------------------
# Common legend
# ------------------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.89),
    ncol=3,
    frameon=True,
)


# ------------------------------------------------------------
# Overall title
# ------------------------------------------------------------

fig.suptitle(
    "Detector Frequency Response Across Prime-Cam Bands\n"
    f"{P95_SUMMARY_SCAN_PATTERN.replace('_', ' ').title()} scan, "
    f"{P95_SUMMARY_DURATION} s",
    fontsize=15,
    y=0.99,
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        0.83,
    ]
)
# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

p95_summary_output = (
    OUTPUT_DIR
    / (
        "cross_band_p95_frequency_response_"
        f"{P95_SUMMARY_SCAN_PATTERN}_"
        f"{P95_SUMMARY_DURATION}s.png"
    )
)

fig.savefig(
    p95_summary_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print(
    "\nSaved cross-band P95 frequency-response figure:\n"
    f"{p95_summary_output}"
)


from pathlib import Path

BASE_DIR = Path(
    "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria"
)

OUTPUTS_DIR = BASE_DIR / "outputs"


def get_tod_path(
    band_ghz,
    elevation_label,
    pwv_mm,
    duration_s=900,
    speed=0.1,
    map_size_deg=1.5,
):
    """
    Construct the path to an existing Daisy TOD.

    The 280 GHz runs use an older naming convention without
    the band label in the run prefix.
    """

    band_ghz = int(band_ghz)

    pwv_tag = (
        f"{pwv_mm:.2f}"
        .replace(".", "p")
    )

    speed_tag = (
        f"{speed:.1f}"
        .replace(".", "p")
    )

    map_tag = (
        f"{map_size_deg:.1f}"
        .replace(".", "p")
    )

    if band_ghz == 280:

        run_prefix = (
            f"OrionA_daisy_"
            f"{elevation_label}_"
            f"speed_{speed_tag}_"
            f"PWV_{pwv_tag}mm_"
            f"duration_{duration_s}s_"
            f"map_{map_tag}deg"
        )

    elif band_ghz in (350, 850):

        run_prefix = (
            f"OrionA_{band_ghz}GHz_"
            f"daisy_"
            f"{elevation_label}_"
            f"speed_{speed_tag}_"
            f"PWV_{pwv_tag}mm_"
            f"duration_{duration_s}s_"
            f"map_{map_tag}deg"
        )

    else:
        raise ValueError(
            f"Unsupported band: {band_ghz}"
        )

    tod_dir = (
        OUTPUTS_DIR
        / f"{run_prefix}_tods"
    )

    fits_path = (
        tod_dir
        / f"{run_prefix}_dim_reduced_tods.fits"
    )

    return fits_path


# ============================================================
# RECALCULATE FREQUENCY-SHIFT PERCENTILES
# 900 s DAISY RUNS ONLY
# ============================================================
#
# This reads the EXISTING TOD FITS files.
# It does NOT rerun any MARIA simulations.
#
# For each:
#   - band:      280, 350, 850 GHz
#   - PWV:       0.36, 0.67, 1.28 mm
#   - elevation: 45-55, 55-65, 65-75 deg
#
# it calculates percentiles of:
#
#       |Delta f / FWHM|
#
# using the same fixed-tone assumption as module_analysis:
# each detector's reference power is its median power over
# the observation.
# ============================================================



DAISY_PERCENTILE_BANDS = [
    280,
    350,
    850,
]

DAISY_PERCENTILE_PWVS = [
    0.36,
    0.67,
    1.28,
]

DAISY_PERCENTILE_ELEVATIONS = [
    "45-55",
    "55-65",
    "65-75",
]

DAISY_PERCENTILE_DURATION_S = 900
DAISY_PERCENTILE_SPEED = 0.1
DAISY_PERCENTILE_MAP_SIZE_DEG = 1.5

# Percentiles of |Delta f/FWHM| to calculate.
PERCENTILES_TO_CALCULATE = [
    25,
    50,
    68,
    75,
    84,
    90,
    95,
    99,
]

DAISY_PERCENTILE_OUTPUT_CSV = (
    OUTPUT_DIR
    / "daisy_900s_frequency_shift_percentiles.csv"
)


# ============================================================
# BUILD EXISTING TOD PATH
# ============================================================

def get_daisy_900s_tod_path(
    band_ghz,
    elevation_label,
    pwv_mm,
):
    """
    Return the path to an existing 900 s Daisy TOD.

    The 280 GHz simulations use the older filename convention,
    while 350 and 850 GHz include the band in the run prefix.
    """

    band_ghz = int(band_ghz)

    pwv_tag = (
        f"{pwv_mm:.2f}"
        .replace(".", "p")
    )

    speed_tag = (
        f"{DAISY_PERCENTILE_SPEED:.1f}"
        .replace(".", "p")
    )

    map_tag = (
        f"{DAISY_PERCENTILE_MAP_SIZE_DEG:.1f}"
        .replace(".", "p")
    )

    if band_ghz == 280:

        run_prefix = (
            f"OrionA_daisy_"
            f"{elevation_label}_"
            f"speed_{speed_tag}_"
            f"PWV_{pwv_tag}mm_"
            f"duration_{DAISY_PERCENTILE_DURATION_S}s_"
            f"map_{map_tag}deg"
        )

    elif band_ghz in (350, 850):

        run_prefix = (
            f"OrionA_{band_ghz}GHz_"
            f"daisy_"
            f"{elevation_label}_"
            f"speed_{speed_tag}_"
            f"PWV_{pwv_tag}mm_"
            f"duration_{DAISY_PERCENTILE_DURATION_S}s_"
            f"map_{map_tag}deg"
        )

    else:
        raise ValueError(
            f"Unsupported band: {band_ghz}"
        )

    tod_directory = (
        Path("outputs")
        / f"{run_prefix}_tods"
    )

    fits_path = (
        tod_directory
        / (
            f"{run_prefix}_"
            "dim_reduced_tods.fits"
        )
    )

    return fits_path, run_prefix


# ============================================================
# BUILD MARIA BAND
# ============================================================

def make_percentile_band(
    band_ghz,
):
    """
    Construct the same MARIA band definition used for the
    atmospheric simulations.
    """

    band_ghz = int(band_ghz)

    if band_ghz == 280:
        center_hz = 280e9
        width_hz = 60e9
        net_cmb = 13e-6

    elif band_ghz == 350:
        center_hz = 350e9
        width_hz = 35e9
        net_cmb = 48e-6

    elif band_ghz == 850:
        center_hz = 850e9
        width_hz = 97e9
        net_cmb = 13e-6

    else:
        raise ValueError(
            f"Unsupported band: {band_ghz}"
        )

    return Band(
        name="m2/f093",
        center=center_hz,
        width=width_hz,
        efficiency=0.5,
        NET_CMB=net_cmb,
        knee=1.0,
        gain_error=5e-2,
    )


# ============================================================
# MKID PARAMETERS
# ============================================================

def get_percentile_mkid_parameters(
    band_ghz,
):
    """
    Return the same MKID parameters used in module_analysis.
    """

    band_ghz = int(band_ghz)

    if band_ghz in (280, 350):
        return {
            "Q_r": 40000.0,
            "R_0": -2.448e9,
            "P_0": 957e-18,
        }

    if band_ghz == 850:
        return {
            "Q_r": 15000.0,
            "R_0": -1.0e7,
            "P_0": 120e-12,
        }

    raise ValueError(
        f"Unsupported band: {band_ghz}"
    )


# ============================================================
# CALCULATE DELTA f / FWHM
# ============================================================

def calculate_delta_f_fwhm_matrix(
    power_pW,
    band_ghz,
):
    """
    Calculate detector-by-time Delta f/FWHM.

    Each detector uses a fixed readout reference corresponding
    to its median optical loading over the 900 s observation.
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

    parameters = (
        get_percentile_mkid_parameters(
            band_ghz
        )
    )

    Q_r = parameters["Q_r"]
    R_0 = parameters["R_0"]
    P_0 = parameters["P_0"]

    # Convert pW to W.
    power_W = (
        power_pW
        * 1e-12
    )

    # Fixed operating point for every detector.
    median_power_W = np.nanmedian(
        power_W,
        axis=1,
        keepdims=True,
    )

    # Responsivity evaluated at that fixed operating point.
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        responsivity = (
            R_0
            / np.sqrt(
                1.0
                + median_power_W / P_0
            )
        )

    delta_power_W = (
        power_W
        - median_power_W
    )

    delta_f_fwhm = (
        Q_r
        * responsivity
        * delta_power_W
    )

    return delta_f_fwhm


# ============================================================
# CALCULATE PERCENTILE SUMMARY
# ============================================================

def calculate_frequency_shift_percentiles(
    delta_f_fwhm,
):
    """
    Calculate pooled percentiles of absolute Delta f/FWHM
    over every valid detector-time sample.
    """

    delta_f_fwhm = np.asarray(
        delta_f_fwhm,
        dtype=np.float64,
    )

    absolute_shift = np.abs(
        delta_f_fwhm[
            np.isfinite(
                delta_f_fwhm
            )
        ]
    )

    if absolute_shift.size == 0:
        raise ValueError(
            "No finite Delta f/FWHM values found."
        )

    metrics = {}

    for percentile in PERCENTILES_TO_CALCULATE:

        metrics[
            f"p{percentile}_abs_delta_f_over_fwhm"
        ] = float(
            np.nanpercentile(
                absolute_shift,
                percentile,
            )
        )

    # These are also useful diagnostics.
    metrics[
        "mean_abs_delta_f_over_fwhm"
    ] = float(
        np.nanmean(
            absolute_shift
        )
    )

    metrics[
        "std_abs_delta_f_over_fwhm"
    ] = float(
        np.nanstd(
            absolute_shift
        )
    )

    metrics[
        "max_abs_delta_f_over_fwhm"
    ] = float(
        np.nanmax(
            absolute_shift
        )
    )

    metrics[
        "n_detector_time_samples"
    ] = int(
        absolute_shift.size
    )

    return metrics


# ============================================================
# RUN EXISTING 900 s DAISY TOD ANALYSIS
# ============================================================

if RUN_DAISY_PERCENTILE_ANALYSIS:

    percentile_rows = []

    site = maria.get_site(
        "cerro_chajnantor",
        altitude=5600,
    )

    total_expected = (
        len(DAISY_PERCENTILE_BANDS)
        * len(DAISY_PERCENTILE_PWVS)
        * len(DAISY_PERCENTILE_ELEVATIONS)
    )

    run_number = 0

    for band_ghz in DAISY_PERCENTILE_BANDS:

        maria_band = (
            make_percentile_band(
                band_ghz
            )
        )

        for pwv_mm in DAISY_PERCENTILE_PWVS:

            for elevation_label in (
                DAISY_PERCENTILE_ELEVATIONS
            ):

                run_number += 1

                (
                    fits_path,
                    run_prefix,
                ) = get_daisy_900s_tod_path(
                    band_ghz=band_ghz,
                    elevation_label=elevation_label,
                    pwv_mm=pwv_mm,
                )

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    f"Percentile analysis "
                    f"{run_number}/{total_expected}"
                )

                print(
                    f"Band: {band_ghz} GHz"
                )

                print(
                    f"PWV: {pwv_mm:.2f} mm"
                )

                print(
                    f"Elevation: "
                    f"{elevation_label} deg"
                )

                print(
                    f"TOD: {fits_path}"
                )

                # --------------------------------------------
                # Skip missing files instead of terminating
                # the entire analysis.
                # --------------------------------------------

                if not fits_path.exists():

                    print(
                        "WARNING: TOD does not exist. "
                        "Skipping this configuration."
                    )

                    continue

                # --------------------------------------------
                # Load existing TOD
                # --------------------------------------------

                tod = maria.tod.load(
                    fits_path,
                    site=site,
                    bands=[
                        maria_band
                    ],
                )

                # Convert the existing K_RJ TOD into optical
                # detector power using MARIA.
                power_pW = np.asarray(
                    tod.to("pW").signal,
                    dtype=np.float64,
                )

                print(
                    f"Power matrix shape: "
                    f"{power_pW.shape}"
                )


                # --------------------------------------------
                # POWER / RESPONSIVITY DIAGNOSTICS
                # --------------------------------------------

                parameters = get_percentile_mkid_parameters(
                    band_ghz
                )

                R_0 = parameters["R_0"]
                P_0 = parameters["P_0"]

                # Convert pW -> W
                power_W = power_pW * 1e-12

                # Per-detector operating/reference power.
                # This is the same P_i used in the frequency
                # response calculation.
                P_i_W = np.nanmedian(
                    power_W,
                    axis=1,
                )

                P_i_pW = P_i_W * 1e12


                # Power excursion about each detector's
                # fixed operating point.
                delta_P_W = (
                    power_W
                    - P_i_W[:, None]
                )

                delta_P_pW = delta_P_W * 1e12


                # Responsivity at each detector's operating point.
                responsivity = (
                    R_0
                    / np.sqrt(
                        1.0
                        + P_i_W / P_0
                    )
                )


                # This isolates the PWV-dependent part of
                # R(P_i) * Delta P, apart from R_0.
                scaled_delta_P_W = (
                    np.abs(delta_P_W)
                    / np.sqrt(
                        1.0
                        + P_i_W[:, None] / P_0
                    )
                )

                abs_delta_P_pW = np.abs(delta_P_pW)

                delta_P_over_sqrt_P = (
                    abs_delta_P_pW / np.sqrt(P_i_pW[:, None])
                )

                fractional_delta_P = (
                    abs_delta_P_pW / P_i_pW[:, None]
                )

                # --------------------------------------------
                # Summarize this observation
                # --------------------------------------------

                power_metrics = {

                    # Typical detector operating power
                    "median_P_i_pW": float(
                        np.nanmedian(P_i_pW)
                    ),

                    # Detector-to-detector range
                    "p16_P_i_pW": float(
                        np.nanpercentile(P_i_pW, 16)
                    ),

                    "p84_P_i_pW": float(
                        np.nanpercentile(P_i_pW, 84)
                    ),

                    # Typical absolute responsivity
                    "median_abs_responsivity_Hz_W": float(
                        np.nanmedian(
                            np.abs(responsivity)
                        )
                    ),

                    # Characteristic power excursions
                    "p50_abs_delta_P_pW": float(
                        np.nanpercentile(
                            np.abs(delta_P_pW),
                            50,
                        )
                    ),

                    "p95_abs_delta_P_pW": float(
                        np.nanpercentile(
                            np.abs(delta_P_pW),
                            95,
                        )
                    ),

                    # Delta P after accounting for the
                    # P_i-dependent responsivity suppression
                    "p50_scaled_delta_P_W": float(
                        np.nanpercentile(
                            scaled_delta_P_W,
                            50,
                        )
                    ),

                    "p95_scaled_delta_P_W": float(
                        np.nanpercentile(
                            scaled_delta_P_W,
                            95,
                        )
                    ),
                    "p95_delta_P_over_sqrt_P": float(
                        np.nanpercentile(
                            delta_P_over_sqrt_P,
                            95,
                        )
                    ),
                    "p95_fractional_delta_P": float(
                        np.nanpercentile(
                            fractional_delta_P,
                            95,
                        )
                    ),
                }


                # --------------------------------------------
                # Frequency response
                # --------------------------------------------

                delta_f_fwhm = (
                    calculate_delta_f_fwhm_matrix(
                        power_pW=power_pW,
                        band_ghz=band_ghz,
                    )
                )

                metrics = (
                    calculate_frequency_shift_percentiles(
                        delta_f_fwhm
                    )
                )

                # --------------------------------------------
                # Build one CSV row
                # --------------------------------------------

                elevation_min = float(
                    elevation_label.split("-")[0]
                )

                elevation_max = float(
                    elevation_label.split("-")[1]
                )

                row = {
                    "run_prefix": run_prefix,
                    "band_ghz": band_ghz,
                    "pwv_mm": pwv_mm,
                    "scan_pattern": "daisy",
                    "duration_s": (
                        DAISY_PERCENTILE_DURATION_S
                    ),
                    "input_speed_deg_s": (
                        DAISY_PERCENTILE_SPEED
                    ),
                    "map_size_deg": (
                        DAISY_PERCENTILE_MAP_SIZE_DEG
                    ),
                    "elevation_label": (
                        elevation_label
                    ),
                    "elevation_min_deg": (
                        elevation_min
                    ),
                    "elevation_max_deg": (
                        elevation_max
                    ),
                    "tod_path": str(
                        fits_path
                    ),
                }

                row.update(
                    metrics
                )

                row.update(
                    power_metrics
                )

                percentile_rows.append(
                    row
                )

                print(
                    "Percentiles:"
                )

                for percentile in (
                    PERCENTILES_TO_CALCULATE
                ):

                    key = (
                        f"p{percentile}_"
                        "abs_delta_f_over_fwhm"
                    )

                    print(
                        f"  P{percentile}: "
                        f"{metrics[key]:.6g}"
                    )

                # Release the large TOD before loading
                # the next file.
                del tod
                del power_pW
                del delta_f_fwhm

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    percentile_df = pd.DataFrame(
        percentile_rows
    )

    percentile_df = (
        percentile_df
        .sort_values(
            [
                "band_ghz",
                "elevation_min_deg",
                "pwv_mm",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    percentile_df.to_csv(
        DAISY_PERCENTILE_OUTPUT_CSV,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "900 s Daisy percentile analysis complete."
    )

    print(
        f"Processed "
        f"{len(percentile_df)} / "
        f"{total_expected} configurations."
    )

    print(
        "Saved percentile CSV to:\n"
        f"{DAISY_PERCENTILE_OUTPUT_CSV}"
    )

    print(
        "\nPercentile summary:"
    )

    display_columns = [
        "band_ghz",
        "pwv_mm",
        "elevation_label",
        "p50_abs_delta_f_over_fwhm",
        "p68_abs_delta_f_over_fwhm",
        "p75_abs_delta_f_over_fwhm",
        "p90_abs_delta_f_over_fwhm",
        "p95_abs_delta_f_over_fwhm",
        "p99_abs_delta_f_over_fwhm",
    ]

    print(
        percentile_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

# ============================================================
# PLOT:
# CROSS-BAND P50 AND P95 FREQUENCY RESPONSE
# ============================================================
#
# Solid line + marker:
#     P50(|Delta f/FWHM|)
#
# Dashed line:
#     P95(|Delta f/FWHM|)
#
# Colour:
#     observing elevation
#
# All three observing bands share the same y-axis so that the
# relative scale of the detector response can be compared
# directly between 280, 350, and 850 GHz.
# ============================================================


PERCENTILE_CSV = (
    OUTPUT_DIR
    / "daisy_900s_frequency_shift_percentiles.csv"
)


# ------------------------------------------------------------
# Load percentile results
# ------------------------------------------------------------

if not PERCENTILE_CSV.exists():
    raise FileNotFoundError(
        "Percentile CSV not found:\n"
        f"{PERCENTILE_CSV}"
    )


percentile_df = pd.read_csv(
    PERCENTILE_CSV
)


# ------------------------------------------------------------
# Check required percentile columns
# ------------------------------------------------------------

percentile_plot_columns = [
    "band_ghz",
    "pwv_mm",
    "elevation_min_deg",
    "elevation_max_deg",
    "p50_abs_delta_f_over_fwhm",
    "p95_abs_delta_f_over_fwhm",
]

missing_percentile_columns = [
    column
    for column in percentile_plot_columns
    if column not in percentile_df.columns
]

if missing_percentile_columns:
    raise KeyError(
        "Percentile CSV is missing:\n"
        + "\n".join(
            missing_percentile_columns
        )
    )


# ------------------------------------------------------------
# Bands and elevation ranges
# ------------------------------------------------------------

bands = [
    280,
    350,
    850,
]


elevation_ranges = (
    percentile_df[
        [
            "elevation_min_deg",
            "elevation_max_deg",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "elevation_min_deg"
    )
)


# ============================================================
# MAKE THREE-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(24, 7.5),
    sharex=True,
    sharey=True,
)


for axis, band in zip(
    axes,
    bands,
):

    band_df = percentile_df[
        percentile_df[
            "band_ghz"
        ] == band
    ].copy()


    for _, elevation_row in (
        elevation_ranges.iterrows()
    ):

        elev_min = elevation_row[
            "elevation_min_deg"
        ]

        elev_max = elevation_row[
            "elevation_max_deg"
        ]


        elevation_df = band_df[
            (
                band_df[
                    "elevation_min_deg"
                ] == elev_min
            )
            & (
                band_df[
                    "elevation_max_deg"
                ] == elev_max
            )
        ].copy()


        elevation_df = (
            elevation_df
            .sort_values(
                "pwv_mm"
            )
        )


        if elevation_df.empty:
            continue


        pwv = (
            elevation_df[
                "pwv_mm"
            ]
            .to_numpy()
        )


        p50 = (
            elevation_df[
                "p50_abs_delta_f_over_fwhm"
            ]
            .to_numpy()
        )


        p95 = (
            elevation_df[
                "p95_abs_delta_f_over_fwhm"
            ]
            .to_numpy()
        )


        # ----------------------------------------------------
        # P50:
        # characteristic / median absolute frequency response
        # ----------------------------------------------------

        line = axis.plot(
            pwv,
            p50,
            marker="o",
            markersize=8,
            linewidth=3,
            linestyle="-",
            label=(
                f"{elev_min:g}-"
                f"{elev_max:g}°"
            ),
        )[0]


        # ----------------------------------------------------
        # P95:
        # upper-tail frequency response
        #
        # Use the same colour as the P50 curve for this
        # elevation, but distinguish it using a dashed line.
        # ----------------------------------------------------

        axis.plot(
            pwv,
            p95,
            linewidth=3,
            linestyle="--",
            color=line.get_color(),
        )


    # --------------------------------------------------------
    # Individual panel formatting
    # --------------------------------------------------------

    axis.set_title(
        f"{band} GHz",
        fontsize=20,
    )

    axis.grid(
        alpha=0.3,
    )

    axis.tick_params(
        labelsize=15,
    )


# ============================================================
# SHARED AXIS LABELS
# ============================================================

# This is deliberately NOT labelled P95 because both P50 and
# P95 are now shown.

axes[0].set_ylabel(
    r"$|\Delta f/\mathrm{FWHM}|$",
    fontsize=20,
)


fig.supxlabel(
    "PWV (mm)",
    fontsize=20,
)


# ============================================================
# COMMON Y-AXIS SCALE
# ============================================================
#
# Using the same scale for every panel makes the much larger
# response of the 850 GHz module immediately visible.
# ============================================================

maximum_p95 = (
    percentile_df[
        "p95_abs_delta_f_over_fwhm"
    ]
    .max()
)


axes[0].set_ylim(
    0,
    1.08 * maximum_p95,
)

# ============================================================
# COMBINED LEGEND
# ============================================================

elevation_handles, elevation_labels = (
    axes[0].get_legend_handles_labels()
)

statistic_handles = [
    Line2D(
        [0],
        [0],
        linestyle="-",
        linewidth=3,
        marker="o",
        markersize=8,
        color="black",
        label=r"$P_{50}$",
    ),

    Line2D(
        [0],
        [0],
        linestyle="--",
        linewidth=3,
        color="black",
        label=r"$P_{95}$",
    ),
]

fig.legend(
    elevation_handles + statistic_handles,
    elevation_labels + [
        r"$P_{50}$",
        r"$P_{95}$",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.90),
    ncol=5,
    frameon=True,
    fontsize=16,
)

# ============================================================
# OVERALL TITLE
# ============================================================

fig.suptitle(
    "Detector Frequency Response Across Prime-Cam Bands\n"
    "900 s Daisy scans",
    fontsize=20,
    y=0.99,
)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        0.87,
    ]
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR
    / (
        "cross_band_frequency_response_"
        "p50_p95_daisy_900s.png"
    )
)


fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


print(
    "\nSaved P50/P95 frequency-response figure:\n"
    f"{output_path}"
)


# ============================================================
# DIAGNOSTIC PLOT:
# POWER, RESPONSIVITY, AND DELTA-P SCALING
# 280 AND 350 GHz
# ============================================================

diagnostic_columns = [
    "band_ghz",
    "pwv_mm",
    "elevation_min_deg",
    "elevation_max_deg",
    "median_P_i_pW",
    "median_abs_responsivity_Hz_W",
    "p95_abs_delta_P_pW",
    "p95_scaled_delta_P_W",
]

missing_diagnostic_columns = [
    column
    for column in diagnostic_columns
    if column not in percentile_df.columns
]

if missing_diagnostic_columns:
    raise KeyError(
        "Diagnostic columns are missing from the percentile CSV:\n"
        + "\n".join(missing_diagnostic_columns)
    )


# Only investigate the 280 and 350 GHz modules for now.
diagnostic_df = percentile_df[
    percentile_df["band_ghz"].isin(
        [280, 350]
    )
].copy()


# ============================================================
# LOOP OVER BANDS
# ============================================================

for band in [280, 350]:

    band_df = diagnostic_df[
        diagnostic_df["band_ghz"] == band
    ].copy()

    if band_df.empty:
        print(
            f"No diagnostic data found for {band} GHz."
        )
        continue


    # --------------------------------------------------------
    # Make four-panel diagnostic figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=True,
    )

    ax_power = axes[0, 0]
    ax_resp = axes[0, 1]
    ax_delta_p = axes[1, 0]
    ax_scaled = axes[1, 1]


    # --------------------------------------------------------
    # Loop over elevation ranges
    # --------------------------------------------------------

    elevation_ranges = (
        band_df[
            [
                "elevation_min_deg",
                "elevation_max_deg",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "elevation_min_deg"
        )
    )

    for _, elevation_row in elevation_ranges.iterrows():

        elev_min = elevation_row[
            "elevation_min_deg"
        ]

        elev_max = elevation_row[
            "elevation_max_deg"
        ]

        elevation_df = band_df[
            (
                band_df[
                    "elevation_min_deg"
                ] == elev_min
            )
            & (
                band_df[
                    "elevation_max_deg"
                ] == elev_max
            )
        ].copy()

        elevation_df = (
            elevation_df
            .sort_values(
                "pwv_mm"
            )
        )

        if elevation_df.empty:
            continue

        pwv = elevation_df[
            "pwv_mm"
        ].to_numpy()

        label = (
            f"{elev_min:g}-"
            f"{elev_max:g}°"
        )


        # ----------------------------------------------------
        # PANEL 1:
        # Median operating power P_i
        # ----------------------------------------------------

        ax_power.plot(
            pwv,
            elevation_df[
                "median_P_i_pW"
            ],
            marker="o",
            linewidth=2.5,
            label=label,
        )


        # ----------------------------------------------------
        # PANEL 2:
        # Median absolute responsivity
        # ----------------------------------------------------

        ax_resp.plot(
            pwv,
            elevation_df[
                "median_abs_responsivity_Hz_W"
            ],
            marker="o",
            linewidth=2.5,
            label=label,
        )


        # ----------------------------------------------------
        # PANEL 3:
        # P95 absolute power excursion
        # ----------------------------------------------------

        ax_delta_p.plot(
            pwv,
            elevation_df[
                "p95_abs_delta_P_pW"
            ],
            marker="o",
            linewidth=2.5,
            label=label,
        )


        # ----------------------------------------------------
        # PANEL 4:
        # P95 Delta-P after responsivity suppression term
        # ----------------------------------------------------

        ax_scaled.plot(
            pwv,
            elevation_df[
                "p95_scaled_delta_P_W"
            ],
            marker="o",
            linewidth=2.5,
            label=label,
        )


    # ========================================================
    # LABELS
    # ========================================================

    ax_power.set_ylabel(
        r"Median $P_i$ (pW)",
        fontsize=13,
    )

    ax_resp.set_ylabel(
        r"Median $|R(P_i)|$ (Hz W$^{-1}$)",
        fontsize=13,
    )

    ax_delta_p.set_ylabel(
        r"$P_{95}(|\Delta P|)$ (pW)",
        fontsize=13,
    )

    ax_scaled.set_ylabel(
        r"$P_{95}\left("
        r"|\Delta P|/"
        r"\sqrt{1+P_i/P_0}"
        r"\right)$ (W)",
        fontsize=12,
    )

    ax_delta_p.set_xlabel(
        "PWV (mm)",
        fontsize=13,
    )

    ax_scaled.set_xlabel(
        "PWV (mm)",
        fontsize=13,
    )


    # ========================================================
    # PANEL TITLES
    # ========================================================

    ax_power.set_title(
        "Detector Operating Power",
        fontsize=14,
    )

    ax_resp.set_title(
        "MKID Responsivity",
        fontsize=14,
    )

    ax_delta_p.set_title(
        "Atmospheric Power Excursions",
        fontsize=14,
    )

    ax_scaled.set_title(
        "Power Excursions After Responsivity Suppression",
        fontsize=14,
    )


    # ========================================================
    # FORMATTING
    # ========================================================

    for axis in axes.flat:

        axis.grid(
            alpha=0.3
        )

        axis.tick_params(
            labelsize=11
        )


    # Use one common legend.
    handles, labels = (
        ax_power.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=True,
        fontsize=12,
    )


    # ========================================================
    # OVERALL TITLE
    # ========================================================

    fig.suptitle(
        "Power and Responsivity Diagnostics\n"
        f"{band} GHz, 900 s Daisy scans",
        fontsize=17,
        y=0.99,
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )


    # ========================================================
    # SAVE
    # ========================================================

    diagnostic_output = (
        OUTPUT_DIR
        / (
            f"{band}GHz_"
            "power_responsivity_diagnostics_"
            "daisy_900s.png"
        )
    )

    fig.savefig(
        diagnostic_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved diagnostic figure:\n"
        f"{diagnostic_output}"
    )

    # ============================================================
# DELTA-P SCALING DIAGNOSTICS
# ============================================================

for band in [280, 350]:

    band_df = percentile_df[
        percentile_df["band_ghz"] == band
    ].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
    )

    elevation_ranges = (
        band_df[
            [
                "elevation_min_deg",
                "elevation_max_deg",
            ]
        ]
        .drop_duplicates()
        .sort_values("elevation_min_deg")
    )

    for _, elevation_row in elevation_ranges.iterrows():

        elev_min = elevation_row["elevation_min_deg"]
        elev_max = elevation_row["elevation_max_deg"]

        elevation_df = band_df[
            (
                band_df["elevation_min_deg"]
                == elev_min
            )
            & (
                band_df["elevation_max_deg"]
                == elev_max
            )
        ].sort_values("pwv_mm")

        label = (
            f"{elev_min:g}-"
            f"{elev_max:g}°"
        )

        axes[0].plot(
            elevation_df["pwv_mm"],
            elevation_df[
                "p95_delta_P_over_sqrt_P"
            ],
            marker="o",
            linewidth=2.5,
            label=label,
        )

        axes[1].plot(
            elevation_df["pwv_mm"],
            elevation_df[
                "p95_fractional_delta_P"
            ],
            marker="o",
            linewidth=2.5,
            label=label,
        )

    axes[0].set_title(
        r"$P_{95}(|\Delta P|/\sqrt{P_i})$"
    )

    axes[1].set_title(
        r"$P_{95}(|\Delta P|/P_i)$"
    )

    axes[0].set_ylabel(
        r"$|\Delta P|/\sqrt{P_i}$"
    )

    axes[1].set_ylabel(
        r"$|\Delta P|/P_i$"
    )

    for axis in axes:
        axis.set_xlabel("PWV (mm)")
        axis.grid(alpha=0.3)

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    fig.suptitle(
        f"{band} GHz Power-Fluctuation Scaling\n"
        "900 s Daisy scans",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.88]
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{band}GHz_delta_P_scaling_"
            "daisy_900s.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

if plot_multi_realization_p95:

    # ============================================================
    # PLOT:
    # CROSS-BAND P95 FREQUENCY RESPONSE
    # FIVE INDEPENDENT REALIZATIONS
    # ============================================================

    SUMMARY_SCAN_PATTERN = "daisy"
    SUMMARY_DURATION = 900

    plot_df = realization_summary_df[
        (
            realization_summary_df["scan_pattern"]
            == SUMMARY_SCAN_PATTERN
        )
        & (
            realization_summary_df["duration_s"]
            == SUMMARY_DURATION
        )
    ].copy()


    bands = [
        280,
        350,
        850,
    ]


    elevation_ranges = (
        plot_df[
            [
                "elevation_min_deg",
                "elevation_max_deg",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "elevation_min_deg"
        )
    )


    # ============================================================
    # FIGURE SETUP
    # ============================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.8),
        sharex=True,
        sharey=True,
    )


    # Exact PWV values used in the simulations.
    pwv_ticks = [
        0.36,
        0.67,
        1.28,
    ]


    for axis, band in zip(
        axes,
        bands,
    ):

        band_df = plot_df[
            plot_df["band_ghz"] == band
        ].copy()


        for _, elevation_row in (
            elevation_ranges.iterrows()
        ):

            elev_min = elevation_row[
                "elevation_min_deg"
            ]

            elev_max = elevation_row[
                "elevation_max_deg"
            ]


            elevation_df = band_df[
                (
                    band_df["elevation_min_deg"]
                    == elev_min
                )
                & (
                    band_df["elevation_max_deg"]
                    == elev_max
                )
            ].sort_values(
                "pwv_mm"
            )


            if elevation_df.empty:
                continue


            pwv = (
                elevation_df[
                    "pwv_mm"
                ]
                .to_numpy()
            )


            p95 = (
                elevation_df[
                    "p95_realization_median"
                ]
                .to_numpy()
            )


            p95_min = (
                elevation_df[
                    "p95_realization_min"
                ]
                .to_numpy()
            )


            p95_max = (
                elevation_df[
                    "p95_realization_max"
                ]
                .to_numpy()
            )


            # ----------------------------------------------------
            # Median P95 across the five atmospheric realizations
            # ----------------------------------------------------

            line = axis.plot(
                pwv,
                p95,
                marker="o",
                markersize=6,
                linewidth=2.2,
                label=(
                    f"{elev_min:g}-"
                    f"{elev_max:g}°"
                ),
                zorder=3,
            )[0]


            # ----------------------------------------------------
            # Full realization-to-realization range
            # ----------------------------------------------------

            axis.errorbar(
                pwv,
                p95,
                yerr=[
                    p95 - p95_min,
                    p95_max - p95,
                ],
                fmt="none",
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
                color=line.get_color(),
                zorder=2,
            )


        # ========================================================
        # PANEL FORMATTING
        # ========================================================

        axis.set_title(
            f"{band} GHz",
            fontsize=15,
            pad=8,
        )


        # Actual atmospheric conditions simulated.
        axis.set_xticks(
            pwv_ticks
        )

        axis.set_xticklabels(
            [
                "0.36",
                "0.67",
                "1.28",
            ]
        )


        # One linewidth reference.
        axis.axhline(
            1.0,
            linestyle=":",
            linewidth=1.3,
            alpha=0.7,
            color="black",
            zorder=1,
        )


        # Shared y-axis scale.
        # axis.set_ylim(
        #     0,
        #     1.05,
        # )


        axis.grid(
            alpha=0.20,
        )


        axis.tick_params(
            labelsize=11,
        )


    # ============================================================
    # SHARED AXIS LABELS
    # ============================================================

    axes[0].set_ylabel(
        r"$P_{95}\left(|\Delta f/\mathrm{FWHM}|\right)$",
        fontsize=14,
    )


    fig.supxlabel(
        "PWV (mm)",
        fontsize=14,
        y=0.04,
    )


    # ============================================================
    # COMMON LEGEND
    # ============================================================

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )


    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.88,
        ),
        ncol=3,
        frameon=False,
        fontsize=11,
    )


    # ============================================================
    # OVERALL TITLE
    # ============================================================

    fig.suptitle(
        "Detector Frequency Response Across Prime-Cam Bands",
        fontsize=16,
        y=0.98,
    )


    # ============================================================
    # LAYOUT
    # ============================================================

    fig.tight_layout(
        rect=[
            0.02,
            0.06,
            0.99,
            0.82,
        ]
    )


    # ============================================================
    # SAVE
    # ============================================================

    output_path = (
        OUTPUT_DIR
        / (
            "cross_band_p95_frequency_response_"
            "daisy_900s_5realizations_noylim.png"
        )
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(fig)


    print(
        "\nSaved five-realization P95 figure:\n"
        f"{output_path}"
    )


# ============================================================
# PLOT:
# P50 AND P95 FREQUENCY RESPONSE
# FIVE INDEPENDENT REALIZATIONS
# ============================================================

if plot_multi_realization_p95:

    SUMMARY_SCAN_PATTERN = "daisy"
    SUMMARY_DURATION = 900

    plot_df = realization_summary_df[
        (
            realization_summary_df["scan_pattern"]
            == SUMMARY_SCAN_PATTERN
        )
        & (
            realization_summary_df["duration_s"]
            == SUMMARY_DURATION
        )
    ].copy()


    bands = [
        280,
        350,
        850,
    ]


    pwv_ticks = [
        0.36,
        0.67,
        1.28,
    ]


    elevation_ranges = (
        plot_df[
            [
                "elevation_min_deg",
                "elevation_max_deg",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "elevation_min_deg"
        )
    )


    # ========================================================
    # FIGURE SETUP
    #
    # Row 0 = P50
    # Row 1 = P95
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        sharex=True,
        sharey="row",
    )


    # ========================================================
    # LOOP OVER BANDS
    # ========================================================

    for column, band in enumerate(bands):

        band_df = plot_df[
            plot_df["band_ghz"] == band
        ].copy()


        ax_p50 = axes[0, column]
        ax_p95 = axes[1, column]


        for _, elevation_row in (
            elevation_ranges.iterrows()
        ):

            elev_min = elevation_row[
                "elevation_min_deg"
            ]

            elev_max = elevation_row[
                "elevation_max_deg"
            ]


            elevation_df = band_df[
                (
                    band_df["elevation_min_deg"]
                    == elev_min
                )
                & (
                    band_df["elevation_max_deg"]
                    == elev_max
                )
            ].sort_values(
                "pwv_mm"
            )


            if elevation_df.empty:
                continue


            pwv = (
                elevation_df[
                    "pwv_mm"
                ]
                .to_numpy()
            )


            # ====================================================
            # P50
            # ====================================================

            p50 = (
                elevation_df[
                    "p50_realization_median"
                ]
                .to_numpy()
            )

            p50_min = (
                elevation_df[
                    "p50_realization_min"
                ]
                .to_numpy()
            )

            p50_max = (
                elevation_df[
                    "p50_realization_max"
                ]
                .to_numpy()
            )


            ax_p50.errorbar(
                pwv,
                p50,
                yerr=[
                    p50 - p50_min,
                    p50_max - p50,
                ],
                marker="o",
                markersize=5,
                linewidth=2,
                elinewidth=1,
                capsize=3,
                label=(
                    f"{elev_min:g}-"
                    f"{elev_max:g}°"
                ),
            )


            # ====================================================
            # P95
            # ====================================================

            p95 = (
                elevation_df[
                    "p95_realization_median"
                ]
                .to_numpy()
            )

            p95_min = (
                elevation_df[
                    "p95_realization_min"
                ]
                .to_numpy()
            )

            p95_max = (
                elevation_df[
                    "p95_realization_max"
                ]
                .to_numpy()
            )


            ax_p95.errorbar(
                pwv,
                p95,
                yerr=[
                    p95 - p95_min,
                    p95_max - p95,
                ],
                marker="o",
                markersize=5,
                linewidth=2,
                elinewidth=1,
                capsize=3,
            )


        # ========================================================
        # COLUMN FORMATTING
        # ========================================================

        ax_p50.set_title(
            f"{band} GHz",
            fontsize=15,
            pad=8,
        )


        for axis in (
            ax_p50,
            ax_p95,
        ):

            axis.set_xticks(
                pwv_ticks
            )

            axis.set_xticklabels(
                [
                    "0.36",
                    "0.67",
                    "1.28",
                ]
            )

            axis.grid(
                alpha=0.2,
            )

            axis.tick_params(
                labelsize=10,
            )


        # One-linewidth reference belongs primarily
        # with the upper-tail response.
        ax_p95.axhline(
            1.0,
            linestyle=":",
            linewidth=1.3,
            color="black",
            alpha=0.65,
        )


    # ========================================================
    # ROW LABELS
    # ========================================================

    axes[0, 0].set_ylabel(
        r"$P_{50}\left(|\Delta f/\mathrm{FWHM}|\right)$",
        fontsize=13,
    )

    axes[1, 0].set_ylabel(
        r"$P_{95}\left(|\Delta f/\mathrm{FWHM}|\right)$",
        fontsize=13,
    )


    fig.supxlabel(
        "PWV (mm)",
        fontsize=14,
        y=0.04,
    )


    # ========================================================
    # Y LIMITS
    # ========================================================

    maximum_p50 = (
        plot_df[
            "p50_realization_max"
        ]
        .max()
    )

    maximum_p95 = (
        plot_df[
            "p95_realization_max"
        ]
        .max()
    )


    axes[0, 0].set_ylim(
        0,
        1.10 ,
    )

    axes[1, 0].set_ylim(
        0,
        1.10,
    )


    # ========================================================
    # SINGLE LEGEND:
    # only elevation needs to be encoded now
    # ========================================================

    handles, labels = (
        axes[0, 0]
        .get_legend_handles_labels()
    )


    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.92,
        ),
        ncol=3,
        frameon=False,
        fontsize=11,
    )


    # ========================================================
    # TITLE
    # ========================================================

    fig.suptitle(
        "Detector Frequency Response Across Prime-Cam Bands",
        fontsize=16,
        y=0.98,
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.tight_layout(
        rect=[
            0.02,
            0.06,
            0.99,
            0.88,
        ],
        h_pad=2.0,
        w_pad=1.2,
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / (
            "cross_band_frequency_response_"
            "p50_p95_separate_"
            "daisy_900s_5realizations.png"
        )
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(fig)


    print(
        "\nSaved separated five-realization "
        "P50/P95 figure:\n"
        f"{output_path}"
    )