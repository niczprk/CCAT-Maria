from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# File paths
# ============================================================

CSV_PATH = Path(
    "outputs/atmospheric_parameter_sweep/"
    "all_band_pwv_elevation_summary.csv"
)

OUTPUT_DIR = Path(
    "outputs/atmospheric_parameter_sweep/"
    "comparison_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Plot settings
# ============================================================

FIGURE_DPI = 300
FIGURE_SIZE = (10, 7)

BANDS = [280, 350, 850]

ELEVATION_LABELS = [
    "45-55",
    "55-65",
    "65-75",
]

PWV_VALUES = [
    0.36,
    0.67,
    1.28,
]


# ============================================================
# Load and validate the summary CSV
# ============================================================

def load_summary_csv(csv_path):
    """
    Load the combined atmospheric sweep CSV and perform
    basic validation and cleaning.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            "Could not find the combined summary CSV:\n"
            f"{csv_path.resolve()}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = [
        "band_ghz",
        "pwv_mm",
        "scan_pattern",
        "elevation_label",
        "atmospheric_median_elevation_deg",
        "atmospheric_median_raw_array_power_pW",
        "atmospheric_median_small_scale_mad_pW",
        "optical_common_model_residual_std_pW",
        "optical_residual_variance_fraction",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "The following required columns are missing "
            "from the CSV:\n"
            + "\n".join(missing_columns)
        )

    # Ensure these columns are numeric.
    numeric_columns = [
        "band_ghz",
        "pwv_mm",
        "atmospheric_median_elevation_deg",
        "atmospheric_median_raw_array_power_pW",
        "atmospheric_median_small_scale_mad_pW",
        "optical_common_model_residual_std_pW",
        "optical_residual_variance_fraction",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=numeric_columns
    )

    dataframe["elevation_label"] = (
        dataframe["elevation_label"]
        .astype(str)
    )

    dataframe = dataframe.sort_values(
        by=[
            "band_ghz",
            "elevation_label",
            "pwv_mm",
        ]
    ).reset_index(drop=True)

    print(
        f"Loaded {len(dataframe)} simulations "
        f"from {csv_path}"
    )

    print(
        "Bands:",
        sorted(
            dataframe["band_ghz"]
            .unique()
            .tolist()
        ),
    )

    print(
        "PWV values:",
        sorted(
            dataframe["pwv_mm"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Elevation ranges:",
        sorted(
            dataframe["elevation_label"]
            .unique()
            .tolist()
        ),
    )

    return dataframe


# ============================================================
# General plotting helper
# ============================================================

def finish_and_save_plot(
    figure,
    output_path,
):
    """
    Apply the final layout, save the figure, and close it.
    """

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Plot one atmospheric quantity versus PWV
# ============================================================

def plot_quantity_vs_pwv_by_band(
    dataframe,
    y_column,
    y_label,
    title,
    filename,
):
    """
    Create one panel per observing band. Within each panel,
    each elevation range is plotted as a separate line.
    """

    figure, axes = plt.subplots(
        len(BANDS),
        1,
        figsize=(10, 14),
        sharex=True,
    )

    for axis, band in zip(
        axes,
        BANDS,
    ):
        band_data = dataframe[
            dataframe["band_ghz"] == band
        ]

        for elevation_label in ELEVATION_LABELS:
            elevation_data = band_data[
                band_data["elevation_label"]
                == elevation_label
            ].sort_values("pwv_mm")

            if elevation_data.empty:
                continue

            axis.plot(
                elevation_data["pwv_mm"],
                elevation_data[y_column],
                marker="o",
                linewidth=1.5,
                label=(
                    f"Elevation "
                    f"{elevation_label}$^\\circ$"
                ),
            )

        axis.set_ylabel(y_label)

        axis.set_title(
            f"{band} GHz"
        )

        axis.grid(
            alpha=0.3
        )

        axis.legend(
            loc="best"
        )

    axes[-1].set_xlabel(
        "PWV (mm)"
    )

    figure.suptitle(
        title,
        fontsize=16,
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    finish_and_save_plot(
        figure,
        output_path,
    )


# ============================================================
# Plot one atmospheric quantity versus elevation
# ============================================================

def plot_quantity_vs_elevation_by_band(
    dataframe,
    y_column,
    y_label,
    title,
    filename,
):
    """
    Create one panel per observing band. Within each panel,
    each PWV value is plotted as a separate line.
    """

    figure, axes = plt.subplots(
        len(BANDS),
        1,
        figsize=(10, 14),
        sharex=True,
    )

    for axis, band in zip(
        axes,
        BANDS,
    ):
        band_data = dataframe[
            dataframe["band_ghz"] == band
        ]

        for pwv_mm in PWV_VALUES:
            pwv_data = band_data[
                np.isclose(
                    band_data["pwv_mm"],
                    pwv_mm,
                )
            ].sort_values(
                "atmospheric_median_elevation_deg"
            )

            if pwv_data.empty:
                continue

            axis.plot(
                pwv_data[
                    "atmospheric_median_elevation_deg"
                ],
                pwv_data[y_column],
                marker="o",
                linewidth=1.5,
                label=(
                    f"PWV = {pwv_mm:.2f} mm"
                ),
            )

        axis.set_ylabel(
            y_label
        )

        axis.set_title(
            f"{band} GHz"
        )

        axis.grid(
            alpha=0.3
        )

        axis.legend(
            loc="best"
        )

    axes[-1].set_xlabel(
        "Median observing elevation (deg)"
    )

    figure.suptitle(
        title,
        fontsize=16,
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    finish_and_save_plot(
        figure,
        output_path,
    )


# ============================================================
# Band-comparison plot
# ============================================================

def plot_band_comparison_vs_pwv(
    dataframe,
    y_column,
    y_label,
    title,
    filename,
    elevation_label="65-75",
):
    """
    Compare all three bands on one axis for a selected
    elevation range.
    """

    figure, axis = plt.subplots(
        figsize=FIGURE_SIZE
    )

    selected_data = dataframe[
        dataframe["elevation_label"]
        == elevation_label
    ]

    for band in BANDS:
        band_data = selected_data[
            selected_data["band_ghz"] == band
        ].sort_values("pwv_mm")

        if band_data.empty:
            continue

        axis.plot(
            band_data["pwv_mm"],
            band_data[y_column],
            marker="o",
            linewidth=1.5,
            label=f"{band} GHz",
        )

    axis.set_xlabel(
        "PWV (mm)"
    )

    axis.set_ylabel(
        y_label
    )

    axis.set_title(
        title
        + "\n"
        + f"Elevation {elevation_label}$^\\circ$"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend(
        loc="best"
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    finish_and_save_plot(
        figure,
        output_path,
    )


# ============================================================
# Optional summary table
# ============================================================

def save_reduced_summary_table(
    dataframe,
):
    """
    Save a smaller CSV containing only the quantities likely
    to be used in the atmospheric-loading Results section.
    """

    selected_columns = [
        "band_ghz",
        "pwv_mm",
        "scan_pattern",
        "elevation_label",
        "atmospheric_median_elevation_deg",
        "atmospheric_median_airmass",
        "atmospheric_median_raw_array_power_pW",
        "atmospheric_median_small_scale_mad_pW",
        "optical_common_model_residual_std_pW",
        "optical_residual_variance_fraction",
        "transmission_tau_0",
        "transmission_transmission_center",
    ]

    available_columns = [
        column
        for column in selected_columns
        if column in dataframe.columns
    ]

    reduced_dataframe = dataframe[
        available_columns
    ].copy()

    reduced_path = (
        OUTPUT_DIR
        / "atmospheric_results_reduced_summary.csv"
    )

    reduced_dataframe.to_csv(
        reduced_path,
        index=False,
    )

    print(
        f"Saved: {reduced_path}"
    )


# ============================================================
# Main program
# ============================================================

def main():
    summary_df = load_summary_csv(
        CSV_PATH
    )

    # --------------------------------------------------------
    # Detector loading
    # --------------------------------------------------------

    plot_quantity_vs_pwv_by_band(
        dataframe=summary_df,
        y_column=(
            "atmospheric_median_raw_array_power_pW"
        ),
        y_label="Median array power (pW)",
        title=(
            "Absolute Detector Loading "
            "versus PWV"
        ),
        filename=(
            "all_bands_detector_loading_"
            "versus_pwv.png"
        ),
    )

    plot_quantity_vs_elevation_by_band(
        dataframe=summary_df,
        y_column=(
            "atmospheric_median_raw_array_power_pW"
        ),
        y_label="Median array power (pW)",
        title=(
            "Absolute Detector Loading "
            "versus Elevation"
        ),
        filename=(
            "all_bands_detector_loading_"
            "versus_elevation.png"
        ),
    )

    # --------------------------------------------------------
    # Small-scale atmospheric fluctuations
    # --------------------------------------------------------

    plot_quantity_vs_pwv_by_band(
        dataframe=summary_df,
        y_column=(
            "atmospheric_median_small_scale_mad_pW"
        ),
        y_label=(
            "Median small-scale MAD estimate (pW)"
        ),
        title=(
            "Small-Scale Detector Fluctuations "
            "versus PWV"
        ),
        filename=(
            "all_bands_small_scale_mad_"
            "versus_pwv.png"
        ),
    )

    plot_quantity_vs_elevation_by_band(
        dataframe=summary_df,
        y_column=(
            "atmospheric_median_small_scale_mad_pW"
        ),
        y_label=(
            "Median small-scale MAD estimate (pW)"
        ),
        title=(
            "Small-Scale Detector Fluctuations "
            "versus Elevation"
        ),
        filename=(
            "all_bands_small_scale_mad_"
            "versus_elevation.png"
        ),
    )

    # --------------------------------------------------------
    # Residual common-mode atmospheric structure
    # --------------------------------------------------------

    plot_quantity_vs_pwv_by_band(
        dataframe=summary_df,
        y_column=(
            "optical_common_model_residual_std_pW"
        ),
        y_label=(
            "Residual common-power standard "
            "deviation (pW)"
        ),
        title=(
            "Common-Power Residual "
            "versus PWV"
        ),
        filename=(
            "all_bands_common_residual_std_"
            "versus_pwv.png"
        ),
    )

    plot_quantity_vs_elevation_by_band(
        dataframe=summary_df,
        y_column=(
            "optical_common_model_residual_std_pW"
        ),
        y_label=(
            "Residual common-power standard "
            "deviation (pW)"
        ),
        title=(
            "Common-Power Residual "
            "versus Elevation"
        ),
        filename=(
            "all_bands_common_residual_std_"
            "versus_elevation.png"
        ),
    )

    # --------------------------------------------------------
    # Residual variance fraction
    # --------------------------------------------------------

    plot_quantity_vs_pwv_by_band(
        dataframe=summary_df,
        y_column=(
            "optical_residual_variance_fraction"
        ),
        y_label=(
            "Residual variance fraction"
        ),
        title=(
            "Residual Variance Fraction "
            "versus PWV"
        ),
        filename=(
            "all_bands_residual_variance_"
            "fraction_versus_pwv.png"
        ),
    )

    plot_quantity_vs_elevation_by_band(
        dataframe=summary_df,
        y_column=(
            "optical_residual_variance_fraction"
        ),
        y_label=(
            "Residual variance fraction"
        ),
        title=(
            "Residual Variance Fraction "
            "versus Elevation"
        ),
        filename=(
            "all_bands_residual_variance_"
            "fraction_versus_elevation.png"
        ),
    )

    # --------------------------------------------------------
    # Direct band comparisons at 65-75 degrees
    # --------------------------------------------------------

    plot_band_comparison_vs_pwv(
        dataframe=summary_df,
        y_column=(
            "atmospheric_median_raw_array_power_pW"
        ),
        y_label="Median array power (pW)",
        title=(
            "Detector Loading across "
            "Prime-Cam Bands"
        ),
        filename=(
            "band_comparison_loading_"
            "versus_pwv.png"
        ),
        elevation_label="65-75",
    )

    plot_band_comparison_vs_pwv(
        dataframe=summary_df,
        y_column=(
            "atmospheric_median_small_scale_mad_pW"
        ),
        y_label=(
            "Median small-scale MAD estimate (pW)"
        ),
        title=(
            "Small-Scale Fluctuations across "
            "Prime-Cam Bands"
        ),
        filename=(
            "band_comparison_small_scale_mad_"
            "versus_pwv.png"
        ),
        elevation_label="65-75",
    )

    save_reduced_summary_table(
        summary_df
    )

    print(
        "\nComparison plotting complete."
    )

    print(
        f"Outputs saved in:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()