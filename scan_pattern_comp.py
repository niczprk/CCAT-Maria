from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIR = Path(
    "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria"
)

OUTPUT_BASE = BASE_DIR / "outputs" / "Scan_Speed_Tests_Medium"

# Scan patterns exactly as they appear in your directory/file names
SCAN_PATTERNS = [
    "back_and_forth",
    "daisy",
    "double_circle",
    "lissajous",
    "raster",
]
MAP_SIZE = "medium"

# Conditions to compare
COMPARISON_SPEED = 0.1
COMPARISON_ELEVATION = "55-65"

# Where to save the comparison figure
SAVE_DIR = BASE_DIR / "outputs" / "scan_pattern_comparisons"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTION TO BUILD EACH CSV PATH
# ============================================================

def get_csv_path(scan_pattern):
    """
    Construct the path to the combined speed/elevation CSV
    for a given scan pattern.
    """

    return (
        OUTPUT_BASE
        / scan_pattern
        / f"OrionA_{MAP_SIZE}_map_{scan_pattern}_speed_tests"
        / "speed_csv"
        / f"maria_{scan_pattern}_speed_elevation_motion_limits_combined_{scan_pattern}_v2.csv"
    )


# ============================================================
# LOAD THE REQUIRED ROW FROM EACH SCAN PATTERN
# ============================================================

comparison_rows = []

for scan_pattern in SCAN_PATTERNS:

    csv_path = get_csv_path(scan_pattern)

    print("\n" + "=" * 70)
    print(f"Scan pattern: {scan_pattern}")
    print(f"Looking for: {csv_path}")
    print(f"Exists: {csv_path.exists()}")

    if not csv_path.exists():
        print(f"WARNING: file not found, skipping {scan_pattern}")
        continue

    df = pd.read_csv(csv_path)

    print("CSV loaded successfully")
    print("Columns:")
    print(df.columns.tolist())

    print("\nAvailable elevations:")
    print(df["elevation_label"].unique())

    print("\nAvailable speeds:")
    print(df["input_speed_deg_s"].unique())

    # Select only the requested speed and elevation range
    selected = df[
        (df["elevation_label"] == COMPARISON_ELEVATION)
        & (
            df["input_speed_deg_s"]
            .sub(COMPARISON_SPEED)
            .abs()
            < 1e-8
        )
    ]

    if selected.empty:
        print(
            f"WARNING: no row found for "
            f"{scan_pattern}, "
            f"speed={COMPARISON_SPEED}, "
            f"elevation={COMPARISON_ELEVATION}"
        )
        continue

    # There should normally be one matching row
    row = selected.iloc[0]

    comparison_rows.append(
        {
            "scan_pattern": scan_pattern,
            "max_motor_az_velocity_deg_s":
                row["max_motor_az_velocity_deg_s"],
            "max_motor_el_velocity_deg_s":
                row["max_motor_el_velocity_deg_s"],
            "max_motor_az_acceleration_deg_s2":
                row["max_motor_az_acceleration_deg_s2"],
            "max_motor_el_acceleration_deg_s2":
                row["max_motor_el_acceleration_deg_s2"],
        }
    )


comparison_df = pd.DataFrame(comparison_rows)

print("\nComparison data:")
print(comparison_df)


# ============================================================
# FRIENDLIER NAMES FOR PLOTTING
# ============================================================

DISPLAY_NAMES = {
    "back_and_forth": "Back-and-forth",
    "daisy": "Daisy",
    "double_circle": "Double circle",
    "lissajous": "Lissajous",
    "raster": "Raster",
}

comparison_df["display_name"] = comparison_df["scan_pattern"].map(
    DISPLAY_NAMES
)


# ============================================================
# MAKE FOUR-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
    constrained_layout=True,
)

# ------------------------------------------------------------
# AZ velocity
# ------------------------------------------------------------

axes[0, 0].bar(
    comparison_df["display_name"],
    comparison_df["max_motor_az_velocity_deg_s"],
)

axes[0, 0].set_title("Maximum Azimuth Motor Velocity")
axes[0, 0].set_ylabel("Velocity (deg/s)")
axes[0, 0].tick_params(axis="x", rotation=25)
axes[0, 0].grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# EL velocity
# ------------------------------------------------------------

axes[0, 1].bar(
    comparison_df["display_name"],
    comparison_df["max_motor_el_velocity_deg_s"],
)

axes[0, 1].set_title("Maximum Elevation Motor Velocity")
axes[0, 1].set_ylabel("Velocity (deg/s)")
axes[0, 1].tick_params(axis="x", rotation=25)
axes[0, 1].grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# AZ acceleration
# ------------------------------------------------------------

axes[1, 0].bar(
    comparison_df["display_name"],
    comparison_df["max_motor_az_acceleration_deg_s2"],
)

axes[1, 0].set_title("Maximum Azimuth Motor Acceleration")
axes[1, 0].set_ylabel("Acceleration (deg/s²)")
axes[1, 0].tick_params(axis="x", rotation=25)
axes[1, 0].grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# EL acceleration
# ------------------------------------------------------------

axes[1, 1].bar(
    comparison_df["display_name"],
    comparison_df["max_motor_el_acceleration_deg_s2"],
)

axes[1, 1].set_title("Maximum Elevation Motor Acceleration")
axes[1, 1].set_ylabel("Acceleration (deg/s²)")
axes[1, 1].tick_params(axis="x", rotation=25)
axes[1, 1].grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# Overall title
# ------------------------------------------------------------

fig.suptitle(
    "Scan-Pattern Dependence of Telescope Motion\n"
    f"Input speed = {COMPARISON_SPEED:.1f} deg/s, "
    f"Elevation = {COMPARISON_ELEVATION} deg",
    fontsize=16,
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    SAVE_DIR
    / (
        f"{MAP_SIZE}_scan_pattern_motor_comparison_"
        f"speed_{str(COMPARISON_SPEED).replace('.', 'p')}_"
        f"elev_{COMPARISON_ELEVATION}.png"
    )
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(f"\nSaved figure to:\n{output_path}")

# ============================================================
# SECOND FIGURE:
# ACCELERATION DEPENDENCE ON MAP SIZE
# ============================================================

MAP_SIZES = {
    "small": 0.5,
    "medium": 1.5,
    "large": 3.0,
}

map_size_rows = []


def get_map_size_csv_path(scan_pattern, map_size_name):
    """
    Construct the CSV path for a given scan pattern and map size.
    """

    output_base = (
        BASE_DIR
        / "outputs"
        / f"Scan_Speed_Tests_{map_size_name.capitalize()}"
    )

    return (
        output_base
        / scan_pattern
        / f"OrionA_{map_size_name}_map_{scan_pattern}_speed_tests"
        / "speed_csv"
        / (
            f"maria_{scan_pattern}_speed_elevation_motion_limits_"
            f"combined_{scan_pattern}_v2.csv"
        )
    )


# ============================================================
# LOAD ALL MAP SIZES
# ============================================================

for map_size_name, map_size_deg in MAP_SIZES.items():

    for scan_pattern in SCAN_PATTERNS:

        csv_path = get_map_size_csv_path(
            scan_pattern,
            map_size_name,
        )

        print("\n" + "=" * 70)
        print(
            f"Map size: {map_size_name} "
            f"({map_size_deg} deg)"
        )
        print(f"Scan pattern: {scan_pattern}")
        print(f"Looking for: {csv_path}")
        print(f"Exists: {csv_path.exists()}")

        if not csv_path.exists():
            print(
                f"WARNING: file not found for "
                f"{map_size_name}, {scan_pattern}"
            )
            continue

        df = pd.read_csv(csv_path)

        selected = df[
            (df["elevation_label"] == COMPARISON_ELEVATION)
            & (
                df["input_speed_deg_s"]
                .sub(COMPARISON_SPEED)
                .abs()
                < 1e-8
            )
        ]

        if selected.empty:
            print(
                f"WARNING: no matching row for "
                f"{map_size_name}, {scan_pattern}, "
                f"speed={COMPARISON_SPEED}, "
                f"elevation={COMPARISON_ELEVATION}"
            )
            continue

        row = selected.iloc[0]

        map_size_rows.append(
            {
                "scan_pattern": scan_pattern,
                "display_name": DISPLAY_NAMES[scan_pattern],
                "map_size_name": map_size_name,
                "map_size_deg": map_size_deg,
                "max_motor_az_acceleration_deg_s2":
                    row["max_motor_az_acceleration_deg_s2"],
                "max_motor_el_acceleration_deg_s2":
                    row["max_motor_el_acceleration_deg_s2"],
            }
        )


map_size_df = pd.DataFrame(map_size_rows)

if map_size_df.empty:
    raise RuntimeError(
        "No map-size comparison data were loaded."
    )

print("\nMap-size comparison data:")
print(map_size_df)


# ============================================================
# MAKE TWO-PANEL MAP-SIZE FIGURE
# ============================================================

fig2, axes2 = plt.subplots(
    1,
    2,
    figsize=(14, 6),
)


# ------------------------------------------------------------
# AZIMUTH ACCELERATION
# ------------------------------------------------------------

for scan_pattern in SCAN_PATTERNS:

    pattern_df = (
        map_size_df[
            map_size_df["scan_pattern"] == scan_pattern
        ]
        .sort_values("map_size_deg")
    )

    if pattern_df.empty:
        continue

    axes2[0].plot(
        pattern_df["map_size_deg"],
        pattern_df["max_motor_az_acceleration_deg_s2"],
        marker="o",
        label=DISPLAY_NAMES[scan_pattern],
    )


axes2[0].set_title(
    "Maximum Azimuth Motor Acceleration"
)

axes2[0].set_xlabel(
    "Map size (deg)"
)

axes2[0].set_ylabel(
    "Acceleration (deg/s²)"
)

axes2[0].set_xticks(
    [0.5, 1.5, 3.0]
)

axes2[0].grid(
    alpha=0.3
)


# ------------------------------------------------------------
# ELEVATION ACCELERATION
# ------------------------------------------------------------

for scan_pattern in SCAN_PATTERNS:

    pattern_df = (
        map_size_df[
            map_size_df["scan_pattern"] == scan_pattern
        ]
        .sort_values("map_size_deg")
    )

    if pattern_df.empty:
        continue

    axes2[1].plot(
        pattern_df["map_size_deg"],
        pattern_df["max_motor_el_acceleration_deg_s2"],
        marker="o",
        label=DISPLAY_NAMES[scan_pattern],
    )


axes2[1].set_title(
    "Maximum Elevation Motor Acceleration"
)

axes2[1].set_xlabel(
    "Map size (deg)"
)

axes2[1].set_ylabel(
    "Acceleration (deg/s²)"
)

axes2[1].set_xticks(
    [0.5, 1.5, 3.0]
)

axes2[1].grid(
    alpha=0.3
)

# ------------------------------------------------------------
# OVERALL TITLE
# ------------------------------------------------------------

fig2.suptitle(
    "Scan-Pattern Dependence of Motor Acceleration with Map Size\n"
    f"Input speed = {COMPARISON_SPEED:.1f} deg/s, "
    f"Elevation = {COMPARISON_ELEVATION} deg",
    fontsize=16,
    y = 0.98
)

# ------------------------------------------------------------
# SHARED LEGEND
# ------------------------------------------------------------

handles, labels = axes2[1].get_legend_handles_labels()

fig2.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 0.895),
    frameon=True,
)

fig2.subplots_adjust(
    top=0.78,
    bottom=0.12,
    left=0.07,
    right=0.98,
    wspace=0.10,
)




# ============================================================
# SAVE SECOND FIGURE
# ============================================================

map_size_output_path = (
    SAVE_DIR
    / (
        "map_size_scan_pattern_acceleration_comparison_"
        f"speed_{str(COMPARISON_SPEED).replace('.', 'p')}_"
        f"elev_{COMPARISON_ELEVATION}.png"
    )
)

fig2.savefig(
    map_size_output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(
    f"\nSaved map-size comparison figure to:\n"
    f"{map_size_output_path}"
)