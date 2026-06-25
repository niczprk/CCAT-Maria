from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import maria
from maria.instrument import Band


SAMPLE_RATE_HZ = 20
DET_IDX = 604

SCAN_PATTERN = "daisy"
ELEV_LABEL = "65-75"
SPEED = 0.2

eta = 0.5
PWV_MM = 0.36

run_prefix = (
    f"OrionA_{SCAN_PATTERN.lower()}_{ELEV_LABEL}_speed_{SPEED:.1f}"
    .replace(".","p")
    )

TOD_OUTDIR = Path(f"outputs/{run_prefix}_tods")
fits_path = TOD_OUTDIR / f"{run_prefix}_dim_reduced_tods.fits"

HIST_OUTDIR = Path(f"outputs/{run_prefix}_motion_histograms")
HIST_OUTDIR.mkdir(parents=True, exist_ok=True)

site = maria.get_site("cerro_chajnantor", altitude = 5600)

band = Band(
    name="m2/f093",
    center=280e9,
    width=60e9,
    efficiency=eta,
    NET_CMB=13e-6,
    knee=1.0,
    gain_error=5e-2,
)

if not fits_path.exists():
    raise FileNotFoundError(f"FITS file not found: {fits_path}. Please run simple_ccat.py first to generate the TODs.")

tod = maria.tod.load(fits_path, site=site, band=[band])

ra_deg = np.rad2deg(tod.ra[DET_IDX, :])
dec_deg = np.rad2deg(tod.dec[DET_IDX, :])

az_deg = np.rad2deg(tod.az[DET_IDX, :])
el_deg = np.rad2deg(tod.el[DET_IDX, :])

time = np.arange(len(ra_deg)) / SAMPLE_RATE_HZ

dt = 1 / SAMPLE_RATE_HZ

valid = np.isfinite(ra_deg) & np.isfinite(dec_deg) & np.isfinite(az_deg) & np.isfinite(el_deg)

ra_deg = ra_deg[valid]
dec_deg = dec_deg[valid]
az_deg = az_deg[valid]
el_deg = el_deg[valid]
time = time[valid]

if len(time) < 3:
    raise RuntimeError("Not enough valid data points to compute motion histograms.")

velocity_ra = (np.cos(np.radians(dec_deg[1:-1])) * (ra_deg[2:] - ra_deg[:-2]) / (2 * dt))

velocity_dec = (dec_deg[2:] - dec_deg[:-2]) / (2 * dt)

acceleration_ra = (np.cos(np.radians(dec_deg[1:-1])) * (ra_deg[2:] - 2 * ra_deg[1:-1] + ra_deg[:-2]) / (dt ** 2))

acceleration_dec = (dec_deg[2:] - 2 * dec_deg[1:-1] + dec_deg[:-2]) / (dt ** 2)

projected_velocity_az = (np.cos(np.radians(el_deg[1:-1])) * (az_deg[2:] - az_deg[:-2]) / (2 * dt))

velocity_el = ((el_deg[2:] - el_deg[:-2]) / (2 * dt))

projected_acceleration_az = (np.cos(np.radians(el_deg[1:-1])) * (az_deg[2:] - 2 * az_deg[1:-1] + az_deg[:-2]) / (dt ** 2))

acceleration_el = ((el_deg[2:] - 2 * el_deg[1:-1] + el_deg[:-2])   / (dt ** 2))

motor_velocity_az = projected_velocity_az / np.cos(np.radians(el_deg[1:-1]))

motor_velocity_el = velocity_el

motor_acceleration_az = projected_acceleration_az / np.cos(np.radians(el_deg[1:-1]))

motor_acceleration_el = acceleration_el

def summarize_quantity(name, values, units):

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    stats = {
        "quantity": name,
        "units": units,
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "max_abs": np.max(np.abs(values)),
        "p95_abs": np.percentile(np.abs(values), 95),
        "p99_abs": np.percentile(np.abs(values), 99),
    }
    print(f"\n{name} [{units}]")
    print("-" * 50)
    print(f"Mean:    {stats['mean']:.6g}")
    print(f"Std:     {stats['std']:.6g}")
    print(f"Min:     {stats['min']:.6g}")
    print(f"Max:     {stats['max']:.6g}")
    print(f"Max |x|: {stats['max_abs']:.6g}")
    print(f"95% |x|: {stats['p95_abs']:.6g}")
    print(f"99% |x|: {stats['p99_abs']:.6g}")

    return stats

def plot_histogram(name, values, units, filename, absolute=False):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if absolute:
        values = np.abs(values)
        xlabel = f"|{name}| ({units})"
        title_name = f"Absolute {name}"
    else:
        xlabel = f"{name} ({units})"
        title_name = name

    plt.figure(figsize=(8, 6))

    plt.hist(
        values,
        bins=80,
        edgecolor="black",
        alpha=0.8,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Number of samples")
    plt.title(
        f"{title_name} Histogram\n"
        f"{SCAN_PATTERN}, Detector {DET_IDX}, Speed={SPEED:.2f} deg/s, Elev={ELEV_LABEL}"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(HIST_OUTDIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Quantities to analyze
# ============================================================

quantities = [
    ("RA Velocity", velocity_ra, "deg/s"),
    ("Dec Velocity", velocity_dec, "deg/s"),
    ("Projected Az Velocity", projected_velocity_az, "deg/s"),
    ("Elevation Velocity", velocity_el, "deg/s"),

    ("Motor Az Velocity", motor_velocity_az, "deg/s"),
    ("Motor El Velocity", motor_velocity_el, "deg/s"),

    ("RA Acceleration", acceleration_ra, "deg/s^2"),
    ("Dec Acceleration", acceleration_dec, "deg/s^2"),
    ("Projected Az Acceleration", projected_acceleration_az, "deg/s^2"),
    ("Elevation Acceleration", acceleration_el, "deg/s^2"),

    ("Motor Az Acceleration", motor_acceleration_az, "deg/s^2"),
    ("Motor El Acceleration", motor_acceleration_el, "deg/s^2"),
]


# ============================================================
# Print stats, save CSV, save histograms
# ============================================================

stats_rows = []

for name, values, units in quantities:
    stats_rows.append(summarize_quantity(name, values, units))

    safe_name = name.lower().replace(" ", "_").replace("/", "_")

    plot_histogram(
        name=name,
        values=values,
        units=units,
        filename=f"{run_prefix}_detector_{DET_IDX}_{safe_name}_histogram.png",
        absolute=False,
    )

    plot_histogram(
        name=name,
        values=values,
        units=units,
        filename=f"{run_prefix}_detector_{DET_IDX}_{safe_name}_absolute_histogram.png",
        absolute=True,
    )

stats_df = pd.DataFrame(stats_rows)
stats_csv_path = HIST_OUTDIR / f"{run_prefix}_detector_{DET_IDX}_motion_histogram_stats.csv"
stats_df.to_csv(stats_csv_path, index=False)

print(f"\nSaved histograms to: {HIST_OUTDIR}")
print(f"Saved statistics CSV to: {stats_csv_path}")