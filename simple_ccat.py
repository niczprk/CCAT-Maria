"""
CCAT-prime Pipeline Simple Execution Script

Workflow:
1)Convert JCMT FITS maps from mJy/arcsec^2 to Jy/sr
2)Fille NaN values in FITS maps
3)Clip FITS maps to smaller area (if necessary)
4)Plot dT/d(el) vs Elevation for varying tau_0
5)Build Instrument + site
6)Load input map, plan scan, simulate TOD
7)TOD diagnostics + plots
8)BinMappper map + plot
9)Plot atmosphere vs elevation (TOD averaged)

"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import maria
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from maria import Instrument
from maria.instrument import Band
from maria import fetch
from maria import Planner
from maria.mappers import BinMapper

import os, sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D



PREFIX = "OrionA"

OUTDIR = Path(f"outputs/{PREFIX}_ccat_test_outputs")

DATA_DIR = Path("data")

RAW_FITS = DATA_DIR / f"{PREFIX}_20170726_850_DR3_ext_HK.fits"
JYSR_FITS = DATA_DIR / f"{PREFIX}_20170726_850_DR3_ext_HK_JySr.fits"
FILLED_FITS = DATA_DIR / f"{PREFIX}_20170726_850_DR3_ext_HK_JySr_nan_filled.fits"
REDUCED_FITS = DATA_DIR / f"{PREFIX}_20170726_850_DR3_ext_HK_JySr_reduced_filled.fits"

CUTOUT_ORIONA = dict(ra_min=83.2, ra_max=84.0, dec_min=-6.0, dec_max=-4.8)

CUTOUT_SERPENSE = dict(ra_min=279.35, ra_max=279.765, dec_min=-2.0, dec_max=-1.0)

CUTOUT = CUTOUT_ORIONA

NU_HZ = 280e9  # 280 GHz
NU_GHZ = NU_HZ / 1e9

PWV_MM = 0.76  #  mm, precip water vapour

EL_LIMITS = (30, 80)  # degrees

T_0 = 278.868 #K, atmospheric ground temp

# -------- Simulation Parameters --------

START_TIME = "2022-02-11T12:00:00"
TOTAL_DURATION_S = 900  # seconds
SIM_DURATION_S = 900  # seconds
SAMPLE_RATE_HZ = 15  # Hz
SCAN_PATTERN = "daisy"
CHUNK_NUMBER = 0

# -------- Physical Constants --------

C = 299792458.0                 # m/s
K_B = 1.380649e-23              # J/K
H = 6.62607015e-34              # J*s
T_CMB = 2.7255                  # K
JY = 1e-26                      # W m^-2 Hz^-1

# -------- Utility Functions --------

def savefig(outdir: Path, name: str, dpi: int = 200) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("Saved:", path)
    return path


def first_image_hdu(hdul: fits.HDUList) -> tuple[int, fits.ImageHDU]:
    """Return (index, hdu) for the first HDU that contains image-like data."""
    idx = next(i for i, h in enumerate(hdul) if getattr(h, "data", None) is not None)
    return idx, hdul[idx]


def squeeze_to_2d(data: np.ndarray) -> np.ndarray:
    """If data is shape (1, ny, nx), return (ny, nx). Otherwise return as-is."""
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    return data


def to_deg_if_rad(x: np.ndarray) -> np.ndarray:
    """Convert to degrees if values look like radians (heuristic)."""
    x = np.asarray(x)
    if np.nanmax(np.abs(x)) < 10.0:
        return np.rad2deg(x)
    return x


# -------------------------
# FITS tools
# -------------------------

def convert_fits_units(IN: Path, OUT: Path, input_unit: str, output_unit: str) -> None:
    """
    Convert FITS image units by applying a multiplicative factor.

    Currently supports:
      mJy/arcsec^2 -> Jy/sr
    """
    unit_conversions = {
        ("mJy/arcsec^2", "Jy/sr"): (1e-3 / (1 / 206265) ** 2),
    }

    key = (input_unit, output_unit)
    if key not in unit_conversions:
        raise ValueError(f"Conversion {input_unit} -> {output_unit} not defined.")

    factor = unit_conversions[key]

    with fits.open(IN) as hdul:
        _, hdu = first_image_hdu(hdul)
        data = np.asarray(hdu.data, dtype=np.float64)

        hdu.data = data * factor
        hdu.header["BUNIT"] = output_unit
        hdu.header.add_history(f"Converted from {input_unit} to {output_unit}")
        hdu.header.add_history(f"Factor used: {factor:.6e} {output_unit} per ({input_unit})")

        hdul.writeto(OUT, overwrite=True)

    print("Wrote:", OUT)


def clip_fits_nans(IN: Path, OUT: Path, fill_value: float = 0.0) -> None:
    """Fill NaNs (and infinities) with fill_value and write new FITS."""
    with fits.open(IN) as hdul:
        _, hdu = first_image_hdu(hdul)
        data2d = squeeze_to_2d(hdu.data)

        data_filled = np.nan_to_num(data2d, nan=fill_value, posinf=fill_value, neginf=fill_value)
        fits.PrimaryHDU(data=data_filled.astype(np.float32), header=hdu.header).writeto(OUT, overwrite=True)

    print("Wrote:", OUT)


def clip_fits_area(
    IN: Path,
    OUT: Path,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> None:
    """Cut a rectangular sky region from a FITS image (using WCS) and write it."""
    ra_c = 0.5 * (ra_min + ra_max)
    dec_c = 0.5 * (dec_min + dec_max)

    # small-angle correction for RA width at dec_c
    width_deg = (ra_max - ra_min) * np.cos(np.deg2rad(dec_c))
    height_deg = (dec_max - dec_min)

    with fits.open(IN) as hdul:
        _, hdu = first_image_hdu(hdul)
        hdr = hdu.header
        wcs = WCS(hdr)

        data2d = squeeze_to_2d(hdu.data)

        center = SkyCoord(ra_c * u.deg, dec_c * u.deg, frame="icrs")
        size = u.Quantity((height_deg, width_deg), u.deg)  # (ny, nx)

        cutout = Cutout2D(
            data2d,
            position=center,
            size=size,
            wcs=wcs,
            mode="partial",
            fill_value=np.nan,
        )

        new_hdr = cutout.wcs.to_header()
        new_hdr["BUNIT"] = hdr.get("BUNIT", "Jy/sr")

        fits.PrimaryHDU(data=cutout.data.astype(np.float32), header=new_hdr).writeto(OUT, overwrite=True)

    print("Wrote:", OUT)
    print("Patch center (deg):", ra_c, dec_c)
    print("Patch size (deg):", height_deg, width_deg)

#------ Unit/Atmosphere Tools -------


def dB_dT(nu_GHz: float, T: float = T_CMB) -> float:
    """Planck-law derivative dB/dT at frequency nu_GHz and temperature T."""
    nu = nu_GHz * 1e9
    x = H * nu / (K_B * T)
    ex = np.exp(x)
    return (2.0 * K_B * nu**2 / C**2) * (x**2 * ex) / (ex - 1.0) ** 2


def beam_solid_angle_gaussian(fwhm_arcsec: float) -> float:
    """Gaussian beam solid angle Ω = (π / (4 ln2)) * θ_FWHM^2."""
    theta_rad = fwhm_arcsec * (np.pi / (180.0 * 3600.0))
    return (np.pi / (4.0 * np.log(2.0))) * theta_rad**2


def convert_noise_equivalent(
    initial: str,
    final: str,
    nu_GHz: float,
    value: float,
    beam_fwhm_arcsec: float | None = None,
    N_det: int | None = None,
    T: float = T_CMB,
) -> float:
    """
    Convert between NEI (Jy/sr*sqrt(s)), NEFD (Jy/beam*sqrt(s)), NET (K*sqrt(s)).
    Notes:
      - NEI<->NEFD requires beam and N_det (as written here assumes array-combined convention).
    """
    initial = initial.upper()
    final = final.upper()
    if initial == final:
        return value

    dBdT_val = dB_dT(nu_GHz, T=T)  # W m^-2 Hz^-1 sr^-1 K^-1

    if initial == "NEI" and final == "NET":
        NEI_SI = value * JY
        return NEI_SI / dBdT_val

    if initial == "NET" and final == "NEI":
        return (value * dBdT_val) / JY

    if beam_fwhm_arcsec is None or N_det is None:
        raise ValueError("NEFD conversions require beam_fwhm_arcsec and N_det.")

    omega_beam = beam_solid_angle_gaussian(beam_fwhm_arcsec)

    if initial == "NEI" and final == "NEFD":
        return value * omega_beam * np.sqrt(N_det)

    if initial == "NEFD" and final == "NEI":
        return value / (omega_beam * np.sqrt(N_det))

    if initial == "NET" and final == "NEFD":
        nei_si = value * dBdT_val  # W m^-2 Hz^-1 sr^-1 sqrt(s)
        nei = nei_si / JY
        return nei * omega_beam * np.sqrt(N_det)

    if initial == "NEFD" and final == "NET":
        nei = value / (omega_beam * np.sqrt(N_det))
        nei_si = nei * JY
        return nei_si / dBdT_val

    raise ValueError("Invalid conversion types. Options: NEI, NET, NEFD.")


def effective_atm_temp_850GHz(*, pwv: float, tau_0: float = None, el_deg: float | np.ndarray, T_0: float = T_0) -> np.ndarray:
    """Effective atmospheric temperature Teff = T0(1-exp(-tau(el))) using JCMT tau0(PWV)."""
    if tau_0 is not None:
        tau_0 = float(tau_0)
        
    elif tau_0 is None:
        tau_0 = 0.179 * (pwv + 0.337) #SCUBA-2 JCMT 850 micron tau0(PWV) relation

    elif pwv is None:
        raise ValueError("Provide either tau_0 or pwv.")
    else:
        raise ValueError("Provide either tau_0 or pwv.")
    
    z_rad = np.deg2rad(90.0 - np.asarray(el_deg))
    tau_z = tau_0 / np.cos(z_rad)
    return T_0 * (1.0 - np.exp(-tau_z))

def tau_0_from_atm_temp(T_atm: float, el_deg: float, T_0: float = T_0) -> float:
    """Given measured atmospheric temperature T_atm at elevation el_deg, return tau_0."""
    z_rad = np.deg2rad(90.0 - el_deg)
    tau_el = -np.log(1.0 - T_atm / T_0)
    tau_0 = tau_el * np.cos(z_rad)
    return tau_0


def inst_effective_atm_temp_850GHz(
    *,
    mode: str,
    tau_0: float | None = None,
    pwv: float | None = None,
    el_deg: float | np.ndarray,
    delta_el: float = 1.0,
    T_0: float = T_0,
) -> np.ndarray:
    """
    Return dTeff/d(el) in K/deg, either analytic ("inst") or finite-difference ("fin_diff").
    Accepts el_deg as scalar or array.
    """
    if tau_0 is None:
        if pwv is None:
            raise ValueError("Provide either tau_0 or pwv.")
        tau_0 = 0.179 * (pwv + 0.337)

    el = np.asarray(el_deg)
    z_rad = np.deg2rad(90.0 - el)
    tau_z = tau_0 / np.cos(z_rad)

    if mode == "inst":
        # dTeff/dz(rad) then convert to per-degree elevation:

        dT_dz_rad = T_0 * np.exp(-tau_z) * (tau_0 * np.sin(z_rad) / (np.cos(z_rad) ** 2))

        dT_del_rad = dT_dz_rad
        return - dT_del_rad * (np.pi / 180.0)  # K per degree

    if mode == "fin_diff":
        # central difference in degrees
        teff_plus = effective_atm_temp_850GHz(pwv=pwv if pwv is not None else 0.0, el_deg=el + 0.5 * delta_el, T_0=T_0)
        teff_minus = effective_atm_temp_850GHz(pwv=pwv if pwv is not None else 0.0, el_deg=el - 0.5 * delta_el, T_0=T_0)
        return np.abs((teff_plus - teff_minus) / delta_el)

    raise ValueError("Invalid mode. Choose 'inst' or 'fin_diff'.")

def compare_maria_atm_temp(
        T_atm: float,
        el_deg: float,
        mode: str = "inst",
        T_0: float = T_0
):
    """
    Compare maria atmosphere temperature derivative dT/del at given T_atm and el_deg. 
    Input a measured atmospheric temperature T_atm at elevation el_deg, and compute tau_0 from that.
    Then compute dT/del using the inst_effective_atm_temp_850GHz function and print the result along with the corresponding tau_0.
    This can be used to check if the maria atmosphere model is consistent with the expected atmospheric approximations
    for temperature and its elevation dependence.
    """

    tau_0 = tau_0_from_atm_temp(
        T_atm=T_atm,
        el_deg=el_deg,
        T_0=T_0
    )
    dT_del = inst_effective_atm_temp_850GHz(
        mode=mode,
        tau_0=tau_0,
        el_deg=el_deg,
        T_0=T_0
    )
    print(f"Atm dT/del: {dT_del:.6f} K/deg for /mathrm$tau_0$={tau_0:.4f} at el={el_deg:.4f} deg")

# ---------- Main Execution Pipeline -----------

def main(
    run_mode: str = "only_atm",
    atm_plot: bool = True,
    temp_mode: str = "inst",
    map_type: str = "BinMapper",
    tod_diagnostics: bool = True,
) -> None:
    """
    run_mode options:
      - "only_atm": run ONLY the atmospheric plots, then exit
      - "only_sim": run simulation/mapmaking pipeline ONLY (skip atmospheric plots)
      - "all": run atmospheric plots + simulation/mapmaking pipeline
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)

    run_mode = run_mode.lower().strip()
    if run_mode not in ("only_atm", "only_sim", "all"):
        raise ValueError("run_mode must be one of: 'only_atm', 'only_sim', 'all'")

    do_atm = (run_mode in ("only_atm", "all")) and atm_plot
    do_sim = (run_mode in ("only_sim", "all"))

    # -----------------------------
    # Atmosphere plots helper
    # -----------------------------
    def run_atmosphere_plots() -> None:
        x = np.linspace(EL_LIMITS[0], EL_LIMITS[1], 150)
        taus = np.linspace(0.05, 0.25, 5)

        plt.figure(figsize=(8, 6))
        for tau in taus:
            y = inst_effective_atm_temp_850GHz(mode=temp_mode, tau_0=float(tau), el_deg=x)
            plt.plot(x, y, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$dT/d\mathrm{el}$ (K/deg)")
        plt.title(r"Instantaneous $dT/d\mathrm{el}$ vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "inst_dT_del_tau0_0p05_to_0p25.png")

        plt.figure(figsize=(8, 6))
        for tau in taus:
            a = effective_atm_temp_850GHz(pwv=PWV_MM, tau_0=float(tau), el_deg=x)
            plt.plot(x, a, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$T_\mathrm{eff}$ (K)")
        plt.title(r"Effective Atmospheric Temperature $T_\mathrm{eff}$ vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "Teff_tau0_0p05_to_0p25.png")

        plt.figure(figsize=(8, 6))
        for tau in taus:
            y = inst_effective_atm_temp_850GHz(mode=temp_mode, tau_0=float(tau), el_deg=x)
            a = effective_atm_temp_850GHz(pwv=PWV_MM, tau_0=float(tau), el_deg=x)
            plt.plot(x, y / a, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$\frac{dT/d\mathrm{el}}{T_\mathrm{eff}}$")
        plt.title(r"Fractional Atmospheric Temperature Derivative vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "frac_dT_del_Teff_tau0_0p05_to_0p25.png")

        plt.figure(figsize=(8, 6))
        for tau in taus:
            y = inst_effective_atm_temp_850GHz(mode=temp_mode, tau_0=float(tau), el_deg=x)
            a = effective_atm_temp_850GHz(pwv=PWV_MM, tau_0=float(tau), el_deg=x)
            plt.plot(x, 1- (y / a), label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$1-\frac{dT/d\mathrm{el}}{T_\mathrm{eff}}$")
        plt.title(r"Fractional Diff Atmospheric Temperature Derivative vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "frac_diff_dT_del_Teff_tau0_0p05_to_0p25.png")

    # -----------------------------
    # 1) Atmosphere-only mode
    # -----------------------------
    if run_mode == "only_atm":
        if do_atm:
            run_atmosphere_plots()
        else:
            print("[skip] atmosphere plots (atm_plot=False)")
        return  # <-- stop here: do not run sim/mapmaking

    # -----------------------------
    # 2) FITS prep + Simulation modes
    #    ("only_sim" or "all")
    # -----------------------------
    if not RAW_FITS.exists():
        raise FileNotFoundError(f"Input FITS file not found: {RAW_FITS}")

    # FITS prep
    convert_fits_units(RAW_FITS, JYSR_FITS, input_unit="mJy/arcsec^2", output_unit="Jy/sr")
    clip_fits_nans(JYSR_FITS, FILLED_FITS, fill_value=0.0)
    clip_fits_area(FILLED_FITS, REDUCED_FITS, **CUTOUT)

    # Atmosphere plots for "all"
    if do_atm:
        run_atmosphere_plots()
    else:
        print("[skip] atmosphere plots (run_mode='only_sim' or atm_plot=False)")

    # -----------------------------
    # Instrument
    # -----------------------------
    f280 = Band(
        center=280e9,
        width=40e9,
        NET_CMB=13e-6,
        knee=1.0,
        gain_error=5e-2,
    )

    array_cfg = {
        "shape": "hexagon",
        "field_of_view": 1.3,
        "beam_spacing": 2.0,
        "primary_size": 6.0,
        "bands": [f280],
        "polarized": False,
    }

    instrument = maria.get_instrument(array=array_cfg)
    print(instrument)
    instrument.plot()
    savefig(OUTDIR, f"{PREFIX}_instrument_overview.png")

    # -----------------------------
    # Site and Map
    # -----------------------------
    site = maria.get_site("cerro_chajnantor", altitude=5600)
    input_map = maria.map.load(str(REDUCED_FITS), nu=NU_HZ)

    print(input_map)
    input_map.to("Jy/sr").plot(cmap="coolwarm")
    savefig(OUTDIR, f"{PREFIX}_input_map_JySr_PWV{PWV_MM:.2f}.png")

    # -----------------------------
    # Scan Planning
    # -----------------------------
    planner = Planner(
        start_time=START_TIME,
        target=input_map,
        site=site,
        constraints={"el": EL_LIMITS},
    )

    plans = planner.generate_plans(
        total_duration=TOTAL_DURATION_S,
        max_chunk_duration=SIM_DURATION_S,
        scan_pattern=SCAN_PATTERN,
        sample_rate=SAMPLE_RATE_HZ,
        scan_options={"radius": input_map.width.deg / 3.0},
    )

    plans[0].plot()
    print(plans)
    savefig(OUTDIR, f"{PREFIX}_daisy_plan_PWV{PWV_MM:.2f}.png")

    # -----------------------------
    # TOD Simulation
    # -----------------------------
    sim = maria.Simulation(
        instrument=instrument,
        plans=plans,
        site=site,
        atmosphere="2d",
        atmosphere_kwargs={"weather": {"pwv": PWV_MM}},
        cmb="generate",
        map=input_map,
    )

    print(sim)
    tods = sim.run()
    print(tods)

    tods[0].plot()
    plt.savefig(
        os.path.join(OUTDIR, f"{PREFIX}_tod_plot_{CHUNK_NUMBER}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    # -----------------------------
    # TOD Diagnostics
    # -----------------------------
    def chunk_summary(tod, i: int) -> None:
        el = to_deg_if_rad(tod.el)
        el0 = np.nanmean(el, axis=0)
        airmass = 1.0 / np.sin(np.deg2rad(el0))
        print(
            f"TOD {i}: el median={np.nanmedian(el0):.2f} deg, "
            f"airmass median={np.nanmedian(airmass):.2f}, "
            f"el range=({np.nanpercentile(el0, 5):.1f}, {np.nanpercentile(el0, 95):.1f})"
        )

    tod0 = tods[CHUNK_NUMBER]

    if tod_diagnostics:
        for i, tod in enumerate(tods):
            chunk_summary(tod, i)

        for k in tod0.data.keys():
            arr = np.asarray(tod0.data[k])
            print(
                f"TOD data key: {k}, shape={arr.shape}, dtype={arr.dtype}, "
                f"min={arr.min(): .3e}, max={arr.max(): .3e}, std={arr.std(): .3e}"
            )

        tod0.plot()
        savefig(OUTDIR, f"{PREFIX}_tod_plot_chunk{CHUNK_NUMBER}_PWV{PWV_MM:.2f}.png")
    else:
        print("[skip] tod diagnostics")

    # -----------------------------
    # Pointing plots (az/el and ra/dec)
    # -----------------------------
    t = np.asarray(tod0.time)
    tsec = t - t[0]

    az0 = np.nanmean(to_deg_if_rad(tod0.az), axis=0)
    el0 = np.nanmean(to_deg_if_rad(tod0.el), axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(tsec, el0, label="el (deg)")
    plt.plot(tsec, az0, label="az (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees")
    plt.title("Telescope Pointing vs Time")
    plt.legend()
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_tod_pointing_azel_PWV{PWV_MM:.2f}.png")

    ra0 = np.nanmean(to_deg_if_rad(tod0.ra), axis=0)
    dec0 = np.nanmean(to_deg_if_rad(tod0.dec), axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(tsec, dec0, label="dec (deg)")
    plt.plot(tsec, ra0, label="ra (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees")
    plt.title("Telescope RA/Dec vs Time")
    plt.legend()
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_tod_pointing_radec_PWV{PWV_MM:.2f}.png")

    # -----------------------------
    # Atmosphere vs elevation (TOD averaged)
    # -----------------------------



    el0 = np.nanmean(to_deg_if_rad(tod0.el), axis=0)
    T_atm0 = np.nanmean(np.asarray(tod0.data["atmosphere"]), axis=0)

    m = np.isfinite(el0) & np.isfinite(T_atm0) & (T_atm0 > 0.0) & (T_atm0 < 0.98 * T_0)

    elv = el0[m]
    Tv = T_atm0[m]

    tau0_samples = tau_0_from_atm_temp(
        T_atm = Tv,
        el_deg = elv,
        T_0 = T_0
    )

    tau0_ref = np.median(tau0_samples)

    el_ref = np.nanmedian(elv)
    T_ref = np.nanmedian(Tv)

    dTdel_ref = inst_effective_atm_temp_850GHz(
        mode="inst",
        tau_0=tau0_ref,
        el_deg=el_ref,
        T_0=T_0
    )
    
    # tau0_est = tau_0_from_atm_temp(
    #     T_atm=T_ref,
    #     el_deg=el_ref,
    #     T_0=T_0
    # )

    print(f"Inferred tau_0 from TOD-avg atmosphere: {tau0_ref:.4f} (PWV={PWV_MM:.2f} mm)")
    print(f"Expected dT/del at el={el_ref:.2f} deg: {dTdel_ref:.6f} K/deg")

    # a, b = np.polyfit(elv, Tv, deg=1)

    # print(f"Atmosphere vs Elevation fit: T_atm = {a:.4f} * el + {b:.4f} [K]")

    xfit = np.linspace(elv.min(), elv.max(), 100)
    yfit = T_ref + dTdel_ref * (xfit - el_ref)

    step = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(elv[::step], Tv[::step], s=1, alpha=0.5, label="TOD samples")
    plt.plot(xfit, yfit, lw=2, alpha= 0.75, color="red", label=
        fr"Model tangent "
        fr"($\tau_0={tau0_ref:.4f}$, "
        fr"$dT/d\mathrm{{el}}={dTdel_ref:.4f}\,\mathrm{{K/deg}}$)")
    plt.xlabel("Elevation (deg)")
    plt.ylabel("Atmospheric Temperature (K)")
    plt.title(f"Atmospheric Temperature vs Elevation (PWV={PWV_MM:.2f} mm, TOD-avg)")
    plt.grid(True)
    plt.legend()
    savefig(OUTDIR, f"{PREFIX}_atmosphere_vs_el_elref{el_ref:.2f}_tau0_inferred_tangent_PWV{PWV_MM:.2f}.png")
    # -----------------------------
    # BinMapper Mapmaking
    # -----------------------------
    map_type_norm = map_type.lower().strip()

    if map_type_norm == "binmapper":
        mapper = BinMapper(
            tod_preprocessing={
                "remove_spline": {"knot_spacing": 60, "remove_el_gradient": True},
                "remove_modes": {"modes_to_remove": 1},
            },
            map_postprocessing={"gaussian_filter": {"sigma": 1}},
            units="mK_RJ",
            tods=tods,
        )

        output_map = mapper.run()
        output_map.plot(nu_index=[0], cmap="coolwarm")
        savefig(OUTDIR, f"{PREFIX}_output_BinMapper_PWV{PWV_MM:.2f}.png")

    elif map_type_norm in ("none", "skip", "no"):
        print("[skip] mapmaking")

    else:
        raise ValueError(f"Unknown map_type={map_type!r}. Try 'BinMapper' or 'skip'.")

if __name__ == "__main__":
    # main(atm_plot=False, map_type="skip", tod_diagnostics=False)
    main(atm_plot=True, run_mode= "all" , temp_mode="inst", map_type="Binmapper", tod_diagnostics=True)

    # compare_maria_atm_temp(T_atm=20.0, el_deg= 33.5, mode="inst", T_0=T_0)

    # plt.figure(figsize=(8, 6))
    # x = np.linspace(30, 33.5, 15)
    # a = effective_atm_temp_850GHz(pwv=PWV_MM, tau_0 = 0.0409, el_deg=x)
    # plt.plot(x, a, label=fr"$\tau_0$={0.0409:.4f}")
    # plt.xlabel("Elevation (deg)")
    # plt.ylabel(r"$T_\mathrm{eff}$ (K)")
    # plt.title(r"Effective Atmospheric Temperature $T_\mathrm{eff}$ vs Elevation for varying $\tau_0$")
    # plt.grid(True)
    # plt.legend(ncol=2, fontsize=9)
    # savefig(OUTDIR, "Teff_tau0_0p0409.png")

raise SystemExit("Stopping After Main Execution Pipeline.")