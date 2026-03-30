from __future__ import annotations
import os, sys

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import maria
from maria import Instrument
from maria.instrument import Band
from maria import fetch
from maria import Planner
from maria.mappers import BinMapper
from maria.spectrum import AtmosphericSpectrum

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

# for multiple runs of maria in the same session

os.environ["OMP_NUM_THREADS"] = "1"  # Avoid multithreading issues in numpy/scipy
os.environ["MKL_NUM_THREADS"] = "1" # Avoid multithreading issues in numpy/scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1" # Avoid multithreading issues in numpy/scipy
os.environ["NUMEXPR_NUM_THREADS"] = "1" # Avoid multithreading issues in numpy/scipy

# -----------------------------
# Fits // Directory Parameters
# -----------------------------

FITS_PREFIX = "OrionA"

DATA_DIR = Path("data")

RAW_FITS = DATA_DIR / f"{FITS_PREFIX}_20170726_850_DR3_ext_HK.fits"
JYSR_FITS = DATA_DIR / f"{FITS_PREFIX}_20170726_850_DR3_ext_HK_JySr.fits"
FILLED_FITS = DATA_DIR / f"{FITS_PREFIX}_20170726_850_DR3_ext_HK_JySr_nan_filled.fits"
REDUCED_FITS = DATA_DIR / f"{FITS_PREFIX}_20170726_850_DR3_ext_HK_JySr_reduced_filled.fits"

CCAT_DATA = DATA_DIR / "atm-table-ccat.dat"

CUTOUT_ORIONA = dict(ra_min=83.2, ra_max=84.0, dec_min=-6.0, dec_max=-4.8)

CUTOUT_SERPENSE = dict(ra_min=279.35, ra_max=279.765, dec_min=-2.0, dec_max=-1.0)

CUTOUT = CUTOUT_ORIONA

PREFIX = "OrionA_220_GHz"#_280GHz_eta0.5_pwr_coupling_polarized" # for output files, e.g. "OrionA_polarized"

OUTDIR = Path(f"outputs/{PREFIX}_ccat_outputs")

# -----------------------------
#  Simulation Parameters 
# -----------------------------
selected_band = "220" #make sure these match

NU_HZ = 220e9  # Hz
bandwidth_hz = 56e9  # GHz bandwidth for 220 GHz band
eta = 0.5 # optical efficiency for this estimate

Polarized = False # whether to include polarization in the simulation

if Polarized:
    p = 0.5 # fractional polarization of the source, for this estimates
else:
    p = 1.0

f_res = 800e6  # Resonant frequency in Hz (800 MHz)

Q_r = 40000 # Quality factor taken from Bayguchi thesis

P_0 = 957e-18 # idk but do not question the mighty jordan wheeler

R_0 = -2.448e9 #avg responsivity in W^-1 from Jordan Wheeler

Del_f = 2200 # Hz, 1/10th of the FWHM is the estimated linear regime limit for 350GHz MKID array

NU_GHZ = NU_HZ / 1e9 #GHz

PWV_MM = 0.36  #  mm, precip water vapour this only affects the main if pwv is None

# 0.36, 0.67, & 1.28 are Q1, Q2, and Q3 zenith PMV values for Chajnantor

EL_LIMITS = (30, 70)  # degrees

T_0 = 278.868 #K, atmospheric ground temp



START_TIME = "2022-02-10T18:55:00"

#"2022-02-10T20:30:00" for around 60 degrees 
#"2022-02-10T18:55:00" for roughly 45 degrees
#"2022-02-10T18:30:00" for roughly 40 degrees
#"2022-02-10T17:00:00" for roughly 30 degrees
 

TOTAL_DURATION_S = 1800  # seconds
SIM_DURATION_S = 1800  # seconds
SAMPLE_RATE_HZ = 15  # Hz
SCAN_PATTERN = "daisy"
CHUNK_NUMBER = 0

# ------------------------------
#  Physical Constants 
# ------------------------------

C = 299792458.0                 # m/s
K_B = 1.380649e-23              # J/K
H = 6.62607015e-34              # J*s
T_CMB = 2.7255                  # K
JY = 1e-26                      # W m^-2 Hz^-1

# -----------------------------
#  Utility Functions 
# -----------------------------

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

def inst_power_per_deg(bandwidth: float, eta: float, dT_del: float) -> float:
    """
    Return the power change per degree of elevation due to atmospheric temperature change, in likely pico-Watts/deg.
    This can be used to estimate how much the atmospheric loading on the instrument changes with elevation, which is important for understanding gain variations and calibration.
    The power change can be approximated as dP/del = K_B * eta * bandwidth * dT/del, where eta is the optical efficiency and bandwidth is the effective bandwidth of the instrument.
    Eta in this case will be assumed to be 1.0 for a simple estimate, and bandwidth can be taken as the width of the band (e.g. 40 GHz for the 280 GHz band).
    The dT/del can be computed using the inst_effective_atm_temp_850GHz function for given tau_0 and elevation.
    """

    #haven't implemented this anywhere yet. It will happen
    return K_B * eta * bandwidth * dT_del

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

def deltaP_for_delta_el(el_ref_deg: float, delta_el_deg: float, tau_0: float, bandwidth: float, eta: float, T_0: float = T_0) -> float:
    """
    Estimate the change in atmospheric power loading (delta_P) for a given change in elevation (delta_el_deg) around a reference elevation (el_ref_deg),
    using the atmospheric temperature derivative and the instrument parameters.
    """


    e1 = el_ref_deg - 0.5 * delta_el_deg
    e2 = el_ref_deg + 0.5 * delta_el_deg

    delta_P = K_B * bandwidth * eta * 1e12 * (effective_atm_temp_850GHz(pwv=None, tau_0=tau_0, el_deg=e2, T_0=T_0) - effective_atm_temp_850GHz(pwv=None, tau_0=tau_0, el_deg=e1, T_0=T_0)) 

    return delta_P

# ----------------------------------------------
# ---------- Main Execution Pipeline -----------
# ----------------------------------------------

def main(
    run_mode: str = "only_atm",
    atm_plot: bool = True,
    temp_mode: str = "inst",
    ccat_band: str = "280",
    map_type: str = "BinMapper",
    tod_diagnostics: bool = True,
    pwv_mm: float = PWV_MM,
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
        taus = np.linspace(0.01, 0.10, 5)

        plt.figure(figsize=(8, 6))
        for tau in taus:
            y = inst_effective_atm_temp_850GHz(mode=temp_mode, tau_0=float(tau), el_deg=x)
            plt.plot(x, y, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$dT/d\mathrm{el}$ (K/deg)")
        plt.title(r"Instantaneous $dT/d\mathrm{el}$ vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "inst_dT_del_tau0_0p01_to_0p10.png")

        plt.figure(figsize=(8, 6))
        for tau in taus:
            a = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x)
            plt.plot(x, a, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$T_\mathrm{eff}$ (K)")
        plt.title(r"Effective Atmospheric Temperature $T_\mathrm{eff}$ vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "Teff_tau0_0p01_to_0p10.png")

        plt.figure(figsize=(8, 6))
        for tau in taus:
            y = inst_effective_atm_temp_850GHz(mode=temp_mode, tau_0=float(tau), el_deg=x)
            a = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x)
            plt.plot(x, y / a, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$\frac{dT/d\mathrm{el}}{T_\mathrm{eff}}$")
        plt.title(r"Fractional Atmospheric Temperature Derivative vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "frac_dT_del_Teff_tau0_0p01_to_0p10.png")

        plt.figure(figsize=(8, 6))
        for tau in taus:
            y = inst_effective_atm_temp_850GHz(mode=temp_mode, tau_0=float(tau), el_deg=x)
            a = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x)
            plt.plot(x, 1- (y / a), label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$1-\frac{dT/d\mathrm{el}}{T_\mathrm{eff}}$")
        plt.title(r"Fractional Diff Atmospheric Temperature Derivative vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "frac_diff_dT_del_Teff_tau0_0p01_to_0p10.png")

        # -----------------------------
        # Power change per degree: dP/del
        # -----------------------------

        plt.figure(figsize=(8, 6))
        for tau in taus:
            P_atm_pW = K_B * effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0) * bandwidth_hz * eta * 1e12 # pW
            plt.plot(x, P_atm_pW, label=fr"$\tau_0$={tau:.2f}")

        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"Power (pW)")
        plt.title(
            fr"Atmospheric power loading vs Elevation "
            fr"($\Delta\nu$={bandwidth_hz/1e9:.0f} GHz, $\eta$={eta:.2f})"
        )
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "atm_power_vs_elevation.png")

        plt.figure(figsize=(8, 6))
        for tau in taus:
            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
                bandwidth=bandwidth_hz,
                eta=eta,
                dT_del=dT_del,
            )  # W/deg

            plt.plot(x, dP_del_W * 1e12, label=fr"$\tau_0$={tau:.2f}")

        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$dP/d\mathrm{el}$ (pW/deg)")
        plt.title(
            fr"Atmospheric loading slope $dP/d\mathrm{{el}}$ vs Elevation "
            fr"($\Delta\nu$={bandwidth_hz/1e9:.0f} GHz, $\eta$={eta:.2f})"
        )
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "inst_dP_del_tau0_0p01_to_0p10.png")

        # -----------------------------
        # Responsivity fit R(P): fit in W, but derived from pW points
        # -----------------------------
        from scipy.optimize import curve_fit

        R_data = np.array(
            [-2.375, -2.1, -1.92, -1.87, -1.81, -1.75, -1.68, -1.61, -1.525,  -1.505, -1.5],
            dtype=float,
        ) * 1e8  # responsivity

        P_data_pW = np.array([0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5], dtype=float)
        def model_responsivity_func_pW(P_pW, a, b, c):
            return a * P_pW**2 + b * P_pW + c

        (a_fit, b_fit, c_fit), cov = curve_fit(model_responsivity_func_pW, P_data_pW, R_data)

        P_grid_pW = np.linspace(P_data_pW.min(), P_data_pW.max(), 200)
        R_fit = model_responsivity_func_pW(P_grid_pW, a_fit, b_fit, c_fit)

        plt.figure(figsize=(8, 6))
        plt.plot(P_data_pW, R_data, label="Estimated data points")
        plt.plot(P_grid_pW, R_fit, lw=2, label=rf"Quadratic fit (P in pW)")
        # $R(P) = ({a_fit:.2e})P^2 + ({b_fit:.2e})P + ({c_fit:.2e})$"
        plt.xlabel("Power (pW)")
        plt.ylabel(r"Responsivity $R$ (paper units)")
        plt.title("Fitted responsivity model R(P)")
        plt.grid(True)
        plt.legend()
        savefig(OUTDIR, "fitted_responsivity_model.png")

        # -----------------------------
        # Piecewise Responsivity Function for powers past 8.5 pW
        # -----------------------------

        P_MAX_PW = 8.5
        R_CONST = -1.5e8

        def R_piecewise(P_pW):
            """
            Use quadratic fit for P<=8.5 pW, else constant R_CONST.
            """
            P = np.asarray(P_pW, dtype=float)
            R_quad = model_responsivity_func_pW(P, a_fit, b_fit, c_fit)
            return np.where(P <= P_MAX_PW, R_quad, R_CONST)

        def R_paper(P_W):
            """
            Model used in 280 GHz result paper, J Wheeler
            """
            return R_0 / (np.sqrt(1+ P_W / P_0))

        # -----------------------------
        # Paper Responsivity Model Plot
        # -----------------------------

        P_plot_pW = np.logspace(-5, 1, 200)          # pW
        P_plot_W  = P_plot_pW * 1e-12                       # W

        plt.figure(figsize=(8, 6))

        # Paper model evaluated in W, but plotted vs pW
        plt.plot(P_plot_pW, R_paper(P_plot_W) / 1e8, label="Paper model (expects W)")

        plt.xlabel("Power (pW)")
        plt.ylabel(r"Responsivity $R$ ($10^8$ $W^{-1}$)")
        plt.title("Approximate Responsivity Model")
        plt.xscale("log")
        plt.grid(True)
        plt.legend()
        savefig(OUTDIR, "new_280_responsivity_model.png")

        # -----------------------------
        # frequency slope vs elevation from power + responsivity fit
        # -----------------------------

        plt.figure(figsize=(8, 6))
        for tau in taus:

            Teff = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0)

            P_atm_W = K_B * Teff * bandwidth_hz * eta # W
            P_atm_pW = P_atm_W * 1e12 # convert to pW

            R_eval = R_paper(P_atm_W) 

            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
                bandwidth=bandwidth_hz,
                eta=eta,
                dT_del=dT_del,
            )  # W/deg

            # R_eval = R_piecewise(P_atm_pW)  # pW

            df_del_Hz_per_deg = f_res * R_eval * dP_del_W  # Hz/deg

            print(
                f"tau={tau:.2f}: dT/del median={np.median(dT_del):.6f} K/deg, "
                f"dP/del median={np.median(dP_del_W)*1e12:.6f} pW/deg"
            )
            print(
                f"tau={tau:.2f}  "
                f"P_atm_pW: min/med/max={P_atm_pW.min():.3g}/{np.median(P_atm_pW):.3g}/{P_atm_pW.max():.3g}  "
                f"R_eval: min/med/max={R_eval.min():.3g}/{np.median(R_eval):.3g}/{R_eval.max():.3g}  "
                f"df/del med={np.median(df_del_Hz_per_deg):.3g} Hz/deg"
            )

            plt.plot(x, df_del_Hz_per_deg / 1e6, label=fr"$\tau_0$={tau:.2f}")

        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$df/d\mathrm{el}$ (MHz/deg)")
        plt.title(r"Predicted frequency slope vs elevation: $df/d\mathrm{el}=f_\mathrm{res}R(P)\,dP/d\mathrm{el}$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "predicted_df_del_vs_elevation.png")

        # -----------------------------
        #  Elevation Change allowed before leaving linear regime
        # -----------------------------


        plt.figure(figsize=(8, 6))
        for tau in taus:
            
            Teff = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0)

            P_atm_W = K_B * Teff * bandwidth_hz * eta # W
            P_atm_pW = P_atm_W * 1e12 # convert to pW

            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
                bandwidth=bandwidth_hz,
                eta=eta,
                dT_del=dT_del,
            )  # W/deg

            # R_eval = R_piecewise(P_atm_pW)  # pW

            R_eval = R_paper(P_atm_W) # convert back to W for paper model

            df_del_Hz_per_deg = f_res * R_eval * dP_del_W  # Hz/deg

            del_el_deg = Del_f / np.abs(df_del_Hz_per_deg)  # deg, elevation change corresponding to 2200 Hz frequency shift

            plt.plot(x, del_el_deg, label=fr"$\tau_0$={tau:.2f}")
        
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r" $\Delta\mathrm{el}$ for $|\Delta f|<2200$ Hz (deg)")
        plt.title(r"Estimated elevation change allowed before leaving linear regime")
        plt.ylim(0, 10.0)
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, f"allowed_del_el_for_linear_regime_eta{eta:.1f}.png")
    # -------------------------------------------------------
    # elevation change expressed in Delta P for 2200 Hz shift
    # -------------------------------------------------------

        plt.figure(figsize=(8, 6))
        for tau in taus:
            Teff = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0)

            P_atm_W = K_B * Teff * bandwidth_hz * eta # W
            P_atm_pW = P_atm_W * 1e12 # convert to pW

            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
                bandwidth=bandwidth_hz,
                eta=eta,
                dT_del=dT_del,
            )  # W/deg

            # R_eval = R_piecewise(P_atm_pW)  # pW

            R_eval = R_paper(P_atm_W) # convert back to W for paper model

            # Del_P = np.abs(1 / (10 * Q_r * R_eval))# pW, power change corresponding to 2200 Hz shift based on responsivity and Q_r

            Del_P = (Del_f/f_res) * np.abs(1/R_eval) # W, power change corresponding to 2200 Hz shift based on responsivity and frequency shift ratio

            Del_P_pW = Del_P * 1e12 # convert to pW

            plt.plot(x, Del_P_pW, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r" $|\Delta P|$ for $|\Delta f|\leq 2200$ ($pW$)")
        plt.title(r"Estimated power change allowed before leaving linear regime")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, f"allowed_Del_P_for_linear_regime_eta{eta:.1f}.png")

    # -------------------------------------------------------
    # change in power for given elevation change, of reference elevation, and tau_0
    # -------------------------------------------------------

        # el_ref = 45.0 # degrees, reference elevation for some of the plots

        # deltas = [1.0, 2.0, 5.0] # degrees, elevation changes


        # plt.figure(figsize=(8, 6))
        # for delta in deltas:

        #     dP = np.array([deltaP_for_delta_el(el_ref_deg=el_ref, delta_el_deg=delta, tau_0=float(tau), bandwidth=bandwidth_hz, eta=eta, T_0=T_0) for tau in taus])
        #     plt.plot(taus, np.abs(dP), label=fr"$\Delta\mathrm{{el}}$={delta:.1f} deg")

        # plt.xlabel(r"Zenith opacity $\tau_0$")
        # plt.ylabel(r"$|\Delta P|$ (pW)")
        # plt.title(fr"Atmospheric loading change vs $\tau_0$ (Δν={bandwidth_hz/1e9:.0f} GHz, η={eta:.2f})")
        # plt.grid(True)
        # plt.legend()
        # savefig(OUTDIR, f"deltaP_vs_tau0_el{el_ref:.0f}_dels_{'_'.join(str(int(d)) for d in deltas)}deg.png") #subtle flex

        # -----------------------------
        # change in power for given elevation change, of reference elevation, and tau_0
        # -----------------------------

        plt.figure(figsize=(8, 6))
        for tau in taus:

            Teff = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0)

            P_atm_W = K_B * Teff * bandwidth_hz * eta # W
            P_atm_pW = P_atm_W * 1e12 # convert to pW

            R_eval = R_paper(P_atm_W) 

            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
                bandwidth=bandwidth_hz,
                eta=eta,
                dT_del=dT_del,
            )  # W/deg

            plt.plot(x, dP_del_W * 1e12, label=fr"$\tau_0$={tau:.2f}")

        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$dP/d\mathrm{el}$ (pW/deg)")
        plt.title(
            fr"Atmospheric loading slope $dP/d\mathrm{{el}}$ vs Elevation "
            fr"($\Delta\nu$={bandwidth_hz/1e9:.0f} GHz, $\eta$={eta:.2f})"
        )
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, f"atm_loading_slope_dP_del_vs_elevation_eta{eta:.1f}.png")




    # -------------------------------------------------------
    # Open CCAT dat file to check for consistency with atmosphere model assumptions
    # -------------------------------------------------------

        plt.figure(figsize=(8, 6))
        for tau in taus:

            Teff = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0)

            P_atm_W = K_B * Teff * bandwidth_hz * eta # W

            R_eval = R_paper(P_atm_W) 

            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
            bandwidth=bandwidth_hz,
            eta=eta,
            dT_del=dT_del,
            )  # W/deg

            # dP_del_W = inst_power_per_deg(
            #     bandwidth=bandwidth_hz,
            #     eta=eta,
            #     dT_del=dT_del,
            # )  # W/deg

            # del_f_fwhm = Q_r * bandwidth_hz * eta * K_B * R_eval * dT_del  # Hz, frequency change corresponding to power change for given dT/del and responsivity
            del_f_fwhm = Q_r * R_eval * dP_del_W  # Hz, frequency change corresponding to power change for given dT/del and responsivity
            plt.plot(x, del_f_fwhm, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$\frac{d}{d(\mathrm{el})}(\delta f / \mathrm{FWHM})$")
        plt.title(
            fr"Frequency change vs Elevation "
            fr"($\Delta\nu$={bandwidth_hz/1e9:.0f} GHz, $\eta$={eta:.2f})"
        )
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, f"freq_change_vs_elevation_eta{eta:.1f}.png")

    # -------------------------------------------------------
    # Open CCAT dat file to check for consistency with atmosphere model assumptions
    # -------------------------------------------------------

    # ccat_tab = pd.read_csv(CCAT_DATA)

    # print(ccat_tab.head(20))

    # ccat_atm_data = np.loadtxt(CCAT_DATA, comments = "!")

    # nu = ccat_atm_data[:, 0]
    # b = ccat_atm_data[:, 1] 
    # c = ccat_atm_data[:, 2] 

    # freqs = [220, 280, 350, 410, 850]
    # pwvs = np.linspace(0.36, 1.28)

    # plt.figure(figsize=(8, 6))
    # for f in freqs: 
    #     idx = np.argmin(np.abs(nu - f))
    #     b_f = b[idx]
    #     c_f = c[idx]

    #     tau_0 = b_f * pwvs + c_f

    #     plt.plot(pwvs, tau_0, label=fr"${f}$ GHz, CCAT fit $\tau_0 = {b_f:.3f} \cdot \mathrm{{PWV}} + {c_f:.3f}$")

    # plt.xlabel("PWV (mm)")
    # plt.ylabel(r"Zenith opacity $\tau_0$")
    # plt.title("Atmospheric opacity vs PWV")
    # plt.grid(True)
    # plt.legend()
    # savefig(OUTDIR, f"ccat_tau0_vs_pwv.png")

# -------------------------------------------------------
# CCAT tau trends for b and c ccat bands GHz
# -------------------------------------------------------

    # freqs = [220, 280, 350, 410, 850]
    # widths = [56, 60, 35, 30, 97]

    # maria_b = [0.0376, 0.0518, 0.1488, 0.3195, 0.7298]
    # maria_c = [0.0109, 0.0151, 0.0387, 0.0594, 0.2034]

    # ccat_atm_tab = np.loadtxt(CCAT_DATA, comments="!")

    # nu = ccat_atm_tab[:, 0]
    # b = ccat_atm_tab[:, 1]
    # c = ccat_atm_tab[:, 2]

    # for f, w, mb, mc in zip(freqs, widths, maria_b, maria_c):
    #     nu_mask = (nu >= f - w/2) & (nu <= f + w/2)

    #     if not np.any(nu_mask):
    #         print(f"No data found in range {f-w/2} to {f+w/2} GHz for band centered at {f} GHz")
    #         continue

    #     b_masked = b[nu_mask]
    #     c_masked = c[nu_mask]
    #     nu_masked = nu[nu_mask]

    #     # ---- b plot ----
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(nu_masked, b_masked, label="CCAT b coefficient (PWV slope)")

    #     if not np.isnan(mb):
    #         plt.axhline(mb, linestyle="--", color = "red",label=f"Maria b coefficient at {f} GHz")
    #     else:
    #         plt.plot([], [], linestyle="--", color="red", label="Maria b: N/A")  # dummy plot for legend

    #     plt.xlabel("Frequency (GHz)")
    #     plt.ylabel("b coefficient")
    #     plt.title(f"CCAT tau trends for b around {f} GHz")
    #     plt.grid(True)
    #     plt.legend()
    #     savefig(OUTDIR, f"ccat_tau_trends_b_{f}GHz.png")
    #     plt.close()

    #     # ---- c plot ----
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(nu_masked, c_masked, label="CCAT c coefficient (PWV intercept)")

    #     if not np.isnan(mc):
    #         plt.axhline(mc, linestyle="--", color = "red", label=f"Maria c coefficient at {f} GHz")
    #     else:
    #         plt.plot([], [], linestyle="--", color="red", label="Maria c: N/A")  # dummy plot for legend

    #     plt.xlabel("Frequency (GHz)")
    #     plt.ylabel("c coefficient")
    #     plt.title(f"CCAT tau trends for c around {f} GHz")
    #     plt.grid(True)
    #     plt.legend()
    #     savefig(OUTDIR, f"ccat_tau_trends_c_{f}GHz.png")
    #     plt.close()

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

    if not REDUCED_FITS.exists():
        convert_fits_units(RAW_FITS, JYSR_FITS, input_unit="mJy/arcsec^2", output_unit="Jy/sr")
        clip_fits_nans(JYSR_FITS, FILLED_FITS, fill_value=0.0)
        clip_fits_area(FILLED_FITS, REDUCED_FITS, **CUTOUT)
    else:
        print(f"Reduced FITS already exists: {REDUCED_FITS}. Skipping FITS prep steps.")

    # if not RAW_FITS.exists():
    #     raise FileNotFoundError(f"Input FITS file not found: {RAW_FITS}")

    # # FITS prep
    # convert_fits_units(RAW_FITS, JYSR_FITS, input_unit="mJy/arcsec^2", output_unit="Jy/sr")
    # clip_fits_nans(JYSR_FITS, FILLED_FITS, fill_value=0.0)
    # clip_fits_area(FILLED_FITS, REDUCED_FITS, **CUTOUT)



    # Atmosphere plots for "all"
    if do_atm:
        run_atmosphere_plots()
    else:
        print("[skip] atmosphere plots (run_mode='only_sim' or atm_plot=False)")

    # -----------------------------
    # Instrument
    # -----------------------------
    if ccat_band == "220":
        f220 = Band(
            center=220e9,
            width=56e9,
            efficiency= eta,
            NET_CMB=6.8e-6,
            knee=1.0,
            gain_error=5e-2,
        )
        band = f220

    elif ccat_band == "280":
        f280 = Band(
            center=280e9,
            width=60e9,
            efficiency= eta,
            NET_CMB=13e-6,
            knee=1.0,
            gain_error=5e-2,
        )
        band = f280

    elif ccat_band == "350":
        f350 = Band(
            center=350e9,
            width=35e9,
            efficiency= eta,
            NET_CMB=48e-6,
            knee=1.0,
            gain_error=5e-2,
        )
        band = f350

    elif ccat_band == "410":
        f410 = Band(
            center=410e9,
            width=30e9,
            efficiency= eta,
            NET_CMB=182e-6,
            knee=1.0,
            gain_error=5e-2,
        )
        band = f410
    
    elif ccat_band == "850":
        f850 = Band(
            center=850e9,
            width=97e9,
            efficiency= eta,
            NET_CMB=310000e-6,
            knee=1.0,
            gain_error=5e-2,
        )
        band = f850
    
    else:
        raise ValueError(f"Invalid ccat_band: {ccat_band}. Choose from '220', '280', '350', '410', '850'.")
    

    array_cfg = {
        "shape": "hexagon",
        "field_of_view": 1.3,
        "beam_spacing": 8.0,
        "primary_size": 6.0,
        "bands": [band],
        "polarized": Polarized,
    }

    #change beam spacing -> 4 to see if the temperature changes, if it doesnt then there is an assumption abt detectors
    #maybe, 
    #what is assumed telescope efficiency 
    #could run pwv multiples with less detectors for efficiency
    # 

    instrument = maria.get_instrument(array=array_cfg)
    print(instrument)
    instrument.plot()
    savefig(OUTDIR, f"{PREFIX}_instrument_overview.png")

    # -----------------------------
    # Site and Map
    # -----------------------------
    site = maria.get_site("cerro_chajnantor", altitude=5600)

    print(site)
    site.plot()
    savefig(OUTDIR, f"{PREFIX}_site_overview.png")

    input_map = maria.map.load(str(REDUCED_FITS), nu=NU_HZ)

    print(input_map)

    map_jysr = input_map.to("Jy/sr")
    print(map_jysr)
    map_jysr.plot(cmap="coolwarm")
    savefig(OUTDIR, f"{PREFIX}_input_map_JySr_PWV{pwv_mm:.2f}.png")

    # map_pW = input_map.to("watts")
    # print(map_pW)
    # map_pW.plot(cmap="coolwarm")
    # savefig(OUTDIR, f"{PREFIX}_input_map_W_PWV{pwv_mm:.2f}.png")

    # map_K = input_map.to("K_RJ", band=f280)
    # print(map_K)
    # map_K.plot(cmap="coolwarm")
    # savefig(OUTDIR, f"{PREFIX}_input_map_KRJ_PWV{pwv_mm:.2f}.png")

    # map_Kcmb = input_map.to("K_CMB", band=f280)    
    # print(map_Kcmb)
    # map_Kcmb.plot(cmap="coolwarm")
    # savefig(OUTDIR, f"{PREFIX}_input_map_KCMB_PWV{pwv_mm:.2f}.png")


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
    savefig(OUTDIR, f"{PREFIX}_daisy_plan_PWV{pwv_mm:.2f}.png")

    # -----------------------------
    # TOD Simulation
    # -----------------------------
    sim = maria.Simulation(
        instrument=instrument,
        plans=plans,
        site=site,
        atmosphere="2d",
        atmosphere_kwargs={"weather": {"pwv": pwv_mm}},
        # cmb = "generate",
        # cmb_kwargs = {"source": "planck"},
        map=input_map,
    )

    print(sim)
    tods = sim.run()
    print(tods)

    tods[0].to("pW").plot()
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
        savefig(OUTDIR, f"{PREFIX}_tod_plot_chunk{CHUNK_NUMBER}_PWV{pwv_mm:.2f}.png")
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
    savefig(OUTDIR, f"{PREFIX}_tod_pointing_azel_PWV{pwv_mm:.2f}.png")

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
    savefig(OUTDIR, f"{PREFIX}_tod_pointing_radec_PWV{pwv_mm:.2f}.png")

    # # -----------------------------
    # # Atmosphere vs elevation (TOD averaged) ORIGINAL METHOD
    # # -----------------------------



    # el0 = np.nanmean(to_deg_if_rad(tod0.el), axis=0)
    # T_atm0 = np.nanmean(np.asarray(tod0.data["atmosphere"]), axis=0)

    # m = np.isfinite(el0) & np.isfinite(T_atm0) & (T_atm0 > 0.0) #& (T_atm0 < 0.98 * T_0)

    # elv = el0[m]
    # Tv = T_atm0[m]

    # tau0_samples = tau_0_from_atm_temp(
    #     T_atm = Tv,
    #     el_deg = elv,
    #     T_0 = T_0
    # )

    # tau0_ref = np.median(tau0_samples)

    # el_ref = np.nanmedian(elv)
    # T_ref = np.nanmedian(Tv)
    
    # # tau0_est = tau_0_from_atm_temp(
    # #     T_atm=T_ref,
    # #     el_deg=el_ref,
    # #     T_0=T_0
    # # )

    # print(f"Inferred tau_0 from TOD-avg atmosphere: {tau0_ref:.4f} (PWV={pwv_mm:.2f} mm)")

    # dTdel_ref = inst_effective_atm_temp_850GHz(
    #     mode="inst",
    #     tau_0=tau0_ref,
    #     el_deg=el_ref,
    #     T_0=T_0
    # )

    # print(f"Expected dT/del at el={el_ref:.2f} deg: {dTdel_ref:.6f} K/deg")

    # # a, b = np.polyfit(elv, Tv, deg=1)

    # # print(f"Atmosphere vs Elevation fit: T_atm = {a:.4f} * el + {b:.4f} [K]")

    # xfit = np.linspace(elv.min(), elv.max(), 100)
    # yfit = T_ref + dTdel_ref * (xfit - el_ref)

    # step = 10
    # plt.figure(figsize=(8, 6))
    # plt.scatter(elv[::step], Tv[::step], s=1, alpha=0.5, label="TOD samples")
    # plt.plot(xfit, yfit, lw=2, alpha= 0.75, color="red", label=
    #     fr"Model tangent "
    #     fr"($\tau_0={tau0_ref:.4f}$, "
    #     fr"$dT/d\mathrm{{el}}={dTdel_ref:.4f}\,\mathrm{{K/deg}}$)")
    # plt.xlabel("Elevation (deg)")
    # plt.ylabel("Atmospheric Temperature (K)")
    # plt.title(f"Atmospheric Temperature vs Elevation (PWV={pwv_mm:.2f} mm, TOD-avg)")
    # plt.grid(True)
    # plt.legend()
    # savefig(OUTDIR, f"{PREFIX}_atmosphere_vs_el_elref{el_ref:.2f}_tau0_inferred_tangent_PWV{pwv_mm:.2f}.png")

    # -----------------------------
    # Atmosphere vs elevation (TOD averaged) THOMAS METHOD
    # -----------------------------

    tod_true_K_RJ = band.cal("pW -> K_RJ")(tod0.to("pW").signal)

    el0 = np.nanmean(to_deg_if_rad(tod0.el), axis=0)
    T_sig0 = np.nanmean(np.asarray(tod_true_K_RJ), axis=0)

    m = np.isfinite(el0) & np.isfinite(T_sig0)

    elv = el0[m]
    Tv_sig = T_sig0[m]

    tau0_samples = tau_0_from_atm_temp(
        T_atm = Tv_sig,
        el_deg = elv,
        T_0 = T_0
    )

    tau0_ref_new = np.median(tau0_samples)

    el_ref = np.nanmedian(elv)
    T_ref = np.nanmedian(Tv_sig)
    
    # tau0_est = tau_0_from_atm_temp(
    #     T_atm=T_ref,
    #     el_deg=el_ref,
    #     T_0=T_0
    # )

    print(f"Inferred tau_0 from TOD-avg atmosphere: {tau0_ref_new:.4f} (PWV={pwv_mm:.2f} mm)")

    dTdel_ref = inst_effective_atm_temp_850GHz(
        mode="inst",
        tau_0=tau0_ref_new,
        el_deg=el_ref,
        T_0=T_0
    )

    print(f"Expected dT/del at el={el_ref:.2f} deg: {dTdel_ref:.6f} K/deg")

    xfit = np.linspace(elv.min(), elv.max(), 100)
    yfit = T_ref + dTdel_ref * (xfit - el_ref)

    step = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(elv[::step], Tv_sig[::step], s=1, alpha=0.5, label="TOD samples")
    plt.plot(
        xfit,
        yfit,
        lw=2,
        color="red",
        alpha=0.8,
        label=
        fr"Model tangent "
        fr"($\tau_0={tau0_ref_new:.4f}$, "
        fr"$dT/d\mathrm{{el}}={dTdel_ref:.4f}\,\mathrm{{K/deg}}$)"
    )
    plt.xlabel("Elevation (deg)")
    plt.ylabel(r"Signal Temperature $T$ (K$_{RJ}$)")
    plt.title(f"Signal in K_RJ vs Elevation (PWV={pwv_mm:.2f} mm, TOD-avg)")
    plt.grid(True)
    plt.legend()
    savefig(OUTDIR, f"{PREFIX}_signal_KRJ_vs_el_PWV{pwv_mm:.2f}.png")

    # -----------------------------
    # Compute Detector Loading Power in Maria direct from TOD Model
    # -----------------------------


    P_det = tod.to("pW").signal
    P = np.asarray(P_det, dtype=np.float64)

    print("P shape:", P.shape)

    P_mean = np.nanmean(P, axis=1).ravel()
    P_std  = np.nanstd(P, axis=1).ravel()
    P_ptp  = (np.nanmax(P, axis=1) - np.nanmin(P, axis=1)).ravel()

    print("P_mean shape:", P_mean.shape)
    print("P_std shape:", P_std.shape)
    print("P_ptp shape:", P_ptp.shape)

    print("P_mean min/max:", np.nanmin(P_mean), np.nanmax(P_mean))
    print("P_std min/max:", np.nanmin(P_std), np.nanmax(P_std))
    print("P_ptp min/max:", np.nanmin(P_ptp), np.nanmax(P_ptp))

    plt.figure(figsize=(8,6))
    plt.hist(P_mean[np.isfinite(P_mean)], bins=30, alpha=0.7, density=False)
    plt.xlabel("Mean Detector Power (pW)")
    plt.ylabel("Number of Detectors")
    plt.title(f"Distribution of Mean Direct Detector Power (PWV={pwv_mm:.2f} mm $\\eta$={eta:.2f})")
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_detector_direct_power_eta_{eta:.2f}_histogram_PWV{pwv_mm:.2f}.png")

    KRJ_det = tod.to("K_RJ").signal
    K_RJ = np.asarray(KRJ_det, dtype=np.float64)

    print("K_RJ shape:", K_RJ.shape)

    K_RJ_mean = np.nanmean(K_RJ, axis=1).ravel()
    K_RJ_std  = np.nanstd(K_RJ, axis=1).ravel()
    K_RJ_ptp  = (np.nanmax(K_RJ, axis=1) - np.nanmin(K_RJ, axis=1)).ravel()

    print("K_RJ_mean shape:", K_RJ_mean.shape)
    print("K_RJ_std shape:", K_RJ_std.shape)
    print("K_RJ_ptp shape:", K_RJ_ptp.shape)

    print("K_RJ_mean min/max:", np.nanmin(K_RJ_mean), np.nanmax(K_RJ_mean))
    print("K_RJ_std min/max:", np.nanmin(K_RJ_std), np.nanmax(K_RJ_std))
    print("K_RJ_ptp min/max:", np.nanmin(K_RJ_ptp), np.nanmax(K_RJ_ptp))

    plt.figure(figsize=(8,6))
    plt.hist(K_RJ_mean[np.isfinite(K_RJ_mean)], bins=30, alpha=0.7, density=False)
    plt.xlabel("Mean Detector Temperature (K_RJ)")
    plt.ylabel("Number of Detectors")
    plt.title(f"Distribution of Mean Direct Detector Temperature (PWV={pwv_mm:.2f} mm $\\eta$={eta:.2f})")
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_detector_direct_KRJ_eta_{eta:.2f}_histogram_PWV{pwv_mm:.2f}.png")





    # plt.figure(figsize=(8,6))
    # plt.hist(P_std[np.isfinite(P_std)], bins=30, alpha=0.7)
    # plt.xlabel("Std Dev of Detector Power (pW)")
    # plt.ylabel("Number of Detectors")
    # plt.title(f"Distribution of Std Dev of Direct Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    # plt.grid(True)
    # savefig(OUTDIR, f"{PREFIX}_detector_direct_power_eta_{eta:.2f}_std_histogram_PWV{pwv_mm:.2f}.png")

    # plt.figure(figsize=(8,6))
    # plt.hist(P_ptp[np.isfinite(P_ptp)], bins=30, alpha=0.7)
    # plt.xlabel("Peak-to-Peak Detector Power (pW)")
    # plt.ylabel("Number of Detectors")
    # plt.title(f"Distribution of Peak-to-Peak Direct Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    # plt.grid(True)
    # savefig(OUTDIR, f"{PREFIX}_detector_direct_power_eta_{eta:.2f}_ptp_histogram_PWV{pwv_mm:.2f}.png")
    cal_factor = band.cal("pW -> K_RJ")
    print("Calibration object:", cal_factor)
    print(type(cal_factor))
    print(cal_factor.__dict__)

    cal_factor_maria = cal_factor(1.0)  # K_RJ / pW

    print(f"Calibration conversion factor from Maria: {cal_factor_maria:.6e} K/pW")

    # -----------------------------
    # Compare Calibration Conversion Factor from Maria to Simple Estimate
    # -----------------------------

    bandwidth_list = [56e9, 60e9, 35e9, 30e9, 97e9]
    band_labels = ["220 Band", "280 Band", "350 Band", "410 Band", "850 GHz"]

    cal_facs = [1e-12 / (p * eta * bw * K_B) for bw in bandwidth_list]

    # compare Maria to the matching analytic band
    idx_match = 0  # 350 GHz band
    cal_fac_match = cal_facs[idx_match]

    percent_diff = 100.0 * (cal_fac_match - cal_factor_maria) / cal_factor_maria

    band_labels_maria = band_labels + [f"Maria ({NU_GHZ} GHz)"]
    cal_facs_maria = cal_facs + [cal_factor_maria]

    for label, bw, cf in zip(band_labels, bandwidth_list, cal_facs):
        print(f"{label:8s}  bw={bw/1e9:6.1f} GHz   cal_fac={cf:.6e} K/pW")

    print(f"Maria ({NU_GHZ} GHz): cal_fac={cal_factor_maria:.6e} K/pW")
    print(f"Percent difference vs matching analytic band: {percent_diff:.2f}%")

    plt.figure(figsize=(8, 6))
    bars = plt.bar(band_labels_maria, cal_facs_maria, color=["blue"] * 5 + ["red"])

    maria_bar = bars[-1]
    plt.text(
        maria_bar.get_x() + maria_bar.get_width() / 2,
        maria_bar.get_height(),
        f"{percent_diff:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

    plt.ylabel("Calibration Conversion Factor (K/pW)")
    plt.title("Calibration Conversion Factor for CCAT Bands")
    plt.grid(True, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig(OUTDIR, "calibration_conversion_factors.png")
    # -----------------------------
    # Compute Detector Loading Power in Maria Atmosphere Model
    # -----------------------------


    P_det_atm_pW = (K_B * np.asarray(tod0.data["atmosphere"]) * bandwidth_hz * eta) * 1e12  # (N_det, N_time)

    P = np.asarray(P_det_atm_pW, dtype=np.float64)

    P_mean = np.mean(P_det_atm_pW, axis=1)
    P_std = np.std(P_det_atm_pW, axis=1)

    P_ptp = np.nanmax(P_det_atm_pW, axis=1) - np.nanmin(P_det_atm_pW, axis=1)

    # Per-detector mean: cheap, keep as-is
    P_mean_per_det = np.mean(P_det_atm_pW, axis=1)

    plt.figure(figsize=(8,6))
    plt.hist(P_mean_per_det, bins=30, alpha=0.7)
    plt.xlabel("Mean Detector Power (pW)")
    plt.ylabel("Number of Detectors")
    plt.title(f"Distribution of Mean On-Sky Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_detector_on_sky_power_eta_{eta:.2f}_histogram_PWV{pwv_mm:.2f}.png")

    # plt.figure(figsize=(8,6))
    # plt.hist(P_std, bins=30, alpha=0.7)
    # plt.xlabel("Std Dev of Detector Power (pW)")
    # plt.ylabel("Number of Detectors")
    # plt.title(f"Distribution of Std Dev of On-Sky Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    # plt.grid(True)
    # savefig(OUTDIR, f"{PREFIX}_detector_on_sky_power_eta_{eta:.2f}_std_histogram_PWV{pwv_mm:.2f}.png")

    # plt.figure(figsize=(8,6))
    # plt.hist(P_ptp, bins=30, alpha=0.7)
    # plt.xlabel("Peak-to-Peak Detector Power (pW)")
    # plt.ylabel("Number of Detectors")
    # plt.title(f"Distribution of Peak-to-Peak On-Sky Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    # plt.grid(True)
    # savefig(OUTDIR, f"{PREFIX}_detector_on_sky_power_eta_{eta:.2f}_ptp_histogram_PWV{pwv_mm:.2f}.png")


    P_det_pW = (K_B * band.cal("pW -> K_RJ")(tod0.to("pW").signal) * bandwidth_hz * eta) * 1e12  # (N_det, N_time)


    P = np.asarray(P_det_pW, dtype=np.float64)

    P_mean = np.mean(P_det_pW, axis=1)
    P_std = np.std(P_det_pW, axis=1)

    P_ptp = np.nanmax(P_det_pW, axis=1) - np.nanmin(P_det_pW, axis=1)

    # Per-detector mean: cheap, keep as-is
    P_mean_per_det = np.mean(P_det_pW, axis=1)

    plt.figure(figsize=(8,6))
    plt.hist(P_mean_per_det, bins=30, alpha=0.7)
    plt.xlabel("Mean Detector Power (pW)")
    plt.ylabel("Number of Detectors")
    plt.title(f"Distribution of Mean Atmospheric Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_detector_atmospheric_power_eta_{eta:.2f}_histogram_PWV{pwv_mm:.2f}.png")

    # plt.figure(figsize=(8,6))
    # plt.hist(P_std, bins=30, alpha=0.7)
    # plt.xlabel("Std Dev of Detector Power (pW)")
    # plt.ylabel("Number of Detectors")
    # plt.title(f"Distribution of Std Dev of Atmospheric Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    # plt.grid(True)
    # savefig(OUTDIR, f"{PREFIX}_detector_atmospheric_power_eta_{eta:.2f}_std_histogram_PWV{pwv_mm:.2f}.png")

    # plt.figure(figsize=(8,6))
    # plt.hist(P_ptp, bins=30, alpha=0.7)
    # plt.xlabel("Peak-to-Peak Detector Power (pW)")
    # plt.ylabel("Number of Detectors")
    # plt.title(f"Distribution of Peak-to-Peak Atmospheric Detector Power (PWV={pwv_mm:.2f} mm $\eta$={eta:.2f} )")
    # plt.grid(True)
    # savefig(OUTDIR, f"{PREFIX}_detector_atmospheric_power_eta_{eta:.2f}_ptp_histogram_PWV{pwv_mm:.2f}.png")

    # -----------------------------
    # Extracting Tau from transmission
    # -----------------------------
    pwv_list = [0.36, 0.64, 0.92, 1.28]  # mm, from Q1 to Q3 zenith PWV values

    atm_spec = AtmosphericSpectrum(region="chajnantor", altitude=5600)

    elev_deg_list = np.linspace(30, 70, 6)
    elev_rad_list = np.radians(elev_deg_list)

    nu = band.nu.Hz

    for pwv in pwv_list:
        plt.figure(figsize=(8, 6))

        for elev_deg, elev_rad in zip(elev_deg_list, elev_rad_list):
            trans = atm_spec.transmission(
                nu=nu,
                elevation=float(elev_rad),
                pwv=float(pwv),
            )
            plt.plot(nu / 1e9, trans, label=f"El={elev_deg:.0f} deg")

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Atmospheric Transmission")
        plt.title(f"Atmospheric Transmission vs Frequency (PWV={pwv:.2f} mm)")
        plt.grid(True)
        plt.legend()
        savefig(OUTDIR, f"{PREFIX}_atmospheric_transmission_vs_frequency_PWV{pwv:.2f}_{selected_band}GHz.png")
    # -----------------------------
    # Inferring Transmission and Tau0 vs PWV
    # -----------------------------

    pwv_list = np.linspace(0.36, 1.28, 25)  # mm
    elev_deg_list = [30.0, 40.0, 50.0, 60.0, 70.0]
    elev_rad_list = np.radians(elev_deg_list)

    atm_spec = AtmosphericSpectrum(region="chajnantor", altitude=5600)
    nu = band.nu.Hz

    nu0_idx = np.argmin(np.abs(nu - band.center.Hz))
    nu0_ghz = nu[nu0_idx] / 1e9

    plt.figure(figsize=(8, 6))
    for elev_deg, elev_rad in zip(elev_deg_list, elev_rad_list):
        # trans_band_list = []
        trans_center_list = []

        for pwv in pwv_list:
            trans = atm_spec.transmission(nu=nu, elevation=elev_rad, pwv=pwv)
            # trans_band = np.mean(trans)
            # trans_band_list.append(trans_band)

            trans_center = trans[nu0_idx]
            trans_center_list.append(trans_center)


        plt.plot(pwv_list, trans_center_list, marker='o', label=f"{elev_deg:.0f} deg")

    plt.xlabel("PWV (mm)")
    plt.ylabel("Atmospheric Transmission at Band Centre")
    plt.title(f"Transmission vs PWV at Band Centre ({nu0_ghz:.1f} GHz)")
    plt.grid(True)
    plt.legend()
    savefig(OUTDIR, f"{PREFIX}_band_center_transmission_vs_pwv_all_elev.png")


    plt.figure(figsize=(8, 6))
    for elev_deg, elev_rad in zip(elev_deg_list, elev_rad_list):
        # tau0_band_list = []
        tau0_center_list = []

        for pwv in pwv_list:
            trans = atm_spec.transmission(nu=nu, elevation=elev_rad, pwv=pwv)
            tau0 = -np.log(trans) * np.sin(elev_rad)
            # tau0_band = np.mean(tau0)
            # tau0_band_list.append(tau0_band)
            tau0_center = tau0[nu0_idx]
            tau0_center_list.append(tau0_center)

        plt.plot(pwv_list, tau0_center_list, marker='o', label=f"{elev_deg:.0f} deg")

    plt.xlabel("PWV (mm)")
    plt.ylabel("Zenith Optical Depth at Band Centre")
    plt.title(f"Zenith Optical Depth vs PWV at Band Centre ({nu0_ghz:.1f} GHz)")
    plt.grid(True)
    plt.legend()
    savefig(OUTDIR, f"{PREFIX}_band_center_tau0_vs_pwv_all_elev.png")

    # -----------------------------
    # Transmission-derived tau0 vs PWV for current Maria band
    # -----------------------------

    pwv_arr = np.linspace(0.36, 1.28, 25)
    elev_deg_ref = 50.0
    elev_rad_ref = np.radians(elev_deg_ref)

    atm_spec = AtmosphericSpectrum(region="chajnantor", altitude=5600)
    nu = band.nu.Hz
    nu_ghz = nu / 1e9

    # choose either "center" or "bandavg"
    mode_tau = "center"

    tau0_trans_list = []
    trans_rep_list = []

    if mode_tau == "center":
        nu0_idx = np.argmin(np.abs(nu - band.center.Hz))

    for pwv in pwv_arr:
        trans = atm_spec.transmission(nu=nu, elevation=elev_rad_ref, pwv=pwv)

        # line-of-sight -> zenith optical depth
        tau0_nu = -np.log(trans) * np.sin(elev_rad_ref)

        if mode_tau == "center":
            tau0_rep = tau0_nu[nu0_idx]
            trans_rep = trans[nu0_idx]
        elif mode_tau == "bandavg":
            tau0_rep = np.mean(tau0_nu)
            trans_rep = np.mean(trans)
        else:
            raise ValueError("mode_tau must be 'center' or 'bandavg'")

        tau0_trans_list.append(tau0_rep)
        trans_rep_list.append(trans_rep)

    tau0_trans_arr = np.asarray(tau0_trans_list, dtype=float)
    trans_rep_arr = np.asarray(trans_rep_list, dtype=float)

    (B_trans, C_trans), cov_trans = np.polyfit(pwv_arr, tau0_trans_arr, deg=1, cov=True)
    B_trans_err, C_trans_err = np.sqrt(np.diag(cov_trans))

    pwv_fit = np.linspace(pwv_arr.min(), pwv_arr.max(), 200)
    tau0_fit_trans = B_trans * pwv_fit + C_trans

    ccat_table = np.loadtxt(CCAT_DATA, comments="!")
    nu_tab = ccat_table[:, 0]
    b_tab = ccat_table[:, 1]
    c_tab = ccat_table[:, 2]

    if mode_tau == "center":
        idx_ccat = np.argmin(np.abs(nu_tab - band.center.Hz / 1e9))
        B_ccat = b_tab[idx_ccat]
        C_ccat = c_tab[idx_ccat]

    elif mode_tau == "bandavg":
        band_lo = (band.center.Hz - band.width.Hz / 2) / 1e9
        band_hi = (band.center.Hz + band.width.Hz / 2) / 1e9

        m_band = (nu_tab >= band_lo) & (nu_tab <= band_hi)

        B_ccat = np.mean(b_tab[m_band])
        C_ccat = np.mean(c_tab[m_band])

    else:
        raise ValueError("mode_tau must be 'center' or 'bandavg'")

    tau0_fit_ccat = B_ccat * pwv_fit + C_ccat

    tau_fit_tod = 0.0376 * pwv_fit + 0.0109 # from TOD-inferred tau0 vs PWV fit for 220 GHZ

    plt.figure(figsize=(8,6))
    # plt.plot(pwv_arr, tau0_trans_arr, "o", label="Maria transmission-derived $\\tau_0$")
    plt.plot(
        pwv_fit,
        tau0_fit_trans,
        lw=2,
        label=fr"Maria trans fit: $\tau_0 = ({B_trans:.4f}\pm{B_trans_err:.4f})\,\mathrm{{PWV}} + ({C_trans:.4f}\pm{C_trans_err:.4f})$"
    )
    plt.plot(
        pwv_fit,
        tau0_fit_ccat,
        "--",
        lw=2,
        label=fr"CCAT: $\tau_0 = {B_ccat:.4f}\,\mathrm{{PWV}} + {C_ccat:.4f}$"
    )
    plt.plot(
        pwv_fit,
        tau_fit_tod,
        "-.",
        lw=2,
        label=fr"Maria TOD fit: $\tau_0 = {0.0376:.4f}\,\mathrm{{PWV}} + {0.0109:.4f}$"
    )
    plt.xlabel("PWV (mm)")
    plt.ylabel("Zenith Optical Depth $\\tau_0$")
    plt.title(f"Transmission-derived $\\tau_0$ vs PWV for {ccat_band} GHz band")
    plt.grid(True)
    plt.legend(fontsize=9)
    savefig(OUTDIR, f"{PREFIX}_transmission_tau0_vs_pwv_{ccat_band}GHz_{mode_tau}.png")
    # -----------------------------
    # B and C value fit comparison between Maria transmission-derived tau0 and CCAT table
    # -----------------------------

    band_lo = band.center.Hz - band.width.Hz / 2
    band_hi = band.center.Hz + band.width.Hz / 2

    band_mask = (nu >= band_lo) & (nu <= band_hi)

    nu_band = nu[band_mask]
    nu_ghz_band = nu_band / 1e9

    tau_grid = []

    for pwv in pwv_arr:
        trans = atm_spec.transmission(nu=nu, elevation=elev_rad_ref, pwv=pwv)
        tau0_nu = -np.log(trans) * np.sin(elev_rad_ref)
        tau_grid.append(tau0_nu)

    tau0_grid = np.asarray(tau_grid, dtype=float)   # shape (N_pwv, N_nu)
    tau0_grid_band = tau0_grid[:, band_mask]        # shape (N_pwv, N_band)

    Maria_B_list = []
    Maria_C_list = []

    for j in range(len(nu_band)):
        y = tau0_grid_band[:, j]
        x = pwv_arr
        B, C = np.polyfit(x, y, deg=1)
        Maria_B_list.append(B)
        Maria_C_list.append(C)

    B_maria = np.array(Maria_B_list)
    C_maria = np.array(Maria_C_list)

    interp_B = interp1d(nu_tab, b_tab, kind="linear", bounds_error=False, fill_value="extrapolate")
    interp_C = interp1d(nu_tab, c_tab, kind="linear", bounds_error=False, fill_value="extrapolate")

    B_ccat_interp = interp_B(nu_ghz_band)
    C_ccat_interp = interp_C(nu_ghz_band)

    plt.figure(figsize=(8,6))
    plt.plot(nu_ghz_band, B_maria, label="Maria B(ν)", color="tab:green")
    plt.plot(nu_ghz_band, B_ccat_interp, "-", label="CCAT B(ν)", color="tab:blue")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("B coefficient")
    plt.legend()
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_B_coefficient_comparison_{ccat_band}GHz_{mode_tau}.png")

    plt.figure(figsize=(8,6))
    plt.plot(nu_ghz_band, B_maria - B_ccat_interp, label="ΔB = Maria - CCAT", color="tab:red")
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("ΔB")
    plt.legend()
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_B_residual_{ccat_band}GHz_{mode_tau}.png")

    plt.figure(figsize=(8,6))
    plt.plot(nu_ghz_band, C_maria, label="Maria C(ν)", color="tab:green")
    plt.plot(nu_ghz_band, C_ccat_interp, "-", label="CCAT C(ν)", color="tab:blue")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("C coefficient")
    plt.legend()
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_C_coefficient_comparison_{ccat_band}GHz_{mode_tau}.png")

    plt.figure(figsize=(8,6))
    plt.plot(nu_ghz_band, C_maria - C_ccat_interp, label="ΔC = Maria - CCAT", color="tab:red")
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("ΔC")
    plt.legend()
    plt.grid(True)
    savefig(OUTDIR, f"{PREFIX}_C_residual_{ccat_band}GHz_{mode_tau}.png")
    # -----------------------------
    # BinMapper Mapmaking
    # -----------------------------
    map_type_norm = map_type.lower().strip()

    # tod_difference = tods.data["atmosphere"] - tod_true_K_RJ = band.cal("pW -> K_RJ")(tods.to("pW").signal)


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
        savefig(OUTDIR, f"{PREFIX}_output_BinMapper_PWV{pwv_mm:.2f}.png")

    elif map_type_norm in ("none", "skip", "no"):
        print("[skip] mapmaking")

    else:
        raise ValueError(f"Unknown map_type={map_type!r}. Try 'BinMapper' or 'skip'.")

    return tau0_ref_new, el_ref

if __name__ == "__main__":

    band_info = { # GHz, Prime-Cam module specs
    "220": {"center": 220.0, "width": 56.0},
    "280": {"center": 280.0, "width": 60.0},
    "350": {"center": 350.0, "width": 35.0},
    "410": {"center": 410.0, "width": 30.0},
    "850": {"center": 850.0, "width": 97.0},
    }

    band_center = band_info[selected_band]["center"]
    band_width = band_info[selected_band]["width"]
    freq_target = band_center # GHz, for CCAT table lookup

    import time

    starting_time = time.perf_counter()
     
    import multiprocessing as mp
    import gc

    mp.set_start_method("spawn", force = True)

    main(atm_plot=True, map_type="skip", temp_mode="inst", ccat_band = selected_band, run_mode="all", tod_diagnostics=True, pwv_mm=0.36)

    # pwv_list = [0.36, 1.28]
    # for pwv in pwv_list:
    #     print(f"\n=== Running for PWV={pwv:.2f} mm ===")
    #     main(atm_plot=True, run_mode= "all" , temp_mode="inst", ccat_band="850", map_type="skip", tod_diagnostics=True, pwv_mm=pwv)
        
    #main(atm_plot=True, run_mode= "all" , temp_mode="inst", ccat_band="280", map_type="Binmapper", tod_diagnostics=True, pwv_mm=0.36)

    raise SystemExit("Stopping after single run. Uncomment the loop below to run multiple PWV values and compare inferred tau_0.")

# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------

    pwv_list = np.linspace(0.36, 1.28, 5)  # mm, from Q1 to Q3 zenith PWV values

    tau0_list = []
    el_ref_list = []

    for pwv in pwv_list:

        main_start_time = time.perf_counter()

        pwv_mm = pwv
        print(f"\n=== Running for PWV={pwv_mm:.2f} mm ===")

        tau0_ref_new, el_ref = main(
            atm_plot=False,
            run_mode="only_sim",
            temp_mode="inst",
            ccat_band=selected_band,
            map_type="skip",
            tod_diagnostics=True,
            pwv_mm=pwv_mm,
        )
        tau0_list.append(tau0_ref_new)
        el_ref_list.append(el_ref)

        main_end_time = time.perf_counter()
        main_elapsed_time = main_end_time - main_start_time
        print(
            f"Finished run for PWV={pwv_mm:.2f} mm, "
            f"inferred tau_0={tau0_ref_new:.4f} "
            f"(Elapsed time: {main_elapsed_time:.2f} seconds)"
        )

        gc.collect()

    el_refs = np.asarray(el_ref_list, dtype=np.float64)
    print(
        f"Elevation references across runs: "
        f"{el_refs.min():.2f} to {el_refs.max():.2f} deg, "
        f"median={np.median(el_refs):.2f} deg"
    )
    el_ref_name = float(np.median(el_refs))

    pwv_arr = np.asarray(pwv_list, dtype=np.float64)
    tau0_arr = np.asarray(tau0_list, dtype=np.float64)

    m = np.isfinite(pwv_arr) & np.isfinite(tau0_arr)
    pwv_arr = pwv_arr[m]
    tau0_arr = tau0_arr[m]

    # ---------------------------------
    # TOD-derived fit
    # ---------------------------------
    (maria_b_fit, maria_c_fit), cov = np.polyfit(pwv_arr, tau0_arr, deg=1, cov=True)
    maria_b_fit_err, maria_c_fit_err = np.sqrt(np.diag(cov))

    pwv_fit = np.linspace(pwv_arr.min(), pwv_arr.max(), 100)
    tau0_fit_tod = maria_b_fit * pwv_fit + maria_c_fit

    print(
        f"TOD fit parameters: "
        f"b={maria_b_fit:.4f} ± {maria_b_fit_err:.4f}, "
        f"c={maria_c_fit:.4f} ± {maria_c_fit_err:.4f}"
    )

    # ---------------------------------
    # Rebuild the selected Maria band
    # ---------------------------------
    if selected_band == "220":
        band = Band(center=220e9, width=56e9, efficiency=eta, NET_CMB=6.8e-6, knee=1.0, gain_error=5e-2)
    elif selected_band == "280":
        band = Band(center=280e9, width=60e9, efficiency=eta, NET_CMB=13e-6, knee=1.0, gain_error=5e-2)
    elif selected_band == "350":
        band = Band(center=350e9, width=35e9, efficiency=eta, NET_CMB=48e-6, knee=1.0, gain_error=5e-2)
    elif selected_band == "410":
        band = Band(center=410e9, width=30e9, efficiency=eta, NET_CMB=182e-6, knee=1.0, gain_error=5e-2)
    elif selected_band == "850":
        band = Band(center=850e9, width=97e9, efficiency=eta, NET_CMB=310000e-6, knee=1.0, gain_error=5e-2)
    else:
        raise ValueError(f"Invalid selected_band: {selected_band}")

    # ---------------------------------
    # CCAT table
    # ---------------------------------
    ccat_table = np.loadtxt(CCAT_DATA, comments="!")
    nu = ccat_table[:, 0]   # GHz
    b = ccat_table[:, 1]
    c = ccat_table[:, 2]

    nu_mask = (nu >= freq_target - band_width / 2) & (nu <= freq_target + band_width / 2)
    nu_masked = nu[nu_mask]
    b_masked = b[nu_mask]
    c_masked = c[nu_mask]

    idx_freq = np.argmin(np.abs(nu_masked - freq_target))
    ccat_b_center = b_masked[idx_freq]
    ccat_c_center = c_masked[idx_freq]

    tau0_fit_ccat = ccat_b_center * pwv_fit + ccat_c_center

    # ---------------------------------
    # Transmission-derived fit at band centre
    # ---------------------------------
    atm_spec = AtmosphericSpectrum(region="chajnantor", altitude=5600)

    nu_band_hz = band.nu.Hz
    nu_band_ghz = nu_band_hz / 1e9

    band_lo_hz = band.center.Hz - band.width.Hz / 2
    band_hi_hz = band.center.Hz + band.width.Hz / 2

    band_mask = (nu_band_hz >= band_lo_hz) & (nu_band_hz <= band_hi_hz)

    nu_band_hz = nu_band_hz[band_mask]
    nu_band_ghz = nu_band_ghz[band_mask]

    nu0_idx = np.argmin(np.abs(nu_band_hz - band.center.Hz))

    tau0_trans_list = []

    for pwv in pwv_arr:
        trans = atm_spec.transmission(
            nu=nu_band_hz,
            elevation=float(np.radians(el_ref_name)),
            pwv=float(pwv),
        )
        tau0_nu = -np.log(trans) * np.sin(np.radians(el_ref_name))
        tau0_trans_list.append(tau0_nu[nu0_idx])

    tau0_trans_arr = np.asarray(tau0_trans_list, dtype=float)

    (maria_b_trans, maria_c_trans), cov_trans = np.polyfit(
        pwv_arr, tau0_trans_arr, deg=1, cov=True
    )
    maria_b_trans_err, maria_c_trans_err = np.sqrt(np.diag(cov_trans))

    tau0_fit_trans = maria_b_trans * pwv_fit + maria_c_trans

    print(
        f"Transmission fit parameters: "
        f"b={maria_b_trans:.4f} ± {maria_b_trans_err:.4f}, "
        f"c={maria_c_trans:.4f} ± {maria_c_trans_err:.4f}"
    )

    # ---------------------------------
    # Transmission-derived B(nu), C(nu) across full Maria band
    # ---------------------------------
    tau_grid = []
    for pwv in pwv_arr:
        trans = atm_spec.transmission(
            nu=nu_band_hz,
            elevation=float(np.radians(el_ref_name)),
            pwv=float(pwv),
        )
        tau0_nu = -np.log(trans) * np.sin(np.radians(el_ref_name))
        tau_grid.append(tau0_nu)

    tau0_grid = np.asarray(tau_grid, dtype=float)  # shape: (N_pwv, N_nu)

    B_trans_nu = []
    C_trans_nu = []

    for j in range(len(nu_band_hz)):
        y = tau0_grid[:, j]
        x = pwv_arr
        Bj, Cj = np.polyfit(x, y, deg=1)
        B_trans_nu.append(Bj)
        C_trans_nu.append(Cj)

    B_trans_nu = np.asarray(B_trans_nu)
    C_trans_nu = np.asarray(C_trans_nu)

    # interpolate CCAT values onto Maria band grid
    interp_B = interp1d(nu, b, kind="linear", bounds_error=False, fill_value="extrapolate")
    interp_C = interp1d(nu, c, kind="linear", bounds_error=False, fill_value="extrapolate")

    B_ccat_interp = interp_B(nu_band_ghz)
    C_ccat_interp = interp_C(nu_band_ghz)

    # ---------------------------------
    # Final 2x2 figure
    # ---------------------------------
    COLOR_TOD = "tab:orange"
    COLOR_TRANS = "tab:green"
    COLOR_CCAT = "tab:blue"
    COLOR_CENTER = "0.4"

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: tau0 vs PWV
    ax = axs[0, 0]

    ax.plot(
        pwv_arr,
        tau0_arr,
        "o",
        color="black",
        ms=5,
        alpha=0.85,
        label="Maria TOD samples",
    )

    ax.plot(
        pwv_fit,
        tau0_fit_tod,
        color=COLOR_TOD,
        lw=2.2,
        linestyle="--",
        label=(
            fr"Maria TOD fit: $\tau_0 = "
            fr"({maria_b_fit:.4f}\pm{maria_b_fit_err:.4f})\,\mathrm{{PWV}} + "
            fr"({maria_c_fit:.4f}\pm{maria_c_fit_err:.4f})$"
        ),
    )

    ax.plot(
        pwv_fit,
        tau0_fit_trans,
        color=COLOR_TRANS,
        lw=2.2,
        linestyle="-",
        label=(
            fr"Maria trans fit: $\tau_0 = "
            fr"({maria_b_trans:.4f}\pm{maria_b_trans_err:.4f})\,\mathrm{{PWV}} + "
            fr"({maria_c_trans:.4f}\pm{maria_c_trans_err:.4f})$"
        ),
    )

    ax.plot(
        pwv_fit,
        tau0_fit_ccat,
        color=COLOR_CCAT,
        lw=2.2,
        linestyle="-",
        label=fr"CCAT table: $\tau_0 = {ccat_b_center:.4f}\,\mathrm{{PWV}} + {ccat_c_center:.4f}$",
    )

    ax.set_xlabel("PWV (mm)")
    ax.set_ylabel(r"Zenith optical depth $\tau_0$")
    ax.set_title(f"(a) $\\tau_0$ vs PWV at {band_center:.0f} GHz")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Top-right: info box
    ax = axs[0, 1]
    ax.axis("off")

    info_text = (
        f"Band summary\n\n"
        f"Center frequency: {band_center:.0f} GHz\n"
        f"Bandwidth: {band_width:.0f} GHz\n"
        f"PWV range: {pwv_arr.min():.2f} to {pwv_arr.max():.2f} mm\n"
        f"Median reference elevation: {el_ref_name:.2f} deg\n\n"
        f"Maria TOD fit:\n"
        f"b = {maria_b_fit:.4f} ± {maria_b_fit_err:.4f}\n"
        f"c = {maria_c_fit:.4f} ± {maria_c_fit_err:.4f}\n\n"
        f"Maria transmission fit:\n"
        f"b = {maria_b_trans:.4f} ± {maria_b_trans_err:.4f}\n"
        f"c = {maria_c_trans:.4f} ± {maria_c_trans_err:.4f}\n\n"
        f"Nearest CCAT table values:\n"
        f"b = {ccat_b_center:.4f}\n"
        f"c = {ccat_c_center:.4f}"
    )

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Bottom-left: B(nu)
    ax = axs[1, 0]

    ax.plot(
        nu_band_ghz, 
        B_ccat_interp,
        color=COLOR_CCAT,
        lw=2.0,
        linestyle="-",
        label="CCAT tabulated $B(\\nu)$",
    )

    ax.plot(
        nu_band_ghz,
        B_trans_nu,
        color=COLOR_TRANS,
        lw=2.0,
        linestyle="-",
        label="Maria transmission $B(\\nu)$",
    )

    ax.axhspan(
        maria_b_fit - maria_b_fit_err,
        maria_b_fit + maria_b_fit_err,
        color=COLOR_TOD,
        alpha=0.22,
        label="Maria TOD $B$ fit ±1σ",
    )

    ax.axhline(
        maria_b_fit,
        color=COLOR_TOD,
        lw=2.0,
        linestyle="--",
        label="Maria TOD $B$ fit",
    )

    ax.axvline(
        band_center,
        color=COLOR_CENTER,
        lw=1.6,
        linestyle=":",
        label=f"Band centre: {band_center:.0f} GHz",
    )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("B coefficient")
    ax.set_title(f"(b) $B(\\nu)$ across the {band_center:.0f} GHz band")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Bottom-right: C(nu)
    ax = axs[1, 1]

    ax.plot(
        nu_band_ghz,
        C_ccat_interp,
        color=COLOR_CCAT,
        lw=2.0,
        linestyle="-",
        label="CCAT tabulated $C(\\nu)$",
    )

    ax.plot(
        nu_band_ghz,
        C_trans_nu,
        color=COLOR_TRANS,
        lw=2.0,
        linestyle="-",
        label="Maria transmission $C(\\nu)$",
    )

    ax.axhspan(
        maria_c_fit - maria_c_fit_err,
        maria_c_fit + maria_c_fit_err,
        color=COLOR_TOD,
        alpha=0.22,
        label="Maria TOD $C$ fit ±1σ",
    )

    ax.axhline(
        maria_c_fit,
        color=COLOR_TOD,
        lw=2.0,
        linestyle="--",
        label="Maria TOD $C$ fit",
    )

    ax.axvline(
        band_center,
        color=COLOR_CENTER,
        lw=1.6,
        linestyle=":",
        label=f"Band centre: {band_center:.0f} GHz",
    )

    c_upper = max(np.nanmax(C_ccat_interp), np.nanmax(C_trans_nu), maria_c_fit + maria_c_fit_err)
    if c_upper > 1.0:
        ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, 1.05 * c_upper)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("C coefficient")
    ax.set_title(f"(c) $C(\\nu)$ across the {band_center:.0f} GHz band")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    savefig(
        OUTDIR,
        f"{PREFIX}_tau0_vs_PWV_and_CCAT_coefficients_PWV{pwv_arr.min():.2f}-{pwv_arr.max():.2f}_elref{el_ref_name:.2f}deg.png",
    )
    plt.close()



    # plt.figure(figsize=(8, 6))

    # plt.plot(pwv_arr, tau0_arr, "o", label="Inferred $\\tau_0$ from TOD")

    # pwv_fit = np.linspace(pwv_arr.min(), pwv_arr.max(), 100)
    # tau0_fit = maria_b_fit * pwv_fit + maria_c_fit
    # plt.plot(pwv_fit, tau0_fit,lw=2, ls = "-", label=f"Linear Fit: $\\tau_0$ = {maria_b_fit:.4f} * PWV + {maria_c_fit:.4f}")

    # plt.plot(pwv_arr, tau0_ccat, label=fr"${NU_GHZ:.1f}$ GHz, CCAT fit $\tau_0 = {b_f:.3f} \cdot \mathrm{{PWV}} + {c_f:.3f}$")

    # plt.xlabel("PWV (mm)")
    # plt.ylabel("Inferred $\\tau_0$")
    # plt.title(f"Inferred $\\tau_0$ vs PWV, {NU_GHZ} GHz")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(OUTDIR / f"inferred_tau0_vs_PWV_N{len(pwv_arr)}_elmedian_{el_ref_name:.2f}.png", dpi=200, bbox_inches="tight")
    # plt.close()

    ending_time = time.perf_counter()
    elapsed_time = ending_time - starting_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    

    raise SystemExit("Stopping After Main Execution Pipeline.")