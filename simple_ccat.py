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


import os, sys

# for multiple runs of maria in the same session

os.environ["OMP_NUM_THREADS"] = "1"  # Avoid multithreading issues in numpy/scipy
os.environ["MKL_NUM_THREADS"] = "1" # Avoid multithreading issues in numpy/scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1" # Avoid multithreading issues in numpy/scipy
os.environ["NUMEXPR_NUM_THREADS"] = "1" # Avoid multithreading issues in numpy/scipy



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



from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D


# -----------------------------
# Fits // Directory Parameters
# -----------------------------

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

# -----------------------------
#  Simulation Parameters 
# -----------------------------

NU_HZ = 280e9  # 280 GHz
NU_GHZ = NU_HZ / 1e9

PWV_MM = 0.36  #  mm, precip water vapour

# 0.36, 0.67, & 1.28 are Q1, Q2, and Q3 zenith PMV values for Chajnantor

EL_LIMITS = (10, 85)  # degrees

T_0 = 278.868 #K, atmospheric ground temp


START_TIME = "2022-02-10T17:00:00"

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



# ----------------------------------------------
# ---------- Main Execution Pipeline -----------
# ----------------------------------------------




def main(
    run_mode: str = "only_atm",
    atm_plot: bool = True,
    temp_mode: str = "inst",
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
            a = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x)
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
            a = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x)
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
            a = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x)
            plt.plot(x, 1- (y / a), label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$1-\frac{dT/d\mathrm{el}}{T_\mathrm{eff}}$")
        plt.title(r"Fractional Diff Atmospheric Temperature Derivative vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "frac_diff_dT_del_Teff_tau0_0p05_to_0p25.png")

        # -----------------------------
        # Power change per degree: dP/del
        # -----------------------------
        bandwidth_hz = 40e9  # 40 GHz bandwidth for 280 GHz band
        eta =  1.0 # Assume perfect optical efficiency for this estimate

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
        savefig(OUTDIR, "inst_dP_del_tau0_0p05_to_0p25.png")

        # -----------------------------
        # Responsivity fit R(P): fit in W, but derived from pW points
        # -----------------------------
        from scipy.optimize import curve_fit

        R_data = np.array(
            [-2.375, -1.92, -1.81, -1.75, -1.68, -1.61, -1.575, -1.535, -1.505, -1.5],
            dtype=float,
        ) * 1e8  # responsivity

        P_data_pW = np.array([0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5], dtype=float)

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

        f_res = 800e6  # Resonant frequency in Hz (800 MHz)

        # -----------------------------
        # frequency slope vs elevation from power + responsivity fit
        # -----------------------------

        plt.figure(figsize=(8, 6))
        for tau in taus:

            Teff = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0)

            P_atm_pW = K_B * Teff * bandwidth_hz * eta # W
            P_atm_pW *= 1e12 # convert to pW

            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
                bandwidth=bandwidth_hz,
                eta=eta,
                dT_del=dT_del,
            )  # W/deg

            R_eval = R_piecewise(P_atm_pW)  # pW

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

        Del_f = 2200 # Hz, 1/10th of the FWHM is the estimated linear regime limit for 280GHz MKID array

        plt.figure(figsize=(8, 6))
        for tau in taus:
            
            Teff = effective_atm_temp_850GHz(pwv=pwv_mm, tau_0=float(tau), el_deg=x, T_0=T_0)

            P_atm_pW = K_B * Teff * bandwidth_hz * eta # W
            P_atm_pW *= 1e12 # convert to pW

            dT_del = inst_effective_atm_temp_850GHz(
                mode=temp_mode, tau_0=float(tau), el_deg=x, T_0=T_0
            )  # K/deg

            dP_del_W = inst_power_per_deg(
                bandwidth=bandwidth_hz,
                eta=eta,
                dT_del=dT_del,
            )  # W/deg

            R_eval = R_piecewise(P_atm_pW)  # pW

            df_del_Hz_per_deg = f_res * R_eval * dP_del_W  # Hz/deg

            del_el_deg = 2200 / df_del_Hz_per_deg  # deg, elevation change corresponding to 2200 Hz frequency shift

            plt.plot(x, del_el_deg, label=fr"$\tau_0$={tau:.2f}")
        
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r" $\Delta\mathrm{el}$ for $|\Delta f|<2200$ Hz (deg)")
        plt.title(r"Estimated elevation change allowed before leaving linear regime")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, f"allowed_del_el_for_linear_regime_eta{eta:.1f}.png")


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

    print(site)
    site.plot()
    savefig(OUTDIR, f"{PREFIX}_site_overview.png")

    input_map = maria.map.load(str(REDUCED_FITS), nu=NU_HZ)

    print(input_map)
    input_map.to("Jy/sr").plot(cmap="coolwarm")
    savefig(OUTDIR, f"{PREFIX}_input_map_JySr_PWV{pwv_mm:.2f}.png")

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
    
    # tau0_est = tau_0_from_atm_temp(
    #     T_atm=T_ref,
    #     el_deg=el_ref,
    #     T_0=T_0
    # )

    print(f"Inferred tau_0 from TOD-avg atmosphere: {tau0_ref:.4f} (PWV={pwv_mm:.2f} mm)")

    dTdel_ref = inst_effective_atm_temp_850GHz(
        mode="inst",
        tau_0=tau0_ref,
        el_deg=el_ref,
        T_0=T_0
    )

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
    plt.title(f"Atmospheric Temperature vs Elevation (PWV={pwv_mm:.2f} mm, TOD-avg)")
    plt.grid(True)
    plt.legend()
    savefig(OUTDIR, f"{PREFIX}_atmosphere_vs_el_elref{el_ref:.2f}_tau0_inferred_tangent_PWV{pwv_mm:.2f}.png")
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
        savefig(OUTDIR, f"{PREFIX}_output_BinMapper_PWV{pwv_mm:.2f}.png")

    elif map_type_norm in ("none", "skip", "no"):
        print("[skip] mapmaking")

    else:
        raise ValueError(f"Unknown map_type={map_type!r}. Try 'BinMapper' or 'skip'.")

    return tau0_ref, el_ref

if __name__ == "__main__":

    import time

    starting_time = time.perf_counter()
     
    import multiprocessing as mp
    import gc

    mp.set_start_method("spawn", force = True)


    main(atm_plot=True, map_type="skip", tod_diagnostics=False, temp_mode="inst", run_mode="only_atm", pwv_mm=0.36)
    # main(atm_plot=True, run_mode= "all" , temp_mode="inst", map_type="Binmapper", tod_diagnostics=True)

    raise SystemExit("Stopping after single run. Uncomment the loop below to run multiple PWV values and compare inferred tau_0.")

    pwv_list = np.linspace(0.36, 1.28, 5) # mm, from Q1 to Q3 zenith PMV values

    tau0_list = []

    for pwv in pwv_list:

        main_start_time = time.perf_counter()

        pwv_mm = pwv
        print(f"\n=== Running for PWV={pwv_mm:.2f} mm ===")

        tau0_ref, el_ref = main(atm_plot=False, run_mode= "only_sim" , temp_mode="inst", map_type="skip", tod_diagnostics=True, pwv_mm=pwv_mm)
        tau0_list.append(tau0_ref)

        main_end_time = time.perf_counter()
        main_elapsed_time = main_end_time - main_start_time
        print(f"Finished run for PWV={pwv_mm:.2f} mm, inferred tau_0={tau0_ref:.4f} (Elapsed time: {main_elapsed_time:.2f} seconds)")

        gc.collect() # Clean up memory after each run
    
    el_refs = np.asarray(el_ref, dtype=np.float64)
    print(f"Elevation references across runs: {el_refs.min():.2f} to {el_refs.max():.2f} deg, median={np.median(el_refs):.2f} deg")
    el_ref_name = float(np.median(el_refs))


    pwv_arr = np.asarray(pwv_list, dtype=np.float64)
    tau0_arr = np.asarray(tau0_list, dtype=np.float64)

    m = np.isfinite(pwv_arr) & np.isfinite(tau0_arr)
    pwv_arr = pwv_arr[m]
    tau0_arr = tau0_arr[m]

    A, B = np.polyfit(pwv_arr, tau0_arr, deg=1)
    print(f"Linear fit: tau_0 = {A:.4f} * PWV + {B:.4f}")

    plt.figure(figsize=(8, 6))

    plt.plot(pwv_arr, tau0_arr, "o", label="Inferred $\\tau_0$ from TOD")

    pwv_fit = np.linspace(pwv_arr.min(), pwv_arr.max(), 100)
    tau0_fit = A * pwv_fit + B
    plt.plot(pwv_fit, tau0_fit,lw=2, ls = "-", label=f"Linear Fit: $\\tau_0$ = {A:.4f} * PWV + {B:.4f}")

    plt.xlabel("PWV (mm)")
    plt.ylabel("Inferred $\\tau_0$")
    plt.title("Inferred $\\tau_0$ vs PWV")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"inferred_tau0_vs_PWV_N{len(pwv_arr)}_elmedian_{el_ref_name:.2f}.png", dpi=200, bbox_inches="tight")
    plt.close()

    ending_time = time.perf_counter()
    elapsed_time = ending_time - starting_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    

    raise SystemExit("Stopping After Main Execution Pipeline.")