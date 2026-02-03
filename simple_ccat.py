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

OUTDIR = Path("outputs/OrionA_ccat_test_outputs")

DATA_DIR = Path("data")

RAW_FITS = DATA_DIR / "OrionA_20170726_850_DR3_ext_HK.fits"
JYSR_FITS = DATA_DIR / "OrionA_20170726_850_DR3_ext_HK_JySr.fits"
FILLED_FITS = DATA_DIR / "OrionA_20170726_850_DR3_ext_HK_JySr_nan_filled.fits"
REDUCED_FITS = DATA_DIR / "OrionA_20170726_850_DR3_ext_HK_JySr_reduced_filled.fits"

CUTOUT = dict(ra_min=83.2, ra_max=84.0, dec_min=-6.0, dec_max=-4.8)

NU_HZ = 280e9  # 280 GHz
NU_GHZ = NU_HZ / 1e9

PWV_MM = 0.38  # mm

EL_LIMITS = (30, 80)  # degrees

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


def effective_atm_temp_850GHz(*, pwv: float, el_deg: float | np.ndarray, T_0: float = 300.0) -> np.ndarray:
    """Effective atmospheric temperature Teff = T0(1-exp(-tau(el))) using JCMT tau0(PWV)."""
    tau_0 = 0.179 * (pwv + 0.337)
    z_rad = np.deg2rad(90.0 - np.asarray(el_deg))
    tau_z = tau_0 / np.cos(z_rad)
    return T_0 * (1.0 - np.exp(-tau_z))


def inst_effective_atm_temp_850GHz(
    *,
    mode: str,
    tau_0: float | None = None,
    pwv: float | None = None,
    el_deg: float | np.ndarray,
    delta_el: float = 1.0,
    T_0: float = 300.0,
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
        # Teff = T0(1-exp(-tau0/cos z))
        # dTeff/dz = T0 * exp(-tauz) * (tau0 * sin z / cos^2 z)
        dT_dz_rad = T_0 * np.exp(-tau_z) * (tau_0 * np.sin(z_rad) / (np.cos(z_rad) ** 2))
        # elevation increases -> z decreases, but we want dT/d(el). since z = 90 - el:
        # dT/d(el) = - dT/dz. Here magnitude is what's usually plotted; keep sign consistent:
        dT_del_rad = -dT_dz_rad
        return dT_del_rad * (np.pi / 180.0)  # K per degree

    if mode == "fin_diff":
        # central difference in degrees
        teff_plus = effective_atm_temp_850GHz(pwv=pwv if pwv is not None else 0.0, el_deg=el + 0.5 * delta_el, T_0=T_0)
        teff_minus = effective_atm_temp_850GHz(pwv=pwv if pwv is not None else 0.0, el_deg=el - 0.5 * delta_el, T_0=T_0)
        return (teff_plus - teff_minus) / delta_el

    raise ValueError("Invalid mode. Choose 'inst' or 'fin_diff'.")

# ---------- Main Execution Pipeline -----------

def main(atm_plot: bool = True, map_type="BinMapper", tod_diagnostics=True) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FITS.exists():
        raise FileNotFoundError(f"Input FITS file not found: {RAW_FITS}")

    convert_fits_units(RAW_FITS, JYSR_FITS, input_unit="mJy/arcsec^2", output_unit="Jy/sr")
    clip_fits_nans(JYSR_FITS, FILLED_FITS, fill_value=0.0)
    clip_fits_area(FILLED_FITS, REDUCED_FITS, **CUTOUT)

    # atmosphere derivation plots

    if atm_plot:
        x = np.linspace(EL_LIMITS[0], EL_LIMITS[1], 150)
        taus = np.linspace(0.05, 0.90, 18)

        plt.figure(figsize=(8, 6))
        for tau in taus:
            y = inst_effective_atm_temp_850GHz(mode="inst", tau_0=float(tau), el_deg=x)
            plt.plot(x, y, label=fr"$\tau_0$={tau:.2f}")
        plt.xlabel("Elevation (deg)")
        plt.ylabel(r"$dT/d\mathrm{el}$ (K/deg)")
        plt.title(r"Instantaneous $dT/d\mathrm{el}$ vs Elevation for varying $\tau_0$")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        savefig(OUTDIR, "inst_dT_del_tau0_0p05_to_0p90.png")
    else:
        print("[skip] atmosphere derivative plot")
    
# ---------- Instrument -----------
    f280 = Band(
        center=290e9,
        width=40e9,
        NET_CMB=13e-6,
        knee=1.0,
        gain_error=5e-2
    )

    array_cfg= {
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
    savefig(OUTDIR, "instrument_overview.png")

    # ---------- Site and Map -----------

    site = maria.get_site("cerro_chajnantor", altitude=5600)
    input_map = maria.map.load(str(REDUCED_FITS), nu=NU_HZ)

    print(input_map)
    input_map.to("Jy/sr").plot(cmap="coolwarm")
    savefig(OUTDIR, "OrionA_input_map_JySr.png")

    # ---------- Scan Planning -----------

    planner = Planner(
        target = input_map,
        site = site,
        constraints={"el": EL_LIMITS}
    )
    plans = planner.generate_plans(
        total_duration=SIM_DURATION_S,
        max_chunk_duration=SIM_DURATION_S,
        scan_pattern=SCAN_PATTERN,
        sample_rate=SAMPLE_RATE_HZ,
        scan_options={"radius": input_map.width.deg / 3.0}  # degrees
    )

    plans[0].plot()
    print(plans)
    savefig(OUTDIR, "OrionA_daisy_plan.png")

    # --------- TOD Simulation -----------

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
    tods=sim.run()

    print(tods)
    tods[0].plot()
    plt.savefig(os.path.join(OUTDIR, f"OrionA_tod_plot_{CHUNK_NUMBER}.png"),dpi=200,bbox_inches="tight")
    plt.close("all")


    # ---------- TOD Diagnostics -----------

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
        savefig(OUTDIR, f"OrionA_tod_plot_chunk{CHUNK_NUMBER}.png")
    else:
        print("[skip] tod diagnostics")

    # --- Pointing plots (az/el and ra/dec) ---
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
    savefig(OUTDIR, "OrionA_tod_pointing_azel.png")

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
    savefig(OUTDIR, "OrionA_tod_pointing_radec.png")

    # --- Atmosphere vs elevation (TOD averaged) ---
    el0 = np.nanmean(to_deg_if_rad(tod0.el), axis=0)
    T_atm0 = np.nanmean(np.asarray(tod0.data["atmosphere"]), axis=0)

    step = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(el0[::step], T_atm0[::step], s=1, alpha=0.5)
    plt.xlabel("Elevation (deg)")
    plt.ylabel("Atmospheric Temperature (K)")
    plt.title(f"Atmospheric Temperature vs Elevation (PWV={PWV_MM:.2f} mm, TOD-avg)")
    plt.grid(True)
    savefig(OUTDIR, f"atmosphere_vs_el_pwv{PWV_MM:.2f}mm.png")

    # --- BinMapper Mapmaking ---

    map_type = map_type.upper()

    # Only BinMapper implemented here for simplicity
    # I will add MaximumLikelihoodMapper later

    if map_type == "BINMAPPER":
        mapper = BinMapper(
            center=input_map.center,
            frame="ra/dec",
            width=input_map.width,
            height=input_map.height,
            resolution=input_map.width / 64,
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
        savefig(OUTDIR, "OrionA_output_BinMapper.png")

    elif map_type in ("NONE", "SKIP", "NO"):
        print("[skip] mapmaking")

    else:
        raise ValueError(f"Unknown map_type={map_type!r}. Try 'BinMapper' or 'skip'.")

if __name__ == "__main__":
    # main(atm_plot=False, map_type="skip", tod_diagnostics=False)
    main(atm_plot=True, map_type="BinMapper", tod_diagnostics=True)

raise SystemExit("Stopping After Main Execution Pipeline.")

outdir_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/maria_outputs"

outdir_pro="/Users/zaparniukn/Documents/maria/maria_outputs"

blank_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/blank_ccat_test_outputs"

blank_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/blank_ccat_test_outputs"

orionA_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/OrionA_ccat_test_outputs"

orionA_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/OrionA_ccat_test_outputs"

serpens_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/Serpens_ccat_test_outputs"

serpens_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/Serpens_ccat_test_outputs"

outdir = serpens_ccat_test_outputs_pers

os.makedirs(outdir, exist_ok=True)

# Physical constants (SI)
C = 299792458.0                 # m/s
K_B = 1.380649e-23              # J/K
H = 6.62607015e-34              # J*s
T_CMB = 2.7255                  # K
JY = 1e-26                      # W m^-2 Hz^-1


Arcsec2_to_Sr = (1/206265)**2
Factor = 1e-3 / Arcsec2_to_Sr  # mJy/arcsec^2 to Jy/sr

# IN = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK.fits"
# OUT = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr.fits"


# IN = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr.fits"
# OUT = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr_nan_filled.fits"


# IN = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr.fits"
# OUT = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr_reduced_filled.fits"

# IN = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/data/SerpensE_20170726_850_DR3_ext_HK.fits"
# OUT = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/data/SerpensE_20170726_850_DR3_ext_HK_JySr_reduced_filled.fits"

def convert_fits_units(
        IN: str,
        OUT: str,
        input_unit: str,
        output_unit: str,
):
    """
    Convert FITS file data units from input_unit to output_unit.

    Parameters
    ----------
    IN : str
        Input FITS filename.
    OUT : str
        Output FITS filename.
    input_unit : str
        Input data unit (e.g., 'mJy/arcsec^2').
    output_unit : str
        Output data unit (e.g., 'Jy/sr').
    """

    unit_conversions = {
        ('mJy/arcsec^2', 'Jy/sr'): (1e-3 / (1/206265)**2),
        # Add more conversions as needed
    }

    if (input_unit, output_unit) not in unit_conversions:
        raise ValueError(f"Conversion from {input_unit} to {output_unit} not defined.")

    factor = unit_conversions[(input_unit, output_unit)]

    with fits.open(IN) as hdul:
        hdu_idx = next(i for i,h in enumerate(hdul) if h.data is not None)
        data = hdul[hdu_idx].data.astype(np.float64)

        hdul[hdu_idx].data = data * factor
        hdul[hdu_idx].header['BUNIT'] = output_unit
        hdul[hdu_idx].header.add_history(f"Converted from {input_unit} to {output_unit}")
        hdul[hdu_idx].header.add_history(f"Factor used: {factor:.6e} {output_unit} per ({input_unit})")

        hdul.writeto(OUT, overwrite=True)

    print("Wrote:", OUT)


def clip_fits_area(
        IN: str,
        OUT: str,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
):
    """
    Clip a rectangular area from a FITS file and save to a new FITS file.

    Parameters
    ----------
    IN : str
        Input FITS filename.
    OUT : str
        Output FITS filename.
    ra_min : float
        Minimum right ascension in degrees.
    ra_max : float
        Maximum right ascension in degrees.
    dec_min : float
        Minimum declination in degrees.
    dec_max : float
        Maximum declination in degrees.
    """

    ra_c = (ra_min + ra_max) / 2
    dec_c = (dec_min + dec_max) / 2

    width_deg = (ra_max - ra_min) * np.cos(np.deg2rad(dec_c))
    height_deg = dec_max - dec_min

    with fits.open(IN) as hdul:
        # pick first image HDU with data
        hdu_idx = next(i for i,h in enumerate(hdul) if h.data is not None)
        data = hdul[hdu_idx].data
        hdr  = hdul[hdu_idx].header
        wcs  = WCS(hdr)

        if data.ndim == 3 and data.shape[0] == 1:
            data2d = data[0]
        else:
            data2d = data

        center = SkyCoord(ra_c*u.deg, dec_c*u.deg, frame="icrs")
        size   = u.Quantity((height_deg, width_deg), u.deg)  # (ny, nx)

        cutout = Cutout2D(data2d, position=center, size=size, wcs=wcs, mode="partial", fill_value=np.nan)

        cut = cutout.data.astype(np.float32)

        # Write new FITS
        new_hdr = cutout.wcs.to_header()
        # Preserve/ensure units
        if "BUNIT" in hdr:
            new_hdr["BUNIT"] = hdr["BUNIT"]
        else:
            new_hdr["BUNIT"] = "Jy/sr"

        fits.PrimaryHDU(data=cut, header=new_hdr).writeto(OUT, overwrite=True)

    print("Wrote:", OUT)
    print("Patch center (deg):", ra_c, dec_c)
    print("Patch size (deg):", height_deg, width_deg)

def clip_fits_nans(
        IN: str,
        OUT: str,
):
    """
    Fill NaN values in a FITS file and save to a new FITS file.

    Parameters
    ----------
    IN : str
        Input FITS filename.
    OUT : str
        Output FITS filename.
    """

    with fits.open(IN) as hdul:
        # pick first image HDU with data
        hdu_idx = next(i for i,h in enumerate(hdul) if h.data is not None)
        data = hdul[hdu_idx].data
        hdr  = hdul[hdu_idx].header
        wcs  = WCS(hdr)

        if data.ndim == 3 and data.shape[0] == 1:
            data2d = data[0]
        else:
            data2d = data

        # Fill NaNs (choose 0.0 for "black" unobserved regions)
        data_filled = np.nan_to_num(data2d, nan=0.0, posinf=0.0, neginf=0.0)

        fits.PrimaryHDU(data=data_filled, header=hdr).writeto(OUT, overwrite=True)

    print("Wrote:", OUT)

# IN = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr.fits"
# OUT = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr_nan_filled.fits"



IN = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/data/OrionA_20170726_850_DR3_ext_HK.fits"
OUT = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/data/OrionA_20170726_850_DR3_ext_HK_JySr.fits"

convert_fits_units(IN, OUT, input_unit="mJy/arcsec^2", output_unit="Jy/sr")

IN = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/data/OrionA_20170726_850_DR3_ext_HK_JySr.fits"
OUT = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/data/OrionA_20170726_850_DR3_ext_HK_JySr_nan_filled.fits"

clip_fits_nans(IN, OUT)

IN = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/data/OrionA_20170726_850_DR3_ext_HK_JySr_nan_filled.fits"
OUT = "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/data/OrionA_20170726_850_DR3_ext_HK_JySr_reduced_filled.fits"

clip_fits_area(IN, OUT,
               ra_min=83.2, ra_max=84.0,
               dec_min=-6.0, dec_max=-4.8)

# raise SystemExit("Stopping CCAT-prime Example Execution After FITS Prep.")

def dB_dT(nu_GHz: float, T: float = T_CMB) -> float:
    """
    Plank-law derivative dB/dT evaluated at inputted temperature
    T and frequency nu (in Ghz)

    Returns:
    dB/dT: float 
        Units: W m^-2 Hz^-1 sr^-1 K^-1
    """
    nu = nu_GHz * 1e9
    x = H * nu / (K_B * T)
    ex = np.exp(x)
    return (2.0 * K_B * nu**2 / C**2) * (x**2 * ex) / (ex - 1.0)**2

def beam_solid_angle_gaussian(fwhm_arcsec: float) -> float:
    """
    Gaussian beam solid angle Omega_beam ≈ 1.133 * theta_FWHM^2.

    Parameters
    ----------
    fwhm_arcsec : float
        Beam FWHM in arcseconds.

    Returns
    -------
    Omega_beam : float
        Beam solid angle in steradians.
    """
    theta_rad = fwhm_arcsec * (np.pi / (180.0 * 3600.0))
    return (np.pi/(4*np.log(2))) * theta_rad**2

def convert_noise_equivalent(
        initial: str,
        final: str,
        nu_GHz: float,
        value: float,
        beam_fwhm_arcsec: float = None,
        N_det: int = None,
        T: float = T_CMB,
) -> float:
    """
    convert noise equivalent values between different units.
    
    Definitions Expected for initial and final:
        -NEI: Noise Equivalent Intensity Jy sr^-1 rt(s) [W m^-2 Hz^-1 sr^-1 rt(s)]
        -NEFD: Noise Equivalent Flux Density Jy beam^-1 rt(s) [W m^-2 Hz^-1 rt(s)]
        -NET: Noise Equivalent Temperature K rt(s)
        - add NET_RJ conversion caculate RJ then multiply by the factor 
        - 
    
    Parameters:
    start, end : str
        'NEI', 'NET', 'NEFD' (case-insensitive)
    nu_GHz : float
        Frequency in GHz.
    value : float
        Value in the units of `start`.
    beam_arcsec : float, optional
        Beam FWHM in arcsec. Required for any conversion involving NEFD.
    Ndet : float, optional
        Total number of detectors. Required for NEI <-> NEFD if your NEI is *array-combined*
        (as in the top table notes).
    yield_frac : float
        Active detector yield (default 0.8).
    pol_mode : str
        'broadband' (Nbeams=Ndet/2) or 'eor_spec' (Nbeams=Ndet).
    T : float
        Temperature for dB/dT (default T_CMB).
    matched_filter : bool
        If True, use Omega_eff = 2*Omega_beam (Gaussian matched-filter convention).
        If False, use Omega_eff = Omega_beam ("Jy per beam" convention).

    Returns
    -------
    float
        Converted value in units of `end`.
    """

    initial = initial.upper()
    final = final.upper()

    if initial == final:
        return value
    
    nu = nu_GHz * 1e9  # Hz
    dBdT_val = dB_dT(nu_GHz, T=T)  # W m^-2 Hz^-1 sr^-1 K^-1

    if initial == "NEI" and final == "NET":
        NEI = value * JY  # W m^-2 Hz^-1 sr^-1 sqrt(s)
        NET = NEI / dBdT_val
        return NET
    
    elif initial == "NEI" and final == "NEFD":
        if beam_fwhm_arcsec is None:
            raise ValueError("beam_fwhm_arcsec must be provided for NEI <-> NEFD conversion.")
        if N_det is None:
            raise ValueError("N_det must be provided for array-combined NEI <-> NEFD conversion.")
        NEI = value
        omega_beam = beam_solid_angle_gaussian(beam_fwhm_arcsec)  # sr
        NEFD = NEI * omega_beam * np.sqrt(N_det)
        return NEFD
    
    elif initial == "NET" and final == "NEI":
        NET = value
        NEI = NET * dBdT_val / JY  # Jy sr^-1 sqrt(s)
        return NEI
    
    elif initial == "NEFD" and final == "NEI":
        if beam_fwhm_arcsec is None:
            raise ValueError("beam_fwhm_arcsec must be provided for NEI <-> NEFD conversion.")
        if N_det is None:
            raise ValueError("N_det must be provided for array-combined NEI <-> NEFD conversion.")
        NEFD = value
        omega_beam = beam_solid_angle_gaussian(beam_fwhm_arcsec)  # sr
        NEI = NEFD / (omega_beam * np.sqrt(N_det))
        return NEI
    
    elif initial == "NET" and final == "NEFD":
        if beam_fwhm_arcsec is None:
            raise ValueError("beam_fwhm_arcsec must be provided for NET <-> NEFD conversion.")
        if N_det is None:
            raise ValueError("N_det must be provided for array-combined NET <-> NEFD conversion.")
        NET = value
        NEI = NET * dBdT_val
        omega_beam = beam_solid_angle_gaussian(beam_fwhm_arcsec)  # sr
        NEFD = NEI * omega_beam * np.sqrt(N_det) 
        return NEFD
    
    elif initial == "NEFD" and final == "NET":
        if beam_fwhm_arcsec is None:
            raise ValueError("beam_fwhm_arcsec must be provided for NET <-> NEFD conversion.")
        if N_det is None:
            raise ValueError("N_det must be provided for array-combined NET <-> NEFD conversion.")
        NEFD = value
        omega_beam = beam_solid_angle_gaussian(beam_fwhm_arcsec)  # sr
        NEI = NEFD / (omega_beam * np.sqrt(N_det))
        NET = NEI / dBdT_val
        return NET
    
    else:
        raise ValueError("Invalid conversion types. Options: 'NEI', 'NET', 'NEFD'.")
    

def effective_atm_temp_850GHz(pwv: float, el_deg: float, T_0: float = 300.0) -> float:

    # put a tau input value for calculations
    """
    Estimate effective atmospheric temperature given PWV and elevation.

    Parameters
    ----------
    pwv : float
        Precipitable water vapor in mm.
    T_atm : float
        Physical temperature of the atmosphere in K (default 300 K).
    elevation_deg : float
        Telescope elevation angle in degrees.

    Returns
    -------
    T_eff : float
        Effective atmospheric temperature in K.
    """
    # Zenith optical depth (JCMT/SCUBA-2 relation)
    tau_0 = 0.179 * (pwv + 0.337)

    z = 90.0 - el_deg  # zenith angle [deg]

    tau_z = tau_0 / np.cos(np.deg2rad(z))

    T_eff = T_0 * (1.0 - np.exp(-tau_z))

    return T_eff


def inst_effective_atm_temp_850GHz(mode: str, tau_0: float = None, pwv: float = None, el_deg: float = None, delta_el: float = None, T_0: float = 300.0) -> float:

    # put a tau input value for calculations
    """
    Estimate effective atmospheric temperature given PWV and elevation.

    Parameters
    ----------
    pwv : float
        Precipitable water vapor in mm.
    T_atm : float
        Physical temperature of the atmosphere in K (default 300 K).
    elevation_deg : float
        Telescope elevation angle in degrees.

    Returns
    -------
    T_eff : float
        Effective atmospheric temperature in K.
    """
    # Zenith optical depth (JCMT/SCUBA-2 relation)
    if tau_0 is None and pwv is not None:

        tau_0 = 0.179 * (pwv + 0.337)

    elif tau_0 is not None:

        tau_0 = tau_0

    else:

        raise ValueError("Either tau_0 or pwv must be provided.")

    z_rad = np.deg2rad(90.0 - el_deg)  # zenith angle [rad]

    tau_z = tau_0 / np.cos(z_rad)

    if mode == "inst":

        dT_dz = T_0 * tau_0 * np.exp(-tau_z) * np.sin(z_rad) / (np.cos(z_rad)**2)
        result = dT_dz * np.pi / 180.0 

    elif mode == "fin_diff":
        delta_el = 1.0  # degree
        T_eff_plus = effective_atm_temp_850GHz(pwv, el_deg + delta_el/2, T_0)
        T_eff_minus = effective_atm_temp_850GHz(pwv, el_deg - delta_el/2, T_0)
        dT_dz = (T_eff_plus - T_eff_minus) / (delta_el)
        result = dT_dz 
    
    else:
        raise ValueError("Invalid mode. Choose 'inst' or 'fin_diff'.")
    return result  # convert to per degree


x = np.linspace(30, 80, 150) #degrees

taus = np.linspace(0.05, 0.90, 18) #from 5% to 25% zenith opacity in steps of 5%

plt.figure(figsize=(8,6))

for tau in taus:
    y = inst_effective_atm_temp_850GHz(mode="inst", tau_0=tau, el_deg=x, pwv=None)#, delta_el=1.0)
    plt.plot(x, y, label=f"$\\tau_0$={tau:.2f}")

plt.xlabel("Elevation (degrees)")
plt.ylabel(r"$dT/d\mathrm{el}$ (K/deg)")   # you're returning per deg of elevation
plt.title(r"Instantaneous $dT/d\mathrm{el}$ vs Elevation for varying $\tau_0$")
plt.grid(True)
plt.legend(ncol=2, fontsize=9)

outfile = os.path.join(outdir, "inst_dT_del_tau0_0p05_to_0p90.png")
plt.savefig(outfile, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", outfile)

# raise SystemExit("Stopping CCAT-prime Example Execution Before Instrument Definition.")

# "/Users/zaparniukn/Documents/maria/"

# temp1 = effective_atm_temp_850GHz(pwv=0.38, el_deg=57.5)

# temp2 = effective_atm_temp_850GHz(pwv=0.38, el_deg=62.5)

# difference1 = np.abs(temp2 - temp1)

# temp3 = effective_atm_temp_850GHz(pwv=0.38, el_deg=27.5)

# temp4 = effective_atm_temp_850GHz(pwv=0.38, el_deg=32.5)

# difference2 = np.abs(temp4 - temp3)

# temp5 = inst_effective_atm_temp_850GHz(mode = "inst", pwv=0.38, el_deg=59.5)
# temp6 = inst_effective_atm_temp_850GHz(mode = "inst", pwv=0.38, el_deg=60.5)


# difference3 = np.abs(temp6 - temp5)


# print("Effective Atmospheric Temperature at 850 GHz:")
# print(f"At 57.5 degrees elevation: {temp1:.2f} K")
# print(f"At 62.5 degrees elevation: {temp2:.2f} K")
# print(f"Difference: {difference1:.2f} K")

# print(f"At 27.5 degrees elevation: {temp3:.2f} K")
# print(f"At 32.5 degrees elevation: {temp4:.2f} K")
# print(f"Difference: {difference2:.2f} K")

# print(f"Inst. dT/dz at 59.5 degrees elevation: {temp5:.4f} K/deg")
# print(f"Inst. dT/dz at 60.5 degrees elevation: {temp6:.4f} K/deg")
# print(f"Difference: {difference3:.4f} K/deg")


# NEFD220 = convert_noise_equivalent(
#     initial="NEI", final="NEFD",
#     nu_GHz=220,
#     value=3300.0,  # Jy sr^-1 sqrt(s)
#     beam_fwhm_arcsec= 59.0,
#     N_det=7938
# )

# NET220 = convert_noise_equivalent(
#     initial="NEI", final="NET",
#     nu_GHz=220,
#     value= 3300,  # Jy beam^-1 sqrt(s)
#     beam_fwhm_arcsec= 59.0,
#     N_det=7938
# )

# NEFD410 = convert_noise_equivalent(
#     initial="NEI", final="NEFD",
#     nu_GHz=410,
#     value=37300.0,  # Jy sr^-1 sqrt(s)
#     beam_fwhm_arcsec= 59.0,
#     N_det=20808
# )

# NET410 = convert_noise_equivalent(
#     initial="NEI", final="NET",
#     nu_GHz=410,
#     value= 37300,  # Jy beam^-1 sqrt(s)
#     beam_fwhm_arcsec= 32.0,
#     N_det=20808
# )

# print("NEFD at 220 GHz (mJy beam^-1 sqrt(s)):", NEFD220*1e3 )
# print("NET at 220 GHz (uK sqrt(s)):", NET220*1e6 )

# print("NEFD at 410 GHz (mJy beam^-1 sqrt(s)):", NEFD410*1e3 )
# print("NET at 410 GHz (uK sqrt(s)):", NET410*1e6 )
 


f280= Band(
    center=280e9, #Hz
    width=56e9, #Hz
    NET_CMB=13e-6, #K*sqrt(s) 
    knee=1e0, #Hz
    gain_error=5e-2
)

# f350 = Band(
#     center=350e9, #Hz
#     width=50e9, #Hz
#     NET_CMB=48e-6, #K*sqrt(s) 
#     knee=1e0, #Hz
#     gain_error=5e-2
# )

#Define the array configuration, specifying the 
#detectors distribution on the focal plane

array = { #"n": 500, 
        "shape": "hexagon",
        "field_of_view": 1.3, #degrees...
         "beam_spacing": 2.0,
         "primary_size": 6, #in meters...
         "bands": [f280], #, f220, f350, f410, f850],
        #  "packing": "triangular",
         "polarized": False,
        #  "offsets": None
        }
# array350 = { #"n": 2000, ``
#         "shape": "hexagon",
#         "field_of_view": 1.3, #degrees...
#          "beam_spacing": 3.3,
#          "primary_size": 6, #in meters...
#          "bands": [f350], #, f220, f350, f410, f850],
#         #  "packing": "triangular",
#          "polarized": False,
#         #  "offsets": None
#         }

# subarray_280 = {"name": "mod_280", "bands": [f280], "focal_plane_offset": (-1.0*1.8, 0.0), **array250}
# subarray_350 = {"name": "mod_350", "bands": [f350], "focal_plane_offset": (-0.5*1.8, -0.866*1.6), **array350}


instrument = maria.get_instrument(array= array)

print(instrument)
instrument.plot()
plt.savefig(os.path.join(outdir, "simple_ccat_test_instrument_plot_280GHz.png"), dpi=200, bbox_inches="tight")
plt.close("all")

# raise SystemExit("Stopping CCAT-prime Example Execution Before Site Definition.")

site = maria.get_site("cerro_chajnantor", altitude=5600)

# print(site)
# site.plot()
# plt.savefig("ccat_test_site_plot.png",dpi=200,bbox_inches="tight")
# plt.close("all")

# input_map = maria.map.load(fetch("maps/cluster2.fits"),
#                           nu=280e9)

input_map = maria.map.load("/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/data/OrionA_20170726_850_DR3_ext_HK_JySr_reduced_filled.fits",
                          nu=280e9,
)

# input_map.data[:] = 0.0

# input_map[..., 256: -256, 256: -256].to("K_RJ").plot(cmap="cmb")
# print(input_map)
# plt.savefig(os.path.join(outdir, "input_map_blank.png"),dpi=200,bbox_inches="tight")
# plt.close("all")


# map_filename= maria.io.fetch("maps/cluster1.fits") 

# input_map = maria.map.load(
#     filename=map_filename,
#     nu=280e9, #what is this used for?
#     center=(291.156, -31.23)) #wcs..?
# input_map.data *= 50e1 #why

print(input_map)
input_map.to("Jy/sr").plot(cmap = "coolwarm")
plt.savefig(os.path.join(outdir, "OrionA_850_reduced_input_map_280GHz.png"),dpi=200,bbox_inches="tight")
plt.close("all")



planner = Planner(target=input_map,
                  site=site,
                  constraints={"el": (30, 80)})


plans = planner.generate_plans(total_duration=900,
                               max_chunk_duration=900,
                               scan_pattern="daisy",
                               sample_rate=15,
                               scan_options={"radius": input_map.width.deg / 3})

plans[0].plot()
print(plans)
plt.savefig(os.path.join(outdir, "OrionA_850_daisy_reduced_1.png"),dpi=200,bbox_inches="tight")
plt.close("all")

# planner = Planner(start_time="2024-08-06T03:00:00",
#                   target=input_map,
#                   site=site,
#                   constraints = {"el": (60,85)}) # telescope elevation limits

# plans = planner.generate_plans(total_duration = 1200, #seconds
#                                 max_chunk_duration = 600, #seconds
#                                scan_pattern = "daisy",
#                                scan_options = {"radius": input_map.width.deg / 3},
#                                sample_rate = 10 #Hz
#                                )

# print(plans)
# plans[0].plot()
# plt.savefig("simple_ccat_plan_plot.png",dpi=200,bbox_inches="tight")
# plt.close("all")

# raise SystemExit("Stopping CCAT-prime Example Execution Before Simulation.")

sim = maria.Simulation(
    instrument=instrument,
    plans=plans,
    site=site,
    atmosphere = "2d",
    atmosphere_kwargs = {"weather":{"pwv":1.44}},
    cmb = "generate", #in mm
    map = input_map)

print(sim)

tods = sim.run()

def chunk_summary(tod, i):
    el = tod.el

    if np.nanmax(np.abs(el)) < 10:
        el = np.rad2deg(el)
    el0 = np.nanmean(el, axis= 0)
    airmass = 1/np.sin(np.deg2rad(el0))
    print(f"TOD {i}: el median={np.nanmedian(el0):.2f} deg, "
          f"airmass median={np.nanmedian(airmass):.2f}, "
          f"el range=({np.nanpercentile(el0,5):.1f},{np.nanpercentile(el0,95):.1f})")


print(tods)

tod = tods[0]   # define this explicitly

for k in tod.data.keys():
    arr = np.asarray(tod.data[k])   # force numpy
    print(
        f"TOD data key: {k}, "
        f"shape: {arr.shape}, "
        f"dtype: {arr.dtype}, "
        f"min={arr.min(): .3e}, "
        f"max={arr.max(): .3e}, "
        f"std={arr.std(): .3e}"
    )

print(tod)

chunk_summary(tods[0], 0)
# chunk_summary(tods[1], 1)


# raise SystemExit("Stopping CCAT-prime Example Execution After Simulation.")



print(tods)
tods[0].plot()
plt.savefig(os.path.join(outdir, "OrionA_850_ccat_tod_plot_reduced_1.png"),dpi=200,bbox_inches="tight")
plt.close("all")

# tods[1].plot()
# plt.savefig(os.path.join(outdir, "OrionA_850_ccat_tod_plot_reduced2_1.png"),dpi=200,bbox_inches="tight")
# plt.close("all")

# tods[-1].plot()
# plt.savefig(os.path.join(outdir, "OrionA_850_ccat_tod_plot_reduced_last_1.png"),dpi=200,bbox_inches="tight")
# plt.close("all")


maria.undebug()

input_map.center

# print(input_map.center)

# from maria.mappers import MaximumLikelihoodMapper

# ml_mapper = MaximumLikelihoodMapper(tods=tods,
#                                     width=0.75 * input_map.width.deg,
#                                     height=0.75 * input_map.height.deg,
#                                     resolution=10 * input_map.resolution.deg,
#                                     units="mK_RJ")
# print(f"{ml_mapper.loss() = }")

# print(ml_mapper.map)
# ml_mapper.map.plot(cmap="cmb")
# plt.savefig(os.path.join(outdir, "simple_ccat_ml_output_map5.png"),dpi=200,bbox_inches="tight")
# plt.close("all")

# ml_mapper.fit(epochs=4, steps_per_epoch=32, lr=2e-1)
# ml_mapper.map.plot(cmap="cmb")
# plt.savefig(os.path.join(outdir, "simple_ccat_ml_output_map_fitted5.png"),dpi=200,bbox_inches="tight")
# plt.close("all")

# raise SystemExit("Stopping CCAT-prime Example Execution Before Binning.")


tod = tods[0]

t = tod.time
tsec = t- t[0]

az = tod.az
el = tod.el

if np.nanmax(np.abs(az)) < 10:
    az = np.rad2deg(az)
if np.nanmax(np.abs(el)) < 10:
    el = np.rad2deg(el)
az0 = np.nanmean(az, axis=0)

el0 = np.nanmean(el, axis=0)

plt.figure(figsize=(8,6))
plt.plot(tsec, el0, label="el (deg)")
plt.plot(tsec, az0, label="az (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Degrees")
plt.title("Telescope Pointing vs Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(outdir, "OrionA_850_ccat_tod_pointing_reduced_1.png"),dpi=200,bbox_inches="tight")
plt.close("all")


ra = tod.ra
dec = tod.dec

if np.nanmax(np.abs(ra)) < 10:
    ra = np.rad2deg(ra)
if np.nanmax(np.abs(dec)) < 10:
    dec = np.rad2deg(dec)

ra0 = np.nanmean(ra, axis=0)
dec0 = np.nanmean(dec, axis=0)

plt.figure(figsize=(8,6))
plt.plot(tsec, dec0, label="dec (deg)")
plt.plot(tsec, ra0, label="ra (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Degrees")   
plt.title("Telescope RA/Dec vs Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(outdir, "OrionA_850_ccat_tod_radec_reduced_1.png"),dpi=200,bbox_inches="tight")
plt.close("all")


from maria.mappers import BinMapper

mapper = BinMapper(
    center=input_map.center,
    frame="ra/dec",
    width=input_map.width,
    height=input_map.height,
    resolution=input_map.width / 64,
    tod_preprocessing={
        # "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
        "remove_spline": {"knot_spacing": 60, "remove_el_gradient": True},
        "remove_modes": {"modes_to_remove": 1},
    },
    map_postprocessing={
        "gaussian_filter": {"sigma": 1},
        # "median_filter": {"size": 1},
    },
    units="mK_RJ",
    tods=tods,
)

mapper.add_tods(tods)

output_map = mapper.run()

output_map.plot(nu_index= [0], cmap="coolwarm")
plt.savefig(os.path.join(outdir, "OrionA_850_ccat_output_BinMapper_map_2d_reduced_280_1.png"),dpi=200,bbox_inches="tight")
plt.close("all")


tod = tods[0]

el = tod.el
if np.nanmax(np.abs(el)) < 10:
    el = np.rad2deg(el)
el0 = np.nanmean(el, axis= 0) # (time)

#atmosphere

T_atm = np.asarray(tod.data["atmosphere"]) # (det, time)
T_atm0 = np.nanmean(T_atm, axis=0)  # (time)

step = 10
plt.figure(figsize=(8,6))
plt.scatter(el0[::step], T_atm0[::step], s=1, alpha=0.5)
plt.xlabel("Elevation (deg)")
plt.ylabel("Atmospheric Temperature (K)")
plt.title("Atmospheric Temperature vs Elevation $PWV = 1.44$ mm (TOD, Detector-Averaged)")
plt.grid(True)
plt.savefig(os.path.join(outdir, "OrionA_850_ccat_tod_atmosphere_vs_el_pwv144_1.png"),dpi=200,bbox_inches="tight")
plt.close("all")
print("Saved:", os.path.join(outdir, "OrionA_850_ccat_tod_atmosphere_vs_el_pwv144_1.png"))