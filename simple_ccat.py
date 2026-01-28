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

import numpy as np


outdir_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/maria_outputs"

outdir_pro="/Users/zaparniukn/Documents/maria/maria_outputs"

blank_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/blank_ccat_test_outputs"

blank_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/blank_ccat_test_outputs"

orionA_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/OrionA_ccat_test_outputs"

orionA_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/OrionA_ccat_test_outputs"

serpens_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/Serpens_ccat_test_outputs"

serpens_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/Serpens_ccat_test_outputs"

outdir = serpens_ccat_test_outputs_pro


# Physical constants (SI)
C = 299792458.0                 # m/s
K_B = 1.380649e-23              # J/K
H = 6.62607015e-34              # J*s
T_CMB = 2.7255                  # K
JY = 1e-26                      # W m^-2 Hz^-1


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

# x = np.linspace(59.5, 60.5, 150) #degrees
# y = inst_effective_atm_temp_850GHz(mode="fin_diff", pwv=0.38, el_deg=x, delta_el=1.0)

# plt.figure(figsize=(8,6))
# plt.plot(x, y)
# plt.xlabel("Elevation (degrees)")
# plt.ylabel("Fin. Diff. dT/dz (K/deg)")
# plt.title("Finite Difference dT/dz at 850 GHz vs Elevation")
# plt.grid(True)
# plt.show()
# plt.savefig(os.path.join("/Users/zaparniukn/Documents/maria/", "fin_diff_dT_dz_850GHz_vs_elevation.png"), dpi=200, bbox_inches="tight")
# plt.close()


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
 
# IN = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK.fits"
# OUT = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr.fits"

# raise SystemExit("Stopping CCAT-prime Example Execution Before Unit Conversion.")


IN = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK.fits"
OUT = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr.fits"


Arcsec2_to_Sr = (1/206265)**2
Factor = 1e-3 / Arcsec2_to_Sr  # mJy/arcsec^2 to Jy/sr

with fits.open(IN) as hdul:

    hdu_idx = next(i for i,h in enumerate(hdul) if h.data is not None)
    data=hdul[hdu_idx].data.astype(np.float64)

    hdul[hdu_idx].data = data * Factor
    hdul[hdu_idx].header['BUNIT'] = 'Jy/sr'
    hdul[hdu_idx].header.add_history("Converted from mJy/arcsec^2 to Jy/sr")
    hdul[hdu_idx].header.add_history(f"Factor used: {Factor:.6e} Jy/sr per (mJy/arcsec^2)")

    hdul.writeto(OUT, overwrite=True)

print("Wrote:", OUT)

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




IN = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr.fits"
OUT = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr_reduced_filled.fits"

ra_min, ra_max = 279.25, 279.75
dec_min, dec_max = -2.0, -1.0

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

    # If your map has an extra leading axis (e.g. (1,ny,nx)), drop it
    if data.ndim == 3 and data.shape[0] == 1:
        data2d = data[0]
    else:
        data2d = data

    center = SkyCoord(ra_c*u.deg, dec_c*u.deg, frame="icrs")
    size   = u.Quantity((height_deg, width_deg), u.deg)  # (ny, nx)

    cutout = Cutout2D(data2d, position=center, size=size, wcs=wcs, mode="partial", fill_value=np.nan)

    cut = cutout.data.astype(np.float32)

    # Fill NaNs (choose 0.0 for "black" unobserved regions)
    cut = np.nan_to_num(cut, nan=0.0, posinf=0.0, neginf=0.0)

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


IN = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr.fits"
OUT = "/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr_nan_filled.fits"

# ra_min, ra_max = 83.25, 84.0
# dec_min, dec_max = -6.0, -5.0

# ra_c = (ra_min + ra_max) / 2
# dec_c = (dec_min + dec_max) / 2

# width_deg = (ra_max - ra_min) * np.cos(np.deg2rad(dec_c))
# height_deg = dec_max - dec_min

with fits.open(IN) as hdul:
    # pick first image HDU with data
    hdu_idx = next(i for i,h in enumerate(hdul) if h.data is not None)
    data = hdul[hdu_idx].data
    hdr  = hdul[hdu_idx].header
    wcs  = WCS(hdr)

    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    else:
        data = data
    
    # Fill NaNs (choose 0.0 for "black" unobserved regions)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    fits.PrimaryHDU(data=data, header=hdr).writeto(OUT, overwrite=True)

print("Wrote:", OUT)

# --define outdirs for different laptops


#hello

f280= Band(
    center=280e9, #Hz
    width=40e9, #Hz
    NET_RJ=30e-6, #K*sqrt(s) 
    knee=1e0, #Hz
    gain_error=5e-2
)

#Define the array configuration, specifying the 
#detectors distribution on the focal plane

array = { #"n": 500, 
        "shape": "hexagon",
        "field_of_view": 1.3, #degrees...
         "beam_spacing": 2.3,
         "primary_size": 6, #in meters...
         "bands": [f280], #, f220, f350, f410, f850],
        #  "packing": "triangular",
         "polarized": False,
        #  "offsets": None
        }

instrument = maria.get_instrument(array=array)

print(instrument)
instrument.plot()
plt.savefig(os.path.join(outdir, "simple_ccat_test_instrument_plot_280GHz.png"), dpi=200, bbox_inches="tight")
plt.close("all")

site = maria.get_site("cerro_chajnantor", altitude=5600)

# print(site)
# site.plot()
# plt.savefig("ccat_test_site_plot.png",dpi=200,bbox_inches="tight")
# plt.close("all")

# input_map = maria.map.load(fetch("maps/cluster2.fits"),
#                           nu=280e9)

input_map = maria.map.load("/Users/zaparniukn/Documents/data/SerpensE_20170724_850_DR3_ext_HK_JySr_reduced_filled.fits",
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
input_map.to("Jy/sr").plot()
plt.savefig(os.path.join(outdir, "input_map_.png"),dpi=200,bbox_inches="tight")
plt.close("all")


planner = Planner(target=input_map,
                  site=site,
                  constraints={"el": (30, 80)})


plans = planner.generate_plans(total_duration=1800,
                               max_chunk_duration=900,
                               scan_pattern="lissajous",
                               sample_rate=5,
                               scan_options={"radius": input_map.width.deg / 3})

plans[0].plot()
print(plans)
plt.savefig(os.path.join(outdir, "SerpensE_850_lissajous__reduced_full.png"),dpi=200,bbox_inches="tight")
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
    atmosphere_kwargs = {"weather":{"pwv":0.67}},
    map = input_map)

print(sim)

tods = sim.run()

print(tods)
tods[0].plot()
plt.savefig(os.path.join(outdir, "SerpensE_850_ccat_tod_plot_lissajous__reduced_full.png"),dpi=200,bbox_inches="tight")
plt.close("all")


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

import matplotlib.pyplot as plt 

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
plt.savefig(os.path.join(outdir, "SerpensE_850_ccat_tod_pointing_lissajous__reduced_full.png"),dpi=200,bbox_inches="tight")
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
plt.savefig(os.path.join(outdir, "SerpensE_850_ccat_tod_radec_lissajous__reduced_full_scaled.png"),dpi=200,bbox_inches="tight")
plt.close("all")


from maria.mappers import BinMapper

mapper = BinMapper(
    center=input_map.center,
    frame="ra/dec",
    width=input_map.width,
    height=input_map.height,
    resolution=input_map.width / 128,
    tod_preprocessing={
        "remove_spline": {"knot_spacing": 60, "remove_el_gradient": True},
        "remove_modes": {"modes_to_remove": 1},
    },
    map_postprocessing={
        "gaussian_filter": {"sigma": 1},
    },
    units="mK_RJ",
    tods=tods,
)

mapper.add_tods(tods)

output_map = mapper.run()

output_map.plot(nu_index= 0)
plt.savefig(os.path.join(outdir, "SerpensE_850_ccat_output_lissajousBinMapper_map_2d__reduced_full_scaled.png"),dpi=200,bbox_inches="tight")
plt.close("all")