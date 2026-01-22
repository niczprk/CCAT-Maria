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

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

IN = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK.fits"
OUT = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr.fits"

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

# IN = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr.fits"
# OUT = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr_reduced_filled.fits"

# ra_min, ra_max = 83.25, 84.0
# dec_min, dec_max = -6.0, -5.0

# ra_c = (ra_min + ra_max) / 2
# dec_c = (dec_min + dec_max) / 2

# width_deg = (ra_max - ra_min) * np.cos(np.deg2rad(dec_c))
# height_deg = dec_max - dec_min

# with fits.open(IN) as hdul:
#     # pick first image HDU with data
#     hdu_idx = next(i for i,h in enumerate(hdul) if h.data is not None)
#     data = hdul[hdu_idx].data
#     hdr  = hdul[hdu_idx].header
#     wcs  = WCS(hdr)

#     # If your map has an extra leading axis (e.g. (1,ny,nx)), drop it
#     if data.ndim == 3 and data.shape[0] == 1:
#         data2d = data[0]
#     else:
#         data2d = data

#     center = SkyCoord(ra_c*u.deg, dec_c*u.deg, frame="icrs")
#     size   = u.Quantity((height_deg, width_deg), u.deg)  # (ny, nx)

#     cutout = Cutout2D(data2d, position=center, size=size, wcs=wcs, mode="partial", fill_value=np.nan)

#     cut = cutout.data.astype(np.float32)

#     # Fill NaNs (choose 0.0 for "black" unobserved regions)
#     cut = np.nan_to_num(cut, nan=0.0, posinf=0.0, neginf=0.0)

#     # Write new FITS
#     new_hdr = cutout.wcs.to_header()
#     # Preserve/ensure units
#     if "BUNIT" in hdr:
#         new_hdr["BUNIT"] = hdr["BUNIT"]
#     else:
#         new_hdr["BUNIT"] = "Jy/sr"

#     fits.PrimaryHDU(data=cut, header=new_hdr).writeto(OUT, overwrite=True)

# print("Wrote:", OUT)
# print("Patch center (deg):", ra_c, dec_c)
# print("Patch size (deg):", height_deg, width_deg)    



IN = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr.fits"
OUT = "/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr_nan_filled.fits"

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

outdir_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/maria_outputs"

outdir_pro="/Users/zaparniukn/Documents/maria/maria_outputs"

blank_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/blank_ccat_test_outputs"

blank_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/blank_ccat_test_outputs"

orionA_ccat_test_outputs_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/OrionA_ccat_test_outputs"

orionA_ccat_test_outputs_pro="/Users/zaparniukn/Documents/maria/OrionA_ccat_test_outputs"

outdir = orionA_ccat_test_outputs_pro

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

array = { #"n": 500, #unknown number of detectors?
        "shape": "hexagon",
        "field_of_view": 1.3, #degrees...
         "beam_spacing": 2.5, #not sure about this one
         "primary_size": 6, #in meters...
         "bands": [f280], #, f220, f350, f410, f850],
        #  "packing": "triangular",
         "polarized": False,
        #  "array_name": "CCAT-prime",
        #  "offsets": None
        }

instrument = maria.get_instrument(array=array)

# print(instrument)
# instrument.plot()
# plt.savefig(os.path.join(outdir, "simple_ccat_test_instrument_plot5.png"), dpi=200, bbox_inches="tight")
# plt.close("all")


site = maria.get_site("cerro_chajnantor", altitude=5600)

# print(site)
# site.plot()
# plt.savefig("ccat_test_site_plot.png",dpi=200,bbox_inches="tight")
# plt.close("all")

# input_map = maria.map.load(fetch("maps/cluster2.fits"),
#                           nu=280e9)

input_map = maria.map.load("/Users/zaparniukn/Documents/data/OrionA_20170726_850_DR3_ext_HK_JySr_nan_filled.fits",
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


planner = Planner(target=input_map, site=site, constraints={"el": (30, 85)})
plans = planner.generate_plans(total_duration=900,
                               max_chunk_duration=900,
                               scan_pattern="lissajous",
                               sample_rate=50,
                               scan_options={"radius": input_map.width.deg / 2})

plans[0].plot()
print(plans)
plt.savefig(os.path.join(outdir, "OrionA_850_lissajous_full.png"),dpi=200,bbox_inches="tight")
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
    atmosphere_kwargs = {"weather":{"pwv":0.5}},
    map = input_map)

print(sim)

tods = sim.run()

print(tods)
tods[0].plot()
plt.savefig(os.path.join(outdir, "orionA_850_ccat_tod_plot_lissajous_full.png"),dpi=200,bbox_inches="tight")
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

from maria.mappers import BinMapper

mapper = BinMapper(
    # center=input_map.center,
    # frame="ra/dec",
    # width=input_map.width,
    # height=input_map.height,
    # resolution=input_map.width / 256,
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
plt.savefig(os.path.join(outdir, "orionA_850_ccat_output_lissajousBinMapper_map_2d_full.png"),dpi=200,bbox_inches="tight")
plt.close("all")