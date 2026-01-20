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


# --define outdirs for different laptops

outdir_pers= "/mnt/c/Users/nickz/OneDrive/Documents/GitHub/CCAT-Maria/maria_outputs"

outdir_pro="/Users/zaparniukn/Documents/maria/maria_outputs"

outdir = outdir_pers

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

array = { "n": 500, #unknown number of detectors?
        "shape": "hexagon",
        "field_of_view": 1.3, #degrees...
         "beam_spacing": 1.8, #not sure about this one
         "primary_size": 6, #in meters...
         "bands": [f280], #, f220, f350, f410, f850],
        #  "packing": "triangular",
         "polarized": False,
        #  "array_name": "CCAT-prime",
        #  "offsets": None
        }

instrument = maria.get_instrument(array=array)

print(instrument)
instrument.plot()
plt.savefig(os.path.join(outdir, "simple_ccat_test_instrument_plot4.png"), dpi=200, bbox_inches="tight")
plt.close("all")

site = maria.get_site("cerro_chajnantor", altitude=5600)

# print(site)
# site.plot()
# plt.savefig("ccat_test_site_plot.png",dpi=200,bbox_inches="tight")
# plt.close("all")

input_map = maria.map.load(fetch("maps/cluster2.fits"),
                          nu=280e9)

input_map.data *= 10e1 

input_map[..., 256: -256, 256: -256].to("K_RJ").plot(cmap="cmb")
print(input_map)
plt.savefig(os.path.join(outdir, "input_map_cmb.png"),dpi=200,bbox_inches="tight")
plt.close("all")


# map_filename= maria.io.fetch("maps/cluster1.fits") 

# input_map = maria.map.load(
#     filename=map_filename,
#     nu=280e9, #what is this used for?
#     center=(291.156, -31.23)) #wcs..?
# input_map.data *= 50e1 #why

# print(input_map)
# input_map.to("K_RJ").plot()
# plt.savefig("input_map.png",dpi=200,bbox_inches="tight")
# plt.close("all")


planner = Planner(target=input_map, site=site, constraints={"el": (65, 85)})
plans = planner.generate_plans(total_duration=900,
                               max_chunk_duration=900,
                               sample_rate=50,
                               scan_options={"radius": input_map.width.deg / 2})

plans[0].plot()
print(plans)
plt.savefig(os.path.join(outdir, "simple_ccat_plan_plot4.png"),dpi=200,bbox_inches="tight")
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
    atmosphere_kwargs = {"weather":{"pwv":1.0}},
    map = input_map)

print(sim)

tods = sim.run()

print(tods)
tods[0].plot()
plt.savefig(os.path.join(outdir, "simple_ccat_tod_plot.png"),dpi=200,bbox_inches="tight")
plt.close("all")

maria.undebug()

input_map.center

# print(input_map.center)

from maria.mappers import MaximumLikelihoodMapper

from maria.mappers import MaximumLikelihoodMapper

ml_mapper = MaximumLikelihoodMapper(tods=tods,
                                    width=0.75 * input_map.width.deg,
                                    height=0.75 * input_map.height.deg,
                                    resolution=10 * input_map.resolution.deg,
                                    units="mK_RJ")
print(f"{ml_mapper.loss() = }")

print(ml_mapper.map)
ml_mapper.map.plot(cmap="cmb")
plt.savefig(os.path.join(outdir, "simple_ccat_ml_output_map4.png"),dpi=200,bbox_inches="tight")
plt.close("all")

ml_mapper.fit(epochs=4, steps_per_epoch=32, lr=2e-1)
ml_mapper.map.plot(cmap="cmb")
plt.savefig(os.path.join(outdir, "simple_ccat_ml_output_map_fitted4.png"),dpi=200,bbox_inches="tight")
plt.close("all")

raise SystemExit("Stopping CCAT-prime Example Execution Before Binning.")

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
plt.savefig(os.path.join(outdir, "simple_ccat_output_map4.png"),dpi=200,bbox_inches="tight")
plt.close("all")