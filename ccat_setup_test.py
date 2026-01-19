import maria
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from maria import Instrument
from maria.instrument import Band
from maria import fetch
from maria import Planner
from maria.mappers import BinMapper


f220 = Band(
    center=220e9, #Hz
    width=40e9, #Hz
    NET_RJ=100e-6, #K*sqrt(s)
    knee=1e0, #Hz
    gain_error=5e-2
)

f280= Band(
    center=280e9, #Hz
    width=50e9, #Hz
    NET_RJ=150e-6, #K*sqrt(s) 
    knee=1e0, #Hz
    gain_error=5e-2
)

f350 = Band(
    center = 350e9,
    width = 60e9,
    NET_RJ = 200e-6,
    knee = 1e0,
    gain_error = 5e-2
)

f410 = Band(
    center = 410e9,
    width = 70e9,
    NET_RJ = 250e-6,
    knee = 1e0,
    gain_error = 5e-2
)

f850 = Band(
    center = 850e9,
    width= 100e9,
    NET_RJ = 500e-6, 
    knee = 1e0,
    gain_error = 5e-2
)

f260_spec1 = Band(
    center=260e9, #Hz
    width=50e9, #Hz
    NET_RJ=120e-6, #K*sqrt(s)
    knee=1e0, #Hz
    gain_error=5e-2
)

f370_spec2 = Band(
    center=370e9, #Hz
    width=70e9, #Hz
    NET_RJ=220e-6, #K*sqrt(s)
    knee=1e0, #Hz
    gain_error=5e-2
)

# f090 = Band(
#     center=90e9, #Hz
#     width=20e9, #Hz
#     NET_RJ=40e-6, #K*sqrt(s)
#     knee=1e0, #Hz
#     gain_error=5e-2
# )

# f150 = Band(
#     center=150e9,
#     width=30e9,
#     NET_RJ=60e-6,
#     knee=1e0,
#     gain_error=5e-2
# )

#Define the array configuration, specifying the 
#detectors distribution on the focal plane

array = { #"n": #unknown number of detectors?
        "shape": "hexagon",
        "field_of_view": 1.15, #degrees...
         "beam_spacing": 1.8, #not sure about this one
         "primary_size": 6, #in meters...
         "bands": [f220, f280, f350, f410, f850],
         "packing": "triangular",
         "polarized": True,
        #  "array_name": "CCAT-prime",
        #  "offsets": None
        }

array_spec = {
        "shape": "hexagon",
        "field_of_view": 1.15, #degrees...
         "beam_spacing": 1.8, #not sure about this one
         "primary_size": 6, #in meters...
         "bands": [f260_spec1, f370_spec2],
         "packing": "triangular",
         "polarized": False,
        #  "array_name": "CCAT-prime EoR Spectrometer",
        #  "offsets": None
        }

subarray_850 = {"name": "mod_850", "bands": [f850], "focal_plane_offset": (0.0, 0.0), **array}
subarray_410 = {"name": "mod_410", "bands": [f410], "focal_plane_offset": (0.5*1.8, -0.866*1.8), **array}
subarray_350 = {"name": "mod_350", "bands": [f350], "focal_plane_offset": (-0.5*1.8, -0.866*1.6), **array}
subarray_280 = {"name": "mod_280", "bands": [f280], "focal_plane_offset": (-1.0*1.8, 0.0), **array}
subarray_220 = {"name": "mod_220", "bands": [f220], "focal_plane_offset": (-0.5*1.8, 0.866*1.8), **array}
subarray_Placeholder_EoRSpec1 = {"name": "mod_EoRSpec1", "bands": [f260_spec1], "focal_plane_offset": (1.0*1.8, 0.0), **array_spec}
subarray_Placeholder_EoRSpec2 = {"name": "mod_EoRSpec2", "bands": [f370_spec2], "focal_plane_offset": (0.5*1.8, 0.866*1.8), **array_spec}

instrument = Instrument(arrays=[subarray_850, subarray_410, subarray_350, subarray_280, subarray_220, subarray_Placeholder_EoRSpec1, subarray_Placeholder_EoRSpec2])

print(instrument)
instrument.plot()
plt.savefig("ccat_test_instrument_plot6.png", dpi=200, bbox_inches="tight")
plt.close("all")

raise SystemExit("Stopping CCAT-prime Example Execution Before Simulation.")

site = maria.get_site("cerro_chajnantor", altitude=5600)

print(site)
site.plot()
plt.savefig("ccat_test_site_plot.png",dpi=200,bbox_inches="tight")
plt.close("all")


map_filename= maria.io.fetch("maps/cluster1.fits") 

input_map = maria.map.load(
    filename=map_filename,
    nu=220e9, #what is this used for?
    center=(291.156, -31.23)) #wcs..?
input_map.data *= 5e1 #why

print(input_map)
input_map.to("K_RJ").plot()
plt.savefig("input_map.png",dpi=200,bbox_inches="tight")
plt.close("all")

from maria import Planner

planner = Planner(start_time="2024-08-06T03:00:00",
                  target=input_map,
                  site=site,
                  constraints = {"el": (0,90)}) # telescope elevation limits

plans = planner.generate_plans(total_duration = 600, #seconds
                                max_chunk_duration = 600, #seconds
                               scan_pattern = "daisy",
                               scan_options = {"radius": input_map.width.deg / 3},
                               sample_rate = 25 #Hz
                               )

print(plans)
plans[0].plot()
plt.savefig("ccat_plan_plot2.png",dpi=200,bbox_inches="tight")
plt.close("all")

# raise SystemExit("Stopping CCAT-prime Example Execution Before Simulation.")

sim = maria.Simulation(
    instrument=instrument,
    plans=plans,
    site=site,
    atmosphere = "2d",
    # atmosphere_kwargs = {"weather":{"pwv":0.5}},
    map = input_map)

print(sim)

tods = sim.run()

print(tods)
tods[0].plot()
plt.savefig("ccat_tod_plot.png",dpi=200,bbox_inches="tight")
plt.close("all")

# maria.undebug()

# input_map.center

# print(input_map.center)

from maria.mappers import BinMapper

mapper = BinMapper(
    # center=input_map.center,
    # frame="ra/dec",
    # width=input_map.width,
    # height=input_map.height,
    # resolution=input_map.width / 256,
    tod_preprocessing={
        "remove_modes": {"modes_to_remove": 1},
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

output_map.plot(nu_index = [0,1])
plt.savefig("ccat_output_map.png",dpi=200,bbox_inches="tight")

plt.close("all")