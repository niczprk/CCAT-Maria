import maria
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from maria.instrument import Band
from maria import fetch
from maria import Planner
from maria.mappers import BinMapper


# print("Matplotlib version:", matplotlib.__version__)
# print("Matplotlib backend:", matplotlib.get_backend())

# raise SystemExit("Stopping execution for testing purposes.")
#Band defines and determiens the
#array's sensitivity to different spectra
#--------- MUSTANG-2 Example -----------------

input_map = maria.map.load(fetch("maps/crab_nebula.fits"), nu = 93e9)

input_map.plot()
print(input_map)
plt.savefig("crab_nebula_map.png",dpi=200,bbox_inches="tight")
plt.close("all")

planner = Planner(target=input_map,
                  site= "green_bank",
                  constraints={"el": (60,90)})

plans = planner.generate_plans(total_duration = 900, sample_rate = 100)

plans[0].plot()
print(plans)
plt.savefig("mustang2_plan.png",dpi=200,bbox_inches="tight")
plt.close("all")

instrument = maria.get_instrument("MUSTANG-2")

print(instrument)
instrument.plot()
plt.savefig("mustang2_instrument.png",dpi=200,bbox_inches="tight")
plt.close("all")

sim = maria.Simulation(
    instrument=instrument,
    plans=plans,
    site="green_bank",
    map = input_map,
    atmosphere="2d",
)

print(sim)

tods = sim.run()
tods[0].plot()
plt.savefig("mustang2_tod.png",dpi=200,bbox_inches="tight")
plt.close("all")

mapper = BinMapper(
    tod_preprocessing={
        "remove_modes": {"modes_to_remove": 1},
        "remove_spline": {"knot_spacing": 60, "remove_el_gradient": True},
    },
    map_postprocessing={
        "gaussian_filter": {"sigma": 1},
    },
    units="uK_RJ",
    tods=tods,
)

self = mapper

output_map = mapper.run()

output_map.plot()
plt.savefig("mustang2_output_map.png",dpi=200,bbox_inches="tight")
plt.close("all")

raise SystemExit("Stopping MUSTANG-2 Example Execution.")

#--------- Tutorial Example -----------------


f090 = Band(
    center=90e9, #Hz
    width=20e9, #Hz
    NET_RJ=40e-6, #K*sqrt(s)
    knee=1e0, #Hz
    gain_error=5e-2
)

f150 = Band(
    center=150e9,
    width=30e9,
    NET_RJ=60e-6,
    knee=1e0,
    gain_error=5e-2
)

#Define the array configuration, specifying the 
#detectors distribution on the focal plane

array = {"field_of_view": 0.5,
         "beam_spacing": 1.5,
         "primary_size": 25,
         "bands": [f090, f150]
        }

instrument = maria.get_instrument(array=array)

print(instrument)
instrument.plot()
plt.savefig("instrument_plot.png",dpi=200,bbox_inches="tight")
plt.close("all")

site = maria.get_site("llano_de_chajnantor", altitude=5065)

print(site)
site.plot()
plt.savefig("site_plot.png",dpi=200,bbox_inches="tight")
plt.close("all")

# map_filename = maria.io.fetch("maps/einstein.h5")

# input_map = maria.map.load(filename = map_filename,
#                            nu = 150e9,
#                            resolution=1/1024,
#                            center = (150,10),
#                            frame = "ra/dec",
#                            units = "Jy/pixel")


# print(input_map)
# input_map.to(units = "uK_RJ").plot()
# plt.savefig("input_map.png",dpi=200,bbox_inches="tight")
# plt.close("all")


from maria.io import fetch

map_filename= maria.io.fetch("maps/cluster1.fits") 

input_map = maria.map.load(
    filename=map_filename,
    nu=150e9,
    center=(291.156, -31.23))
input_map.data *= 5e1 #why

print(input_map)
input_map.to("K_RJ").plot()
plt.savefig("input_map.png",dpi=200,bbox_inches="tight")
plt.close("all")

from maria import Planner

planner = Planner(start_time="2024-08-06T03:00:00",
                  target=input_map,
                  site=site,
                  constraints = {"el": (70,90)})

plans = planner.generate_plans(total_duration = 1200, #seconds
                                max_chunk_duration = 600, #seconds
                               scan_pattern = "daisy",
                               scan_options = {"radius": input_map.width.deg / 3},
                               sample_rate = 10 #Hz
                               )

print(plans)
plans[0].plot()
plt.savefig("plan_plot.png",dpi=200,bbox_inches="tight")
plt.close("all")

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
plt.savefig("tod_plot.png",dpi=200,bbox_inches="tight")
plt.close("all")

maria.undebug()

input_map.center

print(input_map.center)

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

output_map.plot(nu_index = [0,1])
plt.savefig("output_map.png",dpi=200,bbox_inches="tight")
plt.close("all")