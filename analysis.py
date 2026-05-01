from pathlib import Path

from . import simple_ccat
from maria import TOD


selected_band = "280" #make sure these match

NU_HZ = 280e9  # Hz
bandwidth_hz = 60e9  # GHz bandwidth for 850 GHz band
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

EL_LIMITS = (68, 85)  # degrees

T_0 = 278.868 #K, atmospheric ground temp



START_TIME = "2022-02-10T23:05:00"

# "2022-02-10T22:45:00" for around 75 degrees ?
#"2022-02-10T20:30:00" for around 60 degrees 
#"2022-02-10T18:55:00" for roughly 45 degrees
#"2022-02-10T18:30:00" for roughly 40 degrees
#"2022-02-10T17:00:00" for roughly 30 degrees
 

TOTAL_DURATION_S = 1800  # seconds
SIM_DURATION_S = 1800  # seconds
SAMPLE_RATE_HZ = 50  # Hz
SCAN_PATTERN = "daisy"
CHUNK_NUMBER = 0

PREFIX = "OrionA_tod_test"
ANALYSIS_OUTDIR = Path(f"outputs/{PREFIX}_analysis_outputs")
TOD_OUTDIR = Path(f"outputs/{PREFIX}_tod_files")


simple_ccat.tod_analysis(maps = False,
    save_all_plots = False,
    run_mode = "fits",
    atm_plot = True,
    temp_mode = "inst",
    ccat_band = "280",
    map_type = "BM",
    pwv_mm = PWV_MM,
)

tod = TOD.from_fits(TOD_OUTDIR / f"{PREFIX}_tods.fits")