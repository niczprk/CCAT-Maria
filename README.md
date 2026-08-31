# CCAT-Maria

# FYST Time-Domain Analysis

This repository contains Python scripts used to investigate potential time-domain observing strategies for CCAT Observatory's Fred Young Submillimeter Telescope (FYST) using the MARIA simulation framework.

The analysis focuses on telescope motion, atmospheric loading, and the response of Prime-Cam detectors under different scan patterns and observing conditions.

## Scripts

* `simple_ccat.py` — MARIA simulation setup and TOD generation.
* `module_analysis.py` — atmospheric loading and detector-response analysis.
* `analysis.py` — telescope velocity and acceleration analysis.
* `histogram.py` — motion distribution diagnostics.
* `scan_pattern_comp.py` — comparison of telescope motion between scan patterns.
* `comparison_plots.py` — comparison plots from the atmospheric parameter sweep.
* `observing_strategies.py` — combines detector, motion, and coverage results to compare observing strategies.

## Outputs

Simulation and analysis outputs are stored locally in `outputs/` and `Serpens_ccat_outputs/`. These directories contain a large number of generated files and are therefore excluded from the repository.

Older and superseded analysis scripts are retained in `archive/`.


## Reports

For a more in depth description of the work done with using this analysis the following overleaf reports are available:

1. Report 1: [Forecasting Atmospheric Opacity and Detector Loading for CCAT
Prime-Cam Using Mari](https://www.overleaf.com/read/gckbwfyhhdpd#cdc218)

2. Report 2: [An Early Understanding of Scan Dynamics and Detector Response
for Time-Domain Observations with the CCAT Observatory](https://www.overleaf.com/read/crntfsfrsdqx#cbe543)

3. Appendix: [CCAT Appendix](https://www.overleaf.com/read/kqnvngmwbwym#e88a37) 
