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

For a more in depth description of the work done with using this analysis the following overleaf reports are available:

1. Report 1: https://www.overleaf.com/project/698bada3c02216c4decf956c

2. Report 2: https://www.overleaf.com/project/6a4c391190f148df1ddd4a47

3. Appendix: https://www.overleaf.com/project/6a85f7cc0932475b8d02ae17
