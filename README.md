
# triple-check

This code constrains the progenitor outer orbits of sequential black-hole mergers in hierarchical triples. Given an adopted upper bound on the first-merger recoil velocity, the observational inputs are the black-hole masses, the inter-merger delay, and the eccentricity of the second binary at a reference gravitational-wave frequency. The code additionally supports posterior-distribution inputs from merger events to obtain constraints that account for the measurement uncertainties.

## Setup and Dependencies

Run the commands below from this repository's root, where the Python scripts live.
A virtual environment keeps the dependencies separate from other projects:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy matplotlib pyvista h5py pillow
```

NumPy and SciPy are sufficient for single-system summaries. Matplotlib is needed
for 2D maps and posterior figures, PyVista for 3D rendering, h5py for HDF5 posterior
files, and Pillow for saving 3D renders as PDF. Interactive 3D plots require a
working graphics environment.

## Quick Start

Show the available options:

```bash
python3 minimum_recoil_maps_2d.py --help
python3 minimum_recoil_maps_3d.py --help
python3 system_summary.py --help
```

Generate a simple 2D minimum-recoil-velocity map slice:

```bash
python3 minimum_recoil_maps_2d.py --e-out 0.9 \
  --n-aout 300 --n-nu 360 --n-epost 1000 --outdir kick_grid_maps/quickstart
```

The example uses the default system parameters (which are the parameters for the two mergers GW190514→GW190521). In order to create the corresponding full 3D map, run:

```bash
python3 minimum_recoil_maps_3d.py \
  --n-eout 100 --n-aout 300 --n-nu 360 --n-epost 1000
```

For a more detailed system summary using the default tolerances, run:

```bash
python3 system_summary.py --workers 8
```


## Main scripts

| Script | Purpose |
| --- | --- |
| [minimum_recoil_maps_2d.py](minimum_recoil_maps_2d.py) | Plot axis-aligned 2D slices of a minimum-required-recoil velocity map, highlighting allowed regions |
| [minimum_recoil_maps_3d.py](minimum_recoil_maps_3d.py) | Plot 3D minimum-recoil-velocity map with volume rendering and allowed-region visualization |
| [system_summary.py](system_summary.py) | Single-system and posterior constraint summaries |
| [posterior_plots.py](posterior_plots.py) | Plot figures from saved posterior summary JSON |

The scripts share their physical model in `triple_core.py` and `peters_tools.py`,
command-line options in `triple_cli.py`, and input/output helpers in `triple_io.py`.

For individual-system summaries, `edge_tracking.py`, `endpoint_refinement.py`,
and `gap_validation.py` provide local refinement and validation of projected
constraints, while `support_decision.py` supplies the shared support-classification
and interval-handling rules.

## Documentation

- [Minimum recoil velocity maps: 2D and 3D](docs/recoil_maps.md) — generate slices and volume visualizations of minimum-recoil-velocity maps.
- [System summary](docs/system_summary.md) — summarize the results for a single system, inspect
  convergence and projected constraints, and propagate event posterior samples.
