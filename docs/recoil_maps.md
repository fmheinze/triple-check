# Minimum-recoil-velocity map plots: 2D and 3D

[Back to README](../README.md) · [System summary](system_summary.md)

Run all commands from the repository root, where
`minimum_recoil_maps_2d.py` and `minimum_recoil_maps_3d.py` live.
See the [setup instructions](../README.md#setup-and-dependencies) for dependencies.

## What the maps show

The progenitor coordinates are outer semimajor axis `a_out`, outer eccentricity
`e_out`, and orbital phase at the first merger. At each point, the code minimizes
the required recoil over accessible post-merger Peters target orbits and both
post-kick radial directions. Targets must meet the specified merger delay and
reference-frequency eccentricity filter.

By default, maps use **true anomaly**. Use `--phase-measure mean_anomaly` for a
mean-anomaly axis. The system-summary tool defaults to mean anomaly; explicitly
match this setting when comparing results.

The recoil ceiling highlights compatible progenitors. The Mardling–Aarseth stability
mask is enabled by default.

## Shared physical settings

Both plotting scripts accept these options:

| Options | Meaning |
| --- | --- |
| `--m1-inner`, `--m2-inner` | Inner masses in solar masses |
| `--mrem`, `--m3` | First remnant and third-body masses in solar masses |
| `--delay-days` | Delay between the first and the second merger in days |
| `--f-ref-hz` | Reference GW peak frequency in Hz |
| `--e-ref-range EMIN EMAX` | Admitted reference-eccentricity interval |
| `--recoil-ceiling-kms` | Recoil velocity ceiling in km/s |
| `--min-peters-orbits` | Minimum delay divided by the initial post-kick orbital period for assessing the Peters applicability |
| `--min-peters-rp-rg` | Minimum initial pericenter in units of post-merger `GM/c²` for assessing the Peters applicability |
| `--secular-only` | Restrict to targets passing the Peters applicability conditions|
| `--inclination-deg` | Mutual inclination for stability in degrees (set to 180° for the least restrictive stability cut) |

Without `--secular-only`, all targets admitted by the reference-eccentricity
filter are considered. The orbit-ratio cut is a secular-applicability proxy,
not the integrated number of inspiral orbits.

The search domain is controlled by `--aout-min-au`, `--aout-max-au`,
`--eout-min`, `--eout-max`, `--nu-min-deg`, and `--nu-max-deg`. Grid counts are
`--n-aout`, `--n-eout`, `--n-nu`, and `--n-epost` (the target eccentricity grid).
Separation limits always use AU; `--aout-axis rg` changes displayed separation
units to `G(mrem + m3)/c²`.

Use `--stability-days-before` to set the epoch for the inferred inner separation,
or supply `--inner-separation-au` in AU or `--alpha-in` in units of
`G(m1_inner + m2_inner)/c²`. The two explicit
separation options are mutually exclusive. `--no-ma-mask` disables stability masking.

## 2D slices

Choose one or more planes with `--slices`:

| Plane | Scanned coordinates | Fixed-coordinate option |
| --- | --- | --- |
| `aout-nu` (default) | Semi-major axis and phase | `--e-out` |
| `eout-aout` | Eccentricity and semi-major axis | `--nu-m-deg` |
| `eout-nu` | Eccentricity and phase | `--a-out-au` |

Each fixed-coordinate option accepts multiple values. For example:

```bash
python3 minimum_recoil_maps_2d.py --slices aout-nu --e-out 0 0.6 0.9 \
  --n-aout 120 --n-nu 180 --n-epost 1000 \
  --workers 3 --output-format pdf --outdir kick_grid_maps/2d
```

To request all three slice orientations:

```bash
python3 minimum_recoil_maps_2d.py --slices aout-nu eout-aout eout-nu \
  --e-out 0.6 --nu-m-deg 0 90 --a-out-au 0.1 \
  --n-aout 60 --n-eout 40 --n-nu 72 --n-epost 300 \
  --outdir kick_grid_maps/all_planes
```

Figures are saved in `--outdir` as PNG by default; `--output-format pdf` selects
PDF. Additional outputs are available with:

- `--fixed-epost VALUE [VALUE ...]`: maps for fixed post-merger eccentricities.
- `--output-minimizing-epost`: additional maps containing the target eccentricity producing minimum recoil.
- `--output-minimizing-eref`: additional maps containing that target's reference-frequency eccentricity.
- `--write-candidate-csv`: a diagnostic target table, including rejected candidates.

The default palette is `viridis`, with crimson compatibility highlighting.
Use `--cmap`, `--highlight-color`, `--vmin`, and `--vmax` to change these choices.
The default color limits are 100–5000 km/s.
`--contour-smoothing-sigma 0` disables display contour smoothing; smoothing does
not change the underlying recoil calculation.

## 3D visualization

A coarse interactive example:

```bash
python3 minimum_recoil_maps_3d.py \
  --n-eout 24 --n-aout 60 --n-nu 72 --n-epost 300 --workers 4
```

The display uses phase, logarithmic separation, and outer eccentricity. It shows
a velocity-colored volume and the highlighted recoil-compatible region.
Use `--no-volume` or `--no-highlight` to hide either component.

Save a render with an eccentricity slice:

```bash
python3 minimum_recoil_maps_3d.py \
  --n-eout 24 --n-aout 60 --n-nu 72 --n-epost 300 \
  --show-slice --slice-axis z --slice-value 0.6 \
  --offscreen --output kick_grid_maps/kick_map_3d.png
```

Slice coordinates are:

| `--slice-axis` | Meaning of `--slice-value` |
| --- | --- |
| `x` | Phase in degrees |
| `y` | `log10(a_out)` in the selected `--aout-axis` units |
| `z` | Outer eccentricity |

Use `--output` to set the output image path. PNG and PDF are supported, and
`--output-format` can override the filename extension. A 3D PDF contains a raster
render.

Use `--offscreen` to render and save an image without opening a visible plot
window. To inspect and rotate the plot interactively after saving, omit
`--offscreen` and add `--interactive`. Without `--interactive`, the plot window
closes after saving. When `--offscreen` is set, `--interactive` has no effect.
Offscreen rendering still requires a working VTK graphics backend.

## Resolution and interpretation

The examples use reduced grids for exploratory plots. Increase both progenitor
and target-grid resolution and check the search domain before interpreting thin
regions or boundaries. Plotting scripts do not certify numerical convergence.
Adaptive local refinement of projected constraints is performed by
`system_summary.py`; the 2D and 3D plotting scripts do not perform this refinement.


Both scripts reuse equivalent reflected phases and restore the full phase grid
for plotting. `--workers` controls parallel execution; defaults are one worker
for 2D and four for 3D. Larger 3D grids and more workers require more memory.

For numerical volumes, projected support, convergence diagnostics, and posterior
uncertainty propagation, see the [system-summary guide](system_summary.md).

### Sampling near unit eccentricity

The 3D map defaults to `--eout-spacing refined`. It keeps all `--n-eout` linearly
spaced base planes and adds calculated planes where adjacent values differ by
more than 0.1 in `log(1 - e_out)`. The printed grid dimensions include these extra
planes. This resolves rapidly moving regions near unit eccentricity without
increasing the resolution of the entire cube.

Nonuniform planes are drawn at their actual eccentricities, with a linear
physical eccentricity axis. Recoil calculations, target cuts, and compatible
points are unchanged. Use `--eout-spacing linear` to reproduce the original grid
with exactly `--n-eout` planes. This setting only affects the 3D plotting script.

For responsive interaction, the faint colored volume is interpolated once onto a
regular display grid. The highlighted compatible surface and physical-coordinate
slices still use the full refined grid. This changes only background rendering;
it does not alter computed recoil values or compatibility decisions.
