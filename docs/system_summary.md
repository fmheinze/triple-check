# System constraint summary

[Back to README](../README.md) · [Recoil map plots](recoil_maps.md)

Use `system_summary.py` to calculate outer-orbit constraints for one system or
propagate the uncertainty from two merger-event posteriors. It reports compatible
regions, projected constraints, and Peters-applicability diagnostics, with adaptive
resolution and local refinement.

Run commands from the repository root. See [Setup and Dependencies](../README.md#setup-and-dependencies):
NumPy and SciPy are required; posterior HDF5 input also needs h5py, and posterior
figures need Matplotlib. Use `python3 system_summary.py --help` for all options
and their defaults.

## Single-system analysis

Run with the default system parameters and save a report:

```bash
python3 system_summary.py --workers 4 --json-output system_report.json
```

The text report is printed to the terminal; progress is written separately to
stderr. Optional JSON retains the configuration, results and refinement history.
`--workers` sets the number of processes; more workers require more memory.

## Physical settings and search domain

Use masses, times and frequencies in a consistent source or detector frame.

| Options | Meaning |
| --- | --- |
| `--m1-inner`, `--m2-inner` | First binary's component masses in solar masses |
| `--mrem`, `--m3` | First remnant and tertiary masses in solar masses |
| `--delay-days` | Inter-merger delay in days for a single-system run |
| `--f-ref-hz` | Reference GW peak frequency in Hz |
| `--e-ref-range EMIN EMAX` | Admitted eccentricity interval at the reference frequency |
| `--recoil-ceiling-kms` | Maximum allowed recoil velocity in km/s |
| `--min-peters-orbits` | Minimum ratio of merger delay to initial post-kick orbital period for Peters applicability |
| `--min-peters-rp-rg` | Minimum initial pericenter in units of post-merger `GM/c²` for Peters applicability |
| `--r-sec-threshold` | Threshold used to classify `R_sec` as sufficiently high |
| `--inclination-deg` | Mutual inclination in degrees for the stability criterion (180° gives the least restrictive cut) |

The Mardling–Aarseth stability cut is enabled by default. Use
`--stability-days-before` to set the epoch for the inferred inner separation, or
specify it with `--inner-separation-au` in AU or `--alpha-in` in units of
`G(m1_inner + m2_inner)/c²`.
These two explicit separation options are mutually
exclusive. `--no-ma-mask` disables the stability cut.

The search domain is set by `--aout-min-au`, `--aout-max-au`, `--eout-min`,
`--eout-max`, `--nu-min-deg` and `--nu-max-deg`. Separation bounds always use AU;
`--aout-axis rg` changes the displayed units. The summary defaults to **mean
anomaly**, whereas the map scripts default to true anomaly. Match
`--phase-measure` when comparing them.

## Resolution and convergence

Adaptive refinement is enabled by default. Set the starting grid counts with
`--n-eout`, `--n-aout`, `--n-nu` and `--n-epost`, and use
`--refinement-factor` to specify the multiplicative increase in all four counts
at each step toward higher resolution, rounding upward. The factor defaults to
1.5 when starting counts are supplied without an explicit factor. Specifying
any of these options selects geometric refinement; otherwise, the built-in
resolution sequence is used.

Choose the convergence scope with `--convergence-support`:

| Mode | What agreement checks |
| --- | --- |
| `envelope` (default) | Bulk diagnostics and outer boundaries of projected support; internal gaps may remain unresolved |
| `intervals` | Bulk diagnostics, every support-interval endpoint, and interval topology, including gaps |
| `bulk` | Bulk diagnostics only; local refinement and boundary checks are disabled |

Adaptive runs use local endpoint refinement and gap validation where enabled.
A `converged` result applies only to the selected checks and tolerances.
`--convergence-streak 2` requests two consecutive passing comparisons.

If a run reports `not_converged`, inspect its unresolved checks and stop reason.
Adjust the relevant limit: `--max-refinements`, `--max-grid-points` or
`--max-epost` for global refinement; `--gap-validation-max-points` or
`--endpoint-refinement-max-points` for local checks. Local budgets count initial
scan points, not every evaluation performed during refinement. All tolerances
and further controls are listed by `--help`.

An unresolved adaptive run saves its latest completed result and exits with
status 1. Successful runs exit with 0; success in fixed-resolution mode does not
mean convergence. Expand the search domain if support reaches its edges:
refinement does not expand it automatically.

## Reading the results

Both `all_targets` and `secular_targets` are reported. The first contains all
admitted post-merger targets; the second retains those passing the adopted
Peters-applicability conditions.

| Quantity | Meaning |
| --- | --- |
| Compatible volume fraction `V` | Compatible fraction of the stability-selected domain, measured uniformly in outer eccentricity, logarithmic separation and the selected phase coordinate |
| Strength `S = -log10(V)` | Smaller compatible volume gives greater constraint strength |
| Minimum recoil | Lowest required recoil over reachable, stability-selected progenitors, including those above the chosen ceiling |
| Projected support | Supported ranges of separation, eccentricity and phase, with unresolved regions and endpoint uncertainty reported separately |
| `R_sec` | Compatible secular-target volume divided by compatible all-target volume; undefined if the latter is empty |

`R_sec` is an operational applicability diagnostic, not an error estimate or a
probability that Peters evolution is accurate.

## Two-event posterior analysis

Supply local posterior files for the earlier and later mergers. Supported formats
include public GWTC/PESummary HDF5, PESummary/Bilby JSON, headered text/CSV and NPZ.
If a file contains several analyses, select the appropriate table with
`--first-dataset` or `--second-dataset`; otherwise the reader lists the available
paths rather than choosing a waveform model.

```bash
python3 system_summary.py \
  --posterior-first FIRST.h5 --first-dataset FIRST_LABEL/posterior_samples \
  --posterior-second SECOND.h5 --second-dataset SECOND_LABEL/posterior_samples \
  --remnant-component secondary --posterior-delay-days 6.84 \
  --e-ref-range 0 0.04 --posterior-samples 128 --posterior-seed 12345 \
  --workers 4 --json-output pair_report.json
```

Replace the filenames, dataset paths, delay and eccentricity interval with your
inputs. `secondary` assigns the first remnant to the lighter component of the
second event; `primary` assigns it to the heavier component.

Essential input requirements:

- The first file must contain correlated component and remnant masses. Original
  GWTC-1 files without remnant samples need an updated release containing them.
- Posterior calculations use **detector-frame masses, observed delay and observed  frequency**. Source masses require row-by-row redshift information or equivalent
  mass-frame information for conversion. Reported AU separations are detector-frame
  lengths; divide by a common `1+z` for source-frame lengths.
- If `--posterior-delay-days` is omitted, the delay is obtained from timing samples.
  It never falls back to the single-system delay default.
- `--e-ref-range` supplies an assumed or measured interval for every draw. Alternatively,
  use `--e-ref-min` and/or `--e-ref-max`; omitted posterior bounds default to 0 and 1.
  With no eccentricity options, all reference eccentricities are admitted. When supplied,
  `--e-ref-range` overrides the separate bounds.
- Default prior-corrected matching requires the second analysis's **prior samples**,
  either embedded in its file or supplied with `--second-prior FILE` and, if needed,
  `--second-prior-dataset PATH`. If unavailable, `--pairing-mode overlap` permits a sensitivity
  study, not a properly combined joint posterior.

Matching uses remnant/component mass and luminosity distance by default;
`--match-parameters mass` uses mass alone. Inspect effective sample sizes and
weight-concentration warnings. Vary `--posterior-proposals`, `--posterior-samples`,
`--match-mass-width` and `--match-log-distance-width` to assess sampling sensitivity.

### Outputs and resuming

Posterior mode requires `--json-output` and saves a four-panel PNG beside it.
The report includes weighted support curves, `R_sec` distributions, per-sample
results and numerical-status weights. Check unresolved, failed and undefined
weight as well as the converged subset. These outputs describe feasibility under
measurement uncertainty; they are not an outer-orbit posterior or an association
probability.

JSON is checkpointed after each completed draw. To continue an interrupted run,
stop the old process and repeat the same command with `--resume` and the same
JSON path. Inputs and analysis settings must match; worker count may change.
Completed draws are retained, including failed or nonconverged draws; interrupted
draws restart. Without `--resume`, a new run can overwrite the report.

### Plot saved results

Generate figures without rerunning the dynamics:

```bash
python3 posterior_plots.py pair_report.json \
  --include-r-sec --output pair_support.pdf
python3 posterior_plots.py pair_report.json --distributions \
  --converged-only --output pair_distributions.pdf
```

Use `--family all_targets` or `--family secular_targets` to select a target
family. For distribution plots, omit `--converged-only` to include all saved
samples. Selected samples retain their original weights. Use
`--volume-scale linear` to include zero volumes in distribution histograms;
the default volume axis is logarithmic. Run `python3 posterior_plots.py --help`
for further display options.
