"""Shared filename formatting, image/table/report output, and posterior input.

Plot construction and analysis remain in their respective modules. HDF5 support is optional;
posterior waveform analyses are never silently combined.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from triple_core import AnalysisSettings, KickMapSlice, PostFirstMergerCandidates


# -------------------------------------------------------------------------------------------------
# Sample decoding and dataset discovery
# -------------------------------------------------------------------------------------------------

POSTERIOR_DATA_ALIASES = {
    'mass_1': ('mass_1', 'm1_detector_frame_Msun', 'm1', 'mass1'),
    'mass_2': ('mass_2', 'm2_detector_frame_Msun', 'm2', 'mass2'),
    'final_mass': ('final_mass', 'mf_detector_frame_Msun', 'mf', 'final_mass_non_evolved'),
    'mass_1_source': ('mass_1_source', 'm1_source_frame_Msun'),
    'mass_2_source': ('mass_2_source', 'm2_source_frame_Msun'),
    'final_mass_source': ('final_mass_source', 'mf_source_frame_Msun',
                          'final_mass_source_non_evolved'),
    'luminosity_distance': ('luminosity_distance', 'luminosity_distance_Mpc', 'distance', 'dist'),
    'redshift': ('redshift', 'z'),
    'geocent_time': ('geocent_time', 'geocent_time_GPS', 'time', 'tc'),
}


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _columns(obj):
    """Decode a structured array, PESummary matrix, or column dictionary."""
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return {name: np.asarray(obj[name]) for name in obj.dtype.names}
    if isinstance(obj, dict):
        if obj.get('__dataframe__') is True or obj.get('__array__') is True:
            return _columns(obj['content'])
        if 'parameter_names' in obj and 'samples' in obj:
            names = [_decode(x) for x in obj['parameter_names']]
            values = np.asarray(obj['samples'])
            if values.ndim == 2 and values.shape[1] == len(names):
                return dict(zip(names, values.T))
        arrays = {}
        for name, value in obj.items():
            if isinstance(value, dict) and value.get('__array__') is True:
                value = value['content']
            if isinstance(value, (list, np.ndarray)):
                a = np.asarray(value)
                if a.ndim == 1 and a.dtype.kind in 'fiu':
                    arrays[name] = a
        if arrays:
            return arrays
    return None


def _is_sample_path(path):
    leaf = path.rsplit('/', 1)[-1].lower()
    prior_path = any(part in ('prior', 'priors', 'prior_samples', 'overall_prior')
                     for part in path.lower().split('/'))
    return ('posterior' in leaf or leaf in ('samples', 'table')) and not prior_path


def _hdf_columns(obj):
    import h5py
    if isinstance(obj, h5py.Dataset):
        return _columns(obj[()]) if obj.dtype.names else None
    if 'parameter_names' in obj and 'samples' in obj:
        return _columns({key: obj[key][()] for key in ('parameter_names', 'samples')})
    arrays = {k: v[()] for k, v in obj.items() if isinstance(v, h5py.Dataset)
              and v.ndim == 1 and v.dtype.kind in 'fiu'}
    return arrays or None


def _hdf_description(obj):
    """Inspect potentially large release files without reading sample arrays."""
    import h5py
    if isinstance(obj, h5py.Dataset):
        if obj.dtype.names and obj.ndim == 1:
            return {'columns': list(obj.dtype.names), 'rows': obj.shape[0]}
        return None
    if 'parameter_names' in obj and 'samples' in obj:
        return {'columns': [_decode(x) for x in obj['parameter_names'][()]],
                'rows': obj['samples'].shape[0]}
    datasets = {k: v for k, v in obj.items() if isinstance(v, h5py.Dataset)
                and v.ndim == 1 and v.dtype.kind in 'fiu'}
    if datasets:
        return {'columns': list(datasets), 'rows': next(iter(datasets.values())).shape[0]}
    return None


def _discover_sample_tables(path):
    """List posterior dataset paths and columns without loading waveform/strain data."""
    path = Path(path)
    found = {}
    if path.suffix.lower() in ('.h5', '.hdf5', '.hdf'):
        try:
            import h5py
        except ImportError as exc:
            raise ValueError('Reading HDF5 requires h5py: python -m pip install h5py') from exc
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if not _is_sample_path(name):
                    return
                description = _hdf_description(obj)
                if description:
                    found[name] = description
            f.visititems(visit)
    elif path.suffix.lower() == '.json':
        root = json.loads(path.read_text())
        def walk(obj, name=''):
            if not isinstance(obj, dict):
                return
            if _is_sample_path(name) or name == '':
                cols = _columns(obj)
                if cols:
                    found[name or '/'] = {
                        'columns': list(cols), 'rows': len(next(iter(cols.values())))}
                    return
            for key, value in obj.items():
                walk(value, (name + '/' + key).strip('/'))
        walk(root)
    else:
        cols = _text_columns(path)
        found['/'] = {'columns': list(cols), 'rows': len(next(iter(cols.values())))}
    return found


def _text_columns(path):
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    data = np.genfromtxt(path, names=True, delimiter=',' if path.suffix.lower() == '.csv' else None,
                         encoding='utf-8', ndmin=1)
    if not data.dtype.names:
        raise ValueError('Text posterior files need a header row with column names.')
    return {key: data[key] for key in data.dtype.names}


# -------------------------------------------------------------------------------------------------
# Posterior and prior sample loading
# -------------------------------------------------------------------------------------------------

@dataclass
class Samples:
    columns: dict
    provenance: dict

    @property
    def size(self):
        return len(next(iter(self.columns.values())))

    def get(self, name, required=True):
        for key in POSTERIOR_DATA_ALIASES.get(name, (name,)):
            if key in self.columns:
                return np.asarray(self.columns[key], dtype=float)
        if required:
            raise ValueError(
                f'{self.provenance["path"]}: missing {name}; columns: {", ".join(self.columns)}')
        return None

    def weights(self):
        for key in ('weights', 'weight', 'posterior_weight'):
            if key in self.columns:
                w = self.get(key)
                break
        else:
            if 'log_weight' in self.columns:
                lw = self.get('log_weight')
                if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)) or not np.any(np.isfinite(lw)):
                    raise ValueError('Invalid log_weight column.')
                w = np.exp(lw - np.max(lw))
            else:
                w = np.ones(self.size)
        if np.any(~np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
            raise ValueError('Posterior weights must be finite, nonnegative and have positive sum')
        return w / w.sum()

    def detector_mass(self, name):
        values = self.get(name, required=False)
        if values is not None:
            return values
        source = self.get(name + '_source', required=False)
        if source is None:
            return self.get(name)  # informative missing-column error
        z = self.get('redshift', required=False)
        if z is None:
            for mass in ('mass_1', 'mass_2'):
                det, src = self.get(mass, False), self.get(mass + '_source', False)
                if det is not None and src is not None:
                    z = det / src - 1
                    break
        if z is None or np.any(~np.isfinite(z)) or np.any(z < 0):
            raise ValueError(f'Converting {name}_source requires valid redshift samples.')
        return source * (1 + z)


def read_samples(path, dataset=None):
    path = Path(path)
    if dataset is None:
        options = _discover_sample_tables(path)
        if len(options) != 1:
            raise ValueError(f'{path}: select a dataset explicitly; available posterior tables: '
                             + ', '.join(options))
        dataset = next(iter(options))
    dataset = dataset.strip('/')
    if path.suffix.lower() in ('.h5', '.hdf5', '.hdf'):
        import h5py
        with h5py.File(path, 'r') as f:
            if dataset not in f and dataset:
                raise ValueError(f'{path}: dataset {dataset!r} not found.')
            cols = _hdf_columns(f[dataset] if dataset else f)
    elif path.suffix.lower() == '.json':
        obj = json.loads(path.read_text())
        try:
            for key in dataset.split('/') if dataset else []:
                obj = obj[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f'{path}: dataset {dataset!r} not found.') from exc
        cols = _columns(obj)
    else:
        if dataset:
            raise ValueError('Text and NPZ files use dataset / only.')
        cols = _text_columns(path)
    if not cols:
        raise ValueError(f'{path}: {dataset} is not a supported sample table.')
    if any(np.asarray(v).ndim != 1 for v in cols.values()):
        raise ValueError('Sample columns must be nonempty one-dimensional arrays of equal length.')
    lengths = {len(v) for v in cols.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError('Sample columns must be nonempty one-dimensional arrays of equal length.')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return Samples(cols, {'path': str(path.resolve()), 'dataset': dataset or '/',
                          'sha256': digest.hexdigest(), 'rows': next(iter(lengths))})


def find_prior_samples(path, posterior_dataset, *, try_default=False):
    """Read a prior table from known release layouts, or return None if absent.

    ``posterior_dataset`` selects the analysis-specific PESummary prior first. For a standalone
    prior file, ``try_default`` also tries the ordinary single-table reader before the known prior
    paths. Explicit dataset selection uses ``read_samples`` directly. The caller decides what a
    missing prior means.
    """
    if try_default:
        try:
            return read_samples(path)
        except ValueError:
            pass  # A standalone HDF5 prior may use one of the known paths below.
    label = posterior_dataset.rsplit('/', 1)[0]
    paths = ([label + '/priors/samples'] if '/' in posterior_dataset else [])
    paths += ['prior', 'prior_samples', 'Overall_prior', 'priors/samples']
    for dataset in paths:
        try:
            return read_samples(path, dataset)
        except (ValueError, KeyError):
            continue
    return None


def _load_prior(args, second):
    """Select the second-event prior; a missing prior never implies a flat one."""
    path = args.second_prior or args.posterior_second
    if args.second_prior_dataset:
        return read_samples(path, args.second_prior_dataset)
    prior = find_prior_samples(path, second.provenance['dataset'],
                               try_default=bool(args.second_prior))
    if prior is not None:
        return prior
    raise ValueError(
        'No usable second-event prior samples found. '
        'Supply --second-prior/--second-prior-dataset, or explicitly select '
        '--pairing-mode overlap for a sensitivity study (not a joint posterior).'
    )


# -------------------------------------------------------------------------------------------------
# Filename and suffix helpers
# -------------------------------------------------------------------------------------------------

def format_e(x: float) -> str:
    return f"{x:.4g}".replace(".", "p").replace("-", "m")


def unique_filename_values(values, label):
    """Keep first occurrences, rejecting distinct values with the same filename tag."""
    unique = list(dict.fromkeys(values))
    tags = {}
    for value in unique:
        tag = format_e(value)
        if tag in tags:
            raise ValueError(
                f"{label}: {tags[tag]:g} and {value:g} share filename tag {tag!r}. "
                "Run them separately in different output directories."
            )
        tags[tag] = value
    return unique


def filter_suffix(e_ref_min: Optional[float], e_ref_max: Optional[float]) -> str:
    if e_ref_min is None and e_ref_max is None:
        return ""
    if e_ref_min is None:
        return f"_erefmax_{format_e(e_ref_max)}"
    if e_ref_max is None:
        return f"_erefmin_{format_e(e_ref_min)}"
    return f"_erefrange_{format_e(e_ref_min)}_{format_e(e_ref_max)}"


def axis_suffix(aout_axis: str) -> str:
    return "" if aout_axis == "au" else f"_aout_{aout_axis}"


def phase_suffix(phase_measure: str) -> str:
    """File-name tag for the phase measure.

    The measure changes the map on every plane, whether the phase is scanned or held, so two runs
    that differ only in it would otherwise overwrite each other. The default measure adds nothing,
    keeping existing file names.
    """
    from triple_core import PHASE_TRUE_ANOMALY

    return "" if phase_measure == PHASE_TRUE_ANOMALY else "_M"


def _slice_file_stem(kick_slice: KickMapSlice) -> str:
    from triple_core import PHASE_TRUE_ANOMALY

    if kick_slice.fixed_name == "e_out":
        fixed = f"eout_{format_e(kick_slice.fixed_value)}"
    elif kick_slice.fixed_name == "nu_m_deg":
        phase = "nu_m" if kick_slice.phase_measure == PHASE_TRUE_ANOMALY else "M_m"
        fixed = f"{phase}_{format_e(kick_slice.fixed_value)}deg"
    else:
        fixed = f"aout_{format_e(kick_slice.fixed_value)}AU"
    return f"{kick_slice.plane.replace('-', '_')}_{fixed}"


# -------------------------------------------------------------------------------------------------
# Figure and candidate-table export
# -------------------------------------------------------------------------------------------------

def save_plot_output(plotter, args):
    """Save a rendered PNG or PDF, then optionally keep the 3D window open."""
    if args.output is None and args.output_format is None:
        plotter.show()
        return None
    outpath = Path(args.output or "kick_map_3d").expanduser().resolve()
    if args.output_format is not None:
        outpath = outpath.with_suffix("." + args.output_format)
    elif not outpath.suffix:
        outpath = outpath.with_suffix(".png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.show(interactive=False, auto_close=False)
        if outpath.suffix.lower() == ".pdf":
            from PIL import Image
            pixels = plotter.screenshot(return_img=True)
            with Image.fromarray(pixels).convert("RGB") as rendered:
                rendered.save(outpath, format="PDF", resolution=220.0)
        else:
            plotter.screenshot(str(outpath))
        print(f"Saved image to {outpath}")
        if args.interactive and not args.offscreen:
            plotter.show()
    finally:
        plotter.close()
    return outpath


def write_candidate_table(
    outpath: Path,
    candidates: PostFirstMergerCandidates,
    settings: AnalysisSettings,
    f_ref_hz: float,
) -> None:
    from triple_core import PETERS_SECULAR_OK, passes_e_ref_filter

    fieldnames = [
        "e_post",
        "a_AU",
        "rp_AU",
        "ra_AU",
        "f_peak_initial_Hz",
        "e_ref",
        "f_ref_hz",
        "passes_e_ref_filter",
        "orbital_period_days",
        "n_orbits_to_merger",
        "rp_over_rg",
        "peters_applicability_label",
        "selected_by_run",
    ]
    with outpath.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for values in zip(
            candidates.epost,
            candidates.a_au,
            candidates.rp_au,
            candidates.ra_au,
            candidates.f_peak_initial_hz,
            candidates.e_ref,
            candidates.orbital_period_days,
            candidates.n_orbits_to_merger,
            candidates.rp_over_rg,
            candidates.applicability_label,
        ):
            e, a, rp, ra, fpk, eref, period, norbits, rprg, label = values
            passes_reference = passes_e_ref_filter(
                float(eref), settings.e_ref_min, settings.e_ref_max
            )
            selected = passes_reference and (
                not settings.require_peters_applicable or label == PETERS_SECULAR_OK
            )
            writer.writerow(
                {
                    "e_post": e,
                    "a_AU": a,
                    "rp_AU": rp,
                    "ra_AU": ra,
                    "f_peak_initial_Hz": fpk,
                    "e_ref": eref,
                    "f_ref_hz": f_ref_hz,
                    "passes_e_ref_filter": int(passes_reference),
                    "orbital_period_days": period,
                    "n_orbits_to_merger": norbits,
                    "rp_over_rg": rprg,
                    "peters_applicability_label": label,
                    "selected_by_run": int(selected),
                }
            )


# -------------------------------------------------------------------------------------------------
# System summary report formatting and output
# -------------------------------------------------------------------------------------------------

def write_report(report, path):
    """Write a strict JSON system report with a trailing newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def encode_report_sample(sample):
    """Encode a completed sample compactly, reusing its final result when it also ends history."""
    encode = json.JSONEncoder(allow_nan=False, separators=(',', ':')).encode
    result = sample.get('result')
    encoded_result = encode(result)
    fields = []
    for key, value in sample.items():
        if key == 'result':
            encoded_value = encoded_result
        elif key == 'history' and isinstance(value, list):
            levels = [
                encoded_result if level is result or (index == len(value) - 1 and level == result)
                else encode(level)
                for index, level in enumerate(value)
            ]
            encoded_value = '[' + ','.join(levels) + ']'
        else:
            encoded_value = encode(value)
        fields.append(encode(key) + ':' + encoded_value)
    return '{' + ','.join(fields) + '}'


def save_report(report, path, *, serialized_samples=None):
    """Atomically write a report, optionally reusing encoded immutable completed samples.

    Serialized samples must correspond to the report's sample list in the same order. Keep all
    report fields and histories in ordinary JSON; compact sample records reduce file size.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if serialized_samples is None:
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    else:
        if len(serialized_samples) != len(report['samples']):
            raise ValueError('Serialized samples must match the report sample count.')
        metadata = {key: value for key, value in report.items() if key != 'samples'}
        prefix = json.dumps(metadata, indent=2, allow_nan=False)[:-1].rstrip()
        with temporary.open('w') as stream:
            stream.write(prefix + (',\n' if metadata else '\n') + '  "samples": [')
            for index, encoded_sample in enumerate(serialized_samples):
                stream.write((',' if index else '') + '\n    ')
                stream.write(encoded_sample)
            stream.write('\n  ]\n}\n')
    temporary.replace(path)


def _report_tolerance_flags(report):
    """Read the convergence checks recorded by the current report format."""
    checks = report['convergence_checks']
    return checks['bulk'], checks['endpoints']


def format_report(report):
    import triple_core as core

    def number(value):
        return 'undefined' if value is None else f'{value:.7g}'

    def bounds(values):
        return 'none' if values is None else f'[{number(values[0])}, {number(values[1])}]'

    show_bulk, show_endpoints = _report_tolerance_flags(report)
    final = report['final']
    system = core.TripleSystem(**report['system'])
    settings = report['settings']

    config = report['configuration']
    inner_separation = core.inner_separation_for_stability_au(
        system.m1_inner, system.m2_inner, core.AnalysisSettings(**settings))
    rows = [
        ('System', system.describe()),
        ('Numerical status', report['numerical_status']),
        ('Stop reason', report['stop_reason']),
        ('Domain', f'a_out={report["domain"]["a_out_au"]} AU; '
         f'e_out={report["domain"]["e_out"]}; phase={report["domain"]["phase_deg"]} deg'),
        ('Reference frequency', f'{config["f_ref_hz"]:g} Hz'),
        ('Reference eccentricity', f'[{settings["e_ref_min"]}, '
         f'{settings["e_ref_max"]}] (None means no bound)'),
        ('Recoil ceiling', f'{settings["recoil_ceiling_kms"]:g} km/s'),
        ('Stability required', str(settings['require_stability'])),
        ('Inner separation', f'{inner_separation:.7g} AU'),
        ('Peters cuts', f'N_orb >= {config["min_peters_orbits"]:g}, '
         f'r_p/r_g >= {config["min_peters_rp_rg"]:g}'),
        ('Final resolution', ', '.join(
            f'{key}={value}' for key, value in final['resolution'].items())),
    ]
    if show_bulk:
        rows.append(('Convergence tolerances',
            f'R_sec absolute={config["convergence_r_sec_atol"]:g}; '
            f'support coverage absolute={config["convergence_coverage_atol"]:g}; '
            f'log10(volume fraction)={config["convergence_log_volume_atol"]:g} dex'))
    if show_endpoints:
        rows.append(('Endpoint tolerances',
            f'e_out={config["convergence_endpoint_eout_atol"]:g}; '
            f'log10(a_out)={config["convergence_endpoint_loga_atol"]:g} dex; '
            f'phase={config["convergence_endpoint_phase_atol_deg"]:g} deg'))
    rows.extend([
        ('Projection filter', f'at least {config["projection_min_transverse_cells"]} cells and '
         f'transverse fraction >= {config["projection_min_transverse_fraction"]:g}'),
        ('Peters reliability', final['reliability_verdict']),
        ('R_sec', number(final['r_sec'])),
        ('R_sec threshold', f'{settings["r_sec_threshold"]:g}'),
    ])
    label_width = max(len(label) for label, _ in rows)
    lines = ['System summary', '==============', '']
    lines.extend(f'{label + ":":<{label_width + 1}}  {value}' for label, value in rows)

    for family, stats in final['families'].items():
        strength = '+infinity' if stats['strength_is_infinite'] else number(stats['strength'])
        classification = ('Peters-incompatible'
                          if stats['constraint_class'] == 'peters-incompatible'
                          else stats['constraint_class'])
        family_lines = [
            f'  Class: {classification}',
            f'  Admitted targets: {final["targets"][family]["count"]}',
            f'  Post-merger eccentricity range: {bounds(final["targets"][family]["e_post_range"])}',
            f'  Reference eccentricity range: {bounds(final["targets"][family]["e_ref_range"])}',
            f'  Compatible / physical cells: {stats["compatible_cells"]} / '
            f'{stats["physical_cells"]}',
            f'  Reachable physical cells: {stats["reachable_physical_cells"]}',
            f'  Volume fraction (V): {number(stats["volume_fraction"])}',
            f'  Strength (S=-log10(V)): {strength}',
            f'  Minimum required recoil: {number(stats["minimum_recoil_kms"])} km/s',
        ]
        for axis, values in stats['projections'].items():
            family_lines.extend([
                '', '  ' + {'e_out': 'Outer eccentricity',
                             'a_out': 'Outer separation', 'phase': 'Orbital phase'}[axis] + ':',
                f'    C_support: {values["support_coverage"]:.7g}',
                f'    C_envelope: {values["envelope_coverage"]:.7g}',
            ])
            axis_units = 'AU' if axis == 'a_out' else values['units'].replace('au', 'AU')
            unit_label = '' if axis == 'e_out' else f' ({axis_units})'

            def coordinate(value):
                return 10**value if axis == 'a_out' else value

            def interval_text(intervals):
                return ', '.join(
                    f'[{coordinate(lo):.7g}, {coordinate(hi):.7g}]'
                    for lo, hi in intervals) or 'none'

            formatted = interval_text(values['support_intervals'])
            label = ('Gap-checked support'
                     if (values.get('gap_validation')
                         and report['configuration'].get('validate_gaps'))
                     else 'Filtered support')
            family_lines.append(f'    {label}{unit_label}: {formatted}')
            if values.get('undecided_intervals'):
                undecided = interval_text(values['undecided_intervals'])
                family_lines.append(f'    Undecided at this resolution{unit_label}: {undecided}; '
                    f'coverage={values.get("undecided_coverage", 0.):.7g}')
            if values.get('endpoint_uncertainty'):
                brackets = interval_text(values['endpoint_uncertainty'])
                family_lines.append(f'    Endpoint uncertainty brackets{unit_label}: {brackets}')
            envelope = values['envelope']
            envelope_text = (
                f'[{coordinate(envelope["start"]):.7g}, {coordinate(envelope["end"]):.7g}]'
                + (' across the phase seam' if envelope['wraps'] else '')
            ) if envelope else 'none'
            family_lines.append(f'    Shortest envelope{unit_label}: {envelope_text}')
        family_width = max(label_width + 1, max(
            len(line.partition(': ')[0]) for line in family_lines if ': ' in line))
        heading = {'all_targets': 'All targets', 'secular_targets': 'Secular targets'}[family]
        lines.extend(['', heading, '-' * len(heading), ''])
        for line in family_lines:
            label, separator, value = line.partition(': ')
            lines.append(f'{label + ":":<{family_width + 1}}  {value}' if separator else line)
    boundary_warnings = [warning for warning in final['warnings']
                         if ': support reaches ' in warning and 'search edge' in warning]
    if boundary_warnings:
        lines.append('')
        lines.extend('Note: ' + warning for warning in boundary_warnings)
    return '\n'.join(lines)


def matching_quality_warnings(matching):
    """Flag matching ESS below 20 or a single proposal carrying more than 10% of the weight."""
    warnings = []
    for key, label in (
        ('importance_effective_sample_size', 'Proposal ESS'),
        ('first_event_effective_sample_size', 'First-event row ESS'),
        ('second_event_effective_sample_size', 'Second-event row ESS'),
    ):
        value = matching.get(key)
        if value is not None and value < 20:
            warnings.append(f'{label} is {value:.6g}, below the diagnostic threshold of 20.')
    maximum_weight = matching.get('maximum_proposal_weight')
    if maximum_weight is not None and maximum_weight > 0.1:
        warnings.append(
            f'One proposal carries {maximum_weight:.1%} of the matching weight '
            '(diagnostic threshold: 10%).'
        )
    return warnings


def format_posterior_report(report):
    summary = report['summary']
    r_sec = summary['r_sec']

    def number(value):
        return 'undefined' if value is None else f'{value:.6g}'

    def interval(distribution):
        quantiles = distribution['quantiles']
        if quantiles is None:
            return 'undefined'
        return (f'{quantiles["median"]:.6g} '
                f'[{quantiles["q05"]:.6g}, {quantiles["q95"]:.6g}]')

    def readable(name):
        if name == 'peters-incompatible':
            return 'Peters-incompatible'
        return name.replace('_', ' ').capitalize()

    rows = [
        ('Prior correction', 'Applied' if report['matching']['mode'] == 'prior-corrected'
         else 'Not applied'),
        ('Numerical status', readable(report['numerical_status'])),
    ]
    rows.extend((f'{readable(status)} weight', number(weight))
                for status, weight in summary['numerical_status_weights'].items())
    rows.extend([
        ('R_sec median [5%, 95%]', interval(r_sec)),
        *([('R_sec undefined weight', number(r_sec['undefined_weight']))]
          if r_sec['undefined_weight'] > 0 else []),
        ('R_sec threshold', number(r_sec['threshold'])),
        ('Weight at or above threshold', number(r_sec['weight_at_or_above_threshold'])),
        ('Fraction among defined R_sec', number(r_sec['fraction_at_or_above_given_defined'])),
        ('Converged R_sec median [5%, 95%]', interval(summary['r_sec_converged'])),
        ('Converged weight above threshold',
         number(summary['r_sec_converged']['weight_at_or_above_threshold'])),
    ])
    sections = [('Posterior summary', rows)]
    matching_rows = [
        (label, number(report['matching'][key]))
        for key, label in (
            ('importance_effective_sample_size', 'Proposal ESS'),
            ('first_event_effective_sample_size', 'First-event row ESS'),
            ('second_event_effective_sample_size', 'Second-event row ESS'),
            ('unique_selected_pairs', 'Unique selected pairs'),
            ('maximum_proposal_weight', 'Maximum proposal weight'),
        )
        if key in report['matching']
    ]
    if matching_rows:
        sections.append(('Posterior matching quality', matching_rows))
    if summary.get('gap_validation'):
        gaps = summary['gap_validation']
        sections.append(('Gap validation', [
            ('Checked systems', gaps['checked_samples']),
            ('Recovered bins', gaps['recovered_bins']),
            ('Localized boundary bins', gaps['boundary_bins']),
            ('Unresolved bins', gaps['unresolved_bins']),
            ('Unresolved sample weight', number(gaps['unresolved_sample_weight'])),
        ]))
    if summary.get('endpoint_refinement', {}).get('checked_samples'):
        local = summary['endpoint_refinement']
        sections.append(('Local endpoint/gap refinement', [
            ('Checked systems', local['checked_samples']),
            ('Unresolved windows', local['unresolved_windows']),
            ('Scan-budget points used', f'{local["points_used"]:,}'),
        ]))
    for family, metrics in summary['families'].items():
        rows = [
            ('Volume fraction median [5%, 95%]', interval(metrics['volume_fraction'])),
            ('Minimum recoil median [5%, 95%]', interval(metrics['minimum_recoil_kms']) + ' km/s'),
        ]
        class_counts = {}
        for sample in report['samples']:
            stats = sample.get('result', {}).get('families', {}).get(family)
            name = stats['constraint_class'] if stats else 'evaluation_failed'
            class_counts[name] = class_counts.get(name, 0) + sample['multiplicity']
        total = report['configuration']['posterior_samples']
        rows.extend((readable(name), f'{class_counts[name]}/{total} ({100 * weight:.1f}%)')
                    for name, weight in metrics['sampled_class_weights'].items())
        for axis, label in (
                ('e_out', 'Outer eccentricity support union'),
                ('a_out', 'Outer separation support union (AU)'),
                ('phase', 'Orbital phase support union (deg)')):
            intervals = metrics['projections'][axis]['support_union']
            if axis == 'a_out':
                intervals = [(10**lo, 10**hi) for lo, hi in intervals]
            text = ' ∪ '.join(f'[{lo:.7g}, {hi:.7g}]' for lo, hi in intervals) or 'none'
            rows.append((label, text))
        sections.append((readable(family), rows))

    width = max(len(label) for _, rows in sections for label, _ in rows) + 1
    lines = []
    for heading, rows in sections:
        if lines:
            lines.append('')
        lines.extend([heading, '-' * len(heading), ''])
        lines.extend(f'{label + ":":<{width}}  {value}' for label, value in rows)
    warnings = matching_quality_warnings(report['matching'])
    if warnings:
        lines.extend(['', 'Matching-quality warnings (separate from grid convergence):'])
        lines.extend('Warning: ' + warning for warning in warnings)
        lines.append(
            'Inspect overlap and matching widths; more final draws alone do not improve ESS.'
        )
    return '\n'.join(lines)
