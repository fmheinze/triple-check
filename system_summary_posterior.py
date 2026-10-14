"""Propagate paired gravitational-wave event posteriors through the system analysis.

Read and validate event samples, approximately match the first event's remnant to one component of
the second event by mass and, optionally, distance, and evaluate hierarchical-triple feasibility
for each distinct sampled pair. Matching supports prior correction or an explicitly selected
overlap-only mode.

Combine the weighted results into distributions of recoil, support, and Peters reliability,
retaining each sample's numerical status. Evaluate samples in parallel when requested and save
progress after each completed sample to support resuming. Dynamics use detector-frame masses,
times and frequencies, and redshifted lengths.
"""
from collections import Counter
from copy import copy
import math
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

from triple_io import (
    read_samples, _load_prior, save_report, encode_report_sample, matching_quality_warnings,
)
from triple_core import completed_tasks
from triple_cli import validate_system_summary_args
from support_decision import interval_union


PRIOR_KDE_MAX_SAMPLES = 5000

FAMILIES = ('all_targets', 'secular_targets')
AXES = ('e_out', 'a_out', 'phase')


# -------------------------------------------------------------------------------------------------
# Matching helpers
# -------------------------------------------------------------------------------------------------

def _finite_positive(values, name, allow_zero=False):
    invalid = ~np.isfinite(values) | (values < 0 if allow_zero else values <= 0)
    if np.any(invalid):
        raise ValueError(
            f'{name}: {int(invalid.sum())} invalid rows; '
            'repair/filter inputs explicitly before propagation.'
        )


def _features(mass, luminosity_distance=None):
    """Build matching columns from masses and optional log luminosity distances."""
    _finite_positive(mass, 'detector mass')
    features = [mass]
    if luminosity_distance is not None:
        _finite_positive(luminosity_distance, 'luminosity_distance')
        features.append(np.log(luminosity_distance))
    return np.column_stack(features)


def _weighted_variance(values, weights):
    """Return each column's population variance using normalized sample weights."""
    return np.sum(
        weights[:, None] * (values - np.sum(weights[:, None] * values, axis=0)) ** 2,
        axis=0,
    )


# -------------------------------------------------------------------------------------------------
# Sample pairing and preparation
# -------------------------------------------------------------------------------------------------

def kernel_pair_draws(first, second, args, rng):
    """Draw event-sample pairs whose remnant/component masses and optional distances agree.

    Use the two posterior tables, matching settings, and random generator to propose pairs and
    weight their agreement with a finite-width Gaussian kernel. Reject distant or incorrectly
    ordered pairs and, when requested, divide their weights by an estimate of the second event's
    marginal prior density.

    Return a Counter of selected (first-row, second-row) pairs and their draw counts, plus
    diagnostics describing bandwidths, sample diversity, and matching residuals.
    """
    # Prepare matching coordinates and posterior sample weights.
    component = 'mass_1' if args.remnant_component == 'primary' else 'mass_2'
    match_distance = args.match_parameters == 'mass-distance'
    first_features = _features(
        first.detector_mass('final_mass'),
        luminosity_distance=first.get('luminosity_distance') if match_distance else None,
    )
    second_features = _features(
        second.detector_mass(component),
        luminosity_distance=second.get('luminosity_distance') if match_distance else None,
    )
    first_weights, second_weights = first.weights(), second.weights()

    # Estimate matching bandwidths, applying explicit widths when supplied
    dimensions = first_features.shape[1]
    effective_samples = min(1 / np.sum(first_weights ** 2), 1 / np.sum(second_weights ** 2))
    scale = np.sqrt(0.5 * (
        _weighted_variance(first_features, first_weights)
        + _weighted_variance(second_features, second_weights)
    ))
    bandwidth = scale * effective_samples ** (-1 / (dimensions + 4))
    if args.match_mass_width is not None:
        bandwidth[0] = args.match_mass_width
    if dimensions == 2 and args.match_log_distance_width is not None:
        bandwidth[1] = args.match_log_distance_width
    _finite_positive(bandwidth, 'matching bandwidth (supply explicit widths for constant samples)')

    # Propose pairs and reject excessive mismatches or inconsistent mass ordering
    first_indices = rng.choice(first.size, args.posterior_proposals, p=first_weights)
    second_indices = rng.choice(second.size, args.posterior_proposals, p=second_weights)
    squared_distance = np.sum((
        (first_features[first_indices] - second_features[second_indices]) / bandwidth
    ) ** 2, axis=1)
    log_weights = -0.5 * squared_distance
    # The cutoff prevents normalization from promoting remote Gaussian tails
    log_weights[squared_distance > 16] = -np.inf
    tertiary_mass = second.detector_mass('mass_2' if component == 'mass_1' else 'mass_1')
    correctly_ordered = (
        first_features[first_indices, 0] >= tertiary_mass[second_indices]
        if component == 'mass_1'
        else first_features[first_indices, 0] <= tertiary_mass[second_indices]
    )
    log_weights[~correctly_ordered] = -np.inf

    # Optionally correct pair weights for the second event's marginal prior
    prior_info = None
    if args.pairing_mode == 'prior-corrected':
        prior = _load_prior(args, second)
        prior_features = _features(
            prior.detector_mass(component),
            luminosity_distance=prior.get('luminosity_distance') if match_distance else None,
        )
        prior_weights = prior.weights()
        if len(prior_features) > PRIOR_KDE_MAX_SAMPLES:
            prior_features = prior_features[
                rng.choice(len(prior_features), PRIOR_KDE_MAX_SAMPLES, p=prior_weights)
            ]
            prior_weights = np.full(len(prior_features), 1 / len(prior_features))
        try:
            prior_kde = gaussian_kde(prior_features.T, weights=prior_weights)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                'Second-event marginal prior KDE is singular; supply a sufficiently sampled prior.'
            ) from exc
        eligible = np.isfinite(log_weights)
        if np.any(eligible):
            unique_second_indices, inverse_indices = np.unique(
                second_indices[eligible], return_inverse=True
            )
            log_prior_density = prior_kde.logpdf(
                second_features[unique_second_indices].T
            )[inverse_indices]
            if not np.all(np.isfinite(log_prior_density)):
                raise ValueError('Nonfinite marginal prior density; cannot correct pair weights.')
            log_weights[eligible] -= log_prior_density
        prior_info = dict(
            prior.provenance, kde_samples=len(prior_features), kde_factor=float(prior_kde.factor),
            coordinates=['detector_mass_msun', 'ln_distance_mpc'][:dimensions],
        )

    # Normalize surviving proposal weights and draw the requested sample pairs
    supported_proposals = np.isfinite(log_weights)
    if not np.any(supported_proposals):
        raise ValueError(
            'No pair proposals overlap inside the 4-bandwidth matching ellipsoid. '
            'Increase proposal count or inspect the data/assignment and bandwidth; '
            'no constraints were inferred.'
        )
    pair_weights = np.exp(log_weights - logsumexp(log_weights))
    pair_effective_samples = float(1 / np.sum(pair_weights ** 2))
    selected_proposals = rng.choice(len(first_indices), args.posterior_samples, p=pair_weights)
    counts = Counter(
        (int(first_indices[index]), int(second_indices[index])) for index in selected_proposals
    )

    # Report matching quality and concentration of weight on individual event rows
    first_row_weights = np.bincount(first_indices, weights=pair_weights, minlength=first.size)
    second_row_weights = np.bincount(second_indices, weights=pair_weights, minlength=second.size)
    diagnostics = {
        'mode': args.pairing_mode,
        'shared_parameters': args.match_parameters,
        'bandwidth': bandwidth.tolist(),
        'kernel_cutoff_mahalanobis': 4,
        'proposal_count': len(first_indices),
        'supported_proposals': int(supported_proposals.sum()),
        'importance_effective_sample_size': pair_effective_samples,
        'first_event_effective_sample_size': float(1 / np.sum(first_row_weights ** 2)),
        'second_event_effective_sample_size': float(1 / np.sum(second_row_weights ** 2)),
        'maximum_proposal_weight': float(pair_weights.max()),
        'unique_selected_pairs': len(counts),
        'weighted_rms_matching_residual': np.sqrt(np.sum(
            pair_weights[:, None]
            * (first_features[first_indices] - second_features[second_indices]) ** 2,
            axis=0,
        )).tolist(),
        'prior': prior_info,
    }
    return counts, diagnostics


def prepare_samples(args):
    """Build weighted dynamics inputs from matched posterior samples of two events.

    Read and validate the event tables, draw sample pairs, and assemble detector-frame masses,
    observed delays, and reference-eccentricity ranges. Preserve correlations within each event row
    and combine duplicate pairs using their draw multiplicities. Return the prepared samples,
    source provenance, and matching diagnostics.
    """
    # Load event tables and initialize reproducible sampling
    rng = np.random.default_rng(args.posterior_seed)
    first = read_samples(args.posterior_first, args.first_dataset)
    second = read_samples(args.posterior_second, args.second_dataset)
    if first.get('final_mass', False) is None and first.get('final_mass_source', False) is None:
        raise ValueError(
            'First-event file lacks remnant masses. Use a posterior release with '
            'final_mass or final_mass_source samples.'
        )

    # Validate masses and, when needed, event times before costly matching
    inner_primary_mass, inner_secondary_mass, remnant_mass = [
        first.detector_mass(name) for name in ('mass_1', 'mass_2', 'final_mass')
    ]
    tertiary_mass = second.detector_mass(
        'mass_2' if args.remnant_component == 'primary' else 'mass_1'
    )
    for name, values in (
        ('m1_inner', inner_primary_mass), ('m2_inner', inner_secondary_mass),
        ('mrem', remnant_mass), ('m3', tertiary_mass),
    ):
        _finite_positive(values, name)
    if np.any(remnant_mass > inner_primary_mass + inner_secondary_mass):
        raise ValueError('First-event remnant mass exceeds the inner total mass in some rows.')
    event_times = None
    if args.posterior_delay_days is None:
        event_times = (first.get('geocent_time'), second.get('geocent_time'))
        if not all(np.all(np.isfinite(times)) for times in event_times):
            raise ValueError('Nonfinite geocent_time samples; supply --posterior-delay-days.')

    # Select matched rows and calculate their observed merger delays
    pair_counts, diagnostics = kernel_pair_draws(first, second, args, rng)
    pair_indices = np.array(list(pair_counts), dtype=int)
    first_indices, second_indices = pair_indices[:, 0], pair_indices[:, 1]
    input_columns = dict(
        m1_inner=inner_primary_mass[first_indices],
        m2_inner=inner_secondary_mass[first_indices],
        mrem=remnant_mass[first_indices],
        m3=tertiary_mass[second_indices],
    )
    if event_times is None:
        delays_days = np.full(len(pair_indices), args.posterior_delay_days)
    else:
        delays_days = (
            event_times[1][second_indices] - event_times[0][first_indices]
        ) / 86400
    _finite_positive(delays_days, 'delay_days')

    # Package each distinct pair with its multiplicity weight and provenance
    sources = {'first': first.provenance, 'second': second.provenance}
    samples = []
    for index, ((first_row, second_row), count) in enumerate(pair_counts.items()):
        inputs = {name: float(column[index]) for name, column in input_columns.items()}
        inputs['delay_days'] = float(delays_days[index])
        eccentricity_range = [args.e_ref_min, args.e_ref_max]
        samples.append(dict(
            rows={'first_row': first_row, 'second_row': second_row},
            multiplicity=count,
            weight=count / args.posterior_samples,
            inputs=inputs,
            e_ref_range=eccentricity_range,
        ))
    return samples, sources, diagnostics


# -------------------------------------------------------------------------------------------------
# Weighted summaries
# -------------------------------------------------------------------------------------------------

def distribution(values, weights, threshold=None, histogram=False):
    """Summarize weighted values, keeping defined and undefined weight separate.

    Compute the mean and 5th, 50th, and 95th percentiles after normalizing weights over finite
    values only. None and nonfinite values contribute to undefined weight. Optionally report weight
    at or above a threshold and its fraction among defined values, or a 20-bin histogram spanning
    [0, 1] with unnormalized input weights. These histograms summarize R_sec across sampled systems
    The plotting code uses the saved all-sample histogram for its reliability panel. Weights must
    be finite, nonnegative, and aligned with the values; their sum need not be one.
    """
    w = np.asarray(weights, dtype=float)
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    valid = np.isfinite(v)
    mass = float(w[valid].sum())
    result = {'defined_weight': mass, 'undefined_weight': float(w[~valid].sum()),
              'quantiles': None, 'mean': None}
    if mass > 0:
        x, p = v[valid], w[valid] / mass
        order = np.argsort(x)
        cdf = np.cumsum(p[order])
        indexes = np.minimum(np.searchsorted(cdf, [.05, .5, .95]), len(x) - 1)
        result.update(quantiles=dict(zip(('q05', 'median', 'q95'), map(float, x[order][indexes]))),
                      mean=float(np.sum(x * p)))
    if threshold is not None:
        above = float(w[valid & (v >= threshold)].sum())
        result.update(threshold=threshold, weight_at_or_above_threshold=above,
                      fraction_at_or_above_given_defined=above / mass if mass else None)
    if histogram:
        edges = np.linspace(0, 1, 21)
        counts = np.histogram(v[valid], bins=edges, weights=w[valid])[0]
        result['histogram'] = {'edges': edges.tolist(), 'weight_per_bin': counts.tolist()}
    return result


def _support_occupancy(intervals, edges):
    """Fraction of each common bin covered by the (disjoint) allowed intervals."""
    coverage = np.zeros(len(edges) - 1)
    for lo, hi in intervals:
        coverage += np.maximum(
            0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)
        ) / np.diff(edges)
    return np.clip(coverage, 0, 1)


def aggregate(samples, args):
    """Combine evaluated sample pairs into weighted statistics and projected support.

    Summarize numerical statuses, reliability, refinement diagnostics, and both target families on
    common projection bins. Preserve missing-result weight and report support from converged,
    sufficiently reliable samples separately. Return the summary used in the posterior JSON report
    and plots without renormalizing sample weights.
    """
    # Summarize numerical status and reliability for all and converged samples
    sample_weights = [sample['weight'] for sample in samples]
    reliability_values = [sample.get('result', {}).get('r_sec') for sample in samples]
    converged_weights = [
        sample['weight'] if sample['numerical_status'] == 'converged' else 0.
        for sample in samples
    ]
    status_weights = {}
    for sample in samples:
        status = sample['numerical_status']
        status_weights[status] = status_weights.get(status, 0.) + sample['weight']
    result = {
        'numerical_status_weights': status_weights,
        'r_sec': distribution(reliability_values, sample_weights, args.r_sec_threshold, True),
        'r_sec_converged': distribution(
            reliability_values, converged_weights, args.r_sec_threshold, True
        ),
        'families': {},
    }

    # Collect gap-validation and endpoint-refinement diagnostics
    gap_reports = [sample.get('result', {}).get('gap_validation') for sample in samples]
    result['gap_validation'] = {
        'checked_samples': sum(bool(report and report['enabled']) for report in gap_reports),
        'recovered_bins': sum(report['recovered_bins'] for report in gap_reports if report),
        'boundary_bins': sum(report.get('boundary_bins', 0) for report in gap_reports if report),
        'unresolved_bins': sum(report['unresolved_bins'] for report in gap_reports if report),
        'unresolved_sample_weight': sum(
            sample['weight'] for sample, report in zip(samples, gap_reports)
            if report and report['unresolved_bins']
        ),
    }
    refinement_reports = [
        sample.get('result', {}).get('endpoint_refinement') for sample in samples
    ]
    result['endpoint_refinement'] = {
        'checked_samples': sum(report is not None for report in refinement_reports),
        'points_used': sum(report['points_used'] for report in refinement_reports if report),
        'point_budget_unit': 'initial_scan_points',
        'unresolved_windows': sum(
            report['unresolved_windows'] for report in refinement_reports if report
        ),
        'unresolved_sample_weight': sum(
            sample['weight'] for sample, report in zip(samples, refinement_reports)
            if report and report['unresolved_windows']
        ),
    }

    # Define common projection domains and summarize each target family
    bounds = {
        'e_out': [args.eout_min, args.eout_max],
        'a_out': [np.log10(args.aout_min_au), np.log10(args.aout_max_au)],
        'phase': [args.nu_min_deg, args.nu_max_deg],
    }
    for family in FAMILIES:
        family_results = [
            sample.get('result', {}).get('families', {}).get(family) for sample in samples
        ]
        family_report = {
            metric: distribution(
                [stats[metric] if stats else None for stats in family_results], sample_weights
            )
            for metric in ('volume_fraction', 'minimum_recoil_kms', 'strength')
        }
        family_report['zero_volume_weight'] = sum(
            weight for stats, weight in zip(family_results, sample_weights)
            if stats and stats['volume_fraction'] == 0
        )
        family_report['sampled_class_weights'] = {}
        family_report['projections'] = {}
        for stats, weight in zip(family_results, sample_weights):
            classification = stats['constraint_class'] if stats else 'evaluation_failed'
            family_report['sampled_class_weights'][classification] = (
                family_report['sampled_class_weights'].get(classification, 0.) + weight
            )

        # Accumulate support on each axis, retaining the original sample weights
        for axis in AXES:
            edges = np.linspace(*bounds[axis], args.posterior_plot_projection_bins + 1)
            support_probability = np.zeros(len(edges) - 1)
            reliable_support_probability = np.zeros(len(edges) - 1)
            reliable_sample_weight = 0.
            for sample, stats in zip(samples, family_results):
                if stats is None:
                    continue
                bin_coverage = _support_occupancy(
                    stats['projections'][axis]['support_intervals'], edges
                )
                support_probability += sample['weight'] * bin_coverage
                reliability = sample['result']['r_sec']
                if (
                    sample['numerical_status'] == 'converged'
                    and reliability is not None
                    and reliability >= args.r_sec_threshold
                ):
                    reliable_support_probability += sample['weight'] * bin_coverage
                    reliable_sample_weight += sample['weight']

            # Store support curves, interval unions, and projection diagnostics
            projections = [
                stats['projections'][axis] if stats else None for stats in family_results
            ]
            family_report['projections'][axis] = {
                'edges': edges.tolist(),
                'units': {
                    'e_out': 'dimensionless',
                    'a_out': 'log10(detector-frame AU)',
                    'phase': 'deg',
                }[axis],
                'support_probability': support_probability.tolist(),
                'support_union': interval_union(
                    interval for sample, projection in zip(samples, projections)
                    if sample['weight'] > 0 and projection is not None
                    for interval in projection['support_intervals']
                ),
                'converged_reliable_joint_support_probability': (
                    reliable_support_probability.tolist()
                ),
                'converged_reliable_weight': reliable_sample_weight,
                'support_probability_given_converged_reliable': (
                    (reliable_support_probability / reliable_sample_weight).tolist()
                    if reliable_sample_weight else None
                ),
                **{
                    metric: distribution(
                        [projection[metric] if projection else None for projection in projections],
                        sample_weights,
                    )
                    for metric in ('support_coverage', 'envelope_coverage', 'support_width')
                },
                'touches_search_edge_weight': sum(
                    weight for projection, weight in zip(projections, sample_weights)
                    if projection
                    and (projection['touches_lower_edge'] or projection['touches_upper_edge'])
                    and not projection['circular']
                ),
            }
        result['families'][family] = family_report
    return result


# -------------------------------------------------------------------------------------------------
# Evaluation and resuming
# -------------------------------------------------------------------------------------------------

def _evaluate_posterior_sample(task, progress=None):
    """Run the system analysis for one selected event pair and return its updated record.

    Unpack the task's sample index, original sample, run settings, and evaluator. Copy the sample
    and settings, insert its masses, delay, and eccentricity range, then validate the settings and
    evaluate the system. Use one grid worker because parallel execution is already managed across
    posterior samples.

    Attach the final result, numerical status, stopping reason, and refinement history. Record
    ValueError, FloatingPointError, and OverflowError from validation or evaluation as an
    evaluation_failed sample; other exceptions propagate. Return the sample index and updated
    sample record.
    """
    index, original, args, evaluator = task
    sample = dict(original)
    current = copy(args)
    current.workers = 1
    for key, value in sample['inputs'].items():
        setattr(current, key, value)
    current.e_ref_min, current.e_ref_max = sample['e_ref_range']
    current.e_ref_range = sample['e_ref_range']
    if progress:
        progress(f'Sample {index + 1} (weight {sample["weight"]:.5g})')
    try:
        validate_system_summary_args(current)
        single = evaluator(current, progress=progress)
        sample.update(numerical_status=single['numerical_status'], result=single['final'],
                      stop_reason=single['stop_reason'], history=single['history'])
    except (ValueError, FloatingPointError, OverflowError) as exc:
        sample.update(numerical_status='evaluation_failed', error=str(exc))
    return index, sample


def restore_completed_draws(path, args, samples, sources, matching):
    """Validate a saved posterior checkpoint and return reusable samples by current index.

    Require matching analysis settings, source and prior provenance, and draw identities, inputs,
    and weights. Allow changes to execution/output options and file locations; compare provenance
    without paths so relocated inputs can still be recognized. Raise ValueError for incompatible
    checkpoints or draw records. Restore all recorded terminal statuses, including
    evaluation_failed, so failed draws are not retried.
    """
    # Load the checkpoint and verify its report format
    with Path(path).open() as stream:
        checkpoint = json.load(stream)
    if checkpoint.get('analysis_mode') != 'posterior' or checkpoint.get('schema_version') != 3:
        raise ValueError('Resume requires a version-3 posterior checkpoint.')

    # Compare analysis settings, allowing execution, output, and path changes
    current_configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    ignored_settings = {
        'resume', 'workers', 'quiet', 'json_output', 'posterior_plot_projection_bins',
        'posterior_first', 'posterior_second', 'second_prior',
    }
    saved_configuration = checkpoint.get('configuration', {})
    changed_settings = sorted(
        key for key in set(saved_configuration) | set(current_configuration)
        if key not in ignored_settings
        and saved_configuration.get(key) != current_configuration.get(key)
    )
    if changed_settings:
        raise ValueError('Resume settings differ: ' + ', '.join(changed_settings))

    # Verify input provenance and matching diagnostics independently of file locations
    def without_paths(value):
        if isinstance(value, dict):
            return {key: without_paths(item) for key, item in value.items() if key != 'path'}
        if isinstance(value, list):
            return [without_paths(item) for item in value]
        return value

    if without_paths(checkpoint.get('sources')) != without_paths(sources):
        raise ValueError('Resume posterior input provenance differs.')
    if without_paths(checkpoint.get('matching')) != without_paths(matching):
        raise ValueError('Resume matching or prior provenance differs.')

    # Map saved event-row pairs to the draws prepared for the current run
    def draw_identity(sample):
        return tuple(sorted(sample['rows'].items()))

    expected_draws = {
        draw_identity(sample): (index, sample) for index, sample in enumerate(samples)
    }
    completed_draws = {}
    for saved_sample in checkpoint.get('samples', []):
        expected_entry = expected_draws.get(draw_identity(saved_sample))
        if expected_entry is None:
            raise ValueError('Checkpoint contains an unexpected posterior draw.')
        index, expected_sample = expected_entry
        if index in completed_draws or any(
            saved_sample.get(key) != value for key, value in expected_sample.items()
        ):
            raise ValueError('Checkpoint draw identity, inputs, or weights differ.')

        # Require a terminal status and results for every nonfailed draw
        status = saved_sample.get('numerical_status')
        if status not in ('converged', 'not_converged', 'fixed_resolution', 'evaluation_failed'):
            raise ValueError('Checkpoint contains an invalid draw status.')
        if status != 'evaluation_failed' and not all(
            key in saved_sample for key in ('result', 'history')
        ):
            raise ValueError('Checkpoint is missing completed draw results.')
        completed_draws[index] = saved_sample
    return completed_draws


def _update_posterior_report(report, completed_samples, total_samples, args):
    """Refresh ordered samples, completion, comparison counts, and the weighted summary."""
    # Worker completion order must not change sample ordering or weighted sums
    report['samples'] = [completed_samples[index] for index in sorted(completed_samples)]
    report['complete'] = len(completed_samples) == total_samples
    report['convergence_comparisons'] = sum(
        level.get('comparison_level') is not None
        for sample in report['samples'] for level in sample.get('history', [])
    )
    report['summary'] = aggregate(report['samples'], args)


def run_posterior_analysis(args, progress=None, evaluator=None):
    """Evaluate matched event pairs, checkpoint progress, and return the posterior report.

    Prepare weighted samples, optionally restore completed draws, and evaluate remaining pairs
    serially or across workers. Rebuild and save the summary after each completed pair, including
    handled failures, without renormalizing partial sample weights. Finally assign the overall
    numerical status and save the finished report.
    """
    from system_summary import run_analysis, convergence_checks

    # Prepare sample pairs and initialize report metadata
    if evaluator is None:
        evaluator = run_analysis
    samples, sources, matching = prepare_samples(args)
    if progress:
        progress(
            f'Propagating {args.posterior_samples} draws '
            f'({len(samples)} distinct sample pairs/rows).'
        )
        progress(f'Using {min(args.workers, len(samples))} worker process(es) across samples.')
        for warning in matching_quality_warnings(matching):
            progress('Matching-quality warning: ' + warning)
    report = {
        'schema_version': 3,
        'analysis_mode': 'posterior',
        'sources': sources,
        'frame': (
            'detector: masses redshifted, delays/frequencies observed; AU lengths are redshifted'
        ),
        'configuration': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'matching': matching,
        'samples': [],
        'convergence_checks': convergence_checks(args),
        'convergence_comparisons': 0,
    }

    # Restore validated draws and schedule only the remaining pairs
    resume_requested = getattr(args, 'resume', False)
    completed_samples = (
        restore_completed_draws(args.json_output, args, samples, sources, matching)
        if resume_requested else {}
    )
    # Completed records are immutable; serialize each only once, including restored samples
    encoded_samples = {
        index: encode_report_sample(sample) for index, sample in completed_samples.items()
    }
    if progress and resume_requested:
        progress(
            f'Restored {len(completed_samples)}/{len(samples)} completed draws; '
            f'evaluating {len(samples) - len(completed_samples)} remaining draws.'
        )
    tasks = [
        (index, sample, args, evaluator) for index, sample in enumerate(samples)
        if index not in completed_samples
    ]
    if args.workers == 1:
        results = (_evaluate_posterior_sample(task, progress) for task in tasks)
    else:
        results = completed_tasks(_evaluate_posterior_sample, tasks, args.workers, progress)

    # Save a checkpoint after each completed pair, including handled failures
    for index, sample in results:
        completed_samples[index] = sample
        encoded_samples[index] = encode_report_sample(sample)
        if progress and args.workers > 1:
            progress(
                f'Completed {len(completed_samples)}/{len(samples)}: sample {index + 1}, '
                f'{sample["numerical_status"]} (weight {sample["weight"]:.5g})'
            )
        _update_posterior_report(report, completed_samples, len(samples), args)
        save_report(report, args.json_output, serialized_samples=[
            encoded_samples[index] for index in sorted(completed_samples)
        ])

    # Finalize the report, including runs restored entirely from a checkpoint
    _update_posterior_report(report, completed_samples, len(samples), args)
    if any(
        sample['numerical_status'] in ('not_converged', 'evaluation_failed')
        for sample in report['samples']
    ):
        report['numerical_status'] = 'not_converged'
    elif all(sample['numerical_status'] == 'converged' for sample in report['samples']):
        report['numerical_status'] = 'converged'
    else:
        report['numerical_status'] = 'fixed_resolution'
    save_report(report, args.json_output, serialized_samples=[
        encoded_samples[index] for index in sorted(completed_samples)
    ])
    return report
