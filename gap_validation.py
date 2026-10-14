"""Validate gaps and locate boundaries in projected orbital support.

Evaluate local transverse probes at increasing estimator resolutions and classify each as
supported, excluded, or undecided. Compare successive rounds, refine support transitions, and
retain probe histories so checks can resume within a shared point budget.

Combine completed checks with the grid projections and update support geometry, uncertainty
brackets, classifications, and diagnostics. Endpoint refinement reuses the same validation
workers and budget allocation helpers for its local windows.
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import math
import time
import numpy as np
import triple_core as core
import edge_tracking as et
import support_decision as decide
from triple_core import reusable_task_runner

AXES = ('e_out', 'a_out', 'phase')
RESOLVED = ('recovered', 'persistent_at_probes', 'boundary_resolved')


# -------------------------------------------------------------------------------------------------
# Support masks and probe configuration
# -------------------------------------------------------------------------------------------------

def support_mask(projection, bin_count):
    """Convert normalized support intervals into a Boolean mask with one entry per grid bin."""
    mask = np.zeros(bin_count, dtype=bool)
    for start_fraction, end_fraction in projection['normalized_intervals']:
        mask[round(start_fraction * bin_count) : round(end_fraction * bin_count)] = True
    return mask


def gap_runs(mask, circular=False):
    """Return unsupported bin groups enclosed by support, joining circular gaps across the seam."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or mask.all():
        return []
    transitions = np.diff(np.r_[False, ~mask, False].astype(int))
    gap_groups = [
        list(range(start_index, stop_index))
        for start_index, stop_index in zip(
            np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        )
    ]
    if circular:
        if len(gap_groups) > 1 and gap_groups[0][0] == 0 and gap_groups[-1][-1] == len(mask) - 1:
            gap_groups = [gap_groups[-1] + gap_groups[0]] + gap_groups[1:-1]
        return gap_groups
    return [gap_bins for gap_bins in gap_groups if gap_bins[0] > 0 and gap_bins[-1] < len(mask) - 1]


def probe_key(coordinate, descriptor):
    """Build a cache key from the probe coordinate and complete estimator descriptor."""
    items = tuple(
        sorted(
            (str(key), tuple(sorted(value.items())) if isinstance(value, dict) else value)
            for key, value in descriptor.items()
        )
    )
    return (float(coordinate), items)


def estimator_options(args):
    """Collect shared edge-tracking and support-classification settings from the run arguments."""
    return dict(
        method=et.EDGE_METHOD,
        scan_points=getattr(args, 'edge_tracking_scan', 512),
        support_sweeps=getattr(args, 'edge_tracking_sweeps', 2),
        column_growth=getattr(args, 'edge_tracking_growth', 2.0),
        sigma_factor=getattr(args, 'projection_support_sigma', 2.0),
        single_round_when_empty=getattr(args, 'gap_validation_single_round_when_empty', True),
        min_cells=args.projection_min_transverse_cells,
        min_fraction=args.projection_min_transverse_fraction,
    )


def descriptor_cost(descriptor):
    """Return the reserved point count for one probe using this estimator descriptor."""
    return et.estimated_probe_cost(descriptor['columns'], descriptor['scan'])


def estimator_plan(axis, sizes, level, options):
    """Return the estimator descriptor and reserved point count for a refinement level."""
    _, column = et.transverse_axes(axis)
    columns = max(2, int(math.ceil(sizes[column] * options.get('column_growth', 2.0) ** level)))
    scan = max(8, int(options.get('scan_points', 512)))
    descriptor = {
        'method': et.EDGE_METHOD,
        'columns': columns,
        'scan': scan,
        'bisection_rtol': options.get('bisection_rtol', et.BISECTION_RTOL),
        'bisection_iterations': options.get('bisection_iterations', 52),
        'support_sweeps': options.get('support_sweeps', 2),
    }
    return descriptor, descriptor_cost(descriptor)


def run_probe(
    axis, coordinate, descriptor, edges, targets, system, settings, options, probe, seed=None
):
    """Evaluate transverse occupancy with the selected estimator settings and optional seed."""
    return (probe or et.estimate_occupancy)(
        axis,
        coordinate,
        edges,
        targets,
        system,
        settings,
        descriptor['columns'],
        descriptor['scan'],
        seed=seed,
        bisection_iterations=descriptor.get('bisection_iterations', 52),
        bisection_rtol=descriptor.get('bisection_rtol', 0.0),
        support_sweeps=descriptor.get('support_sweeps', options.get('support_sweeps', 2)),
    )


# -------------------------------------------------------------------------------------------------
# Boundary localization and saved rounds
# -------------------------------------------------------------------------------------------------

def _retain_brackets(result, brackets, pairs, left_label):
    """Save completed brackets and the current transition bounds for a later refinement step."""
    bracket_start, bracket_end = min(pair[0] for pair in pairs), max(pair[1] for pair in pairs)
    result['endpoint_brackets_partial'] = brackets + [
        {
            'bounds': [float(bracket_start), float(bracket_end)],
            'left_supported': left_label == decide.SUPPORTED,
        }
    ]


def _seed_for(rounds, result):
    """Collect interval seeds from the selected rounds and saved boundary probes."""
    samples = [sample for round_record in rounds for sample in round_record['probes']]
    samples += [boundary_record['probe'] for boundary_record in result.get('boundary_probes', [])]
    return et.seed_intervals_from_probes(*samples)


def _decided_transitions(states):
    """Return index pairs and left states for support transitions, skipping undecided probes."""
    decided = [(index, state) for index, state in enumerate(states) if decide.decided(state)]
    return [
        (left, right, state)
        for (left, state), (right, other) in zip(decided, decided[1:])
        if state != other
    ]


def _matching_boundary_transitions(rounds, options):
    """Return matching transitions and the initial support state, or None if unresolved."""
    labels = [
        [decide.probe_state(sample, options) for sample in round_record['probes']]
        for round_record in rounds
    ]
    transitions = [_decided_transitions(flags) for flags in labels]
    first_decided = [
        next((state for state in flags if decide.decided(state)), None) for flags in labels
    ]
    if (
        not transitions[0]
        or len(transitions[0]) != len(transitions[1])
        or first_decided[0] != first_decided[1]
    ):
        return
    if any(not decide.decided(flags[side]) for flags in labels for side in (0, -1)):
        return  # An undecided window edge leaves the outermost interval unbounded
    for flags, changes in zip(labels, transitions):
        if any(
            not decide.decided(state)
            and not any(
                left_index < probe_index < right_index for left_index, right_index, _ in changes
            )
            for probe_index, state in enumerate(flags)
        ):
            return
    return transitions, first_decided[-1]


def _prepare_boundary_queries(
    selected_rounds, pairs, unknowns, tolerance, boundary_probes, options
):
    """Plan the next bisection probes, reuse saved samples, and flag stalled brackets."""
    stalled = False
    queries = []
    for (descriptor, _, _), bounds, unknown in zip(selected_rounds, pairs, unknowns):
        if unknown and max(unknown) - min(unknown) > tolerance:
            stalled = True
            break  # The unresolved band itself is wider than allowed
        positions = (
            [(bounds[0] + min(unknown)) / 2, (max(unknown) + bounds[1]) / 2]
            if unknown
            else [sum(bounds) / 2]
        )
        for midpoint in positions:
            if midpoint <= bounds[0] or midpoint >= bounds[1] or midpoint in unknown:
                stalled = True
                break
            cached = next(
                (
                    entry['probe']
                    for entry in boundary_probes
                    if entry['grid'] == descriptor and entry['probe']['coordinate'] == midpoint
                ),
                None,
            )
            if cached is None:
                cached = options.get('probe_cache', {}).get(probe_key(midpoint, descriptor))
                if cached is not None:
                    boundary_probes.append({'grid': descriptor, 'probe': cached})
            queries.append((descriptor, bounds, unknown, midpoint, cached))
    return queries, stalled


def _store_boundary_intervals(
    result, brackets, transition_count, initial_state, tolerance, stalled
):
    """Store partial brackets or assemble resolved support intervals and their uncertainty."""
    # Retain partial brackets or assemble the resolved support intervals
    if stalled or len(brackets) != transition_count:
        result['status'] = 'unresolved'
        result['endpoint_brackets_partial'] = brackets
        result['stalled_reason'] = 'estimator_cannot_separate_rounds'
        return
    # Overlapping uncertainty brackets cannot certify separate components
    if any(
        previous_bracket['bounds'][1] >= bracket['bounds'][0]
        for previous_bracket, bracket in zip(brackets, brackets[1:])
    ):
        result['status'] = 'unresolved'
        return
    intervals, start = [], result['bounds'][0] if initial_state == decide.SUPPORTED else None
    for bracket in brackets:
        boundary = sum(bracket['bounds']) / 2
        if bracket['left_supported']:
            intervals.append([start, boundary])
            start = None
        else:
            start = boundary
    if start is not None:
        intervals.append([start, result['bounds'][1]])
    result.pop('stalled_reason', None)
    result.pop('endpoint_brackets_partial', None)
    result.update(
        status='boundary_resolved',
        support_intervals=intervals,
        endpoint_brackets=brackets,
        endpoint_tolerance=tolerance,
    )


def _resolve_boundaries(result, axis, edges, targets, system, settings, options, budget, probe):
    """Refine matching support transitions from the last two validation rounds.

    Check transition agreement, reuse cached probes, and narrow each boundary bracket within the
    remaining point budget. Update the result with resolved support intervals or retain partial
    brackets and the reason refinement stopped.
    """

    # Check that consecutive rounds identify matching support transitions
    rounds = result['rounds'][-2:]
    tolerance = options.get('endpoint_tolerances', {}).get(axis)
    if len(rounds) != 2 or tolerance is None or tolerance <= 0:
        return
    matching = _matching_boundary_transitions(rounds, options)
    if matching is None:
        return
    transitions, initial_state = matching

    # Prepare cached probes and both resolutions for boundary localization
    boundary_probes = result.setdefault('boundary_probes', [])
    descriptors = [round_record['grid'] for round_record in rounds]
    brackets, stalled = [], False

    # Refine each transition within the available point budget
    for first, second in zip(*transitions):
        selected_rounds = [
            (descriptors[probe_index], rounds[probe_index], (first, second)[probe_index])
            for probe_index in (0, 1)
        ]
        pairs = [
            [
                round_record['probes'][left_index]['coordinate'],
                round_record['probes'][right_index]['coordinate'],
            ]
            for _, round_record, (left_index, right_index, _) in selected_rounds
        ]
        left_label = first[2]
        unknowns = [
            [
                sample['coordinate']
                for sample in round_record['probes'][left_index + 1 : right_index]
                if not decide.decided(decide.probe_state(sample, options))
            ]
            for _, round_record, (left_index, right_index, _) in selected_rounds
        ]
        # Both resolutions must fit inside the same tolerance-sized bracket
        while (
            max(bracket[1] for bracket in pairs) - min(bracket[0] for bracket in pairs) > tolerance
        ):
            if max(bracket[1] - bracket[0] for bracket in pairs) <= tolerance / 4:
                stalled = True
                break
            queries, stalled = _prepare_boundary_queries(
                selected_rounds, pairs, unknowns, tolerance, boundary_probes, options
            )
            if stalled:
                break
            step_cost = sum(
                descriptor_cost(descriptor)
                for descriptor, _, _, _, cached in queries
                if cached is None
            )
            if result['points_used'] + step_cost > budget:
                result.update(
                    status='budget_exhausted',
                    next_step_points=step_cost,
                    next_step_kind='boundary_refinement',
                )
                _retain_brackets(result, brackets, pairs, left_label)
                return
            for descriptor, bounds, unknown, midpoint, sample in queries:
                if sample is None:
                    result['points_used'] += descriptor_cost(descriptor)
                    sample = run_probe(
                        axis,
                        midpoint,
                        descriptor,
                        edges,
                        targets,
                        system,
                        settings,
                        options,
                        probe,
                        seed=_seed_for(rounds, result),
                    )
                    boundary_probes.append({'grid': descriptor, 'probe': sample})
                result['feasible_witness_found'] |= sample['witness'] is not None
                state = decide.probe_state(sample, options)
                if not decide.decided(state):
                    unknown.append(midpoint)
                elif state == left_label:
                    if unknown and midpoint >= min(unknown):
                        stalled = True  # Do not discard unresolved evidence
                    else:
                        bounds[0] = midpoint
                else:
                    if unknown and midpoint <= max(unknown):
                        stalled = True
                    else:
                        bounds[1] = midpoint
            if stalled:
                break
        bracket_start, bracket_end = min(bracket[0] for bracket in pairs), max(
            bracket[1] for bracket in pairs
        )
        brackets.append(
            {
                'bounds': [float(bracket_start), float(bracket_end)],
                'round_brackets': deepcopy(pairs),
                'left_supported': left_label == decide.SUPPORTED,
            }
        )
        if stalled:
            break

    # Retain partial brackets or assemble the resolved support intervals
    _store_boundary_intervals(
        result, brackets, len(transitions[0]), initial_state, tolerance, stalled
    )


def _round_state(samples, options):
    """Summarize probe classifications as supported, below threshold, mixed, or undecided."""
    states = [decide.probe_state(sample, options) for sample in samples]
    return (
        'undecided'
        if any(not decide.decided(state) for state in states)
        else (
            'supported'
            if all(state == decide.SUPPORTED for state in states)
            else (
                'below_threshold' if all(state == decide.EXCLUDED for state in states) else 'mixed'
            )
        )
    )


def _invalidate_window_geometry(result):
    """Clear geometry and stopping decisions while retaining paid rounds and boundary probes."""
    result['status'] = 'unresolved'
    for key in (
        'support_intervals',
        'endpoint_brackets',
        'endpoint_tolerance',
        'endpoint_brackets_partial',
        'stalled_reason',
        'next_step_points',
        'next_step_kind',
    ):
        result.pop(key, None)


def _needs_boundary_agreement(result):
    """Identify saved boundary results that still need brackets from both estimator rounds."""
    return result['status'] == 'boundary_resolved' and (
        not result.get('endpoint_brackets')
        or any(
            len(bracket.get('round_brackets', [])) != 2
            for bracket in result['endpoint_brackets']
        )
    )


def _settled_empty_samples(samples, options):
    """Return whether all samples satisfy the configured single-round empty-scan rule."""
    return bool(
        options.get('single_round_when_empty', True)
        and samples
        and all(
            sample.get('method') == et.EDGE_METHOD
            and not sample.get('scan_limited')
            and not sample.get('compatible_cells')
            and decide.probe_state(sample, options) == decide.EXCLUDED
            for sample in samples
        )
    )


def _rebuild_rounds(result, positions, options):
    """Rebuild saved rounds on current probe positions using cached samples until a sample is
    missing.
    """
    saved = result.get('rounds')
    if not saved:
        return
    requested_positions = [float(position) for position in positions]
    if all(
        len(round_record['probes']) == len(requested_positions)
        and all(
            sample['coordinate'] == position
            for sample, position in zip(round_record['probes'], requested_positions)
        )
        for round_record in saved
    ):
        return
    cache = options.get('probe_cache', {})
    rebuilt = []
    for record in saved:
        descriptor = record['grid']
        saved_probes = {sample['coordinate']: sample for sample in record['probes']}
        samples = [
            saved_probes.get(position) or cache.get(probe_key(position, descriptor))
            for position in requested_positions
        ]
        if any(sample is None for sample in samples):
            break
        state = _round_state(samples, options)
        rebuilt.append(
            dict(
                record,
                probes=samples,
                state=state,
                settled_empty=state == 'below_threshold'
                and _settled_empty_samples(samples, options),
            )
        )
    result['rounds'] = rebuilt
    _invalidate_window_geometry(result)


# -------------------------------------------------------------------------------------------------
# Validation workers and result summaries
# -------------------------------------------------------------------------------------------------

def validate_gap_bin(task, probe=None):
    """Evaluate or resume one gap bin or local window within its assigned point budget.

    Unpack the task, recover completed rounds, evaluate finer probes, and localize support
    transitions when successive rounds agree. Return the updated result with probe history, used
    points, support status, and stopping diagnostics.
    """

    # Prepare probe positions and restore completed rounds
    axis, index, edges, sizes, targets, system, settings, options, budget = task
    bin_start, bin_end = edges[axis][index : index + 2]
    positions = (
        np.linspace(bin_start, bin_end, options['probes'])
        if options.get('include_endpoints')
        else bin_start
        + (np.arange(options['probes']) + 0.5) * (bin_end - bin_start) / options['probes']
    )
    if 'probe_positions' in options:
        positions = np.asarray(options['probe_positions'], dtype=float)
    result = deepcopy(options.get('previous_result')) or {
        'bin': int(index),
        'bounds': [float(bin_start), float(bin_end)],
        'status': 'unresolved',
        'points_used': 0,
        'rounds': [],
        'feasible_witness_found': False,
    }
    bounds = [float(bin_start), float(bin_end)]
    if result['bounds'] != bounds:
        result['bounds'] = bounds
        _invalidate_window_geometry(result)
    result['bin'] = int(index)
    if axis == 'a_out':
        result['bounds_au'] = [10 ** bound for bound in bounds]
    _rebuild_rounds(result, positions, options)
    if _needs_boundary_agreement(result):
        _invalidate_window_geometry(result)
    # A required round count beyond the configured limit can never be reached,
    # so a window carrying one would stay unresolved no matter what is spent.
    required_rounds = min(max(2, options.get('minimum_levels', 2)), max(2, options['levels']))
    if result['status'] in RESOLVED and len(result['rounds']) >= required_rounds:
        _update_validation_diagnostics(result)
        return result
    result['status'] = 'unresolved'
    result.pop('next_step_points', None)
    result.pop('next_step_kind', None)

    # Define the round-agreement and boundary-resolution rules
    def resolve_last_rounds():
        """Update the result from round agreement or refine the latest support transitions."""
        rounds = result['rounds']
        if rounds and rounds[-1]['state'] == 'below_threshold' and rounds[-1].get('settled_empty'):
            result['status'] = 'persistent_at_probes'
            return
        if len(rounds) < required_rounds:
            return
        previous, current = rounds[-2:]
        if all(
            round_record['state'] in ('mixed', 'undecided') for round_record in (previous, current)
        ):
            _resolve_boundaries(
                result, axis, edges, targets, system, settings, options, budget, probe
            )
        elif current['state'] == previous['state'] and current['state'] != 'undecided':
            result['status'] = (
                'recovered' if current['state'] == 'supported' else 'persistent_at_probes'
            )

    # Resume saved work and evaluate finer estimator rounds
    try:
        # Saved rounds can already bracket a boundary under the new policy.
        resolve_last_rounds()
        if result['status'] in RESOLVED or result['status'] == 'budget_exhausted':
            return result
        for level in range(len(result['rounds']), options['levels']):
            descriptor, probe_cost = estimator_plan(axis, sizes, level, options)
            cached = [
                options.get('probe_cache', {}).get(probe_key(position, descriptor))
                for position in positions
            ]
            round_cost = probe_cost * sum(cached_sample is None for cached_sample in cached)
            if result['points_used'] + round_cost > budget:
                result.update(
                    status='budget_exhausted',
                    next_step_points=round_cost,
                    next_step_kind='estimator_round',
                )
                break
            samples = []
            for position, sample in zip(positions, cached):
                if sample is None:
                    result['points_used'] += probe_cost
                    sample = run_probe(
                        axis,
                        position,
                        descriptor,
                        edges,
                        targets,
                        system,
                        settings,
                        options,
                        probe,
                        seed=et.seed_intervals_from_probes(
                            *samples, *[sample for sample in cached if sample]
                        ),
                    )
                samples.append(sample)
            state = _round_state(samples, options)
            result['feasible_witness_found'] |= any(
                sample['witness'] is not None for sample in samples
            )
            record = {
                'grid': descriptor,
                'state': state,
                'probes': samples,
                'settled_empty': state == 'below_threshold'
                and _settled_empty_samples(samples, options),
            }
            result['rounds'].append(record)
            result['status'] = 'unresolved'
            resolve_last_rounds()
            if result['status'] in RESOLVED or result['status'] == 'budget_exhausted':
                break
    except (ValueError, FloatingPointError, OverflowError) as exc:
        result.update(status='evaluation_failed', error=str(exc))
    finally:
        _update_validation_diagnostics(result)
    return result


def _update_validation_diagnostics(result):
    """Translate the validation outcome into evidence, support, and stopping labels for reports."""
    status = result['status']

    # Describe whether the probes found a compatible example
    if result.get('feasible_witness_found'):
        result['feasibility_evidence'] = 'solutions_found'
    elif result.get('rounds') or result.get('boundary_probes'):
        result['feasibility_evidence'] = 'no_solutions_found_at_probes'
    else:
        result['feasibility_evidence'] = 'not_established'

    # Translate resolved outcomes into support labels
    support_labels = {
        'recovered': 'supported_at_probes',
        'persistent_at_probes': 'below_threshold_at_probes',
        'boundary_resolved': 'localized_support_intervals',
        'disabled': 'not_checked',
    }
    result['filtered_support_status'] = support_labels.get(status, 'unresolved')

    # Select the stopping reason, preferring a specific obstruction when available
    default_stop_reasons = {
        'unresolved': 'round_limit',
        'budget_exhausted': 'insufficient_budget_for_next_step',
        'evaluation_failed': 'evaluation_failed',
        'disabled': 'disabled',
    }
    result['stop_reason'] = default_stop_reasons.get(status, 'probe_agreement')
    if status == 'unresolved':
        if result.get('stalled_reason'):
            result['stop_reason'] = result['stalled_reason']
        elif result.get('rounds') and result['rounds'][-1]['state'] == 'undecided':
            result['stop_reason'] = 'occupancy_undecided_at_probes'

    # Keep next-step budget fields only when more budget is needed
    if status != 'budget_exhausted':
        result.pop('next_step_points', None)
        result.pop('next_step_kind', None)


def interval_geometry(intervals, edges, circular):
    """Merge support intervals and return widths, coverage, normalized bounds, and the covering
    envelope.
    """

    # Merge intervals and measure their total support
    domain_start, domain_end = float(edges[0]), float(edges[-1])
    domain_span = domain_end - domain_start
    merged = []
    for interval_start, interval_end in sorted(intervals):
        if merged and interval_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval_end)
        else:
            merged.append([float(interval_start), float(interval_end)])
    width = sum(interval_end - interval_start for interval_start, interval_end in merged)

    # Construct the linear envelope or shortest circular covering arc
    envelope = None
    if merged:
        start, end = merged[0][0], merged[-1][1]
        if circular:
            # Include the gap across the coordinate seam, then remove the
            # longest gap to obtain the shortest covering arc
            gaps = [(merged[0][0] + domain_span - merged[-1][1], len(merged) - 1)]
            gaps += [
                (merged[gap_index + 1][0] - merged[gap_index][1], gap_index)
                for gap_index in range(len(merged) - 1)
            ]
            _, gap_index = max(gaps)
            start, end = merged[(gap_index + 1) % len(merged)][0], merged[gap_index][1]
        arc_width = end - start if end >= start else end - start + domain_span
        envelope = dict(start=start, end=end, wraps=end < start, width=arc_width)
    return dict(
        support_intervals=merged,
        support_width=width,
        support_coverage=width / domain_span,
        envelope=envelope,
        envelope_coverage=envelope['width'] / domain_span if envelope else 0.0,
        normalized_intervals=[
            [
                (interval_start - domain_start) / domain_span,
                (interval_end - domain_start) / domain_span,
            ]
            for interval_start, interval_end in merged
        ],
        normalized_envelope=(
            [
                (envelope['start'] - domain_start) / domain_span,
                (envelope['start'] - domain_start + envelope['width']) / domain_span,
            ]
            if envelope
            else None
        ),
    )


def validation_stop_explanation(reasons):
    """Translate unresolved-check counts into a readable explanation of why validation stopped."""
    messages = []
    if reasons.get('insufficient_budget_for_next_step'):
        messages.append('The point budget prevented further checks.')
    if reasons.get('round_limit'):
        messages.append('Some checks reached the refinement-round limit.')
    if reasons.get('occupancy_undecided_at_probes'):
        messages.append('Some sampled regions remain too close to the support threshold.')
    if reasons.get('estimator_cannot_separate_rounds'):
        messages.append('Boundary estimates or unresolved bands could not fit within tolerance.')
    if reasons.get('evaluation_failed'):
        messages.append('Some checks could not be evaluated.')
    return ' '.join(messages)


# -------------------------------------------------------------------------------------------------
# Budget allocation and projection updates
# -------------------------------------------------------------------------------------------------

def allocate_validation_steps(
    jobs, results, metadata, task_for, runner, priority=None
):
    """Allocate complete validation steps within a shared point budget.

    Resume cached work to determine each next step, prioritize affordable jobs, execute them,
    and update results and cumulative point usage until no further step can run.
    """

    # Initialize pending work
    pending = jobs
    while pending:
        remaining_points = max(0, metadata['max_points'] - metadata['points_used'])

        # Determine the next complete step for each pending job
        planned = []
        for key in pending:
            saved = results.get(key)
            if saved and saved['status'] == 'evaluation_failed':
                continue
            previous_points = saved['points_used'] if saved else 0
            result = validate_gap_bin(task_for(key, previous_points))
            results[key] = result
            step_cost = result.get('next_step_points')
            if step_cost is not None:
                planned.append((step_cost, key))

        # Select affordable steps in priority order
        selected, tasks, old_costs = [], [], []
        # Finish affordable convergence-critical rounds before spending on diagnostic gaps
        ranked = [(priority(key) if priority else 0, step_cost, key) for step_cost, key in planned]
        for tier in sorted({job_priority for job_priority, _, _ in ranked}):
            for job_priority, step_cost, key in sorted(ranked):
                if job_priority != tier or step_cost > remaining_points:
                    continue
                previous_points = results[key]['points_used']
                selected.append(key)
                old_costs.append(previous_points)
                tasks.append(task_for(key, previous_points + step_cost))
                remaining_points -= step_cost
            if tasks:
                break
        if not tasks:
            break

        # Execute selected jobs and update cumulative point usage
        for number, result in runner(list(enumerate(tasks))):
            results[selected[number]] = result
            metadata['points_used'] += result['points_used'] - old_costs[number]
        pending = [
            key
            for _, key in planned
            if results[key]['status'] not in RESOLVED
            and results[key]['status'] != 'evaluation_failed'
        ]


def _prepare_gap_validation_jobs(families, targets, sizes, previous_metadata):
    """Group identical targets and return pending jobs, projection records, and saved checks."""
    # Group families with identical admitted targets
    groups = []
    for family in families:
        match = next(
            (
                target_group
                for target_group in groups
                if np.array_equal(
                    targets[target_group[0]].source_index, targets[family].source_index
                )
            ),
            None,
        )
        if match is None:
            groups.append([family])
        else:
            match.append(family)

    # Collect gap-bin jobs and restore saved checks
    jobs, records, results = [], [], {}
    for group in groups:
        family = group[0]
        for axis in AXES:
            projection = families[family]['projections'][axis]
            original_projection = projection.get('grid_projection', projection)
            gaps = gap_runs(
                support_mask(original_projection, sizes[axis]), original_projection['circular']
            )
            records.append((group, axis, gaps))
            saved = {
                bin_group['bin']: bin_group
                for target_group in projection.get('gap_validation', [])
                for bin_group in target_group['bins']
            }
            for bins in gaps:
                for index in bins:
                    key = (family, axis, index)
                    if previous_metadata is not None and index in saved:
                        results[key] = deepcopy(saved[index])
                    if (
                        key not in results
                        or results[key]['status'] not in RESOLVED
                        or _needs_boundary_agreement(results[key])
                    ):
                        jobs.append(key)

    return jobs, records, results


def _merge_gap_validation_projection(families, group, axis, gaps, results, edges, system):
    """Merge one projection's gap checks into support intervals, uncertainty, and diagnostics."""
    projection = families[group[0]]['projections'][axis]
    original_projection = deepcopy(projection.get('grid_projection', projection))
    intervals = deepcopy(original_projection['support_intervals'])

    # Collect recovered intervals and label each contiguous gap
    diagnostics = []
    for bins in gaps:
        checks = [
            results.get(
                (group[0], axis, bin_index),
                {'bin': int(bin_index), 'status': 'disabled', 'points_used': 0},
            )
            for bin_index in bins
        ]
        for check in checks:
            _update_validation_diagnostics(check)
            if check['status'] == 'recovered':
                intervals.append(check['bounds'])
            elif check['status'] == 'boundary_resolved':
                intervals.extend(check['support_intervals'])
        statuses = {check['status'] for check in checks}
        if statuses == {'recovered'}:
            status = 'recovered'
        elif statuses == {'persistent_at_probes'}:
            status = 'persistent_at_probes'
        elif statuses <= set(RESOLVED) and 'boundary_resolved' in statuses:
            status = 'boundary_resolved'
        elif statuses <= set(RESOLVED):
            status = 'partially_recovered'
        elif statuses == {'disabled'}:
            status = 'disabled'
        else:
            status = 'unresolved'
        diagnostics.append(
            {'status': status, 'wraps': bool(bins[-1] < bins[0]), 'bins': checks}
        )

    # Rebuild support geometry and separation units
    if gaps:
        projection['grid_projection'] = original_projection
        projection.update(interval_geometry(intervals, edges[axis], projection['circular']))
        if axis == 'a_out':
            projection['support_intervals_au'] = [
                [10**interval_start, 10**interval_end]
                for interval_start, interval_end in projection['support_intervals']
            ]
            projection['support_intervals_rg'] = [
                [10**interval_start / system.rg_post_au, 10**interval_end / system.rg_post_au]
                for interval_start, interval_end in projection['support_intervals']
            ]

    # Replace uncertainty in resolved bins with localized endpoint brackets
    checks = [check for gap in diagnostics for check in gap['bins']]
    covered = [check['bounds'] for check in checks if check['status'] in RESOLVED]
    if covered:
        domain = [float(edges[axis][0]), float(edges[axis][-1])]
        inherited = decide.subtract_intervals(
            decide.normalize_intervals_to_domain(
                original_projection.get('endpoint_uncertainty', []),
                domain,
                projection['circular'],
            ),
            covered,
        )
        localized = [
            bracket['bounds']
            for check in checks
            if check['status'] == 'boundary_resolved'
            for bracket in check.get('endpoint_brackets', [])
        ]
        unknown = decide.subtract_intervals(
            original_projection.get('undecided_intervals', []), covered
        )
        projection['endpoint_uncertainty'] = inherited + localized
        projection['undecided_intervals'] = decide.interval_union(
            unknown + inherited + localized
        )
        possible = interval_geometry(
            decide.normalize_intervals_to_domain(
                projection['support_intervals'] + projection['undecided_intervals'],
                domain,
                projection['circular'],
            ),
            edges[axis],
            projection['circular'],
        )
        projection['possible_support_intervals'] = possible['support_intervals']
        projection['undecided_coverage'] = max(
            0.0, possible['support_coverage'] - projection['support_coverage']
        )

    # Share the updated projection with equivalent target families
    projection['gap_validation'] = diagnostics
    for family in group[1:]:
        families[family]['projections'][axis] = deepcopy(projection)


def _summarize_gap_validation(results, metadata):
    """Update aggregate outcome counts, stopping reasons, and remaining validation budget."""
    for result in results.values():
        metadata['recovered_bins'] += result['status'] == 'recovered'
        metadata['boundary_bins'] += result['status'] == 'boundary_resolved'
        metadata['unresolved_bins'] += result['status'] not in RESOLVED

    # Summarize validation outcomes and remaining work
    metadata['points_remaining'] = max(0, metadata['max_points'] - metadata['points_used'])
    metadata['persistent_bins'] = sum(
        validation_result['status'] == 'persistent_at_probes'
        for validation_result in results.values()
    )
    metadata['bins_with_solutions'] = sum(
        validation_result.get('feasible_witness_found', False)
        for validation_result in results.values()
    )
    metadata['unresolved_bins_with_solutions'] = sum(
        validation_result.get('feasible_witness_found', False)
        and validation_result['status'] not in RESOLVED
        for validation_result in results.values()
    )
    metadata['stop_reasons'] = {
        reason: sum(
            validation_result['stop_reason'] == reason for validation_result in results.values()
        )
        for reason in sorted(
            {validation_result['stop_reason'] for validation_result in results.values()}
        )
    }
    costs = [
        validation_result['next_step_points']
        for validation_result in results.values()
        if 'next_step_points' in validation_result
    ]
    metadata['minimum_next_step_points'] = min(costs) if costs else None


def _validate_indexed_task(item):
    """Evaluate an indexed validation task and return its index with the result."""
    number, task = item
    return number, validate_gap_bin(task)


def validate_projection_gaps(
    families,
    edges,
    targets,
    system,
    settings,
    args,
    progress=None,
    execute=None,
    previous_metadata=None,
):
    """Validate projected gaps for all target families and update their support geometry in place.

    Group identical target sets, resume saved checks, distribute the point budget, and merge
    recovered intervals and boundary brackets. Return aggregate validation diagnostics.
    """
    from system_summary import classify_constraint

    # Prepare estimator settings and group identical target sets
    sizes = {axis: len(edges[axis]) - 1 for axis in AXES}
    options = dict(
        estimator_options(args),
        probes=args.gap_validation_probes,
        levels=args.gap_validation_levels,
        endpoint_tolerances={
            'e_out': args.convergence_endpoint_eout_atol,
            'a_out': args.convergence_endpoint_loga_atol,
            'phase': args.convergence_endpoint_phase_atol_deg,
        },
    )
    jobs, records, results = _prepare_gap_validation_jobs(
        families, targets, sizes, previous_metadata
    )

    # Initialize the budget and build validation tasks
    metadata = {
        'enabled': args.validate_gaps,
        'candidate_bins': sum(len(bin_group) for _, _, gaps in records for bin_group in gaps),
        'points_used': previous_metadata['points_used'] if previous_metadata else 0,
        'max_points': args.gap_validation_max_points,
        'point_budget_unit': 'initial_scan_points',
        'unresolved_bins': 0,
        'recovered_bins': 0,
        'boundary_bins': 0,
        'scope': (
            'Finite edge-tracking probes and localized boundary brackets; '
            'no proof of exclusion or continuous feasibility.'
        ),
    }

    def task_for(key, budget):
        """Package one gap bin and its saved result with the assigned point budget."""
        family, axis, index = key
        return (
            axis,
            index,
            edges,
            sizes,
            targets[family],
            system,
            replace(settings, require_peters_applicable=(family == 'secular_targets')),
            dict(options, previous_result=results.get(key)),
            budget,
        )

    # Prioritize envelope-sensitive checks and allocate validation steps
    if args.validate_gaps and jobs:
        if progress:
            progress(f'  Validating filtered support in {len(jobs)} projected gap bins')

        def priority(key):
            """Prioritize checks that can change the reported covering envelope."""
            if args.convergence_support != 'envelope':
                return 0
            from system_summary import has_unresolved_envelope_uncertainty

            family, axis, index = key
            projection = families[family]['projections'][axis]
            intervals = deepcopy(projection.get('grid_projection', projection)['support_intervals'])
            for (other_family, other_axis, _), check in results.items():
                if (other_family, other_axis) != (family, axis):
                    continue
                if check['status'] == 'recovered':
                    intervals.append(check['bounds'])
                elif check['status'] == 'boundary_resolved':
                    intervals.extend(check['support_intervals'])
            probe_projection = dict(
                projection, **interval_geometry(intervals, edges[axis], projection['circular'])
            )
            check = results.get(
                key, dict(status='unresolved', bounds=list(edges[axis][index : index + 2]))
            )
            probe_projection['gap_validation'] = [{'bins': [check]}]
            return (
                0
                if has_unresolved_envelope_uncertainty(
                    probe_projection, float(edges[axis][0]), float(edges[axis][-1] - edges[axis][0])
                )
                else 1
            )

        context = (
            nullcontext(execute) if execute is not None
            else reusable_task_runner(_validate_indexed_task, args.workers, progress)
        )
        with context as runner:
            allocate_validation_steps(
                jobs, results, metadata, task_for, runner, priority=priority
            )

    # Merge validation results into projected support
    for result in results.values():
        _update_validation_diagnostics(result)
    for group, axis, gaps in records:
        _merge_gap_validation_projection(families, group, axis, gaps, results, edges, system)
    _summarize_gap_validation(results, metadata)

    # Report validation outcomes and remaining work
    if progress and args.validate_gaps and jobs:
        explanation = validation_stop_explanation(metadata["stop_reasons"])
        progress(
            f'  Gap validation complete: {metadata["recovered_bins"]} recovered, '
            f'{metadata["boundary_bins"]} boundary-resolved, '
            f'{metadata["persistent_bins"]} below threshold at probes, '
            f'{metadata["unresolved_bins"]} filtered-support bins unresolved '
            f'({metadata["unresolved_bins_with_solutions"]} with solutions found).'
            + (f' {explanation}' if explanation else '')
        )

    # Refresh each target family classification
    for stats in families.values():
        stats['constraint_class'] = classify_constraint(stats)
    return metadata


def revisit_unresolved_gaps(result, args, progress=None):
    """Resume unresolved checks and update the saved projections and validation diagnostics."""
    from system_summary import Resolution, make_axes
    from triple_cli import system_from_args, settings_from_args

    # Restore the saved grid and rebuild its target sets
    metadata = result.get('gap_validation', {})
    if not args.validate_gaps or not metadata.get('unresolved_bins'):
        return
    started = time.monotonic()
    system, settings = system_from_args(args), settings_from_args(args)
    resolution = Resolution(**result['resolution'])
    edges, _ = make_axes(args, resolution)
    candidates = core.compute_post_first_merger_candidates(
        core.make_epost_grid(resolution.n_epost, args.one_minus_epost_min),
        system.mrem,
        system.m3,
        system.delay_days,
        args.f_ref_hz,
        args.min_peters_orbits,
        args.min_peters_rp_rg,
    )
    targets = {
        name: core.select_admitted_targets(
            candidates, replace(settings, require_peters_applicable=name == 'secular_targets')
        )
        for name in result['families']
    }
    if progress:
        progress('  Revisiting unresolved gaps on the comparison baseline')

    # Save the previous projections and resume unresolved checks
    result.setdefault('gap_validation_revisits', []).append(
        {
            'gap_validation': deepcopy(metadata),
            'projections': {
                name: deepcopy(family_result['projections'])
                for name, family_result in result['families'].items()
            },
        }
    )
    result['gap_validation'] = validate_projection_gaps(
        result['families'],
        edges,
        targets,
        system,
        settings,
        args,
        progress,
        previous_metadata=metadata,
    )

    # Refresh warnings and elapsed time
    result['warnings'] = [
        warning
        for warning in result.get('warnings', [])
        if not warning.endswith('gap bins remain unresolved after targeted validation.')
    ]
    unresolved = result['gap_validation']['unresolved_bins']
    if unresolved:
        result['warnings'].append(
            f'{unresolved} gap bins remain unresolved after targeted validation.'
        )
    elapsed = time.monotonic() - started
    result['gap_validation_revisits'][-1]['elapsed_seconds'] = elapsed
    result['elapsed_seconds'] = result.get('elapsed_seconds', 0.0) + elapsed
