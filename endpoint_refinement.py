"""Refine projected support boundaries locally after bulk grid agreement.

Build refinement windows from two grid results, reuse stored probes and validation rounds, and
allocate the available point budget to unresolved endpoints and gaps. Merge the refined
intervals and uncertainty brackets into each result, then repeat the comparison while local
progress is possible. Global volume counts and the target samples stay fixed during refinement.
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import time
import numpy as np
import triple_core as core
import gap_validation as gaps
import edge_tracking as et
import support_decision as decide
from triple_core import reusable_task_runner


# Limit repeated local refinement passes between two grid results
MAX_LOCAL_COMPARISON_PASSES = 3


# -------------------------------------------------------------------------------------------------
# Refinement windows and probe reuse
# -------------------------------------------------------------------------------------------------

def base_projection(projection):
    """Return the saved projection from before local refinement, or the supplied projection."""
    return projection.get('endpoint_base_projection', projection)


def make_windows(left, right, domain, cell_width):
    """Build windows around support endpoints, gaps, and unresolved regions on both grids.

    Merge overlapping windows and return each window with probe positions that include interior
    support samples and its endpoints.
    """

    # Collect endpoint and gap windows
    domain_start, domain_end = map(float, domain)
    circular = left['circular']
    spans, seeds = [], {domain_start, domain_end}
    for projection in (left, right):
        intervals = projection['support_intervals']
        for start, end in decide.unbracketed_possible(projection, domain, envelope_only=False):
            spans.append((max(domain_start, start - cell_width), min(domain_end, end + cell_width)))
            seeds.add((start + end) / 2)
        possible_endpoints = [
            coordinate
            for start, end in projection.get('possible_support_intervals', [])
            for coordinate in (start, end)
        ]
        for start, end in intervals:
            seeds.add((start + end) / 2)
            if end - start > cell_width:
                seeds.update((start + cell_width / 2, end - cell_width / 2))
        for coordinate in [
            coordinate for start, end in intervals for coordinate in (start, end)
        ] + possible_endpoints:
            supported_seam = (
                circular
                and intervals
                and intervals[0][0] == domain_start
                and intervals[-1][1] == domain_end
                and coordinate in (domain_start, domain_end)
            )
            if not supported_seam and (circular or domain_start < coordinate < domain_end):
                spans.append(
                    (
                        max(domain_start, coordinate - 2 * cell_width),
                        min(domain_end, coordinate + 2 * cell_width),
                    )
                )
        gaps = [(start[1], end[0]) for start, end in zip(intervals, intervals[1:])]
        if circular and intervals:
            gaps += [(domain_start, intervals[0][0]), (intervals[-1][1], domain_end)]
        for start, end in gaps:
            if start < end:
                spans.append(
                    (max(domain_start, start - cell_width), min(domain_end, end + cell_width))
                )
                seeds.add((start + end) / 2)

    # Merge windows and select their initial probe positions
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        (
            start,
            end,
            sorted(
                {
                    *np.linspace(start, end, 5),
                    *(coordinate for coordinate in seeds if start <= coordinate <= end),
                }
            ),
        )
        for start, end in merged
    ]


def make_envelope_windows(
    left, right, domain, cell_width, blocking_projections=None, current_projection=None
):
    """Build probe windows for unresolved outer envelopes on the two grids.

    Include endpoint brackets, exposed possible support, and gap checks that affect the covering
    envelope. Return the window bounds and their seed positions.
    """
    from system_summary import circular_envelope_uncertainty, has_unresolved_envelope_uncertainty

    domain_start, domain_end = map(float, domain)
    domain_span = domain_end - domain_start
    spans, seeds = [], set()
    for projection in (left, right):
        for start, end in decide.unbracketed_possible(projection, domain):
            spans.append(
                (max(domain_start, start - 2 * cell_width), min(domain_end, end + 2 * cell_width))
            )
            seeds.add((start + end) / 2)
        envelope = projection.get('normalized_envelope')
        if envelope is None:
            continue
        full_circle = projection['circular'] and abs(envelope[1] - envelope[0] - 1) <= 1e-12
        points = (
            []
            if full_circle
            else [
                domain_start
                + (coordinate % 1 if projection['circular'] else coordinate) * domain_span
                for coordinate in envelope
            ]
        )
        endpoint_points = list(points)
        refine_internal_gaps = False
        if projection['circular']:
            _, refine_internal_gaps = circular_envelope_uncertainty(
                projection, domain_start, domain_span
            )
        for left_bound, right_bound in projection.get('endpoint_uncertainty', []):
            for shift in (-domain_span, 0, domain_span) if projection['circular'] else (0,):
                start, end = left_bound + shift, right_bound + shift
                if (
                    full_circle
                    or refine_internal_gaps
                    or any(
                        start - 1e-10 <= coordinate + endpoint_shift <= end + 1e-10
                        for coordinate in endpoint_points
                        for endpoint_shift in (
                            (-domain_span, 0, domain_span) if projection['circular'] else (0,)
                        )
                    )
                ):
                    points += [start, end]
                    spans.append(
                        (
                            max(domain_start, start - 2 * cell_width),
                            min(domain_end, end + 2 * cell_width),
                        )
                    )
        for coordinate in points:
            if projection['circular'] or domain_start < coordinate < domain_end:
                spans.append(
                    (
                        max(domain_start, coordinate - 2 * cell_width),
                        min(domain_end, coordinate + 2 * cell_width),
                    )
                )
        for start, end in projection['support_intervals']:
            seeds.update(
                (
                    (start + end) / 2,
                    start + min(cell_width / 2, (end - start) / 2),
                    end - min(cell_width / 2, (end - start) / 2),
                )
            )

    # Include possible support exposed by the current envelope
    if current_projection is not None:
        for start, end in decide.unbracketed_possible(current_projection, domain):
            spans.append(
                (max(domain_start, start - 2 * cell_width), min(domain_end, end + 2 * cell_width))
            )
            seeds.update((start, (start + end) / 2, end))
    merged = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Retain gap checks that can change the envelope
    blocking_bounds = []
    for projection in (left, right) if blocking_projections is None else blocking_projections:
        checks = [check for gap in projection.get('gap_validation', []) for check in gap['bins']]
        for check in checks:
            if 'bounds' in check and has_unresolved_envelope_uncertainty(
                dict(projection, gap_validation=[{'bins': [check]}]), domain_start, domain_span
            ):
                blocking_bounds.append(check['bounds'])
    for bounds in sorted(blocking_bounds):
        pieces = decide.subtract_intervals(
            decide.normalize_intervals_to_domain([bounds], domain, left['circular']), merged
        )
        merged.extend(pieces)
    return [
        (
            start,
            end,
            sorted(
                {
                    *np.linspace(start, end, 5),
                    *(coordinate for coordinate in seeds if start <= coordinate <= end),
                }
            ),
        )
        for start, end in sorted(merged)
    ]


def shared_probe_cache(cache, family, axis, projection):
    """Collect reusable probes for one target family and axis, keyed by coordinate and estimator."""
    samples = {}
    checks = [value for key, value in cache.items() if key[:2] == (family, axis)]
    checks += [check for group in projection.get('gap_validation', []) for check in group['bins']]
    for check in checks:
        for round_record in check.get('rounds', []):
            for sample in round_record['probes']:
                samples[
                    gaps.probe_key(sample['coordinate'], round_record['grid'])
                ] = sample
        for boundary_record in check.get('boundary_probes', []):
            sample = boundary_record['probe']
            samples[
                gaps.probe_key(sample['coordinate'], boundary_record['grid'])
            ] = sample
    return samples


def distinct_probe_positions(positions, span, limit, relative=1e-9, *, required=()):
    """Limit optional probe positions while retaining endpoints and all required samples."""
    required = set(map(float, required))
    ordered = sorted({*map(float, positions), *required})
    if not ordered:
        return []
    required.update((ordered[0], ordered[-1]))
    tolerance = abs(span) * relative
    unique = [
        position
        for i, position in enumerate(ordered)
        if position in required or position - ordered[i - 1] > tolerance
    ]
    if limit and len(unique) > limit:
        optional = sorted(
            (i for i in range(1, len(unique) - 1) if unique[i] not in required),
            key=lambda i: unique[i + 1] - unique[i - 1],
            reverse=True,
        )[: max(0, limit - len(required))]
        unique = sorted(required | {unique[i] for i in optional})
    return unique


def _required_probe_positions(positions, bounds, projections, stored):
    """Preserve samples in each known support component and gap, plus saved window positions."""
    window_start, window_end = bounds
    required = {window_start, window_end}

    # Keep one interior sample per support component and separating gap
    for projection in projections:
        intervals = projection['support_intervals']
        gaps = list(
            zip(
                [window_start] + [end for _, end in intervals],
                [start for start, _ in intervals] + [window_end],
            )
        )
        for start, end in [*intervals, *gaps]:
            start, end = max(start, window_start), min(end, window_end)
            if start < end:
                midpoint = (start + end) / 2
                required.add(
                    min(
                        (position for position in positions if start < position < end),
                        key=lambda position: abs(position - midpoint),
                        default=midpoint,
                    )
                )

    # Retain previous sampling positions when windows are rebuilt or merged
    required.update(
        position
        for record in stored
        if record.get('kind', 'window') == 'window'
        for position in record.get('positions', [])
        if window_start <= position <= window_end
    )
    return required


def merge_active_windows(proposed, previous, cached_windows, merge_adjacent=True):
    """Merge overlapping refinement windows while retaining prior samples and round requirements.

    Return bounds, sorted probe positions, and minimum round counts for the active windows. Use
    merge_adjacent to control whether touching proposed windows are combined.
    """

    # Combine proposed windows with saved refinement work
    pending = [(start, end, set(positions), 2) for start, end, positions in proposed]
    selected = []
    remaining = list(previous)
    while pending:
        start, end, positions, minimum_rounds = pending.pop(0)
        changed = True
        while changed:
            changed = False
            for check in list(remaining):
                other_start, other_end = check['bounds']
                if other_start <= end and other_end >= start:
                    start, end = min(start, other_start), max(end, other_end)
                    positions.update((other_start, other_end))
                    positions.update(
                        sample['coordinate']
                        for round_record in check.get('rounds', [])[:1]
                        for sample in round_record['probes']
                    )
                    for value in cached_windows.values():
                        if value['bounds'] == check['bounds']:
                            positions.update(value.get('positions', ()))
                    minimum_rounds = max(
                        minimum_rounds, len(check.get('rounds', [])), check.get('minimum_levels', 2)
                    )
                    remaining.remove(check)
                    changed = True
            for item in list(pending):
                other_start, other_end, other_positions, levels = item
                overlaps = (
                    other_start <= end and other_end >= start
                    if merge_adjacent
                    else other_start < end and other_end > start
                )
                if overlaps:
                    start, end = min(start, other_start), max(end, other_end)
                    positions.update(other_positions)
                    minimum_rounds = max(minimum_rounds, levels)
                    pending.remove(item)
                    changed = True
        selected.append((start, end, sorted(positions), minimum_rounds))
    return selected


# -------------------------------------------------------------------------------------------------
# Consistency checks and refinement planning
# -------------------------------------------------------------------------------------------------

def consistent_with_paid_probes(check, samples, options):
    """Mark a resolved window unresolved if comparable cached probes contradict its support."""
    if check['status'] not in gaps.RESOLVED or not check.get('rounds'):
        return check

    # Identify comparable cached estimators
    rounds = check['rounds'][-2:]
    grids = {gaps.probe_key(0.0, round_record['grid'])[1] for round_record in rounds}
    descriptors = [round_record['grid'] for round_record in rounds]

    def is_comparable_estimator(descriptor):
        """Return whether a cached estimator matches or refines both accepted round descriptors."""
        if descriptor in grids:
            return True
        candidate = dict(descriptor)
        if candidate.get('method') != et.EDGE_METHOD:
            return False
        # More columns with fewer discovery points is not uniformly finer.
        dimensions = ('columns', 'scan')
        fixed = {field: value for field, value in candidate.items() if field not in dimensions}
        return all(
            fixed == {field: value for field, value in reference.items() if field not in dimensions}
            and all(
                candidate.get(field, 0) >= reference.get(field, float('inf'))
                for field in dimensions
            )
            for reference in descriptors
        )

    intervals = (
        [check['bounds']]
        if check['status'] == 'recovered'
        else check.get('support_intervals', []) if check['status'] == 'boundary_resolved' else []
    )
    brackets = [bracket['bounds'] for bracket in check.get('endpoint_brackets', [])]

    # Check cached classifications against the accepted support
    for (coordinate, descriptor), sample in samples.items():
        if (
            not is_comparable_estimator(descriptor)
            or not check['bounds'][0] <= coordinate <= check['bounds'][1]
            or any(start <= coordinate <= bracket for start, bracket in brackets)
        ):
            continue
        state = decide.probe_state(sample, options)
        if not decide.decided(state):
            continue
        if (state == decide.SUPPORTED) != any(
            start <= coordinate <= bracket for start, bracket in intervals
        ):
            check = deepcopy(check)
            check.update(
                status='cached_evidence_disagrees',
                filtered_support_status='unresolved',
                stop_reason='cached_evidence_disagrees',
                minimum_levels=len(check['rounds']) + 1,
            )
            return check
    return check


def _replace_window_intervals(intervals, bounds, replacement):
    """Replace support inside the given window while preserving interval fragments outside it."""
    window_start, window_end = bounds
    remaining = []
    for interval_start, interval_end in intervals:
        if interval_start < window_start:
            remaining.append([interval_start, min(interval_end, window_start)])
        if interval_end > window_end:
            remaining.append([max(interval_start, window_end), interval_end])
    return remaining + deepcopy(replacement)


def _guard_window_boundaries(check, projection, domain, options):
    """Keep a resolved window only when its edge probes agree with the surrounding support.

    Compare decided edge probes from the last two rounds with the grid support immediately
    outside each window edge, so the window cut does not create an artificial support boundary.
    Skip domain edges and undecided probes. Return the original check when consistent,
    or an unresolved copy when an edge disagrees with the surrounding grid.
    """
    # Prepare the surrounding support for resolved windows
    if check['status'] not in gaps.RESOLVED:
        return check
    domain_start, domain_end = domain
    decided_support = projection['support_intervals']
    possible_support = projection.get('possible_support_intervals', decided_support)

    # Inspect the nearest representable coordinate outside each interior window edge
    for side, coordinate in enumerate(check['bounds']):
        if coordinate in (domain_start, domain_end):
            continue
        is_lower_edge = side == 0
        outside_direction = -np.inf if is_lower_edge else np.inf
        outside_coordinate = np.nextafter(coordinate, outside_direction)
        outside_is_supported = any(
            start <= outside_coordinate <= end for start, end in decided_support
        )
        outside_support_is_possible = any(
            start <= outside_coordinate <= end for start, end in possible_support
        )

        # Compare decided edge probes from both recent refinement rounds
        for round_record in check['rounds'][-2:]:
            edge_probe = round_record['probes'][0 if is_lower_edge else -1]
            state = decide.probe_state(edge_probe, options)
            if not decide.decided(state):
                continue
            edge_is_supported = state == decide.SUPPORTED
            support_meets_excluded_grid = edge_is_supported and not outside_support_is_possible
            exclusion_meets_supported_grid = not edge_is_supported and outside_is_supported

            # Reopen a contradictory window without modifying the saved check
            if support_meets_excluded_grid or exclusion_meets_supported_grid:
                check = deepcopy(check)
                check.update(
                    status='unbracketed_boundary',
                    filtered_support_status='unresolved',
                    stop_reason='window_edge_disagreement',
                )
                return check
    return check


def _window_is_covered(bounds, checks):
    """Return whether resolved checks cover the entire supplied window."""

    # Retain grid uncertainty outside resolved windows
    covered = [check['bounds'] for check in checks if check['status'] in gaps.RESOLVED]
    return not decide.subtract_intervals([bounds], covered)


def refinement_focus(previous, current, args):
    """Select axes and coordinate ranges needing further endpoint or gap refinement."""
    from system_summary import support_convergence_failures, has_unresolved_envelope_uncertainty

    # Collect convergence failures and coordinate tolerances
    failures = support_convergence_failures(previous, current, args, args.convergence_support)
    focus = {}
    domains = dict(
        e_out=(args.eout_min, args.eout_max, args.convergence_endpoint_eout_atol),
        a_out=(
            np.log10(args.aout_min_au),
            np.log10(args.aout_max_au),
            args.convergence_endpoint_loga_atol,
        ),
        phase=(args.nu_min_deg, args.nu_max_deg, args.convergence_endpoint_phase_atol_deg),
    )

    def add_focus_range(axis, bounds):
        """Add a refinement range unless the whole axis is already selected."""
        if axis not in focus:
            focus[axis] = []
        if focus[axis] is not None:
            focus[axis].append(bounds)

    # Select unresolved boundaries and blocking gap ranges
    for family in previous['families']:
        for axis, (domain_start, domain_end, tolerance) in domains.items():
            projections = [
                grid_result['families'][family]['projections'][axis]
                for grid_result in (previous, current)
            ]
            relevant = [failure for failure in failures if failure.startswith(f'{family}.{axis}.')]
            if relevant:
                if (
                    args.convergence_support == 'intervals'
                    or projections[0]['circular']
                    or any(
                        failure.endswith(
                            (
                                'invalid_envelope',
                                'envelope_topology',
                                'endpoint_uncertainty',
                                'possible_support_unresolved',
                            )
                        )
                        for failure in relevant
                    )
                ):
                    focus[axis] = None
                elif any(failure.endswith('envelope_endpoints') for failure in relevant):
                    for side in (0, 1):
                        previous_endpoint, current_endpoint = [
                            projection['normalized_envelope'][side] for projection in projections
                        ]
                        distance = (
                            abs((previous_endpoint - current_endpoint + 0.5) % 1 - 0.5)
                            if projections[0]['circular']
                            else abs(previous_endpoint - current_endpoint)
                        )
                        if distance * (domain_end - domain_start) > tolerance + 1e-12:
                            for endpoint in (previous_endpoint, current_endpoint):
                                coordinate = domain_start + (
                                    endpoint % 1 if projections[0]['circular'] else endpoint
                                ) * (domain_end - domain_start)
                                add_focus_range(axis, [coordinate, coordinate])
                    # Arc-complement disagreement can involve near-equal endpoints.
                    if axis not in focus:
                        focus[axis] = None
            for projection in projections:
                checks = [
                    check for gap in projection.get('gap_validation', []) for check in gap['bins']
                ]
                checks += projection.get('endpoint_refinement', [])
                for check in checks:
                    if check['status'] not in (*gaps.RESOLVED, 'disabled') and not check.get(
                        'superseded_by_endpoint_refinement'
                    ):
                        if (
                            args.convergence_support == 'envelope'
                            and not has_unresolved_envelope_uncertainty(
                                dict(projection, gap_validation=[{'bins': [check]}]),
                                domain_start,
                                domain_end - domain_start,
                            )
                        ):
                            continue
                        if 'bounds' in check:
                            add_focus_range(axis, check['bounds'])
                        else:
                            focus[axis] = None
    return focus


def native_gap_plans(projection, stored, domain, args):
    """Prepare resumable gap checks and separate checks that need new refinement windows."""
    from system_summary import has_unresolved_envelope_uncertainty

    # Separate resumable bins from checks requiring new windows
    native, fallback = [], []
    for gap in projection.get('gap_validation', []):
        for check in gap['bins']:
            if not has_unresolved_envelope_uncertainty(
                dict(projection, gap_validation=[{'bins': [check]}]),
                domain[0],
                domain[1] - domain[0],
            ):
                continue
            if 'bounds' not in check:
                continue
            record = next(
                (
                    stored_record
                    for stored_record in stored
                    if stored_record.get('kind') == 'gap_bin'
                    and stored_record['bounds'] == check['bounds']
                ),
                None,
            )
            saved = deepcopy(record['result'] if record else check)
            saved.setdefault('points_used', 0)
            saved.setdefault('rounds', [])
            saved.setdefault('feasible_witness_found', False)
            levels = args.endpoint_refinement_levels
            if (
                len(saved.get('rounds', [])) >= levels
                and saved['status'] not in gaps.RESOLVED
                and not saved.get('next_step_points')
            ):
                fallback.append(check)
                continue
            start, end = check['bounds']
            positions = (
                record['positions']
                if record
                else (
                    [sample['coordinate'] for sample in saved['rounds'][0]['probes']]
                    if saved.get('rounds')
                    else list(
                        start
                        + (np.arange(args.gap_validation_probes) + 0.5)
                        * (end - start)
                        / args.gap_validation_probes
                    )
                )
            )
            native.append(
                dict(
                    bounds=[start, end],
                    positions=positions,
                    result=deepcopy(saved),
                    initial_gap_points=(
                        record.get('initial_gap_points', 0)
                        if record
                        else saved.get('points_used', 0)
                    ),
                )
            )
    return native, fallback


# -------------------------------------------------------------------------------------------------
# Refinement preparation
# -------------------------------------------------------------------------------------------------


def _prepare_refinement_inputs(result, reference, args):
    """Build the grids and targets, grouping families that share admitted candidates."""
    from system_summary import Resolution, make_axes
    from triple_cli import system_from_args, settings_from_args

    resolution = Resolution(**result['resolution'])
    reference_resolution = Resolution(**reference['resolution'])
    edges, _ = make_axes(args, resolution)
    sizes = dict(zip(gaps.AXES, (resolution.n_eout, resolution.n_aout, resolution.n_nu)))
    reference_sizes = dict(
        zip(
            gaps.AXES,
            (reference_resolution.n_eout, reference_resolution.n_aout, reference_resolution.n_nu),
        )
    )
    system, settings = system_from_args(args), settings_from_args(args)
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
    groups = []
    for name in targets:
        group = next(
            (
                target_group
                for target_group in groups
                if np.array_equal(targets[target_group[0]].source_index, targets[name].source_index)
            ),
            None,
        )
        if group is None:
            groups.append([name])
        else:
            group.append(name)

    return edges, sizes, reference_sizes, system, settings, targets, groups


def _prepare_refinement_jobs(
    result, reference, args, edges, sizes, reference_sizes, groups, stored, focus
):
    """Plan focused windows and gap jobs, restoring cached checks and probe positions."""
    envelope_only = args.convergence_support == 'envelope'
    records, windows, minimum_levels, probe_positions, cache = [], {}, {}, {}, {}
    kinds, initial_gap_points = {}, {}
    claimed = set()

    # Build refinement jobs for the selected axes
    for group in groups:
        for axis in gaps.AXES:
            if focus is not None and axis not in focus:
                continue
            projection = result['families'][group[0]]['projections'][axis]
            if 'endpoint_base_projection' not in projection:
                projection['endpoint_base_projection'] = deepcopy(projection)
            original_projection = base_projection(projection)
            reference_projection = base_projection(
                reference['families'][group[0]]['projections'][axis]
            )
            domain = [edges[axis][0], edges[axis][-1]]
            cell_width = (domain[1] - domain[0]) / min(sizes[axis], reference_sizes[axis])
            keys = []
            scoped = [
                (index, record)
                for index, record in enumerate(stored)
                if (record['family'], record['axis']) == (group[0], axis) and 'bounds' in record
            ]
            native, fallback = (
                native_gap_plans(
                    projection, [stored_record for _, stored_record in scoped], domain, args
                )
                if envelope_only
                else ([], [])
            )
            if native:
                for ordinal, plan in enumerate(native):
                    key = (group[0], axis, ordinal)
                    windows[key] = tuple(plan['bounds'])
                    minimum_levels[key] = min(
                        max(2, plan['result'].get('minimum_levels', 2)),
                        args.endpoint_refinement_levels,
                    )
                    probe_positions[key] = plan['positions']
                    kinds[key] = 'gap_bin'
                    initial_gap_points[key] = plan['initial_gap_points']
                    cache[key] = plan['result']
                    claimed.update(
                        index
                        for index, record in scoped
                        if record.get('kind') == 'gap_bin' and record['bounds'] == plan['bounds']
                    )
                    keys.append(key)
            else:
                window_builder = make_envelope_windows if envelope_only else make_windows
                window_options = (
                    dict(
                        blocking_projections=(
                            dict(projection, gap_validation=[{'bins': fallback}]),
                        ),
                        current_projection=projection,
                    )
                    if envelope_only
                    else {}
                )
                proposed = [
                    (window_start, window_end, positions)
                    for window_start, window_end, positions in window_builder(
                        original_projection,
                        reference_projection,
                        domain,
                        cell_width,
                        **window_options,
                    )
                    if focus is None
                    or focus[axis] is None
                    or any(
                        interval_start <= window_end and interval_end >= window_start
                        for interval_start, interval_end in focus[axis]
                    )
                ]
                for ordinal, (window_start, window_end, positions, minimum_rounds) in enumerate(
                    merge_active_windows(
                        proposed,
                        [] if envelope_only else projection.get('endpoint_refinement', []),
                        {index: record for index, record in scoped},
                        merge_adjacent=not envelope_only,
                    )
                ):
                    key = (group[0], axis, ordinal)
                    windows[key] = (window_start, window_end)
                    minimum_levels[key] = min(
                        minimum_rounds, max(2, args.endpoint_refinement_levels)
                    )
                    probe_positions[key] = distinct_probe_positions(
                        positions,
                        window_end - window_start,
                        args.endpoint_refinement_max_positions,
                        required=_required_probe_positions(
                            positions,
                            windows[key],
                            (original_projection, reference_projection),
                            [record for _, record in scoped],
                        ),
                    )
                    kinds[key] = 'window'
                    initial_gap_points[key] = 0
                    match = next(
                        (
                            index
                            for index, record in scoped
                            if index not in claimed
                            and record.get('kind', 'window') == 'window'
                            and (
                                record['bounds'] == [window_start, window_end]
                                if envelope_only
                                else record['bounds'][0] <= window_end
                                and record['bounds'][1] >= window_start
                            )
                        ),
                        None,
                    )
                    if match is not None:
                        claimed.add(match)
                        cache[key] = deepcopy(stored[match]['result'])
                        minimum_levels[key] = min(
                            max(minimum_levels[key], cache[key].get('minimum_levels', 2)),
                            max(2, args.endpoint_refinement_levels),
                        )
                    keys.append(key)
            records.append((group, axis, keys))

    return dict(
        records=records,
        windows=windows,
        minimum_levels=minimum_levels,
        probe_positions=probe_positions,
        cache=cache,
        kinds=kinds,
        initial_gap_points=initial_gap_points,
        claimed=claimed,
    )


def _archived_refinement_probes(stored):
    """Collect paid probes from saved windows by target family and projected axis."""
    archive = {}
    for index, record in enumerate(stored):
        scope = (record['family'], record['axis'])
        archive.setdefault(scope, {}).update(
            shared_probe_cache({(*scope, index): record['result']}, *scope, {})
        )

    return archive


# -------------------------------------------------------------------------------------------------
# Refined projections and reporting
# -------------------------------------------------------------------------------------------------

def _refinement_probe_samples(archive, cache, family, axis, projection):
    """Combine archived and active probes for one target family and axis."""
    return {
        **archive.get((family, axis), {}),
        **shared_probe_cache(cache, family, axis, projection),
    }


def _update_projection_uncertainty(projection, original_projection, checks, domain, axis_edges):
    """Combine refined brackets with uncovered grid uncertainty and update possible support."""
    # Collect completed and partially refined endpoint brackets
    localized = [
        interval_end['bounds']
        for check in checks
        if check['status'] == 'boundary_resolved'
        for interval_end in check['endpoint_brackets']
    ]
    localized += [
        interval_end['bounds']
        for check in checks
        if check['status'] not in gaps.RESOLVED
        for interval_end in check.get('endpoint_brackets_partial', [])
    ]

    # Retain grid uncertainty outside resolved windows
    covered = [check['bounds'] for check in checks if check['status'] in gaps.RESOLVED]
    inherited = decide.subtract_intervals(
        decide.normalize_intervals_to_domain(
            original_projection.get('endpoint_uncertainty', []), domain, projection['circular']
        ),
        covered,
    )
    projection['endpoint_uncertainty'] = localized + inherited

    # Rebuild possible support from refined brackets and remaining unknown regions
    remaining_unknown = decide.subtract_intervals(
        decide.normalize_intervals_to_domain(
            original_projection.get(
                'undecided_intervals',
                decide.subtract_intervals(
                    original_projection.get('possible_support_intervals', []),
                    original_projection['support_intervals'],
                ),
            ),
            domain,
            projection['circular'],
        ),
        covered,
    )
    possible = gaps.interval_geometry(
        decide.normalize_intervals_to_domain(
            projection['support_intervals']
            + projection['endpoint_uncertainty']
            + remaining_unknown,
            domain,
            projection['circular'],
        ),
        axis_edges,
        projection['circular'],
    )
    projection.update(
        possible_support_intervals=possible['support_intervals'],
        undecided_intervals=decide.interval_union(
            projection['endpoint_uncertainty'] + remaining_unknown
        ),
        undecided_coverage=max(0.0, possible['support_coverage'] - projection['support_coverage']),
    )


def _merge_refined_projections(result, jobs, edges, system, options, archive, envelope_only):
    """Apply consistent window results, update each projection, and copy shared-family geometry."""
    records, cache, kinds = jobs['records'], jobs['cache'], jobs['kinds']

    # Merge accepted support and update endpoint uncertainty
    for group, axis, keys in records:
        projection = result['families'][group[0]]['projections'][axis]
        original_projection = base_projection(projection)
        intervals = deepcopy(projection['support_intervals'])
        domain = [float(edges[axis][0]), float(edges[axis][-1])]
        samples = _refinement_probe_samples(archive, cache, group[0], axis, projection)
        checks = [
            consistent_with_paid_probes(
                (
                    cache[window_key]
                    if kinds[window_key] == 'gap_bin'
                    else _guard_window_boundaries(
                        cache[window_key], original_projection, domain, options
                    )
                ),
                samples,
                options,
            )
            for window_key in keys
        ]
        for key, check in zip(keys, checks):
            cache[key] = check
        for check in checks:
            status = check['status']
            if status in gaps.RESOLVED:
                replacement = (
                    [check['bounds']]
                    if status == 'recovered'
                    else check['support_intervals'] if status == 'boundary_resolved' else []
                )
                intervals = _replace_window_intervals(intervals, check['bounds'], replacement)
        projection.update(gaps.interval_geometry(intervals, edges[axis], projection['circular']))
        if axis == 'a_out':
            projection['support_intervals_au'] = [
                [10**interval_start, 10**interval_end]
                for interval_start, interval_end in projection['support_intervals']
            ]
            projection['support_intervals_rg'] = [
                [10**interval_start / system.rg_post_au, 10**interval_end / system.rg_post_au]
                for interval_start, interval_end in projection['support_intervals']
            ]
        retained = [
            check
            for check in projection.get('endpoint_refinement', [])
            if not any(
                other_check['bounds'][0] <= check['bounds'][0]
                and other_check['bounds'][1] >= check['bounds'][1]
                for other_check in checks
            )
        ]
        if envelope_only:
            retained = [
                check
                for check in retained
                if check['status'] in gaps.RESOLVED
                and not any(
                    other_check['bounds'][0] < check['bounds'][1]
                    and other_check['bounds'][1] > check['bounds'][0]
                    for other_check in checks
                )
            ]
        checks = retained + checks
        projection['local_domain_limited'] = bool(
            not projection['circular']
            and projection['support_intervals']
            and (
                projection['support_intervals'][0][0] == domain[0]
                or projection['support_intervals'][-1][1] == domain[1]
            )
        )
        projection['endpoint_refinement'] = checks
        _update_projection_uncertainty(projection, original_projection, checks, domain, edges[axis])
        for gap in projection.get('gap_validation', []):
            for check in gap['bins']:
                check['superseded_by_endpoint_refinement'] = _window_is_covered(
                    check.get('bounds', [float('-inf'), float('inf')]), checks
                )
        for family in group[1:]:
            result['families'][family]['projections'][axis] = deepcopy(projection)


def _update_refinement_metadata(result, groups, metadata, stored, jobs):
    """Save active and retired checks, point usage, and remaining refinement work."""
    claimed, windows, probe_positions = jobs['claimed'], jobs['windows'], jobs['probe_positions']
    kinds, initial_gap_points, cache = jobs['kinds'], jobs['initial_gap_points'], jobs['cache']

    # Summarize active checks and retain cached refinement history
    active_checks = [
        check
        for group in groups
        for projection in result['families'][group[0]]['projections'].values()
        for check in projection.get('endpoint_refinement', [])
    ]
    unresolved_gap_count = sum(
        check['status'] not in (*gaps.RESOLVED, 'disabled')
        and not check.get('superseded_by_endpoint_refinement')
        for group in groups
        for projection in result['families'][group[0]]['projections'].values()
        for gap in projection.get('gap_validation', [])
        for check in gap['bins']
    )
    retained = [record for index, record in enumerate(stored) if index not in claimed]
    metadata.update(
        cache=retained
        + [
            dict(
                family=window_key[0],
                axis=window_key[1],
                bounds=list(windows[window_key]),
                positions=list(probe_positions[window_key]),
                kind=kinds[window_key],
                initial_gap_points=initial_gap_points[window_key],
                result=window_result,
            )
            for window_key, window_result in cache.items()
        ],
        points_remaining=max(0, metadata['max_points'] - metadata['points_used']),
        windows=len(active_checks),
        unresolved_windows=sum(check['status'] not in gaps.RESOLVED for check in active_checks),
        unresolved_gap_bins=unresolved_gap_count,
        minimum_next_step_points=min(
            (check['next_step_points'] for check in active_checks if 'next_step_points' in check),
            default=None,
        ),
        stop_reasons={
            reason: sum(check['stop_reason'] == reason for check in active_checks)
            for reason in sorted({check['stop_reason'] for check in active_checks})
        },
    )


def _refresh_refinement_warnings(result, edges, metadata, envelope_only):
    """Report remaining possible support and refresh warnings and constraint classifications."""
    from system_summary import classify_constraint

    unresolved_gap_count = metadata['unresolved_gap_bins']

    # Record remaining possible support and refresh warnings
    metadata['unbracketed_possible_support'] = []
    for name, stats in result['families'].items():
        for axis, projection in stats['projections'].items():
            fragments = decide.unbracketed_possible(
                projection, [edges[axis][0], edges[axis][-1]], envelope_only
            )
            if fragments:
                metadata['unbracketed_possible_support'].append(
                    dict(
                        family=name,
                        axis=axis,
                        intervals=fragments,
                        largest_width=max(
                            interval_end - interval_start
                            for interval_start, interval_end in fragments
                        ),
                    )
                )
    metadata.setdefault(
        'original_gap_warnings',
        [
            warning
            for warning in result.get('warnings', [])
            if warning.endswith('gap bins remain unresolved after targeted validation.')
        ],
    )
    result['warnings'] = [
        warning
        for warning in result.get('warnings', [])
        if not (
            warning.endswith('gap bins remain unresolved after targeted validation.')
            or warning.endswith('gap bins remain unresolved after local endpoint refinement.')
            or warning.endswith(
                (
                    'local support windows remain unresolved.',
                    'local support windows remain unresolved; see convergence scope.',
                )
            )
            or warning.endswith('locally refined support reaches a search-domain edge.')
        )
    ]
    if unresolved_gap_count:
        result['warnings'].append(
            f'{unresolved_gap_count} gap bins remain unresolved after local endpoint refinement.'
        )
    if metadata['unresolved_windows']:
        result['warnings'].append(
            f'{metadata["unresolved_windows"]} local support windows remain unresolved.'
        )
    for name, stats in result['families'].items():
        for axis, projection in stats['projections'].items():
            if projection.get('local_domain_limited'):
                result['warnings'].append(
                    f'{name}.{axis}: locally refined support reaches a search-domain edge.'
                )
        stats['constraint_class'] = classify_constraint(stats)


# -------------------------------------------------------------------------------------------------
# Refinement coordination
# -------------------------------------------------------------------------------------------------

def refine_endpoints(result, reference, args, progress=None, execute=None, focus=None):
    """Coordinate local support refinement against a comparison grid and return its metadata.

    Prepare windows and cached probes, evaluate within the point budget, then merge support and
    uncertainty into the result in place. Refresh cached history, diagnostics, and warnings.
    """
    started = time.monotonic()

    # Prepare grids and target groups
    edges, sizes, reference_sizes, system, settings, targets, groups = _prepare_refinement_inputs(
        result, reference, args
    )

    # Configure local tolerances and restore the refinement cache
    envelope_only = args.convergence_support == 'envelope'
    options = dict(
        gaps.estimator_options(args),
        probes=5,
        levels=args.endpoint_refinement_levels,
        include_endpoints=True,
        endpoint_tolerances=dict(
            e_out=args.convergence_endpoint_eout_atol / 2,
            a_out=args.convergence_endpoint_loga_atol / 2,
            phase=args.convergence_endpoint_phase_atol_deg / 2,
        ),
    )
    metadata = result.setdefault('endpoint_refinement', dict(points_used=0, cache=[]))
    metadata['max_points'] = args.endpoint_refinement_max_points
    metadata['point_budget_unit'] = 'initial_scan_points'
    metadata['scope'] = (
        'Local endpoint and gap probes after bulk agreement; numerical brackets, '
        'not continuous feasibility or physical exclusion.'
    )
    stored = [dict(record) for record in metadata.get('cache', [])]
    jobs = _prepare_refinement_jobs(
        result, reference, args, edges, sizes, reference_sizes, groups, stored, focus
    )
    cache = jobs['cache']
    archive = _archived_refinement_probes(stored)

    def task_for(key, budget):
        """Package a window, cached probes, estimator settings, and point budget for validation."""
        family, axis, _ = key
        local_edges = dict(edges, **{axis: np.asarray(jobs['windows'][key])})
        samples = _refinement_probe_samples(
            archive, cache, family, axis, result['families'][family]['projections'][axis]
        )
        return (
            axis,
            0,
            local_edges,
            sizes,
            targets[family],
            system,
            replace(settings, require_peters_applicable=family == 'secular_targets'),
            dict(
                options,
                probe_positions=jobs['probe_positions'][key],
                previous_result=cache.get(key),
                minimum_levels=jobs['minimum_levels'][key],
                probe_cache=samples,
            ),
            budget,
        )

    # Allocate the point budget and evaluate resumable steps
    if progress:
        progress(f'  Local endpoint/gap refinement: {len(jobs["windows"])} windows')
    previous_native_costs = {
        window_key: window_result.get('points_used', 0)
        for window_key, window_result in cache.items()
        if jobs['kinds'][window_key] == 'gap_bin'
    }
    context = (
        nullcontext(execute) if execute is not None
        else reusable_task_runner(gaps._validate_indexed_task, args.workers, progress)
    )
    with context as runner:
        gaps.allocate_validation_steps(
            list(jobs['windows']), cache, metadata, task_for, runner,
        )
    metadata['gap_resume_points_used'] = metadata.get('gap_resume_points_used', 0) + sum(
        cache[window_key]['points_used'] - used
        for window_key, used in previous_native_costs.items()
    )

    # Merge support, preserve uncertainty, and refresh the report
    _merge_refined_projections(result, jobs, edges, system, options, archive, envelope_only)
    _update_refinement_metadata(result, groups, metadata, stored, jobs)
    _refresh_refinement_warnings(result, edges, metadata, envelope_only)

    # Update elapsed time and report refinement progress
    result['elapsed_seconds'] = result.get('elapsed_seconds', 0.0) + time.monotonic() - started
    if progress:
        progress(
            f'  Local refinement: {metadata["unresolved_windows"]}/{metadata["windows"]} '
            'windows unresolved.'
        )
        explanation = gaps.validation_stop_explanation(metadata["stop_reasons"])
        if explanation:
            progress(f'  {explanation}')
    return metadata


def _refinement_comparison_state(result):
    """Capture the point count and support geometry used to detect refinement progress."""
    return (
        result.get('endpoint_refinement', {}).get('points_used', 0),
        repr(
            [
                (
                    name,
                    axis,
                    projection.get('support_intervals'),
                    projection.get('possible_support_intervals'),
                    projection.get('endpoint_uncertainty'),
                    [
                        (check.get('bounds'), check['status'])
                        for check in projection.get('endpoint_refinement', [])
                    ],
                )
                for name, family in result['families'].items()
                for axis, projection in family['projections'].items()
            ]
        ),
    )


def refine_comparison(baseline, current, args, progress=None):
    """Refine both grid results while bulk agreement holds and local support improves.

    Keep each result's point budget and probe cache, record comparison passes, and report
    remaining unresolved windows and stopping reasons.
    """
    from system_summary import bulk_convergence_failures, support_convergence_failures

    started = False

    # Refine both results while the local comparison makes progress
    for index in range(MAX_LOCAL_COMPARISON_PASSES):
        if bulk_convergence_failures(baseline, current, args):
            break
        focus = refinement_focus(baseline, current, args)
        if not focus:
            break
        before = (
            _refinement_comparison_state(baseline),
            _refinement_comparison_state(current),
        )
        if not started:
            if progress:
                progress('  Refining endpoints and gaps across both compared levels')
            started = True
        refine_endpoints(baseline, current, args, None, focus=focus)
        refine_endpoints(current, baseline, args, None, focus=focus)
        failures = support_convergence_failures(baseline, current, args, args.convergence_support)
        changed = before != (
            _refinement_comparison_state(baseline),
            _refinement_comparison_state(current),
        )
        current.setdefault('local_comparison_passes', []).append(
            dict(
                pass_number=index + 1,
                support_failures=failures,
                made_progress=changed,
                points_used=[
                    grid_result.get('endpoint_refinement', {}).get('points_used', 0)
                    for grid_result in (baseline, current)
                ],
            )
        )
        if not failures or not changed:
            break

    # Report remaining unresolved work and stopping reasons
    if started and progress:
        metadata = [
            grid_result.get('endpoint_refinement', {}) for grid_result in (baseline, current)
        ]
        unresolved = sum(
            refinement_metadata.get('unresolved_windows', 0) for refinement_metadata in metadata
        )
        reasons = {}
        for item in metadata:
            for reason, count in item.get('stop_reasons', {}).items():
                reasons[reason] = reasons.get(reason, 0) + count
        explanation = gaps.validation_stop_explanation(reasons)
        progress(
            '  Endpoint refinement complete: '
            f'{unresolved} windows unresolved across both grids.'
            + (f' {explanation.strip()}' if explanation else '')
        )
