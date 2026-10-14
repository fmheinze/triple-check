"""Estimate projected support fractions by tracking compatible interval boundaries.

At a fixed orbital coordinate, scan the two transverse axes for recoil-compatible regions.
Refine interval edges by bisection, follow narrow regions across neighbouring columns, and
integrate their widths to obtain compatible area. Normalize by the stable transverse area and
return the fraction, numerical uncertainty, evaluation counts, and a compatible example.

Gap validation and endpoint refinement use these probe results to classify local support.
"""

import math
from functools import lru_cache, partial

import numpy as np

import triple_core as core

AXES = ('e_out', 'a_out', 'phase')
EDGE_METHOD = 'edge_tracking'
BISECTION_RTOL = 1e-7
THIN_AXIS = {'a_out': 'e_out', 'phase': 'e_out', 'e_out': 'a_out'}


# -------------------------------------------------------------------------------------------------
# Transverse coordinates and stable area
# -------------------------------------------------------------------------------------------------

def transverse_axes(axis):
    """Return the thin axis used for bisection and the transverse column axis."""
    thin = THIN_AXIS[axis]
    return thin, next(key for key in AXES if key not in (axis, thin))


def _evaluate_compatibility(
    axis, coordinate, thin_axis, thin_values, column_axis, column_values, targets, system, settings
):
    """Evaluate recoil compatibility at paired transverse coordinates."""
    values = {
        axis: coordinate,
        thin_axis: np.asarray(thin_values, dtype=float),
        column_axis: np.asarray(column_values, dtype=float),
    }
    separation_au, eccentricity, phase_rad = np.broadcast_arrays(
        10 ** np.asarray(values['a_out'], dtype=float),
        np.asarray(values['e_out'], dtype=float),
        np.radians(np.asarray(values['phase'], dtype=float)),
    )
    if not separation_au.size:
        return np.zeros(separation_au.shape, dtype=bool), 0

    # Apply the stability and radial-reach checks
    compatible = np.zeros(separation_au.shape, dtype=bool)
    inner_separation_au = core.inner_separation_for_stability_au(
        system.m1_inner, system.m2_inner, settings
    )
    minimum_stable_separation = core.mardling_aarseth_aout_min_au(
        inner_separation_au,
        eccentricity,
        system.m1_inner + system.m2_inner,
        system.m3,
        system.inclination_rad,
    )
    stable = (
        separation_au >= minimum_stable_separation
        if settings.require_stability
        else np.ones(separation_au.shape, dtype=bool)
    )
    if not len(targets.epost) or not stable.any():
        return compatible, int(separation_au.size)

    # Radial reach over the whole admitted family is necessary for every target
    tolerance = 32 * np.finfo(float).eps
    minimum_pericentre, maximum_apocentre = float(np.min(targets.rp_si)), float(
        np.max(targets.ra_si)
    )
    orbital_radius, _, _ = core.pre_outer_state(
        separation_au,
        eccentricity,
        core.as_true_anomaly_rad(phase_rad, eccentricity, settings.phase_measure),
        system.m1_inner + system.m2_inner,
        system.m3,
    )
    viable = (
        stable
        & (orbital_radius >= minimum_pericentre * (1 - tolerance))
        & (orbital_radius <= maximum_apocentre * (1 + tolerance))
    )

    # Evaluate the remaining compatible candidates
    if viable.any():
        region = core.compute_allowed_region(
            separation_au[viable],
            eccentricity[viable],
            phase_rad[viable],
            targets,
            system.m1_inner,
            system.m2_inner,
            system.mrem,
            system.m3,
            settings,
            system.inclination_rad,
        )
        compatible[viable] = region.recoil_compatible
    return compatible, int(separation_au.size)


def stable_domain_is_empty(axis, coordinate, edges, system, settings):
    """Return whether the stability boundary excludes the entire transverse domain."""
    if not settings.require_stability:
        return False
    eccentricity = coordinate if axis == 'e_out' else float(edges['e_out'][0])
    inner_separation_au = core.inner_separation_for_stability_au(
        system.m1_inner, system.m2_inner, settings
    )
    minimum_stable_separation = core.mardling_aarseth_aout_min_au(
        inner_separation_au,
        eccentricity,
        system.m1_inner + system.m2_inner,
        system.m3,
        system.inclination_rad,
    )
    maximum_log_separation = coordinate if axis == 'a_out' else float(edges['a_out'][-1])
    return bool(maximum_log_separation < math.log10(minimum_stable_separation))


def stable_transverse_area(axis, coordinate, edges, system, settings):
    """Return the stable transverse area at one projected coordinate.

    Use direct stability-boundary solves for separation and eccentricity probes, and cached
    midpoint quadrature for phase probes.
    """

    # Prepare the transverse domain and stability boundary
    domain_lower = {key: float(edges[key][0]) for key in AXES}
    domain_upper = {key: float(edges[key][-1]) for key in AXES}
    domain_widths = {key: domain_upper[key] - domain_lower[key] for key in AXES}
    thin_axis, column_axis = transverse_axes(axis)
    if not settings.require_stability:
        return domain_widths[thin_axis] * domain_widths[column_axis]
    inner_separation_au = core.inner_separation_for_stability_au(
        system.m1_inner, system.m2_inner, settings
    )
    eccentricity_upper = min(domain_upper['e_out'], 1.0 - 1e-15)

    def minimum_stable_log_separation(eccentricity):
        """Return the minimum stable log-separation at the supplied eccentricity."""
        return np.log10(
            core.mardling_aarseth_aout_min_au(
                inner_separation_au,
                eccentricity,
                system.m1_inner + system.m2_inner,
                system.m3,
                system.inclination_rad,
            )
        )

    # Integrate the stable area for the selected projected axis
    if axis == 'a_out':
        if float(minimum_stable_log_separation(domain_lower['e_out'])) > coordinate:
            return 0.0
        if float(minimum_stable_log_separation(eccentricity_upper)) <= coordinate:
            return domain_widths['e_out'] * domain_widths['phase']
        lower_eccentricity, upper_eccentricity = domain_lower['e_out'], eccentricity_upper
        for _ in range(100):
            midpoint = 0.5 * (lower_eccentricity + upper_eccentricity)
            if midpoint <= lower_eccentricity or midpoint >= upper_eccentricity:
                break
            lower_eccentricity, upper_eccentricity = (
                (midpoint, upper_eccentricity)
                if float(minimum_stable_log_separation(midpoint)) <= coordinate
                else (lower_eccentricity, midpoint)
            )
        return (lower_eccentricity - domain_lower['e_out']) * domain_widths['phase']
    if axis == 'e_out':
        return (
            max(
                0.0,
                domain_upper['a_out']
                - max(domain_lower['a_out'], float(minimum_stable_log_separation(coordinate))),
            )
            * domain_widths['phase']
        )
    return _phase_stable_measure(
        domain_lower['e_out'],
        eccentricity_upper,
        domain_lower['a_out'],
        domain_upper['a_out'],
        float(inner_separation_au),
        float(system.m1_inner + system.m2_inner),
        float(system.m3),
        float(system.inclination_rad),
    )


@lru_cache(maxsize=128)
def _phase_stable_measure(
    eccentricity_lower,
    eccentricity_upper,
    log_separation_lower,
    log_separation_upper,
    inner_separation_au,
    inner_mass,
    tertiary_mass,
    inclination_rad,
):
    """Integrate stable log-separation widths over eccentricity and cache the resulting area."""
    knots = np.linspace(eccentricity_lower, eccentricity_upper, 200001)
    eccentricities = 0.5 * (knots[:-1] + knots[1:])
    minimum_log_separation = np.log10(
        core.mardling_aarseth_aout_min_au(
            inner_separation_au, eccentricities, inner_mass, tertiary_mass, inclination_rad
        )
    )
    stable_widths = np.clip(
        log_separation_upper - np.maximum(log_separation_lower, minimum_log_separation), 0.0, None
    )
    return float(np.sum(stable_widths) * (knots[1] - knots[0]))


# -------------------------------------------------------------------------------------------------
# Interval tracking and integration
# -------------------------------------------------------------------------------------------------

def _bisect_interval_edges(
    column_indices, inside, outside, evaluate, iterations, relative_tolerance=0.0
):
    """Bisect paired interval edges and return inside points, outside points, and steps used.

    Stop each interval when its remaining edge brackets meet the relative width tolerance, or
    when the iteration count or floating-point midpoint resolution is reached.
    """
    inside, outside = np.array(inside, dtype=float), np.array(outside, dtype=float)
    steps_used = 0
    for _ in range(iterations):
        midpoints = 0.5 * (inside + outside)
        active_edges = (midpoints != inside) & (midpoints != outside)
        if relative_tolerance > 0:
            interval_widths = np.abs(inside[1::2] - inside[0::2])
            remaining_bracket_widths = np.abs(outside - inside).reshape(-1, 2).sum(axis=1)
            active_edges &= np.repeat(
                remaining_bracket_widths > relative_tolerance * interval_widths, 2
            )
        if not active_edges.any():
            break
        compatible = evaluate(column_indices[active_edges], midpoints[active_edges])
        move_inside, move_outside = np.zeros(active_edges.shape, bool), np.zeros(
            active_edges.shape, bool
        )
        move_inside[active_edges], move_outside[active_edges] = compatible, ~compatible
        inside = np.where(move_inside, midpoints, inside)
        outside = np.where(move_outside, midpoints, outside)
        steps_used += 1
    return inside, outside, steps_used


def _column_interval_brackets(flags, positions, domain_start, domain_end):
    """Return inside/outside edge pairs for each compatible stretch in a sampled column."""
    transitions = np.diff(np.r_[False, flags, False].astype(int))
    brackets = []
    for start_index, stop_index in zip(
        np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    ):
        brackets.append(
            (
                (
                    positions[start_index],
                    positions[start_index - 1] if start_index > 0 else domain_start,
                    start_index > 0,
                ),
                (
                    positions[stop_index - 1],
                    positions[stop_index] if stop_index < len(flags) else domain_end,
                    stop_index < len(flags),
                ),
            )
        )
    return brackets


def quadrature_sigma(widths, column_spacing):
    """Estimate the area-integration uncertainty from column widths and spacing.

    Combine partial end-cell contributions from each occupied stretch with the curvature of the
    width profile.
    """
    widths = np.asarray(widths, dtype=float)
    occupied_indices = np.flatnonzero(widths > 0.0)
    if not len(occupied_indices):
        return 0.0

    # Account for partial end cells inside the domain
    occupied = widths > 0.0
    starts = occupied & np.r_[False, ~occupied[:-1]]
    stops = occupied & np.r_[~occupied[1:], False]
    end_cell_error = 0.5 * column_spacing * float(np.sum(widths[starts]) + np.sum(widths[stops]))

    # Account for curvature in the column-width profile
    curvature_error = (
        float(np.sum(np.abs(np.diff(widths, 2)))) * column_spacing / 24.0
        if len(widths) > 2
        else 0.0
    )

    return end_cell_error + curvature_error


# -------------------------------------------------------------------------------------------------
# Column scanning, continuation, and boundary refinement
# -------------------------------------------------------------------------------------------------

def _scan_column_samples(columns, scan_positions, block, evaluate_columns):
    """Evaluate the shared scan in column blocks and return its compatibility mask."""
    initial_hits = np.zeros((columns, len(scan_positions)), dtype=bool)
    columns_per_block = max(1, block // len(scan_positions))
    for start in range(0, columns, columns_per_block):
        stop = min(columns, start + columns_per_block)
        initial_hits[start:stop] = evaluate_columns(
            np.arange(start, stop)[:, None], scan_positions[None, :]
        )
    return initial_hits


def _column_samples(index, scan_positions, initial_hits, updated_samples):
    """Return updated samples for a column, falling back to its initial scan samples."""
    return updated_samples.get(index, (scan_positions, initial_hits[index]))


def _compatible_sample_span(index, column_samples):
    """Return the outermost compatible sample positions in a column, or None."""
    positions, flags = column_samples(index)
    return (float(positions[flags].min()), float(positions[flags].max())) if flags.any() else None


def _predict_search_windows(index, occupied_columns, column_samples, scan_spacing):
    """Extrapolate search windows from nearby occupied columns and their boundary drift."""
    columns = len(occupied_columns)
    windows = []
    for step in (-1, 1):
        nearby_index = next(
            (
                candidate_index
                for candidate_index in range(index + step, columns if step > 0 else -1, step)
                if occupied_columns[candidate_index]
            ),
            None,
        )
        if nearby_index is None:
            continue
        predicted_lower, predicted_upper = _compatible_sample_span(nearby_index, column_samples)
        farther_index = next(
            (
                candidate_index
                for candidate_index in range(nearby_index + step, columns if step > 0 else -1, step)
                if occupied_columns[candidate_index]
            ),
            None,
        )
        drift = 0.0
        if farther_index is not None and _compatible_sample_span(farther_index, column_samples):
            other_lower, other_upper = _compatible_sample_span(farther_index, column_samples)
            reach = (index - nearby_index) / (nearby_index - farther_index)
            lower_shift, upper_shift = (predicted_lower - other_lower) * reach, (
                predicted_upper - other_upper
            ) * reach
            predicted_lower, predicted_upper = (
                predicted_lower + lower_shift,
                predicted_upper + upper_shift,
            )
            drift = abs(lower_shift) + abs(upper_shift)
        predicted_lower, predicted_upper = min(predicted_lower, predicted_upper), max(
            predicted_lower, predicted_upper
        )
        margin = max(scan_spacing, 2 * drift, predicted_upper - predicted_lower)
        windows.append((predicted_lower - margin, predicted_upper + margin))
    return windows


def _neighbour_interval_width(index, occupied_columns, column_samples, scan_spacing):
    """Estimate the local interval width from adjacent occupied columns or the scan spacing."""
    columns = len(occupied_columns)
    widths = [
        (
            _compatible_sample_span(side, column_samples)[1]
            - _compatible_sample_span(side, column_samples)[0]
        )
        or scan_spacing
        for side in (index - 1, index + 1)
        if 0 <= side < columns
        and occupied_columns[side]
        and _compatible_sample_span(side, column_samples)
    ]
    return min(widths) if widths else scan_spacing


def _continue_column_intervals(
    initial_hits, scan_positions, thin_bounds, scan, seed, evaluate_columns
):
    """Follow compatible intervals into neighbouring columns using local rescans.

    Reuse the initial scan and supplied seeds, extrapolate windows from occupied columns,
    and return updated samples, the initial discovery status, and missed-width estimates.
    """
    thin_lower, thin_upper = thin_bounds
    columns = len(initial_hits)
    scan_spacing = (thin_upper - thin_lower) / scan

    # Keep evolving column samples and the resampling helper together
    updated_samples = {}

    column_samples = partial(
        _column_samples,
        scan_positions=scan_positions,
        initial_hits=initial_hits,
        updated_samples=updated_samples,
    )

    def rescan(index, windows, widen=1.0, resolve=None):
        """Resample predicted windows and merge the new samples into a column.

        Choose the sampling density from the neighbouring interval width and return whether
        compatibility was found, together with the scan spacing estimate.
        """
        spans = [
            (
                max(thin_lower, interval_start - widen * scan_spacing),
                min(thin_upper, interval_end + widen * scan_spacing),
            )
            for interval_start, interval_end in windows
        ]
        reach = sum(interval_end - interval_start for interval_start, interval_end in spans)
        sample_count = max(16, scan // 8)
        if resolve:
            sample_count = int(
                min(max(sample_count, math.ceil(6 * reach / (resolve / 16.0))), 16 * scan)
            )
        points = np.unique(
            np.concatenate(
                [
                    np.linspace(interval_start, interval_end, sample_count)
                    for interval_start, interval_end in spans
                ]
            )
        )
        flags = evaluate_columns(np.full(points.shape, index), points)
        base_positions, base_flags = column_samples(index)
        order = np.argsort(np.concatenate([base_positions, points]), kind='stable')
        merged_positions = np.concatenate([base_positions, points])[order]
        merged_flags = np.concatenate([base_flags, flags])[order]
        unique_positions = np.r_[True, np.diff(merged_positions) > 0]
        updated_samples[index] = (
            merged_positions[unique_positions],
            merged_flags[unique_positions],
        )
        return bool(flags.any()), reach / max(sample_count - 1, 1)

    # Continue discovered intervals into neighbouring columns
    seed_intervals = [
        [float(interval_start), float(interval_end)]
        for interval_start, interval_end in (seed or [])
    ]
    occupied_columns = initial_hits.any(axis=1)
    if not occupied_columns.any() and seed_intervals:
        for index in range(columns):
            occupied_columns[index], _ = rescan(index, seed_intervals, widen=4.0)
    scan_limited = not occupied_columns.any()
    rescan_attempts = np.zeros(columns, dtype=int)
    missed_widths = {}
    while occupied_columns.any() and not occupied_columns.all():
        frontier_columns = [
            index
            for index in range(columns)
            if not occupied_columns[index]
            and rescan_attempts[index] < 2
            and any(
                0 <= side < columns and occupied_columns[side] for side in (index - 1, index + 1)
            )
        ]
        if not frontier_columns:
            break
        for index in frontier_columns:
            rescan_attempts[index] += 1
            windows = _predict_search_windows(index, occupied_columns, column_samples, scan_spacing)
            if not windows:
                continue
            occupied_columns[index], rescan_spacing = rescan(
                index,
                windows,
                widen=1.0 if rescan_attempts[index] == 1 else 8.0,
                resolve=_neighbour_interval_width(
                    index, occupied_columns, column_samples, scan_spacing
                ),
            )
            if occupied_columns[index]:
                missed_widths.pop(index, None)
            else:
                missed_widths[index] = rescan_spacing

    return updated_samples, scan_limited, missed_widths


def _refine_column_boundaries(
    columns, column_samples, thin_bounds, evaluate_columns, bisection_iterations, bisection_rtol
):
    """Bisect interval boundaries and return column widths, edge errors, and diagnostics."""
    thin_lower, thin_upper = thin_bounds

    # Collect and bisect interval boundary pairs
    column_indices, inside_edges, outside_edges, open_edges = [], [], [], []
    for index in range(columns):
        positions, flags = column_samples(index)
        if not flags.any():
            continue
        for (left_in, left_out, left_open), (
            right_in,
            right_out,
            right_open,
        ) in _column_interval_brackets(flags, positions, thin_lower, thin_upper):
            column_indices += [index, index]
            inside_edges += [left_in, right_in]
            outside_edges += [left_out, right_out]
            open_edges += [left_open, right_open]
    widths = np.zeros(columns)
    width_errors = np.zeros(columns)
    bisection_steps = 0
    max_intervals_per_column, thin_range, intervals = 0, None, []
    if column_indices:
        column_indices = np.asarray(column_indices, dtype=int)
        inside_edges, outside_edges = np.asarray(inside_edges, float), np.asarray(
            outside_edges, float
        )
        refinable_edges = np.asarray(open_edges, dtype=bool)
        outside_edges[~refinable_edges] = inside_edges[~refinable_edges]
        inside_edges, outside_edges, bisection_steps = _bisect_interval_edges(
            column_indices,
            inside_edges,
            outside_edges,
            evaluate_columns,
            bisection_iterations,
            bisection_rtol,
        )
        np.add.at(width_errors, column_indices, np.abs(outside_edges - inside_edges))
        intervals = [
            (int(index), min(left, right), max(left, right))
            for index, left, right in zip(
                column_indices[0::2].tolist(),
                inside_edges[0::2].tolist(),
                inside_edges[1::2].tolist(),
            )
        ]
        for index, left, right in intervals:
            widths[index] += right - left
        max_intervals_per_column = int(np.bincount(column_indices[0::2], minlength=columns).max())
        thin_range = [float(inside_edges.min()), float(inside_edges.max())]

    return {
        'widths': widths,
        'width_errors': width_errors,
        'bisection_steps': bisection_steps,
        'thin_range': thin_range,
        'max_intervals_per_column': max_intervals_per_column,
        'intervals': intervals,
    }


def _track_column_intervals(
    axis,
    coordinate,
    thin_axis,
    column_axis,
    column_values,
    edges,
    targets,
    system,
    settings,
    scan,
    seed,
    bisection_iterations,
    block,
    bisection_rtol=0.0,
):
    """Track compatible intervals across columns and return widths, edge errors, and diagnostics.

    Start with a shared scan, resample predicted windows from neighbouring columns or seeds, and
    bisect the discovered interval boundaries.
    """

    # Prepare the initial transverse scan
    thin_lower, thin_upper = float(edges[thin_axis][0]), float(edges[thin_axis][-1])
    columns = len(column_values)

    scan_positions = np.r_[
        thin_lower,
        thin_lower + (np.arange(scan) + 0.5) * (thin_upper - thin_lower) / scan,
        thin_upper,
    ]
    evaluation_count = 0

    def evaluate_columns(column_indices, thin_values):
        """Evaluate selected columns at thin-axis positions and accumulate the point count."""
        nonlocal evaluation_count
        flags, cost = _evaluate_compatibility(
            axis,
            coordinate,
            thin_axis,
            thin_values,
            column_axis,
            column_values[np.asarray(column_indices, dtype=int)],
            targets,
            system,
            settings,
        )
        evaluation_count += cost
        return flags

    # Scan the full column range and continue intervals into neighbouring columns
    initial_hits = _scan_column_samples(columns, scan_positions, block, evaluate_columns)
    updated_samples, scan_limited, missed_widths = _continue_column_intervals(
        initial_hits, scan_positions, (thin_lower, thin_upper), scan, seed, evaluate_columns
    )

    column_samples = partial(
        _column_samples,
        scan_positions=scan_positions,
        initial_hits=initial_hits,
        updated_samples=updated_samples,
    )

    # Refine interval boundaries and record the evaluations used by bisection
    evaluations_before_bisection = evaluation_count
    refined = _refine_column_boundaries(
        columns,
        column_samples,
        (thin_lower, thin_upper),
        evaluate_columns,
        bisection_iterations,
        bisection_rtol,
    )
    bisection_evaluations = evaluation_count - evaluations_before_bisection

    # Combine interval measurements and sampling diagnostics
    return {
        'widths': refined['widths'],
        'width_errors': refined['width_errors'],
        'bisection_evaluations': bisection_evaluations,
        'bisection_steps': refined['bisection_steps'],
        'evaluations': evaluation_count,
        'missed': missed_widths,
        'thin_range': refined['thin_range'],
        'scan_limited': bool(scan_limited),
        'max_intervals_per_column': refined['max_intervals_per_column'],
        'intervals': refined['intervals'],
    }


# -------------------------------------------------------------------------------------------------
# Occupancy probes and reusable seeds
# -------------------------------------------------------------------------------------------------

def _column_midpoints(range_start, range_end, columns):
    """Return uniformly spaced midpoint coordinates for the requested range and column count."""
    return range_start + (np.arange(columns) + 0.5) * (range_end - range_start) / columns


def estimate_occupancy(
    axis,
    coordinate,
    edges,
    targets,
    system,
    settings,
    columns,
    scan,
    seed=None,
    bisection_iterations=52,
    block=131072,
    support_sweeps=2,
    bisection_rtol=BISECTION_RTOL,
):
    """Estimate the compatible transverse fraction at one fixed orbital coordinate.

    Track interval boundaries across columns, concentrate additional sweeps on the occupied
    column range, and divide the integrated compatible area by the stable area. Return a probe
    record with the fraction, integration uncertainty, remaining edge error, compatible example,
    and sampling diagnostics.
    """

    # Validate settings and prepare the transverse axes
    if not math.isfinite(bisection_rtol) or bisection_rtol < 0:
        raise ValueError('bisection_rtol must be finite and nonnegative')
    thin_axis, column_axis = transverse_axes(axis)
    column_low, column_high = float(edges[column_axis][0]), float(edges[column_axis][-1])
    columns, scan = max(int(columns), 2), max(int(scan), 8)

    # Handle an empty stable domain
    if stable_domain_is_empty(axis, coordinate, edges, system, settings):
        return {
            'coordinate': float(coordinate),
            'method': EDGE_METHOD,
            'physical_cells': 0,
            'compatible_cells': 0,
            'physical_measure': 0.0,
            'compatible_measure': 0.0,
            'physical_domain_empty': True,
            'fraction': None,
            'fraction_sigma': None,
            'fraction_error_bound': 0.0,
            'witness': None,
            'bisection_measure_bound': 0.0,
            'bisection_evaluations': 0,
            'bisection_steps': 0,
            'bisection_rtol': bisection_rtol,
            'columns': columns,
            'scan_points': scan,
            'evaluations': 0,
            'column_bounds': [column_low, column_high],
            'sweeps': 0,
            'thin_axis': thin_axis,
            'thin_range': None,
            'scan_limited': False,
            'columns_not_tracked': 0,
            'unseen_measure_bound': 0.0,
            'max_intervals_per_column': 0,
        }

    # Track intervals and concentrate subsequent sweeps on occupied columns
    bounds = (column_low, column_high)
    sweep = _track_column_intervals(
        axis,
        coordinate,
        thin_axis,
        column_axis,
        _column_midpoints(*bounds, columns),
        edges,
        targets,
        system,
        settings,
        scan,
        seed,
        bisection_iterations,
        block,
        bisection_rtol,
    )
    bisection_evaluations = sweep['bisection_evaluations']
    bisection_steps = sweep['bisection_steps']
    evaluations, sweeps = sweep['evaluations'], 1
    for _ in range(max(0, support_sweeps - 1)):
        occupied_columns = np.flatnonzero(sweep['widths'] > 0.0)
        if not len(occupied_columns):
            break
        column_spacing = (bounds[1] - bounds[0]) / columns
        column_positions = _column_midpoints(*bounds, columns)
        refined_bounds = (
            max(column_low, float(column_positions[occupied_columns[0]]) - column_spacing),
            min(column_high, float(column_positions[occupied_columns[-1]]) + column_spacing),
        )
        if refined_bounds[1] - refined_bounds[0] >= 0.5 * (bounds[1] - bounds[0]):
            break  # Already resolved: refining the same count buys nothing.
        refined_sweep = _track_column_intervals(
            axis,
            coordinate,
            thin_axis,
            column_axis,
            _column_midpoints(*refined_bounds, columns),
            edges,
            targets,
            system,
            settings,
            scan,
            seed_intervals_from_probes(dict(sweep, method=EDGE_METHOD)),
            bisection_iterations,
            block,
            bisection_rtol,
        )
        bisection_evaluations += refined_sweep['bisection_evaluations']
        bisection_steps = max(bisection_steps, refined_sweep['bisection_steps'])
        evaluations += refined_sweep['evaluations']
        sweeps += 1
        bounds, sweep = refined_bounds, refined_sweep

    # Integrate compatible area and combine numerical uncertainties
    widths = sweep['widths']
    column_spacing = (bounds[1] - bounds[0]) / columns
    compatible_area = float(np.sum(widths) * column_spacing)
    stable_area = stable_transverse_area(axis, coordinate, edges, system, settings)
    compatible_fraction = compatible_area / stable_area if stable_area > 0 else None
    unseen_area_bound = float(sum(sweep['missed'].values())) * column_spacing
    boundary_area_error = float(np.sum(sweep['width_errors']) * column_spacing)
    quadrature_error = quadrature_sigma(widths, column_spacing) + (7.0 / 6.0) * boundary_area_error
    fraction_uncertainty = (
        (quadrature_error + unseen_area_bound) / stable_area if stable_area > 0 else None
    )

    # Record a compatible example and assemble the probe result
    witness = None
    if sweep['intervals']:
        column_positions = _column_midpoints(*bounds, columns)
        index, left, right = max(sweep['intervals'], key=lambda item: item[2] - item[1])
        witness = _compatible_witness(
            axis,
            coordinate,
            thin_axis,
            0.5 * (left + right),
            column_axis,
            float(column_positions[index]),
            targets,
            system,
            settings,
        )
    return {
        'coordinate': float(coordinate),
        'method': EDGE_METHOD,
        'physical_cells': columns if stable_area > 0 else 0,
        'compatible_cells': int(np.count_nonzero(widths > 0)),
        'compatible_measure': compatible_area,
        'physical_measure': float(stable_area),
        'fraction': compatible_fraction,
        'fraction_sigma': fraction_uncertainty,
        'witness': witness,
        'fraction_error_bound': boundary_area_error / stable_area if stable_area > 0 else 0.0,
        'bisection_measure_bound': boundary_area_error,
        'bisection_rtol': bisection_rtol,
        'bisection_evaluations': bisection_evaluations,
        'bisection_steps': bisection_steps,
        'columns': columns,
        'scan_points': scan,
        'evaluations': evaluations,
        'column_bounds': [float(bounds[0]), float(bounds[1])],
        'sweeps': sweeps,
        'thin_axis': thin_axis,
        'thin_range': sweep['thin_range'],
        'scan_limited': bool(sweep['scan_limited']),
        'columns_not_tracked': len(sweep['missed']),
        'unseen_measure_bound': unseen_area_bound,
        'max_intervals_per_column': sweep['max_intervals_per_column'],
    }


def _compatible_witness(
    axis, coordinate, thin_axis, thin_value, column_axis, column_value, targets, system, settings
):
    """Return a compatible orbital configuration and its recoil and eccentricity values, or None."""

    # Prepare broadcast coordinates
    values = {
        axis: float(coordinate),
        thin_axis: float(thin_value),
        column_axis: float(column_value),
    }
    separation_au, eccentricity, phase_rad = (
        10 ** values['a_out'],
        values['e_out'],
        math.radians(values['phase']),
    )
    region = core.compute_allowed_region(
        np.array([separation_au]),
        np.array([eccentricity]),
        np.array([phase_rad]),
        targets,
        system.m1_inner,
        system.m2_inner,
        system.mrem,
        system.m3,
        settings,
        system.inclination_rad,
    )
    if not bool(region.recoil_compatible[0]):
        return None
    return {
        'a_out_au': float(separation_au),
        'e_out': float(eccentricity),
        'phase_deg': float(values['phase']),
        'recoil_kms': float(region.vk_min_kms[0]),
        'e_post': float(region.epost_star[0]),
        'e_ref': float(region.eref_star[0]),
    }


def seed_intervals_from_probes(*samples):
    """Combine tracked interval ranges from neighbouring probes into a seed for the next scan."""
    spans = [
        sample['thin_range']
        for sample in samples
        if sample and sample.get('method') == EDGE_METHOD and sample.get('thin_range')
    ]
    if not spans:
        return None
    return [[min(lower for lower, _ in spans), max(upper for _, upper in spans)]]


def estimated_probe_cost(columns, scan):
    """Return the reserved point count for the initial column scan, including domain endpoints."""
    return int(max(int(columns), 2) * (max(int(scan), 8) + 2))
