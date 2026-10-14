#!/usr/bin/env python3
"""Report system constraints without storing a minimum-recoil-velocity map.

Volumes use equal cells in (e_out, log10(a_out), phase). The default phase is mean anomaly, so
uniform phase represents uniform time on the outer orbit. Uniform refinement grows all dimensions
together. Mirrored phases are computed once and restored before accumulating cell counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import sys
import time

import numpy as np

import support_decision as decide
import triple_core as core
from triple_core import completed_tasks
from triple_io import format_report, format_posterior_report, write_report
from triple_cli import (
    is_posterior_mode,
    parse_system_summary_args as parse_args,
    validate_system_summary_args as validate_args,
    settings_from_args,
    system_from_args,
)


# -------------------------------------------------------------------------------------------------
# Resolution settings and refinement schedule
# -------------------------------------------------------------------------------------------------

# Resolution sequence in (outer eccentricity, log separation, phase, post-merger eccentricity)
DEFAULT_RESOLUTION_SEQUENCE = (
    (75, 270, 270, 3750),
    (100, 360, 360, 5000),
    (125, 450, 450, 6250),
    (150, 540, 540, 7500),
    (200, 720, 720, 10000),
    (250, 900, 900, 12500),
)


@dataclass(frozen=True)
class Resolution:
    n_eout: int
    n_aout: int
    n_nu: int
    n_epost: int

    @property
    def points(self):
        return self.n_eout * self.n_aout * self.n_nu

    def refined(self, refinement_factor):
        return Resolution(**{key: max(value + 1, math.ceil(value * refinement_factor))
                             for key, value in asdict(self).items()})


class RefinementSchedule:
    def __init__(self, initial, refinement_factor, required_streak, sequence=None):
        self.sequence = sequence
        self.anchor = initial
        self.refinement_factor = refinement_factor
        self.required_streak = required_streak
        self.anchor_index = None
        self.streak = 0
        self.done = False

    def next_resolution(self):
        if self.sequence:
            index = 0 if self.anchor_index is None else self.anchor_index + 1
            return self.sequence[index] if index < len(self.sequence) else None
        return (self.anchor if self.anchor_index is None
                else self.anchor.refined(self.refinement_factor))

    def record_evaluation(self, index, resolution, failures):
        initial = self.anchor_index is None
        self.anchor, self.anchor_index = resolution, index
        if not initial:
            self.streak = 0 if failures else self.streak + 1
            self.done = self.streak >= self.required_streak
        return True

    def pending(self):
        return [] if self.done else ['uniform_confirmation']


def make_axes(args, resolution):
    """Equal-width cells, sampled at their centers, with no duplicate phase endpoint."""
    edges = {
        'e_out': np.linspace(args.eout_min, args.eout_max, resolution.n_eout + 1),
        'a_out': np.linspace(math.log10(args.aout_min_au), math.log10(args.aout_max_au),
                             resolution.n_aout + 1),
        'phase': np.linspace(args.nu_min_deg, args.nu_max_deg, resolution.n_nu + 1),
    }
    centers = {key: (value[:-1] + value[1:]) / 2 for key, value in edges.items()}
    return edges, centers


# -------------------------------------------------------------------------------------------------
# Projected support geometry and statistics
# -------------------------------------------------------------------------------------------------

def covering_arc(support, circular):
    """Smallest cell-edge envelope, as (start index, number of bins)."""
    idx = np.flatnonzero(support)
    n = len(support)
    if not len(idx):
        return None
    if not circular or len(idx) == n:
        return int(idx[0]), int(idx[-1] - idx[0] + 1)
    gaps = np.diff(np.r_[idx, idx[0] + n]) - 1
    largest = int(np.argmax(gaps))
    return int(idx[(largest + 1) % len(idx)]), int(n - gaps[largest])


def support_geometry(support, edges, circular):
    """Cell-edge interval geometry for a sampled or gap-validated support mask."""
    n = len(support)
    runs = decide.true_runs(support)
    arc = covering_arc(support, circular)
    span = float(edges[-1] - edges[0])
    envelope = None
    if arc:
        start, width = arc
        end = start + width
        envelope = {'start': float(edges[start]),
                    'end': float(edges[end % n] if end > n else edges[end]),
                    'wraps': bool(end > n), 'width': span * width / n}
    return {
        'support_coverage': float(np.mean(support)),
        'envelope_coverage': arc[1] / n if arc else 0.0,
        'support_intervals': [[float(edges[start]), float(edges[end])] for start, end in runs],
        'envelope': envelope, 'support_width': span * float(np.mean(support)),
        'normalized_intervals': [[start / n, end / n] for start, end in runs],
        'normalized_envelope': ([arc[0] / n, (arc[0] + arc[1]) / n] if arc else None),
    }


def projection_summary(mass, physical_mass, edges, circular, args):
    """Three-valued projected support: decided support, and the bins beside it.

    Counting cell centers cannot decide a bin whose occupancy sits inside the aliasing spread of
    its own count, and calling such a bin supported or empty is what makes a projected endpoint
    march with resolution. Those bins are undecided instead. Transitions carry center-to-center
    positional brackets, spanning any adjacent undecided cells; targeted probes then narrow them.
    """
    supported, undecided, _ = decide.classify(mass, physical_mass,
                                              args.projection_min_transverse_fraction,
                                              getattr(args, 'projection_support_sigma', 2.),
                                              args.projection_min_transverse_cells)
    possible = supported | undecided
    geometry = support_geometry(supported, edges, circular)
    outer = support_geometry(possible, edges, circular)
    return {
        **geometry,
        'possible_support_intervals': outer['support_intervals'],
        'undecided_coverage': float(np.mean(undecided)),
        'undecided_intervals': [[float(edges[start]), float(edges[end])]
                                for start, end in decide.true_runs(undecided)],
        'endpoint_uncertainty': [b['bounds'] for b in
                                 decide.endpoint_brackets(supported, undecided, edges, circular)],
        'touches_lower_edge': bool(mass[0] > 0), 'touches_upper_edge': bool(mass[-1] > 0),
        'circular': circular,
    }


def classify_constraint(stats):
    """Ordered operational strength classes; Peters reliability is separate."""
    if stats['physical_cells'] == 0:
        return 'no_physical_domain'
    if stats['compatible_cells'] == 0:
        return 'peters-incompatible'
    f = stats['volume_fraction']
    ce, ca, cp = (stats['projections'][key]['support_coverage']
                  for key in ('e_out', 'a_out', 'phase'))
    if f < 0.01 and ca < 0.20 and min(ce, cp) < 0.30:
        return 'very_strong'
    if f < 0.03 and ca < 0.30 and min(ce, cp) < 0.50:
        return 'strong'
    if f < 0.10 or ca < 0.50:
        return 'moderate'
    return 'weak'


# -------------------------------------------------------------------------------------------------
# Grid evaluation and result accumulation
# -------------------------------------------------------------------------------------------------

def empty_accumulator(resolution):
    """Initializes an accumulator."""
    return {
        'physical_cells': 0, 'compatible_cells': 0, 'reachable_physical_cells': 0,
        'minimum_recoil_kms': None,
        'allowed': {key: np.zeros(n, dtype=np.int64) for key, n in
                    zip(('e_out', 'a_out', 'phase'),
                        (resolution.n_eout, resolution.n_aout, resolution.n_nu))},
        'physical': {key: np.zeros(n, dtype=np.int64) for key, n in
                     zip(('e_out', 'a_out', 'phase'),
                         (resolution.n_eout, resolution.n_aout, resolution.n_nu))},
    }


def accumulate(acc, region, start, stop):
    """Update  the accumulator ``acc`` in place with one computed grid block's cell counts,
    per-axis counts, and minimum recoil. start:stop selects eccentricity bins, since each block
    covers all separation and phase bins.
    """
    allowed = region.recoil_compatible
    physical = region.dynamically_stable
    reachable = physical & region.reachable
    acc['physical_cells'] += int(physical.sum())
    acc['compatible_cells'] += int(allowed.sum())
    acc['reachable_physical_cells'] += int(reachable.sum())
    if reachable.any():
        minimum = float(np.min(region.vk_min_kms[reachable]))
        acc['minimum_recoil_kms'] = (
            min(minimum, acc['minimum_recoil_kms'])
            if acc['minimum_recoil_kms'] is not None else minimum
        )
    for name, mask in (('allowed', allowed), ('physical', physical)):
        acc[name]['e_out'][start:stop] += mask.sum(axis=(1, 2))
        acc[name]['a_out'] += mask.sum(axis=(0, 2))
        acc[name]['phase'] += mask.sum(axis=(0, 1))


def finish_summary(acc, edges, args, rg_au):
    """Turn accumulated grid counts into the final summary. Add the compatible fraction,
    constraint strength, per-axis projections, units, and constraint classification.
    """
    n, allowed = acc['physical_cells'], acc['compatible_cells']
    fraction = allowed / n if n else None
    circular = math.isclose(args.nu_max_deg - args.nu_min_deg, 360, abs_tol=1e-9)
    result = {key: acc[key] for key in (
        'physical_cells', 'compatible_cells', 'reachable_physical_cells', 'minimum_recoil_kms'
    )}
    result.update(volume_fraction=fraction,
                  strength=-math.log10(fraction) if fraction is not None and fraction > 0 else None,
                  strength_is_infinite=(fraction == 0),
                  projections={})
    for key in edges:
        projection = projection_summary(
            acc['allowed'][key], acc['physical'][key], edges[key], circular and key == 'phase', args
        )
        projection['units'] = {'a_out': 'log10(AU)', 'e_out': 'dimensionless', 'phase': 'deg'}[key]
        if key == 'a_out':
            projection['support_intervals_au'] = [
                [10**lo, 10**hi] for lo, hi in projection['support_intervals']
            ]
            projection['support_intervals_rg'] = [
                [10**lo / rg_au, 10**hi / rg_au] for lo, hi in projection['support_intervals']
            ]
        result['projections'][key] = projection
    result['constraint_class'] = classify_constraint(result)
    return result


def candidate_diagnostics(targets):
    def bounds(values):
        return [float(np.min(values)), float(np.max(values))] if len(values) else None
    return {'count': len(targets.epost), 'e_post_range': bounds(targets.epost),
            'e_ref_range': bounds(targets.e_ref)}


def _evaluate_grid_block(task):
    """Reuse mirrored phases, count the full block, and return compact reductions."""
    start, stop, centers, targets, system, settings = task
    resolution = Resolution(stop - start, len(centers['a_out']), len(centers['phase']), 1)
    accumulators = {key: empty_accumulator(resolution) for key in targets}
    a = 10**centers['a_out'][None, :, None]
    e = centers['e_out'][start:stop, None, None]
    phase = np.radians(centers['phase'][None, None, :])
    all_region = None
    for key, family in targets.items():
        if key == 'secular_targets' and len(family.epost) == len(targets['all_targets'].epost):
            region = all_region
        else:
            region = core.compute_allowed_region(
                a, e, phase, family, system.m1_inner, system.m2_inner, system.mrem, system.m3,
                settings=replace(settings, require_peters_applicable=(key == 'secular_targets')),
                inclination_rad=system.inclination_rad,
                phase_axis=2,
            )
        accumulate(accumulators[key], region, 0, stop - start)
        if key == 'all_targets':
            all_region = region
    return start, stop, accumulators


def _merge_grid_block(accumulators, start, stop, block):
    """Merge one block's counts and minimum recoil into the full-grid accumulators in place."""
    for family, local in block.items():
        acc = accumulators[family]
        for key in ('physical_cells', 'compatible_cells', 'reachable_physical_cells'):
            acc[key] += local[key]
        minimum = local['minimum_recoil_kms']
        if minimum is not None:
            acc['minimum_recoil_kms'] = (minimum if acc['minimum_recoil_kms'] is None
                                       else min(minimum, acc['minimum_recoil_kms']))
        for name in ('allowed', 'physical'):
            acc[name]['e_out'][start:stop] += local[name]['e_out']
            for axis in ('a_out', 'phase'):
                acc[name][axis] += local[name][axis]


def evaluate_resolution(args, resolution, progress=None):
    """Coordinate evaluation of one complete grid at the chosen resolution.

    Prepare the candidates and grid, split the grid into blocks, send them to workers, and merge
    their results. Build the final summaries, check projection gaps, and determine the overall
    verdict. The block functions handle the individual pieces; this function manages the whole
    calculation.
    """
    # Set up the system and candidate groups
    started = time.monotonic()
    system = system_from_args(args)
    settings = settings_from_args(args)
    candidates = core.compute_post_first_merger_candidates(
        core.make_epost_grid(resolution.n_epost, args.one_minus_epost_min),
        system.mrem, system.m3, system.delay_days, args.f_ref_hz,
        args.min_peters_orbits, args.min_peters_rp_rg,
    )
    targets = {
        'all_targets': core.select_admitted_targets(candidates,
            replace(settings, require_peters_applicable=False)),
        'secular_targets': core.select_admitted_targets(candidates,
            replace(settings, require_peters_applicable=True)),
    }

    # Build the grid and divide it into blocks
    edges, centers = make_axes(args, resolution)
    accumulators = {key: empty_accumulator(resolution) for key in targets}
    block_size = max(1, 65536 // (resolution.n_aout * resolution.n_nu))
    # Small grids must still have enough blocks to use the requested workers
    block_size = min(block_size, math.ceil(resolution.n_eout / args.workers))
    tasks = [(start, min(resolution.n_eout, start + block_size), centers, targets, system, settings)
             for start in range(0, resolution.n_eout, block_size)]

    # Evaluate the blocks and merge their results
    last_progress = time.monotonic()
    completed_layers = 0
    for start, stop, block in completed_tasks(_evaluate_grid_block, tasks, args.workers, progress):
        _merge_grid_block(accumulators, start, stop, block)
        completed_layers += stop - start
        if progress and (
            completed_layers == resolution.n_eout or time.monotonic() - last_progress >= 10
        ):
            progress(f'  Evaluated {completed_layers}/{resolution.n_eout} eccentricity layers')
            last_progress = time.monotonic()

    # Build summaries and check projection gaps
    families = {key: finish_summary(acc, edges, args, system.rg_post_au)
                for key, acc in accumulators.items()}
    from gap_validation import validate_projection_gaps
    gap_checks = validate_projection_gaps(
        families, edges, targets, system, settings, args, progress
    )

    # Determine the overall reliability verdict
    n_all, n_sec = (families[key]['compatible_cells'] for key in ('all_targets', 'secular_targets'))
    r_sec = n_sec / n_all if n_all else None
    verdict = (core.VERDICT_NO_COMPATIBLE_PROGENITOR if r_sec is None else
               core.VERDICT_PETERS_RELIABLE if r_sec >= settings.r_sec_threshold else
               core.VERDICT_RECOIL_COMPATIBLE_UNRELIABLE)

    # Collect warnings about resolution and search limits
    warnings = []
    if gap_checks['unresolved_bins']:
        warnings.append(
            f"{gap_checks['unresolved_bins']} gap bins remain unresolved after targeted validation."
        )
    if not families['all_targets']['physical_cells']:
        warnings.append('No physical-domain cells were sampled; volume fractions are undefined.')
    for key, stats in families.items():
        if not stats['compatible_cells']:
            warnings.append(
                f'{key}: no compatible progenitor was sampled; this alone does not prove exclusion.'
            )
        elif any(not axis['support_intervals'] for axis in stats['projections'].values()):
            warnings.append(
                f'{key}: a nonempty region has an empty filtered projection; '
                'its class may be under-resolved.'
            )
        for name, axis in stats['projections'].items():
            if name == 'a_out' and (axis['touches_lower_edge'] or axis['touches_upper_edge']):
                warnings.append(
                    f'{key}: support reaches an a_out search edge; '
                    'constraints may depend on domain limits.'
                )
            if name == 'e_out' and axis['touches_upper_edge']:
                warnings.append(f'{key}: support reaches the upper e_out search edge.')
            if (name == 'phase' and not axis['circular']
                    and (axis['touches_lower_edge'] or axis['touches_upper_edge'])):
                warnings.append(f'{key}: support reaches a partial-phase search edge.')

    # Return the results for this resolution
    return {
        'resolution': asdict(resolution), 'grid_points': resolution.points,
        'elapsed_seconds': time.monotonic() - started,
        'targets': {key: candidate_diagnostics(value) for key, value in targets.items()},
        'families': families, 'r_sec': r_sec, 'reliability_verdict': verdict,
        'selected_family': (
            'secular_targets' if settings.require_peters_applicable else 'all_targets'
        ),
        'warnings': warnings, 'gap_validation': gap_checks,
    }


# -------------------------------------------------------------------------------------------------
# Convergence checks and diagnostics
# -------------------------------------------------------------------------------------------------

def interval_endpoint_failure(old, new, tolerance):
    """Check whether old and new support intervals agree within the tolerance.

    Validate intervals on a zero-to-one scale, check that their structure is unchanged, and compare
    endpoints while accounting for circular wrapping. Return None if they agree, or a reason for
    invalid intervals, changed structure, or endpoints that moved too far.
    """
    def components(projection):
        """Validate intervals, join circular wraparound, and flag full-axis support."""
        intervals = np.asarray(projection.get('normalized_intervals'), dtype=float)
        if intervals.size == 0:
            return [], False
        if (intervals.ndim != 2 or intervals.shape[1] != 2
                or not np.all(np.isfinite(intervals))
                or np.any(intervals < 0) or np.any(intervals > 1)
                or np.any(intervals[:, 0] >= intervals[:, 1])
                or np.any(intervals[1:, 0] <= intervals[:-1, 1])):
            raise ValueError('Invalid support intervals')
        values = intervals.tolist()
        full = values == [[0., 1.]]
        if (projection.get('circular', False) and len(values) > 1
                and values[0][0] == 0 and values[-1][1] == 1):
            values = values[1:-1] + [[values[-1][0], values[0][1] + 1]]
        return values, full

    try:
        left, left_full = components(old)
        right, right_full = components(new)
    except (TypeError, ValueError):
        return 'invalid_intervals'
    circular = old.get('circular', False)
    if circular != new.get('circular', False):
        return 'interval_topology'
    if len(left) != len(right) or (circular and left_full != right_full):
        return 'interval_topology'
    if not left:
        return None
    left, right = np.asarray(left), np.asarray(right)
    # Linear components have a unique order; circular ones may cross the seam
    for offset in range(len(right) if circular else 1):
        candidate = np.roll(right, offset, axis=0)
        distances = np.abs(left - candidate)
        if circular:
            distances = np.abs((left - candidate + .5) % 1 - .5)
            # Distinguish a short arc from its almost-full-circle complement
            if np.any(
                np.abs(np.diff(left, axis=1) - np.diff(candidate, axis=1)) > 2 * tolerance + 1e-12
            ):
                continue
        if np.all(distances <= tolerance + 1e-12):
            return None
    return 'interval_endpoints'


def circular_envelope_distance(left, right):
    """Measure how far two covering arcs differ, accounting for circular wrapping.

    Return the largest distance from a point in either arc to the nearest point in the other
    (Hausdorff distance), on a circle of circumference one. This checks whether the overall
    supported phase range stays stable as the grid is refined; gaps inside the covering arcs
    are not compared.
    """

    # Helpers for a single point on the circle
    def contains(arc, position):
        """Check whether the position lies inside the arc, including wraparound."""
        return (position - arc[0]) % 1 <= arc[1] - arc[0] + 1e-12

    def point_distance(position, arc):
        """Return the shortest circular distance from a point to an arc."""
        if contains(arc, position):
            return 0.
        return min(abs((position - endpoint + .5) % 1 - .5) for endpoint in arc)

    # Measure the largest distance from the source arc to the target arc
    def directed_distance(source, target):
        """Find the source point farthest from the target arc."""
        target_width = target[1] - target[0]
        if target_width >= 1 - 1e-12:
            return 0.

        # The maximum occurs at a source endpoint or the target gap's midpoint
        candidates = [source[0], source[1]]
        gap_midpoint = (target[1] + (1 - target_width) / 2) % 1
        if contains(source, gap_midpoint):
            candidates.append(gap_midpoint)
        return max(point_distance(position, target) for position in candidates)

    # Check both directions so either arc can contribute the largest mismatch
    return max(directed_distance(left, right), directed_distance(right, left))


def envelope_endpoint_failure(old, new, tolerance):
    """Check whether old and new support envelopes agree within the tolerance.

    Validate the envelopes and compare linear endpoints or circular covering arcs to check
    convergence. Return None if they agree, or a failure reason for invalid bounds, mismatched
    emptiness or axis type, or excessive movement.
    """
    def envelope(projection):
        """Validate and return normalized envelope bounds, or None for empty support."""
        value = projection['normalized_envelope']
        if value is None:
            return None
        bounds = np.asarray(value, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError('Invalid envelope')
        start, end = bounds
        width = end-start
        circular = projection.get('circular', False)
        if (start < 0 or start >= 1 or width <= 0 or width > 1 + 1e-12
                or (not circular and end > 1)):
            raise ValueError('Invalid envelope')
        return bounds

    try:
        left, right = envelope(old), envelope(new)
    except (KeyError, TypeError, ValueError):
        return 'invalid_envelope'
    circular = old.get('circular', False)
    if circular != new.get('circular', False) or (left is None) != (right is None):
        return 'envelope_topology'
    if left is None:
        return None
    distances = np.abs(left-right)
    if circular:
        return (None if circular_envelope_distance(left, right) <= tolerance + 1e-12
                else 'envelope_endpoints')
    return None if np.all(distances <= tolerance + 1e-12) else 'envelope_endpoints'


def has_unresolved_envelope_uncertainty(projection, lower, span):
    """Return True if unresolved checks could still change the support envelope.

    Ignore checks already resolved, disabled, or superseded by endpoint refinement. Uncertain bins
    inside the envelope do not affect its extent; bins outside it may move its boundaries or change
    the circular covering arc. Missing or invalid bounds also count as uncertainty. This guards
    against premature convergence.
    """
    for gap in projection.get('gap_validation', []):
        for check in gap['bins']:
            if (check.get('superseded_by_endpoint_refinement')
                    or check['status'] in (
                        'recovered', 'persistent_at_probes', 'boundary_resolved', 'disabled'
                    )):
                continue
            envelope = projection.get('normalized_envelope')
            bounds = check.get('bounds')
            if envelope is None or bounds is None:
                return True
            if projection.get('circular', False) and abs(envelope[1]-envelope[0]-1) <= 1e-12:
                continue
            lo, hi = (np.asarray(bounds)-lower)/span
            if not np.all(np.isfinite([lo, hi])) or lo >= hi:
                return True
            shifts = (-1, 0, 1) if projection.get('circular', False) else (0,)
            if not any(lo+shift >= envelope[0]-1e-12 and hi+shift <= envelope[1]+1e-12
                       for shift in shifts):
                return True
    return False


def convergence_checks(args):
    """Return a dictionary showing which convergence checks are enabled by the run settings."""
    adaptive = not args.fixed_resolution
    endpoints = adaptive and args.convergence_support != 'bulk'
    return {
        'bulk': adaptive,
        'endpoints': endpoints,
        'interval_topology': endpoints and args.convergence_support == 'intervals',
        'constraint_class': adaptive and args.convergence_check_class,
        'previously_positive_guard': adaptive,
    }


def circular_envelope_uncertainty(projection, lower, span):
    """Estimate how much uncertain support boundaries could change the circular envelope.

    The covering arc excludes the largest gap. If another gap could become largest, internal gaps
    need refinement too; otherwise only the envelope's endpoints matter. This prevents declaring
    convergence while the arc is uncertain. Return (radius, internal_refinement_needed): a
    conservative distance bound as a fraction of the full circle, and whether internal gaps need
    refinement.
    """
    # Normalize uncertain spans and retain only definite support
    endpoint_brackets = projection.get('endpoint_uncertainty', [])
    if not endpoint_brackets:
        return 0., False
    uncertain_intervals = decide.normalize_intervals_to_domain(
        [[(a - lower) / span, (b - lower) / span] for a, b in endpoint_brackets],
        [0., 1.], True)
    support = projection.get('normalized_intervals', [])
    definite_support = decide.subtract_intervals(support, uncertain_intervals)
    if not definite_support:
        return .5, True

    # Find the largest gaps that the remaining uncertainty could allow
    possible_gaps = [[a[1], b[0]] for a, b in zip(definite_support, definite_support[1:])
                     if b[0] > a[1]]
    if definite_support[0][0] + 1 > definite_support[-1][1]:
        possible_gaps.append([definite_support[-1][1], definite_support[0][0] + 1])
    largest_possible_gap = max((b - a for a, b in possible_gaps), default=0.)
    envelope = projection.get('normalized_envelope')
    if envelope is None:
        return .5, True
    if envelope[1] - envelope[0] >= 1 - 1e-12:
        return largest_possible_gap / 2, True

    # Compare the guaranteed exterior gap with competing internal gaps
    exterior_start = envelope[1] % 1
    exterior_end = exterior_start + 1 - (envelope[1] - envelope[0])
    exterior_midpoint = (exterior_start + exterior_end) / 2
    shifted_uncertain_intervals = [
        [a + k, b + k] for a, b in uncertain_intervals for k in (-1, 0, 1, 2)
    ]
    guaranteed_exterior_parts = decide.subtract_intervals(
        [[exterior_start, exterior_end]], shifted_uncertain_intervals
    )
    competing_gap_lengths = [
        b - a for a, b in possible_gaps
        if not any(a - 1e-12 <= exterior_midpoint + k <= b + 1e-12 for k in (-1, 0, 1))
    ]

    # Measure how far each envelope endpoint could move within its bracket
    endpoint_radius = 0.
    for x in envelope:
        for a, b in endpoint_brackets:
            a, b = (a - lower) / span, (b - lower) / span
            for k in (-1, 0, 1):
                if a - 1e-10 <= x + k <= b + 1e-10:
                    endpoint_radius = max(endpoint_radius, x + k - a, b - x - k)

    # Refine internal gaps too unless the exterior gap is certainly largest
    if (len(guaranteed_exterior_parts) == 1 and
            guaranteed_exterior_parts[0][1] - guaranteed_exterior_parts[0][0]
            > max(competing_gap_lengths, default=0.) + 1e-12):
        return endpoint_radius, False
    return max(endpoint_radius, largest_possible_gap / 2), True


def endpoint_uncertainty_failure(old, new, lower, span, tolerance, mode):
    """Return True if boundary movement plus uncertainty exceeds the tolerance.

    Compare old and new endpoints using their movement and both uncertainty radii all normalized to
    the axis span. Circular envelopes use arc distance and envelope uncertainty instead. Mismatched
    endpoints also fail. Return False when matching comparison fits within the tolerance.
    """
    # Compare circular envelopes directly, without building endpoint lists
    if mode == 'envelope' and old['circular'] and new['circular']:
        arcs = [projection.get('normalized_envelope') for projection in (old, new)]
        if any(arc is None for arc in arcs):
            return (arcs[0] is None) != (arcs[1] is None)
        old_radius, _ = circular_envelope_uncertainty(old, lower, span)
        new_radius, _ = circular_envelope_uncertainty(new, lower, span)
        return (circular_envelope_distance(*arcs)
                + old_radius + new_radius > tolerance + 1e-12)

    # Collect endpoint positions, sides, and uncertainty radii
    def endpoints(projection):
        """Return each normalized endpoint, its side, and its uncertainty radius."""
        if mode == 'envelope':
            envelope = projection.get('normalized_envelope')
            if (envelope is None
                    or (projection['circular'] and abs(envelope[1] - envelope[0] - 1) <= 1e-12)):
                return []
            positions = [(envelope[0], 0), (envelope[1], 1)]
        else:
            intervals = projection['normalized_intervals']
            positions = [(position, side) for interval in intervals
                         for side, position in enumerate(interval)]
            if (projection['circular'] and intervals
                    and intervals[0][0] == 0 and intervals[-1][1] == 1):
                positions = [item for item in positions if item[0] not in (0, 1)]

        result = []
        for position, side in positions:
            radius = 0.
            for bracket_start, bracket_end in projection.get('endpoint_uncertainty', []):
                bracket_lower = (bracket_start - lower) / span
                bracket_upper = (bracket_end - lower) / span
                for shift in ((-1, 0, 1) if projection['circular'] else (0,)):
                    if bracket_lower - 1e-10 <= position + shift <= bracket_upper + 1e-10:
                        radius = max(radius, abs(position + shift - bracket_lower),
                                     abs(bracket_upper - position - shift))
            result.append((position, side, radius))
        return result

    # Handle changed endpoint counts and empty support
    old_endpoints, new_endpoints = endpoints(old), endpoints(new)
    if len(old_endpoints) != len(new_endpoints):
        return True
    if not old_endpoints:
        return False

    # Match endpoints in order, allowing cyclic shifts for circular axes
    for offset in range(len(new_endpoints) if old['circular'] else 1):
        matched_endpoints = new_endpoints[offset:] + new_endpoints[:offset]
        within_tolerance = True
        for (old_position, old_side, old_radius), (new_position, new_side, new_radius) in zip(
            old_endpoints, matched_endpoints
        ):
            distance = (abs((old_position - new_position + .5) % 1 - .5)
                        if old['circular'] else abs(old_position - new_position))
            if old_side != new_side or distance + old_radius + new_radius > tolerance + 1e-12:
                within_tolerance = False
                break
        if within_tolerance:
            return False
    return True


def support_convergence_failures(previous, current, args, mode):
    """Return reasons why support geometry or its uncertainty has not converged."""
    failures = []

    # Check unresolved gaps when comparing individual support intervals
    if mode == 'intervals' and any(
            level.get('endpoint_refinement', {}).get(
                'unresolved_gap_bins', level.get('gap_validation', {}).get('unresolved_bins', 0))
            for level in (previous, current)):
        failures.append('gap_validation.unresolved')

    # Set each axis's lower bound, span, and endpoint tolerance
    domains = {
        'e_out': (args.eout_min, args.eout_max - args.eout_min,
                  args.convergence_endpoint_eout_atol),
        'a_out': (math.log10(args.aout_min_au), math.log10(args.aout_max_au / args.aout_min_au),
                  args.convergence_endpoint_loga_atol),
        'phase': (args.nu_min_deg, args.nu_max_deg - args.nu_min_deg,
                  args.convergence_endpoint_phase_atol_deg),
    }
    compare_geometry = (
        envelope_endpoint_failure if mode == 'envelope' else interval_endpoint_failure
    )
    for family_name in ('all_targets', 'secular_targets'):
        for axis, (lower, span, tolerance) in domains.items():
            projections = [level['families'][family_name]['projections'][axis]
                           for level in (previous, current)]

            # Compare possible support, geometry, and then endpoint uncertainty
            if any(decide.unbracketed_possible(
                       projection, [lower, lower + span], mode == 'envelope')
                   for projection in projections):
                failures.append(f'{family_name}.{axis}.possible_support_unresolved')
            geometry_failure = compare_geometry(*projections, tolerance / span)
            if geometry_failure:
                failures.append(f'{family_name}.{axis}.{geometry_failure}')
            elif endpoint_uncertainty_failure(*projections, lower, span, tolerance / span, mode):
                failures.append(f'{family_name}.{axis}.endpoint_uncertainty')

            # Check unresolved local endpoints in interval mode
            local_checks = [check for projection in projections
                            for check in projection.get('endpoint_refinement', [])]
            if mode == 'intervals' and any(
                    check['status'] not in (
                        'recovered', 'persistent_at_probes', 'boundary_resolved'
                    )
                    for check in local_checks):
                failures.append(f'{family_name}.{axis}.local_endpoints_unresolved')

            # Check remaining envelope uncertainty only after geometry agrees
            if mode == 'envelope' and not geometry_failure and any(
                    has_unresolved_envelope_uncertainty(
                        dict(projection, gap_validation=[{
                            'bins': projection.get('endpoint_refinement', [])
                        }]),
                        lower, span)
                    for projection in projections):
                failures.append(f'{family_name}.{axis}.local_envelope_unresolved')
            if mode == 'envelope' and not geometry_failure and any(
                    has_unresolved_envelope_uncertainty(projection, lower, span)
                    for projection in projections):
                failures.append(f'{family_name}.{axis}.envelope_boundary_unresolved')
    return failures


def bulk_convergence_failures(previous, current, args):
    """Return failure reasons for changes in bulk statistics, verdicts, and optional class."""
    failures = []
    old_all_stats, new_all_stats = (
        level['families']['all_targets'] for level in (previous, current)
    )
    old_secular_stats, new_secular_stats = (
        level['families']['secular_targets'] for level in (previous, current)
    )

    # Check compatibility, reliability, and optional classification changes
    if (old_all_stats['compatible_cells'] > 0) != (new_all_stats['compatible_cells'] > 0):
        failures.append('recoil_compatible')
    if previous['reliability_verdict'] != current['reliability_verdict']:
        failures.append('reliability_verdict')
    if (args.convergence_check_class
            and old_secular_stats['constraint_class'] != new_secular_stats['constraint_class']):
        failures.append('secular_targets.constraint_class')

    # Empty compatible regions require zero volumes; skip the remaining comparisons
    if new_all_stats['compatible_cells'] == 0:
        if not all(stats['volume_fraction'] == 0
                   for stats in (
                       old_all_stats, new_all_stats, old_secular_stats, new_secular_stats
                   )):
            failures.append('empty_region_volume_agreement')
        return failures

    # Compare the secular fraction and support coverages with absolute tolerances
    def check_absolute_difference(failure_name, old_value, new_value, tolerance):
        if (old_value is None or new_value is None or not math.isfinite(old_value)
                or not math.isfinite(new_value) or abs(old_value - new_value) > tolerance):
            failures.append(failure_name)

    check_absolute_difference(
        'r_sec', previous['r_sec'], current['r_sec'], args.convergence_r_sec_atol
    )
    for axis in ('e_out', 'a_out', 'phase'):
        check_absolute_difference(
            f'secular_targets.{axis}.support_coverage',
            old_secular_stats['projections'][axis]['support_coverage'],
            new_secular_stats['projections'][axis]['support_coverage'],
            args.convergence_coverage_atol)

    # Compare volume fractions using their logarithmic ratio; matching zeros pass
    for family_name, old_stats, new_stats in (
            ('all_targets', old_all_stats, new_all_stats),
            ('secular_targets', old_secular_stats, new_secular_stats)):
        old_value, new_value = old_stats['volume_fraction'], new_stats['volume_fraction']
        if old_value == new_value == 0:
            continue
        if (old_value is None or new_value is None
                or not math.isfinite(old_value) or not math.isfinite(new_value)
                or old_value <= 0 or new_value <= 0
                or abs(math.log10(new_value / old_value)) > args.convergence_log_volume_atol):
            failures.append(f'{family_name}.volume_fraction')
    return failures


def convergence_failures(previous, current, args):
    """Bulk agreement and the selected support geometry, including uncertainty."""
    if args.convergence_support == 'bulk':
        return bulk_convergence_failures(previous, current, args)
    return (support_convergence_failures(previous, current, args, args.convergence_support)
            + bulk_convergence_failures(previous, current, args))


def describe_convergence_issue(name):
    """Turn convergence failure or pending verification identifier into a readable explanation."""
    descriptions = {
        'uniform_confirmation': 'Confirm agreement on a finer grid',
        'Initial evaluation': 'A finer grid is needed for the first comparison',
        'recoil_compatible': 'Whether compatible solutions exist changed between grids',
        'reliability_verdict': 'The Peters reliability classification changed',
        'empty_region_volume_agreement': 'The grids disagree on whether compatible volume is zero',
        'previously_compatible_region_missing': 'Previously sampled compatible solutions missing',
        'gap_validation.unresolved': 'Some support gaps remain unresolved',
        'r_sec': 'The secular fraction has not met its tolerance',
        'constraint_class': 'The constraint classification changed',
        'volume_fraction': 'The compatible volume fraction has not met its tolerance',
        'support_coverage': 'The support coverage has not met its tolerance',
        'invalid_intervals': 'Support intervals could not be compared',
        'interval_topology': 'The number or arrangement of support intervals changed',
        'interval_endpoints': 'Support interval boundaries have not met their tolerance',
        'invalid_envelope': 'The outer support envelope could not be compared',
        'envelope_topology': 'The outer support envelope changed structure',
        'envelope_endpoints': 'Outer support boundaries have not met their tolerance',
        'endpoint_uncertainty': 'Support boundary uncertainty exceeds the tolerance',
        'possible_support_unresolved': 'Possible additional support remains unresolved',
        'local_endpoints_unresolved': 'Local support boundary checks remain unresolved',
        'local_envelope_unresolved': 'Local outer-boundary checks remain unresolved',
        'envelope_boundary_unresolved': 'The outer support boundary remains unresolved',
    }
    if name in descriptions:
        return descriptions[name]
    parts = name.split('.')
    labels = {'all_targets': 'All targets', 'secular_targets': 'Secular targets',
              'e_out': 'outer eccentricity', 'a_out': 'outer separation', 'phase': 'orbital phase'}
    context = ' / '.join(labels.get(part, part.replace('_', ' ')) for part in parts[:-1])
    detail = descriptions.get(parts[-1], parts[-1].replace('_', ' ').capitalize())
    return f'{context}: {detail}' if context else detail


# -------------------------------------------------------------------------------------------------
# Adaptive analysis and report assembly
# -------------------------------------------------------------------------------------------------

def run_analysis(args, evaluate=evaluate_resolution, progress=None):
    """Run a fixed grid or refine successive grids until convergence or a limit.

    Evaluate each resolution, refine support boundaries when needed, and record convergence checks.
    Return the settings, final result, history, and stop reason.
    """
    # Set up the grid schedule, history, and initial status
    resolution = Resolution(args.n_eout, args.n_aout, args.n_nu, args.n_epost)
    sequence = (tuple(Resolution(*counts) for counts in DEFAULT_RESOLUTION_SEQUENCE)
                if getattr(args, 'resolution_sequence', 'geometric') == 'sequence' else None)
    schedule = RefinementSchedule(
        resolution, args.refinement_factor, args.convergence_streak, sequence
    )
    history = []
    status = 'fixed_resolution' if args.fixed_resolution else 'not_converged'
    stop_reason = 'Fixed resolution requested; convergence was not tested.'

    # Evaluate each grid level and decide whether another is needed
    for level in range(1 if args.fixed_resolution else args.max_refinements + 1):
        # Evaluate this resolution and report its size
        comparison_index = schedule.anchor_index
        if progress:
            progress(f'Level {level}: e_out={resolution.n_eout}, a_out={resolution.n_aout}, '
                     f'phase={resolution.n_nu}, e_post={resolution.n_epost} '
                     f'({resolution.points:,} progenitor points)')
        try:
            result = evaluate(args, resolution, progress)
        except ValueError as exc:
            if not history:
                raise
            stop_reason = f'Refinement could not be evaluated: {exc}'
            break

        # Revisit gaps and refine boundaries when bulk results agree
        if comparison_index is not None and args.convergence_support != 'bulk':
            from gap_validation import revisit_unresolved_gaps
            if 'endpoint_refinement' not in history[comparison_index]:
                revisit_unresolved_gaps(history[comparison_index], args, progress)
            baseline = history[comparison_index]
            if args.refine_endpoints and not bulk_convergence_failures(baseline, result, args):
                from endpoint_refinement import refine_comparison
                refine_comparison(baseline, result, args, progress)

        # Compare results and guard against losing previously found solutions
        failures = (convergence_failures(history[comparison_index], result, args)
                    if comparison_index is not None else ['Initial evaluation'])
        if (history and result['families']['all_targets']['compatible_cells'] == 0
                and any(h['families']['all_targets']['compatible_cells'] > 0 for h in history)):
            failures.append('previously_compatible_region_missing')

        # Update the schedule and save this level's diagnostics
        accepted = schedule.record_evaluation(level, resolution, failures)
        result['comparison_level'] = comparison_index
        result['accepted_as_baseline'] = accepted
        result['remaining_verification'] = [] if args.fixed_resolution else schedule.pending()
        result['bulk_convergence_failures'] = (
            bulk_convergence_failures(history[comparison_index], result, args)
            if comparison_index is not None else ['Initial evaluation']
        )
        result['support_interval_diagnostics'] = (support_convergence_failures(
            history[comparison_index], result, args, 'intervals')
            if comparison_index is not None and args.convergence_support != 'bulk' else [])
        result['convergence_failures'] = failures
        result['stable_comparisons'] = schedule.streak
        history.append(result)

        # Print a compact summary of this level
        if progress:
            stats = result['families'][result['selected_family']]
            classification = ('Peters-incompatible'
                              if stats['constraint_class'] == 'peters-incompatible'
                              else stats['constraint_class'])
            level_rows = [
                ('Class', classification),
                ('Volume fraction (V)', stats['volume_fraction']),
                ('R_sec', result['r_sec']),
                ('Convergence checks outstanding', len(failures)),
                ('Time needed for this level', f'{result["elapsed_seconds"]:.1f} s'),
            ]
            if failures and comparison_index is not None:
                level_rows.append((
                    'Failed checks',
                    '; '.join(describe_convergence_issue(item) for item in failures)
                ))
            width = max(len(label) for label, _ in level_rows) + 1
            progress(f'  Results for level {level}:')
            for label, value in level_rows:
                value = 'undefined' if value is None else value
                progress(f'    {label + ":":<{width}}  {value}')

        # Stop for a fixed grid, confirmed convergence, or exhausted limits
        if args.fixed_resolution:
            break
        if schedule.done:
            status = 'converged'
            if args.convergence_support == 'bulk':
                stop_reason = (f'Bulk quantities agree for {schedule.streak} successive refinement '
                               'comparison(s); boundaries and gaps were not convergence-tested.')
            else:
                stop_reason = (
                    'Bulk diagnostics and selected support geometry agree for '
                    f'{schedule.streak} successive refinement comparison(s).'
                )
                stop_reason += (' Envelope convergence; internal gaps remain diagnostic.'
                                if args.convergence_support == 'envelope'
                                else ' Strict support-interval convergence.')
            break
        if sequence and level == len(sequence) - 1:
            stop_reason = 'Resolution sequence exhausted without convergence.'
            break
        if level == args.max_refinements:
            stop_reason = 'Maximum number of refinements reached.'
            break

        # Check the next grid against resource limits before continuing
        next_resolution = schedule.next_resolution()
        if (next_resolution.points > args.max_grid_points
                or next_resolution.n_epost > args.max_epost):
            stop_reason = 'Next refinement would exceed --max-grid-points or --max-epost.'
            break
        resolution = next_resolution

    # Complete the stop reason and prepare the system metadata.
    system = system_from_args(args)
    if status == 'not_converged' and not history[-1]['convergence_failures']:
        stop_reason += (
            ' Remaining verification: '
            f'{"; ".join(describe_convergence_issue(item) for item in schedule.pending())}; '
            f'{schedule.streak}/{args.convergence_streak} confirmations passed.'
        )

    # Return the run settings, final result, and full evaluation history
    return {
        'schema_version': 2,
        'system': asdict(system),
        'settings': asdict(settings_from_args(args)),
        'configuration': {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        'domain': {'a_out_au': [args.aout_min_au, args.aout_max_au],
                   'a_out_rg': [args.aout_min_au / system.rg_post_au,
                                args.aout_max_au / system.rg_post_au],
                   'e_out': [args.eout_min, args.eout_max],
                   'phase_deg': [args.nu_min_deg, args.nu_max_deg]},
        'resolution_schedule': {'mode': 'sequence' if sequence else 'geometric',
                                'sequence': ([asdict(item) for item in sequence]
                                             if sequence else None),
                                'factor': None if sequence else args.refinement_factor},
        'convergence_checks': convergence_checks(args),
        'convergence_comparisons': sum(h['comparison_level'] is not None for h in history),
        'numerical_status': status, 'stop_reason': stop_reason,
        'final': history[-1], 'history': history,
    }


# -------------------------------------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    progress = None if args.quiet else lambda message: print(message, file=sys.stderr, flush=True)
    from system_summary_posterior import run_posterior_analysis
    if is_posterior_mode(args):
        try:
            report = run_posterior_analysis(args, progress=progress)
            print('\n\n' + format_posterior_report(report))
            from posterior_plots import plot_report
            plot_report(report, args.json_output.with_suffix('.png'), include_r_sec=True)
        except (ValueError, OSError, ImportError) as exc:
            print(f'Posterior analysis error: {exc}', file=sys.stderr)
            return 1
        return 1 if report['numerical_status'] == 'not_converged' else 0
    report = run_analysis(args, progress=progress)
    print('\n\n' + format_report(report))
    if args.json_output:
        write_report(report, args.json_output)
    return 1 if report['numerical_status'] == 'not_converged' else 0


if __name__ == '__main__':
    sys.exit(main())
