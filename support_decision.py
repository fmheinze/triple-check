"""Shared classification and interval handling for filtered projected support.

Classify grid bins and local probes as supported, excluded, or undecided using compatible
fractions, the configured support threshold, numerical uncertainty, and minimum-count checks.
Uncertainty that straddles the threshold remains undecided.

Construct brackets around uncertain endpoints, handle circular phase wrapping, merge overlapping
intervals while preserving gaps, and identify possible support outside the reported intervals or
their uncertainty brackets.

The initial grid, gap validation, and endpoint refinement share these rules. This module interprets
their counts and occupancy estimates; it does not compute recoil velocities, run edge tracking, or
perform the local refinement itself.
"""
import math

import numpy as np

SUPPORTED, UNDECIDED, EXCLUDED = 'supported', 'undecided', 'excluded'
STATES = (SUPPORTED, UNDECIDED, EXCLUDED)


def occupancy_sigma(compatible, physical):
    """Estimate grid-sampling uncertainty in the compatible fraction for each bin.

    Given compatible-cell counts m and physically admissible counts N, return sqrt(m + 1) / N where
    both counts are positive, and zero otherwise. Inputs may be scalars or broadcastable arrays;
    the result has their broadcast shape.

    This heuristic allows for narrow compatible regions being hit or missed by grid-cell centres.
    The +1 increases the estimate for small counts. Filtered support checks use it to flag
    fractions too close to the support threshold.
    """
    compatible_counts = np.asarray(compatible, dtype=float)
    physical_counts = np.asarray(physical, dtype=float)
    broadcast_shape = np.broadcast_shapes(compatible_counts.shape, physical_counts.shape)
    count_uncertainty = np.where(
        compatible_counts > 0., np.sqrt(compatible_counts + 1.), 0.
    )
    return np.divide(
        np.broadcast_to(count_uncertainty, broadcast_shape),
        np.broadcast_to(physical_counts, broadcast_shape),
        out=np.zeros(broadcast_shape),
        where=np.broadcast_to(physical_counts, broadcast_shape) > 0.,
    )


def classify(compatible, physical, min_fraction, sigma_factor, min_cells=1):
    """Return supported, undecided, and excluded masks for projected grid bins.

    For a positive fraction threshold, support requires the compatible fraction minus its
    uncertainty margin to exceed the threshold and the minimum count to be met. Exclude bins with
    zero compatible hits or whose fraction plus the margin is below the threshold; other physically
    admissible bins are undecided. The margin is sigma_factor times the heuristic occupancy_sigma
    estimate.
    """
    compatible_counts = np.asarray(compatible, dtype=float)
    physical_counts = np.asarray(physical, dtype=float)
    has_physical_cells = physical_counts > 0.
    if not min_fraction > 0.:
        supported = has_physical_cells & (compatible_counts >= max(min_cells, 1))
        return supported, np.zeros(supported.shape, dtype=bool), has_physical_cells & ~supported
    compatible_fraction = np.divide(
        compatible_counts, physical_counts,
        out=np.zeros(np.broadcast_shapes(compatible_counts.shape, physical_counts.shape)),
        where=has_physical_cells,
    )
    uncertainty_margin = sigma_factor * occupancy_sigma(compatible_counts, physical_counts)
    supported = (
        has_physical_cells
        & (compatible_counts >= max(min_cells, 1))
        & (compatible_fraction - uncertainty_margin > min_fraction)
    )
    excluded = has_physical_cells & (
        (compatible_counts <= 0.) | (compatible_fraction + uncertainty_margin < min_fraction)
    )
    return supported, has_physical_cells & ~supported & ~excluded, excluded


def probe_sigma(sample):
    """Return one probe's fraction uncertainty for the support classification.

    Use the stored fraction_sigma when available; otherwise estimate uncertainty from compatible
    and physically admissible cell counts.
    """
    if sample.get('fraction_sigma') is not None:
        return float(sample['fraction_sigma'])
    return float(
        occupancy_sigma(sample.get('compatible_cells', 0), sample.get('physical_cells', 0))
    )


def probe_state(sample, options):
    """Classify a local probe as supported, excluded, or undecided.

    For a positive fraction threshold, compare the compatible fraction with the threshold using
    a margin of scaled probe uncertainty plus boundary-tracking error. Support also requires the
    minimum compatible count. With a nonpositive threshold, use only that count requirement.

    A confirmed empty physical domain is excluded; no physical cells without that confirmation
    leaves the probe undecided. Probe estimators supply a fraction whenever physical cells exist;
    the existing fallback excludes records that have physical cells but lack a fraction.
    """
    # Handle empty physical domains and unavailable fractions
    physical_count = sample.get('physical_cells', 0)
    if (
        sample.get('physical_domain_empty') is True
        and physical_count == 0
        and sample.get('compatible_cells', 0) == 0
    ):
        return EXCLUDED
    if not physical_count or sample.get('fraction') is None:
        return EXCLUDED if physical_count else UNDECIDED

    # Use the count-only criterion when the fraction filter is disabled
    compatible_count = sample.get('compatible_cells', 0)
    min_fraction = options.get('min_fraction', 0.)
    min_cells = max(options.get('min_cells', 1), 1)
    if not min_fraction > 0.:
        return SUPPORTED if compatible_count >= min_cells else EXCLUDED

    # Compare the fraction with the threshold, allowing for both uncertainty sources
    boundary_error = float(sample.get('fraction_error_bound', 0.))
    compatible_fraction = float(sample['fraction'])
    uncertainty_margin = options.get('sigma_factor', 2.) * probe_sigma(sample) + boundary_error
    if compatible_count >= min_cells and compatible_fraction - uncertainty_margin > min_fraction:
        return SUPPORTED
    if (
        (compatible_count <= 0 and boundary_error == 0.)
        or compatible_fraction + uncertainty_margin < min_fraction
    ):
        return EXCLUDED
    return UNDECIDED


def decided(state):
    return state in (SUPPORTED, EXCLUDED)


def true_runs(mask):
    """Return (start, stop) indices for each True stretch in a 1D mask, with stop excluded."""
    mask = np.asarray(mask, dtype=bool)
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(transitions == 1).tolist(),
                    np.flatnonzero(transitions == -1).tolist()))


def endpoint_brackets(supported, undecided, edges, circular=False):
    """Build sampling-based uncertainty brackets around supported stretches.

    Given supported and undecided bin masks and their edges, scan outward from each supported
    stretch through adjacent undecided bins. Bracket each transition between the supported bin's
    centre and the next decided centre, or the domain edge if the scan reaches it. A supported
    stretch already touching a linear domain edge has no outward transition to bracket.

    Circular axes allow unwrapped coordinates across the seam; continuous support across that
    seam is not a boundary. Return records containing the nominal endpoint, sorted bounds, side
    (0 for lower, 1 for upper), and source.
    """
    # Prepare bin centres and the domain span used for circular wrapping
    supported, undecided = np.asarray(supported, bool), np.asarray(undecided, bool)
    edges = np.asarray(edges, float)
    centres = (edges[:-1] + edges[1:]) / 2
    bin_count, domain_span = len(supported), edges[-1] - edges[0]
    brackets = []

    # Inspect both ends of each supported stretch
    for start, stop in true_runs(supported):
        for side, inside_index, step, endpoint in (
            (0, start, -1, edges[start]),
            (1, stop - 1, 1, edges[stop]),
        ):
            outside_index = inside_index + step
            if circular and supported[outside_index % bin_count]:
                continue
            if not circular and not 0 <= outside_index < bin_count:
                continue

            # Pass undecided bins, stopping at a decided bin or a domain limit
            while (
                (circular or 0 <= outside_index < bin_count)
                and undecided[outside_index % bin_count]
            ):
                outside_index += step
                if abs(outside_index - inside_index) >= bin_count:
                    break

            # Convert the stopping index to a coordinate, unwrapping circular axes
            if circular:
                outside_coordinate = (
                    centres[outside_index % bin_count] + (outside_index // bin_count) * domain_span
                )
            elif outside_index < 0 or outside_index >= bin_count:
                outside_coordinate = edges[0 if outside_index < 0 else -1]
            else:
                outside_coordinate = centres[outside_index]
            brackets.append(dict(
                endpoint=float(endpoint),
                bounds=sorted((float(centres[inside_index]), float(outside_coordinate))),
                side=side,
                source='grid_transition',
            ))
    return brackets


def interval_union(intervals):
    """Merge overlapping/touching linear intervals, preserving genuine gaps."""
    merged = []
    for interval_start, interval_end in sorted(
        (float(interval_start), float(interval_end)) for interval_start, interval_end in intervals
    ):
        if merged and interval_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval_end)
        else:
            merged.append([interval_start, interval_end])
    return merged


def subtract_intervals(intervals, covered):
    """Return the parts of intervals left after removing all covered spans.

    Merge overlapping intervals in each collection, then trim or split the remaining intervals
    around each covered span. Preserve uncovered fragments and discard fully covered intervals.
    """
    remaining = interval_union(intervals)
    for covered_start, covered_end in interval_union(covered):
        pieces = []
        for interval_start, interval_end in remaining:
            if interval_end <= covered_start or interval_start >= covered_end:
                pieces.append([interval_start, interval_end])
            else:
                if interval_start < covered_start:
                    pieces.append([interval_start, covered_start])
                if interval_end > covered_end:
                    pieces.append([covered_end, interval_end])
        remaining = pieces
    return remaining


def normalize_intervals_to_domain(intervals, domain, circular=False):
    """Clip or wrap intervals into the domain and merge overlapping or touching pieces.

    Linear intervals are clipped to the domain, discarding portions outside it. Circular intervals
    are shifted into the domain and split at the seam; an interval spanning at least one full
    circle covers the entire domain. Preserve already in-domain circular endpoints to avoid
    unnecessary rounding changes.

    Expect ordered interval endpoints and a domain with positive width. Circular intervals that
    cross the seam must use unwrapped endpoints, with the end at or beyond the start.
    """
    domain_start, domain_end = map(float, domain)
    domain_span = domain_end - domain_start
    result = []
    for interval_start, interval_end in intervals:
        # Wrap circular intervals and split any portion crossing the domain seam
        if circular:
            if interval_end - interval_start >= domain_span:
                return [[domain_start, domain_end]]
            # Preserve in-domain endpoints exactly; shifting can introduce rounding differences
            shift = (
                math.floor((interval_start - domain_start) / domain_span) * domain_span
                if not domain_start <= interval_start < domain_end else 0.
            )
            wrapped_start, wrapped_end = interval_start - shift, interval_end - shift
            result.append([wrapped_start, min(wrapped_end, domain_end)])
            if wrapped_end > domain_end:
                result.append([domain_start, wrapped_end - domain_span])

        # Keep only the positive-width intersection for linear intervals
        elif min(interval_end, domain_end) > max(interval_start, domain_start):
            result.append([max(interval_start, domain_start), min(interval_end, domain_end)])
    return interval_union(result)


def unbracketed_possible(projection, domain, envelope_only=True):
    """Return possible-support fragments outside the reported geometry and endpoint brackets.

    Subtract reported support and endpoint uncertainty brackets from possible support, falling
    back to undecided intervals when possible-support intervals are absent. In envelope-only mode,
    also cover the entire reported envelope, since internal gaps do not affect its extent.

    Normalize both interval collections to the domain, including circular wrapping. Ignore only
    fragments whose two endpoints lie within rounding tolerance of the same covered boundary;
    retain tiny isolated fragments elsewhere so they remain available for refinement.
    """
    domain_start, domain_end = map(float, domain)
    domain_span = domain_end - domain_start

    # Assemble the reported support and, when requested, its covering envelope
    covered_intervals = list(projection.get('support_intervals', []))
    if envelope_only:
        envelope = projection.get('normalized_envelope')
        if envelope is not None:
            if projection.get('circular'):
                for shift in (-1, 0, 1):
                    if envelope[1] + shift > 0 and envelope[0] + shift < 1:
                        covered_intervals.append([
                            max(domain_start, domain_start + (envelope[0] + shift) * domain_span),
                            min(domain_end, domain_start + (envelope[1] + shift) * domain_span),
                        ])
            else:
                covered_intervals.append([
                    domain_start + envelope[0] * domain_span,
                    domain_start + envelope[1] * domain_span,
                ])

    # Include endpoint brackets, then subtract covered spans from possible support
    covered_intervals += projection.get('endpoint_uncertainty', [])
    possible_intervals = projection.get(
        'possible_support_intervals', projection.get('undecided_intervals', [])
    )
    circular = projection.get('circular', False)

    # Normalize each bracket once to avoid rounding from repeated circular shifts
    covered_intervals = normalize_intervals_to_domain(covered_intervals, domain, circular)
    possible_intervals = normalize_intervals_to_domain(possible_intervals, domain, circular)
    fragments = subtract_intervals(possible_intervals, covered_intervals)

    # Discard rounding fragments beside one covered boundary, preserving isolated islands
    roundoff_tolerance = 8 * max(
        math.ulp(domain_start), math.ulp(domain_end), math.ulp(domain_span)
    )
    covered_boundaries = [endpoint for interval in covered_intervals for endpoint in interval]
    unresolved_fragments = []
    for fragment_start, fragment_end in fragments:
        if fragment_end > fragment_start and not any(
            max(abs(fragment_start - boundary), abs(fragment_end - boundary)) <= roundoff_tolerance
            for boundary in covered_boundaries
        ):
            unresolved_fragments.append([fragment_start, fragment_end])
    return unresolved_fragments
