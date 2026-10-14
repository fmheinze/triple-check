#!/usr/bin/env python3
"""Shared physics core for the triple-check scripts, including:

* physical-unit conversions, orbital diagnostics, and Peters wrappers
* the Mardling-Aarseth hierarchical triple stability boundary
* the post-first-merger eccentricity grid and candidate orbits
* the progenitor phase grid functions
* the pre-kick state of the outer orbit
* the recoil kinematics and the minimum-recoil map over an axis-aligned slice
* the allowed progenitor region computation
* the system-level Peters-reliability assessment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as dataclass_fields, replace
from contextlib import contextmanager, ExitStack, nullcontext
from typing import Optional

import numpy as np

from constants import AU_SI, C_SI, DAY_SI, G_SI, MSUN_SI, MSUN_TIME_SI
import peters_tools as tools


# How the progenitor phase coordinate is measured
PHASE_MEAN_ANOMALY = "mean_anomaly"
PHASE_TRUE_ANOMALY = "true_anomaly"
PHASE_MEASURES = (PHASE_MEAN_ANOMALY, PHASE_TRUE_ANOMALY)

# Peters secular applicability labels
PETERS_SECULAR_OK = "peters_secular_ok"
ALREADY_AT_OR_ABOVE_REFERENCE = "already_at_or_above_reference"
FEW_PASSAGE_PETERS_UNRELIABLE = "few_passage_Peters_unreliable"

# System-level classifications
VERDICT_PETERS_RELIABLE = "peters_reliable"
VERDICT_RECOIL_COMPATIBLE_UNRELIABLE = "recoil_compatible_but_peters_unreliable"
VERDICT_NO_COMPATIBLE_PROGENITOR = "no_compatible_progenitor"


# Shared execution utility used by maps and posterior/grid evaluations
def completed_tasks(function, tasks, workers, progress=None, *, executor=None):
    """Yield results as jobs finish, keeping at most two jobs per worker queued."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    import multiprocessing

    if workers < 1:
        raise ValueError('workers must be at least 1.')
    workers = min(workers, len(tasks))
    if workers <= 1:
        for task in tasks:
            yield function(task)
        return
    context = (nullcontext(executor) if executor is not None else ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn')
    ))
    with context as pool:
        jobs = iter(tasks)
        pending = {pool.submit(function, task) for _, task in zip(range(2 * workers), jobs)}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    yield result
                    task = next(jobs, None)
                    if task is not None:
                        pending.add(pool.submit(function, task))
        finally:
            for future in pending:
                future.cancel()


@contextmanager
def reusable_task_runner(function, workers, progress=None):
    """Reuse a lazily started worker pool across batches, closing it when the operation finishes."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    executor = None
    with ExitStack() as resources:
        def run(tasks):
            """Run one batch, creating workers only when there is parallel work to do."""
            nonlocal executor
            if workers > 1 and len(tasks) > 1 and executor is None:
                executor = resources.enter_context(ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context('spawn')
                ))
            return completed_tasks(function, tasks, workers, progress, executor=executor)

        yield run


# -------------------------------------------------------------------------------------------------
# Unit helpers
# -------------------------------------------------------------------------------------------------

def mass_time_unit_seconds(total_mass_msun):
    """One code time unit, G M / c^3, in seconds."""
    return total_mass_msun * MSUN_TIME_SI


def mass_length_unit_m(total_mass_msun):
    """One code length unit, G M / c^2, in meters."""
    return G_SI * (total_mass_msun * MSUN_SI) / C_SI**2


def mass_length_unit_au(total_mass_msun):
    """One code length unit, G M / c^2, in astronomical units."""
    return mass_length_unit_m(total_mass_msun) / AU_SI


def eta_from_component_masses(m1, m2):
    """Symmetric mass ratio from any consistent pair of component masses."""
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError("component masses must be positive.")
    return m1 * m2 / (m1 + m2)**2


# -------------------------------------------------------------------------------------------------
# Orbital diagnostics
# -------------------------------------------------------------------------------------------------

def orbital_period_seconds(a_si, total_mass_msun):
    """Newtonian orbital period for a semi-major axis in SI units."""
    if a_si <= 0.0:
        raise ValueError("a_si must be positive.")
    if total_mass_msun <= 0.0:
        raise ValueError("total_mass_msun must be positive.")
    mu_si = G_SI * total_mass_msun * MSUN_SI
    return 2.0 * math.pi * math.sqrt(a_si**3 / mu_si)


def semi_major_axis_for_merger_time_au(time_days, e, m1_msun, m2_msun):
    """Semi-major axis whose Peters merger time is ``time_days``, in AU.

    Physical-units form of :func:`peters_tools.solve_semi_major_axis_for_merger_time`, which works
    in total-mass units and therefore needs only the symmetric mass ratio. Supply the component
    masses here and the conversion in and out is done for you.
    """
    if time_days <= 0.0:
        raise ValueError("time_days must be positive.")
    total_mass_msun = m1_msun + m2_msun
    eta = eta_from_component_masses(m1_msun, m2_msun)
    t_code = time_days * DAY_SI / mass_time_unit_seconds(total_mass_msun)
    a_code = tools.solve_semi_major_axis_for_merger_time(t_code, e, eta)
    return a_code * mass_length_unit_au(total_mass_msun)


def circular_inner_separation_from_time_au(m1_msun, m2_msun, time_days):
    """Circular separation giving a requested Peters merger time, in AU."""
    return semi_major_axis_for_merger_time_au(time_days, 0.0, m1_msun, m2_msun)


# -------------------------------------------------------------------------------------------------
# System properties, settings and specifications
# -------------------------------------------------------------------------------------------------

# Default threshold on R_sec above which a system is called Peters-reliable
DEFAULT_R_SEC_THRESHOLD = 0.85

# Default epoch at which the inner binary's separation is evaluated for the stability limit
DEFAULT_STABILITY_DAYS_BEFORE = 30.0

# Eccentricity at which an e_post grid changes its spacing
E_SPLIT = 0.9


@dataclass(frozen=True)
class TripleSystem:
    """The hierarchical triple, independent of any grid or plotting choice.

    ``m1_inner`` and ``m2_inner`` are the inner binary component masses before their merger,
    ``mrem`` the mass of the remnant they leave behind, and ``m3`` the mass of the tertiary.
    ``delay_days`` is the requested delay between the two mergers, and ``inclination_rad`` the
    mutual inclination of the inner and outer binaries. Further properties are computed from these.
    """
    m1_inner: float
    m2_inner: float
    mrem: float
    m3: float
    delay_days: float
    inclination_rad: float = math.pi

    @property
    def m_inner_pre(self) -> float:
        """Total inner-binary mass before the first merger [Msun]."""
        return self.m1_inner + self.m2_inner

    @property
    def m_post(self) -> float:
        """Total mass of the post-merger binary, remnant plus tertiary [Msun]."""
        return self.mrem + self.m3

    @property
    def mu_post(self) -> float:
        """Standard gravitational parameter of the post-merger binary [SI]."""
        return G_SI * self.m_post * MSUN_SI

    @property
    def eta_post(self) -> float:
        """Symmetric mass ratio of the post-merger binary."""
        return eta_from_component_masses(self.mrem, self.m3)

    @property
    def rg_post_au(self) -> float:
        """Length unit GM/c^2 of the post-merger binary, in AU."""
        return mass_length_unit_au(self.m_post)

    @property
    def inclination_deg(self) -> float:
        return math.degrees(self.inclination_rad)

    def describe(self) -> str:
        return (
            f"m1={self.m1_inner:g} m2={self.m2_inner:g} -> mrem={self.mrem:g}, "
            f"m3={self.m3:g} Msun; delay={self.delay_days:g} d; "
            f"inclination={self.inclination_deg:g} deg"
        )


@dataclass(frozen=True)
class AnalysisSettings:
    """The modeling choices and settings for one run.

    ``e_ref_min`` and ``e_ref_max`` bound the reference eccentricity interval of the admitted
    post-first-merger targets, and ``require_peters_applicable`` restricts them further to those
    meeting the Peters applicability criteria. ``recoil_ceiling_kms`` is the largest recoil
    considered possible. ``require_stability`` applies the Mardling-Aarseth hierarchical triple
    stability condition. The inner separation it needs is by default the one a circular inner
    binary has ``stability_days_before`` days before it merges. Alternatively,
    ``inner_separation_au`` can be directly supplied, or ``alpha_in`` sets it in units of
    G(m1_inner + m2_inner)/c^2 for each system.
    """
    e_ref_min: Optional[float] = None
    e_ref_max: Optional[float] = None
    require_peters_applicable: bool = False
    recoil_ceiling_kms: Optional[float] = None
    require_stability: bool = True
    stability_days_before: float = DEFAULT_STABILITY_DAYS_BEFORE
    inner_separation_au: Optional[float] = None
    alpha_in: Optional[float] = None
    phase_measure: str = PHASE_MEAN_ANOMALY
    r_sec_threshold: float = DEFAULT_R_SEC_THRESHOLD


# -------------------------------------------------------------------------------------------------
# Dynamical stability of the hierarchical triple
# -------------------------------------------------------------------------------------------------

def mardling_aarseth_aout_min_au(
    a_in_au,
    e_out,
    m_inner_msun: float,
    m3_msun: float,
    inclination_rad: float = 0.0,
):
    """Minimum outer semi-major axis for a dynamically stable triple, in AU.

    Empirical stability criterion of Mardling & Aarseth (2001), MNRAS 321, 398. They state it as
    a condition on the pericenter of the outer orbit, r_p,out = a_out (1 - e_out):

        r_p,out / a_in > 2.8 [ (1 + q_out) (1 + e_out) / (1 - e_out)^(1/2) ]^(2/5) (1 - 0.3 i/pi)

    with q_out = m3 / (m1 + m2) the mass ratio of the tertiary to the inner binary and i the
    mutual inclination in radians. Dividing by (1 - e_out) puts the criterion on the semi-major
    axis, which is the form evaluated below:

        a_out,min = 2.8 a_in (1 + q_out)^(2/5) (1 + e_out)^(2/5) (1 - e_out)^(-6/5) (1 - 0.3 i/pi)

    ``e_out`` may be a scalar or an array, the return type is the same as the input.
    """
    e_out = np.asarray(e_out, dtype=float)
    if np.any(e_out < 0.0) or np.any(e_out >= 1.0):
        raise ValueError("e_out must satisfy 0 <= e_out < 1.")

    q_out = m3_msun / m_inner_msun
    inclination_factor = 1.0 - 0.3 * inclination_rad / math.pi

    a_out_min = (
        a_in_au
        * 2.8
        * (1.0 + q_out)**(2.0 / 5.0)
        * (1.0 + e_out)**(2.0 / 5.0)
        / (1.0 - e_out)**(6.0 / 5.0)
        * inclination_factor
    )
    return float(a_out_min) if np.ndim(a_out_min) == 0 else a_out_min


def inner_separation_for_stability_au(
    m1_inner_msun: float,
    m2_inner_msun: float,
    settings: AnalysisSettings,
) -> float:
    """Inner-binary semi-major axis entering the stability criterion, in AU.

    Use the explicit AU value or alpha_in times the pre-merger inner mass length.
    Otherwise use the circular separation at ``settings.stability_days_before``.
    """
    if getattr(settings, "alpha_in", None) is not None:
        if settings.inner_separation_au is not None:
            raise ValueError("alpha-in and inner-separation-au are mutually exclusive.")
        if not (math.isfinite(settings.alpha_in) and settings.alpha_in > 0):
            raise ValueError("alpha-in must be finite and positive.")
        return settings.alpha_in * mass_length_unit_au(m1_inner_msun + m2_inner_msun)
    if settings.inner_separation_au is not None:
        return float(settings.inner_separation_au)
    return circular_inner_separation_from_time_au(
        m1_inner_msun,
        m2_inner_msun,
        time_days=settings.stability_days_before,
    )


# -------------------------------------------------------------------------------------------------
# Post-first-merger candidate orbits
# -------------------------------------------------------------------------------------------------

def make_epost_grid(
    n_epost: int,
    one_minus_epost_min: float = 1e-12,
) -> np.ndarray:
    """Construct a post-first-merger eccentricity grid.

    ``one_minus_epost_min`` is the smallest 1 - e_post the grid reaches, i.e., how close the
    most eccentric candidate comes to unity. The grid covers 0 <= e_post <= 1-one_minus_epost_min.

    Which spacing is used depends on whether the range reaches past ``E_SPLIT``:

    * A range reaching past it is split in two. Roughly one third of the grid is spaced linearly
      over 0 <= e_post < ``E_SPLIT``, and the remaining two thirds are uniform in
      log(1 - e_post) from ``E_SPLIT`` up to the maximum.
    * A range ending at or below ``E_SPLIT`` is covered by a single cubic grid
      e = max * (1 - (1 - x)^3) over a uniform x, which spaces points more widely at low e and more
      densely near the upper limit.

    Either way the grid holds exactly ``n_epost`` strictly increasing samples, or a ValueError
    says that the requested resolution is finer than double precision resolves near e_post = 1.
    """
    if n_epost < 3:
        raise ValueError("n_epost must be at least 3.")
    if not (0.0 < one_minus_epost_min < 1.0):
        raise ValueError("one-minus-epost-min must satisfy 0 < value < 1.")

    effective_max = 1.0 - one_minus_epost_min
    if effective_max >= 1.0:
        raise ValueError(
            "one-minus-epost-min is too small for double precision: "
            "1 - value rounds to 1. Increase one-minus-epost-min."
        )

    if effective_max <= E_SPLIT:
        # Use a single cubic grid.
        x = np.linspace(0.0, 1.0, n_epost)
        e = effective_max * (1.0 - (1.0 - x)**3)
        e[0] = 0.0
        e[-1] = effective_max
    else:
        # Split into a linear moderate-e (one third) and a logarithmic high-e part (two thirds).
        n_low = max(1, n_epost // 3)
        n_high = n_epost - n_low
        e_low = np.linspace(0.0, E_SPLIT, n_low, endpoint=False)
        delta = np.geomspace(1.0 - E_SPLIT, one_minus_epost_min, n_high)
        e_high = 1.0 - delta
        e_high[-1] = effective_max
        e = np.concatenate((e_low, e_high))

    # Equal neighbors mean the requested spacing has fallen below what a double resolves.
    collisions = int(np.count_nonzero(np.diff(e) <= 0.0))
    if collisions:
        raise ValueError(
            f"n_epost={n_epost} with one_minus_epost_min={one_minus_epost_min:g} asks for a finer "
            f"grid than double precision resolves: {collisions} neighboring sample pair(s) "
            f"coincide. Lower n_epost or raise one-minus-epost-min."
        )

    return e


@dataclass
class PostFirstMergerCandidates:
    """Candidate orbits and diagnostics.

    ``e_ref`` is evaluated at ``f_ref_hz``. For an initial peak frequency
    already at or above that reference, it is the initial eccentricity;
    ``applicability_label`` then flags ``already_at_or_above_reference``.
    """
    epost: np.ndarray
    a_si: np.ndarray
    rp_si: np.ndarray
    ra_si: np.ndarray
    e_ref: np.ndarray
    f_peak_initial_hz: np.ndarray
    orbital_period_days: np.ndarray
    n_orbits_to_merger: np.ndarray
    rp_over_rg: np.ndarray
    applicability_label: np.ndarray

    # Orbits are held in SI units, which is what the recoil kinematics use. AU is a presentation
    # unit, only needed for written output, so it is derived on demand.
    @property
    def a_au(self) -> np.ndarray:
        return self.a_si / AU_SI

    @property
    def rp_au(self) -> np.ndarray:
        return self.rp_si / AU_SI

    @property
    def ra_au(self) -> np.ndarray:
        return self.ra_si / AU_SI


def peters_applicability_label(
    n_orbits,
    rp_over_rg,
    f_peak_initial_hz,
    f_ref_hz,
    min_orbits,
    min_rp_rg,
):
    """Conservative validity label for orbit-averaged Peters evolution.

    ``n_orbits`` is the merger delay divided by the initial orbital period.
    The classification is binary once a target is below the reference frequency: it
    either satisfies both applicability conditions, or it does not.  Every
    threshold is supplied by the caller, so the label has no hidden scales of
    its own.
    """
    if f_peak_initial_hz >= f_ref_hz:
        return ALREADY_AT_OR_ABOVE_REFERENCE
    if n_orbits >= min_orbits and rp_over_rg >= min_rp_rg:
        return PETERS_SECULAR_OK
    return FEW_PASSAGE_PETERS_UNRELIABLE


def compute_post_first_merger_candidates(
    epost_grid: np.ndarray,
    mrem_msun: float,
    m3_msun: float,
    merger_time_days: float,
    f_ref_hz: float = 10.0,
    min_peters_orbits: float = 10.0,
    min_peters_rp_rg: float = 20.0,
) -> PostFirstMergerCandidates:
    """Tabulate the candidate post-first-merger orbits, one per grid eccentricity.

    Each entry is the orbit the remnant and the tertiary must be on for them to merge
    ``merger_time_days`` later under Peters evolution: for a given e_post the merger time fixes
    the semi-major axis, so the grid traces a one-parameter family from circular to nearly radial.

    These are candidates, not yet post-first-merger target orbits. A candidate becomes an
    admitted target once its reference eccentricity falls in a prescribed interval, and the
    column ``e_ref`` is what is needed for this test. Each candidate also carries its
    pericenter in units of GM/c^2, its initial peak GW frequency, the merger delay in units of the
    initial orbital period, and the applicability label set by ``min_peters_orbits`` and
    ``min_peters_rp_rg``.

    The Peters applicability cuts and the eccentricity condition are applied later from
    :class:`AnalysisSettings`, so which candidates a run admits is not decided here.
    """
    M_post_msun = mrem_msun + m3_msun
    eta = eta_from_component_masses(mrem_msun, m3_msun)

    t_target_code = merger_time_days * DAY_SI / mass_time_unit_seconds(M_post_msun)
    length_unit_m = mass_length_unit_m(M_post_msun)

    a_M = np.array([
        tools.solve_semi_major_axis_for_merger_time(t_target_code, float(e), eta)
        for e in epost_grid
    ])

    # In total-mass units a length is its value in GM/c^2, so rp_M is already r_p/r_g.
    rp_M = a_M * (1.0 - epost_grid)
    ra_M = a_M * (1.0 + epost_grid)
    rp_over_rg = rp_M

    # Convert to SI units.
    a_si = a_M * length_unit_m
    rp_si = rp_M * length_unit_m
    ra_si = ra_M * length_unit_m

    e_ref = np.empty_like(epost_grid, dtype=float)
    f_peak_initial_hz = np.empty_like(epost_grid, dtype=float)
    period_days = np.empty_like(epost_grid, dtype=float)

    for k, (a0_si, e0) in enumerate(zip(a_si, epost_grid)):
        a0_si, e0 = float(a0_si), float(e0)
        f_peak_initial_hz[k] = tools.eccentric_peak_gw_frequency_hz(a0_si, e0, M_post_msun)
        e_ref[k] = tools.peters_eccentricity_at_gw_frequency(
            a0_si, e0, M_post_msun, f_ref_hz
        )
        period_days[k] = orbital_period_seconds(a0_si, M_post_msun) / DAY_SI

    n_orbits = merger_time_days / period_days
    labels = np.array([
        peters_applicability_label(
            float(norb),
            float(rprg),
            float(fpk),
            f_ref_hz,
            min_peters_orbits,
            min_peters_rp_rg,
        )
        for norb, rprg, fpk in zip(n_orbits, rp_over_rg, f_peak_initial_hz)
    ], dtype=object)

    return PostFirstMergerCandidates(
        epost=epost_grid,
        a_si=a_si,
        rp_si=rp_si,
        ra_si=ra_si,
        e_ref=e_ref,
        f_peak_initial_hz=f_peak_initial_hz,
        orbital_period_days=period_days,
        n_orbits_to_merger=n_orbits,
        rp_over_rg=rp_over_rg,
        applicability_label=labels,
    )


def passes_e_ref_filter(
    e_ref,
    e_ref_min: Optional[float] = None,
    e_ref_max: Optional[float] = None,
):
    """Return True where ``e_ref`` lies inside the requested reference eccentricity range.

    Elementwise, so this takes one eccentricity or a whole column of them.
    """
    e_ref = np.asarray(e_ref, dtype=float)
    inside = np.ones(e_ref.shape, dtype=bool)
    if e_ref_min is not None:
        inside &= e_ref >= e_ref_min
    if e_ref_max is not None:
        inside &= e_ref <= e_ref_max
    return inside


def target_is_selected(e_ref, applicability_label, settings: AnalysisSettings):
    """Return True where a candidate survives the reference-eccentricity and applicability cuts.

    Elementwise, so this judges one candidate or a whole table at once.
    """
    keep = passes_e_ref_filter(e_ref, settings.e_ref_min, settings.e_ref_max)
    if settings.require_peters_applicable:
        keep = keep & (np.asarray(applicability_label) == PETERS_SECULAR_OK)
    return keep


@dataclass
class PostFirstMergerTargets(PostFirstMergerCandidates):
    """The admitted post-first-merger target family: the candidates a run actually minimizes over.

    Same columns as :class:`PostFirstMergerCandidates`, restricted to the rows that pass the
    reference eccentricity interval and, if the run asks for it, the Peters applicability
    criteria. ``source_index`` gives each target's row in the candidate table it came from.
    """
    source_index: np.ndarray


def select_admitted_targets(
    candidates: PostFirstMergerCandidates,
    settings: AnalysisSettings,
) -> PostFirstMergerTargets:
    """Apply the post-first-merger-target family cuts.

    Accepts a candidate table or an already-admitted one, so a family can be narrowed further;
    ``source_index`` keeps pointing at the original candidate rows either way.
    """
    keep = target_is_selected(candidates.e_ref, candidates.applicability_label, settings)
    columns = {
        field.name: getattr(candidates, field.name)[keep]
        for field in dataclass_fields(PostFirstMergerCandidates)
    }
    kept = np.flatnonzero(keep)
    source_index = (
        candidates.source_index[kept]
        if isinstance(candidates, PostFirstMergerTargets)
        else kept
    )
    return PostFirstMergerTargets(source_index=source_index, **columns)


# -------------------------------------------------------------------------------------------------
# Progenitor phase grid
# -------------------------------------------------------------------------------------------------

def phase_reflection_indices(phase_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find sampled phases equivalent under reflection and full-orbit wrapping.

    Return representative indexes and an inverse map restoring the original order. Both true and
    mean anomaly have this symmetry for the direction-unrestricted minimum recoil.
    """
    phase = np.asarray(phase_rad, dtype=float)
    if phase.ndim != 1 or not np.all(np.isfinite(phase)):
        raise ValueError("Phases must be a finite one-dimensional array.")

    # fmod preserves tiny phases near zero, unlike adding pi before wrapping
    wrapped = np.fmod(phase, 2.0 * np.pi)
    wrapped = np.where(wrapped > np.pi, wrapped - 2.0 * np.pi, wrapped)
    wrapped = np.where(wrapped < -np.pi, wrapped + 2.0 * np.pi, wrapped)
    folded = np.abs(wrapped)
    order = np.argsort(folded, kind="stable")
    tolerance = 8.0 * np.finfo(float).eps * min(2.0 * np.pi, np.max(np.abs(phase), initial=0.0))
    distinct = np.unique(phase)
    if len(distinct) > 1:
        tolerance = min(tolerance, float(np.min(np.diff(distinct))) / 4.0)

    def equivalent(left, right):
        if left == right:
            return True
        difference_turns = np.rint((left - right) / (2.0 * np.pi))
        reflection_turns = np.rint((left + right) / (2.0 * np.pi))
        return ((difference_turns != 0 and
                 abs(left - right - difference_turns * (2.0 * np.pi)) <= tolerance)
                or abs(left + right - reflection_turns * (2.0 * np.pi)) <= tolerance)

    representatives = []
    inverse = np.empty(len(phase), dtype=np.intp)
    for index in order:
        if (not representatives or folded[index] - folded[representatives[-1]] > tolerance
                or not equivalent(phase[index], phase[representatives[-1]])):
            representatives.append(int(index))
        inverse[index] = len(representatives) - 1
    return np.asarray(representatives, dtype=np.intp), inverse


def phase_grid_rad(phase_min_deg: float, phase_max_deg: float, n_phase: int) -> np.ndarray:
    """Progenitor phase grid over the requested range, in radians.

    A range spanning a whole orbit wraps, so the upper endpoint is dropped: -180 and +180 deg are
    the same phase, and sampling both would count it twice and stretch the spacing to 360/(n-1)
    instead of 360/n. A partial range does not wrap, so both ends are kept.
    """
    if n_phase < 1:
        raise ValueError("n_phase must be at least 1.")
    wraps = math.isclose(phase_max_deg - phase_min_deg, 360.0, rel_tol=0.0, abs_tol=1e-9)

    return np.linspace(
        np.radians(phase_min_deg),
        np.radians(phase_max_deg),
        n_phase,
        endpoint=not wraps,
    )


def _kepler_residual(eccentric_anomaly, eccentricity, mean_anomaly):
    """Evaluate Kepler's equation accurately for eccentric anomalies close to zero."""
    anomaly = np.asarray(eccentric_anomaly, dtype=float)
    squared = anomaly ** 2
    # Series for E - sin(E), avoiding cancellation near periapsis
    difference = np.where(
        np.abs(anomaly) < 0.1,
        anomaly * squared * (
            1 / 6 + squared * (-1 / 120 + squared * (
                1 / 5040 + squared * (-1 / 362880 + squared * (
                    1 / 39916800 - squared / 6227020800
                ))
            ))
        ),
        anomaly - np.sin(anomaly),
    )
    return (1 - eccentricity) * anomaly + eccentricity * difference - mean_anomaly


def mean_to_true_anomaly_rad(mean_anomaly, e_out) -> np.ndarray:
    """Convert outer mean anomaly to true anomaly for 0 <= e_out < 1, in radians.

    Use vectorized Newton iteration, verify each root, and apply a bracketed fallback where
    needed. Broadcast finite inputs and return angles in [-pi, pi).
    """
    from scipy.optimize import brentq

    M, e = np.broadcast_arrays(
        np.asarray(mean_anomaly, dtype=float), np.asarray(e_out, dtype=float)
    )
    if np.any(~np.isfinite(e) | (e < 0.0) | (e >= 1.0)):
        raise ValueError("e_out must be finite and satisfy 0 <= e_out < 1.")
    if np.any(~np.isfinite(M)):
        raise ValueError("mean_anomaly must be finite.")

    # Preserve tiny anomalies already in range instead of rounding them through addition of pi
    M_wrapped = np.where(
        (M >= -np.pi) & (M < np.pi), M, (M + np.pi) % (2 * np.pi) - np.pi
    )

    # Shortcut for a circular orbit, where the true anomaly is the mean anomaly.
    if not M_wrapped.size or np.all(e == 0.0):
        return M_wrapped

    # Initial guess
    E = np.where(
        np.abs(M_wrapped) < np.pi / 2.0,
        M_wrapped / np.maximum(1.0 - e, 1e-3),
        np.sign(M_wrapped) * np.pi,
    )
    E = np.clip(E, -np.pi, np.pi)

    # Fast Newton iteration; failed elements are repaired below
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(80):
            residual = E - e * np.sin(E) - M_wrapped
            derivative = 1.0 - e * np.cos(E)
            E_new = E - residual / np.maximum(derivative, 1e-14)
            if np.max(np.abs(E_new - E)) < 2e-13:
                E = E_new
                break
            E = E_new

    # Verify each root with a stable residual and its local derivative
    bounded_anomaly = np.clip(E, -np.pi, np.pi)
    residual = _kepler_residual(bounded_anomaly, e, M_wrapped)
    derivative = (1 - e) + 2 * e * np.sin(bounded_anomaly / 2) ** 2
    root_tolerance = 8 * np.finfo(float).eps * np.abs(bounded_anomaly)
    converged = (
        np.isfinite(E) & (np.abs(E) <= np.pi)
        & (np.abs(residual) <= derivative * root_tolerance)
    )

    # Kepler's equation is monotone, so each remaining root has a bracket in [0, pi]
    E = np.array(E, copy=True)
    for index in np.flatnonzero(~converged):
        mean = float(M_wrapped.flat[index])
        root = brentq(
            _kepler_residual, 0., np.pi, args=(float(e.flat[index]), abs(mean)),
            xtol=np.nextafter(0., 1.), rtol=4 * np.finfo(float).eps, maxiter=256,
        )
        E.flat[index] = np.copysign(root, mean)

    sin_half = np.sqrt(1.0 + e) * np.sin(E / 2.0)
    cos_half = np.sqrt(1.0 - e) * np.cos(E / 2.0)
    true_anomaly = 2 * np.arctan2(sin_half, cos_half)
    return np.where(true_anomaly >= np.pi, true_anomaly - 2 * np.pi, true_anomaly)


def as_true_anomaly_rad(phase_rad, e_out, phase_measure: str) -> np.ndarray:
    """Interpret the progenitor phase coordinate and return the true anomaly, in radians.

    The kinematics only ever need the true anomaly; the phase measure decides what the grid
    coordinate means on the way in.
    """
    if phase_measure == PHASE_MEAN_ANOMALY:
        return mean_to_true_anomaly_rad(phase_rad, e_out)
    if phase_measure == PHASE_TRUE_ANOMALY:
        return np.asarray(phase_rad, dtype=float)
    raise ValueError(f"phase_measure must be one of {PHASE_MEASURES}, got {phase_measure!r}.")


# -------------------------------------------------------------------------------------------------
# Pre-kick outer orbit
# -------------------------------------------------------------------------------------------------

def pre_outer_state(
    a_out_au,
    e_out,
    nu_out_rad,
    m_inner_pre_msun: float,
    m3_msun: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separation and relative velocity of the outer orbit just before the first merger.

    The three orbital parameters can be a full grid, a 2D slice with one coordinate held fixed,
    or three scalars for a single progenitor. Returns the separation R and the radial and
    transverse velocity components in SI units.
    """
    e_out = np.asarray(e_out, dtype=float)
    if np.any(e_out < 0.0) or np.any(e_out >= 1.0):
        raise ValueError("e_out must satisfy 0 <= e_out < 1.")

    a_si = a_out_au * AU_SI
    mu_pre_si = G_SI * (m_inner_pre_msun + m3_msun) * MSUN_SI

    p_si = a_si * (1.0 - e_out) * (1.0 + e_out)
    # Avoid subtracting nearly equal numbers close to a high-e apocenter.
    denom = (1.0 - e_out) + 2.0 * e_out * np.cos(nu_out_rad / 2.0)**2

    R_si = p_si / denom
    sqrt_mu_over_p = np.sqrt(mu_pre_si / p_si)

    Vr_pre_si = sqrt_mu_over_p * e_out * np.sin(nu_out_rad)
    Vt_pre_si = sqrt_mu_over_p * denom

    return R_si, Vr_pre_si, Vt_pre_si


# -------------------------------------------------------------------------------------------------
# Recoil kinematics
# -------------------------------------------------------------------------------------------------

def kick_speeds_for_target(
    R_si: np.ndarray,
    Vr_pre_si: np.ndarray,
    Vt_pre_si: np.ndarray,
    mu_post_si: float,
    a_post_si: float,
    rp_si: float,
    ra_si: float,
    epost: float,
) -> np.ndarray:
    """Squared recoil speed needed to reach a post-first-merger target orbit.

    The recoil is instantaneous, so the separation R is the same before and after it. The target
    must therefore reach R, i.e. r_p <= R <= r_a, and there the post-kick state follows from the
    conservation of energy and angular-momentum:

        V^2      = mu (2/R - 1/a_post)
        Vt_post  = h_post / R,   h_post = sqrt(mu a_post (1 - e_post^2))
        Vr_post  = +/- sqrt(V^2 - Vt_post^2)
        vk^2     = (Vr_post - Vr_pre)^2 + (Vt_post - Vt_pre)^2

    Both signs of Vr_post are physically allowed, so the smaller of the two branches is returned.

    Returns the squared kick speed in SI units, ``nan`` wherever the target cannot be reached.
    """
    R_si = np.asarray(R_si, dtype=float)
    a_post_si = float(a_post_si)
    epost = float(epost)
    rp_si, ra_si = float(rp_si), float(ra_si)

    # Use a separate local scale at each endpoint: scaling the pericenter
    # tolerance by the apocenter would admit large errors for nearly radial orbits.
    rtol = 16.0 * np.finfo(float).eps
    near_rp = np.abs(R_si - rp_si) <= rtol * rp_si
    near_ra = np.abs(R_si - ra_si) <= rtol * ra_si
    allowed_R = ((R_si >= rp_si) | near_rp) & ((R_si <= ra_si) | near_ra)
    if not np.any(allowed_R):
        return np.full(R_si.shape, np.nan)

    R_eval = np.where(near_rp, rp_si, np.where(near_ra, ra_si, R_si))
    h_post = math.sqrt(mu_post_si * a_post_si * (1.0 - epost) * (1.0 + epost))
    Vt_post = h_post / R_eval
    Vr2 = (mu_post_si / a_post_si) * ((R_eval - rp_si) / R_eval) * ((ra_si - R_eval) / R_eval)
    Vr_abs = np.sqrt(np.maximum(Vr2, 0.0))

    ok = allowed_R & (Vr2 >= 0.0)

    # Both signs of the post-kick radial velocity are allowed. Choose the minimum of both branches.
    vk2_outgoing = (Vr_abs - Vr_pre_si)**2 + (Vt_post - Vt_pre_si)**2
    vk2_ingoing = (-Vr_abs - Vr_pre_si)**2 + (Vt_post - Vt_pre_si)**2
    vk2_si = np.minimum(vk2_outgoing, vk2_ingoing)

    return np.where(ok, vk2_si, np.nan)


@dataclass
class KickSolution:
    """Best target orbit at every point of a grid.

    The star quantities are the quantities for which ``vk2_si`` is minimal. ``vk2_si`` is ``inf``
    and the star quantities are ``nan`` where no target survives the cuts or is kinematically
    reachable. ``valid`` is the corresponding boolean mask.
    """
    vk2_si: np.ndarray
    epost_star: np.ndarray
    eref_star: np.ndarray
    valid: np.ndarray

    @property
    def vk_kms(self) -> np.ndarray:
        """Minimum recoil speed in km/s, ``nan`` where invalid."""
        out = np.full(self.vk2_si.shape, np.nan)
        out[self.valid] = np.sqrt(self.vk2_si[self.valid]) / 1000.0
        return out


def minimize_kick_over_targets(
    R_si: np.ndarray,
    Vr_pre_si: np.ndarray,
    Vt_pre_si: np.ndarray,
    targets: PostFirstMergerTargets,
    mu_post_si: float,
) -> KickSolution:
    """Minimize the recoil speed over the admitted target family.

    Every target given is minimized over. Targets are scanned for increasing post-first-merger
    eccentricity, and a strictly smaller ``vk2_si`` is needed to replace the current best, so
    exact ties keep the least eccentric target.
    """
    R_si = np.asarray(R_si, dtype=float)
    shape = R_si.shape

    # Sort cells once so each target visits only its potentially reachable radius range
    order = np.argsort(R_si, axis=None)
    radii = R_si.ravel()[order]
    radial_velocities = np.broadcast_to(Vr_pre_si, shape).ravel()[order]
    tangential_velocities = np.broadcast_to(Vt_pre_si, shape).ravel()[order]
    best_vk2 = np.full(radii.shape, np.inf)
    best_epost = np.full(radii.shape, np.nan)
    best_eref = np.full(radii.shape, np.nan)
    # Enclose the recoil kernel's 16-epsilon endpoint tolerance, including boundary roundoff
    search_tolerance = 32 * np.finfo(float).eps

    for epost, a_post_si, rp_si, ra_si, e_ref in zip(
        targets.epost, targets.a_si, targets.rp_si, targets.ra_si, targets.e_ref
    ):
        start = np.searchsorted(radii, rp_si * (1 - search_tolerance), side='left')
        stop = np.searchsorted(radii, ra_si * (1 + search_tolerance), side='right')
        if start >= stop:
            continue
        vk2_si = kick_speeds_for_target(
            radii[start:stop], radial_velocities[start:stop], tangential_velocities[start:stop],
            mu_post_si, a_post_si, rp_si, ra_si, epost,
        )
        # Keep target order and strict improvement so ties retain the original chosen target
        improve = vk2_si < best_vk2[start:stop]
        best_vk2[start:stop][improve] = vk2_si[improve]
        best_epost[start:stop][improve] = float(epost)
        best_eref[start:stop][improve] = float(e_ref)

    # Restore the original cell order and shape, including scalar and empty grids
    restored = []
    for values in (best_vk2, best_epost, best_eref):
        original_order = np.empty(values.shape, dtype=float)
        original_order[order] = values
        restored.append(original_order.reshape(shape))
    return KickSolution(
        vk2_si=restored[0], epost_star=restored[1], eref_star=restored[2],
        valid=np.isfinite(restored[0]),
    )


# -------------------------------------------------------------------------------------------------
# Minimum-recoil-velocity map slices
# -------------------------------------------------------------------------------------------------

SLICE_CHOICES = ("aout-nu", "eout-aout", "eout-nu")


@dataclass
class KickMapSlice:
    """One axis-aligned slice through the 3D minimum-recoil-velocity map over (a_out, e_out, nu_m).

    ``plane`` names the two scanned coordinates and ``fixed_name`` the fixed one;
    ``fixed_value`` is in the unit that ``fixed_name`` states:

    ==============  ============  ==================  =====================
    plane           fixed_name    fixed_value unit    scanned axes (y, x)
    ==============  ============  ==================  =====================
    ``aout-nu``     e_out         dimensionless       a_out [AU], nu_m [rad]
    ``eout-aout``   nu_m_deg      degrees             a_out [AU], e_out
    ``eout-nu``     a_out_AU      AU                  e_out, nu_m [rad]
    ==============  ============  ==================  =====================

    Every array has shape ``(len(y_values), len(x_values))``, so a field can be passed straight to
    ``pcolormesh(x_values, y_values, field)``.
    """

    plane: str
    fixed_name: str
    fixed_value: float
    x_values: np.ndarray
    y_values: np.ndarray
    vk_min_kms: np.ndarray
    epost_star: np.ndarray
    eref_star: np.ndarray
    kinematic_valid: np.ndarray
    ma_valid: np.ndarray
    selected_domain: np.ndarray
    valid: np.ndarray
    phase_measure: str = PHASE_MEAN_ANOMALY


def _slice_coordinate_grids(
    plane: str,
    fixed_value: float,
    a_grid_au: np.ndarray,
    e_out_grid: np.ndarray,
    nu_grid_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return x, y, and broadcast A/E/NU grids for one requested plane."""
    if plane == "aout-nu":
        if not (0.0 <= fixed_value < 1.0):
            raise ValueError("Fixed e_out must satisfy 0 <= e_out < 1.")
        NU, A = np.meshgrid(nu_grid_rad, a_grid_au, indexing="xy")
        E = np.full_like(A, fixed_value)
        return nu_grid_rad, a_grid_au, A, E, NU, "e_out"

    if plane == "eout-aout":
        E, A = np.meshgrid(e_out_grid, a_grid_au, indexing="xy")
        NU = np.full_like(A, math.radians(fixed_value))
        return e_out_grid, a_grid_au, A, E, NU, "nu_m_deg"

    if plane == "eout-nu":
        if fixed_value <= 0.0:
            raise ValueError("Fixed a_out must be positive.")
        NU, E = np.meshgrid(nu_grid_rad, e_out_grid, indexing="xy")
        A = np.full_like(E, fixed_value)
        return nu_grid_rad, e_out_grid, A, E, NU, "a_out_AU"

    raise ValueError(f"Unknown slice {plane!r}; choose from {', '.join(SLICE_CHOICES)}.")


def compute_min_kick_map_slice(
    plane: str,
    fixed_value: float,
    a_grid_au: np.ndarray,
    e_out_grid: np.ndarray,
    nu_grid_rad: np.ndarray,
    targets: PostFirstMergerTargets,
    m1_inner_msun: float,
    m2_inner_msun: float,
    mrem_msun: float,
    m3_msun: float,
    inclination_rad: float = math.pi,
    settings: AnalysisSettings = AnalysisSettings(),
    use_phase_symmetry: bool = True,
) -> KickMapSlice:
    """Minimize the recoil over an axis-aligned 2D plane of the progenitor grid.

    See :class:`KickMapSlice` for the plane names and the unit of ``fixed_value``. A 3D cube can be
    assembled by stacking slices. Mirrored phase columns are computed once and restored before
    returning. Set ``use_phase_symmetry=False`` for a full-grid reference calculation.
    """
    x, y, A_au, E_out, NU_rad, fixed_name = _slice_coordinate_grids(
        plane,
        float(fixed_value),
        np.asarray(a_grid_au, dtype=float),
        np.asarray(e_out_grid, dtype=float),
        np.asarray(nu_grid_rad, dtype=float),
    )

    region = compute_allowed_region(
        A_au,
        E_out,
        NU_rad,
        targets,
        m1_inner_msun,
        m2_inner_msun,
        mrem_msun,
        m3_msun,
        settings=replace(settings, recoil_ceiling_kms=None),
        inclination_rad=inclination_rad,
        phase_axis=1 if use_phase_symmetry and plane != "eout-aout" else None,
    )

    ma_valid = A_au >= region.ma_aout_min_au
    selected_domain = region.dynamically_stable
    valid = region.recoil_compatible

    vk_min_kms = np.where(valid, region.vk_min_kms, np.nan)
    epost_star = np.where(valid, region.epost_star, np.nan)
    eref_star = np.where(valid, region.eref_star, np.nan)

    return KickMapSlice(
        plane=plane,
        fixed_name=fixed_name,
        fixed_value=float(fixed_value),
        x_values=x,
        y_values=y,
        vk_min_kms=vk_min_kms,
        epost_star=epost_star,
        eref_star=eref_star,
        kinematic_valid=region.reachable,
        ma_valid=ma_valid,
        selected_domain=selected_domain,
        valid=valid,
        phase_measure=settings.phase_measure,
    )


# -------------------------------------------------------------------------------------------------
# Allowed recoil-compatible progenitor region
# -------------------------------------------------------------------------------------------------

@dataclass
class AllowedRegion:
    """The allowed progenitor region A.

    Each array holds one entry per progenitor point, laid out like the grid itself.
    ``reachable``, ``dynamically_stable`` and ``within_recoil_ceiling`` say True or False for
    each condition, and ``recoil_compatible`` is True where all three hold. That is the allowed
    recoil-compatible progenitor region. Projecting it onto the individual axes gives the
    constraints on a_out, e_out and nu_m.
    """
    vk_min_kms: np.ndarray
    epost_star: np.ndarray
    eref_star: np.ndarray
    reachable: np.ndarray
    dynamically_stable: np.ndarray
    within_recoil_ceiling: np.ndarray
    recoil_compatible: np.ndarray
    ma_aout_min_au: np.ndarray
    settings: AnalysisSettings

    @property
    def n_points(self) -> int:
        return int(self.recoil_compatible.size)

    @property
    def n_compatible(self) -> int:
        return int(np.count_nonzero(self.recoil_compatible))

    @property
    def is_empty(self) -> bool:
        """True when no progenitor point survives, i.e. there is no recoil-compatible region."""
        return self.n_compatible == 0

    def reasons(self) -> list[str]:
        """Say in words which condition removed the points that are not compatible.

        A point is blamed on the first condition it fails, so the counts do not overlap and,
        together with the compatible points, add up to the whole grid.
        """
        out: list[str] = []
        total = self.n_points
        not_reachable = int(np.count_nonzero(~self.reachable))
        if not_reachable:
            out.append(
                f"no admitted Peters target is kinematically reachable "
                f"({not_reachable}/{total} points)"
            )
        not_stable = int(np.count_nonzero(self.reachable & ~self.dynamically_stable))
        if not_stable:
            out.append(
                f"outer orbit does not satisfy the Mardling-Aarseth stability criterion "
                f"({not_stable}/{total} points)"
            )
        not_recoil_compatible = int(np.count_nonzero(
            self.reachable & self.dynamically_stable & ~self.within_recoil_ceiling
        ))
        if not_recoil_compatible:
            ceiling = self.settings.recoil_ceiling_kms
            out.append(
                f"required recoil exceeds the {ceiling:g} km/s ceiling "
                f"({not_recoil_compatible}/{total} points)"
            )
        if not out:
            out.append("all progenitor points are recoil-compatible")
        return out


def compute_allowed_region(
    a_out_au,
    e_out,
    nu_out_rad,
    targets: PostFirstMergerTargets,
    m1_inner_msun: float,
    m2_inner_msun: float,
    mrem_msun: float,
    m3_msun: float,
    settings: AnalysisSettings = AnalysisSettings(),
    inclination_rad: float = math.pi,
    phase_axis: Optional[int] = None,
) -> AllowedRegion:
    """Compute the allowed region on a progenitor grid, or judge a single progenitor.

    ``a_out_au``, ``e_out`` and ``nu_out_rad`` can be either a full 3D grid, a 2D slice, or three
    scalars describing one system. The target family is fixed by ``targets``; ``settings`` decides
    which progenitor points survive.

    Set ``phase_axis`` to the array dimension that varies orbital phase to avoid repeating
    calculations for mirrored phases. The function calculates each mirrored pair once, then copies
    its results to both phase positions. The returned arrays still contain every requested grid
    point, so counts, projected distributions and plots include both positions.
    Leave ``phase_axis=None`` to calculate every point separately.
    """
    a_out_au, E_out, NU_rad = np.broadcast_arrays(
        np.asarray(a_out_au, dtype=float),
        np.asarray(e_out, dtype=float),
        np.asarray(nu_out_rad, dtype=float),
    )

    if phase_axis is not None and a_out_au.size:
        if not -a_out_au.ndim <= phase_axis < a_out_au.ndim:
            raise ValueError("phase_axis is outside the broadcast grid dimensions.")
        axis = phase_axis % a_out_au.ndim
        selector = [0] * a_out_au.ndim
        selector[axis] = slice(None)
        phases = NU_rad[tuple(selector)]
        phase_shape = [1] * a_out_au.ndim
        phase_shape[axis] = len(phases)
        if (not np.all(NU_rad == phases.reshape(phase_shape))
                or not np.all(a_out_au == np.take(a_out_au, [0], axis=axis))
                or not np.all(E_out == np.take(E_out, [0], axis=axis))):
            raise ValueError("phase_axis must describe an independent Cartesian phase grid.")
        representatives, inverse = phase_reflection_indices(phases)
        if len(representatives) < len(phases):
            reduced = compute_allowed_region(
                np.take(a_out_au, representatives, axis=axis),
                np.take(E_out, representatives, axis=axis),
                np.take(NU_rad, representatives, axis=axis),
                targets, m1_inner_msun, m2_inner_msun, mrem_msun, m3_msun,
                settings=settings, inclination_rad=inclination_rad,
            )
            return replace(reduced, **{
                field.name: np.take(getattr(reduced, field.name), inverse, axis=axis)
                for field in dataclass_fields(reduced) if field.name != "settings"
            })

    m_inner_pre_msun = m1_inner_msun + m2_inner_msun
    M_post_msun = mrem_msun + m3_msun
    mu_post_si = G_SI * M_post_msun * MSUN_SI

    NU_true = as_true_anomaly_rad(NU_rad, E_out, settings.phase_measure)
    R_si, Vr_pre_si, Vt_pre_si = pre_outer_state(
        a_out_au,
        e_out=E_out,
        nu_out_rad=NU_true,
        m_inner_pre_msun=m_inner_pre_msun,
        m3_msun=m3_msun,
    )

    solution = minimize_kick_over_targets(
        R_si,
        Vr_pre_si,
        Vt_pre_si,
        targets,
        mu_post_si,
    )

    vk_min_kms = solution.vk_kms
    reachable = solution.valid

    a_in_stability_au = inner_separation_for_stability_au(
        m1_inner_msun, m2_inner_msun, settings
    )
    ma_aout_min_au = np.broadcast_to(
        mardling_aarseth_aout_min_au(
            a_in_stability_au,
            e_out=E_out,
            m_inner_msun=m_inner_pre_msun,
            m3_msun=m3_msun,
            inclination_rad=inclination_rad,
        ),
        a_out_au.shape,
    )

    if settings.require_stability:
        dynamically_stable = a_out_au >= ma_aout_min_au
    else:
        dynamically_stable = np.ones(a_out_au.shape, dtype=bool)

    if settings.recoil_ceiling_kms is None:
        within_recoil_ceiling = np.ones(a_out_au.shape, dtype=bool)
    else:
        within_recoil_ceiling = vk_min_kms <= settings.recoil_ceiling_kms

    return AllowedRegion(
        vk_min_kms=vk_min_kms,
        epost_star=solution.epost_star,
        eref_star=solution.eref_star,
        reachable=reachable,
        dynamically_stable=dynamically_stable,
        within_recoil_ceiling=within_recoil_ceiling,
        recoil_compatible=reachable & dynamically_stable & within_recoil_ceiling,
        ma_aout_min_au=ma_aout_min_au,
        settings=settings,
    )


# -------------------------------------------------------------------------------------------------
# System-level Peters-reliability analysis
# -------------------------------------------------------------------------------------------------

@dataclass
class SystemReliability:
    """System-level Peters-reliability classification.

    ``r_sec`` is the recoil-compatible progenitor volume retained when the target family is
    restricted to targets satisfying the Peters applicability conditions. It is ``None`` when the
    recoil-compatible progenitor region for all post-first-merger target orbits is empty.
    """
    volume_all_targets: float
    volume_secular_targets: float
    r_sec: Optional[float]
    threshold: float
    verdict: str
    region_all_targets: AllowedRegion
    region_secular_targets: AllowedRegion

    @property
    def n_physical_cells(self) -> int:
        """Cells in the physical domain: those passing the stability condition."""
        return int(np.count_nonzero(self.region_all_targets.dynamically_stable))

    @property
    def f_all(self) -> Optional[float]:
        """Fraction of the physical domain that is recoil-compatible, over all targets."""
        n = self.n_physical_cells
        return self.volume_all_targets / n if n else None

    @property
    def f_sec(self) -> Optional[float]:
        """Same fraction, over the targets meeting the Peters applicability conditions."""
        n = self.n_physical_cells
        return self.volume_secular_targets / n if n else None

    @property
    def peters_reliable(self) -> bool:
        return self.verdict == VERDICT_PETERS_RELIABLE

    @property
    def recoil_compatible(self) -> bool:
        return self.volume_all_targets > 0.0

    def summary(self) -> str:
        if self.r_sec is None:
            return (
                f"{self.verdict}: no recoil-compatible progenitor found, "
                f"so R_sec is undefined"
            )
        return (
            f"{self.verdict}: R_sec = {self.r_sec:.3f} "
            f"({'>=' if self.peters_reliable else '<'} {self.threshold:g})"
        )


def assess_peters_reliability(
    a_out_au,
    e_out,
    nu_out_rad,
    candidates: PostFirstMergerCandidates,
    m1_inner_msun: float,
    m2_inner_msun: float,
    mrem_msun: float,
    m3_msun: float,
    settings: AnalysisSettings = AnalysisSettings(),
    inclination_rad: float = math.pi,
) -> SystemReliability:
    """Classify one system as Peters-reliable, unreliable, or incompatible.

    One progenitor region, two target families: the same region machinery is run once over every
    reference-eccentricity-admitted target, and once over the subset that also meets the Peters
    applicability conditions. R_sec is the ratio of the retained volumes.

    Volume is the number of compatible cells, so the progenitor grid must already be laid out in
    the measure the volume is meant to carry; this does not reweight it.
    """
    targets_all = select_admitted_targets(
        candidates, replace(settings, require_peters_applicable=False)
    )
    targets_secular = select_admitted_targets(
        candidates, replace(settings, require_peters_applicable=True)
    )

    common = dict(
        m1_inner_msun=m1_inner_msun,
        m2_inner_msun=m2_inner_msun,
        mrem_msun=mrem_msun,
        m3_msun=m3_msun,
        settings=settings,
        inclination_rad=inclination_rad,
    )
    region_all = compute_allowed_region(a_out_au, e_out, nu_out_rad, targets_all, **common)
    region_sec = compute_allowed_region(a_out_au, e_out, nu_out_rad, targets_secular, **common)

    volume_all = float(np.count_nonzero(region_all.recoil_compatible))
    volume_sec = float(np.count_nonzero(region_sec.recoil_compatible))

    if volume_all <= 0.0:
        r_sec = None
        verdict = VERDICT_NO_COMPATIBLE_PROGENITOR
    else:
        r_sec = volume_sec / volume_all
        verdict = (
            VERDICT_PETERS_RELIABLE
            if r_sec >= settings.r_sec_threshold
            else VERDICT_RECOIL_COMPATIBLE_UNRELIABLE
        )

    return SystemReliability(
        volume_all_targets=volume_all,
        volume_secular_targets=volume_sec,
        r_sec=r_sec,
        threshold=settings.r_sec_threshold,
        verdict=verdict,
        region_all_targets=region_all,
        region_secular_targets=region_sec,
    )
