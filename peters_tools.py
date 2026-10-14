#!/usr/bin/env python3
"""Peters inspiral equations and eccentric-frequency diagnostics.

The inspiral equations use geometric code units with G = c = M_total = 1.
Component masses enter through eta = m1 m2 / M_total^2. Frequency diagnostics
use SI lengths, solar masses, and hertz.
"""

import math
from functools import lru_cache
from math import expm1, log1p, sqrt

from scipy.integrate import quad

from constants import G_SI, MSUN_SI


# -------------------------------------------------------------------------------------------------
# Peters coefficients and numerical tolerance
# -------------------------------------------------------------------------------------------------

# Below this eccentricity the circular merger-time and separation formulas are used
CIRCULAR_ECCENTRICITY_TOLERANCE = 1e-12

_E_POWER = 29.0 / 19.0
_POLYNOMIAL_COEFFICIENT = 121.0 / 304.0
_POLYNOMIAL_POWER = 1181.0 / 2299.0
_ENDPOINT_VALUE = (1.0 + _POLYNOMIAL_COEFFICIENT)**_POLYNOMIAL_POWER


def _check_eccentricity(e):
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1.")


def _beta(eta):
    """Radiation-reaction coefficient beta.

    beta = (64/5) G^3 m1 m2 (m1+m2) / c^5 in physical units;
    beta = (64/5) eta in the geometric units used here.
    """
    if eta <= 0.0:
        raise ValueError("eta must be positive.")
    return (64.0 / 5.0) * eta


# -------------------------------------------------------------------------------------------------
# Peters eccentric inspiral
# -------------------------------------------------------------------------------------------------

def _peters_integrand_regularized(u):
    """Regularized integrand of the Peters eccentricity integral.

    Set u = e'/sqrt(1-e'^2), hence e' = u/sqrt(1+u^2). Since de'/(1-e'^2)^(3/2) = du, the
    transformed integrand is F(u) = e'^(29/19) [1 + (121/304) e'^2]^(1181/2299).
    """
    ep = u / sqrt(1.0 + u * u)
    return ep**_E_POWER * (1.0 + _POLYNOMIAL_COEFFICIENT * ep * ep)**_POLYNOMIAL_POWER


def _peters_tail_correction(v):
    """Stable (F(1/v) - F(infinity))/v^2 for the regularized tail.

    This evaluates the tail of the Peters eccentricity integral using
    v = 1/u and F from :func:`_peters_integrand_regularized`.
    """
    if v == 0.0:
        coefficient = (
            -0.5 * _E_POWER
            - _POLYNOMIAL_POWER * _POLYNOMIAL_COEFFICIENT / (1.0 + _POLYNOMIAL_COEFFICIENT)
        )
        return _ENDPOINT_VALUE * coefficient

    v2 = v * v
    log_ratio = -0.5 * _E_POWER * log1p(v2) + _POLYNOMIAL_POWER * (
        log1p(v2 / (1.0 + _POLYNOMIAL_COEFFICIENT)) - log1p(v2)
    )
    return _ENDPOINT_VALUE * expm1(log_ratio) / v2


@lru_cache(maxsize=1)
def _peters_regularized_correction_to_infinity():
    """Finite correction K for the regularized Peters eccentricity integral.

    With C = F(infinity), K = integral_0^infinity [F(u) - C] du.
    Thus I(e) = C U + K - integral_U^infinity [F(u) - C] du, where U = e/sqrt(1-e^2).
    The tail is evaluated using v = 1/u.
    """
    lower, _ = quad(
        lambda u: _peters_integrand_regularized(u) - _ENDPOINT_VALUE,
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    tail, _ = quad(
        _peters_tail_correction,
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    return lower + tail


def peters_eccentricity_integral(e):
    """The Peters integral I(e), with its endpoint regularized.

        I(e) = integral_0^e e'^(29/19) [1 + (121/304) e'^2]^(1181/2299) / (1-e'^2)^(3/2) de'.

    With u = e'/sqrt(1-e'^2), the singular measure cancels exactly:

        de' / (1-e'^2)^(3/2) = du.

    For u > 1, the transformed integrand's constant asymptote is integrated analytically. Its
    finite remainder is evaluated on v = 1/u in [0, 1], avoiding a quadrature interval that
    grows without bound as e -> 1.
    """
    _check_eccentricity(e)
    if e < CIRCULAR_ECCENTRICITY_TOLERANCE:
        return 0.0

    one_minus_e2 = (1.0 - e) * (1.0 + e)
    u_max = e / sqrt(one_minus_e2)

    if u_max <= 1.0:
        integral, _ = quad(
            _peters_integrand_regularized,
            0.0,
            u_max,
            epsabs=0.0,
            epsrel=1e-12,
        )
        return integral

    omitted_tail, _ = quad(
        _peters_tail_correction,
        0.0,
        1.0 / u_max,
        epsabs=1e-13,
        epsrel=1e-12,
    )
    return _ENDPOINT_VALUE * u_max + _peters_regularized_correction_to_infinity() - omitted_tail


def peters_c0(a, e):
    """Peters invariant c0 for 0 < e < 1.

        c0 = a (1-e^2) e^(-12/19) [1 + (121/304) e^2]^(-870/2299).

    c0 has the same length units as a and is conserved under orbit-averaged Peters evolution.
    """
    if not (0.0 < e < 1.0):
        raise ValueError("c0 requires 0 < e < 1.")
    if a <= 0.0:
        raise ValueError("a must be positive.")
    one_minus_e2 = (1.0 - e) * (1.0 + e)
    return a * one_minus_e2 * e**(-12.0 / 19.0) * (1.0 + 121.0 / 304.0 * e**2)**(-870.0 / 2299.0)


def peters_a_from_c0(c0, e):
    """Semi-major axis along a Peters trajectory.

        a(e) = c0 e^(12/19) [1 + (121/304) e^2]^(870/2299) / (1-e^2).

    This inverts :func:`peters_c0` for 0 < e < 1 and preserves the units of c0.
    """
    if not (0.0 < e < 1.0):
        raise ValueError("a(c0, e) requires 0 < e < 1.")
    if c0 <= 0.0:
        raise ValueError("c0 must be positive.")
    one_minus_e2 = (1.0 - e) * (1.0 + e)
    return c0 * e**(12.0 / 19.0) * (1.0 + 121.0 / 304.0 * e**2)**(870.0 / 2299.0) / one_minus_e2


def peters_time_eccentric(a, e, eta):
    """Peters merger time in G = c = M = 1 units.

        t_m(a, e) = (12/19) c0^4 I(e) / beta,    e > 0,
        t_m(a, 0) = a^4 / (4 beta),    e = 0.

    c0, I(e), and beta are evaluated by :func:`peters_c0`,
    :func:`peters_eccentricity_integral`, and :func:`_beta`, respectively.
    The circular branch is used below CIRCULAR_ECCENTRICITY_TOLERANCE.

    Parameters
    ----------
    a : float
        Initial semi-major axis in total-mass units, a/M.
    e : float
        Initial eccentricity, 0 <= e < 1.
    eta : float
        Symmetric mass ratio, eta = m1*m2/(m1+m2)^2.

    Returns
    -------
    t : float
        Merger time in total-mass time units.
    """
    _check_eccentricity(e)
    if a <= 0.0:
        raise ValueError("a must be positive.")

    beta = _beta(eta)

    if e < CIRCULAR_ECCENTRICITY_TOLERANCE:
        return a**4 / (4.0 * beta)

    c0 = peters_c0(a, e)
    return (12.0 / 19.0) * c0**4 * peters_eccentricity_integral(e) / beta


def solve_semi_major_axis_for_merger_time(t_target, e, eta):
    """Semi-major axis whose Peters merger time is t_target.

    Invert the eccentric merger-time relation, then recover a from the Peters invariant:

        c0 = [19 beta t_target / (12 I(e))]^(1/4),
        a = c0 e^(12/19) [1 + (121/304) e^2]^(870/2299) / (1-e^2).

    In the circular limit, a = (4 beta t_target)^(1/4), used below CIRCULAR_ECCENTRICITY_TOLERANCE.
    Only I(e) requires quadrature; no root-finding is needed for a.

    Parameters
    ----------
    t_target : float
        Target merger time in total-mass time units.
    e : float
        Initial eccentricity, 0 <= e < 1.
    eta : float
        Symmetric mass ratio.

    Returns
    -------
    a : float
        Semi-major axis in total-mass units.
    """
    _check_eccentricity(e)
    if t_target <= 0.0:
        raise ValueError("t_target must be positive.")

    beta = _beta(eta)

    if e < CIRCULAR_ECCENTRICITY_TOLERANCE:
        return (4.0 * beta * t_target)**0.25

    integral = peters_eccentricity_integral(e)
    if integral <= 0.0:
        raise ValueError(f"I(e) evaluated to {integral} at e={e}; cannot invert.")

    c0 = (19.0 * beta * t_target / (12.0 * integral))**0.25
    return peters_a_from_c0(c0, e)


# -------------------------------------------------------------------------------------------------
# Eccentricity-frequency diagnostics
# -------------------------------------------------------------------------------------------------

def eccentric_peak_gw_frequency_hz(a_si, e, total_mass_msun):
    """Wen peak GW frequency approximation in hertz.

        f_peak = sqrt(GM) / [pi (a (1-e^2))^(3/2)] * (1+e)^1.1954.

    Here a is in meters and M = total_mass_msun * MSUN_SI. This is a frequency approximation, not a
    Peters evolution equation. At e = 0 it reduces to twice the Keplerian orbital frequency.
    """
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1.")
    if total_mass_msun <= 0.0:
        raise ValueError("total_mass_msun must be positive.")
    if a_si <= 0.0:
        return math.inf

    mu_si = G_SI * total_mass_msun * MSUN_SI
    semi_latus_rectum_si = a_si * (1.0 - e) * (1.0 + e)
    return math.sqrt(mu_si) / (math.pi * semi_latus_rectum_si**1.5) * (1.0 + e)**1.1954


def peters_eccentricity_at_gw_frequency(a0_si, e0, total_mass_msun, f_ref_hz):
    """Eccentricity when Peters evolution reaches the reference peak GW frequency.

    For an orbit with initial f_peak(a0_si, e0) < f_ref_hz, solve:

    f_peak(a(e_ref), e_ref) = f_ref_hz,    0 <= e_ref <= e0

    Use :func:`peters_a_from_c0` with c0 fixed by :func:`peters_c0`, and the frequency from
    :func:`eccentric_peak_gw_frequency_hz`. Equivalently,

    f_peak(e) = sqrt(GM)/(pi c0^(3/2)) * e^(-18/19) * [1+(121/304)e^2]^(-1305/2299) * (1+e)^1.1954

    The code bisects the composed a(e), f_peak(a,e) relation. An initially circular orbit stays
    circular. When the initial peak frequency is already at or above f_ref_hz, return e0 as the
    reference eccentricity by convention; no backwards extrapolation is performed.
    """
    if f_ref_hz <= 0.0:
        raise ValueError("f_ref_hz must be positive.")
    if not (0.0 <= e0 < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1.")
    if e0 < 1e-14:
        return 0.0

    f0 = eccentric_peak_gw_frequency_hz(a0_si, e0, total_mass_msun)
    if f0 >= f_ref_hz:
        return e0

    c0_si = peters_c0(a0_si, e0)

    def f_of_e(e):
        a_si = peters_a_from_c0(c0_si, e)
        return eccentric_peak_gw_frequency_hz(a_si, e, total_mass_msun)

    lo = 1e-14
    hi = e0
    if f_of_e(lo) < f_ref_hz:
        return 0.0

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        # Once no distinct midpoint is representable, further bisections do nothing
        if mid == lo or mid == hi:
            break
        if f_of_e(mid) >= f_ref_hz:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
