#!/usr/bin/env python3
"""Two-dimensional slices through the analytic three-dimensional kick map.

The progenitor map has coordinates ``(a_out, e_out, nu_m)``. Any of its three axis-aligned 2D
slices can be requested from this one script using the argument ``--slices``:

``aout-nu``
    Scan ``(a_out, nu_m)`` at one or more fixed ``e_out`` values.
``eout-aout``
    Scan ``(e_out, a_out)`` at one or more fixed ``nu_m`` values.
``eout-nu``
    Scan ``(e_out, nu_m)`` at one or more fixed ``a_out`` values.

At every progenitor point the recoil is minimized over the accessible Peters targets and both
post-kick radial branches. By default every target whose reference eccentricity falls in the
admitted interval is considered. ``--secular-only`` restricts a run to targets that also satisfy
the Peters applicability conditions, N_orb >= --min-peters-orbits and r_p/r_g >= --min-peters-rp-rg

Points whose minimum recoil falls below ``--recoil-ceiling-kms`` are the recoil-compatible ones,
shaded in the kick maps.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from triple_core import (
    PHASE_MEAN_ANOMALY,
    AnalysisSettings,
    KickMapSlice,
    compute_min_kick_map_slice,
    compute_post_first_merger_candidates,
    select_admitted_targets,
    make_epost_grid,
    phase_reflection_indices,
)

from triple_core import completed_tasks

from triple_io import (
    format_e, filter_suffix, axis_suffix, phase_suffix,
    _slice_file_stem, write_candidate_table, unique_filename_values,
)

from triple_cli import (
    parse_kick_maps_2d_args as parse_args,
    system_from_args,
    settings_from_args,
)


# -------------------------------------------------------------------------------------------------
# Build fixed post-first-merger eccentricity kick maps
# -------------------------------------------------------------------------------------------------

def compute_fixed_epost_kick_map(
    e_out: float,
    e_post_fixed: float,
    a_grid_au: np.ndarray,
    nu_grid_rad: np.ndarray,
    merger_time_days: float,
    m1_inner_msun: float,
    m2_inner_msun: float,
    mrem_msun: float,
    m3_msun: float,
    settings: AnalysisSettings = AnalysisSettings(),
    inclination_rad: float = math.pi,
    f_ref_hz: float = 10.0,
    min_peters_orbits: float = 10.0,
    min_peters_rp_rg: float = 20.0,
) -> np.ndarray:
    """Compute the minimum-recoil-velocity map at an exact e_post."""
    candidate = compute_post_first_merger_candidates(
        np.array([e_post_fixed], dtype=float),
        mrem_msun=mrem_msun,
        m3_msun=m3_msun,
        merger_time_days=merger_time_days,
        f_ref_hz=f_ref_hz,
        min_peters_orbits=min_peters_orbits,
        min_peters_rp_rg=min_peters_rp_rg,
    )
    kick_slice = compute_min_kick_map_slice(
        plane="aout-nu",
        fixed_value=e_out,
        a_grid_au=a_grid_au,
        e_out_grid=np.array([e_out]),
        nu_grid_rad=nu_grid_rad,
        targets=select_admitted_targets(candidate, settings),
        m1_inner_msun=m1_inner_msun,
        m2_inner_msun=m2_inner_msun,
        mrem_msun=mrem_msun,
        m3_msun=m3_msun,
        inclination_rad=inclination_rad,
        settings=settings,
    )
    return kick_slice.vk_min_kms


# -------------------------------------------------------------------------------------------------
# Coordinate and label helpers
# -------------------------------------------------------------------------------------------------

def eccentricity_filter_title(
    e_ref_min: Optional[float],
    e_ref_max: Optional[float],
    f_ref_hz: float,
) -> str:
    label = rf"e_{{{f_ref_hz:g}\,\mathrm{{Hz}}}}"
    if e_ref_min is None and e_ref_max is None:
        return ""
    if e_ref_min is None:
        return rf"requiring ${label}\leq {e_ref_max:.3g}$"
    if e_ref_max is None:
        return rf"requiring ${label}\geq {e_ref_min:.3g}$"
    return rf"requiring ${e_ref_min:.3g}\leq {label}\leq {e_ref_max:.3g}$"


def aout_plot_values(
    a_grid_au: np.ndarray,
    aout_axis: str,
    aout_rg_au: float,
) -> tuple[np.ndarray, str, float]:
    """Return y-axis values, y-label, and AU-to-axis conversion factor."""
    if aout_axis == "au":
        return a_grid_au, r"Pre-merger outer semi-major axis $a_{\rm out}$ [AU]", 1.0
    if aout_axis == "rg":
        return (
            a_grid_au / aout_rg_au,
            r"Pre-merger outer semi-major axis $a_{\rm out}/(GM/c^2)$",
            1.0 / aout_rg_au,
        )
    raise ValueError("aout_axis must be 'au' or 'rg'.")


def _phase_label(phase_measure: str) -> str:
    """Axis label for the progenitor phase coordinate, which --phase-measure selects."""
    if phase_measure == PHASE_MEAN_ANOMALY:
        return r"Outer mean anomaly at first merger $M_{\rm m}$ [deg]"
    return r"Outer true anomaly at first merger $\nu_{\rm m}$ [deg]"


def _slice_plot_axes(
    kick_slice: KickMapSlice,
    aout_axis: str,
    aout_rg_au: float,
) -> tuple[np.ndarray, np.ndarray, str, str, bool]:
    """Return display coordinates, labels, and whether y is logarithmic."""
    if kick_slice.plane == "aout-nu":
        y, ylabel, _ = aout_plot_values(kick_slice.y_values, aout_axis, aout_rg_au)
        return (
            np.degrees(kick_slice.x_values),
            y,
            _phase_label(kick_slice.phase_measure),
            ylabel,
            True,
        )
    if kick_slice.plane == "eout-aout":
        y, ylabel, _ = aout_plot_values(kick_slice.y_values, aout_axis, aout_rg_au)
        return (
            kick_slice.x_values,
            y,
            r"Pre-merger outer eccentricity $e_{\rm out}$",
            ylabel,
            True,
        )
    return (
        np.degrees(kick_slice.x_values),
        kick_slice.y_values,
        _phase_label(kick_slice.phase_measure),
        r"Pre-merger outer eccentricity $e_{\rm out}$",
        False,
    )


def _slice_title(kick_slice: KickMapSlice, secular_only: bool) -> str:
    mode = "Peters-applicable targets only" if secular_only else "all targets"
    if kick_slice.fixed_name == "e_out":
        fixed = rf"$e_{{\rm out}}={kick_slice.fixed_value:g}$"
    elif kick_slice.fixed_name == "nu_m_deg":
        symbol = "M" if kick_slice.phase_measure == PHASE_MEAN_ANOMALY else r"\nu"
        fixed = rf"${symbol}_{{\rm m}}={kick_slice.fixed_value:g}^\circ$"
    else:
        fixed = rf"$a_{{\rm out}}={kick_slice.fixed_value:g}\,\mathrm{{AU}}$"
    return f"{fixed}; {mode}"


# -------------------------------------------------------------------------------------------------
# Contour preparation and stability overlays
# -------------------------------------------------------------------------------------------------

def prepare_kick_contour_fields(
    Z: np.ndarray,
    smoothing_sigma: float = 1.5,
    edge_trim: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the velocity grid so shaded regions and contour lines look cleaner.

    ``Z`` is a 2D grid of minimum required recoil velocities in km/s. This helper smooths
    neighboring values to make plotted boundaries less jagged.

    ``smoothing_sigma`` controls the smoothing width in grid cells; zero disables smoothing.
    Smoothing uses log10(positive velocity). Nonfinite or negative values remain blank;
    exact zero-kick points are preserved. The recoil ceiling is applied by the caller,
    so smoothing can move the displayed threshold boundary relative to the raw grid.

    Returns
    -------
    Z_fill
        The smoothed velocity grid, keeping every originally valid location.
        Use this to draw filled regions, such as velocities below the ceiling.
    Z_contour
        The same grid with a thin border left blank around missing/invalid
        data and the array edges. This avoids messy contour-line fragments.
        ``edge_trim`` sets the border thickness in grid cells; zero keeps it.
    """
    # Select valid velocities, preserving exact zero-kick solutions
    Z = np.asarray(Z, dtype=float)
    finite = np.isfinite(Z) & (Z >= 0.0)
    positive = finite & (Z > 0.0)

    if not np.any(finite):
        empty = np.full_like(Z, np.nan, dtype=float)
        return empty, empty.copy()

    # Smooth the field for filled regions, leaving invalid cells blank
    Z_fill = np.full_like(Z, np.nan, dtype=float)

    if smoothing_sigma > 0.0 and np.any(positive):
        try:
            from scipy.ndimage import gaussian_filter

            logZ = np.zeros_like(Z, dtype=float)
            logZ[positive] = np.log10(Z[positive])

            weights = positive.astype(float)
            numerator = gaussian_filter(logZ*weights, sigma=float(smoothing_sigma), mode="nearest")
            denominator = gaussian_filter(weights, sigma=float(smoothing_sigma), mode="nearest")

            good = positive & (denominator > 1e-12)
            Z_fill[good] = 10.0 ** (numerator[good] / denominator[good])

        except Exception as exc:
            print(
                "WARNING: contour smoothing failed; using the unsmoothed field. "
                f"Reason: {exc}"
            )
            Z_fill[finite] = Z[finite]
    else:
        Z_fill[finite] = Z[finite]

    Z_fill[finite & ~positive] = 0.0

    # Trim boundary cells only for contour lines
    contour_valid = finite.copy()

    if edge_trim > 0:
        try:
            from scipy.ndimage import binary_erosion
            contour_valid = binary_erosion(contour_valid, iterations=int(edge_trim), border_value=0)
        except Exception as exc:
            print(
                "WARNING: contour-edge trimming failed; using the full finite "
                f"domain. Reason: {exc}"
            )
            contour_valid = finite

    # Return the full smoothed field and the trimmed contour field
    Z_contour = np.where(contour_valid, Z_fill, np.nan)
    return Z_fill, Z_contour


def _label_empty_map(ax):
    """Explain blank maps without implying that an unsampled region is excluded."""
    ax.text(0.5, 0.5, "No valid solutions on this grid", transform=ax.transAxes,
            ha="center", va="center", color="dimgray", fontsize=10, zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=4))


def _overlay_ma_invalid_region(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
    ma_valid: np.ndarray,
) -> list:
    """Draw the Mardling-Aarseth-invalid region for all three slice orientations."""
    unstable = ~ma_valid
    handles = []
    if not np.any(unstable):
        return handles

    ma_color = "tab:blue"
    hatch_set = ax.contourf(
        x_values,
        y_values,
        unstable.astype(float),
        levels=[0.5, 1.5],
        colors="none",
        hatches=["///"],
        alpha=0.75,
        zorder=6.5,
    )

    try:
        hatch_set.set_facecolor("none")
        hatch_set.set_edgecolor(ma_color)
        hatch_set.set_linewidth(0.0)
    except AttributeError:
        for collection in hatch_set.collections:
            collection.set_facecolor("none")
            collection.set_edgecolor(ma_color)
            collection.set_linewidth(0.0)
    handles.append(
        Patch(
            facecolor="none",
            edgecolor=ma_color,
            hatch="///",
            label="Mardling--Aarseth unstable region",
        )
    )
    if np.any(ma_valid):
        ax.contour(
            x_values,
            y_values,
            ma_valid.astype(float),
            levels=[0.5],
            colors=[ma_color],
            linestyles="--",
            linewidths=1.5,
            zorder=7,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=ma_color,
                linestyle="--",
                linewidth=1.5,
                label=r"Mardling--Aarseth boundary",
            )
        )
    return handles


# -------------------------------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------------------------------

def plot_fixed_epost_eout_map(
    vk_kms: np.ndarray,
    a_grid_au: np.ndarray,
    nu_grid_rad: np.ndarray,
    e_out: float,
    e_post_fixed: float,
    outdir: Path,
    vmax: float = 5000.0,
    e_ref_min: Optional[float] = None,
    e_ref_max: Optional[float] = None,
    f_ref_hz: float = 10.0,
    aout_axis: str = "au",
    aout_rg_au: float = 1.0,
    phase_measure: str = PHASE_MEAN_ANOMALY,
    show_title: bool = True,
    output_format: str = "pdf",
    vmin: float = 100.0,
) -> Path:
    """Plot the minimum-recoil-velocity map for fixed post-merger and pre-merger eccentricities."""
    NU_deg = np.degrees(nu_grid_rad)
    A_plot, y_label, _ = aout_plot_values(a_grid_au, aout_axis, aout_rg_au)

    finite = np.isfinite(vk_kms)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    display = np.maximum(vk_kms, np.finfo(float).tiny)
    pcm = ax.pcolormesh(NU_deg, A_plot, display, shading="auto", norm=norm, rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"Required kick $v_k$ [km/s]")
    if not np.any(finite):
        _label_empty_map(ax)

    ax.set_yscale("log")
    ax.set_xlim(NU_deg[0], NU_deg[-1])
    ax.set_ylim(A_plot[0], A_plot[-1])
    ax.set_xlabel(_phase_label(phase_measure))
    ax.set_ylabel(y_label)
    title = rf"Fixed $e_{{post}}={e_post_fixed:.3g}$, $e_{{out}}={e_out:.3g}$"
    filter_title = eccentricity_filter_title(e_ref_min, e_ref_max, f_ref_hz)
    if filter_title:
        title += "\n" + filter_title
    if show_title:
        ax.set_title(title)
    fig.tight_layout()

    outpath = outdir / (
        f"kick_fixed_epost_{format_e(e_post_fixed)}_eout_{format_e(e_out)}"
        f"{filter_suffix(e_ref_min, e_ref_max)}{axis_suffix(aout_axis)}"
        f"{phase_suffix(phase_measure)}.{output_format}"
    )
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_minimum_recoil_velocity_map(
    kick_slice: KickMapSlice,
    outdir: Path,
    secular_only: bool,
    recoil_ceiling_kms: float,
    vmax: float,
    e_ref_min: Optional[float],
    e_ref_max: Optional[float],
    f_ref_hz: float,
    aout_axis: str,
    aout_rg_au: float,
    allowed_alpha: float,
    contour_smoothing_sigma: float,
    contour_edge_trim: int,
    show_title: bool,
    output_format: str = "pdf",
    vmin: float = 100.0,
    fixed_epost: Optional[float] = None,
    cmap: str = "viridis",
    highlight_color: str = "crimson",
) -> Path:
    """Plot the minimum-required recoil-velocity map for a 2D slice."""
    x, y, xlabel, ylabel, y_log = _slice_plot_axes(kick_slice, aout_axis, aout_rg_au)
    Z = kick_slice.vk_min_kms
    finite = np.isfinite(Z)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    display = np.maximum(Z, np.finfo(float).tiny)
    pcm = ax.pcolormesh(x, y, display, shading="auto", norm=norm, cmap=cmap, rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"Minimum required recoil $v_{k,\min}$ [km/s]")
    if not np.any(finite):
        _label_empty_map(ax)

    Z_fill, Z_contour = prepare_kick_contour_fields(
        Z,
        smoothing_sigma=contour_smoothing_sigma,
        edge_trim=contour_edge_trim,
    )
    legend_handles = []
    ceiling = float(recoil_ceiling_kms)
    ceiling_color = highlight_color
    finite_fill = np.isfinite(Z_fill)
    if np.any(finite_fill) and np.nanmin(Z_fill[finite_fill]) < ceiling:
        lower = 0.0
        ax.contourf(
            x,
            y,
            Z_fill,
            levels=[lower, ceiling],
            colors=[ceiling_color],
            alpha=allowed_alpha,
            antialiased=True,
            zorder=3,
        )
        legend_handles.append(
            Patch(
                facecolor=ceiling_color,
                edgecolor=ceiling_color,
                alpha=allowed_alpha,
                label=rf"$v_{{k,\min}}<{ceiling:g}\ {{\rm km\,s^{{-1}}}}$",
            )
        )

    finite_contour = np.isfinite(Z_contour)
    if np.any(finite_contour):
        zmin = float(np.nanmin(Z_contour[finite_contour]))
        zmax = float(np.nanmax(Z_contour[finite_contour]))
        requested = np.unique(np.asarray([100.0, 300.0, 500.0, ceiling, 1000.0, 2000.0, 3000.0, 5000.0]))
        white = requested[
            (np.abs(requested - ceiling) > 1e-12)
            & (requested > zmin)
            & (requested < zmax)
        ]
        if white.size:
            contours = ax.contour(
                x, y, Z_contour, levels=white, colors="white", linewidths=0.9, zorder=5
            )
            ax.clabel(contours, fmt=lambda value: f"{value:g}", fontsize=7, colors="white")
        if zmin < ceiling < zmax:
            highlighted = ax.contour(
                x,
                y,
                Z_contour,
                levels=[ceiling],
                colors=[ceiling_color],
                linewidths=1.6,
                zorder=6,
            )
            ax.clabel(
                highlighted,
                fmt={ceiling: f"{ceiling:g}"},
                fontsize=7,
                colors=[ceiling_color],
            )

    legend_handles.extend(_overlay_ma_invalid_region(ax, x, y, kick_slice.ma_valid))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=1.0).set_zorder(10)
    if y_log:
        ax.set_yscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor("whitesmoke")
    if show_title:
        title = _slice_title(kick_slice, secular_only)
        if fixed_epost is not None:
            title += rf"; fixed $e_{{post}}={fixed_epost:.3g}$"
        reference_filter = eccentricity_filter_title(e_ref_min, e_ref_max, f_ref_hz)
        if reference_filter:
            title += "\n" + reference_filter
        ax.set_title(title)
    fig.tight_layout()

    prefix = "kick_min_map" if fixed_epost is None else f"kick_fixed_epost_{format_e(fixed_epost)}"
    outpath = outdir / (
        f"{prefix}_{_slice_file_stem(kick_slice)}"
        f"{filter_suffix(e_ref_min, e_ref_max)}{axis_suffix(aout_axis)}"
        f"{phase_suffix(kick_slice.phase_measure)}.{output_format}"
    )
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_minimum_recoil_velocity_eccentricity_map(
    kick_slice: KickMapSlice,
    field_name: str,
    outdir: Path,
    secular_only: bool,
    epost_grid_max: float,
    f_ref_hz: float,
    aout_axis: str,
    aout_rg_au: float,
    show_title: bool,
    e_ref_min: Optional[float] = None,
    e_ref_max: Optional[float] = None,
    output_format: str = "pdf",
    fixed_epost: Optional[float] = None,
    cmap: str = "viridis",
) -> Path:
    """Plot the kick-minimizing e_post or reference eccentricity for a 2D kick-map slice."""
    x, y, xlabel, ylabel, y_log = _slice_plot_axes(kick_slice, aout_axis, aout_rg_au)
    if field_name == "epost":
        Z = kick_slice.epost_star
        vmax = epost_grid_max
        cbar_label = r"Post-merger eccentricity giving $v_{k,\min}$"
    elif field_name == "eref":
        Z = kick_slice.eref_star
        vmax = 1.0
        cbar_label = rf"Eccentricity at ${f_ref_hz:g}\,\mathrm{{Hz}}$ for $v_{{k,\min}}$ target"
    else:
        raise ValueError("field_name must be 'epost' or 'eref'.")

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    pcm = ax.pcolormesh(x, y, Z, shading="auto", vmin=0.0, vmax=vmax, cmap=cmap, rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)
    if not np.any(np.isfinite(Z)):
        _label_empty_map(ax)
    handles = _overlay_ma_invalid_region(ax, x, y, kick_slice.ma_valid)
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8, framealpha=1.0).set_zorder(10)
    if y_log:
        ax.set_yscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_title:
        title = _slice_title(kick_slice, secular_only)
        if fixed_epost is not None:
            title += rf"; fixed $e_{{post}}={fixed_epost:.3g}$"
        reference_filter = eccentricity_filter_title(e_ref_min, e_ref_max, f_ref_hz)
        if reference_filter:
            title += "\n" + reference_filter
        ax.set_title(title)
    fig.tight_layout()

    fixed_suffix = "" if fixed_epost is None else f"_fixed_epost_{format_e(fixed_epost)}"
    outpath = outdir / (
        f"{field_name}_star_map{fixed_suffix}_{_slice_file_stem(kick_slice)}"
        f"{filter_suffix(e_ref_min, e_ref_max)}{axis_suffix(aout_axis)}"
        f"{phase_suffix(kick_slice.phase_measure)}.{output_format}"
    )
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


# -------------------------------------------------------------------------------------------------
# Slice workers and phase reuse
# -------------------------------------------------------------------------------------------------

def _compute_slice_task(task):
    """Compute either a general slice or its fixed-e_post maps, without plotting."""
    index, slice_kwargs, fixed_targets = task
    kick_slice = None
    fixed_maps = []
    for epost, targets in fixed_targets:
        fixed_slice = compute_min_kick_map_slice(**dict(slice_kwargs, targets=targets))
        fixed_maps.append((epost, fixed_slice))
        kick_slice = fixed_slice
    if kick_slice is None:
        kick_slice = compute_min_kick_map_slice(**slice_kwargs)
    return index, kick_slice, fixed_maps


def _reuse_fixed_phase_tasks(tasks):
    """Group equivalent fixed-phase slices; preserve each requested output label."""
    phase_tasks = [task for task in tasks if task[1]["plane"] == "eout-aout"]
    phases = np.radians([task[1]["fixed_value"] for task in phase_tasks])
    representatives, inverse = phase_reflection_indices(phases)
    representative_ids = {
        task[0]: phase_tasks[representatives[inverse[i]]][0]
        for i, task in enumerate(phase_tasks)
    }
    aliases = {}
    for index, kwargs, _ in tasks:
        representative = representative_ids.get(index, index)
        aliases.setdefault(representative, []).append((index, kwargs["fixed_value"]))
    return [task for task in tasks if task[0] in aliases], aliases


def _restore_fixed_phase_results(results, aliases):
    for index, kick_slice, fixed_maps in results:
        for output_index, fixed_value in aliases[index]:
            yield (output_index, replace(kick_slice, fixed_value=fixed_value),
                   [(epost, replace(s, fixed_value=fixed_value)) for epost, s in fixed_maps])


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Skip repeated slice orientations, keeping their requested order
    slices = list(dict.fromkeys(args.slices))
    fixed_values = {"aout-nu": args.e_out, "eout-aout": args.nu_m_deg, "eout-nu": args.a_out_au}
    try:
        fixed_values = {plane: unique_filename_values(fixed_values[plane], plane) for plane in slices}
        fixed_epost = unique_filename_values(args.fixed_epost, "fixed e_post")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    target_mode = "secular_only" if args.secular_only else "all_targets"
    outdir = Path(args.outdir) / target_mode
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Target mode: {target_mode}; writing only {outdir}")

    system = system_from_args(args)
    settings = settings_from_args(args)
    aout_rg_au = system.rg_post_au
    if args.aout_axis == "rg":
        print(f"Using GM/c^2 = {aout_rg_au:.6e} AU for the a_out axis.")

    reference_filter = eccentricity_filter_title(
        args.e_ref_min, args.e_ref_max, args.f_ref_hz
    )
    if reference_filter:
        print(f"Applying reference eccentricity filter: {reference_filter}")

    epost_grid_max = 1.0 - args.one_minus_epost_min
    candidates = None
    targets = None
    if not fixed_epost or args.write_candidate_csv:
        epost_grid = make_epost_grid(args.n_epost, one_minus_epost_min=args.one_minus_epost_min)
        candidates = compute_post_first_merger_candidates(
            epost_grid,
            mrem_msun=args.mrem,
            m3_msun=args.m3,
            merger_time_days=args.delay_days,
            f_ref_hz=args.f_ref_hz,
            min_peters_orbits=args.min_peters_orbits,
            min_peters_rp_rg=args.min_peters_rp_rg,
        )
        if not fixed_epost:
            # Apply the target-family cuts once for all slices
            targets = select_admitted_targets(candidates, settings)
            print(
                f"Selected {len(targets.epost)} of {len(candidates.epost)} "
                f"post-first-merger candidate orbits for this run."
            )

    a_grid_au = np.geomspace(args.aout_min_au, args.aout_max_au, args.n_aout)
    e_out_grid = np.linspace(args.eout_min, args.eout_max, args.n_eout)
    nu_grid_rad = np.radians(np.linspace(args.nu_min_deg, args.nu_max_deg, args.n_nu))
    inclination_rad = math.radians(args.inclination_deg)

    # Prepare exact-e_post targets once for all requested slice orientations
    fixed_targets = []
    if fixed_epost:
        for epost_fixed in fixed_epost:
            if not (0.0 <= epost_fixed <= epost_grid_max):
                print(f"  skipping fixed e_post={epost_fixed:g}: outside candidate grid")
                continue
            fixed_candidates = compute_post_first_merger_candidates(
                np.array([epost_fixed]), mrem_msun=args.mrem, m3_msun=args.m3,
                merger_time_days=args.delay_days, f_ref_hz=args.f_ref_hz,
                min_peters_orbits=args.min_peters_orbits,
                min_peters_rp_rg=args.min_peters_rp_rg,
            )
            admitted = select_admitted_targets(fixed_candidates, settings)
            fixed_targets.append((epost_fixed, admitted))
            print(f"Selected {len(admitted.epost)} of 1 target at fixed e_post={epost_fixed:g}.")

    # Each unique slice is one job, shared by all requested plot products
    common = dict(
        a_grid_au=a_grid_au, e_out_grid=e_out_grid, nu_grid_rad=nu_grid_rad,
        targets=targets, m1_inner_msun=args.m1_inner, m2_inner_msun=args.m2_inner,
        mrem_msun=args.mrem, m3_msun=args.m3, inclination_rad=inclination_rad,
        settings=settings,
    )
    tasks = []
    for plane in slices:
        for fixed_value in fixed_values[plane]:
            tasks.append((len(tasks), dict(common, plane=plane, fixed_value=float(fixed_value)),
                          fixed_targets))
    requested_count = len(tasks)
    tasks, aliases = _reuse_fixed_phase_tasks(tasks)
    print(f"Computing {len(tasks)} distinct slices for {requested_count} requested slices "
          f"using {min(args.workers, len(tasks))} worker process(es).")
    files_by_slice = {}
    results = completed_tasks(_compute_slice_task, tasks, args.workers, progress=print)
    for index, kick_slice, fixed_maps in _restore_fixed_phase_results(results, aliases):
        print(f"Computed {kick_slice.plane} slice at fixed value {kick_slice.fixed_value:g} "
              f"({len(files_by_slice) + 1}/{requested_count}); writing plots ...")
        slice_files = []
        kick_plots = fixed_maps if fixed_epost else [(None, kick_slice)]
        for epost_fixed, plotted_slice in kick_plots:
            slice_files.append(
                plot_minimum_recoil_velocity_map(
                    plotted_slice,
                    fixed_epost=epost_fixed,
                    outdir=outdir,
                    secular_only=args.secular_only,
                    recoil_ceiling_kms=args.recoil_ceiling_kms,
                    vmax=args.vmax,
                    vmin=args.vmin,
                    e_ref_min=args.e_ref_min,
                    e_ref_max=args.e_ref_max,
                    f_ref_hz=args.f_ref_hz,
                    aout_axis=args.aout_axis,
                    aout_rg_au=aout_rg_au,
                    allowed_alpha=args.recoil_region_alpha,
                    cmap=args.cmap,
                    highlight_color=args.highlight_color,
                    contour_smoothing_sigma=args.contour_smoothing_sigma,
                    contour_edge_trim=args.contour_edge_trim,
                    show_title=not args.no_title,
                    output_format=args.output_format,
                )
            )
            for product, enabled in (("epost", args.output_minimizing_epost),
                                     ("eref", args.output_minimizing_eref)):
                if enabled:
                    slice_files.append(
                        plot_minimum_recoil_velocity_eccentricity_map(
                            plotted_slice,
                            fixed_epost=epost_fixed,
                            field_name=product,
                            cmap=args.cmap,
                            e_ref_min=args.e_ref_min,
                            e_ref_max=args.e_ref_max,
                            outdir=outdir,
                            secular_only=args.secular_only,
                            epost_grid_max=epost_grid_max,
                            f_ref_hz=args.f_ref_hz,
                            aout_axis=args.aout_axis,
                            aout_rg_au=aout_rg_au,
                            show_title=not args.no_title,
                            output_format=args.output_format,
                        )
                    )

        files_by_slice[index] = slice_files

    # Preserve requested ordering in the final file list, regardless of completion order
    made_files = [path for index in range(requested_count) for path in files_by_slice[index]]

    if args.write_candidate_csv:
        candidate_csv = outdir / "post_merger_peters_candidates.csv"
        write_candidate_table(candidate_csv, candidates, settings, args.f_ref_hz)
        made_files.append(candidate_csv)

    print("\nCreated files:")
    for path in made_files:
        print(f"  {path}")

if __name__ == "__main__":
    main()
