#!/usr/bin/env python3
"""Command-line interfaces for the triple-check scripts.

Shared arguments and their common validation are defined once. Each script adds its own parser
defaults and specialized validation below.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from triple_core import (
    AnalysisSettings,
    TripleSystem,
    PHASE_MEAN_ANOMALY,
    PHASE_MEASURES,
    PHASE_TRUE_ANOMALY,
    DEFAULT_STABILITY_DAYS_BEFORE,
    DEFAULT_R_SEC_THRESHOLD,
    SLICE_CHOICES,
)


# -------------------------------------------------------------------------------------------------
# Shared arguments
# -------------------------------------------------------------------------------------------------

def add_map_color_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared map palette, recoil-compatible region color, and color limits."""
    parser.add_argument("--cmap", type=str, default="viridis", help="Map colormap name")
    parser.add_argument("--highlight-color", type=str, default="crimson", help="Color of the recoil-compatible region")
    parser.add_argument("--vmin", type=float, default=100.0, help="Minimum colorbar value [km/s]")
    parser.add_argument("--vmax", type=float, default=5000.0, help="Maximum colorbar value [km/s]")


def add_worker_argument(parser: argparse.ArgumentParser, *, default: int = 1) -> None:
    """Worker count for independent slices, grid blocks, or posterior samples."""
    parser.add_argument("--workers", type=int, default=default, help="Worker processes for parallelizing work. Use 1 for serial execution")


def add_system_arguments(parser: argparse.ArgumentParser) -> None:
    """Masses, delay, and the geometry entering the stability limit."""
    group = parser.add_argument_group("triple system")
    group.add_argument("--m1-inner", type=float, default=67.0, help="First inner binary mass [Msun]")
    group.add_argument("--m2-inner", type=float, default=47.3, help="Second inner binary mass [Msun]")
    group.add_argument("--mrem", type=float, default=109.7, help="First-merger remnant mass [Msun]")
    group.add_argument("--m3", type=float, default=152.4, help="Third BH mass [Msun]")
    group.add_argument("--delay-days", type=float, default=6.839, help="Target delay to second merger [days]")
    group.add_argument("--inclination-deg", type=float, default=180.0, help="Mutual inclination for MA line [deg]; 180 deg gives the least restrictive baseline")


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    """The admitted family of post-first-merger Peters targets."""
    group = parser.add_argument_group("post-merger target family")
    group.add_argument("--one-minus-epost-min", type=float, default=1e-12, help="Closest approach of the candidate grid to e_post=1: e_post,max = 1 - this value")
    group.add_argument("--n-epost", type=int, default=2500, help="Exact number of post-first-merger candidate eccentricities to sample")
    group.add_argument("--min-peters-orbits", type=float, default=10.0, help="Minimum merger delay in units of the initial post-kick orbital period")
    group.add_argument("--min-peters-rp-rg", type=float, default=20.0, help="Minimum initial pericenter in GM/c^2 required by the Peters applicability conditions")
    group.add_argument("--secular-only", dest="secular_only", action="store_true", help="Use only targets satisfying the Peters applicability conditions")
    group.add_argument("--f-ref-hz", dest="f_ref_hz", type=float, default=10.0, help="Reference GW peak frequency [Hz]")
    group.add_argument("--e-ref-min", dest="e_ref_min", type=float, default=0.0, help="Discard candidates whose Peters-evolved eccentricity at the reference frequency is below this value")
    group.add_argument("--e-ref-max", dest="e_ref_max", type=float, default=0.02, help="Discard candidates whose Peters-evolved eccentricity at the reference frequency exceeds this value")
    group.add_argument("--e-ref-range", dest="e_ref_range", type=float, default=None, help="Allowed Peters-evolved eccentricity range at the reference frequency", nargs=2, metavar=("EMIN", "EMAX"))


def add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    """The progenitor grid in (a_out, e_out, nu_m)."""
    group = parser.add_argument_group("progenitor grid")
    group.add_argument("--aout-min-au", type=float, default=1e-3, help="Minimum a_out for grid [AU]")
    group.add_argument("--aout-max-au", type=float, default=10.0, help="Maximum a_out for grid [AU]")
    group.add_argument("--aout-axis", choices=["au", "rg"], default="au", help="Units for the plotted a_out axis: AU or rg = G(mrem+m3)/c^2. Grid limits remain in AU")
    group.add_argument("--n-aout", type=int, default=300, help="Number of a_out grid points")
    group.add_argument("--eout-min", type=float, default=0.0, help="Minimum e_out for slices that scan eccentricity")
    group.add_argument("--eout-max", type=float, default=0.999, help="Maximum e_out for slices that scan eccentricity")
    group.add_argument("--n-eout", type=int, default=100, help="Number of e_out grid points")
    group.add_argument("--nu-min-deg", type=float, default=-180.0, help="Minimum nu_m [deg]")
    group.add_argument("--nu-max-deg", type=float, default=180.0, help="Maximum nu_m [deg]")
    group.add_argument("--n-nu", type=int, default=360, help="Number of phase grid points")
    group.add_argument("--phase-measure", choices=list(PHASE_MEASURES), default=PHASE_TRUE_ANOMALY, help="Meaning of the phase axis ('true_anomaly' or 'mean_anomaly')", )


def add_stability_arguments(parser: argparse.ArgumentParser) -> None:
    """Mardling-Aarseth masking and the inner separation it is evaluated at."""
    group = parser.add_argument_group("stability")
    group.add_argument("--stability-days-before", type=float, default=DEFAULT_STABILITY_DAYS_BEFORE, help="Epoch used to calculate the inner separation for the MA criterion")
    group.add_argument("--alpha-in", type=float, default=None, help="Inner-binary semi-major axis in units G(m1_inner+m2_inner)/c^2 for the Mardling-Aarseth criterion")
    group.add_argument("--inner-separation-au", type=float, default=None, help="Inner-binary semi-major axis for the Mardling-Aarseth criterion")
    group.add_argument("--use-ma-mask", dest="use_ma_mask", action="store_true", default=True, help="Mask points below the Mardling-Aarseth stability limit")
    group.add_argument("--no-ma-mask", dest="use_ma_mask", action="store_false", help="Disable the Mardling-Aarseth physical-domain mask")


def add_recoil_ceiling_argument(parser: argparse.ArgumentParser) -> None:
    """The recoil ceiling used for highlighting and for compatibility."""
    parser.add_argument("--recoil-ceiling-kms", dest="recoil_ceiling_kms", type=float, default=500, help="Recoil ceiling highlighted in maps and used to determine recoil-compatible progenitors")


def validate_color_limits(vmin, vmax, *, minimum_name="vmin", maximum_name="vmax") -> None:
    """Validate finite, ordered color limits; None selects an automatic maximum."""
    if not math.isfinite(vmin):
        raise ValueError(f"{minimum_name} must be finite.")
    if vmax is not None:
        if not math.isfinite(vmax):
            raise ValueError(f"{maximum_name} must be finite.")
        if vmax <= vmin:
            raise ValueError(f"{maximum_name} must exceed {minimum_name} ({vmin:g} km/s).")


def validate_shared_args(args: argparse.Namespace) -> None:
    """Resolve and validate common physical inputs and grid settings."""
    resolve_e_ref_range(args)
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    for name in ("m1_inner", "m2_inner", "mrem", "m3", "delay_days", "f_ref_hz",
                 "stability_days_before", "aout_min_au", "aout_max_au"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name.replace('_', '-')} must be finite and positive.")
    if args.mrem > args.m1_inner + args.m2_inner:
        raise ValueError("mrem cannot exceed the pre-merger inner mass.")
    if not (math.isfinite(args.inclination_deg) and 0.0 <= args.inclination_deg <= 180.0):
        raise ValueError("inclination-deg must be between 0 and 180.")
    if args.inner_separation_au is not None and not (math.isfinite(args.inner_separation_au)
        and args.inner_separation_au > 0.0):
        raise ValueError("inner-separation-au must be finite and positive.")
    alpha_in = getattr(args, "alpha_in", None)
    if alpha_in is not None and args.inner_separation_au is not None:
        raise ValueError("Specify only one of --alpha-in and --inner-separation-au.")
    if alpha_in is not None and not (math.isfinite(alpha_in) and alpha_in > 0.0):
        raise ValueError("alpha-in must be finite and positive.")
    for name in ("recoil_ceiling_kms", "min_peters_orbits", "min_peters_rp_rg"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative.")
    if not (0.0 < args.one_minus_epost_min < 1.0 and 1.0 - args.one_minus_epost_min < 1.0):
        raise ValueError("one-minus-epost-min must produce a representable eccentricity below 1.")
    if min(args.n_aout, args.n_eout, args.n_nu) < 2 or args.n_epost < 3:
        raise ValueError("Progenitor axes need at least 2 points; n-epost needs at least 3.")
    if args.aout_max_au <= args.aout_min_au:
        raise ValueError("Require 0 < aout-min-au < aout-max-au.")
    if not (0.0 <= args.eout_min < args.eout_max < 1.0):
        raise ValueError("Require 0 <= eout-min < eout-max < 1.")
    if not (math.isfinite(args.nu_min_deg) and math.isfinite(args.nu_max_deg)
            and 0.0 < args.nu_max_deg - args.nu_min_deg <= 360.0):
        raise ValueError("The phase interval must span more than 0 and at most 360 degrees.")
    for name in ("e_ref_min", "e_ref_max"):
        value = getattr(args, name)
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"{name.replace('_', '-')} must lie in [0, 1].")
    if args.e_ref_min is not None and args.e_ref_max is not None and args.e_ref_min > args.e_ref_max:
        raise ValueError("e-ref-min must not exceed e-ref-max.")


# -------------------------------------------------------------------------------------------------
# 2D kick maps
# -------------------------------------------------------------------------------------------------

def build_kick_maps_2d_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Axis-aligned 2D slices through the full 3D kick map.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add shared arguments
    add_system_arguments(parser)
    add_grid_arguments(parser)
    add_target_arguments(parser)
    add_stability_arguments(parser)
    add_map_color_arguments(parser)
    add_recoil_ceiling_argument(parser)
    add_worker_argument(parser)

    parser.add_argument("--slices", "--slice", dest="slices", nargs="+", choices=SLICE_CHOICES, default=["aout-nu"], help="Slice plane(s), each name lists the two scanned axes; the third is held fixed")
    parser.add_argument("--e-out", type=float, nargs="+", default=[0.0], help="Fixed e_out values for aout-nu slices")
    parser.add_argument("--nu-m-deg", type=float, nargs="+", default=[0.0], help="Fixed orbital phases [deg] for eout-aout slices; interpreted using --phase-measure")
    parser.add_argument("--a-out-au", type=float, nargs="+", default=[0.1], help="Fixed a_out values [AU] for eout-nu slices")

    parser.add_argument("--recoil-region-alpha", dest="recoil_region_alpha", type=float, default=0.25, help="Alpha of the shaded region below the recoil ceiling")
    parser.add_argument("--contour-smoothing-sigma", type=float, default=1.5, help="Gaussian smoothing sigma in grid cells, applied only to plotting contours in log10(v_k). Set to 0 to disable")
    parser.add_argument("--contour-edge-trim", type=int, default=2, help="Number of grid cells trimmed from invalid-domain boundaries only for line contours, to suppress noisy/jagged edge fragments")

    parser.add_argument("--fixed-epost", type=float, nargs="+", default=[], help="Output maps at these fixed e_post values")
    parser.add_argument("--output-minimizing-epost", action="store_true", help="Also plot the post-merger eccentricity giving the minimum recoil")
    parser.add_argument("--output-minimizing-eref", action="store_true", help="Also plot the reference-frequency eccentricity of the minimum-recoil target")
    parser.add_argument("--no-title", action="store_true", help="Suppress plot titles")
    parser.add_argument("--write-candidate-csv", action="store_true", help="Write the diagnostic post_merger_peters_candidates.csv table, including rejected candidates")
    parser.add_argument("--outdir", type=str, default="kick_grid_maps", help="Output directory")
    parser.add_argument("--output-format", choices=("pdf", "png"), default="png", help="Format for all 2D maps")
    return parser


def validate_plot_colors(cmap: str, highlight_color: str) -> None:
    """Reject invalid colormaps and highlight colors supplied on the command line."""
    from matplotlib import colormaps
    from matplotlib.colors import is_color_like
    if cmap not in colormaps:
        raise ValueError(f"Unknown colormap for --cmap: {cmap!r}.")
    if not is_color_like(highlight_color):
        raise ValueError(f"Invalid color for --highlight-color: {highlight_color!r}.")


def validate_kick_maps_2d_args(args: argparse.Namespace) -> None:
    """Validate shared inputs, then 2D slice and plotting options."""
    validate_shared_args(args)
    validate_color_limits(args.vmin, args.vmax)
    validate_plot_colors(args.cmap, args.highlight_color)

    if args.vmin <= 0.0:
        raise ValueError("vmin must be positive for the logarithmic 2D color scale.")
    if not (0.0 <= args.recoil_region_alpha <= 1.0):
        raise ValueError("recoil-region-alpha must lie in [0, 1].")
    if not (math.isfinite(args.contour_smoothing_sigma) and args.contour_smoothing_sigma >= 0.0
            and args.contour_edge_trim >= 0):
        raise ValueError("Contour smoothing and edge trimming must be finite and non-negative.")
    if any(not (0.0 <= value < 1.0) for value in args.e_out):
        raise ValueError("Every fixed e-out value must satisfy 0 <= value < 1.")
    if any(not math.isfinite(value) for value in args.nu_m_deg):
        raise ValueError("Every fixed nu-m-deg value must be finite.")
    if any(not (math.isfinite(value) and value > 0.0) for value in args.a_out_au):
        raise ValueError("Every fixed a-out-au value must be finite and positive.")
    if any(value > 1.0 - args.one_minus_epost_min for value in args.fixed_epost):
        raise ValueError("fixed-epost values must not exceed the candidate eccentricity limit.")
    if any(not (0.0 <= value < 1.0) for value in args.fixed_epost):
        raise ValueError("Every fixed-epost value must satisfy 0 <= value < 1.")

    # Conservative warning threshold, not a guaranteed numerical accuracy bound
    requested_eout = list(args.e_out) if "aout-nu" in args.slices else []
    if any(plane in args.slices for plane in ("eout-aout", "eout-nu")):
        requested_eout.append(args.eout_max)
    if any(value >= 0.999999 for value in requested_eout):
        print("WARNING: 2D maps with e_out >= 0.999999 are not validated for numerical accuracy. ", file=sys.stderr)


def parse_kick_maps_2d_args(argv=None):
    """Parse, resolve, and validate arguments for this script."""
    parser = build_kick_maps_2d_parser()
    args = parser.parse_args(argv)
    try:
        validate_kick_maps_2d_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


# -------------------------------------------------------------------------------------------------
# 3D kick map
# -------------------------------------------------------------------------------------------------

def build_kick_map_3d_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyVista 3D visualization of the minimum-recoil cube", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Load shared command-line arguments
    add_system_arguments(parser)
    add_grid_arguments(parser)
    add_target_arguments(parser)
    add_stability_arguments(parser)
    add_recoil_ceiling_argument(parser)
    add_map_color_arguments(parser)
    add_worker_argument(parser, default=4)

    parser.add_argument("--eout-spacing", choices=("refined", "linear"), default="refined", help="Refined keeps the n-eout base planes and adds samples near e=1; linear uses exactly n-eout planes")

    # Visualization
    parser.add_argument("--no-volume", dest="show_volume", action="store_false", default=True, help="Hide the velocity-colored volume")

    parser.add_argument("--no-highlight", dest="show_highlight", action="store_false", default=True, help="Hide the highlighted recoil-compatible region (shown by default)")
    parser.add_argument("--highlight-opacity", type=float, default=0.30, help="Opacity of the highlighted low-kick region")
    parser.add_argument("--highlight-specular", type=float, default=0.10, help="Specular highlight strength for the highlighted region")

    parser.add_argument("--show-slice", action="store_true")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z", help="x: fixed phase; y: fixed log10 a_out; z: fixed e_out")
    parser.add_argument("--slice-value", type=float, default=0.6, help="Slice value in physical coordinates: degrees for x, log10 a_out in --aout-axis units for y, and e_out for z")
    parser.add_argument("--slice-opacity", type=float, default=0.8)
    parser.add_argument("--slice-background-color", type=str, default="lightgray", help="Color used for invalid/non-colored regions of the slice")
    parser.add_argument("--slice-background-opacity", type=float, default=0.4, help="Opacity of the gray background in invalid slice regions")
    parser.add_argument("--slice-outline-color", type=str, default="gray", help="Color of the thick square outline drawn around the slice")
    parser.add_argument("--slice-outline-width", type=float, default=5.0, help="Line width of the slice outline")
    parser.add_argument("--slice-ambient", type=float, default=1.0, help="Ambient lighting strength for the slice. Use 1 for flat, vivid colors")
    parser.add_argument("--slice-diffuse", type=float, default=0.5, help="Diffuse lighting strength for the slice. Use 0 to avoid dark shading")
    parser.add_argument("--slice-specular", type=float, default=0.5, help="Specular lighting strength for the slice")

    parser.add_argument("--axis-label-font-size", type=int, default=22, help="Font size of numerical axis labels")
    parser.add_argument("--axis-title-font-size", type=int, default=24, help="Requested font size of axis titles")
    parser.add_argument("--axis-nlabels", type=int, default=5, help="Number of numerical labels shown along each axis")

    parser.add_argument("--background", type=str, default="white")
    parser.add_argument("--outline-color", type=str, default="black")
    parser.add_argument("--outline-width", type=float, default=2.0)
    parser.add_argument("--text-color", type=str, default="black")
    parser.add_argument("--camera-zoom", type=float, default=1.15)

    # Output / interaction
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Keep the interactive window open after saving an image")
    parser.add_argument("--output", type=str, default=None, help="Output image path; --output-format overrides its extension")
    parser.add_argument("--output-format", choices=("pdf", "png"), default=None, help="Saved image format; inferred from the output filename when omitted. Without a path, saves kick_map_3d.FORMAT")

    return parser


def validate_kick_map_3d_args(args):
    """Validate shared inputs, then 3D rendering and slice options."""
    validate_shared_args(args)
    validate_color_limits(args.vmin, args.vmax)

    for name in ("camera_zoom", "outline_width", "highlight_specular",
                 "slice_ambient", "slice_diffuse", "slice_specular", "highlight_opacity",
                 "slice_opacity", "slice_background_opacity", "slice_outline_width", "slice_value"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name.replace('_', '-')} must be finite.")
    if args.camera_zoom <= 0.0:
        raise ValueError("camera-zoom must be positive.")
    if args.outline_width < 0.0:
        raise ValueError("outline-width must be non-negative.")
    for name in ("highlight_specular", "slice_ambient", "slice_diffuse", "slice_specular"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must lie in [0, 1].")
    if not (0.0 <= args.highlight_opacity <= 1.0):
        raise ValueError("highlight-opacity must lie in [0, 1].")
    if not (0.0 <= args.slice_opacity <= 1.0):
        raise ValueError("slice-opacity must lie in [0, 1].")
    if not (0.0 <= args.slice_background_opacity <= 1.0):
        raise ValueError("slice-background-opacity must lie in [0, 1].")
    if args.axis_label_font_size <= 0 or args.axis_title_font_size <= 0:
        raise ValueError("Axis font sizes must be positive.")
    if args.axis_nlabels < 2:
        raise ValueError("axis-nlabels must be at least 2.")
    if args.slice_outline_width < 0.0:
        raise ValueError("slice-outline-width must be non-negative.")
    if args.show_slice:
        if args.slice_axis == "x":
            limits = (args.nu_min_deg, args.nu_max_deg)
        elif args.slice_axis == "y":
            limits = (math.log10(args.aout_min_au), math.log10(args.aout_max_au))
            if args.aout_axis == "rg":
                offset = math.log10(system_from_args(args).rg_post_au)
                limits = tuple(value - offset for value in limits)
        else:
            limits = (args.eout_min, args.eout_max)
        if not limits[0] <= args.slice_value <= limits[1]:
            raise ValueError(f"slice-value must lie in [{limits[0]:g}, {limits[1]:g}].")


def parse_kick_map_3d_args(argv=None):
    """Parse, resolve, and validate arguments for this script."""
    parser = build_kick_map_3d_parser()
    args = parser.parse_args(argv)
    try:
        validate_kick_map_3d_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


# -------------------------------------------------------------------------------------------------
# System summary
# -------------------------------------------------------------------------------------------------

def add_posterior_arguments(parser):
    group = parser.add_argument_group('posterior propagation (detector frame)')
    group.add_argument('--resume', action='store_true', help='Resume completed posterior draws from --json-output; analysis settings and inputs must match')
    group.add_argument('--posterior-first', type=Path, help='Public posterior file for the earlier event')
    group.add_argument('--posterior-second', type=Path, help='Public posterior file for the later event')
    group.add_argument('--first-dataset', help='Exact sample-table path; mandatory if several analyses exist')
    group.add_argument('--second-dataset', help='Exact sample-table path; mandatory if several analyses exist')
    group.add_argument('--remnant-component', choices=('primary', 'secondary'), help='Which component of the later event is the earlier remnant; run assignments separately')
    group.add_argument('--posterior-samples', type=int, default=128, help='Number of draws propagated; duplicate systems are evaluated once with multiplicity weight')
    group.add_argument('--posterior-seed', type=int, default=12345)
    group.add_argument('--posterior-proposals', type=int, default=50000, help='Independent event-pair proposals before kernel conditioning/resampling')
    group.add_argument('--posterior-delay-days', type=float, help='Observed delay; otherwise obtained from geocent_time samples. Never defaults to --delay-days')
    group.add_argument('--pairing-mode', choices=('prior-corrected', 'overlap'), default='prior-corrected', help='Prior-corrected divides by the second-event marginal PE prior; overlap is only a posterior-overlap sensitivity study')
    group.add_argument('--match-parameters', choices=('mass-distance', 'mass'), default='mass-distance', help='Shared detector remnant mass and optionally log luminosity distance')
    group.add_argument('--match-mass-width', type=float, help='Gaussian matching bandwidth [detector Msun]; default pooled standard deviation times Scott factor')
    group.add_argument('--match-log-distance-width', type=float, help='Bandwidth in natural log(distance/Mpc); default pooled standard deviation times Scott factor')
    group.add_argument('--second-prior', type=Path, help='Second-event PE prior samples. Default: embedded priors in the second posterior file')
    group.add_argument('--second-prior-dataset', help='Exact prior sample-table path')
    group.add_argument('--posterior-plot-projection-bins', type=int, default=100, help='Common bins for posterior-weighted projected support')

def is_posterior_mode(args):
    return bool(args.posterior_first or args.posterior_second)

def validate_posterior_arguments(args):
    if not is_posterior_mode(args):
        if args.resume:
            raise ValueError('--resume requires posterior mode.')
        return
    if not (args.posterior_first and args.posterior_second and args.remnant_component):
        raise ValueError('Paired analysis requires --posterior-first, --posterior-second and --remnant-component.')
    if args.posterior_samples < 1 or args.posterior_proposals < args.posterior_samples or args.posterior_seed < 0:
        raise ValueError('Require 1 <= posterior-samples <= posterior-proposals and a nonnegative seed.')
    if args.posterior_plot_projection_bins < 2:
        raise ValueError('Require at least 2 projection bins.')
    for name in ('posterior_delay_days', 'match_mass_width', 'match_log_distance_width'):
        x = getattr(args, name)
        if x is not None and (not math.isfinite(x) or x <= 0):
            raise ValueError(f'{name} must be finite and positive.')
    if not args.json_output:
        raise ValueError('Posterior runs require --json-output to retain samples, weights, and numerical diagnostics.')
    inputs = {p.resolve() for p in (args.posterior_first, args.posterior_second, args.second_prior) if p}
    outputs = [p.resolve() for p in (args.json_output, args.json_output.with_suffix('.png')) if p]
    if inputs.intersection(outputs) or len(outputs) != len(set(outputs)):
        raise ValueError('Output paths must be distinct and cannot overwrite input posterior/prior files.')


def build_system_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=('Omitted reference-eccentricity bounds resolve to [0, 0.02] for a single system '
                'and [0, 1] in posterior mode. --e-ref-range overrides the separate bounds.'),
    )

    # Load external command-line arguments
    add_grid_arguments(parser)
    add_system_arguments(parser)
    add_target_arguments(parser)
    add_stability_arguments(parser)
    add_recoil_ceiling_argument(parser)
    add_posterior_arguments(parser)
    add_worker_argument(parser)

    # General settings
    parser.set_defaults(n_eout=None, n_aout=None, n_nu=None, n_epost=None, phase_measure=PHASE_MEAN_ANOMALY)
    parser.set_defaults(e_ref_min=None, e_ref_max=None)
    parser.add_argument('--fixed-resolution', action='store_true', help='Evaluate --n-eout/--n-aout/--n-nu/--n-epost once')
    parser.add_argument('--r-sec-threshold', type=float, default=DEFAULT_R_SEC_THRESHOLD)
    parser.add_argument('--json-output', type=Path, help='Also write the complete report and refinement history as JSON')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress on stderr')

    # Projection settings
    parser.add_argument('--projection-min-transverse-cells', type=int, default=2, help='Minimum compatible-cell count per projected bin')
    parser.add_argument('--projection-min-transverse-fraction', type=float, default=1e-4, help='Minimum transverse occupancy defining support endpoints')
    parser.add_argument('--projection-support-sigma', type=float, default=2.0, help='Uncertainty-band width around the support threshold [standard deviations]; 0 uses a hard threshold')

    # Edge tracking settings
    parser.add_argument('--edge-tracking-scan', type=int, default=512, help='Coarse samples per column before bisection; increase for scan_limited results')
    parser.add_argument('--edge-tracking-sweeps', type=int, default=2, help='Column sweeps per evaluation; later sweeps focus on the located compatible interval')
    parser.add_argument('--edge-tracking-growth', type=float, default=2.0, help='Column-count growth factor per targeted round')

    # Gap validation settings
    parser.add_argument('--validate-gaps', action=argparse.BooleanOptionalAction, default=True, help='Validate internal support gaps on all three axes with targeted edge tracking')
    parser.add_argument('--gap-validation-single-round-when-empty', action=argparse.BooleanOptionalAction, default=True, help='Stop a gap check after one round if every column scan is empty')
    parser.add_argument('--gap-validation-max-points', type=int, default=300_000_000, help='Initial-scan point budget per resolution, shared across axes and target families')
    parser.add_argument('--gap-validation-levels', type=int, default=3, help='Maximum gap-check rounds; normally requires two agreeing rounds')
    parser.add_argument('--gap-validation-probes', type=int, default=3, help='Uniform interior probe positions per gap bin (minimum 3)')

    # Endpoint refinement settings
    parser.add_argument('--refine-endpoints', action=argparse.BooleanOptionalAction, default=True, help='Refine endpoints and gaps on both comparison grids after bulk diagnostics stabilize')
    parser.add_argument('--endpoint-refinement-max-points', type=int, default=1_000_000_000, help='Local endpoint/gap initial-scan point budget per resolution')
    parser.add_argument('--endpoint-refinement-max-positions', type=int, default=12, help='Target position limit per local window')
    parser.add_argument('--endpoint-refinement-levels', type=int, default=4, help='Maximum edge-tracking rounds per local window (minimum 2); boundary bisection reuses these rounds')

    # General refinement settings
    parser.add_argument('--refinement-factor', type=float, default=None, help='Use geometric refinement of all four grid dimensions (factor 1.5 when only a starting grid is supplied)')
    parser.add_argument('--max-refinements', type=int, default=5, help='Maximum additional grid evaluations after the initial evaluation')
    parser.add_argument('--max-grid-points', type=int, default=300_000_000, help='Progenitor-point limit per adaptive evaluation')
    parser.add_argument('--max-epost', type=int, default=20000, help='Target-grid limit per adaptive evaluation')

    # Convergence criteria
    parser.add_argument('--convergence-r-sec-atol', type=float, default=0.03, help='Absolute tolerance for R_sec')
    parser.add_argument('--convergence-coverage-atol', type=float, default=0.03, help='Absolute tolerance for secular projection support coverage')
    parser.add_argument('--convergence-log-volume-atol', type=float, default=0.03, help='Tolerance on changes in log10 of each compatible volume fraction (dex)')
    parser.add_argument('--convergence-endpoint-eout-atol', type=float, default=0.03, help='Absolute endpoint tolerance in e_out for both target families')
    parser.add_argument('--convergence-endpoint-loga-atol', type=float, default=0.03 , help='Endpoint tolerance in log10(a_out) [dex]')
    parser.add_argument('--convergence-endpoint-phase-atol-deg', type=float, default=1.0, help='Phase endpoint tolerance [degrees]; circular matching for full-orbit domains')
    parser.add_argument('--convergence-support', choices=('bulk', 'envelope', 'intervals'), default='envelope', help='Check only bulk quantities (bulk), outer endpoints (envelope), or all support intervals and gaps (intervals)')
    parser.add_argument('--convergence-check-class', action=argparse.BooleanOptionalAction, default=False, help='Also require an unchanged secular constraint class')
    parser.add_argument('--convergence-streak', type=int, default=1, help='Required consecutive successful refinement comparisons')

    return parser


def validate_system_summary_args(args):
    """Validate shared inputs, then summary projection and convergence options."""
    # Resolve omitted bounds according to the analysis mode before validating the interval
    if args.e_ref_min is None:
        args.e_ref_min = 0.0
    if args.e_ref_max is None:
        args.e_ref_max = 1.0 if is_posterior_mode(args) else 0.02
    grid_names = ('n_eout', 'n_aout', 'n_nu', 'n_epost')
    if not hasattr(args, 'resolution_sequence'):
        explicit = args.refinement_factor is not None or any(
            getattr(args, name) is not None for name in grid_names)
        args.resolution_sequence = 'geometric' if explicit else 'sequence'
    for name, default in zip(grid_names, (75, 270, 270, 3750)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.refinement_factor is None:
        args.refinement_factor = 1.5
    validate_shared_args(args)
    if args.convergence_support == 'bulk':
        args.validate_gaps = False
        args.refine_endpoints = False
        args.convergence_check_class = False
    if args.gap_validation_max_points < 1 or args.gap_validation_levels < 2 or args.gap_validation_probes < 3:
        raise ValueError('Gap validation requires a positive point budget, at least 2 levels and at least 3 probes per bin.')
    if args.endpoint_refinement_max_points < 1 or args.endpoint_refinement_levels < 2:
        raise ValueError('Endpoint refinement requires a positive point budget and at least 2 levels.')
    if not (0 <= args.r_sec_threshold <= 1):
        raise ValueError('r-sec-threshold must lie in [0, 1].')
    if not (args.projection_min_transverse_cells >= 1):
        raise ValueError('projection-min-transverse-cells must be >= 1.')
    if not (0 <= args.projection_min_transverse_fraction <= 1):
        raise ValueError('projection-min-transverse-fraction must lie in [0, 1].')
    if not (math.isfinite(args.projection_support_sigma) and args.projection_support_sigma >= 0):
        raise ValueError('projection-support-sigma must be finite and nonnegative.')
    if args.edge_tracking_scan < 8 or not (math.isfinite(args.edge_tracking_growth) and args.edge_tracking_growth > 1):
        raise ValueError('Edge tracking needs at least 8 scan samples per column and a growth factor > 1.')
    if args.edge_tracking_sweeps < 1:
        raise ValueError('edge-tracking-sweeps must be at least 1.')
    if args.endpoint_refinement_max_positions < 3:
        raise ValueError('endpoint-refinement-max-positions must be at least 3.')
    for name in ('convergence_r_sec_atol', 'convergence_coverage_atol', 'convergence_log_volume_atol',
                 'convergence_endpoint_eout_atol', 'convergence_endpoint_loga_atol', 'convergence_endpoint_phase_atol_deg'):
        if not (math.isfinite(getattr(args, name)) and getattr(args, name) >= 0):
            raise ValueError(f'{name} must be finite and nonnegative.')
    if not (math.isfinite(args.refinement_factor) and args.refinement_factor > 1):
        raise ValueError('refinement-factor must be finite and > 1.')
    if not (args.max_refinements >= 0 and args.convergence_streak >= 1):
        raise ValueError('Invalid refinement count or streak.')
    if not (args.max_grid_points >= 8 and args.max_epost >= 3):
        raise ValueError('Adaptive resource limits are too small.')
    if not args.fixed_resolution:
        if not (args.n_eout * args.n_aout * args.n_nu <= args.max_grid_points and args.n_epost <= args.max_epost):
            raise ValueError('Initial resolution exceeds adaptive resource limits.')


def parse_system_summary_args(argv=None):
    """Parse, resolve, and validate arguments for this script."""
    parser = build_system_summary_parser()
    args = parser.parse_args(argv)
    try:
        validate_system_summary_args(args)
        validate_posterior_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    return args


# -------------------------------------------------------------------------------------------------
# Posterior plots
# -------------------------------------------------------------------------------------------------

def build_posterior_plots_parser() -> argparse.ArgumentParser:
    """Build the parser for saved posterior support curves and sample distributions."""
    parser = argparse.ArgumentParser(
        description='Plot posterior support curves or weighted distributions from a saved report.'
    )
    parser.add_argument('json_file', type=Path, help='Posterior system-summary JSON')
    parser.add_argument('--output', type=Path, help='Output PNG, PDF, or SVG (default: filename based on selected figure)')
    parser.add_argument('--family', choices=('both', 'all_targets', 'secular_targets'), default='both')
    parser.add_argument('--include-r-sec', action='store_true', help='Add an R_sec histogram panel before the orbital projections')
    parser.add_argument('--distributions', action='store_true', help='Plot five sample distributions instead of support curves')
    parser.add_argument('--converged-only', action='store_true', help='Select converged samples for distributions, keeping original weights')
    parser.add_argument('--bins', type=int, default=20, help='Bins per distribution')
    parser.add_argument('--volume-scale', choices=('log', 'linear'), default='log', help='Volume-fraction axis for distributions (default: log)')
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def validate_posterior_plots_args(args: argparse.Namespace) -> None:
    """Validate plotting options and resolve an output path that preserves the input report."""
    if args.dpi < 1:
        raise ValueError('--dpi must be positive')
    if args.bins < 1:
        raise ValueError('--bins must be positive')
    if args.converged_only and not args.distributions:
        raise ValueError('--converged-only requires --distributions')
    if args.include_r_sec and args.distributions:
        raise ValueError('--distributions already includes R_sec; omit --include-r-sec')
    suffix = '_distributions_converged.png' if args.converged_only else '_distributions.png'
    if not args.distributions:
        suffix = '_support.png'
    args.output = args.output or args.json_file.with_name(args.json_file.stem + suffix)
    if args.output.resolve() == args.json_file.resolve():
        raise ValueError('Output must not overwrite the input JSON')


def parse_posterior_plots_args(argv=None):
    """Parse plotting arguments, validate option combinations, and resolve the output path."""
    parser = build_posterior_plots_parser()
    args = parser.parse_args(argv)
    try:
        validate_posterior_plots_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


# -------------------------------------------------------------------------------------------------
# Post-processing shared by every script
# -------------------------------------------------------------------------------------------------

def resolve_e_ref_range(args: argparse.Namespace) -> argparse.Namespace:
    """Convert ``e_ref_range`` into ``e_ref_min`` / ``e_ref_max``."""
    e_ref_range = getattr(args, "e_ref_range", None)
    if e_ref_range is not None:
        args.e_ref_min, args.e_ref_max = e_ref_range
    return args


def system_from_args(args: argparse.Namespace) -> TripleSystem:
    """Construct the physical system from CLI values, converting degrees to radians."""
    return TripleSystem(
        m1_inner=args.m1_inner,
        m2_inner=args.m2_inner,
        mrem=args.mrem,
        m3=args.m3,
        delay_days=args.delay_days,
        inclination_rad=math.radians(args.inclination_deg),
    )


def settings_from_args(args: argparse.Namespace) -> AnalysisSettings:
    """Build the :class:`AnalysisSettings` record from parsed arguments."""
    return AnalysisSettings(
        e_ref_min=getattr(args, "e_ref_min", None),
        e_ref_max=getattr(args, "e_ref_max", None),
        require_peters_applicable=getattr(args, "secular_only", False),
        recoil_ceiling_kms=getattr(args, "recoil_ceiling_kms", None),
        require_stability=getattr(args, "use_ma_mask", True),
        inner_separation_au=getattr(args, "inner_separation_au", None),
        alpha_in=getattr(args, "alpha_in", None),
        phase_measure=getattr(args, "phase_measure", PHASE_MEAN_ANOMALY),
        stability_days_before=getattr(args, "stability_days_before", DEFAULT_STABILITY_DAYS_BEFORE),
        r_sec_threshold=getattr(args, "r_sec_threshold", DEFAULT_R_SEC_THRESHOLD),
    )
