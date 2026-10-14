#!/usr/bin/env python3
"""PyVista visualization of the three-dimensional minimum-recoil-velocity map cube."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np

from triple_core import (
    PHASE_MEAN_ANOMALY,
    PHASE_MEASURES,
    PHASE_TRUE_ANOMALY,
    compute_min_kick_map_slice,
    compute_post_first_merger_candidates,
    select_admitted_targets,
    make_epost_grid,
)

from triple_io import save_plot_output

from triple_cli import (
    parse_kick_map_3d_args as parse_args,
    validate_kick_map_3d_args as validate_args,
    system_from_args,
    settings_from_args,
    resolve_e_ref_range,
)


# -------------------------------------------------------------------------------------------------
# Build the 3D minimum-recoil-velocity map
# -------------------------------------------------------------------------------------------------

# Bound changes in log(1 - e) between neighboring eccentricity planes
MAX_LOG_ECCENTRICITY_GAP = 0.1


# Each worker keeps its own inputs, populated once when the process starts
_worker_slice_inputs = {}


def _initialize_slice_worker(slice_kwargs):
    """Load the grids and target table once per worker process."""
    _worker_slice_inputs.clear()
    _worker_slice_inputs.update(slice_kwargs)


def _compute_eout_slice(e_out, slice_kwargs):
    """Compute mirrored phases once, then return the full (phase, separation) plane."""
    return compute_min_kick_map_slice(
        plane="aout-nu",
        fixed_value=float(e_out),
        **slice_kwargs).vk_min_kms.T


def _compute_eout_slice_worker(e_out):
    return _compute_eout_slice(e_out, _worker_slice_inputs)


def eccentricity_grid(lower, upper, count, spacing):
    """Keep the base eccentricity grid and fill large gaps in log(1 - e) when refined."""
    base = np.linspace(lower, upper, count)
    if spacing == "linear":
        return base
    planes = [float(base[0])]
    for left, right in zip(base[:-1], base[1:]):
        log_left, log_right = np.log1p(-left), np.log1p(-right)
        subdivisions = int(np.ceil((log_left - log_right) / MAX_LOG_ECCENTRICITY_GAP))
        interior = -np.expm1(np.linspace(log_left, log_right, subdivisions + 1)[1:-1])
        planes.extend(interior)
        planes.append(float(right))
    return np.asarray(planes)


def build_vk_cube(args):
    system = system_from_args(args)
    settings = settings_from_args(args)

    # 1D physical grids
    e_out_grid = eccentricity_grid(
        args.eout_min, args.eout_max, args.n_eout, args.eout_spacing
    )
    a_grid_au = np.logspace(np.log10(args.aout_min_au), np.log10(args.aout_max_au), args.n_aout)
    phase_deg_grid = np.linspace(args.nu_min_deg, args.nu_max_deg, args.n_nu)
    phase_rad_grid = np.radians(phase_deg_grid)
    loga_grid = np.log10(a_grid_au)

    # Build the candidate-orbits
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

    # Apply the target-family cuts
    targets = select_admitted_targets(candidates, settings)

    # Plotting convention:
    # x-axis -> phase (nu_m or M_m)
    # y-axis -> log10 a_out (units chosen by --aout-axis)
    # z-axis -> e_out

    cube = np.full((args.n_nu, args.n_aout, len(e_out_grid)), np.nan, dtype=float)
    valid = np.zeros_like(cube, dtype=np.uint8)

    print("Building v_k,min cube: ",
          f"n_phase={args.n_nu}, n_x={args.n_aout}, n_e={len(e_out_grid)}")

    if len(e_out_grid) > args.n_eout:
        print(f"Added {len(e_out_grid) - args.n_eout} high-eccentricity planes.", flush=True)

    slice_kwargs = dict(
        a_grid_au=a_grid_au,
        e_out_grid=e_out_grid,
        nu_grid_rad=phase_rad_grid,
        targets=targets,
        m1_inner_msun=args.m1_inner,
        m2_inner_msun=args.m2_inner,
        mrem_msun=args.mrem,
        m3_msun=args.m3,
        inclination_rad=system.inclination_rad,
        settings=settings,
    )
    workers = min(args.workers, len(e_out_grid))
    print(f"Computing e_out slices with {workers} worker(s).", flush=True)

    def store_slice(k, plane, completed):
        cube[:, :, k] = plane
        valid[:, :, k] = np.isfinite(plane).astype(np.uint8)
        print(f"  Completed {completed:>3d}/{len(e_out_grid)} slices: e_out={e_out_grid[k]:.5f}",
            flush=True)

    if workers == 1:
        for k, e_out in enumerate(e_out_grid):
            store_slice(k, _compute_eout_slice(e_out, slice_kwargs), k + 1)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_slice_worker,
            initargs=(slice_kwargs,),
        ) as pool:
            pending = {pool.submit(_compute_eout_slice_worker,
                float(e)): k for k, e in enumerate(e_out_grid)}
            try:
                for completed, future in enumerate(as_completed(pending), start=1):
                    k = pending.pop(future)
                    store_slice(k, future.result(), completed)
            except BaseException:
                for future in pending:
                    future.cancel()
                raise

    return {
        "cube": cube,
        "valid": valid,
        "phase_deg_grid": phase_deg_grid,
        "loga_grid": loga_grid,
        "e_out_grid": e_out_grid,
        "rg_au": system.rg_post_au,
    }


def make_plot_grid(cube, valid, phase_deg_grid, loga_grid, e_out_grid):
    """Place cube values on their normalized coordinates, preserving nonuniform spacing."""
    import pyvista as pv

    if len(phase_deg_grid) < 2 or len(loga_grid) < 2 or len(e_out_grid) < 2:
        raise ValueError("All three axes must have at least 2 points.")

    coordinates = [np.asarray(axis, dtype=float)
                   for axis in (phase_deg_grid, loga_grid, e_out_grid)]
    if any(not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0)
           for axis in coordinates):
        raise ValueError("Grid coordinates must be finite and strictly increasing.")
    dimensions = tuple(len(axis) for axis in coordinates)
    if np.shape(cube) != dimensions or np.shape(valid) != dimensions:
        raise ValueError("Cube and validity arrays must match the coordinate dimensions.")
    normalized = [(axis - axis[0]) / (axis[-1] - axis[0]) for axis in coordinates]
    uniform = all(np.allclose(axis, np.linspace(0., 1., len(axis)), rtol=0., atol=1e-14)
                  for axis in normalized)
    if uniform:
        grid = pv.ImageData(
            dimensions=dimensions, spacing=tuple(1 / (size - 1) for size in dimensions),
            origin=(0., 0., 0.),
        )
    else:
        grid = pv.RectilinearGrid(*normalized)

    vk_data = np.asarray(cube, dtype=float)
    valid_data = np.asarray(valid, dtype=np.uint8)

    grid.point_data["vk_min"] = vk_data.ravel(order="F")
    grid.point_data["valid"] = valid_data.ravel(order="F")
    grid.set_active_scalars("vk_min")

    return grid


def volume_rendering_grid(grid, map_data):
    """Resample the faint background volume uniformly while retaining the refined analysis grid."""
    import pyvista as pv

    if isinstance(grid, pv.ImageData):
        return grid

    # Interpolate along eccentricity at physical positions on a regular display grid
    eccentricities = map_data["e_out_grid"]
    uniform_eccentricities = np.linspace(
        eccentricities[0], eccentricities[-1], len(eccentricities)
    )
    source = map_data["cube"]
    values = np.empty(source.shape, dtype=np.float32)
    for index, eccentricity in enumerate(uniform_eccentricities):
        right = min(np.searchsorted(eccentricities, eccentricity, side="right"),
                    len(eccentricities) - 1)
        left = right - 1
        weight = ((eccentricity - eccentricities[left])
                  / (eccentricities[right] - eccentricities[left]))
        # Exact samples stay valid even when their neighboring plane is invalid
        if weight == 0.:
            values[:, :, index] = source[:, :, left]
        elif weight == 1.:
            values[:, :, index] = source[:, :, right]
        else:
            values[:, :, index] = (
                (1 - weight) * source[:, :, left] + weight * source[:, :, right]
            )

    # Keep NaN gaps transparent; use this grid only for the decorative volume layer
    volume = pv.ImageData(
        dimensions=source.shape, spacing=tuple(1 / (size - 1) for size in source.shape)
    )
    volume.point_data["vk_min"] = values.ravel(order="F")
    return volume


# -------------------------------------------------------------------------------------------------
# Coordinate helpers
# -------------------------------------------------------------------------------------------------

def physical_to_unit(value: float, coords: np.ndarray, name: str) -> float:
    """Map a physical coordinate value to the normalized [0, 1] cube axis."""
    lo = float(coords[0])
    hi = float(coords[-1])

    if not lo <= value <= hi:
        raise ValueError(f"Requested {name}={value:g} lies outside scanned range [{lo:g}, {hi:g}]")

    if hi == lo:
        return 0.0

    return (float(value) - lo) / (hi - lo)


def aout_axis_values(map_data, aout_axis: str) -> tuple[np.ndarray, str]:
    """Return the plotted a_out axis values and its physical-coordinate name."""
    if aout_axis == "au":
        return map_data["loga_grid"], "Log10 outer semi-major axis (AU)"
    if aout_axis == "rg":
        return (
            map_data["loga_grid"] - np.log10(map_data["rg_au"]),
            "Log10 outer semi-major axis (gravitational radii)",
        )
    raise ValueError("aout_axis must be 'au' or 'rg'.")


def phase_axis_name(phase_measure: str) -> str:
    """Plain-text phase-axis title, also used in slice-position errors."""
    if phase_measure == PHASE_MEAN_ANOMALY:
        return "Mean anomaly (degrees)"
    if phase_measure == PHASE_TRUE_ANOMALY:
        return "True anomaly (degrees)"
    raise ValueError(f"phase_measure must be one of {PHASE_MEASURES}, got {phase_measure!r}.")


def slice_definition(map_data, args):
    """Return normalized slice origin/normal and a human-readable description."""
    axis = args.slice_axis.lower()

    if axis == "x":
        # Phase coordinate in degrees
        u = physical_to_unit(
            args.slice_value,
            map_data["phase_deg_grid"],
            phase_axis_name(args.phase_measure)
        )
        origin = (u, 0.5, 0.5)
        normal = (1.0, 0.0, 0.0)
        description = f"{phase_axis_name(args.phase_measure)} = {args.slice_value:g}"

    elif axis == "y":
        # Semimajor axis --aout-axis
        aout_values, aout_name = aout_axis_values(map_data, args.aout_axis)
        u = physical_to_unit(args.slice_value, aout_values, aout_name)
        origin = (0.5, u, 0.5)
        normal = (0.0, 1.0, 0.0)
        description = f"{aout_name} = {args.slice_value:g}"

    elif axis == "z":
        # Outer eccentricity
        u = physical_to_unit(args.slice_value, map_data["e_out_grid"], "e_out")
        origin = (0.5, 0.5, u)
        normal = (0.0, 0.0, 1.0)
        description = f"Outer eccentricity = {args.slice_value:g}"

    else:
        raise ValueError("slice-axis must be one of x, y, z")

    return origin, normal, description


# -------------------------------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------------------------------

def extract_axis_slice(grid, origin, normal):
    """Extract boundary planes exactly; VTK's cutter omits the lower boundary."""
    axis = int(np.argmax(np.abs(normal)))
    if origin[axis] in (0.0, 1.0):
        extent = [0, grid.dimensions[0] - 1, 0, grid.dimensions[1] - 1, 0, grid.dimensions[2] - 1]
        index = 0 if origin[axis] == 0.0 else grid.dimensions[axis] - 1
        extent[2 * axis:2 * axis + 2] = [index, index]
        import pyvista as pv

        if isinstance(grid, pv.RectilinearGrid):
            from vtkmodules.vtkFiltersExtraction import vtkExtractRectilinearGrid

            extractor = vtkExtractRectilinearGrid()
            extractor.SetInputData(grid)
            extractor.SetVOI(*extent)
            extractor.Update()
            return pv.wrap(extractor.GetOutput()).extract_surface()
        return grid.extract_subset(extent).extract_surface()
    return grid.slice(normal=normal, origin=origin)


def plot_cube(grid, map_data, args):
    import pyvista as pv

    cube = map_data["cube"]
    finite = np.isfinite(cube)

    if not np.any(finite):
        raise RuntimeError(
            "The 3D kick cube contains no finite v_k,min values. "
            "Check the mass/delay/eccentricity filters and the scanned a_out range."
        )

    vk_data_min = float(np.nanmin(cube))
    vk_data_max = float(np.nanmax(cube))
    highlight_value = float(args.recoil_ceiling_kms)
    if args.offscreen:
        try:
            pv.start_xvfb()
        except Exception:
            pass

    plotter = pv.Plotter(off_screen=args.offscreen, window_size=(1600, 1100))
    plotter.set_background(args.background)

    # Transparency generally looks cleaner with depth peeling
    try:
        plotter.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.0)
    except Exception:
        pass

    # Unit-cube outline
    plotter.add_mesh(grid.outline(), color=args.outline_color, line_width=args.outline_width)

    # ---------------------------------------------------------------------------------------------
    # Volume rendering
    # ---------------------------------------------------------------------------------------------

    if args.show_volume:
        # Keep the volume deliberately faint so the highlighted region remains visible
        opacity = [0.00, 0.01, 0.01, 0.01, 0.01, 0.025]

        volume_kwargs = dict(
            scalars="vk_min",
            cmap=args.cmap,
            clim=[args.vmin, args.vmax],
            opacity=opacity,
            shade=False,
            blending="composite",
            opacity_unit_distance=0.01,
            show_scalar_bar=False,
        )

        volume_grid = volume_rendering_grid(grid, map_data)

        # Newer PyVista versions support nan_opacity
        try:
            plotter.add_volume(volume_grid, nan_opacity=0.0, **volume_kwargs)
        except TypeError:
            plotter.add_volume(volume_grid, **volume_kwargs)

    # ---------------------------------------------------------------------------------------------
    # Highlight the low-kick region
    # ---------------------------------------------------------------------------------------------

    # Extract the highlighted region directly from the calculated grid
    if args.show_highlight and highlight_value >= vk_data_min:
        try:
            upper = min(highlight_value, vk_data_max)
            low_region = grid.threshold(
                value=(vk_data_min, upper),
                scalars="vk_min",
                preference="point",
            )
            if low_region.n_points > 0 and low_region.n_cells > 0:
                low_surface = low_region.extract_surface().clean()
                plotter.add_mesh(
                    low_surface,
                    color=args.highlight_color,
                    opacity=args.highlight_opacity,
                    smooth_shading=True,
                    show_scalar_bar=False,
                    specular=args.highlight_specular,
                )
            else:
                print(f"WARNING: no geometric region was extracted below {highlight_value:g} km/s.")
        except Exception as exc:
            print(f"WARNING: failed to extract the region with "
                  f"v_k,min <= {highlight_value:g} km/s: {exc}")

    # ---------------------------------------------------------------------------------------------
    # Optional physical-coordinate slice
    # ---------------------------------------------------------------------------------------------

    if args.show_slice:
        origin, normal, slice_description = slice_definition(map_data, args)

        # Slice the original grid. NaN regions naturally remain absent/blank.
        slice_mesh = extract_axis_slice(grid, origin, normal)

        print(
            f"Slice {slice_description}: "
            f"{slice_mesh.n_points} points, {slice_mesh.n_cells} cells"
        )

        # Split the slice into valid (colored) and invalid (gray background)
        try:
            slice_valid = slice_mesh.threshold(
                value=0.5,
                scalars="valid",
                preference="point",
            )
            slice_invalid = slice_mesh.threshold(
                value=0.5,
                scalars="valid",
                preference="point",
                invert=True,
            )
        except Exception:
            slice_valid = slice_mesh
            slice_invalid = None

        # Slightly transparent gray background only where no finite kick value is available
        if (slice_invalid is not None and slice_invalid.n_points > 0 and slice_invalid.n_cells > 0):
            plotter.add_mesh(
                slice_invalid,
                color=args.slice_background_color,
                opacity=args.slice_background_opacity,
                lighting=False,
                show_scalar_bar=False,
            )

        # Render only valid values with the viridis map
        if slice_valid.n_points > 0 and slice_valid.n_cells > 0:
            plotter.add_mesh(
                slice_valid,
                scalars="vk_min",
                cmap=args.cmap,
                clim=[args.vmin, args.vmax],
                opacity=args.slice_opacity,
                show_scalar_bar=False,
                lighting=True,
                ambient=args.slice_ambient,
                diffuse=args.slice_diffuse,
                specular=args.slice_specular,
            )

        # Add a thick square outline around the full slice
        try:
            slice_outline = slice_mesh.outline()
        except Exception:
            slice_outline = slice_mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            )

        if slice_outline.n_points > 0 and slice_outline.n_cells > 0:
            plotter.add_mesh(
                slice_outline,
                color=args.slice_outline_color,
                line_width=args.slice_outline_width,
                opacity=1.0,
                render_lines_as_tubes=True,
            )

    # ---------------------------------------------------------------------------------------------
    # Physical axis labels on the normalized cube
    # ---------------------------------------------------------------------------------------------

    aout_values, aout_title = aout_axis_values(map_data, args.aout_axis)
    physical_axes_ranges = (
        float(map_data["phase_deg_grid"][0]),
        float(map_data["phase_deg_grid"][-1]),
        float(aout_values[0]),
        float(aout_values[-1]),
        float(map_data["e_out_grid"][0]),
        float(map_data["e_out_grid"][-1]),
    )

    bounds_kwargs = dict(
        mesh=grid,
        location="outer",
        ticks="outside",
        axes_ranges=physical_axes_ranges,
        n_xlabels=args.axis_nlabels,
        n_ylabels=args.axis_nlabels,
        n_zlabels=args.axis_nlabels,
        color=args.text_color,
        bold=False,
    )
    label_axes = plotter.show_bounds(
        **bounds_kwargs, grid="back", xtitle="", ytitle="", ztitle="",
        font_size=args.axis_label_font_size,
    )
    title_axes = plotter.show_bounds(
        **bounds_kwargs,
        xtitle=phase_axis_name(args.phase_measure),
        ytitle=aout_title,
        ztitle="Outer eccentricity",
        font_size=args.axis_title_font_size,
    )
    for index, axis in enumerate(("X", "Y", "Z")):
        title_axes.GetLabelTextProperty(index).SetOpacity(0.0)
        getattr(title_axes, f"Get{axis}AxesLinesProperty")().SetOpacity(0.0)
        getattr(title_axes, f"Set{axis}AxisTickVisibility")(False)
    plotter.add_actor(label_axes, reset_camera=False, pickable=False)
    label_axes.x_axis_range = physical_axes_ranges[0:2]
    label_axes.y_axis_range = physical_axes_ranges[2:4]
    label_axes.z_axis_range = physical_axes_ranges[4:6]

    # Camera: slight 3D perspective
    plotter.camera_position = [
        (2.9, 0.6, 2.9),  # camera position
        (0.50, 0.50, 0.50),  # focal point
        (0.0, 1.0, 0.0),  # view-up direction
    ]
    plotter.camera.zoom(args.camera_zoom)

    save_plot_output(plotter, args)


def main():
    args = parse_args()

    map_data = build_vk_cube(args)

    grid = make_plot_grid(
        cube=map_data["cube"],
        valid=map_data["valid"],
        phase_deg_grid=map_data["phase_deg_grid"],
        loga_grid=map_data["loga_grid"],
        e_out_grid=map_data["e_out_grid"],
    )

    plot_cube(grid, map_data, args)


if __name__ == "__main__":
    main()
