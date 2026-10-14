#!/usr/bin/env python3
"""Plot posterior support curves or weighted distributions from a saved system-summary report."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
import numpy as np

from triple_cli import parse_posterior_plots_args as parse_args


FAMILY_STYLES = {
    'all_targets': ('All compatible targets', '#236A9F', '-', 2.2),
    'secular_targets': ('Secular targets', '#C46B25', (0, (4, 2.5)), 1.8),
}


# -------------------------------------------------------------------------------------------------
# Projected support curves
# -------------------------------------------------------------------------------------------------

def union_label(intervals, axis):
    """Format disjoint support intervals with axis units, placing two intervals on each line."""
    if not intervals:
        return r'$\varnothing$'
    if axis == 'a_out':
        intervals = [
            (10**interval_start, 10**interval_end) for interval_start, interval_end in intervals
        ]
    parts = [
        f'[{interval_start:.4g}, {interval_end:.4g}]' for interval_start, interval_end in intervals
    ]
    # Wrap interval labels in pairs
    lines = [' ∪ '.join(parts[index : index + 2]) for index in range(0, len(parts), 2)]
    units = {'a_out': ' AU', 'phase': '°', 'e_out': ''}[axis]
    return '\n'.join(lines) + units


def plot_report(
    report,
    output,
    *,
    family='both',
    dpi=300,
    include_r_sec=False,
    panel_height=2.98,
    show_legend=True,
    font_scale=1.0,
):
    """Plot saved support probabilities and an optional weighted R_sec histogram.

    Overlay the selected target families, label their support intervals, and save the figure
    using the report's bin values and sample weights.
    """
    # Validate projected support and convert separation coordinates to AU
    try:
        families = report['summary']['families']
    except KeyError as exc:
        raise ValueError('Expected a posterior summary JSON with summary.families.') from exc
    selected_families = ['all_targets', 'secular_targets'] if family == 'both' else [family]
    family_styles = FAMILY_STYLES
    projection_data = {}
    for name in selected_families:
        for axis in ('e_out', 'a_out', 'phase'):
            try:
                projection = families[name]['projections'][axis]
                edges = np.asarray(projection['edges'], dtype=float)
                values = np.asarray(projection['support_probability'], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Missing or invalid {name}.{axis} projection.') from exc
            if (
                edges.ndim != 1
                or values.ndim != 1
                or len(edges) != len(values) + 1
                or not len(values)
                or not np.all(np.isfinite(edges))
                or not np.all(np.diff(edges) > 0)
                or not np.all(np.isfinite(values))
                or np.any(values < -1e-10)
                or np.any(values > 1 + 1e-10)
            ):
                raise ValueError(f'Invalid edges or probabilities for {name}.{axis}.')
            if axis == 'a_out':
                if not str(projection.get('units', '')).startswith('log10('):
                    raise ValueError('Expected a_out edges in log10(AU); refusing to guess units.')
                edges = 10.0 ** edges
                if not np.all(np.isfinite(edges)) or np.any(edges <= 0):
                    raise ValueError('a_out edges cannot be converted to positive AU values.')
            projection_data[name, axis] = edges, values

    # Prepare interval labels and reserve their header space
    interval_labels = {}
    for name in selected_families:
        for axis in ('a_out', 'e_out', 'phase'):
            intervals = families[name]['projections'][axis]['support_union']
            interval_labels[name, axis] = union_label(intervals, axis)
    header_lines = sum(
        max(interval_labels[name, axis].count('\n') + 1 for axis in ('a_out', 'e_out', 'phase'))
        for name in selected_families
    )
    if include_r_sec:
        header_lines = max(header_lines, 2)
    figure_height = panel_height + 1.17 + 0.18 * header_lines

    # Configure the figure and shared support axes
    with plt.rc_context(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 11 * font_scale,
            'axes.labelsize': 12 * font_scale,
            'axes.titlesize': 12 * font_scale,
            'axes.titlepad': 12,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': '#9AA4AF',
            'axes.labelcolor': '#263444',
            'xtick.color': '#526171',
            'ytick.color': '#526171',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    ):
        figure, panels = plt.subplots(
            1, 4 if include_r_sec else 3, figsize=(17.6 if include_r_sec else 13.2, figure_height)
        )
        support_axes = panels[1:] if include_r_sec else panels
        for panel in support_axes[1:]:
            panel.sharey(support_axes[0])
            panel.tick_params(labelleft=False)
        phase_measure = report['configuration']['phase_measure']
        phase_label = (
            r'Mean anomaly $\mathcal{M}_{\rm m}$ [deg]'
            if phase_measure == 'mean_anomaly'
            else r'True anomaly $\nu$ [deg]'
        )
        labels = [
            r'Outer semimajor axis $a_{\rm out}$ [AU]',
            r'Outer eccentricity $e_{\rm out}$',
            phase_label,
        ]

        # Draw the orbital support curves and interval labels
        for panel, axis, label in zip(support_axes, ('a_out', 'e_out', 'phase'), labels):
            for name in selected_families:
                edges, values = projection_data[name, axis]
                legend, color, linestyle, width = family_styles[name]
                panel.stairs(
                    values, edges, baseline=0, fill=True, color=color, alpha=0.08, linewidth=0
                )
                panel.stairs(
                    values,
                    edges,
                    baseline=None,
                    label=legend,
                    color=color,
                    linestyle=linestyle,
                    linewidth=width,
                )
            annotation_offset = 0
            for name in selected_families:
                prefix = 'All' if name == 'all_targets' else 'Secular'
                text = interval_labels[name, axis]
                panel.annotate(
                    f'{prefix}: {text}',
                    xy=(0.5, 1),
                    xycoords='axes fraction',
                    xytext=(0, 6 + 11 * font_scale * (header_lines - annotation_offset)),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    fontsize=9 * font_scale,
                    color=family_styles[name][1],
                )
                annotation_offset += text.count('\n') + 1
            panel.set(xlabel=label, ylim=(0, 1.035))
            panel.set_axisbelow(True)
            panel.grid(axis='y', color='#DDE3E9', linewidth=0.7)
            panel.yaxis.set_major_locator(MultipleLocator(0.2))
            panel.tick_params(direction='out', length=4, pad=5)
            panel.set_xlim(
                min(projection_data[n, axis][0][0] for n in selected_families),
                max(projection_data[n, axis][0][-1] for n in selected_families),
            )
            if axis == 'a_out':
                panel.set_xscale('log')
                panel.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
                panel.xaxis.set_minor_formatter(NullFormatter())
            elif axis == 'e_out':
                panel.set_xlim(right=1.0)
            elif axis == 'phase':
                panel.xaxis.set_major_locator(MultipleLocator(90))
        support_axes[0].set_ylabel('Probability of projected support', fontsize=12 * font_scale)

        # Add the optional reliability distribution
        if include_r_sec:
            panel = panels[0]
            summary = report['summary']
            median = (summary['r_sec'].get('quantiles') or {}).get('median')
            median_text = f'{median:.3f}' if median is not None else 'undefined'
            class_weights = summary['families']['all_targets'].get('sampled_class_weights')
            incompatible_text = 'unavailable'
            if class_weights is not None:
                # Older reports used "excluded" for the same empty-region classification.
                incompatible_weight = sum(
                    class_weights.get(label, 0.0) for label in ('peters-incompatible', 'excluded')
                )
                incompatible_text = f'{incompatible_weight:.1%}'
            panel.annotate(
                r'Median $R_{\rm sec}$ = ' + median_text
                + f'\nPeters-incompatible: {incompatible_text}',
                xy=(0.5, 1),
                xycoords='axes fraction',
                xytext=(0, 6 + 11 * font_scale * header_lines),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=11 * font_scale,
                color='#263444',
            )
            histogram = summary['r_sec']['histogram']
            panel.stairs(
                histogram['weight_per_bin'],
                histogram['edges'],
                fill=True,
                color='#236A9F',
                alpha=0.3,
                linewidth=1.8,
            )
            panel.axvline(
                summary['r_sec']['threshold'], color='#647282', linestyle=':', linewidth=1
            )
            panel.set(
                xlabel=r'$R_{\rm sec}$',
                ylabel='Sample weight per bin',
                xlim=(0, 1),
                ylim=(0, 1.035),
            )
            panel.set_axisbelow(True)
            panel.grid(axis='y', color='#DDE3E9', linewidth=0.7)

        # Arrange the legend and panel spacing
        handles, legends = support_axes[0].get_legend_handles_labels()
        if show_legend:
            figure.legend(
                handles,
                legends,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.99),
                ncol=len(selected_families),
                frameon=False,
                handlelength=3,
            )
        figure.subplots_adjust(
            left=0.075,
            right=0.99,
            bottom=0.69 / figure_height,
            top=1 - (0.48 + 0.18 * header_lines) / figure_height,
            wspace=0.30 if include_r_sec else 0.14,
        )
        if include_r_sec:
            # Reserve extra space beside the support probability label
            left, right = 0.075, 0.99
            gaps = (0.075, 0.035, 0.035)
            panel_width = (right - left - sum(gaps)) / 4
            panel_left = left
            for index, panel in enumerate(panels):
                position = panel.get_position()
                panel.set_position([panel_left, position.y0, panel_width, position.height])
                if index < 3:
                    panel_left += panel_width + gaps[index]

        # Fit the vertical support label and save the figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        label = support_axes[0].yaxis.label
        label_height = label.get_window_extent(renderer).height
        axis_height = support_axes[0].get_window_extent(renderer).height
        if label_height > 1.06 * axis_height:
            label.set_fontsize(label.get_fontsize() * 1.06 * axis_height / label_height)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(figure)


# -------------------------------------------------------------------------------------------------
# Weighted distributions across posterior samples
# -------------------------------------------------------------------------------------------------

def _select_distribution_samples(report, converged_only):
    """Select saved sample records and their original nonnegative posterior weights."""
    samples = report.get('samples')
    if not isinstance(samples, list):
        raise ValueError('Distribution plots require the saved per-sample records in samples.')
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError('Invalid posterior sample record.')
    try:
        weights = np.array([sample['weight'] for sample in samples], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Each posterior sample must have a numeric weight.') from exc
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() > 1 + 1e-8:
        raise ValueError('Posterior weights must be finite, nonnegative, and sum to at most one.')
    selected = np.array([
        not converged_only or sample.get('numerical_status') == 'converged'
        for sample in samples
    ], dtype=bool)
    return [sample for sample, keep in zip(samples, selected) if keep], weights[selected]


def _sample_metric(samples, weights, metric, family=None, axis=None):
    """Collect one fraction per sample and measure its zero and undefined posterior weight."""
    values = []
    for sample in samples:
        record = sample.get('result') or {}
        if family is not None:
            record = record.get('families', {}).get(family) or {}
        if axis is not None:
            record = record.get('projections', {}).get(axis) or {}
        value = record.get(metric)
        try:
            values.append(float(value) if value is not None else np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid saved {metric} value.') from exc
    values = np.asarray(values)
    defined = np.isfinite(values)
    if np.any((values[defined] < -1e-12) | (values[defined] > 1 + 1e-12)):
        raise ValueError(f'Saved {metric} values must lie between zero and one.')
    values[defined] = np.clip(values[defined], 0., 1.)
    return {
        'values': values,
        'defined': defined,
        'zero_weight': float(weights[defined & (values == 0)].sum()),
        'undefined_weight': float(weights[~defined].sum()),
    }


def _volume_bin_edges(distributions, weights, bins):
    """Choose shared logarithmic bins spanning positive, nonzero-weight volume fractions."""
    positive = [
        distribution['values'][
            distribution['defined'] & (distribution['values'] > 0) & (weights > 0)
        ]
        for distribution in distributions
    ]
    values = np.concatenate(positive)
    if not len(values):
        return np.geomspace(1e-6, 1., bins + 1)
    lower = max(np.nextafter(0., 1.), float(values.min()) / 2)
    upper = min(1., float(values.max()) * 2)
    return np.unique(np.geomspace(lower, upper, bins + 1))


def plot_distributions(
    report, output, *, family='both', converged_only=False, bins=20, volume_scale='log', dpi=300
):
    """Plot five sample distributions using their original posterior weights.

    Overlay target families for volume fraction and each axis's support coverage. R_sec uses one
    shared distribution. Logarithmic volume bins omit zeros; other panels include zeros in their
    first bin. Omit undefined values and do not renormalize selected samples.
    """
    if family not in ('both', *FAMILY_STYLES):
        raise ValueError('Unknown target family.')
    if not isinstance(bins, int) or bins < 1 or dpi < 1:
        raise ValueError('Bin count and DPI must be positive.')
    if volume_scale not in ('log', 'linear'):
        raise ValueError('Volume scale must be log or linear.')

    # Collect each metric with a common set of sample weights
    samples, weights = _select_distribution_samples(report, converged_only)
    families = list(FAMILY_STYLES) if family == 'both' else [family]
    panels = [
        (r'$R_{\rm sec}$', {'reliability': _sample_metric(samples, weights, 'r_sec')}),
        ('Compatible volume fraction', {
            name: _sample_metric(samples, weights, 'volume_fraction', name) for name in families
        }),
    ]
    for axis, label in (
        ('e_out', 'Eccentricity support fraction'),
        ('a_out', 'Separation support fraction'),
        ('phase', 'Phase support fraction'),
    ):
        panels.append((label, {
            name: _sample_metric(samples, weights, 'support_coverage', name, axis)
            for name in families
        }))

    # Plot weighted histograms
    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#263444', 'axes.edgecolor': '#9AA4AF',
        'xtick.color': '#526171', 'ytick.color': '#526171',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    }):
        fig, axes = plt.subplots(1, 5, figsize=(21, 4.8))
        try:
            for index, (ax, (label, distributions)) in enumerate(zip(axes, panels)):
                logarithmic = index == 1 and volume_scale == 'log'
                edges = (
                    _volume_bin_edges(distributions.values(), weights, bins)
                    if logarithmic else np.linspace(0., 1., bins + 1)
                )
                largest_bin = 0.
                for name, distribution in distributions.items():
                    legend, color, style, width = FAMILY_STYLES.get(
                        name, ('All selected samples', '#263444', '-', 2.)
                    )
                    values, valid = distribution['values'], distribution['defined'].copy()
                    if logarithmic:
                        valid &= values > 0
                    counts = np.histogram(values[valid], bins=edges, weights=weights[valid])[0]
                    largest_bin = max(largest_bin, float(counts.max()))
                    ax.stairs(counts, edges, color=color, linestyle=style, linewidth=width,
                              label=legend)
                ax.set(xlabel=label, xlim=(edges[0], edges[-1]),
                       ylim=(0., largest_bin * 1.15 if largest_bin else 1.))
                ax.grid(axis='y', color='#DDE3E9', linewidth=.7)
                ax.set_axisbelow(True)
                if logarithmic:
                    ax.set_xscale('log')
                    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                    if largest_bin == 0:
                        ax.text(.5, .5, 'No positive volume weight', transform=ax.transAxes,
                                ha='center', color='#526171')
                if index == 0:
                    threshold = report.get('configuration', {}).get('r_sec_threshold')
                    if threshold is not None and np.isfinite(float(threshold)):
                        ax.axvline(float(threshold), color='#647282', linestyle=':', linewidth=1)
            axes[0].set_ylabel('Posterior weight per bin')

            # Add the family legend and save the distribution figure
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .98),
                       ncol=len(families), frameon=False)
            fig.subplots_adjust(left=.05, right=.99, bottom=.17, top=.87, wspace=.32)
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)


# -------------------------------------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------------------------------------

def main(argv=None):
    """Load a saved report and render the figure selected by the command-line arguments."""
    args = parse_args(argv)
    try:
        report = json.loads(args.json_file.read_text())
        if args.distributions:
            plot_distributions(
                report, args.output, family=args.family, dpi=args.dpi, bins=args.bins,
                converged_only=args.converged_only, volume_scale=args.volume_scale,
            )
        else:
            plot_report(report, args.output, family=args.family, dpi=args.dpi,
                        include_r_sec=args.include_r_sec)
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
