import sys
import logging
import itertools

import daiquiri
import numpy as np
import demes
import demesdraw
import msprime
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_steps(dbg):
    last_N = max(dbg.population_size_history[:, dbg.num_epochs - 1])
    last_epoch = dbg.epoch_start_time[-1]
    steps = sorted(
        list(
            set(np.linspace(0, last_epoch + 12 * last_N, 1001)).union(
                set(dbg.epoch_start_time)
            )
        )
    )
    return steps


def coalescence_rate_trajectories(graph, leaves=None):
    """
    Return a matrix of coalecence times between pairs of populations.
    """
    if leaves is None:
        leaves = [deme.name for deme in graph.demes if deme.end_time == 0]
    assert len(leaves) > 0
    dbg = msprime.Demography.from_demes(graph).debug()
    steps = get_steps(dbg)
    crt = dict()
    for j, a in enumerate(leaves):
        for k, b in enumerate(leaves[j:], j):
            lineages = {a: 1, b: 1} if a != b else {a: 2}
            logger.debug(f"Getting coalescence_rate_trajectory: {a} and {b}")
            crt[(a, b)] = dbg.coalescence_rate_trajectory(
                steps, lineages, double_step_validation=False
            )
    return steps, crt


def get_axes(aspect=9 / 16, scale=1.5, **subplot_kwargs):
    """Make a matplotlib axes."""
    figsize = scale * plt.figaspect(aspect)
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    fig.set_tight_layout(True)
    return fig, ax


def get_line_plot_styles():
    linestyles = ["solid", "dashed", "dashdot"]
    linewidths = [1, 2, 4, 8]
    path_effects_lists = [
        [matplotlib.patheffects.withStroke(linewidth=2, foreground="white", alpha=0.5)],
        [matplotlib.patheffects.withStroke(linewidth=3, foreground="white", alpha=0.5)],
        [matplotlib.patheffects.withStroke(linewidth=5, foreground="white", alpha=0.5)],
        [],
    ]
    z_top = 1000  # Top of the z order stacking.
    z_adjust = dict(solid=-2, dashed=0, dashdot=-1)
    return (
        dict(
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=z_top - 10 * linewidth + z_adjust[linestyle],
            alpha=0.7,
            solid_capstyle="butt",
            path_effects=path_effects,
        )
        for linestyle, linewidth, path_effects in zip(
            *map(itertools.cycle, (linestyles, linewidths, path_effects_lists))
        )
    )


def plot_figure(graph, steps, crt):
    fig, axs = get_axes(nrows=2, ncols=2, gridspec_kw=dict(width_ratios=[1, 1]))

    # axs[1, 0].set_axis_off()
    ax_tubes = axs[0, 0]
    ax_cr1 = axs[0, 1]
    ax_cr2 = axs[1, 1]
    ax_cp = axs[1, 0]

    w = 1.3 * demesdraw.utils.size_max(graph)
    positions = dict(C=0, B=w, D=2 * w, A=3 * w)
    demesdraw.tubes(graph, ax=ax_tubes, log_time=True, positions=positions)
    # for tick in ax_tubes.get_xticklabels():
    #    tick.set_rotation(45)

    style_cr1 = get_line_plot_styles()
    style_cr2 = get_line_plot_styles()
    style_cp = get_line_plot_styles()

    for (a, b), (r, p) in crt.items():
        ax_cr = ax_cr1 if a == b else ax_cr2
        label = a if a == b else f"{a}/{b}"
        style_cr = style_cr1 if a == b else style_cr2
        ax_cr.plot(steps, r, label=label, **next(style_cr))
        ax_cp.plot(steps, p, label=label, **next(style_cp))

    ax_cr1.set_title("Coalescence rate")
    ax_cr2.set_title("Cross-coalescence rate")
    ax_cp.set_title("Probability of not having coalesced")
    for ax in (ax_cr1, ax_cr2):
        ax.set_ylabel("rate of coalescence")
    ax_cp.set_ylabel("Pr[not coalecesed]")
    for ax in (ax_cr1, ax_cr2, ax_cp):
        ax.set_xlabel("time ago (generations)")
        ax.legend()
        # ax.set_xscale("log")
        ax.set_yscale("log")

    return fig


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} model.yaml")
        exit(1)

    input_file = sys.argv[1]
    graph = demes.load(input_file)

    # daiquiri.setup(level="DEBUG")
    steps, crt = coalescence_rate_trajectories(graph, ["A", "C"])
    # daiquiri.setup(level="WARNING")

    fig = plot_figure(graph, steps, crt)
    fig.savefig("/tmp/crt.pdf")
