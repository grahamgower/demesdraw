import itertools

import daiquiri
import numpy as np
import demes
import demesdraw
import msprime
import matplotlib
import matplotlib.pyplot as plt


def get_steps2(dbg):
    last_N = max(dbg.population_size_history[:, dbg.num_epochs - 1])
    last_epoch = dbg.epoch_start_time[-1]
    times = list(dbg.epoch_start_time) + [last_epoch + 12 * last_N]
    steps = set()
    for a, b in zip(times[:-1], times[1:]):
        steps.update(np.linspace(a, b, 101))
    steps = np.array(sorted(steps))
    return steps


def get_steps(dbg):
    # Get initial steps. Copied from mean_coalescence_time().
    last_N = max(dbg.population_size_history[:, dbg.num_epochs - 1])
    last_epoch = dbg.epoch_start_time[-1]
    steps = sorted(
        list(
            set(np.linspace(0, last_epoch + 12 * last_N, 101)).union(
                set(dbg.epoch_start_time)
            )
        )
    )
    return steps


def refine_steps(steps):
    # double the number of steps
    inter = steps[:-1] + np.diff(steps) / 2
    steps = np.concatenate([steps, inter])
    steps.sort()
    return steps


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
        [matplotlib.patheffects.withStroke(linewidth=2, foreground="white", alpha=0.7)],
        [matplotlib.patheffects.withStroke(linewidth=3, foreground="white", alpha=0.7)],
        [matplotlib.patheffects.withStroke(linewidth=5, foreground="white", alpha=0.7)],
        [],
    ]
    z_top = 1000  # Top of the z order stacking.
    return (
        dict(
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=z_top - linewidth,
            alpha=0.7,
            solid_capstyle="butt",
            path_effects=path_effects,
        )
        for linestyle, linewidth, path_effects in zip(
            *map(itertools.cycle, (linestyles, linewidths, path_effects_lists))
        )
    )


def plot_figure(graph):
    fig, axs = get_axes(nrows=2, ncols=2, gridspec_kw=dict(width_ratios=[1, 2]))

    axs[1, 0].set_axis_off()
    ax_tubes = axs[0, 0]
    ax_cr = axs[0, 1]
    ax_cp = axs[1, 1]

    w = 1.3 * demesdraw.utils.size_max(graph)
    positions = dict(C=0, B=w, D=2 * w, A=3 * w)
    demesdraw.tubes(graph, ax=ax_tubes, log_time=True, positions=positions)

    style_cr = get_line_plot_styles()
    style_cp = get_line_plot_styles()

    dbg = msprime.Demography.from_demes(graph).debug()
    steps = get_steps(dbg)

    for _ in range(4):
        r, p = dbg.coalescence_rate_trajectory(
            steps, lineages=dict(A=1, C=1), double_step_validation=False
        )
        assert np.all(p[:-1] >= p[1:])
        ax_cr.plot(steps, r, label=f"{len(steps)}", **next(style_cr))
        ax_cp.plot(steps, p, label=f"{len(steps)}", **next(style_cp))
        steps = refine_steps(steps)

    ax_cr.set_title("coalescence rate (lineages: A=1, C=1")
    ax_cp.set_title("Pr{A and C not coalesced}")
    ax_cr.set_ylabel("rate")
    ax_cp.set_ylabel("probability")
    for ax in (ax_cr, ax_cp):
        ax.set_xlabel("time ago (generations)")
        ax.legend(title="len(steps)")

    return fig


test_case = """\
time_units: generations
defaults:
  epoch: {start_size: 1000}
demes:
- name: A
- name: B
  ancestors: [A]
  start_time: 6000
- name: C
  ancestors: [B]
  start_time: 2000
- name: D
  ancestors: [C]
  start_time: 1000
migrations:
- demes: [A, D]
  rate: 1e-5
"""

graph = demes.loads(test_case)
fig = plot_figure(graph)
fig.savefig("/tmp/cr.pdf", dpi=200)

dbg = msprime.Demography.from_demes(graph).debug()
# steps = get_steps(dbg)
# steps = refine_steps(steps)
# steps = refine_steps(steps)

daiquiri.setup(level="DEBUG")

# r, p = dbg.coalescence_rate_trajectory(
#    steps, lineages=dict(A=1, C=1), double_step_validation=False
# )
# print(steps)
# print(r)
# print(p)

t = dbg.mean_coalescence_time({"A": 1, "C": 1}, steps=get_steps(dbg))
