import demes
import numpy as np
import matplotlib

from demesdraw import utils


def size_history(
    graph: demes.Graph,
    ax: matplotlib.axes.Axes = None,
    inf_ratio: float = 0.1,
    inf_label: bool = False,
    invert_x: bool = False,
    annotate_epochs: bool = False,
    colours: utils.ColourOrColourMapping = None,
    log_time: bool = False,
    log_size: bool = False,
    title: str = None,
    num_points: int = 100,
) -> matplotlib.axes.Axes:
    """
    Plot population size as a function of time for each deme in the graph.

    :param demes.Graph graph: The demes graph to plot.
    :param matplotlib.axes.Axes ax: The matplotlib axes onto which the figure
        will be drawn. If None, an empty axes will be created for the figure.
    :param float inf_ratio: The proportion of the time axis that will be
        used for the time interval which stretches towards infinity.
    :param bool inf_label: Write "inf" by the arrow that points towards infinity.
    :param bool invert_x: If true, the horizontal axis will have infinity
        on the left and zero on the right, and the vertical axis will be drawn
        on the right.
    :param bool annotate_epochs: Annotate the figure with epoch indices
        over the relevant parts of the lines. This is mostly useful as a
        pedagogical tool.
    :param colours: A mapping from deme name to matplotlib colour. Alternately,
        ``colours`` may be a named colour that will be used for all demes.
    :type colours: dict or str
    :param bool log_time: Use a log-10 scale for the time axis.
    :param bool log_size: Use a log-10 scale for the deme size axis.
    :param str title: The title of the figure.

    :return: The matplotlib axes onto which the figure was drawn.
    :rtype: matplotlib.axes.Axes
    """
    ax = utils.get_axes(ax)

    if invert_x:
        arrowhead = "<k"
    else:
        arrowhead = ">k"

    colours = utils.get_colours(graph, colours)
    inf_start_time = utils.inf_start_time(graph, inf_ratio, log_time)

    linestyles = ["solid"]  # , "dashed", "dashdot"]
    linewidths = [2, 4, 8, 1]
    legend_handles = []
    # Top of the z order stacking.
    z_top = 1 + len(graph.demes) + max(linewidths)

    for j, deme in enumerate(graph.demes):
        colour = colours[deme.name]
        linestyle = linestyles[j % len(linestyles)]
        linewidth = linewidths[j % len(linewidths)]
        plot_kwargs = dict(
            color=colour,
            linestyle=linestyle,
            linewidth=linewidth,
            label=deme.name,
            alpha=0.7,
            zorder=z_top - linewidth,
            capstyle="butt",
            fill=False,
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=3, foreground="white")
            ],
        )
        discontinuity_kwargs = plot_kwargs.copy()
        discontinuity_kwargs.update(linestyle=":")
        legend_kwargs = plot_kwargs.copy()
        legend_kwargs.pop("fill")
        # Line2D and Patch use different keywords for the capstyle. *Sigh*
        legend_kwargs.update(solid_capstyle=legend_kwargs.pop("capstyle"))

        legend_handles.append(matplotlib.lines.Line2D([], [], **legend_kwargs))

        # Path for the main line (solid).
        vertices_main = []
        codes_main = []
        # Path for the discontinuity lines (dashed).
        vertices_discontinuity = []
        codes_discontinuity = []

        for k, epoch in enumerate(deme.epochs):
            start_time = epoch.start_time
            if np.isinf(start_time):
                start_time = inf_start_time
            end_time = epoch.end_time
            if log_time:
                end_time = max(1, end_time)

            if epoch.size_function == "constant":
                x = np.array([start_time, end_time])
                y = np.array([epoch.start_size, epoch.end_size])
            elif epoch.size_function == "exponential":
                x = np.linspace(start_time, end_time, num=num_points)
                dt = np.linspace(0, 1, num=num_points)
                r = np.log(epoch.end_size / epoch.start_size)
                y = epoch.start_size * np.exp(r * dt)
            else:
                raise ValueError(
                    f"Don't know how to plot epoch {k} with "
                    f'"{epoch.size_function}" size_function.'
                )

            vertices_main.extend(list(zip(x, y)))
            if k == 0 or deme.epochs[k - 1].end_size != epoch.start_size:
                codes_main.append(matplotlib.path.Path.MOVETO)
            else:
                codes_main.append(matplotlib.path.Path.LINETO)
            codes_main.extend([matplotlib.path.Path.LINETO] * (len(x) - 1))

            if k > 0 and deme.epochs[k - 1].end_size != epoch.start_size:
                # Size discontinuity.
                vertices_discontinuity.extend(
                    [
                        (deme.epochs[k - 1].end_time, deme.epochs[k - 1].end_size),
                        (epoch.start_time, epoch.start_size),
                    ]
                )
                codes_discontinuity.extend(
                    [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO]
                )

            if annotate_epochs:
                if log_time:
                    text_x = np.exp((np.log(start_time) + np.log(end_time)) / 2)
                else:
                    text_x = (start_time + end_time) / 2
                if log_size:
                    text_y = np.exp(
                        (np.log(epoch.start_size) + np.log(max(1, epoch.end_size))) / 2
                    )
                else:
                    text_y = (epoch.start_size + epoch.end_size) / 2
                ax.annotate(
                    f"epoch {k}",
                    (text_x, text_y),
                    ha="center",
                    va="bottom",
                    xytext=(0, 4 + linewidth / 2),  # vertical offset
                    textcoords="offset points",
                    # Give the text some contrast with its background.
                    bbox=dict(
                        boxstyle="round", fc="white", ec="none", alpha=0.6, pad=0
                    ),
                    # This is only really a useful feature with 1 deme,
                    # but at least try to do something reasonable for more demes.
                    color="black" if len(graph.demes) == 1 else colour,
                    zorder=z_top,
                )

        # Indicate population size discontinuities from ancestor demes.
        for ancestor_id in deme.ancestors:
            anc = graph[ancestor_id]
            anc_N = utils.size_of_deme_at_time(anc, deme.start_time)
            deme_N = deme.epochs[0].start_size
            if anc_N != deme_N:
                vertices_discontinuity.extend(
                    [(deme.start_time, anc_N), (deme.start_time, deme_N)]
                )
                codes_discontinuity.extend(
                    [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO]
                )

        size_path_patch = matplotlib.patches.PathPatch(
            matplotlib.path.Path(vertices_main, codes_main), **plot_kwargs
        )
        ax.add_patch(size_path_patch)
        if len(vertices_discontinuity) > 0:
            discontinuity_path_patch = matplotlib.patches.PathPatch(
                matplotlib.path.Path(vertices_discontinuity, codes_discontinuity),
                **discontinuity_kwargs,
            )
            ax.add_patch(discontinuity_path_patch)

        if np.isinf(deme.start_time):
            # Plot an arrow at the end of the line, to indicate this
            # line extends towards infinity.
            ax.plot(
                inf_start_time,
                deme.epochs[0].start_size,
                arrowhead,
                color=colour,
                clip_on=False,
                zorder=z_top,
            )
            if inf_label:
                ax.annotate(
                    "inf",
                    (inf_start_time, deme.epochs[0].start_size),
                    xytext=(0, -6),  # vertical offset
                    textcoords="offset points",
                    clip_on=False,
                    ha="center",
                    va="top",
                )

    # Update the axes view. ax.add_patch() doesn't do this itself.
    ax.autoscale_view()

    if len(graph.demes) > 1:
        leg = ax.legend(handles=legend_handles, ncol=len(graph.demes) // 2)
        leg.set_zorder(z_top)

    if title is not None:
        ax.set_title(title)

    # Arrange the axes spines, ticks and labels.

    ax.set_xlim(1 if log_time else 0, inf_start_time)
    if not log_size:
        ax.set_ylim(bottom=0)

    for spine in ax.spines.values():
        spine.set_zorder(z_top)

    ax.spines["top"].set_visible(False)
    if invert_x:
        ax.spines["left"].set_visible(False)
        ax.invert_xaxis()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        ax.spines["right"].set_visible(False)

    ax.set_xlabel(f"time ago ({graph.time_units})")
    # ax.set_ylabel("N", rotation=0, ha="left" if invert_x else "right")
    ax.set_ylabel("deme\nsize", rotation=0, labelpad=20)

    if log_time:
        ax.set_xscale("log", base=10)
    if log_size:
        ax.set_yscale("log", base=10)

    return ax
