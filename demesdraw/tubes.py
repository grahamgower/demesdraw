from __future__ import annotations
from typing import Dict, Mapping, List, Tuple

import demes
import numpy as np
import matplotlib
import matplotlib.patheffects

from . import utils


class Tube:
    """
    A deme represented as a tube. The tube has a length along the time
    dimension and a width equal to the deme's population size, which may
    change over time.

    :ivar list[float] time: Coordinates along the time dimension.
    :ivar list[float] size1: Coordinates along the non-time dimension,
        corresponding to the first side of the tube.
    :ivar list[float] size2: Coordinates along the non-time dimension,
        corresponding to the second side of the tube.
    """

    def __init__(
        self,
        deme: demes.Deme,
        mid: float,
        inf_start_time: float,
        log_time: bool = False,
    ):
        """
        :param demes.Deme deme: The deme for which to calculate coordinates.
        :param float mid: The mid point of the deme along the non-time dimension.
        :param float inf_start_time: The value along the time dimension which
            is used instead of infinity (for epochs with infinite start times).
        :param bool log_time: The time axis uses a log-10 scale.
        """
        self.deme = deme
        self.mid = mid
        self._coords(deme, mid, inf_start_time, log_time)

    def _coords(
        self,
        deme: demes.Deme,
        mid: float,
        inf_start_time: float,
        log_time: bool,
        num_points: int = 100,
    ) -> None:
        """Calculate tube coordinates."""
        time: List[float] = []
        size1: List[float] = []
        size2: List[float] = []
        for k, epoch in enumerate(deme.epochs):
            start_time = epoch.start_time
            if np.isinf(start_time):
                start_time = inf_start_time
            end_time = epoch.end_time

            if epoch.size_function == "constant":
                t = np.array([start_time, end_time])
                N1 = [mid - epoch.start_size / 2] * 2
                N2 = [mid + epoch.end_size / 2] * 2
            elif epoch.size_function == "exponential":
                if log_time:
                    t = np.exp(
                        np.linspace(
                            np.log(start_time), np.log(max(1, end_time)), num=num_points
                        )
                    )
                else:
                    t = np.linspace(start_time, end_time, num=num_points)
                dt = (start_time - t) / (start_time - end_time)
                r = np.log(epoch.end_size / epoch.start_size)
                N = epoch.start_size * np.exp(r * dt)
                N1 = mid - N / 2
                N2 = mid + N / 2
            elif epoch.size_function == "linear":
                if log_time:
                    t = np.exp(
                        np.linspace(
                            np.log(start_time), np.log(max(1, end_time)), num=num_points
                        )
                    )
                else:
                    t = np.linspace(start_time, end_time, num=num_points)
                dt = (start_time - t) / (start_time - end_time)
                N = epoch.start_size + (epoch.end_size - epoch.start_size) * dt
                N1 = mid - N / 2
                N2 = mid + N / 2
            else:
                raise ValueError(
                    f"Don't know how to draw epoch {k} with "
                    f'"{epoch.size_function}" size_function.'
                )

            time.extend(t)
            size1.extend(N1)
            size2.extend(N2)

        self.time = time
        self.size1 = size1
        self.size2 = size2

    def sizes_at(self, time: float) -> Tuple[float, float]:
        """Return the size coordinates of the tube at the given time."""
        N = self.deme.size_at(time)
        N1 = self.mid - N / 2
        N2 = self.mid + N / 2
        return N1, N2

    def to_path(self) -> matplotlib.path.Path:
        """Return a matplotlib.path.Path for the tube outline."""
        # Compared with separately drawing each side of the tube,
        # drawing a path that includes both sides of the tube means:
        #  * vector formats use one "stroke" rather than two; and
        #  * matplotlib path effects apply once, avoiding artifacts for
        #    very narrow tubes, where the path effect from the side
        #    drawn second is on top of the line drawn for the first side.
        N = list(self.size1) + list(self.size2)
        t = list(self.time) * 2
        vertices = list(zip(N, t))
        codes = (
            [matplotlib.path.Path.MOVETO]
            + [matplotlib.path.Path.LINETO] * (len(self.time) - 1)
        ) * 2
        return matplotlib.path.Path(vertices, codes)


def find_positions(graph: demes.Graph) -> Dict[str, float]:
    """
    Find optimal deme positions along a single dimension by:

      - ordering the demes to minimise the number of lines crossing demes,
      - minimising the distance from each parent deme to the mean position of
        its children,
      - minimising the distance between interacting demes (where interactions
        are either migrations or pulses).

    In addition, the solution is constrained so that contemporary demes
    have a minimum separation distance.

    :param graph:
        The graph for which positions should be obtained.
    :return:
        A dictionary mapping deme names to positions.
    """
    sep = utils.separation_heuristic(graph)
    positions = utils.minimal_crossing_positions(
        graph, sep=sep, unique_interactions=False
    )
    return utils.optimise_positions(
        graph, positions=positions, sep=sep, unique_interactions=False
    )


def tubes(
    graph: demes.Graph,
    ax: matplotlib.axes.Axes | None = None,
    colours: utils.ColourOrColourMapping | None = None,
    log_time: bool = False,
    title: str | None = None,
    inf_ratio: float = 0.2,
    positions: Mapping[str, float] | None = None,
    num_lines_per_migration: int = 10,
    seed: int | None = None,
    max_time: float | None = None,
    labels: str = "xticks-mid",
    fill: bool = True,
    scale_bar: bool = False,
) -> matplotlib.axes.Axes:
    """
    Plot a demes-as-tubes schematic of the graph and the demes' relationships.

    Each deme is depicted as a tube, where the tube’s width is
    proportional to the deme’s size at any given time.
    Horizontal lines with arrows indicate either:

     * an ancestor/descendant relation (thick solid lines, open arrow heads),
     * an admixture pulse (dashed lines, closed arrow heads),
     * or a period of continuous migration (thin solid lines, closed arrow heads).

    Lines are drawn in the colour of the ancestor or source deme, and arrows
    point from ancestor to descendant or from source to dest.
    For each period of continuous migration, multiple thin lines are drawn.
    The time for each migration line is sampled uniformly at
    random from the migration's time interval (or log-uniformly for a
    log-scaled time axis). Symmetric migrations have lines in both directions.

    If ``positions`` are not specified, the positions will be chosen
    automatically such that line crossings are minimised and related demes
    are close together. If the automatically chosen positions are unexpected,
    please open an issue at https://github.com/grahamgower/demesdraw/issues/.

    :param demes.Graph graph:
        The demes graph to plot.
    :param ax:
        The matplotlib axes onto which the figure
        will be drawn. If None, an empty axes will be created for the figure.
    :type ax: Optional[matplotlib.axes.Axes]
    :param colours:
        A mapping from deme name to matplotlib colour. Alternately,
        ``colours`` may be a named colour that will be used for all demes.
    :type colours: Optional[dict or str]
    :param log_time:
        If True, use a log-10 scale for the time axis.
        If False (*default*), a linear scale will be used.
    :param title:
        The title of the figure.
    :param inf_ratio:
        The proportion of the time axis that will be
        used for the time interval which stretches towards infinity.
    :param positions:
        A dictionary mapping deme names to horizontal coordinates
        (the mid point of the deme's tube).
        Note that the width of a deme is the deme's (max) size,
        so the positions should allow sufficient space to avoid overlapping.
    :param num_lines_per_migration:
        The number of lines to draw per migration. For symmetric migrations,
        this number of lines will be drawn in each direction.
    :param seed:
        Seed for the random number generator. The generator is
        used to sample times for migration lines.
    :param max_time:
        The maximum time value shown in the figure.
        If demographic events (e.g. size changes, migrations, common ancestor
        events) occur before this time, those events will not be visible.
        If no demographic events occur before this time, the root demes will be
        drawn so they extend to the given time.
    :param labels:
        A string indicating where the deme names should be drawn, or ``None``.
        The options are:

         * "xticks": for extant demes, labels are written under the x-axis
           as matplotlib xticklabels. Labels are not written for extinct demes.
         * "legend": labels for colour patches are written in a legend.
         * "mid": labels are written in the middle of each deme's tube.
         * "xticks-legend": for extant demes, labels are written under the
           x-axis as matplotlib xticklabels. For extinct demes, labels for
           colour patches are written in a legend.
         * "xticks-mid" (*default*): for extant demes, labels are written
           under the x-axis as matplotlib xticklabels. For extinct demes,
           labels are written in the middle of each deme's tube.
         * ``None``: no labels are written.

    :type labels: str or None
    :param fill:
        If True, the inside of the tubes will be painted.
        If False, only the outline of the tubes will be drawn.
    :param scale_bar:
        If True, draw a scale bar that indicates population size.
        If False (*default*), no scale bar will be shown.
    :return:
        The matplotlib axes onto which the figure was drawn.
    """
    if labels not in ("xticks", "legend", "mid", "xticks-legend", "xticks-mid", None):
        raise ValueError(f"Unexpected value for labels: '{labels}'")

    if ax is None:
        _, ax = utils.get_fig_axes()

    if log_time:
        ax.set_yscale("log", base=10)

    colours = utils._get_colours(graph, colours)

    rng = np.random.default_rng(seed)
    inf_start_time = max_time
    if inf_start_time is None:
        inf_start_time = utils._inf_start_time(graph, inf_ratio, log_time)

    if positions is None:
        positions = find_positions(graph)

    tubes = {}
    ancestry_arrows = []

    for j, deme in enumerate(graph.demes):
        colour = colours[deme.name]
        plot_kwargs = dict(
            color=colour,
            # solid_capstyle="butt",
            # capstyle="butt",
            zorder=1,
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=3, foreground="white")
            ],
        )

        mid = positions[deme.name]

        tube = Tube(deme, mid, inf_start_time, log_time=log_time)
        tubes[deme.name] = tube

        path_patch = matplotlib.patches.PathPatch(
            tube.to_path(), capstyle="butt", fill=False, **plot_kwargs
        )
        ax.add_patch(path_patch)
        if fill:
            poly = ax.fill_betweenx(
                tube.time,
                tube.size1,
                tube.size2,
                facecolor=colour,
                edgecolor="none",
                alpha=0.5,
                zorder=1,
            )
            poly._demesdraw_data = dict(deme=deme)

        # Indicate ancestry from ancestor demes.
        tube_frac = np.linspace(
            tube.size1[0], tube.size2[0], max(2, len(deme.ancestors))
        )
        left = 0
        right = 0
        arrow_kwargs = dict(
            arrowstyle="-|>",
            mutation_scale=10,
            alpha=1,
            facecolor=(1, 1, 1, 0),
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=3, foreground="white")
            ],
            zorder=2,
        )
        for ancestor_id, proportion in zip(deme.ancestors, deme.proportions):
            anc_size1, anc_size2 = tubes[ancestor_id].sizes_at(deme.start_time)
            if anc_size2 < tube.size1[0]:
                # Ancestor is to the left.
                x_pos = [anc_size2, tube_frac[left]]
                ancestry_arrows.append(
                    (
                        tube.time[0],
                        min(x_pos),
                        max(x_pos),
                        colours[ancestor_id],
                    )
                )
                left += 1
            elif tube.size2[0] < anc_size1:
                # Ancestor is to the right.
                x_pos = [anc_size1, tube_frac[len(tube_frac) - right - 1]]
                ancestry_arrows.append(
                    (
                        tube.time[0],
                        max(x_pos),
                        min(x_pos),
                        colours[ancestor_id],
                    )
                )
                right += 1

    slots: Dict[int, int] = {}
    for j, (time, x1, x2, colour) in enumerate(ancestry_arrows):
        taken = set()
        for k, slot in slots.items():
            k_time, k_x1, k_x2, _ = ancestry_arrows[k]
            if np.isclose(time, k_time) and not (
                min(x1, x2) > max(k_x1, k_x2) or min(k_x1, k_x2) > max(x1, x2)
            ):
                taken.add(slot)
        slot = 0
        i = 0
        while slot in taken:
            i += 1
            odd = i % 2
            delta = i // 2
            slot = -odd * delta + (1 - odd) * delta
        slots[j] = slot
        radius = slot * 0.15
        arr = matplotlib.patches.FancyArrowPatch(
            (x1, time),
            (x2, time),
            connectionstyle=f"arc3,rad={radius}",
            edgecolor=colour,
            **arrow_kwargs,
        )
        arr.set_sketch_params(1, 100, 2)
        # Ignore potential overflow when using log scale. Infinity is fine.
        with np.errstate(over="ignore"):
            ax.add_patch(arr)

    # Update the axes view. ax.add_patch() doesn't do this itself.
    ax.autoscale_view()

    # Calculate an offset for the space between arrowhead and tube.
    xlim = ax.get_xlim()
    offset = 0.01 * (xlim[1] - xlim[0])

    def migration_line(source, dest, time, metadata, **mig_kwargs):
        """Draw a migration line from source to dest at the given time."""
        source_size1, source_size2 = tubes[source].sizes_at(time)
        dest_size1, dest_size2 = tubes[dest].sizes_at(time)

        if source_size2 < dest_size1:
            x = [source_size2 + offset, dest_size1 - offset]
            arrow = ">"
            lr = "r"
        else:
            x = [source_size1 - offset, dest_size2 + offset]
            arrow = "<"
            lr = "l"

        metadata = dict(lr=lr, **metadata)

        colour = colours[source]
        lines = ax.plot(
            x,
            [time, time],
            color=colour,
            path_effects=[matplotlib.patheffects.Normal()],
            zorder=-1,
            **mig_kwargs,
        )
        for line in lines:
            line.set_sketch_params(1, 100, 2)
            line._demesdraw_data = metadata
        lines = ax.plot(
            x[1],
            time,
            arrow,
            markersize=2,
            color=colour,
            path_effects=[matplotlib.patheffects.Normal()],
            **mig_kwargs,
        )
        for line in lines:
            line._demesdraw_data = metadata

    def random_migration_time(migration, log_scale: bool) -> float:
        start_time = migration.start_time
        if np.isinf(start_time):
            start_time = inf_start_time
        if log_scale:
            t = np.exp(
                rng.uniform(np.log(max(1, migration.end_time)), np.log(start_time))
            )
        else:
            t = rng.uniform(migration.end_time, start_time)
        return t

    # Plot migration lines.
    migration_kwargs = dict(linewidth=0.2, alpha=0.5)
    for migration in graph.migrations:
        for _ in range(num_lines_per_migration):
            t = random_migration_time(migration, log_time)
            migration_line(
                migration.source,
                migration.dest,
                t,
                dict(migration=migration),
                **migration_kwargs,
            )

    # Plot pulse lines.
    for pulse in graph.pulses:
        for j, source in enumerate(pulse.sources):
            migration_line(
                source,
                pulse.dest,
                pulse.time,
                dict(pulse=pulse, source=j),
                linestyle="--",
                linewidth=1,
            )

    xticks = []
    xticklabels = []
    deme_labels = [deme.name for deme in graph.demes]

    if labels in ("xticks", "xticks-legend", "xticks-mid"):
        # Put labels for the leaf nodes underneath the figure.
        leaves = [deme.name for deme in graph.demes if deme.end_time == 0]
        deme_labels = [deme.name for deme in graph.demes if deme.end_time != 0]
        xticks = [positions[leaf] for leaf in leaves]
        xticklabels = leaves

    if len(deme_labels) > 0 and labels in ("legend", "xticks-legend"):
        # Add legend.
        ax.legend(
            handles=[
                matplotlib.patches.Patch(
                    edgecolor=matplotlib.colors.to_rgba(colours[deme_name], 1.0),
                    facecolor=matplotlib.colors.to_rgba(colours[deme_name], 0.3)
                    if fill
                    else None,
                    label=deme_name,
                )
                for deme_name in deme_labels
            ],
            # Use a horizontal layout, rather than vertical.
            ncol=max(1, len(deme_labels) // 2),
        )

    if labels in ("mid", "xticks-mid"):
        for deme_name in deme_labels:
            if log_time:
                tmid = np.exp(
                    (
                        np.log(tubes[deme_name].time[0])
                        + np.log(max(1, tubes[deme_name].time[-1]))
                    )
                    / 2
                )
            else:
                tmid = (tubes[deme_name].time[0] + tubes[deme_name].time[-1]) / 2
            ax.text(
                positions[deme_name],
                tmid,
                deme_name,
                ha="center",
                va="center",
                # Give the text some contrast with its background.
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.6, pad=0.2),
            )

    if title is not None:
        ax.set_title(title)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    ax.set_ylabel(f"time ago ({graph.time_units})")

    ax.set_ylim(1 if log_time else 0, inf_start_time)

    if scale_bar:
        # Plot a scale bar that indicates the population size.
        width = utils.size_max(graph)
        transform = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )
        x_offset = min(min(t.size1) for t in tubes.values())
        ax_sb = ax.inset_axes([x_offset, -0.16, width, 1e-6], transform=transform)

        ax_sb.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_sb.spines[["left", "right", "top"]].set_visible(False)
        ax_sb.xaxis.set_ticks_position("top")
        locator = matplotlib.ticker.AutoLocator()
        locator.set_params(integer=True)
        ax_sb.xaxis.set_major_locator(locator)
        ax_sb.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_sb.set_xlim(0, width)
        ax_sb.set_xlabel("deme size (individuals)")
        ax_sb.xaxis.set_label_position("bottom")

    # Status bar text when in interactive mode.
    def format_coord(x, y):
        return f"time={y:.1f}"

    ax.format_coord = format_coord

    # Note: we must hold a reference to the motion_notify_event callback,
    # or the callback will get garbage collected and thus disabled.
    # So we just add an attribute to the returned Axes object.
    ax._demesdraw_mouseoverpopup = _MouseoverPopup(ax.figure, ax)

    return ax


# Copied and modified from:
# https://matplotlib.org/stable/tutorials/advanced/blitting.html
class _BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        if hasattr(canvas, "copy_from_bbox") and hasattr(canvas, "restore_region"):
            # Webagg and related backends support blitting as of matplotlib 3.4,
            # but supports_blit is False because of some rendering artifacts.
            # However, not using blitting is very laggy, so we use it anyway.
            self.canvas.supports_blit = True
        if not self.canvas.supports_blit:
            return

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                return
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        assert art.figure == self.canvas.figure
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        if not self.canvas.supports_blit:
            self.canvas.draw_idle()
            return
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)


class _MouseoverPopup:
    def __init__(self, fig, ax):
        """Setup interactive mouseover annotations."""

        self.ax = ax
        self.popup = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            multialignment="left",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(boxstyle="round", fc="yellow"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.popup.set_visible(False)
        self.bm = _BlitManager(fig.canvas, [self.popup])

        # Detect mouseover for artists with a _demesdraw_data attribute,
        # which were added in demesdraw.tubes().
        self.artists = []
        for artist in reversed(ax.get_children()):
            if hasattr(artist, "_demesdraw_data"):
                self.artists.append(artist)

        # Get the midpoint of the figure in data coords.
        axis_to_data = self.ax.transAxes + self.ax.transData.inverted()
        self.xmid, self.ymid = axis_to_data.transform((0.5, 0.5))

        # Save a reference to the callback so that it doesn't get garbage collected.
        self.cid = fig.canvas.mpl_connect("motion_notify_event", self.motion_callback)

    def motion_callback(self, event):
        visible = self.popup.get_visible()
        mouse_inside_artist = False
        if event.inaxes == self.ax:
            for artist in self.artists:
                contained, _ = artist.contains(event)
                if contained:
                    data = artist._demesdraw_data
                    label = self.get_mouseover_label(data, event)
                    x = event.xdata
                    y = event.ydata
                    xt = 50
                    yt = 50
                    if x > self.xmid:
                        xt *= -1
                    if y > self.ymid:
                        yt *= -1
                    self.popup.xy = (x, y)
                    self.popup.xyann = (xt, yt)  # xytext
                    self.popup.set_text(label)
                    self.popup.set_visible(True)
                    mouse_inside_artist = True
                    self.bm.update()
                    break

        if not mouse_inside_artist and visible:
            self.popup.set_visible(False)
            self.bm.update()

    def get_arrow(self, lr, source, dest):
        assert lr in ("l", "r")
        arrow = r"$\rightarrow$" if lr == "r" else r"$\leftarrow$"
        if lr == "r":
            return f"{source} {arrow} {dest}"
        else:
            return f"{dest} {arrow} {source}"

    def get_mouseover_label(self, data, event):
        if "deme" in data:
            deme = data["deme"]
            time = max(min(event.ydata, deme.start_time - 1e-6), deme.end_time)
            size = deme.size_at(time)
            ancestors = ", ".join(deme.ancestors)
            proportions = ", ".join(map(str, deme.proportions))
            label = (
                f"deme: {deme.name}\n"
                f"time: ({deme.start_time}, {deme.end_time}]\n"
                f"ancestors: {ancestors}\n"
                f"proportions: {proportions}\n"
                f"size: {int(size)}"
            )
        elif "migration" in data:
            migration = data["migration"]
            arrow = self.get_arrow(data["lr"], migration.source, migration.dest)
            label = (
                f"migration: {arrow}\n"
                f"time: ({migration.start_time}, {migration.end_time}]\n"
                f"rate: {migration.rate}"
            )
        elif "pulse" in data:
            pulse = data["pulse"]
            j = data["source"]
            arrow = self.get_arrow(data["lr"], pulse.sources[j], pulse.dest)
            label = (
                f"pulse: {arrow}\n"
                f"time: {pulse.time}\n"
                f"proportion: {pulse.proportions[j]}"
            )
        return label
