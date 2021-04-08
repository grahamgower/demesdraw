from typing import Any, Dict, Iterable, Mapping, List, Tuple
import itertools
import math

import demes
import numpy as np
import scipy.optimize
import matplotlib
import matplotlib.patheffects

from demesdraw import utils


class Tube:
    """
    A deme represented as a tube. The tube has a length along the time
    dimension and a width equal to the deme's population size, which may
    change over time.

    :ivar time: Coordinates along the time dimension.
    :type time: list of float
    :ivar size1: Coordinates along the non-time dimension, corresponding to the
        first side of the tube.
    :type size1: list of float
    :ivar size2: Coordinates along the non-time dimension, corresponding to the
        second side of the tube.
    :type size2: list of float
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
        N = utils.size_of_deme_at_time(self.deme, time)
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


def coexist(deme_j: demes.Deme, deme_k: demes.Deme) -> bool:
    """Returns true if deme_j and deme_k exist simultaneously."""
    return not (
        deme_j.end_time >= deme_k.start_time or deme_k.end_time >= deme_j.start_time
    )


def coexistence_indexes(graph: demes.Graph) -> List[Tuple[int, int]]:
    """Pairs of indices of demes that exist simultaneously."""
    contemporaries = []
    for j, deme_j in enumerate(graph.demes):
        for k, deme_k in enumerate(graph.demes[j + 1 :], j + 1):
            if coexist(deme_j, deme_k):
                contemporaries.append((j, k))
    return contemporaries


def successors_indexes(graph: demes.Graph) -> Mapping[int, List[int]]:
    """Graph successors, but use indexes rather than deme names"""
    idx = {deme.name: j for j, deme in enumerate(graph.demes)}
    successors = dict()
    for deme, children in graph.successors().items():
        if len(children) > 0:
            successors[idx[deme]] = [idx[child] for child in children]
    return successors


def interactions_indexes(graph: demes.Graph, *, unique: bool) -> List[Tuple[int, int]]:
    """Pairs of indices of demes that exchange migrants (migrations or pulses)."""
    idx = {deme.name: j for j, deme in enumerate(graph.demes)}
    interactions = []
    for migration in graph.migrations:
        if isinstance(migration, demes.AsymmetricMigration):
            interactions.append((idx[migration.source], idx[migration.dest]))
        else:
            for source, dest in itertools.permutations(
                migration.demes, 2  # type:ignore  # noqa
            ):
                interactions.append((idx[source], idx[dest]))
    for pulse in graph.pulses:
        interactions.append((idx[pulse.source], idx[pulse.dest]))

    if unique:
        # Remove duplicates.
        i2 = set()
        for a, b in interactions:
            if (b, a) in i2:
                continue
            i2.add((a, b))
        interactions = list(i2)

    return interactions


def topdown_placement(graph: demes.Graph) -> Mapping[str, int]:
    """
    Assign integer positions to demes by traversing the graph top down,
    avoiding positions already given to contemporary demes.
    """
    positions = {graph.demes[0].name: 0}
    for deme in graph.demes[1:]:
        taken = set()
        for other, pos in positions.items():
            if coexist(deme, graph[other]):
                taken.add(pos)
        pos = 0
        while pos in taken:
            pos += 1
        positions[deme.name] = pos
    return positions


def find_positions(
    graph: demes.Graph, sep: float, rounds: int = None, seed: int = None
) -> Mapping[str, float]:
    """
    Find optimal positions for the demes along a single dimension by minimising:

      - the distance from each parent deme to the mean position of its children,
      - the distance between interacting demes (where interactions are either
        migrations or pulses),
      - the distance from zero.

    In addition, the solution is constrained so that contemporary demes
    have a minimum separation distance, ``sep``.

    :param demes.Graph graph: The graph for which positions should be obtained.
    :param float sep: The minimum separation distance between contemporary demes.
    :param int rounds: Number of rounds of optimisation to perform.
    :param int seed: Seed for the random number generator.

    :return: A dictionary mapping deme names to positions.
    :rtype: dict
    """
    if rounds is None:
        if len(graph.demes) <= 5:
            # explore all orderings of 5 demes
            rounds = 120
        else:
            # just use topdown placement
            rounds = 0

    contemporaries = coexistence_indexes(graph)
    if len(contemporaries) == 0:
        # There are no constraints, so stack demes on top of each other.
        return {deme.name: 0 for deme in graph.demes}
    successors = successors_indexes(graph)
    interactions = interactions_indexes(graph, unique=True)

    def fseparation(x: np.ndarray) -> np.ndarray:
        """The separation distance between coexisting demes."""
        return np.array([np.abs(x[j] - x[k]) for j, k in contemporaries])

    def fmin(x: np.ndarray) -> float:
        """Function to be minimised."""
        z = 0
        # Minimise the distance from each deme to its children.
        for parent, children in successors.items():
            a = x[parent]
            b = np.mean([x[child] for child in children])
            z += (a - b) ** 2
            # z += sum((a - x[child]) ** 2 for child in children)
        # Minimise the distance between interacting demes.
        # (either migrations or pulses).
        z += sum((x[j] - x[k]) ** 2 for j, k in interactions)
        # Also penalise large positions.
        z += sum(x)
        return z

    topdown_positions = topdown_placement(graph)
    topdown = np.array(list(topdown_positions.values())) * sep
    x0 = topdown
    fmin_best = fmin(x0)
    x_best = x0.copy()

    def initial_states(rounds: int) -> Iterable[Any]:
        """Generate initial states for the optimisation procedure."""
        # Try the canonical top-down ordering.
        yield topdown

        n = len(graph.demes)
        if math.factorial(n) <= rounds:
            # generate all permutations
            yield from itertools.permutations(x0, n)
        else:
            rng = np.random.default_rng(seed)
            for _ in range(rounds):
                x = x0.copy()
                rng.shuffle(x)
                yield x

    # We optimise with the "SLSQP" method, which quickly converges to a
    # local solution, then repeat for many distinct starting positions.
    # This seems to work better than using the slower "trust-constr"
    # constrained-optimisation method.
    for x in initial_states(rounds):
        res = scipy.optimize.minimize(
            fmin,
            x,
            method="SLSQP",
            bounds=scipy.optimize.Bounds(0, np.inf),
            constraints=scipy.optimize.NonlinearConstraint(
                fseparation, lb=sep, ub=np.inf
            ),
        )
        if res.success and res.fun < fmin_best:
            x_best = res.x
            fmin_best = res.fun

    return {deme.name: position for deme, position in zip(graph.demes, x_best)}


def tubes(
    graph: demes.Graph,
    ax: matplotlib.axes.Axes = None,
    colours: utils.ColourOrColourMapping = None,
    log_time: bool = False,
    title: str = None,
    inf_ratio: float = 0.2,
    positions: Mapping[str, float] = None,
    num_lines_per_migration: int = 10,
    seed: int = None,
    optimisation_rounds: int = None,
    # TODO: docstring
    labels: str = "xticks-mid",
    fill: bool = True,
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
    The time of each migration line is drawn uniformly at
    random from the migration's time interval (or log-uniformly for a
    log-scaled time axis). Symmetric migrations have lines in both directions.

    We encourage users to manually specify the horizontal ``positions``
    for demes. If not specified, the positions will be chosen automatically,
    but these positions may be unexpected or otherwise non-optimal.

    :param demes.Graph graph: The demes graph to plot.
    :param matplotlib.axes.Axes ax: The matplotlib axes onto which the figure
        will be drawn. If None, an empty axes will be created for the figure.
    :param colours: A mapping from deme name to matplotlib colour. Alternately,
        ``colours`` may be a named colour that will be used for all demes.
    :type colours: dict or str
    :param bool log_time: Use a log-10 scale for the time axis.
    :param str title: The title of the figure.
    :param float inf_ratio: The proportion of the time axis that will be
        used for the time interval which stretches towards infinity.
    :param dict positions: A dictionary mapping deme names to horizontal
        coordinates. Note that the width of a deme is the deme's (max) size,
        so the positions should allow sufficient space to avoid overlapping.
    :param int num_lines_per_migration: The number of lines to draw per
        migration. For symmetric migrations, this number of lines will be
        drawn in each direction.
    :param int optimisation_rounds: Number of rounds of optimisation to perform
        when searching for reasonable horizontal positions for the demes.
        We use an optimisation method that quickly converges to a local
        solution, but repeat the procedure with many distinct starting
        positions. The layout for graphs with many demes may benefit from
        increasing this parameter.
    :param int seed: Seed for the random number generator. The generator is
        used to draw times for migration lines and during the optimisation
        procedure used to determine deme positions on the horizontal axis.

    :return: The matplotlib axes onto which the figure was drawn.
    :rtype: matplotlib.axes.Axes
    """
    if labels not in ("xticks", "legend", "mid", "xticks-legend", "xticks-mid", None):
        raise ValueError(f"Unexpected value for labels: '{labels}'")

    ax = utils.get_axes(ax)

    if log_time:
        ax.set_yscale("log", base=10)

    colours = utils.get_colours(graph, colours)

    rng = np.random.default_rng(seed)
    seed2 = rng.integers(2 ** 63)
    inf_start_time = utils.inf_start_time(graph, inf_ratio, log_time)

    size_max = max(
        max(epoch.start_size, epoch.end_size)
        for deme in graph.demes
        for epoch in deme.epochs
    )
    if positions is None:
        positions = find_positions(
            graph, size_max * 1.1, rounds=optimisation_rounds, seed=seed2
        )

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
            ax.fill_betweenx(
                tube.time,
                tube.size1,
                tube.size2,
                facecolor=colour,
                edgecolor="none",
                alpha=0.5,
                zorder=1,
            )

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
        for ancestor_id in deme.ancestors:
            anc_size1, anc_size2 = tubes[ancestor_id].sizes_at(deme.start_time)
            if anc_size1 + anc_size2 < tube.size1[0] + tube.size2[0]:
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
            else:
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
        ax.add_patch(arr)

    # Update the axes view. ax.add_patch() doesn't do this itself.
    ax.autoscale_view()

    # Calculate an offset for the space between arrowhead and tube.
    xlim = ax.get_xlim()
    offset = 0.01 * (xlim[1] - xlim[0])

    def migration_line(source, dest, time, **mig_kwargs):
        """Draw a migration line from source to dest at the given time."""
        source_size1, source_size2 = tubes[source].sizes_at(time)
        dest_size1, dest_size2 = tubes[dest].sizes_at(time)

        if source_size2 < dest_size1:
            x = [source_size2 + offset, dest_size1 - offset]
            arrow = ">k"
        else:
            x = [source_size1 - offset, dest_size2 + offset]
            arrow = "<k"

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
        ax.plot(
            x[1],
            time,
            arrow,
            markersize=2,
            color=colour,
            path_effects=[matplotlib.patheffects.Normal()],
            **mig_kwargs,
        )

    def random_migration_time(migration, log_scale: bool) -> float:
        start_time = migration.start_time
        if np.isinf(start_time):
            start_time = inf_start_time
        if log_scale:
            t = np.exp(
                rng.uniform(np.log(start_time), np.log(max(1, migration.end_time)))
            )
        else:
            t = rng.uniform(start_time, migration.end_time)
        return t

    # Plot migration lines.
    migration_kwargs = dict(linewidth=0.2, alpha=0.5)
    for migration in graph.migrations:
        if isinstance(migration, demes.AsymmetricMigration):
            for _ in range(num_lines_per_migration):
                t = random_migration_time(migration, log_time)
                migration_line(migration.source, migration.dest, t, **migration_kwargs)
        else:
            for a, b in itertools.permutations(migration.demes, 2):  # type: ignore  # noqa
                for _ in range(num_lines_per_migration):
                    t = random_migration_time(migration, log_time)
                    migration_line(a, b, t, **migration_kwargs)

    # Plot pulse lines.
    for pulse in graph.pulses:
        migration_line(
            pulse.source, pulse.dest, pulse.time, linestyle="--", linewidth=1
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
            ncol=len(deme_labels) // 2,
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
    ax.tick_params("x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_ylabel(f"time ago ({graph.time_units})")

    ax.set_ylim(1 if log_time else 0, inf_start_time)

    return ax
