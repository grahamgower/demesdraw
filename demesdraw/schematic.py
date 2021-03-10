from typing import List, Tuple, Dict
import itertools
import math

import demes
import numpy as np
import scipy.optimize
import matplotlib
import matplotlib.patheffects
import matplotlib.pyplot as plt

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
        log_w: bool = False,
    ):
        """
        :param demes.Deme deme: The deme for which to calculate coordinates.
        :param float mid: The mid point of the deme along the non-time dimension.
        :param float inf_start_time: The value along the time dimension which
            is used instead of infinity (for epochs with infinite start times).
        :param bool log_w: Use log-10 of the deme size for the tube width.
        """
        self._coords(deme, mid, inf_start_time, log_w)

    def _coords(
        self,
        deme: demes.Deme,
        mid: float,
        inf_start_time: float,
        log_w: bool,
        num_exp_points: int = 100,
    ):
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
                if log_w:
                    N1 = [mid - np.log(epoch.start_size) / 2] * 2
                    N2 = [mid + np.log(epoch.end_size) / 2] * 2
                else:
                    N1 = [mid - epoch.start_size / 2] * 2
                    N2 = [mid + epoch.end_size / 2] * 2
            elif epoch.size_function == "exponential":
                t = np.linspace(start_time, end_time, num=num_exp_points)
                dt = np.linspace(0, 1, num=num_exp_points)
                if log_w:
                    # WARNING: this isn't the log of the deme size,
                    # except at the start_time and end_time.
                    r = np.log(np.log(epoch.end_size) / np.log(epoch.start_size))
                    N = np.log(epoch.start_size) * np.exp(r * dt)
                else:
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

    def sizes_at(self, time):
        """Return the size coordinates of the tube at the given time."""
        # Find an index into the coords that gives the deme's sizes at the
        # requested time. We could interpolate between the nearest points,
        # but this is good enough in the size_function="constant" and
        # size_function="exponential" cases.
        # Note that np.searchsorted() does a binary search through an array
        # in value-ascending order. Our self.time array is value-descending,
        # so must first be reversed. The result then gives an index relative
        # to the end of the self.time array.
        i = len(self.time) - int(np.searchsorted(self.time[::-1], time))
        if i == len(self.time):
            i -= 1
        return self.size1[i], self.size2[i]


def coexistence_indexes(graph: demes.Graph) -> List[Tuple[int, int]]:
    """Pairs of indices of demes that exist simultaneously."""
    coexist = []
    for j, deme_j in enumerate(graph.demes):
        for k, deme_k in enumerate(graph.demes[j + 1 :], j + 1):
            if (deme_j.start_time >= deme_k.start_time > deme_j.end_time) or (
                deme_j.start_time > deme_k.end_time >= deme_j.end_time
            ):
                # Demes j and k are contemporaries.
                coexist.append((j, k))
    return coexist


def successors_indexes(graph: demes.Graph) -> Dict[int, List[int]]:
    """Graph successors, but use indexes rather than deme IDs"""
    idx = {deme.id: j for j, deme in enumerate(graph.demes)}
    successors = dict()
    for deme, children in graph.successors().items():
        if len(children) > 0:
            successors[idx[deme]] = [idx[child] for child in children]
    return successors


def interactions_indexes(graph: demes.Graph, *, unique: bool) -> List[Tuple[int, int]]:
    """Pairs of indices of demes that exchange migrants (migrations or pulses)."""
    idx = {deme.id: j for j, deme in enumerate(graph.demes)}
    interactions = []
    for migration in graph.migrations:
        if isinstance(migration, demes.SymmetricMigration):
            for a, b in itertools.permutations(migration.demes, 2):
                interactions.append((idx[a], idx[b]))
        elif isinstance(migration, demes.AsymmetricMigration):
            interactions.append((idx[migration.source], idx[migration.dest]))
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


def find_positions(
    graph: demes.Graph, sep: float, rounds: int = None, seed: int = None
) -> Dict[str, float]:
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

    :return: A dictionary mapping deme IDs to positions.
    :rtype: dict
    """
    if len(graph.demes) == 1:
        return {graph.demes[0].id: 0}
    if rounds is None:
        rounds = 100

    coexist = coexistence_indexes(graph)
    successors = successors_indexes(graph)
    interactions = interactions_indexes(graph, unique=True)
    idx = {deme.id: j for j, deme in enumerate(graph.demes)}

    def fseparation(x):
        """The separation distance between coexisting demes."""
        return np.array([np.abs(x[j] - x[k]) for j, k in coexist])

    def fmin(x):
        """Function to be minimised."""
        z = 0
        # Minimise the distance from each deme to its children.
        for parent, children in successors.items():
            a = x[parent]
            b = np.mean([x[child] for child in children])
            z += (a - b) ** 2
        # Minimise the distance between interacting demes.
        # (either migrations or pulses).
        for j, k in interactions:
            z += (x[j] - x[k]) ** 2
        # Also penalise large positions.
        z += sum(x)
        return z

    x0 = np.arange(len(graph.demes)) * sep
    fmin_best = fmin(x0)
    x_best = x0.copy()

    def initial_states(rounds):
        """Generate initial states for the optimisation procedure."""
        n = len(graph.demes)
        if math.factorial(n) <= rounds:
            # generate all permutations
            yield from itertools.permutations(x0, n)
        else:
            rng = np.random.default_rng(seed)
            for _ in range(rounds):
                x = list(x0)
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
            bounds=scipy.optimize.Bounds(0, max(x0) * 1.01),
            constraints=scipy.optimize.NonlinearConstraint(
                fseparation,
                lb=sep,
                ub=np.inf,
            ),
        )
        if res.success:
            z = fmin(res.x)
            if z < fmin_best:
                x_best = res.x
                fmin_best = z

    return {deme.id: position for deme, position in zip(graph.demes, x_best)}


def schematic(
    graph: demes.Graph,
    ax: matplotlib.axes.Axes = None,
    cmap: matplotlib.colors.Colormap = None,
    log_y: bool = False,
    log_w: bool = False,
    title: str = None,
    inf_ratio: float = 0.2,
    positions: Dict[str, float] = None,
    num_lines_per_migration: int = 10,
    seed: int = None,
    optimisation_rounds: int = None,
) -> matplotlib.axes.Axes:
    """
    Plot a schematic of the demes in the graph and their relationships.

    The width of each deme is proportional to its size, and ancestor/descendant
    relationships are drawn as dotted lines.
    Pulses of admixture are drawn as dashed lines in the colour of the source
    deme, with an arrow pointing from the source to the destination.
    For each period of continuous migration, multiple thin lines are drawn in
    the colour of the source deme, with an arrow pointing from the source to
    the destination. The time of each migration line is drawn uniformly at
    random from the migration's time interval (or log-uniformly for a
    log-scaled time axis). Symmetric migrations have lines in both directions.

    The horizontal position of demes are obtained by non-linear optimisation.
    Specifically, we minimise the distance between ancestors and their
    decendants, and the distance between interacting demes (migrations or pulses),
    subject to the contraint that coexisting demes must have a minimum
    separation distance. Larger graphs may benefit from increasing
    ``optimisation_rounds``. Note that the optimisation procedure may not work
    well for all graphs, so users may prefer to manually specify the
    ``positions`` instead.

    :param demes.Graph graph: The demes graph to plot.
    :param matplotlib.axes.Axes ax: The matplotlib axes onto which the figure
        will be drawn. If None, an empty axes will be created for the figure.
    :param matplotlib.colors.Colormap cmap: A matplotlib colour map to be used
        for the different demes. Get one with :func:`matplotlib.cm.get_cmap()`.
        If None, tab10 or tab20 will be used, depending on the number of demes.
    :param bool log_y: Use a log-10 scale for the vertical axis.
    :param bool log_w: Use a log-10 scale for the width of demes.
        Note that exponentially changing epochs are drawn using a heuristic,
        and are not log-proportional to deme size except at the start/end times
        of the epoch.
    :param str title: The title of the figure.
    :param float inf_ratio: The proportion of the horizontal axis that will be
        used for the time interval which stretches towards infinity.
    :param dict positions: A dictionary mapping deme IDs to horizontal
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
    if ax is None:
        fig_w, fig_h = plt.figaspect(9.0 / 16.0)
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

    if cmap is None:
        if len(graph.demes) <= 10:
            cmap = matplotlib.cm.get_cmap("tab10")
        elif len(graph.demes) <= 20:
            cmap = matplotlib.cm.get_cmap("tab20")
        else:
            raise ValueError(
                "Graph has more than 20 demes, so cmap must be specified. Good luck!"
            )
    colours = {deme.id: cmap(j) for j, deme in enumerate(graph.demes)}

    rng = np.random.default_rng(seed)
    seed2 = rng.integers(2 ** 63)
    inf_start_time = utils.inf_start_time(graph, inf_ratio, log_y)

    size_max = max(
        max(epoch.start_size, epoch.end_size)
        for deme in graph.demes
        for epoch in deme.epochs
    )
    if log_w:
        size_max = np.log(size_max)
    if positions is None:
        positions = find_positions(
            graph, size_max * 1.1, rounds=optimisation_rounds, seed=seed2
        )
    width = (max(positions.values()) - min(positions.values())) / len(graph.demes)

    tubes = {}

    for j, deme in enumerate(graph.demes):
        colour = colours[deme.id]
        plot_kwargs = dict(color=colour, solid_capstyle="butt", zorder=1)

        mid = positions[deme.id]

        tube = Tube(deme, mid, inf_start_time, log_w=log_w)
        tubes[deme.id] = tube

        ax.plot(tube.size1, tube.time, **plot_kwargs)
        ax.plot(tube.size2, tube.time, **plot_kwargs)
        ax.fill_betweenx(
            tube.time,
            tube.size1,
            tube.size2,
            facecolor=colour,
            edgecolor="none",
            alpha=0.3,
            zorder=1,
        )

        # Indicate ancestry from ancestor demes.
        ancestry_kwargs = dict(linestyle=":", **plot_kwargs)
        for ancestor_id in deme.ancestors:
            anc_size1, anc_size2 = tubes[ancestor_id].sizes_at(deme.start_time)
            y = [tube.time[0], tube.time[0]]
            if anc_size2 < tube.size1[0]:
                ax.plot([anc_size2, tube.size1[0]], y, **ancestry_kwargs)
            elif tube.size2[0] < anc_size1:
                ax.plot([tube.size2[0], anc_size1], y, **ancestry_kwargs)
            else:
                ax.plot([anc_size1, tube.size1[0]], y, **ancestry_kwargs)
                ax.plot([anc_size2, tube.size2[0]], y, **ancestry_kwargs)

    def migration_line(source, dest, time, **mig_kwargs):
        """Draw a migration line from source to dest at the given time."""
        source_size1, source_size2 = tubes[source].sizes_at(time)
        dest_size1, dest_size2 = tubes[dest].sizes_at(time)

        offset = 0.15 * width  # the space between arrow head and tube
        if positions[source] < positions[dest]:
            x = [source_size2 + offset, dest_size1 - offset]
            arrow = ">k"
        else:
            x = [source_size1 - offset, dest_size2 + offset]
            arrow = "<k"

        colour = colours[source]
        ax.plot(
            x,
            [time, time],
            color=colour,
            path_effects=[matplotlib.patheffects.Normal()],
            zorder=-1,
            **mig_kwargs,
        )
        ax.plot(
            x[1],
            time,
            arrow,
            markersize=2,
            color=colour,
            path_effects=[matplotlib.patheffects.Normal()],
            **mig_kwargs,
        )

    def random_migration_time(migration, log_scale):
        if log_scale:
            t = np.exp(
                rng.uniform(
                    np.log(migration.start_time), np.log(1 + migration.end_time)
                )
            )
        else:
            t = rng.uniform(migration.start_time, migration.end_time)
        return t

    # Plot migration lines.
    migration_kwargs = dict(linewidth=0.2, alpha=0.5)
    for migration in graph.migrations:
        if isinstance(migration, demes.SymmetricMigration):
            for a, b in itertools.permutations(migration.demes, 2):
                for _ in range(num_lines_per_migration):
                    t = random_migration_time(migration, log_y)
                    migration_line(a, b, t, **migration_kwargs)
        elif isinstance(migration, demes.AsymmetricMigration):
            for _ in range(num_lines_per_migration):
                t = random_migration_time(migration, log_y)
                migration_line(migration.source, migration.dest, t, **migration_kwargs)

    # Plot pulse lines.
    for pulse in graph.pulses:
        migration_line(
            pulse.source, pulse.dest, pulse.time, linestyle="--", linewidth=1
        )

    # Add legend.
    if len(graph.demes) > 1:
        ax.legend(
            handles=[
                matplotlib.patches.Patch(
                    edgecolor=matplotlib.colors.to_rgba(colours[deme.id], 1.0),
                    facecolor=matplotlib.colors.to_rgba(colours[deme.id], 0.3),
                    label=deme.id,
                )
                for deme in graph.demes
            ],
            # Use a horizontal layout, rather than vertical.
            ncol=len(graph.demes) // 2,
        )

    if title is not None:
        ax.set_title(title)

    # ax.set_xticks([position[deme.id] for deme in graph.demes])
    # ax.set_xticklabels([deme.id for deme in graph.demes])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_ylabel(f"time ago ({graph.time_units})")

    if log_y:
        ax.set_yscale("log", base=10)

    ax.set_ylim(1 if log_y else 0, None)

    ax.figure.tight_layout()
    return ax


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot a schematic of the population relationships in the graph."
    )
    parser.add_argument(
        "--log-y", action="store_true", help="Use a log scale for the vertical axis."
    )
    parser.add_argument(
        "--log-w",
        action="store_true",
        help="Use a log scale for the width of demes (representing deme size).",
    )
    parser.add_argument(
        "--optimisation-rounds",
        type=int,
        default=None,
        help=(
            "Number of rounds of optimisation to perform when searching for "
            "reasonable horizontal positions for the demes. "
        ),
    )
    parser.add_argument(
        "--xkcd", action="store_true", help="Plot using the XKCD cartoon style."
    )
    parser.add_argument(
        "yaml_filename",
        metavar="demes.yaml",
        help="The Demes graph to plot.",
    )
    parser.add_argument(
        "plot_filename",
        metavar="figure.pdf",
        help=(
            "Output filename for the figure. "
            "Any file extension supported by matplotlib may be provided "
            "(pdf, eps, png, svg)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.xkcd:
        plt.xkcd()
    graph = demes.load(args.yaml_filename)
    ax = schematic(
        graph,
        log_y=args.log_y,
        log_w=args.log_w,
        optimisation_rounds=args.optimisation_rounds,
    )
    ax.figure.savefig(args.plot_filename)
