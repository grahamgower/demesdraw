import itertools
import math
from typing import Dict, List, Mapping, Set, Tuple, Union
import warnings

import cvxpy as cp
import demes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "get_fig_axes",
    "size_max",
    "size_min",
    "log_size_heuristic",
    "log_time_heuristic",
    "separation_heuristic",
]


# Override the symbols that are returned when calling dir(<module-name>).
def __dir__():
    return sorted(__all__)


# A colour is either a colour name string (e.g. "blue"), or an RGB triple,
# or an RGBA triple.
Colour = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]
# A mapping from a string to a colour, or just a single colour
ColourOrColourMapping = Union[Mapping[str, Colour], Colour]


def _get_times(graph: demes.Graph) -> Set[float]:
    """
    Get all times in the graph.

    :param demes.Graph: The graph.
    :return: The set of times found in the graph.
    :rtype: set[float]
    """
    times = set()
    for deme in graph.demes:
        times.add(deme.start_time)
        for epoch in deme.epochs:
            times.add(epoch.end_time)
    for migration in graph.migrations:
        times.add(migration.start_time)
        times.add(migration.end_time)
    for pulse in graph.pulses:
        times.add(pulse.time)
    return times


def _inf_start_time(graph: demes.Graph, inf_ratio: float, log_scale: bool) -> float:
    """
    Calculate the value on the time axis that will be used instead of infinity.

    :param demes.Graph graph: The graph.
    :param float inf_ratio: The proportion of the time axis that will be used
        for the time interval which stretches towards infinity.
    :param bool log_scale: The time axis uses a log scale.
    :return: The time.
    :rtype: float
    """
    times = _get_times(graph)
    times.discard(math.inf)
    oldest_noninf_time = max(times)

    if oldest_noninf_time == 0:
        # All demes are root demes and have a constant size.
        # About 100 generations is a nice time scale to draw, and we use 113
        # specifically so that the infinity line extends slightly beyond the
        # last tick mark that matplotlib autogenerates.
        inf_start_time: float = 113
    else:
        if log_scale:
            if oldest_noninf_time <= 1:
                # A log scale is a terrible choice for this graph.
                warnings.warn(
                    "Graph contains features at 0 < time <= 1, which will not "
                    "be visible on a log scale."
                )
                oldest_noninf_time = 2
            inf_start_time = math.exp(math.log(oldest_noninf_time) / (1 - inf_ratio))
        else:
            inf_start_time = oldest_noninf_time / (1 - inf_ratio)
    return inf_start_time


def _get_colours(
    graph: demes.Graph,
    colours: ColourOrColourMapping = None,
    default_colour: Colour = "gray",
) -> Mapping[str, Colour]:
    """
    Convert the polymorphic ``colours`` into a dictionary of colours,
    keyed by deme name.

    :param demes.Graph graph: The graph to which colours will apply.
    :param colours: The colour or colours.
        * If ``colours`` is ``None``, the default colour map will be used.
        * If ``colours`` is a dict, it must map deme names to colours.
          All demes not in the dict will be drawn with ``default_colour``.
        * Otherwise, if ``colours`` can be interpreted as a matplotlib
          colour, all demes will be drawn with this colour.
    :type: dict or str
    """
    if colours is None:
        if len(graph.demes) <= 10:
            cmap = matplotlib.cm.get_cmap("tab10")
        elif len(graph.demes) <= 20:
            cmap = matplotlib.cm.get_cmap("tab20")
        else:
            raise ValueError(
                "Graph has more than 20 demes, so colours must be specified."
            )
        new_colours = {deme.name: cmap(j) for j, deme in enumerate(graph.demes)}
    elif isinstance(colours, Mapping):
        bad_names = list(colours.keys() - set([deme.name for deme in graph.demes]))
        if len(bad_names) > 0:
            raise ValueError(
                f"Colours given for deme(s) {bad_names}, but deme(s) were "
                "not found in the graph."
            )
        new_colours = {deme.name: default_colour for deme in graph.demes}
        new_colours.update(**colours)
    else:
        # Try to interpret as a matplotlib colour.
        try:
            colour = matplotlib.colors.to_rgba(colours)
        except ValueError as e:
            raise ValueError(
                f"Colour '{colours}' not interpretable as a matplotlib colour"
            ) from e
        new_colours = {deme.name: colour for deme in graph.demes}
    return new_colours


def get_fig_axes(
    aspect: float = None, scale: float = None, **kwargs
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Get a matplotlib figure and axes.

    The default width and height of a matplotlib figure is 6.4 x 4.8 inches.
    To create axes on a figure with other sizes, the ``figsize`` argument
    can be passed to :func:`matplotlib.pyplot.subplots`.
    The ``get_fig_axes()`` function creates an axes on a figure with an
    alternative parameterisation that is useful for screen display of
    vector images.

    An ``aspect`` parameter sets the aspect ratio, and a ``scale`` parameter
    multiplies the figure size. Increasing the scale will have the effect of
    decreasing the size of objects in the figure (including fonts),
    and increasing the amount of space between objects.

    :param aspect: The aspect ratio (height/width) of the figure.
        This value will be passed to :func:`matplotlib.figure.figaspect` to
        obtain the figure's width and height dimensions.
        If not specified, 9/16 will be used.
    :param scale: Multiply the figure width and height by this value.
        If not specified, 1.0 will be used.
    :param kwargs: Further keyword args will be passed directly to
        :func:`matplotlib.pyplot.subplots`.
    :return: A 2-tuple containing the matplotlib figure and axes,
        as returned by :func:`matplotlib.pyplot.subplots`.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if aspect is None:
        aspect = 9.0 / 16.0
    if scale is None:
        scale = 1.0
    fig, ax = plt.subplots(figsize=scale * plt.figaspect(aspect), **kwargs)
    if not fig.get_constrained_layout():
        fig.set_tight_layout(True)
    return fig, ax


def size_max(graph: demes.Graph) -> float:
    """
    Get the maximum deme size in the graph.

    :param demes.Graph graph: The graph.
    :return: The maximum deme size.
    :rtype: float
    """
    return max(
        max(epoch.start_size, epoch.end_size)
        for deme in graph.demes
        for epoch in deme.epochs
    )


def size_min(graph: demes.Graph) -> float:
    """
    Get the minimum deme size in the graph.

    :param demes.Graph graph: The graph.
    :return: The minimum deme size.
    :rtype: float
    """
    return min(
        min(epoch.start_size, epoch.end_size)
        for deme in graph.demes
        for epoch in deme.epochs
    )


def log_size_heuristic(graph: demes.Graph) -> bool:
    """
    Decide whether or not to use log scale for sizes.

    :param demes.Graph graph: The graph.
    :return: True if log scale should be used or False otherwise.
    :rtype: bool
    """
    if size_max(graph) / size_min(graph) > 4:
        log_size = True
    else:
        log_size = False
    return log_size


def log_time_heuristic(graph: demes.Graph) -> bool:
    """
    Decide whether or not to use log scale for times.

    :param demes.Graph graph:
        The graph.
    :return:
        True if log scale should be used or False otherwise.
    """
    times = _get_times(graph)
    times.discard(0)
    times.discard(math.inf)
    if len(times) > 0 and max(times) / min(times) > 4:
        log_time = True
    else:
        log_time = False
    return log_time


def _contemporaries_max(graph: demes.Graph) -> int:
    """The maximum number of contemporary demes at any time."""
    c_max = 1
    for deme_j in graph.demes:
        c = 1
        for deme_k in graph.demes:
            if deme_j is not deme_k and _intersect(deme_j, deme_k):
                c += 1
        if c > c_max:
            c_max = c
    return c_max


def separation_heuristic(graph: demes.Graph) -> float:
    """
    Find a reasonable separation distance for deme positions.

    :param demes.Graph graph:
        The graph.
    :return:
        The separation distance.
    """
    # This looks ok. See spacing.pdf produced by tests/plot_examples.py.
    c = _contemporaries_max(graph)
    return (1 + 0.5 * math.log(c)) * size_max(graph)


def _intersect(
    dm_j: Union[demes.Deme, demes.AsymmetricMigration],
    dm_k: Union[demes.Deme, demes.AsymmetricMigration],
) -> bool:
    """Returns true if dm_j and dm_k intersect in time."""
    return not (dm_j.end_time >= dm_k.start_time or dm_k.end_time >= dm_j.start_time)


def coexistence_indices(graph: demes.Graph) -> List[Tuple[int, int]]:
    """Pairs of indices of demes that exist simultaneously."""
    contemporaries = []
    for j, deme_j in enumerate(graph.demes):
        for k, deme_k in enumerate(graph.demes[j + 1 :], j + 1):
            if _intersect(deme_j, deme_k):
                contemporaries.append((j, k))
    return contemporaries


def successors_indices(graph: demes.Graph) -> Dict[int, List[int]]:
    """Graph successors, but use indices rather than deme names"""
    idx = {deme.name: j for j, deme in enumerate(graph.demes)}
    successors = dict()
    for deme, children in graph.successors().items():
        if len(children) > 0:
            successors[idx[deme]] = [idx[child] for child in children]
    return successors


def interactions_indices(graph: demes.Graph, *, unique: bool) -> List[Tuple[int, int]]:
    """Pairs of indices of demes that exchange migrants (migrations or pulses)."""
    idx = {deme.name: j for j, deme in enumerate(graph.demes)}
    interactions = []
    for migration in graph.migrations:
        interactions.append((idx[migration.source], idx[migration.dest]))
    for pulse in graph.pulses:
        for source in pulse.sources:
            interactions.append((idx[source], idx[pulse.dest]))

    if unique:
        # Remove duplicates.
        i2 = set()
        for a, b in interactions:
            if (b, a) in i2:
                continue
            i2.add((a, b))
        interactions = list(i2)

    return interactions


def _get_line_candidates(graph: demes.Graph, unique: bool):
    candidates = []
    # Ancestry lines.
    for child in graph.demes:
        for parent_name in child.ancestors:
            for deme in graph.demes:
                if deme is child or deme.name == parent_name:
                    continue
                if deme.start_time > child.start_time > deme.end_time:
                    candidates.append((child.name, deme.name, parent_name))
    # Pulse lines.
    for pulse in graph.pulses:
        for source in pulse.sources:
            for deme in graph.demes:
                if deme.name in [pulse.dest, source]:
                    continue
                if deme.start_time > pulse.time > deme.end_time:
                    candidates.append((source, deme.name, pulse.dest))
    # Migration blocks.
    for migration in graph.migrations:
        for deme in graph.demes:
            if deme.name in (migration.dest, migration.source):
                continue
            if _intersect(deme, migration):
                candidates.append((migration.source, deme.name, migration.dest))

    if unique:
        # Remove duplicates.
        c2 = set()
        for a, b, c in candidates:
            if (c, b, a) in c2:
                continue
            c2.add((a, b, c))
        candidates = list(c2)

    idx = {deme.name: j for j, deme in enumerate(graph.demes)}
    return np.array([(idx[a], idx[b], idx[c]) for a, b, c in candidates])


def _line_crossings(xx: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Count the number of lines that cross one or more demes.

    :param xx:
        The positions of the demes in the graph. This is a 2d array,
        where each row in xx is a distinct vector of proposed positions.
        Specifically, xx[j, k] is the position of deme k in the j'th
        proposed positions vector.
    :param candidates:
        A 2d array of candidate indices.
    :return:
        The number of lines crossing demes. This is a 1d array
        of counts, one for each vector of proposed positions.
    """
    if candidates.size == 0:
        return np.array(0)
    a = xx[..., candidates[:, 0]]
    b = xx[..., candidates[:, 1]]
    c = xx[..., candidates[:, 2]]
    return np.logical_or(
        np.logical_and(a < b, b < c), np.logical_and(a > b, b > c)
    ).sum(axis=-1)


def minimal_crossing_positions(
    graph: demes.Graph,
    *,
    sep: float,
    unique_interactions: bool,
    maxiter: int = 1000000,
    seed: int = 1234,
) -> Dict[str, float]:
    """
    Find an ordering of demes that minimises line crossings.

    Lines may be for ancestry, pulses, or migrations. A naive algorithm is
    used to search for the optimal ordering: if :math:`n!` <= ``maxiter``,
    where ``n`` is the number of demes in the graph, then all possible
    orderings may be evaluated; otherwise up to ``maxiter`` random
    permutations are evaluated. If a proposed ordering produces zero line
    crossings, the algorithm terminates early.

    :param graph:
        Graph for which positions should be obtained.
    :param sep:
        The separation distance between demes.
    :param maxiter:
        Maximum number of orderings to search through.
    :param seed:
        Seed for the random number generator.
    :return:
        A dictionary mapping deme names to positions.
    """
    # Initial ordering.
    x0 = np.arange(0, sep * len(graph.demes), sep)

    def propose_positions_batches(num_proposals: int, batch_size=1000):
        n = len(graph.demes)
        if math.factorial(n) <= num_proposals:
            # Generate all permutations.
            iperms = itertools.permutations(x0, n)
            while True:
                x_batch = np.array(list(itertools.islice(iperms, batch_size)))
                if x_batch.size == 0:
                    break
                yield x_batch
        else:
            # Generate random permutations.
            rng = np.random.default_rng(seed)
            remaining = num_proposals
            while remaining > 0:
                batch_size = min(batch_size, remaining)
                x_batch = np.array([rng.permutation(x0) for _ in range(batch_size)])
                yield x_batch
                remaining -= batch_size

    candidates = _get_line_candidates(graph, unique=unique_interactions)
    crosses = _line_crossings(x0, candidates)
    x_best = x0
    if crosses > 0:
        for x_proposal_batch in propose_positions_batches(maxiter):
            proposal_crosses = _line_crossings(x_proposal_batch, candidates)
            best = np.argmin(proposal_crosses)
            if proposal_crosses[best] < crosses:
                crosses = proposal_crosses[best]
                x_best = x_proposal_batch[best]
                if crosses == 0:
                    break

    return {deme.name: float(pos) for deme, pos in zip(graph.demes, x_best)}


def cvxpy_optimise(
    graph: demes.Graph,
    positions: Mapping[str, float],
    *,
    sep: float,
    unique_interactions: bool,
) -> Dict[str, float]:
    """
    Optimise the given positions into a tree-like layout.

    Convex optimisation is used to minimise the distances:

      - from each parent deme to the mean position of its children,
      - and between interacting demes (where interactions are either
        migrations or pulses).

    Subject to the constraints that:

      - demes are ordered like in the input ``positions``,
      - and demes have a minimum separation distance ``sep``.

    :param graph:
        Graph for which positions should be optimised.
    :param positions:
        A dictionary mapping deme names to positions.
    :param sep:
        The minimum separation distance between contemporary demes.
    :return:
        A dictionary mapping deme names to positions.
    """
    contemporaries = coexistence_indices(graph)
    if len(contemporaries) == 0:
        # There are no constraints, so stack demes on top of each other.
        return {deme.name: 0 for deme in graph.demes}
    successors = successors_indices(graph)
    interactions = interactions_indices(graph, unique=unique_interactions)

    vs = [cp.Variable() for _ in graph.demes]
    # Place the root at position 0.
    vs[0].value = 0

    z = 0
    for parent, children in successors.items():
        a = vs[parent]
        b = sum(vs[child] for child in children) / len(children)
        z += (a - b) ** 2
    if len(interactions) > 0:
        z += cp.sum_squares(cp.hstack([vs[j] - vs[k] for j, k in interactions]))
    objective = cp.Minimize(z)

    constraints = []
    x = np.array([positions[deme.name] for deme in graph.demes])
    for j, k in contemporaries:
        if x[j] < x[k]:
            j, k = k, j
        constraints.append(vs[j] - vs[k] >= sep)

    prob = cp.Problem(objective, constraints)
    prob.solve("OSQP")
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Failed to optimise: {prob}")

    return {graph.demes[j].name: float(v.value) for j, v in enumerate(vs)}
