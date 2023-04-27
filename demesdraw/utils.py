from __future__ import annotations
import itertools
import math
from typing import Dict, List, Mapping, Set, Tuple, Union
import warnings

import demes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

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
    colours: ColourOrColourMapping | None = None,
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
        if hasattr(matplotlib.colormaps, "get_cmap"):
            get_cmap = matplotlib.colormaps.get_cmap
        else:
            # XXX: matplotlib < 3.6 compat
            get_cmap = matplotlib.cm.get_cmap
        if len(graph.demes) <= 10:
            cmap = get_cmap("tab10")
        elif len(graph.demes) <= 20:
            cmap = get_cmap("tab20")
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
    aspect: float | None = None, scale: float | None = None, **kwargs
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
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine("tight")
        else:
            # matplotlib < 3.6 compat
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
    """
    Construct candidate orderings that, if present, would cause lines to cross.

    Each candidate is a triplet of demes, [A, B, C], which says that when the
    demes A,B,C are ordered with B between A and C, then there will be a line
    drawn between A and C that will cross deme B.

    :return:
        A 2d numpy array of candidate indices, where each row is a candidate
        triplet, and the array entries are the deme indexes (which correspond
        to the ordering of the demes in the graph).
    """
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
        A 2d array of candidate indices. See _get_line_candidates().
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
    maxiter: int = 1_000_000,
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

    The default positions matches the ordering of demes in the graph,
    and this order will be used in preference, if no better ordering
    is found.

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

    def propose_positions_batches(num_proposals: int, batch_size=200):
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
                yield rng.permuted(
                    np.tile(x0, batch_size).reshape(batch_size, -1), axis=1
                )
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


def optimise_positions(
    graph: demes.Graph,
    positions: Mapping[str, float],
    *,
    sep: float,
    unique_interactions: bool,
) -> Dict[str, float]:
    """
    Optimise the given positions into a tree-like layout.

    The objective is to minimise the distances:
      - from each parent deme to the mean position of its children,
      - and between interacting demes (where interactions are either
        migrations or pulses).

    Subject to the constraints that contemporaneous demes:
      - are ordered like in the input ``positions``,
      - and have a minimum separation distance ``sep``.

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
    successors = successors_indices(graph)
    interactions = interactions_indices(graph, unique=unique_interactions)
    if len(contemporaries) == 0 or (len(successors) == 0 and len(interactions) == 0):
        # There are no constraints, so stack demes on top of each other.
        return {deme.name: 0 for deme in graph.demes}

    x0 = np.array([positions[deme.name] for deme in graph.demes])
    # Place the root at position 0.
    x0 -= x0[0]

    def objective(x):
        z = 0
        for parent, children in successors.items():
            a = x[parent]
            b = sum(x[child] for child in children) / len(children)
            z += (a - b) ** 2
        for j, k in interactions:
            z += (x[j] - x[k]) ** 2
        return z

    # Ensure that contemporaries are ordered and separated by sep.
    constraints = []
    for j, k in contemporaries:
        if x0[j] < x0[k]:
            j, k = k, j
        c = [0] * len(x0)
        c[j] = 1
        c[k] = -1
        constraints.append(c)
    linear_constraint = scipy.optimize.LinearConstraint(constraints, lb=sep, ub=np.inf)

    with warnings.catch_warnings():
        # Ignore warnings asking us if we're trying to optimise a linear function.
        # I guess this is triggered when x0 is the optimal solution.
        warnings.filterwarnings(
            "ignore", message="delta_grad == 0.0", category=UserWarning, module="scipy"
        )
        res = scipy.optimize.minimize(
            objective,
            x0,
            method="trust-constr",
            constraints=[linear_constraint],
        )

    if not res.success:
        raise RuntimeError(f"Failed to optimise: {res}")

    x = res.x

    return {graph.demes[j].name: float(xj) for j, xj in enumerate(x)}
