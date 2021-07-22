import math
from typing import Mapping, Set, Tuple, Union
import warnings

import demes
import matplotlib
import matplotlib.pyplot as plt

__all__ = [
    "get_fig_axes",
    "size_max",
    "size_min",
    "log_size_heuristic",
    "log_time_heuristic",
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
        for the time intervale which stretches towards infinity.
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
    But what does this mean for how the figure will look on your laptop?
    Or on your phone? In practice, the device (or the user) will magnify or
    shrink the figure as appropriate. This means you should save your figure
    in a vector format (such as pdf or svg) which permits arbitrary zooming
    without loss of quality.

    When making a vector-format figure, the size of the figure is not usually
    as important as the aspect ratio and the relative sizes of the objects
    within the figure. This function accepts an ``aspect`` parameter that
    sets the aspect ratio, and a ``scale`` parameter that multiplies the
    figure size. Increasing the scale will have the effect of decreasing the
    size of objects in the figure (including fonts), and increasing the amount
    of space between objects.

    :param float aspect: The aspect ratio (height/width) of the figure.
        This value will be passed to :func:`matplotlib.figure.figaspect` to
        obtain the figure's width and height dimensions.
        If not specified, 9/16 will be used.
    :param float scale: Multiply the figure width and height by this value.
        If not specified, 1.0 will be used.
    :param dict kwargs: Further keyword args will be passed directly to
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

    :param demes.Graph graph: The graph.
    :return: True if log scale should be used or False otherwise.
    :rtype: bool
    """
    times = _get_times(graph)
    times.discard(0)
    times.discard(math.inf)
    if len(times) > 0 and max(times) / min(times) > 4:
        log_time = True
    else:
        log_time = False
    return log_time
