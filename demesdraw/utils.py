from typing import Mapping, Tuple, Union
import warnings

import demes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# A colour is either a colour name string (e.g. "blue"), or an RGB triple,
# or an RGBA triple.
Colour = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]
# A mapping from a string to a colour, or just a single colour
ColourOrColourMapping = Union[Mapping[str, Colour], Colour]


def inf_start_time(graph: demes.Graph, inf_ratio: float, log_scale: bool) -> float:
    """
    Calculate the value on the time axis that will be used instead of infinity.

    :param float inf_ratio:
    :param bool log_scale: The time axis uses a log scale.
    :return: The time
    :rtype: float
    """
    # Find the oldest non-infinite time in the graph.
    times = []
    for deme in graph.demes:
        times.append(deme.epochs[0].end_time)
        if not np.isinf(deme.epochs[0].start_time):
            times.append(deme.epochs[0].start_time)
    for migration in graph.migrations:
        times.append(migration.end_time)
        if not np.isinf(migration.start_time):
            times.append(migration.start_time)
    times.extend([pulse.time for pulse in graph.pulses])
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
            inf_start_time = np.exp(np.log(oldest_noninf_time) / (1 - inf_ratio))
        else:
            inf_start_time = oldest_noninf_time / (1 - inf_ratio)
    return inf_start_time


def size_of_deme_at_time(deme: demes.Deme, time: float) -> float:
    """
    Return the population size of the deme at the given time.
    """
    for epoch in deme.epochs:
        if epoch.start_time >= time >= epoch.end_time:
            break
    else:
        raise ValueError(f"deme {deme.name} doesn't exist at time {time}")

    if np.isclose(time, epoch.end_time) or epoch.start_size == epoch.end_size:
        N = epoch.end_size
    else:
        assert epoch.size_function == "exponential"
        dt = (epoch.start_time - time) / epoch.time_span
        r = np.log(epoch.end_size / epoch.start_size)
        N = epoch.start_size * np.exp(r * dt)
    return N


def get_colours(
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


def get_axes(
    ax: matplotlib.axes.Axes = None, aspect: float = 9.0 / 16.0, scale: float = 1.0
) -> matplotlib.axes.Axes:
    """
    Make a default axes if one isn't provided.
    """
    if ax is None:
        fig_w, fig_h = plt.figaspect(aspect)
        fig, ax = plt.subplots(figsize=(scale * fig_w, scale * fig_h))
        fig.set_tight_layout(True)
    return ax
