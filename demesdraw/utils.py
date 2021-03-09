import demes
import numpy as np


def inf_start_time(graph: demes.Graph, inf_ratio: float, log_scale: bool) -> float:
    """
    Calculate the value on the time axis that will be used instead of infinity.

    :param float inf_ratio:
    :param bool log_scale: The time axis uses a log scale.
    :return: The time
    :rtype: float
    """
    oldest_end_time = max(deme.epochs[0].end_time for deme in graph.demes)
    if oldest_end_time == 0:
        # All demes are root demes and have a constant size.
        # About 100 generations is a nice time scale to draw, and we use 113
        # specifically so that the infinity line extends slightly beyond the
        # last tick mark that matplotlib autogenerates.
        inf_start_time: float = 113
    else:
        if log_scale:
            inf_start_time = np.exp(np.log(oldest_end_time) / (1 - inf_ratio))
        else:
            inf_start_time = oldest_end_time / (1 - inf_ratio)
    return inf_start_time
