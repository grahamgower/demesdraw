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
    epoch0_times = []
    for deme in graph.demes:
        epoch0_times.append(deme.epochs[0].end_time)
        if not np.isinf(deme.epochs[0].start_time):
            epoch0_times.append(deme.epochs[0].start_time)
    oldest_noninf_time = max(epoch0_times)
    if oldest_noninf_time == 0:
        # All demes are root demes and have a constant size.
        # About 100 generations is a nice time scale to draw, and we use 113
        # specifically so that the infinity line extends slightly beyond the
        # last tick mark that matplotlib autogenerates.
        inf_start_time: float = 113
    else:
        if log_scale:
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
        raise ValueError(f"deme {deme.id} doesn't exist at time {time}")

    if np.isclose(time, epoch.end_time) or epoch.start_size == epoch.end_size:
        N = epoch.end_size
    else:
        assert epoch.size_function == "exponential"
        dt = (epoch.start_time - time) / epoch.time_span
        r = np.log(epoch.end_size / epoch.start_size)
        N = epoch.start_size * np.exp(r * dt)
    return N
