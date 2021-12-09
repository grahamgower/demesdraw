import sys
import logging
import itertools
import time

import daiquiri
import numpy as np
import demes
import demesdraw
import msprime
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_steps(dbg):
    last_N = max(dbg.population_size_history[:, dbg.num_epochs - 1])
    last_epoch = dbg.epoch_start_time[-1]
    times = list(dbg.epoch_start_time) + [last_epoch + 12 * last_N]
    steps = set()
    for a, b in zip(times[:-1], times[1:]):
        steps.update(np.linspace(a, b, 21))
    steps = np.array(sorted(steps))
    return steps


def coalescence_times_matrix(graph):
    """
    Return a matrix of coalecence times between pairs of populations.
    """
    leaves = [deme.name for deme in graph.demes if deme.end_time == 0]
    assert len(leaves) > 0
    ddb = msprime.Demography.from_demes(graph).debug()
    steps = get_steps(ddb)
    ET = np.zeros((len(leaves), len(leaves)))
    for j, a in enumerate(leaves):
        for k, b in enumerate(leaves[j:], j):
            lineages = {a: 1, b: 1} if a != b else {a: 2}
            logger.debug(f"lineages: {a} and {b}")
            ET[j, k] = ddb.mean_coalescence_time(lineages, steps=steps)
            ET[k, j] = ET[j, k]
    return ET


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} model.yaml")
        exit(1)

    input_file = sys.argv[1]
    graph = demes.load(input_file)

    # daiquiri.setup(level="DEBUG")
    for _ in range(10):
        coalescence_times_matrix(graph)
