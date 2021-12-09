import sys
import logging
import itertools

import daiquiri
import numpy as np
import demes
import demesdraw
import msprime
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def coalescence_times_matrix(graph):
    """
    Return a matrix of coalecence times between pairs of populations.
    """
    leaves = [deme.name for deme in graph.demes if deme.end_time == 0]
    assert len(leaves) > 0
    dbg = msprime.Demography.from_demes(graph).debug()
    steps = get_steps(dbg)
    ET = np.zeros((len(leaves), len(leaves)))
    for j, a in enumerate(leaves):
        for k, b in enumerate(leaves[j:], j):
            lineages = {a: 1, b: 1} if a != b else {a: 2}
            ET[j, k] = dbg.mean_coalescence_time(lineages)
            ET[k, j] = ET[j, k]
    return ET


def f2_matrix(graph):
    """
    Return a matrix of f2 stats between pairs of populations.

    Peter (2016), https://doi.org/10.1534/genetics.115.183913
    """
    ET = coalescence_times_matrix(graph)
    f2 = np.zeros_like(ET)
    for j in range(f2.shape[0]):
        for k in range(j + 1, f2.shape[1]):
            f2[j, k] = ET[j, k] - (ET[j, j] + ET[k, k]) / 2
            f2[k, j] = f2[j, k]
    return f2


def PCA(f2):
    """
    PCA decomposition of f2 matrix.

    Peter (2021), https://doi.org/10.1101/2021.07.13.452141
    """
    assert len(f2.shape) == 2
    assert f2.shape[0] == f2.shape[1]
    n = f2.shape[0]
    C = np.eye(n) - np.ones((n, n)) / n
    X = -C @ f2 @ C / 2
    evals, evecs = np.linalg.eigh(X)
    # sort by decreasing eigenvalue
    idx = np.argsort(np.abs(evals))[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    return evecs, evals


def get_plot_styles():
    markers = itertools.cycle("oxd|+_s")
    fcs = itertools.cycle(["none", None, "none", None, None, None, "none"])
    ecs = itertools.cycle(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])
    return (
        dict(marker=marker, edgecolors=ec, c=fc)
        for marker, ec, fc in zip(markers, ecs, fcs)
    )


def plot_scree(ax, lambda_):
    nscree = min(20, len(lambda_))
    scree_x = np.arange(nscree)
    scree_y = lambda_[:nscree] / sum(lambda_)
    ax.bar(scree_x, scree_y)
    ax.set_xticks(scree_x)
    ax.set_xticklabels(scree_x + 1)
    ax.set_ylabel("proportion of variance explained")
    ax.set_xlabel("principal dimension")


def get_axes(aspect=9 / 16, scale=1.5, **subplot_kwargs):
    """Make a matplotlib axes."""
    figsize = scale * plt.figaspect(aspect)
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    fig.set_tight_layout(True)
    return fig, ax


def plot_figure(graph, P, lambda_):
    fig, axs = get_axes(nrows=2, ncols=3)
    leaves = [deme.name for deme in graph.demes if deme.end_time == 0]
    for k, ax in enumerate([axs[0, 1], axs[0, 2], axs[1, 1], axs[1, 2]]):
        style = get_plot_styles()
        pc_x, pc_y = P[:, k], P[:, k + 1]
        for j, deme_name in enumerate(leaves):
            ax.scatter(pc_x[j], pc_y[j], label=deme_name, **next(style))
        ax.set_xlabel(f"PC{k+1}")
        ax.set_ylabel(f"PC{k+2}")
        if k == 0:
            ax.legend(ncol=2)

    demesdraw.tubes(graph, ax=axs[0, 0], log_time=True)
    for tick in axs[0, 0].get_xticklabels():
        tick.set_rotation(45)

    plot_scree(axs[1, 0], lambda_)
    return fig


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} model.yaml")
        exit(1)

    input_file = sys.argv[1]
    graph = demes.load(input_file)

    ET = coalescence_times_matrix(graph)

    # daiquiri.setup(level="DEBUG")
    f2 = f2_matrix(graph)
    # daiquiri.setup(level="WARNING")

    P, lambda_ = PCA(f2)
    np.savetxt("/tmp/P.txt", P)
    np.savetxt("/tmp/lambda.txt", lambda_)
    P = np.loadtxt("/tmp/P.txt")
    lambda_ = np.loadtxt("/tmp/lambda.txt")
    fig = plot_figure(graph, P, lambda_)
    fig.savefig("/tmp/f2_pca.pdf")
