# https://tskit.dev/msprime/docs/stable/ancestry.html#migration-events

import demes
import demesdraw
import msprime
import numpy as np
import tskit

N = 10
demography = msprime.Demography.stepping_stone_model(
    [20] * N, migration_rate=0.1, boundaries=True
)
ts = msprime.sim_ancestry(
    {0: 1, N - 1: 1},
    demography=demography,
    record_migrations=True,
    # record_full_arg=True,
    random_seed=6,
)

tree = ts.first()
graph = demography.to_demes()
w = 1.5 * demesdraw.utils.size_max(graph)
positions = {deme.name: j * w for j, deme in enumerate(graph.demes)}
ax = demesdraw.tubes(
    graph,
    positions=positions,
    # extend figure past the TMRCA
    max_time=1.1 * tree.time(tree.root),
    # don't draw migration lines, as they won't match the simulation
    num_lines_per_migration=0,
)

for u, linestyle in zip(ts.samples()[::2], ["dotted", "dashdot"]):
    mig = ts.tables.migrations
    loc = []
    time = []
    while u != tskit.NULL:
        migs = np.where(mig.node == u)[0]
        for cur_mig in migs:
            cur_mig = mig[cur_mig]
            loc.append(positions[graph.demes[cur_mig.dest].name])
            time.append(cur_mig.time)
        u = tree.parent(u)
    ax.plot(loc, time, linestyle=linestyle, color="black")


ax.figure.savefig("/tmp/mig.pdf")
