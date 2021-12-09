# https://tskit.dev/msprime/docs/stable/ancestry.html#migration-events
import sys

import demes
import demesdraw
import msprime
import numpy as np
import tskit

if len(sys.argv) != 4:
    print(f"usage: {sys.argv[0]} model.yaml deme1 deme2")
    exit(1)

graph = demes.load(sys.argv[1]).in_generations()
deme1 = sys.argv[2]
deme2 = sys.argv[3]
for deme_name in (deme1, deme2):
    if deme_name not in graph:
        raise RuntimeError(f"{deme_name} not in graph")

ts = msprime.sim_ancestry(
    samples=[
        msprime.SampleSet(
            1, population=deme_name, time=graph[deme_name].end_time, ploidy=1
        )
        for deme_name in (deme1, deme2)
    ],
    demography=msprime.Demography.from_demes(graph),
    record_migrations=True,
    # record_full_arg=True,
    # random_seed=6,
)

tree = ts.first()
w = 1.5 * demesdraw.utils.size_max(graph)
positions = {deme.name: j * w for j, deme in enumerate(graph.demes)}
ax = demesdraw.tubes(
    graph,
    positions=positions,
    # extend figure past the TMRCA
    # max_time=1.1 * tree.time(tree.root),
    # don't draw migration lines, as they won't match the simulation
    num_lines_per_migration=0,
    log_time=True,
)

mig = ts.tables.migrations
for u, linestyle in zip(ts.samples()[:2], ["dotted", "dashdot"]):
    while u != tskit.NULL:
        migs = np.where(mig.node == u)[0]
        for cur_mig in migs:
            cur_mig = mig[cur_mig]
            dest_pos = positions[graph.demes[cur_mig.dest].name]
            source_pos = positions[graph.demes[cur_mig.source].name]
            time = cur_mig.time
            ax.plot(
                [source_pos, dest_pos], [time, time], linestyle=linestyle, color="black"
            )
            if dest_pos < source_pos:
                arrow = "<"
            else:
                arrow = ">"
            ax.plot([dest_pos], [time], arrow, color="black")
        u = tree.parent(u)


ax.figure.savefig("/tmp/mig.pdf")
