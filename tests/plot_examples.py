import math

import demes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import demesdraw
from tests import example_files


fig_w, fig_h = plt.figaspect(9.0 / 16.0)
scale = 1.5

with PdfPages("examples.pdf") as pdf:

    for filename in sorted(example_files()):
        title = filename.name.replace("__", " / ")[: -len(".yaml")]
        graph = demes.load(filename)
        sizes = []
        times = []
        for deme in graph.demes:
            for epoch in deme.epochs:
                sizes.extend([epoch.start_size, epoch.end_size])
                if not math.isinf(epoch.start_time):
                    times.append(epoch.start_time)
                if epoch.end_time > 0:
                    times.append(epoch.end_time)

        if max(sizes) / min(sizes) > 4:
            log_size = True
        else:
            log_size = False
        if max(times) / min(times) > 4:
            log_time = True
        else:
            log_time = False

        fig, axs = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(scale * fig_w, scale * fig_h),
            # constrained_layout=True,
        )
        ax1, ax2 = axs[0], axs[1]

        demesdraw.size_history(
            graph,
            ax=ax1,
            invert_x=True,
            log_time=log_time,
            log_size=log_size,
            title=title,
        )

        if title.endswith("AncientEurasia_9K19"):
            print("AncientEurasia_9K19 legend hack!")
            # ax1.legend_.set_visible(False)
            ax1.legend(
                handles=ax1.legend_.get_lines(),
                ncol=len(graph.demes) // 2,
                loc="lower left",
                bbox_to_anchor=(0.1, 0.1),
            )

        demesdraw.schematic(graph, ax=ax2, log_time=log_time)

        pdf.savefig(fig)
        plt.close(fig)
