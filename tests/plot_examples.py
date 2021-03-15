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
        title = filename.name
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
        if len(times) > 0 and max(times) / min(times) > 4:
            log_time = True
        else:
            log_time = False

        fig, axs = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(scale * fig_w, scale * fig_h),
            constrained_layout=True,
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

        demesdraw.tubes(
            graph,
            ax=ax2,
            log_time=log_time,
            # fill=False,
            # colours="black",
            # optimisation_rounds=0,
        )

        pdf.savefig(fig)
        plt.close(fig)
