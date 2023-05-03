import warnings

import demes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import demesdraw
from demesdraw import utils
from tests import example_files


with PdfPages("spacing.pdf") as pdf:
    for _n in range(30):
        n = _n + 1
        print(n)
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        for j in range(n):
            b.add_deme(f"d{j}")
            if j > 0:
                b.add_migration(demes=[f"d{j - 1}", f"d{j}"], rate=1e-5)
        graph = b.resolve()

        colours = None
        if n > 20:
            colours = "black"
        ax = demesdraw.tubes(graph, colours=colours)
        pdf.savefig(ax.figure)
        plt.close(ax.figure)

with PdfPages("examples.pdf") as pdf:
    for filename in sorted(example_files()):
        title = filename.name
        print(title)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Multiple pulses", UserWarning, "demes")
            graph = demes.load(filename)

        fig, (ax1, ax2) = utils.get_fig_axes(
            scale=1.5, nrows=2, constrained_layout=True
        )
        log_size = utils.log_size_heuristic(graph)
        log_time = utils.log_time_heuristic(graph)

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
            scale_bar=True,
        )

        pdf.savefig(fig)
        plt.close(fig)
