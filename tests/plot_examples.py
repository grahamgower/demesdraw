import demes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import demesdraw
from demesdraw import utils
from tests import example_files


with PdfPages("examples.pdf") as pdf:

    for filename in sorted(example_files()):
        title = filename.name
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
            # optimisation_rounds=0,
        )

        pdf.savefig(fig)
        plt.close(fig)
