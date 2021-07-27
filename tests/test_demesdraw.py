import math

import demes
import pytest
import matplotlib
import matplotlib.pyplot as plt

import demesdraw
import tests


class TestSizeHistory:
    def check_size_history(self, graph, **kwargs):
        ax = demesdraw.size_history(graph, **kwargs)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    @pytest.mark.parametrize("log_size", [True, False])
    @pytest.mark.parametrize("log_time", [True, False])
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_log_params(self, graph, log_time, log_size):
        self.check_size_history(graph, log_time=log_time, log_size=log_size)

    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_annotate_epochs_invert_x(self, graph):
        self.check_size_history(graph, invert_x=True, annotate_epochs=True)


class TestTubes:
    def check_tubes(self, graph, seed=1234, optimisation_rounds=None, **kwargs):
        ax = demesdraw.tubes(
            graph, seed=seed, optimisation_rounds=optimisation_rounds, **kwargs
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    @pytest.mark.parametrize("log_time", [True, False])
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_log_params(self, graph, log_time):
        self.check_tubes(graph, log_time=log_time)

    @pytest.mark.parametrize(
        "labels", ["xticks", "legend", "mid", "xticks-legend", "xticks-mid"]
    )
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_labels_params(self, graph, labels):
        self.check_tubes(graph, labels=labels)

    @pytest.mark.parametrize(
        "colours", [None, "black", dict(), dict(A="red", B="blue", C="pink")]
    )
    def test_colours_params(self, colours):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A", epochs=[dict(end_time=100)])
        b.add_deme("B", ancestors=["A"])
        b.add_deme("C", ancestors=["A"])
        graph = b.resolve()
        self.check_tubes(graph, colours=colours)

        for j in range(10):
            b.add_deme(f"deme{j}", ancestors=["A"])
        graph = b.resolve()
        # 13 demes. Also fine.
        self.check_tubes(graph, colours=colours)

    def test_bad_colours_params(self):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A", epochs=[dict(end_time=100)])
        b.add_deme("B", ancestors=["A"])
        b.add_deme("C", ancestors=["A"])
        graph = b.resolve()
        for colour in [object(), math.inf, "thisisnotarealcolour"]:
            with pytest.raises(ValueError, match="not.*a matplotlib colour"):
                self.check_tubes(graph, colours=colour)
        with pytest.raises(ValueError, match="deme.*not found in the graph"):
            self.check_tubes(graph, colours=dict(X="black"))

        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        for j in range(25):
            b.add_deme(f"deme{j}")
        b.add_migration(demes=[f"deme{j}" for j in range(25)], rate=1e-5)
        graph = b.resolve()
        with pytest.raises(ValueError, match="colours must be specified"):
            self.check_tubes(graph)

    @pytest.mark.parametrize("log_time", [True, False])
    def test_max_time(self, log_time):
        def check_max_time(graph):
            for max_time in [50, 5000, 1e7]:
                ax = demesdraw.tubes(graph, log_time=log_time, max_time=max_time)
                ylim = ax.get_ylim()
                assert math.isclose(ylim[1], max_time)
                plt.close(ax.figure)

        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A")
        b.add_deme("B")
        graph = b.resolve()
        check_max_time(graph)

        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A", epochs=[dict(end_time=100)])
        b.add_deme("B", ancestors=["A"])
        b.add_deme("C", ancestors=["A"])
        graph = b.resolve()
        check_max_time(graph)
