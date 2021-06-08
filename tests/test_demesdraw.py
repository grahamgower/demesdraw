import math

import demes
import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import demesdraw
import demesdraw.utils
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


class TestAsTubes:
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


class TestUtilsInfStartTime:
    @pytest.mark.parametrize("log_scale", [True, False])
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_time_is_reasonable(self, graph, log_scale):
        t = demesdraw.utils.inf_start_time(graph, 0.1, log_scale)
        assert t > 0
        assert not np.isinf(t)

        times = []
        for deme in graph.demes:
            for epoch in deme.epochs:
                times.extend([epoch.start_time, epoch.end_time])
        for migration in graph.migrations:
            times.extend([migration.start_time, migration.end_time])
        for pulse in graph.pulses:
            times.append(pulse.time)
        time_max = max(time for time in times if not np.isinf(time))

        assert t > time_max

    @pytest.mark.parametrize("log_scale", [True, False])
    def test_one_epoch(self, log_scale):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        graph = b.resolve()
        t = demesdraw.utils.inf_start_time(graph, 0.1, log_scale)
        assert t > 0
        assert not np.isinf(t)
