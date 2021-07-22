import math

import demes
import pytest
import matplotlib.pyplot as plt

import demesdraw
import demesdraw.utils
import tests


class TestInfStartTime:
    @pytest.mark.parametrize("log_scale", [True, False])
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_time_is_reasonable(self, graph, log_scale):
        t = demesdraw.utils._inf_start_time(graph, 0.1, log_scale)
        assert t > 0
        assert not math.isinf(t)

        times = []
        for deme in graph.demes:
            for epoch in deme.epochs:
                times.extend([epoch.start_time, epoch.end_time])
        for migration in graph.migrations:
            times.extend([migration.start_time, migration.end_time])
        for pulse in graph.pulses:
            times.append(pulse.time)
        time_max = max(time for time in times if not math.isinf(time))

        assert t > time_max

    @pytest.mark.parametrize("log_scale", [True, False])
    def test_one_epoch(self, log_scale):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        graph = b.resolve()
        t = demesdraw.utils._inf_start_time(graph, 0.1, log_scale)
        assert t > 0
        assert not math.isinf(t)


class TestGetFigAxes:
    def teardown_method(self):
        plt.close("all")

    def test_return_values_are_consistent(self):
        fig, ax = demesdraw.utils.get_fig_axes()
        assert fig.get_axes() == [ax]
        assert ax.get_figure() == fig

    @pytest.mark.parametrize("aspect", [0.5, 1, 1.5, 3 / 4, 9 / 16, 10 / 16])
    @pytest.mark.parametrize("scale", [0.5, 1, 1.5])
    def test_aspect_and_scale(self, aspect, scale):
        fig1, _ = demesdraw.utils.get_fig_axes(scale=1, aspect=1)
        width1, height1 = fig1.get_size_inches()
        assert math.isclose(height1, width1)
        fig2, _ = demesdraw.utils.get_fig_axes(scale=scale, aspect=1)
        width2, height2 = fig2.get_size_inches()
        assert math.isclose(height2, width2)
        assert math.isclose(scale, width2 / width1)
        assert math.isclose(scale, height2 / height1)

        fig3, _ = demesdraw.utils.get_fig_axes(scale=1, aspect=aspect)
        width3, height3 = fig3.get_size_inches()
        assert math.isclose(aspect, height3 / width3)
        fig4, _ = demesdraw.utils.get_fig_axes(scale=scale, aspect=aspect)
        width4, height4 = fig4.get_size_inches()
        assert math.isclose(aspect, height4 / width4)
        assert math.isclose(scale, width4 / width3)
        assert math.isclose(scale, height4 / height3)

    def test_tight_layout(self):
        # tight layout is a good default
        fig, _ = demesdraw.utils.get_fig_axes()
        assert fig.get_tight_layout()

    def test_constrained_layout(self):
        # constrained layout should be possible
        fig, _ = demesdraw.utils.get_fig_axes(constrained_layout=True)
        assert not fig.get_tight_layout()
        assert fig.get_constrained_layout()

    def test_multiple_axes(self):
        _, axs = demesdraw.utils.get_fig_axes(nrows=2)
        assert axs.shape == (2,)
        _, axs = demesdraw.utils.get_fig_axes(ncols=2)
        assert axs.shape == (2,)
        _, axs = demesdraw.utils.get_fig_axes(nrows=2, ncols=2)
        assert axs.shape == (2, 2)


class TestSizeMax:
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_size_max(self, graph):
        size_max = demesdraw.utils.size_max(graph)
        for deme in graph.demes:
            for epoch in deme.epochs:
                assert size_max >= epoch.start_size
                assert size_max >= epoch.end_size


class TestSizeMin:
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_size_min(self, graph):
        size_min = demesdraw.utils.size_min(graph)
        for deme in graph.demes:
            for epoch in deme.epochs:
                assert size_min <= epoch.start_size
                assert size_min <= epoch.end_size


class TestLogTimeHeuristic:
    # Don't test the behaviour, just the interface.
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_log_time_heuristic(self, graph):
        log_time = demesdraw.utils.log_time_heuristic(graph)
        assert log_time in [True, False]


class TestLogSizeHeuristic:
    # Don't test the behaviour, just the interface.
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_log_size_heuristic(self, graph):
        log_size = demesdraw.utils.log_size_heuristic(graph)
        assert log_size in [True, False]
