import demes
import pytest
import numpy as np
import matplotlib

import demesdraw
import demesdraw.utils
from tests import example_graph, example_deme


class TestSizeHistory:
    @pytest.mark.parametrize("annotate_epochs", [True, False])
    @pytest.mark.parametrize("invert_x", [True, False])
    @pytest.mark.parametrize("log_size", [True, False])
    @pytest.mark.parametrize("log_time", [True, False])
    def test_examples(
        self, example_graph, log_time, log_size, invert_x, annotate_epochs
    ):
        ax = demesdraw.size_history(
            example_graph,
            log_time=log_time,
            log_size=log_size,
            invert_x=invert_x,
            annotate_epochs=annotate_epochs,
        )
        assert isinstance(ax, matplotlib.axes.Axes)


class TestSchematic:
    @pytest.mark.parametrize("log_time", [True, False])
    def test_examples(self, example_graph, log_time):
        ax = demesdraw.schematic(
            example_graph,
            optimisation_rounds=1,
            log_time=log_time,
            seed=1234,
        )
        assert isinstance(ax, matplotlib.axes.Axes)


class TestInfStartTime:
    @pytest.mark.parametrize("log_scale", [True, False])
    def test_examples(self, example_graph, log_scale):
        t = demesdraw.utils.inf_start_time(example_graph, 0.1, log_scale)
        assert t > 0
        assert not np.isinf(t)
        start_times = [
            epoch.start_time
            for deme in example_graph.demes
            for epoch in deme.epochs
            if not np.isinf(epoch.start_time)
        ]
        end_times = [
            epoch.end_time for deme in example_graph.demes for epoch in deme.epochs
        ]
        assert t > np.max(start_times)
        assert t > np.max(end_times)

    @pytest.mark.parametrize("log_scale", [True, False])
    def test_one_epoch(self, log_scale):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        graph = b.resolve()
        t = demesdraw.utils.inf_start_time(graph, 0.1, log_scale)
        assert t > 0
        assert not np.isinf(t)


class TestSizeOfDemeAtTime:
    def test_examples(self, example_deme):
        N = demesdraw.utils.size_of_deme_at_time(example_deme, example_deme.start_time)
        assert N == example_deme.epochs[0].start_size
        N = demesdraw.utils.size_of_deme_at_time(example_deme, example_deme.end_time)
        assert N == example_deme.epochs[-1].end_size

    def test_bad_time(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1, end_time=100)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1)])
        graph = b.resolve()
        with pytest.raises(ValueError, match="doesn't exist"):
            demesdraw.utils.size_of_deme_at_time(graph["A"], 10)
        with pytest.raises(ValueError, match="doesn't exist"):
            demesdraw.utils.size_of_deme_at_time(graph["B"], 200)
