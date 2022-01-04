import collections
import itertools
import math

import demes
import pytest
import matplotlib.pyplot as plt
import numpy as np

from demesdraw import utils
import tests


class TestInfStartTime:
    @pytest.mark.filterwarnings("ignore:.*log scale.*:UserWarning:demesdraw.utils")
    @pytest.mark.parametrize("log_scale", [True, False])
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_time_is_reasonable(self, graph, log_scale):
        t = utils._inf_start_time(graph, 0.1, log_scale)
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
        t = utils._inf_start_time(graph, 0.1, log_scale)
        assert t > 0
        assert not math.isinf(t)


class TestGetFigAxes:
    def teardown_method(self):
        plt.close("all")

    def test_return_values_are_consistent(self):
        fig, ax = utils.get_fig_axes()
        assert fig.get_axes() == [ax]
        assert ax.get_figure() == fig

    @pytest.mark.parametrize("aspect", [0.5, 1, 1.5, 3 / 4, 9 / 16, 10 / 16])
    @pytest.mark.parametrize("scale", [0.5, 1, 1.5])
    def test_aspect_and_scale(self, aspect, scale):
        fig1, _ = utils.get_fig_axes(scale=1, aspect=1)
        width1, height1 = fig1.get_size_inches()
        assert math.isclose(height1, width1)
        fig2, _ = utils.get_fig_axes(scale=scale, aspect=1)
        width2, height2 = fig2.get_size_inches()
        assert math.isclose(height2, width2)
        assert math.isclose(scale, width2 / width1)
        assert math.isclose(scale, height2 / height1)

        fig3, _ = utils.get_fig_axes(scale=1, aspect=aspect)
        width3, height3 = fig3.get_size_inches()
        assert math.isclose(aspect, height3 / width3)
        fig4, _ = utils.get_fig_axes(scale=scale, aspect=aspect)
        width4, height4 = fig4.get_size_inches()
        assert math.isclose(aspect, height4 / width4)
        assert math.isclose(scale, width4 / width3)
        assert math.isclose(scale, height4 / height3)

    def test_tight_layout(self):
        # tight layout is a good default
        fig, _ = utils.get_fig_axes()
        assert fig.get_tight_layout()

    def test_constrained_layout(self):
        # constrained layout should be possible
        fig, _ = utils.get_fig_axes(constrained_layout=True)
        assert not fig.get_tight_layout()
        assert fig.get_constrained_layout()

    def test_multiple_axes(self):
        _, axs = utils.get_fig_axes(nrows=2)
        assert axs.shape == (2,)
        _, axs = utils.get_fig_axes(ncols=2)
        assert axs.shape == (2,)
        _, axs = utils.get_fig_axes(nrows=2, ncols=2)
        assert axs.shape == (2, 2)


class TestSizeMax:
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_size_max(self, graph):
        size_max = utils.size_max(graph)
        for deme in graph.demes:
            for epoch in deme.epochs:
                assert size_max >= epoch.start_size
                assert size_max >= epoch.end_size


class TestSizeMin:
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_size_min(self, graph):
        size_min = utils.size_min(graph)
        for deme in graph.demes:
            for epoch in deme.epochs:
                assert size_min <= epoch.start_size
                assert size_min <= epoch.end_size


class TestLogTimeHeuristic:
    # Don't test the behaviour, just the interface.
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_log_time_heuristic(self, graph):
        log_time = utils.log_time_heuristic(graph)
        assert log_time in [True, False]


class TestLogSizeHeuristic:
    # Don't test the behaviour, just the interface.
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_log_size_heuristic(self, graph):
        log_size = utils.log_size_heuristic(graph)
        assert log_size in [True, False]


class TestInteractionIndices:
    @pytest.mark.parametrize("unique", (True, False))
    @pytest.mark.parametrize("n", (1, 2, 3))
    def test_no_interations_isolated_demes(self, n, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        for j in range(n):
            b.add_deme(f"deme{j}")
        graph = b.resolve()
        interactions = utils.interactions_indices(graph, unique=unique)
        assert len(interactions) == 0

    @pytest.mark.parametrize("unique", (True, False))
    def test_no_interactions_common_ancestor(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", ancestors=["a"])
        b.add_deme("c", ancestors=["a"])
        graph = b.resolve()
        interactions = utils.interactions_indices(graph, unique=unique)
        assert len(interactions) == 0

    @pytest.mark.parametrize("n_migrations", (1, 2, 3))
    def test_asymmetric_migrations(self, n_migrations):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", ancestors=["a"])
        b.add_deme("c", ancestors=["a"])
        times = np.linspace(0, 100, n_migrations + 1)
        for start_time, end_time in zip(times[1:], times[:-1]):
            b.add_migration(
                source="b",
                dest="c",
                start_time=start_time,
                end_time=end_time,
                rate=1e-5,
            )
        graph = b.resolve()
        interactions = utils.interactions_indices(graph, unique=False)
        assert (
            interactions == [(1, 2)] * n_migrations
            or interactions == [(2, 1)] * n_migrations
        )
        interactions = utils.interactions_indices(graph, unique=True)
        assert interactions == [(1, 2)] or interactions == [(2, 1)]

    @pytest.mark.parametrize("n_migrations", (1, 2, 3))
    def test_symmetric_migrations(self, n_migrations):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", ancestors=["a"])
        b.add_deme("c", ancestors=["a"])
        times = np.linspace(0, 100, n_migrations + 1)
        for start_time, end_time in zip(times[1:], times[:-1]):
            b.add_migration(
                demes=["b", "c"], start_time=start_time, end_time=end_time, rate=1e-5
            )
        graph = b.resolve()
        interactions = utils.interactions_indices(graph, unique=False)
        assert len(interactions) == 2 * n_migrations
        counts = collections.Counter(interactions)
        assert len(counts) == 2
        assert counts[(1, 2)] == n_migrations
        assert counts[(2, 1)] == n_migrations
        interactions = utils.interactions_indices(graph, unique=True)
        assert interactions == [(1, 2)] or interactions == [(2, 1)]

    @pytest.mark.parametrize("n_pulses", (1, 2, 3))
    def test_pulses(self, n_pulses):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", ancestors=["a"])
        b.add_deme("c", ancestors=["a"])
        for j in range(n_pulses):
            b.add_pulse(
                sources=["b"], dest="c", time=100 * j / n_pulses + 1, proportions=[0.1]
            )
        graph = b.resolve()
        interactions = utils.interactions_indices(graph, unique=False)
        assert (
            interactions == [(1, 2)] * n_pulses or interactions == [(2, 1)] * n_pulses
        )
        interactions = utils.interactions_indices(graph, unique=True)
        assert interactions == [(1, 2)] or interactions == [(2, 1)]


class TestLineCrossings:
    @pytest.mark.parametrize("unique", (True, False))
    @pytest.mark.parametrize("n", (1, 2, 3))
    def test_no_crossings_isolated_demes(self, n, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        for j in range(n):
            b.add_deme(f"deme{j}")
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 0
        for x in itertools.permutations(range(n)):
            assert utils._line_crossings(x, candidates) == 0

    @pytest.mark.parametrize("unique", (True, False))
    def test_no_crossings_common_ancestor(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", ancestors=["a"])
        b.add_deme("c", ancestors=["a"])
        b.add_pulse(sources=["b"], dest="c", time=50, proportions=[0.1])
        b.add_migration(demes=["b", "c"], rate=1e-5)
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 0
        for x in itertools.permutations(range(3)):
            assert utils._line_crossings(x, candidates) == 0

    @pytest.mark.parametrize("unique", (True, False))
    def test_ancestry_crossing(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a")
        b.add_deme("b", ancestors=["a"], start_time=100)
        b.add_deme("c", ancestors=["b"], start_time=50)
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 1

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 0
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 1
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 0

    @pytest.mark.parametrize("unique", (True, False))
    def test_ancestry_crossing_multiple_ancestors(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a")
        b.add_deme("b", ancestors=["a"], start_time=100)
        b.add_deme("c", ancestors=["a", "b"], proportions=[0.5, 0.5], start_time=50)
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 2

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 1
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 1
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 1

    @pytest.mark.parametrize("unique", (True, False))
    def test_ancestry_crossing_treelike(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", ancestors=["a"])
        b.add_deme("c", ancestors=["a"], epochs=[dict(end_time=50)])
        b.add_deme("d", ancestors=["c"])
        b.add_deme("e", ancestors=["c"])
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 2

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c", "d", "e"]), candidates) == 0
        assert utils._line_crossings(order(["a", "c", "b", "d", "e"]), candidates) == 2
        assert utils._line_crossings(order(["a", "d", "e", "b", "c"]), candidates) == 2
        assert utils._line_crossings(order(["d", "e", "a", "b", "c"]), candidates) == 2
        assert utils._line_crossings(order(["a", "d", "b", "c", "e"]), candidates) == 1
        assert utils._line_crossings(order(["a", "e", "b", "c", "d"]), candidates) == 1
        assert utils._line_crossings(order(["d", "b", "a", "c", "e"]), candidates) == 1
        assert utils._line_crossings(order(["e", "b", "a", "c", "d"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a", "d", "e"]), candidates) == 0

    @pytest.mark.parametrize("unique", (True, False))
    def test_pulse_crossing(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_deme("c")
        b.add_pulse(sources=["b"], dest="c", time=100, proportions=[0.1])
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 1

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 0
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 1
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 0

    @pytest.mark.parametrize("unique", (True, False))
    def test_pulse_crossing_multiple_sources(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_deme("c")
        b.add_pulse(sources=["a", "b"], dest="c", time=100, proportions=[0.1, 0.1])
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 2

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 1
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 1
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 1

    @pytest.mark.parametrize("unique", (True, False))
    def test_asymmetric_migration_crossing(self, unique):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_deme("c")
        b.add_migration(source="b", dest="c", rate=1e-5)
        graph = b.resolve()
        candidates = utils._get_line_candidates(graph, unique=unique)
        assert len(candidates) == 1

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 0
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 1
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 0

    def test_symmetric_migration_crossing(self):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_deme("c")
        b.add_migration(demes=["b", "c"], rate=1e-5)
        graph = b.resolve()
        # unique = False
        candidates = utils._get_line_candidates(graph, unique=False)
        assert len(candidates) == 2

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 0
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 2
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 2
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 0

        # unique = True
        candidates = utils._get_line_candidates(graph, unique=True)
        assert len(candidates) == 1

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 0
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 1
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 0

    def test_multiple_candidates(self):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("a")
        b.add_deme("b", ancestors=["a"], start_time=100)
        b.add_deme("c", ancestors=["b"], start_time=50)
        b.add_pulse(sources=["a"], dest="c", time=20, proportions=[0.1])
        b.add_migration(source="b", dest="c", rate=1e-5)
        graph = b.resolve()
        # unique = False
        candidates = utils._get_line_candidates(graph, unique=False)
        assert len(candidates) == 3

        def order(deme_names):
            pos = {name: j * 150 for j, name in enumerate(deme_names)}
            return np.array([pos[deme.name] for deme in graph.demes])

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 1
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 2
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 2
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 1

        # Check batch mode gives the same result.
        assert all(
            utils._line_crossings(
                np.stack(
                    [
                        order(["a", "b", "c"]),
                        order(["a", "c", "b"]),
                        order(["b", "a", "c"]),
                        order(["c", "a", "b"]),
                        order(["b", "c", "a"]),
                        order(["c", "b", "a"]),
                    ]
                ),
                candidates,
            )
            == [1, 0, 2, 2, 0, 1]
        )

        # unique = True
        candidates = utils._get_line_candidates(graph, unique=True)
        assert len(candidates) == 2

        assert utils._line_crossings(order(["a", "b", "c"]), candidates) == 1
        assert utils._line_crossings(order(["a", "c", "b"]), candidates) == 0
        assert utils._line_crossings(order(["b", "a", "c"]), candidates) == 1
        assert utils._line_crossings(order(["c", "a", "b"]), candidates) == 1
        assert utils._line_crossings(order(["b", "c", "a"]), candidates) == 0
        assert utils._line_crossings(order(["c", "b", "a"]), candidates) == 1

        # Check batch mode gives the same result.
        assert all(
            utils._line_crossings(
                np.stack(
                    [
                        order(["a", "b", "c"]),
                        order(["a", "c", "b"]),
                        order(["b", "a", "c"]),
                        order(["c", "a", "b"]),
                        order(["b", "c", "a"]),
                        order(["c", "b", "a"]),
                    ]
                ),
                candidates,
            )
            == [1, 0, 1, 1, 0, 1]
        )
