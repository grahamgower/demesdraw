import math

import demes
import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from matplotlib.backend_bases import MouseEvent
import numpy as np

import demesdraw
import tests


def images_equal(file1, file2):
    # Matplotlib's compare_images() function has a weird return value.
    # This wrapper just makes test assertions clearer.
    cmp = compare_images(file1, file2, tol=0)
    return cmp is None


class TestSizeHistory:
    def check_size_history(self, graph, **kwargs):
        ax = demesdraw.size_history(graph, **kwargs)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    @pytest.mark.filterwarnings("ignore:.*log scale.*:UserWarning:demesdraw.utils")
    @pytest.mark.parametrize("log_size", [True, False])
    @pytest.mark.parametrize("log_time", [True, False])
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_log_params(self, graph, log_time, log_size):
        self.check_size_history(graph, log_time=log_time, log_size=log_size)

    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_annotate_epochs_invert_x(self, graph):
        self.check_size_history(graph, invert_x=True, annotate_epochs=True)


class TestTubes:
    def check_tubes(self, graph, seed=1234, **kwargs):
        ax = demesdraw.tubes(graph, seed=seed, **kwargs)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    @pytest.mark.filterwarnings("ignore:.*log scale.*:UserWarning:demesdraw.utils")
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

    @pytest.mark.parametrize("scale_bar", [True, False])
    @pytest.mark.parametrize("graph", tests.example_graphs())
    def test_scale_bar(self, graph, scale_bar):
        self.check_tubes(graph, scale_bar=scale_bar)

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

    def test_format_coord(self):
        # The Axes object's format_coord() function gives the text for the
        # status bar in interactive figures. We want this to indicate the
        # time corresponding to the mouse cursor location.
        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A")
        graph = b.resolve()
        ax = demesdraw.tubes(graph)
        w, h = ax.figure.canvas.get_width_height()
        status_text = ax.format_coord(w / 2, h / 2)
        assert "x" not in status_text
        assert "y" not in status_text
        assert "time" in status_text

    @pytest.mark.parametrize("log_time", [True, False])
    @pytest.mark.usefixtures("tmp_path")
    def test_mouseover_deme(self, log_time, tmp_path):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A")
        graph = b.resolve()

        ax = demesdraw.tubes(graph, log_time=log_time)
        ax.figure.savefig(tmp_path / "fig1.png")

        # Simulate mouseover in the middle of the figure.
        # An annotation should appear, showing deme info.
        w, h = ax.figure.canvas.get_width_height()
        event = MouseEvent("motion_notify_event", ax.figure.canvas, w / 2, h / 2)
        ax.figure.canvas.callbacks.process("motion_notify_event", event)
        ax.figure.savefig(tmp_path / "fig2.png")

        assert not images_equal(tmp_path / "fig1.png", tmp_path / "fig2.png")

        # Simulate mouseover at the edge of the figure.
        # The annotation should disappear.
        event = MouseEvent("motion_notify_event", ax.figure.canvas, 1, 1)
        ax.figure.canvas.callbacks.process("motion_notify_event", event)
        ax.figure.savefig(tmp_path / "fig3.png")

        assert images_equal(tmp_path / "fig1.png", tmp_path / "fig3.png")

    @pytest.mark.parametrize("log_time", [True, False])
    @pytest.mark.usefixtures("tmp_path")
    def test_mouseover_pulse(self, log_time, tmp_path):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A")
        b.add_deme("B")
        b.add_pulse(sources=["A"], dest="B", proportions=[0.1], time=100)
        graph = b.resolve()

        ax = demesdraw.tubes(graph, log_time=log_time)
        ax.figure.savefig(tmp_path / "fig1.png")

        # Simulate moving the mouse over the pulse lines.
        # An annotation should appear, showing pulse info.
        num_pulse_lines = 0
        for line in ax.lines:
            if not (
                hasattr(line, "_demesdraw_data") and "pulse" in line._demesdraw_data
            ):
                continue
            xs, ys = line.get_xdata(), line.get_ydata()
            x, y = ax.transData.transform((np.mean(xs), ys[0]))
            event = MouseEvent("motion_notify_event", ax.figure.canvas, x, y)
            ax.figure.canvas.callbacks.process("motion_notify_event", event)
            ax.figure.savefig(tmp_path / "fig2.png")

            assert not images_equal(tmp_path / "fig1.png", tmp_path / "fig2.png")
            num_pulse_lines += 1
        assert num_pulse_lines > 0

        # Simulate mouseover at the edge of the figure.
        # The annotation should disappear.
        event = MouseEvent("motion_notify_event", ax.figure.canvas, 1, 1)
        ax.figure.canvas.callbacks.process("motion_notify_event", event)
        ax.figure.savefig(tmp_path / "fig3.png")

        assert images_equal(tmp_path / "fig1.png", tmp_path / "fig3.png")

    @pytest.mark.parametrize("log_time", [True, False])
    @pytest.mark.usefixtures("tmp_path")
    def test_mouseover_migration(self, log_time, tmp_path):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("A")
        b.add_deme("B")
        b.add_migration(demes=["A", "B"], rate=1e-5)
        graph = b.resolve()

        ax = demesdraw.tubes(graph, log_time=log_time)
        ax.figure.savefig(tmp_path / "fig1.png")

        # Simulate moving the mouse over the migration lines.
        # An annotation should appear, showing migration info.
        num_migration_lines = 0
        for line in ax.lines:
            if not (
                hasattr(line, "_demesdraw_data") and "migration" in line._demesdraw_data
            ):
                continue
            xs, ys = line.get_xdata(), line.get_ydata()
            x, y = ax.transData.transform((np.mean(xs), ys[0]))
            event = MouseEvent("motion_notify_event", ax.figure.canvas, x, y)
            ax.figure.canvas.callbacks.process("motion_notify_event", event)
            ax.figure.savefig(tmp_path / "fig2.png")

            assert not images_equal(tmp_path / "fig1.png", tmp_path / "fig2.png")
            num_migration_lines += 1
        assert num_migration_lines > 0

        # Simulate mouseover at the edge of the figure.
        # The annotation should disappear.
        event = MouseEvent("motion_notify_event", ax.figure.canvas, 1, 1)
        ax.figure.canvas.callbacks.process("motion_notify_event", event)
        ax.figure.savefig(tmp_path / "fig3.png")

        assert images_equal(tmp_path / "fig1.png", tmp_path / "fig3.png")
