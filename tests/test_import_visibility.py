import pytest

from demesdraw import *


def test_public_symbols():
    size_history
    tubes


def test_nonpublic_symbols():
    with pytest.raises(NameError):
        utils


from demesdraw.utils import *  # noqa: E402


def test_utils_public_symbols():
    get_fig_axes
    size_max
    size_min
    log_size_heuristic
    log_time_heuristic
    separation_heuristic


def test_utils_nonpublic_symbols():
    with pytest.raises(NameError):
        _get_colours


def test_dir():
    import demesdraw

    dir_demesdraw = dir(demesdraw)
    assert "size_history" in dir_demesdraw
    assert "tubes" in dir_demesdraw

    assert "utils" not in dir_demesdraw


def test_utils_dir():
    import demesdraw.utils

    dir_ddu = dir(demesdraw.utils)
    assert "get_fig_axes" in dir_ddu
    assert "size_max" in dir_ddu
    assert "size_min" in dir_ddu
    assert "log_size_heuristic" in dir_ddu
    assert "log_time_heuristic" in dir_ddu

    assert "_get_colours" not in dir_ddu
    assert "size_history" not in dir_ddu
    assert "tubes" not in dir_ddu
