import pathlib

import pytest
import demes


def example_files():
    cwd = pathlib.Path(__file__).parent.resolve()
    example_dir = cwd / "example_graphs"
    files = list(example_dir.glob("**/*.yaml"))
    assert len(files) > 1
    return files


def example_graphs():
    return [demes.load(fn) for fn in example_files()]


def example_demes():
    demes = []
    for graph in example_graphs():
        demes.extend(graph.demes)
    return demes


@pytest.fixture(scope="module", params=example_graphs())
def example_graph(request):
    return request.param


@pytest.fixture(scope="module", params=example_demes())
def example_deme(request):
    return request.param
