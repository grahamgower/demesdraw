import pathlib
import functools

import demes
import matplotlib

matplotlib.use("Agg")


@functools.lru_cache(maxsize=None)
def example_files():
    cwd = pathlib.Path(__file__).parent.resolve()
    example_dir = cwd / "../examples"
    files = list(example_dir.glob("**/*.yaml"))
    assert len(files) > 1
    return files


@functools.lru_cache(maxsize=None)
def example_graphs():
    return [demes.load(fn) for fn in example_files()]


@functools.lru_cache(maxsize=None)
def example_demes():
    return [deme for graph in example_graphs() for deme in graph.demes]
