# DemesDraw

`demesdraw` is a Python package that contains drawing functions for
[Demes](https://popsim-consortium.github.io/demes-spec-docs/main/)
demographic models, using `matplotlib` to create the figures.
DemesDraw offers both a command line interface, and a Python API.
Feedback is very welcome.


# Installation

Install with pip:
```
$ python3 -m pip install demesdraw
```

Or with conda:
```
$ conda install -c conda-forge demesdraw
```

# Example usage

## Command line interface (CLI)

The CLI can be used to quickly plot a Demes YAML file.
Any file format supported by matplotlib can be specified,
but a vector format such as svg or pdf is recommended.

```
$ demesdraw tubes --log-time \
	examples/stdpopsim/HomSap__AmericanAdmixture_4B11.yaml \
	AmericanAdmixture_4B11_tubes.svg
```

![stdpopsim/AmericanAdmixture_4B11 as tubes](https://raw.githubusercontent.com/grahamgower/demesdraw/main/docs/_static/AmericanAdmixture_4B11_tubes.svg)


## Python API

Compared with the CLI, the Python API provides additional control.
In the following example, the horizontal positions of the demes
are chosen manually and the names of extinct demes are moved to
a legend.

```python
import demes
import demesdraw

graph = demes.load("examples/stdpopsim/HomSap__AmericanAdmixture_4B11.yaml")
w = demesdraw.utils.separation_heuristic(graph)
positions = dict(ancestral=0, AMH=0, AFR=0, OOA=1.5 * w, EAS=1 * w, EUR=2 * w, ADMIX=-w)
ax = demesdraw.tubes(graph, log_time=True, positions=positions, labels="xticks-legend")
ax.figure.savefig("AmericanAdmixture_4B11_tubes_custom.svg")
```

![stdpopsim/AmericanAdmixture_4B11 tubes_custom](https://raw.githubusercontent.com/grahamgower/demesdraw/main/docs/_static/AmericanAdmixture_4B11_tubes_custom.svg)

# Documentation

Complete API and CLI details are available in the 
[DemesDraw documentation](https://grahamgower.github.io/demesdraw).
