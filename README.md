# DemesDraw

`demesdraw` is a Python package that contains drawing functions for
[Demes](https://popsim-consortium.github.io/demes-spec-docs/main/)
demographic models, using `matplotlib` to create the figures.
DemesDraw offers both a command line interface, and a Python API.
Feedback is very welcome.


# Installation

```
$ python3 -m pip install demesdraw
```

# Usage

## Command line

```
$ demesdraw tubes --log-time \
	examples/stdpopsim/HomSap__AmericanAdmixture_4B11.yaml \
	AmericanAdmixture_4B11_tubes.svg
```

![stdpopsim/AmericanAdmixture_4B11 as tubes](https://raw.githubusercontent.com/grahamgower/demesdraw/main/docs/_static/AmericanAdmixture_4B11_tubes.svg)

## Python API

```
import demes
import demesdraw

graph = demes.load("examples/stdpopsim/HomSap__AmericanAdmixture_4B11.yaml")
ax = demesdraw.size_history(graph, log_time=True)
ax.figure.savefig("AmericanAdmixture_4B11_size_history.svg")
```

![stdpopsim/AmericanAdmixture_4B11 size history](https://raw.githubusercontent.com/grahamgower/demesdraw/main/docs/_static/AmericanAdmixture_4B11_size_history.svg)

# Documentation

Complete API and CLI details are available in the 
[DemesDraw documentation](https://grahamgower.github.io/demesdraw).
