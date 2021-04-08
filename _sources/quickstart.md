---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import demes
import demesdraw

# When plotting inside a notebook, it's best to use the vector format SVG,
# instead of the ipython default (PNG, a raster format).
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("svg")
```

# Quickstart

DemesDraw is a python package for drawing
[Demes](https://github.com/popsim-consortium/demes-spec) demographic models.
It can be installed with pip.
```
python -m pip install git+https://github.com/grahamgower/demesdraw.git
```

```{warning}
DemesDraw is currently focused on providing figures for the Demes
documentation. As such, the functionality and API are very preliminary.
However, we hope to expand DemesDraw's capabilities in the future,
so backwards-incompatible changes should be expected.
```

# Population size history


The {func}`.size_history` function plots the population size as a
function of time for each deme in the graph. This is great for single-deme
graphs like the ZigZag model from Schiffels & Durbin et al. (2014).

```{code-cell}
:tags: [hide-input]

zigzag = demes.load("../examples/stdpopsim/HomSap__Zigzag_1S14.yaml")
demesdraw.size_history(zigzag, invert_x=True, log_time=True);
```

Multi-deme models can also be plotted with {func}`.size_history`.
E.g., below we plot the human out-of-Africa model from Gutenkunst et al. (2009).
Lines for distinct demes are given different colours and widths,
with thinner lines drawn on top of thicker lines. Ancestor/descendant
relationships are indicated as dotted lines.

```{code-cell}
:tags: [hide-input]

gutenkunst_ooa = demes.load("../examples/stdpopsim/HomSap__OutOfAfrica_3G09.yaml")
demesdraw.size_history(gutenkunst_ooa, invert_x=True, log_size=True);
```

# Demes as tubes

The {func}`.tubes()` function plots a "demes as tubes"-style schematic
overview. Compared with the size-history figure above, this more fully
captures the relationships between demes.
Ancestor/descendant relationships are indicated as solid lines with
open arrow heads pointing from ancestor to descendant, and migrations are drawn
as thin lines with closed arrow heads pointing from source to destination.

```{code-cell}
:tags: [hide-input]

gutenkunst_ooa = demes.load("../examples/stdpopsim/HomSap__OutOfAfrica_3G09.yaml")
demesdraw.tubes(gutenkunst_ooa, num_lines_per_migration=6, log_time=True, seed=1234);
```
