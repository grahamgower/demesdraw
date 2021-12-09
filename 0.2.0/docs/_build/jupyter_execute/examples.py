#!/usr/bin/env python
# coding: utf-8

# # Examples

# In[1]:


import math
import pathlib

import demes
import demesdraw
import demesdraw.utils
import matplotlib.pyplot as plt

# Output SVG.
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("svg")


def plot_from_yaml(yaml_filename):
    graph = demes.load(yaml_filename)
    log_time = demesdraw.utils.log_time_heuristic(graph)
    log_size = demesdraw.utils.log_size_heuristic(graph)

    ax1 = demesdraw.size_history(
        graph,
        invert_x=True,
        log_time=log_time,
        log_size=log_size,
        title=example.name,
    )

    ax2 = demesdraw.tubes(
        graph,
        log_time=log_time,
        title=example.name,
    )

    plt.show(ax1.figure)
    plt.show(ax2.figure)
    
    plt.close(ax1.figure)
    plt.close(ax2.figure)
    print("\n\n")

    
# Plot each example yaml in the examples folder.
cwd = pathlib.Path(".").parent.resolve()
examples = list((cwd / ".." / "examples").glob("**/*.yaml"))
for example in sorted(examples):
    plot_from_yaml(example)

