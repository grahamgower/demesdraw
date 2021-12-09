import sys
import demes
import msprime
import daiquiri

test_case = """\
time_units: generations
defaults:
  epoch: {start_size: 1000}
demes:
- name: A
- name: B
  ancestors: [A]
  start_time: 6000
- name: C
  ancestors: [B]
  start_time: 2000
- name: D
  ancestors: [C]
  start_time: 1000
migrations:
- demes: [A, D]
  rate: 1e-5
"""

graph = demes.loads(test_case)
dbg = msprime.Demography.from_demes(graph).debug()
print(dbg)
daiquiri.setup(level="DEBUG")
t = dbg.mean_coalescence_time({"A": 1, "C": 1}, max_iter=20)
