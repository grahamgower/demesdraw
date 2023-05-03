# Changelog

## [0.4.0] - 2023-05-03

* Documented the layout algorithm used in tubes plots.
* Added `scale_bar` option to `demesdraw.tubes()`, and `--scale-bar` option
  to the `demesdraw tubes` CLI, to draw a scale bar that indicates the
  population size.
* Removed dependency on `cvxpy` by going back to `scipy`
  for optimisation of the deme positions. The trust-constr
  method with linear constraints seems to work well.

## [0.3.1] - 2022-09-19

* Added mouseover annotation popups for interactive tubes plots.
* Dropped support for Python 3.6.
* Performance improvement when calculating tube positions.
* Fix incompatibility with "legend" labels and matplotlib 3.6.0.

## [0.3.0] - 2022-01-08

* Improved the default positions of demes in `demesdraw.tubes()`
  for tree-like models and more elaborate models with many demes.
  The previously used optimisation procedure (scipy's SLSQP) has been
  removed in favour of constrained convex optimisation using `cvxpy`.
* Increased the default amount of space that separates demes in
  `demesdraw.tubes()`, and use more space when there are more
  contemporary demes (see `demesdraw.utils.separation_heuristic()`).
* Filter numpy warnings about overflow when using a log scale.
  The overflow in question is harmless and these warnings only serve
  to confuse users.

**Breaking change**

* Removed the `optimisation_rounds` parameter to `demesdraw.tubes()`.

## [0.2.0] - 2021-12-01

**Breaking change**

* Updates for demes 0.2.0, which changes the syntax for pulses
  from using a single `source` and `proportion` to using a list of
  `sources` and a list of `proportions`.

## [0.1.4] - 2021-07-28

* Added `max_time` parameter to `demesdraw.tubes()`.
* Don't draw ancestry lines for vertically stacked demes.
* Pin demes < 0.2 in install_requires. 
* Made some `demesdraw.utils` functions public and cleaned up namespace.
* Added CLI, with `tubes` and `size_history` subcommands.

## [0.1.3] - 2021-06-22

* Fixed problem with numpy 1.21.

## [0.1.2] - 2021-06-10

* Reinstate support for Python 3.6.

## [0.1.1] - 2021-06-09

* Fix plotting of arrowheads (avoids a matplotlib warning; no visible change).
* Add ISC license file.
* Require demes >= 0.1.2 in install_requires.
* Add linear size function ({user}`noscode`, {pr}`23`).

## [0.1.0] - 2021-04-19

* Initial release, including `demes.tubes()` and `demes.size_history()` functions.
