# Changelog

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
