# Non-Hermitian Pseudomodes for Strongly Coupled Open Quantum Systems:<br> Unravelings, Correlations and Thermodynamics

This repository contains the code used to generate the figures in arXiv:XXX.
The code is written in Python using QuTiP, the [Quantum Toolbox in Python](https://github.com/qutip/qutip/).

## Organization

`src/pdp` —
A python module for the simulation of generic piecewise deterministic jump processes as described in Appendix D of the paper.
Comes with a specification for the regular unraveling of Lindblad equations (`src/pdp/lindblad.py`) and with multiple unravelings for the pseudo-Lindblad equations discussed in Section III.D and Appendix C (`src/pdp/pseudomodes.py`).

`tests` —
Unit tests for the module described above.
Run with `pytest tests`.
Due to the stochastic nature of the module, some tests may fail spuriously.

`example1.py` —
Run to generate trajectories for example 1.
The script will generate files named `result-{i}-{class}.qu`, where "i" is a running index and "class" ranges over possible unravelings.
Each such file contains a `qutip.Result` object obtained by averaging over `1000` trajectories.
The script continues to generate trajectories until it is stopped by a keyboard interrupt.

`result-0-{class}.qu` —
Example results obtained from running `example1.py`.
For the plots in the paper, we generated a total of 536 such files per unraveling.
These files are not included here due to space limitations.

`mpl_setup.py` —
Matplotlib configuration and color definitions.

`example1.ipynb` —
Performs short calculations with `qutip.mesolve` and `qutip.HEOMSolver`, and generates Figures 1, 2, and 5 in the paper.

`example1.ipynb` —
Performs short calculations with `qutip.mesolve` and `qutip.HEOMSolver`, and generates Figures 3 and 4 in the paper.
The calculation results are stored in the `example2-{W}.qu` files.
If such a file exists, the calculation result is automatically loaded from the file instead.

## Installation

The code is based on an early alpha version of QuTiP version 5.
As of writing this note, the code does not work any more with current alpha builds of QuTiP 5.
To run this code, check out [QuTiP commit `4ce8829e`](https://github.com/qutip/qutip/commit/4ce8829edf00cbcf8e60b86b6bad60d9621a64f3) and follow the instructions [here](https://qutip.org/docs/latest/installation.html#installing-from-source) to compile QuTiP from source.

After the full release of QuTiP 5, we intend to update this repository.