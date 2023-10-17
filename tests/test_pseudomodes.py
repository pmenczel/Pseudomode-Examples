import pytest
from pdp import (
    PDPSolver, StandardPseudoUnraveling, AlternativePseudoUnraveling,
    StandardPseudoUnravelingEN, AlternativePseudoUnravelingEN)

import numpy as np
import qutip as qt


EPSI = 1e-6
ALL_SOLVERS = [StandardPseudoUnraveling, AlternativePseudoUnraveling,
               StandardPseudoUnravelingEN, AlternativePseudoUnravelingEN]


@pytest.fixture
def constant_solver(request) -> PDPSolver:
    solver_class = request.param

    H = 0 * qt.sigmaz()
    system = solver_class(H, [], [])
    return PDPSolver(system)


@pytest.mark.parametrize('constant_solver', ALL_SOLVERS, indirect=True)
def test_constevo_startstep(constant_solver: PDPSolver) -> None:
    initial_state = qt.basis(2, 0)
    initial_dm = qt.ket2dm(initial_state)

    constant_solver.start(initial_state, 0)
    final_state = constant_solver.step(1)
    assert (final_state - initial_dm).norm() < EPSI


@pytest.mark.parametrize('constant_solver', ALL_SOLVERS, indirect=True)
@pytest.mark.parametrize('map', ['serial', 'parallel'])
def test_constevo_finalstate(constant_solver: PDPSolver, map: str) -> None:
    initial_state = qt.basis(2, 0)
    initial_dm = qt.ket2dm(initial_state)

    options = {'map': map,
               'keep_runs_results': True,
               'store_final_state': True,
               'store_states': False}
    constant_solver.options = options

    n_traj = 3
    n_times = 100
    result = constant_solver.run(initial_state,
                                 np.linspace(0, 10, n_times), ntraj=n_traj)
    assert result.num_trajectories == n_traj
    for traj in result.trajectories:
        assert len(traj.collapse) == 0
        assert (traj.final_state - initial_dm).norm() < EPSI


@pytest.mark.parametrize('solver_class', ALL_SOLVERS)
def test_zero_rate(solver_class) -> None:
    system = solver_class(
        0 * qt.sigmaz(), [qt.sigmam()], [0])
    test_constevo_finalstate(PDPSolver(system), 'serial')


@pytest.mark.parametrize('constant_solver', ALL_SOLVERS, indirect=True)
@pytest.mark.parametrize('map', ['serial', 'parallel'])
def test_constevo_states(constant_solver: PDPSolver, map: str) -> None:
    initial_state = qt.basis(2, 0)
    initial_dm = qt.ket2dm(initial_state)

    options = {'map': map,
               'keep_runs_results': False,
               'store_final_state': False,
               'store_states': True}
    constant_solver.options = options

    n_traj = 3
    n_times = 100
    result = constant_solver.run(initial_state,
                                 np.linspace(0, 10, n_times), ntraj=n_traj)
    assert result.num_trajectories == n_traj
    assert len(result.average_states) == n_times

    for state in result.average_states:
        assert (state - initial_dm).norm() < EPSI