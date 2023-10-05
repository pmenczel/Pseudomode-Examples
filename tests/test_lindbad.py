import pytest
from pdp import LindbladUnraveling, PDPSolver

import numpy as np
import qutip as qt

EPSI = 1e-6

@pytest.fixture
def constant_solver() -> PDPSolver:
    H = 0 * qt.sigmaz()
    system = LindbladUnraveling(H, [], [])
    return PDPSolver(system)

def test_constevo_startstep(constant_solver: PDPSolver) -> None:
    initial_state = qt.basis(2, 0)
    constant_solver.start(initial_state, 0)
    final_state = constant_solver.step(1)
    assert (final_state - initial_state).norm() < EPSI

def test_constevo_serial(constant_solver: PDPSolver) -> None:
    initial_state = qt.basis(2, 0)
    options = {'map': 'serial',
               'keep_runs_results': True,
               'store_final_state': True,
               'store_states': False}
    constant_solver.options = options

    result = constant_solver.run(initial_state,
                                 np.linspace(0, 10, 100), ntraj=3)
    assert result.num_trajectories == 3
    for traj in result.trajectories:
        assert (traj.final_state - initial_state).norm() < EPSI

def test_constevo_parallel(constant_solver: PDPSolver) -> None:
    initial_state = qt.basis(2, 0)
    options = {'map': 'serial',
               'keep_runs_results': False,
               'store_final_state': False,
               'store_states': True}
    constant_solver.options = options

    result = constant_solver.run(initial_state,
                                 np.linspace(0, 10, 100), ntraj=3)
    assert result.num_trajectories == 3

    initial_dm = qt.ket2dm(initial_state)
    for state in result.average_states:
        assert (state - initial_dm).norm() < EPSI