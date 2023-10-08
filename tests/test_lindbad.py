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

@pytest.mark.parametrize('map', ['serial', 'parallel'])
def test_constevo_finalstate(constant_solver: PDPSolver, map: str) -> None:
    initial_state = qt.basis(2, 0)
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
        assert (traj.final_state - initial_state).norm() < EPSI

@pytest.mark.parametrize('map', ['serial', 'parallel'])
def test_constevo_states(constant_solver: PDPSolver, map: str) -> None:
    initial_state = qt.basis(2, 0)
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

    initial_dm = qt.ket2dm(initial_state)
    for state in result.average_states:
        assert (state - initial_dm).norm() < EPSI


def test_unitaryevo():
    H = qt.sigmax()
    system = LindbladUnraveling(H, [], [])
    initial_state = qt.basis(2, 0)
    unitary_solver = PDPSolver(system)

    e_ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    analytical_result = (
        lambda t: 0,
        lambda t: -2 * np.cos(t) * np.sin(t),
        lambda t: np.cos(t)**2 - np.sin(t)**2,
    )
    
    n_traj = 3
    tlist = np.linspace(0, 10, 100)
    unitary_solver.options = {'map': 'serial',
                              'keep_runs_results': True,
                              'max_step': 0.1}
    result = unitary_solver.run(initial_state, tlist,
                                ntraj=n_traj, e_ops=e_ops)
    
    assert result.num_trajectories == n_traj

    expects = [traj.expect for traj in result.trajectories]
    expects.append(result.average_expect)
    for expect in expects:
        assert len(expect) == len(e_ops)
        for numerical, analytical in zip(expect, analytical_result):
            assert len(numerical) == len(tlist)
            for t, x in zip(tlist, numerical):
                assert np.abs(x - analytical(t)) < EPSI
