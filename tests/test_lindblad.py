import pytest
from pdp import LindbladUnraveling, PDPSolver

import numpy as np
import qutip as qt

from pdp.processes import InitialDM


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


def test_zero_rate() -> None:
    system = LindbladUnraveling(
        0 * qt.sigmaz(), [qt.sigmam()], [0])
    test_constevo_finalstate(PDPSolver(system), 'serial')


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


@pytest.mark.parametrize("initial_state", [
    *qt.sigmax().eigenstates()[-1],
    *qt.sigmay().eigenstates()[-1],
    *qt.sigmaz().eigenstates()[-1],
])
def test_decay(initial_state):
    sm = qt.sigmam()
    ground_state = qt.ket2dm(qt.basis(2, 1))
    rate = 1
    system = LindbladUnraveling(0 * qt.sigmaz(), [sm], [rate])

    n_traj = 10
    solver = PDPSolver(system, options={'keep_runs_results': True,
                                        'store_states': True,
                                        'map': 'serial',
                                        'max_step': 0.1})    
    tlist = np.linspace(0, 10, 100)
    result = solver.run(initial_state, tlist, n_traj)

    assert result.num_trajectories == n_traj

    for traj in result.trajectories:
        assert hasattr(traj, "collapse")
        num_jumps = len(traj.collapse)
        assert num_jumps == 0 or num_jumps == 1
        jump_time, ch = traj.collapse[0] if num_jumps == 1 else (np.inf, 0)
        assert ch == 0

        for i in range(1, len(tlist)):
            time = tlist[i]
            state: qt.Qobj = qt.ket2dm(traj.states[i])
            previous_state = qt.ket2dm(traj.states[i-1])

            if time > jump_time:
                assert state == ground_state
            
            state_down = np.abs(state.overlap(ground_state))
            prev_down = np.abs(previous_state.overlap(ground_state))
            assert state_down >= prev_down


def test_mixed_initial():
    initial_state = qt.thermal_dm(3, 1)
    ntraj = 10

    solver = PDPSolver(LindbladUnraveling(
        hamiltonian = 0 * qt.fock_dm(3),
        lindblad_ops = [], rates = []
        ), options={'store_states': True,
                    'keep_runs_results': True,
                    'map': 'serial'})
    
    ic = InitialDM(initial_state, ntraj)
    result = solver.run_mixed(ic, [0])

    assert len(result.average_states) == 1
    assert result.average_final_state == result.average_states[0]
    assert (result.average_states[0] - initial_state).norm() < EPSI


def test_rabi_oscillations():
    ntraj = 1000
    tlist = np.linspace(0, 5, 100)

    initial = qt.basis(2, 0)
    hamiltonian = qt.sigmax()
    lindblad_down = qt.sigmam()
    rate_down = .5
    lindblad_up = qt.sigmap()
    rate_up = .25

    system = LindbladUnraveling(hamiltonian,
                                [lindblad_down, lindblad_up],
                                [rate_down, rate_up])
    solver = PDPSolver(system, options={'map': 'parallel',
                                        'store_states': True})
    result = solver.run(initial, tlist, ntraj)

    for t, state in zip(tlist, result.average_states):
        analytical_y = (
            16 / 137 -
            16 / 139055 * np.exp(-9 * t / 16) * (
                1015 * np.cos(np.sqrt(1015) * t / 16) +
                283 * np.sqrt(1015) * np.sin(np.sqrt(1015) * t / 16)
            ))
        analytical_z = (
            -3 / 137 +
            4 / 139055 * np.exp(-9 * t / 16) * (
                35525 * np.cos(np.sqrt(1015) * t / 16) -
                233 * np.sqrt(1015) * np.sin(np.sqrt(1015) * t / 16)
            ))
        analytical_state = (qt.identity(2) / 2 +
                            analytical_y * qt.sigmay() / 2 +
                            analytical_z * qt.sigmaz() / 2)
        assert (state - analytical_state).norm() < 0.1
