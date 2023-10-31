import pytest
from pdp import (
    PDPSolver, StandardPseudoUnraveling, AlternativePseudoUnraveling,
    BreuerUnraveling, StandardPseudoUnravelingEN,
    UnravelingLikeAppendixC4, NonHermitianIC)

import numpy as np
import qutip as qt


EPSI = 1e-6
NORMAL_SOLVERS = [StandardPseudoUnraveling, AlternativePseudoUnraveling,
                  BreuerUnraveling, StandardPseudoUnravelingEN]
ALL_SOLVERS = NORMAL_SOLVERS + [UnravelingLikeAppendixC4]


@pytest.fixture
def constant_solver(request):
    solver_class = request.param

    H = 0 * qt.sigmaz()
    system = solver_class(H, [], [])
    return PDPSolver(system)


@pytest.mark.parametrize('constant_solver', ALL_SOLVERS, indirect=True)
def test_constevo_startstep(constant_solver):
    initial_state = qt.basis(2, 0)
    initial_dm = qt.ket2dm(initial_state)

    constant_solver.start(initial_state, 0)
    final_state = constant_solver.step(1)
    assert (final_state - initial_dm).norm() < EPSI


@pytest.mark.parametrize('constant_solver', ALL_SOLVERS, indirect=True)
@pytest.mark.parametrize('map', ['serial', 'parallel'])
def test_constevo_finalstate(constant_solver, map):
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


@pytest.mark.parametrize('process_class', ALL_SOLVERS)
def test_zero_rate(process_class):
    system = process_class(
        0 * qt.sigmaz(), [qt.sigmam()], [0])
    test_constevo_finalstate(PDPSolver(system), 'serial')


@pytest.mark.parametrize('constant_solver', ALL_SOLVERS, indirect=True)
@pytest.mark.parametrize('map', ['serial', 'parallel'])
def test_constevo_states(constant_solver, map):
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


def _hamiltonian_only_analytical(hx, hz, x0, z0, tlist):
    return [
        [(hx * (hx*x0+hz*z0) +
          hz * (hz*x0 - hx*z0) * np.cosh(2*t*np.sqrt(-hx**2 - hz**2))) /
          (hx**2 + hz**2) for t in tlist],
        [((hz*x0 - hx*z0) * np.sinh(2*t*np.sqrt(-hx**2 - hz**2)) /
          np.sqrt(-hx**2 - hz**2)) for t in tlist],
        [(hz * (hx*x0+hz*z0) +
          hx * (hx*z0 - hz*x0) * np.cosh(2*t*np.sqrt(-hx**2 - hz**2))) /
          (hx**2 + hz**2) for t in tlist]
    ]


def _hamiltonian_only_solver(process_class, hx, hz):
    H = hx * qt.sigmax() + hz * qt.sigmaz()
    return PDPSolver(process_class(H, [], []),
                     options={'keep_runs_results': True,
                              'map': 'serial',
                              'store_states': False,
                              'max_step': 0.05})


@pytest.mark.parametrize('process_class', NORMAL_SOLVERS)
@pytest.mark.parametrize('hx, hz', [(1, 1), (.5j, 1), (2j, 1),
                                    (1j, 0), (0, 1j)])
@pytest.mark.parametrize('initial_state, x0, z0', [
    (qt.basis(2, 0), 0, 1),
    ((qt.basis(2, 0) + qt.basis(2, 1)) / np.sqrt(2), 1, 0)])
def test_hamiltonian_only_pure(process_class, hx, hz, initial_state, x0, z0):
    tlist = np.linspace(0, 5, 100)
    ntraj = 5

    solution = _hamiltonian_only_analytical(complex(hx), complex(hz),
                                            x0, z0, tlist)
    solver = _hamiltonian_only_solver(process_class, hx, hz)
    result = solver.run(initial_state, tlist, ntraj=ntraj,
                        e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])
    assert result.num_trajectories == ntraj

    for traj in result.trajectories:
        np.testing.assert_allclose(traj.expect, solution, atol=EPSI, rtol=EPSI)


@pytest.mark.parametrize('hx, hz', [(1, 1), (.5j, 1), (2j, 1),
                                    (1j, 0), (0, 1j)])
@pytest.mark.parametrize('initial_state, x0, z0', [
    (qt.basis(2, 0), 0, 1),
    ((qt.basis(2, 0) + qt.basis(2, 1)) / np.sqrt(2), 1, 0)])
def test_hamiltonian_only_C4(hx, hz, initial_state, x0, z0):
    # for hermitian hamiltonian, trajectories are still deterministic
    if np.imag(hx) == 0 and np.imag(hz) == 0:
        test_hamiltonian_only_pure(UnravelingLikeAppendixC4,
                                   hx, hz, initial_state, x0, z0)
        return
    
    # otherwise, trajectories are stochastic now, have to increase ntraj
    tlist = np.linspace(0, 2.5, 50)
    ntraj = 1000

    solution = _hamiltonian_only_analytical(complex(hx), complex(hz),
                                            x0, z0, tlist)
    solver = _hamiltonian_only_solver(UnravelingLikeAppendixC4, hx, hz)
    solver.options = {'map': 'parallel', 'keep_runs_results': False}
    result = solver.run(initial_state, tlist, ntraj=ntraj,
                        e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])
    assert result.num_trajectories == ntraj

    np.testing.assert_allclose(solution, result.average_expect,
                               atol=.5, rtol=0.15)


@pytest.mark.parametrize('process_class', NORMAL_SOLVERS)
@pytest.mark.parametrize('hx, hz', [(1, 1), (.5j, 1), (2j, 1),
                                    (1j, 0), (0, 1j)])
@pytest.mark.parametrize('x0, z0', [(0, 0), (.5, 0), (0, .5j), (1j, -1j),
                                    (.2 + .3j, .5 + .75j)])
def test_hamiltonian_only_mixed(process_class, hx, hz, x0, z0):
    tlist = np.linspace(0, 5, 100)
    ntraj = 5

    initial = NonHermitianIC(
        qt.qeye(2) / 2 + x0 * qt.sigmax() / 2 + z0 * qt.sigmaz() / 2,
        ntraj)
    solution = _hamiltonian_only_analytical(complex(hx), complex(hz),
                                            x0, z0, tlist)
    solver = _hamiltonian_only_solver(process_class, hx, hz)
    result = solver.run_mixed(initial, tlist,
                              e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])
    assert result.num_trajectories == ntraj
    np.testing.assert_allclose(result.average_expect, solution,
                               atol=EPSI, rtol=EPSI)


@pytest.mark.slow
@pytest.mark.parametrize('process_class', ALL_SOLVERS)
@pytest.mark.parametrize('rate', [0, 1, -1, 1j, -1j])
def test_simple_decay(process_class, rate):
    tlist = np.linspace(0, 3, 25)
    if process_class == UnravelingLikeAppendixC4:
        ntraj = 2000
    else:
        ntraj = 1000

    initial = qt.basis(2, 0)
    solution = [
        (np.exp(-rate * t) * qt.fock_dm(2, 0) +
         (1 - np.exp(-rate * t)) * qt.fock_dm(2, 1)) for t in tlist]
    process = process_class(0 * qt.sigmaz(), [qt.sigmam()], [rate])
    solver = PDPSolver(process, options={'map': 'parallel', 'max_step': 0.1,
                                         'keep_runs_results': True,
                                         'store_states': True})
    result = solver.run(initial, tlist, ntraj)
    assert result.num_trajectories == ntraj

    for state1, state2 in zip(result.average_states, solution):
        assert (state1 - state2).norm() < 0.2 + 0.2 * state2.norm()

    for traj in result.trajectories:
        assert hasattr(traj, "collapse")
        if process_class != UnravelingLikeAppendixC4:
            num_jumps = len(traj.collapse)
            assert num_jumps == 0 or num_jumps == 1
            if num_jumps == 1:
                assert traj.collapse[0][1] == 0


@pytest.mark.slow
@pytest.mark.parametrize('process_class', ALL_SOLVERS)
def test_complex_example(process_class):
    tlist = np.linspace(0, 2.5, 100)
    if process_class == UnravelingLikeAppendixC4:
        ntraj = 10000
    else:
        ntraj = 2500

    H = qt.sigmaz() + .5j * qt.sigmax()
    L1 = qt.sigmam()
    rate1 = .5 + 1j
    L2 = qt.sigmaz()
    rate2 = .25j
    initial = qt.qeye(2) / 2 + 1j * qt.sigmax() / 2 - 1j * qt.sigmaz()

    solver = PDPSolver(process_class(H, [L1, L2], [rate1, rate2]),
                       options={'keep_runs_results': False,
                                'map': 'parallel',
                                'store_states': False,
                                'max_step': 0.05})
    mcsol = solver.run_mixed(NonHermitianIC(initial, ntraj), tlist,
                             e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])

    liouvillian = -1j * qt.spre(H) + 1j * qt.spost(H)
    for rate, L in [(rate1, L1), (rate2, L2)]:
        LdL = L.dag() * L
        liouvillian += rate * (
            qt.spre(L) * qt.spost(L.dag()) -
            .5 * qt.spre(LdL) -
            .5 * qt.spost(LdL)
        )
    mesol = qt.mesolve(liouvillian, initial, tlist, c_ops=None,
                       e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
                       options={'normalize_output': False})
    
    for i in range(3):
        expect_mc = mcsol.average_expect[i]
        expect_me = mesol.expect[i]
        half = int(len(tlist) / 2)
        np.testing.assert_allclose(expect_mc[:half], expect_me[:half],
                                   rtol=0, atol=.2)
        np.testing.assert_allclose(expect_mc[half:], expect_me[half:],
                                   rtol=0, atol=.5)
