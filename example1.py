import pdp

import numpy as np
import qutip as qt

import traceback


##### SYSTEM CONFIGURATION ####################################################

try:
    import mkl  # type: ignore
    mkl.set_num_threads(1)
except ImportError:
    pass

NUM_CPUS = 12


##### NUMERICS ################################################################

NUM_MATSUBARA = 0
PM_CUTOFF = [9, 3]
TLIST = np.linspace(0, 75, 250)


##### PARAMETERS ##############################################################

BETA = 1             # beta, inverse bath temp
COUP_STRENGTH = 0.2  # lambda, coupling strength
HALF_WIDTH = 0.025   # gamma, half width of underdamped SD
BATH_FREQ = 1        # omega_0, resonance frequency of underdamped SD
DELTA = 1            # delta, qubit level separation


###############################################################################


def cot(z):
    return 1 / np.tan(z)


def setup_example():
    pm_identity = qt.tensor(*[qt.qeye(c) for c in PM_CUTOFF])

    Hs = (DELTA / 2) * qt.sigmax() & pm_identity
    Q = qt.sigmaz() & pm_identity
    rho0s = .5 * qt.qeye(2)

    pseudomode_info = []  # Omega, Gamma, N, lambda

    # resonant pseudomodes
    Omega = np.sqrt(BATH_FREQ**2 - HALF_WIDTH**2)
    nu_p = HALF_WIDTH + 1j * Omega
    nu_m = HALF_WIDTH - 1j * Omega
    a_p = +COUP_STRENGTH**2 / (4 * Omega) * (1 + 1j * cot(BETA * nu_p / 2))
    a_m = -COUP_STRENGTH**2 / (4 * Omega) * (1 + 1j * cot(BETA * nu_m / 2))
    pseudomode_info.append((
        Omega, 2 * HALF_WIDTH, a_m / (a_p - np.conj(a_m)),
        np.sqrt(a_p - np.conj(a_m))))
    pseudomode_info.append((
        0, 2 * nu_p, 0, np.sqrt(a_p - np.conj(a_p))))

    # matsubara modes
    for k in range(1, NUM_MATSUBARA + 1):
        nu_k = 2 * np.pi * k / BETA
        a_k = -4 * COUP_STRENGTH**2 * HALF_WIDTH / BETA * nu_k / (
            (nu_p**2 + nu_k**2) * (nu_m**2 + nu_k**2))
        pseudomode_info.append((
            0, 2 * nu_k, 0, np.sqrt(a_k)))

    # build up total hamiltonian, initial state, lindblad ops, rates
    Htot = Hs
    rho0 = rho0s
    lindblad_ops = []
    rates = []
    # also collect heat current operators which are -L_n^\dag ( H_{i,n} )
    # where L_n is free evolution generator of that PM and \dag is
    # Hilbert-Schmidt adjoint and H_{i,n} is interaction Hamiltonian
    heat_current_ops = []

    for i, (W, G, N, lam) in enumerate(pseudomode_info):
        identities_before = [
            qt.qeye(2),
            *[qt.qeye(c) for j, c in enumerate(PM_CUTOFF) if j < i]]
        identities_after = [qt.qeye(c)
                            for j, c in enumerate(PM_CUTOFF) if j > i]
        create = qt.tensor(*identities_before, qt.create(PM_CUTOFF[i]),
                           *identities_after)
        destroy = qt.tensor(*identities_before, qt.destroy(PM_CUTOFF[i]),
                            *identities_after)

        Htot += W * create * destroy
        HI = lam * Q * (create + destroy)
        Htot += HI
        if N == 0:
            lindblad_ops.extend([destroy])
            rates.extend([G])
            rho0 = rho0 & qt.fock_dm(PM_CUTOFF[i])
        else:
            lindblad_ops.extend([destroy, create])
            rates.extend([G * (N + 1), G * N])
            rho0 = rho0 & qt.thermal_dm(PM_CUTOFF[i], N)

        heat_current_ops.append(
            -1j * W * (HI * create * destroy - create * destroy * HI) -
            G * (N + 1) * (destroy * HI * create - destroy * create * HI / 2 -
                           HI * destroy * create / 2) -
            G * N * (create * HI * destroy - create * destroy * HI / 2 -
                     HI * create * destroy / 2)
        )

    return {
        'Htot': Htot,
        'Hs': Hs,
        'Q': Q,
        'rho0': rho0,
        'lindblad_ops': lindblad_ops,
        'rates': rates,
        'Hs_system': (DELTA / 2) * qt.sigmax(),
        'Q_system': qt.sigmaz(),
        'rho0_system': rho0s,
        'heat_current_ops': heat_current_ops
    }


###############################################################################

if __name__ == "__main__":
    ex1 = setup_example()

    unravelings = [
        pdp.StandardPseudoUnraveling,
        pdp.AlternativePseudoUnraveling,
        pdp.UnravelingLikeAppendixC4,
    ]
    NTRAJ_PER_RUN = 1000

    i = 0
    while True:
        print(f"Run {i}")

        try:
            for cls in unravelings:
                process = cls(ex1['Htot'], ex1['lindblad_ops'], ex1['rates'])
                initial_state = pdp.NonHermitianIC(
                    ex1['rho0'], ntraj=NTRAJ_PER_RUN)
                solver = pdp.PDPSolver(
                    process, options={'map': 'parallel', 'num_cpus': NUM_CPUS,
                                      'max_step': 0.5,
                                      'store_states': False,
                                      'store_final_state': False,
                                      'keep_runs_results': False,
                                      'progress_bar': 'tqdm'}
                )

                result = solver.run_mixed(
                    initial_state, TLIST,
                    e_ops=[ex1['Hs'], qt.qeye_like(ex1['Hs'])]
                )

                qt.qsave(result, f"./result-{i}-{cls.__name__}")
            
            print()
            i += 1
        except Exception:
            traceback.print_exc()
            continue
