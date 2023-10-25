__all__ = ['setup_example']

from pdp import PDPSolver

import numpy as np
import qutip as qt


##### SYSTEM CONFIGURATION ####################################################

try:
    import mkl  # type: ignore
    mkl.set_num_threads(1)
except ImportError:
    pass

NUM_CPUS = 5


##### NUMERICS ################################################################

NUM_MATSUBARA = 1
PM_CUTOFF = 3
TLIST = np.linspace(0, 100, 500)


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
    pm_identity = qt.tensor(*([qt.qeye(PM_CUTOFF)] * (NUM_MATSUBARA + 2)))

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
    for k in range(NUM_MATSUBARA):
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

    for i, (W, G, N, lam) in enumerate(pseudomode_info):
        identity_before = qt.qeye(2) & qt.tensor(*([qt.qeye(PM_CUTOFF)] * i))
        identity_after = qt.tensor(*([qt.qeye(PM_CUTOFF)] *
                                     (NUM_MATSUBARA + 2 - i)))
        create = identity_before & qt.create(PM_CUTOFF) & identity_after
        destroy = identity_before & qt.destroy(PM_CUTOFF) & identity_after

        Htot += W * create * destroy
        Htot += lam * Q * (create + destroy)
        if N == 0:
            lindblad_ops.extend([destroy])
            rates.extend([G])
            rho0 = rho0 & qt.fock_dm(PM_CUTOFF)
        else:
            lindblad_ops.extend([destroy, create])
            rates.extend([G * (N + 1), G * N])
            rho0 = rho0 & qt.thermal_dm(PM_CUTOFF, N)

    return Htot, rho0, lindblad_ops, rates


###############################################################################

if __name__ == "__main__":
    setup_example()
