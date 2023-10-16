__all__ = ['PseudoUnraveling',
           'StandardPseudoUnraveling',
           'AlternativePseudoUnraveling',
           'StandardPseudoUnravelingEN',
           'AlternativePseudoUnravelingEN',
           'NonHermitianIC']

from .pdprocess import PDProcess
from .multitraj_patch import InitialStateGenerator

import numpy as np
import qutip as qt
from scipy.linalg import block_diag

from abc import abstractmethod
from numpy.typing import NDArray
from typing import Any


class PseudoUnraveling(PDProcess):
    # TODO just work with Qobjs throughout?

    def __init__(self, hamiltonian: qt.Qobj,
                 lindblad_ops: list[qt.Qobj], rates: list[float]):
        if len(rates) != len(lindblad_ops):
            raise ValueError()
        self.hamiltonian = hamiltonian
        self.lindblad_ops = lindblad_ops
        self.rates = rates
        
        self._L_data = [block_diag(L.full(), L.full()) for L in lindblad_ops]
        self._zipped_data = [(g, (L.dag() * L).full())
                             for g, L in zip(rates, lindblad_ops)]
        H_eff1 = hamiltonian
        H_eff2 = hamiltonian.dag()
        for g, L in zip(rates, lindblad_ops):
            LdL = L.dag() * L
            H_eff1 -= .5j * g * LdL
            H_eff2 -= .5j * g.conjugate() * LdL
        self._H_eff = block_diag(H_eff1.full(), H_eff2.full())

        self.dims = hamiltonian.eigenstates(eigvals=1)[-1][0].dims

    def initial_state_to_array(self, state: qt.Qobj) -> NDArray:
        # The initial state is given as a vector
        # Output states will be given as density matrices
        state_vec = state.full()
        return np.block([state_vec, state_vec])

    def array_to_state(self, state: NDArray) -> qt.Qobj:
        half = int(len(state) / 2)
        psi1 = qt.Qobj(state[:half], dims=self.dims)
        psi2 = qt.Qobj(state[half:], dims=self.dims)
        return psi1 * psi2.dag()

    def expect(self, state: NDArray, observable: qt.Qobj) -> complex:
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]
        return np.vdot(psi2, observable.full() @ psi1)
    
    def jump_rates(self, time: float, state: NDArray) -> list[float]:
        return [self.jump_rate(n, time, state)
                for n in range(len(self._L_data))]

    def apply_jump(self, time: float, channel: int, state: NDArray) -> None:
        factor = np.sqrt(self.rates[channel] /
                         self.jump_rate(channel, time, state))
        state[:] = factor * (self._L_data[channel] @ state)

    def deterministic_generator(
        self, time: float, state: NDArray, result: NDArray) -> None:
        alpha = sum(self.jump_rates(time, state)) / 2
        result[:] = (-1j * self._H_eff @ state) + (alpha * state)

    def arguments(self, args: Any) -> None:
        pass

    @abstractmethod
    def jump_rate(self, channel: int, time: float, state: NDArray) -> float:
        pass


class StandardPseudoUnraveling(PseudoUnraveling):
    def jump_rate(self, channel, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]

        g, LdL = self._zipped_data[channel]
        complex_rate = g * np.vdot(psi2, LdL @ psi1) / np.vdot(psi2, psi1)
        return np.abs(complex_rate)
    
    def jump_rates(self, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]
        norm = np.vdot(psi2, psi1)

        return [np.abs(g * np.vdot(psi2, LdL @ psi1) / norm)
                for g, LdL in self._zipped_data]


class AlternativePseudoUnraveling(PseudoUnraveling):
    def jump_rate(self, channel, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]

        g, LdL = self._zipped_data[channel]
        return np.abs(g * np.sqrt(
            np.vdot(psi1, LdL @ psi1) * np.vdot(psi2, LdL @ psi2) /
            np.vdot(psi1, psi1) / np.vdot(psi2, psi2)))
    
    def jump_rates(self, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]
        norm = np.vdot(psi1, psi1) * np.vdot(psi2, psi2)

        return [np.abs(g * np.sqrt(np.vdot(psi1, LdL @ psi1) *
                                   np.vdot(psi2, LdL @ psi2) / norm))
                for g, LdL in self._zipped_data]


class _EqualNormUnraveling(PseudoUnraveling):
    def apply_jump(self, time: float, channel: int, state: NDArray) -> None:
        super().apply_jump(time, channel, state)
        
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]
        norm1 = np.sqrt(np.vdot(psi1, psi1))
        norm2 = np.sqrt(np.vdot(psi2, psi2))

        psi1 *= np.sqrt(norm2 / norm1)
        psi2 *= np.sqrt(norm1 / norm2)

    def deterministic_generator(
        self, time: float, state: NDArray, result: NDArray) -> None:
        super().deterministic_generator(time, state, result)
        
        half = int(len(state) / 2)
        psi1 = state[:half]
        d_psi1 = result[:half]
        psi2 = state[half:]
        d_psi2 = result[half:]

        f_half = np.real((np.vdot(psi2, d_psi2) - np.vdot(psi1, d_psi1)) /
                         (np.vdot(psi1, psi1) + np.vodt(psi2, psi2)))
        d_psi1 += f_half * psi1
        d_psi2 -= f_half * psi2


class StandardPseudoUnravelingEN(StandardPseudoUnraveling,
                                 _EqualNormUnraveling):
    pass


class AlternativePseudoUnravelingEN(AlternativePseudoUnraveling,
                                    _EqualNormUnraveling):
    pass


class NonHermitianIC(InitialStateGenerator):
    def __init__(self, state: qt.Qobj, ntraj: int):
        # Assumes state is diagonalizable
        eigenvalues, eigenstates = state.eigenstates()

        frequencies = np.array([np.abs(lamb) for lamb in eigenvalues])
        frequencies /= sum(frequencies)
        weights = [lamb / freq for lamb, freq in zip(eigenvalues, frequencies)]

        super().__init__(
            zip(eigenstates, frequencies, weights),
            ntraj
        )
