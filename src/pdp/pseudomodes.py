__all__ = ['GeneralPseudoUnraveling',
           'StandardPseudoUnraveling',
           'PseudoUnravelingAlternativeRates',
           'PseudoUnravelingWithEqualNorms',
           'PseudoUnravelingLikePairedEvolution',
           'NonHermitianIC']

from .pdprocess import PDProcess
from .multitraj_patch import InitialStateGenerator

import numpy as np
import qutip as qt
from scipy.linalg import block_diag

from abc import abstractmethod
from numpy.typing import NDArray
from typing import Any


class GeneralPseudoUnraveling(PDProcess):
    # TODO just work with Qobjs throughout?

    def __init__(self, hamiltonian: qt.Qobj,
                 lindblad_ops: list[qt.Qobj], rates: list[float]):
        if len(rates) != len(lindblad_ops):
            raise ValueError()
        self.hamiltonian = hamiltonian
        self.rates = rates
        self.lindblad_ops = lindblad_ops
        
        self._L_data = [block_diag(L.full(), L.full()) for L in lindblad_ops]
        self._zipped_data = [(g, (L.dag() * L).full())
                             for g, L in zip(rates, lindblad_ops)]
        H_eff1 = hamiltonian
        H_eff2 = hamiltonian.dag()
        for g, L in zip(rates, lindblad_ops):
            LdL = L.dag() * L
            H_eff1 += g * LdL
            H_eff2 += g.conjugate() * LdL
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
        return psi2.conj() @ observable.full() @ psi1
    
    def jump_rates(self, time: float, state: NDArray) -> list[float]:
        return [self.jump_rate(n, time, state)
                for n in range(len(self._L_data))]

    def apply_jump(self, time: float, channel: int, state: NDArray) -> None:
        factor = np.sqrt(self.rates[channel] /
                         self.jump_rate(channel, time, state))
        state[:] = factor * (self._L_data[channel] @ state)
        
        g = self.g(channel, time, state)
        if g != 1:
            half = int(len(state) / 2)
            state[:half] *= g
            state[half:] /= g

    def deterministic_generator(
        self, time: float, state: NDArray, result: NDArray) -> None:
        alpha = sum(self.jump_rates(time, state)) / 2
        result[:] = (-1j * self._H_eff @ state) + (alpha * state)

        f = self.f(time, state)
        if f != 0:
            half = int(len(state) / 2)
            result[:half] += f / 2
            result[half:] -= f / 2

    def arguments(self, args: Any) -> None:
        pass
    
    @abstractmethod
    def f(self, time: float, state: NDArray) -> complex:
        # f_1 = sum_a rate_a + f
        # f_2 = sum_a rate_a - f
        pass

    @abstractmethod
    def g(self, channel: int, time: float, state: NDArray) -> complex:
        # g_1 = sqrt( gamma / rate ) * g
        # g_2 = sqrt( gamma / rate ) / g
        pass

    @abstractmethod
    def jump_rate(self, channel: int, time: float, state: NDArray) -> float:
        pass


class StandardPseudoUnraveling(GeneralPseudoUnraveling):
    def f(self, time, state):
        return 0
    
    def g(self, channel, time, state):
        return 1
    
    def jump_rate(self, channel, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]

        g, LdL = self._zipped_data[channel]
        complex_rate = g * (psi2.conj() @ LdL @ psi1) / (psi2.conj() @ psi1)
        return np.abs(complex_rate)
    
    def jump_rates(self, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]
        norm = psi2.conj() @ psi1

        return [np.abs(g * (psi2.conj() @ LdL @ psi1) / norm)
                for g, LdL in self._zipped_data]


class PseudoUnravelingAlternativeRates(GeneralPseudoUnraveling):
    def f(self, time, state):
        return 0
    
    def g(self, channel, time, state):
        return 1
    
    def jump_rate(self, channel, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]

        # TODO
        g, LdL = self._zipped_data[channel]
        complex_rate = g * (psi2.conj() @ LdL @ psi1) / (psi2.conj() @ psi1)
        return np.abs(complex_rate)
    
    def jump_rates(self, time, state):
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]
        norm = psi2.conj() @ psi1

        # TODO
        return [np.abs(g * (psi2.conj() @ LdL @ psi1) / norm)
                for g, LdL in self._zipped_data]


class PseudoUnravelingWithEqualNorms(GeneralPseudoUnraveling):
    pass # TODO


class PseudoUnravelingLikePairedEvolution(GeneralPseudoUnraveling):
    pass # TODO


class NonHermitianIC(InitialStateGenerator):
    pass # TODO