__all__ = ['PDProcess', 'LindbladUnraveling', 'InitialDM']

from .multitraj_patch import InitialStateGenerator

import numpy as np
import qutip as qt
from scipy.linalg import block_diag

from abc import ABC, abstractmethod
from typing import Any
from numpy.typing import NDArray


class PDProcess(ABC):
    """
    Specification of system undergoing piecewise deterministic process with
    Poisson increments, i.e. dX = L(X) dt + sum_a[ (J_a(X) - X) dN_a ].
    """

    @abstractmethod
    def initial_state_to_array(self, state: Any) -> NDArray:
        """
        The state X must be represented as a numpy array, but may also have a
        different external representation (e.g. `QObj`).
        This method takes an initial state `state` in the external
        representation and returns the corresponding numpy array.
        """
        pass

    @abstractmethod
    def array_to_state(self, state: NDArray) -> Any:
        """
        Converts the given state to the external representation.
        """
        pass

    @abstractmethod
    def expect(self, state: NDArray, observable: Any) -> complex:
        """
        Expectation value of the given observable in the given state.
        """
        pass
    
    @abstractmethod
    def jump_rates(self, time: float, state: NDArray) -> list[float]:
        """
        Returns a list of jump rates, that is, the expectation values of the
        increments dN_a at the current time conditioned on the current state
        """
        pass

    @abstractmethod
    def apply_jump(self, time: float, channel: int, state: NDArray) -> None:
        """
        Applies J_a to the given state, where a is specified by `channel`.
        The array representing the state is updated in-place; this method
        returns nothing.
        """
        pass

    @abstractmethod
    def deterministic_generator(
        self, time: float, state: NDArray, result: NDArray) -> None:
        """
        Computes L(X), where X is the current state (`state` argument).
        The result is returned in the numpy array `result`; this method
        returns nothing.
        """
        pass

    @abstractmethod
    def arguments(self, args: Any) -> None:
        pass


class LindbladUnraveling(PDProcess):
    def __init__(self, hamiltonian: qt.Qobj,
                 lindblad_ops: list[qt.Qobj], rates: list[float]):
        if len(rates) != len(lindblad_ops):
            raise ValueError()
        self.hamiltonian = hamiltonian
        self.rates = rates
        self.lindblad_ops = lindblad_ops
        
        self._L_data = [L.full() for L in lindblad_ops]
        self._zipped_data = [(g, (L.dag() * L).full())
                             for g, L in zip(rates, lindblad_ops)]
        self._H_eff = (hamiltonian.full() -
                       0.5j * sum(g * LdL for g, LdL in self._zipped_data))
        
        # need the dimensions of a state in the space H acts on
        # best I can come up with is this:
        self.dims = hamiltonian.eigenstates(eigvals=1)[-1][0].dims

    def initial_state_to_array(self, state: qt.Qobj) -> NDArray:
        return np.copy(state.full())

    def array_to_state(self, state: NDArray) -> qt.Qobj:
        return qt.Qobj(state, dims=self.dims)

    def expect(self, state: NDArray, observable: qt.Qobj) -> complex:
        qt_state = self.array_to_state(state)
        return qt.expect(observable, qt_state)
    
    def jump_rates(self, time: float, state: NDArray) -> list[float]:
        result = [g * np.vdot(state, LdL @ state)
                  for g, LdL in self._zipped_data]
        return np.real(result)

    def apply_jump(self, time: float, channel: int, state: NDArray) -> None:
        L = self._L_data[channel]
        _, LdL = self._zipped_data[channel]
        state[:] = (L @ state) / np.sqrt(np.vdot(state, LdL @ state))

    def deterministic_generator(
        self, time: float, state: NDArray, result: NDArray) -> None:
        alpha = sum(self.jump_rates(time, state)) / 2
        result[:] = (-1j * self._H_eff @ state) + (alpha * state)

    def arguments(self, args: Any) -> None:
        pass


class InitialDM(InitialStateGenerator):
    def __init__(self, state: qt.Qobj, ntraj: int):
        eigenvalues, eigenstates = state.eigenstates()
        weights = [1] * len(eigenvalues)
        super().__init__(
            zip(eigenstates, eigenvalues, weights),
            ntraj
        )


class GeneralPseudoUnraveling(PDProcess):
    # TODO use sparse representations instead of `full`?

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
