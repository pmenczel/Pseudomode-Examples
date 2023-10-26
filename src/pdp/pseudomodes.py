__all__ = ['StandardPseudoUnraveling',
           'AlternativePseudoUnraveling',
           'StandardPseudoUnravelingEN',
           'AlternativePseudoUnravelingEN',
           'UnravelingLikeAppendixC4',
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
        state_vec = state.full().flatten()
        return np.block([state_vec, state_vec])

    def array_to_state(self, time: float, state: NDArray) -> qt.Qobj:
        half = int(len(state) / 2)
        psi1 = qt.Qobj(state[:half], dims=self.dims)
        psi2 = qt.Qobj(state[half:], dims=self.dims)
        return psi1 * psi2.dag()

    def expect(self, time: float, state: NDArray, obs: qt.Qobj) -> complex:
        half = int(len(state) / 2)
        psi1 = state[:half]
        psi2 = state[half:]
        return np.vdot(psi2, obs.full() @ psi1)

    def jump_rates(self, time: float, state: NDArray) -> list[float]:
        return [self.jump_rate(n, time, state)
                for n in range(len(self._L_data))]

    def apply_jump(self, time: float, channel: int, state: NDArray) -> None:
        half = int(len(state) / 2)
        factor = np.emath.sqrt(self.rates[channel] /
                               self.jump_rate(channel, time, state))

        state[:] = self._L_data[channel] @ state
        state[:half] *= factor
        state[half:] *= np.conj(factor)

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
                         (np.vdot(psi1, psi1) + np.vdot(psi2, psi2)))
        d_psi1 += f_half * psi1
        d_psi2 -= f_half * psi2


class StandardPseudoUnravelingEN(StandardPseudoUnraveling,
                                 _EqualNormUnraveling):
    pass


class AlternativePseudoUnravelingEN(AlternativePseudoUnraveling,
                                    _EqualNormUnraveling):
    pass


class UnravelingLikeAppendixC4(PDProcess):
    def __init__(self, hamiltonian: qt.Qobj,
                 lindblad_ops: list[qt.Qobj], rates: list[float]):
        self.hamiltonian = hamiltonian
        self.lindblad_ops = lindblad_ops
        self.rates = rates

        HR = (hamiltonian + hamiltonian.dag()) / 2
        HI = (hamiltonian - hamiltonian.dag()) / 2j
        L_hat = [(qt.qeye(2) & L) for L in lindblad_ops]
        _term = sum(np.imag(gamma) * L.dag() * L
                    for gamma, L in zip(rates, lindblad_ops)) / 2
        H_hat = (
            (qt.qeye(2) & HR) +
            (qt.sigmaz() & (qt.qzero_like(HR) if _term == 0 else _term))
        )
        X = (2 * (qt.sigmaz() & HI) +
             sum((np.abs(gamma) - np.real(gamma)) * L.dag() * L
                 for gamma, L in zip(rates, L_hat)))

        self._Lambda = max(X.eigenenergies())
        L0 = (self._Lambda * qt.qeye(X.dims[0]) - X).sqrtm()

        self._gamma = rates + [0]
        self._Gamma = [np.abs(rate) for rate in rates] + [1]
        self._L_hat = [L.full() for L in L_hat] + [L0.full()]
        self._zipped = ([(np.abs(rate), (L.dag() * L).full())
                         for rate, L in zip(rates, L_hat)] +
                        [(1, (L0.dag() * L0).full())])
        self._H_eff = (H_hat.full() - .5j * sum(
            Gamma * LdL for Gamma, LdL in self._zipped))

        self.dims = hamiltonian.eigenstates(eigvals=1)[-1][0].dims

    def initial_state_to_array(self, state: qt.Qobj) -> NDArray:
        # Like in standard unraveling, but add martingale
        # We normalize so that norm of double vector is one
        state_vec = state.full().flatten() / np.sqrt(2)
        return np.block([state_vec, state_vec, 2])

    def array_to_state(self, time: float, state: NDArray) -> qt.Qobj:
        half = int((len(state) - 1) / 2)
        psi1 = qt.Qobj(state[:half], dims=self.dims)
        psi2 = qt.Qobj(state[half:-1], dims=self.dims)
        mu = state[-1] * np.exp(self._Lambda * time)
        return mu * psi1 * psi2.dag()

    def expect(self, time: float, state: NDArray, obs: qt.Qobj) -> complex:
        half = int((len(state) - 1) / 2)
        psi1 = qt.Qobj(state[:half], dims=self.dims)
        psi2 = qt.Qobj(state[half:-1], dims=self.dims)
        mu = state[-1] * np.exp(self._Lambda * time)
        return mu * np.vdot(psi2, obs.full() @ psi1)

    def jump_rates(self, time: float, state: NDArray) -> list[float]:
        if state[-1] == 0: # once mu is zero, remaining traj doesn't matter
            return [0] * len(self._zipped)

        return [self.jump_rate(channel, time, state)
                for channel in range(len(self._zipped))]

    def jump_rate(self, channel: int, time: float, state: NDArray) -> float:
        if state[-1] == 0: # once mu is zero, remaining traj doesn't matter
            return 0

        Gamma, LdL = self._zipped[channel]
        return np.real(Gamma * np.vdot(state[:-1], LdL @ state[:-1]))

    def apply_jump(self, time: float, channel: int, state: NDArray) -> None:
        if state[-1] == 0: # once mu is zero, remaining traj doesn't matter
            return

        factor = np.sqrt(self._Gamma[channel] /
                         self.jump_rate(channel, time, state))
        state[:-1] = factor * self._L_hat[channel] @ state[:-1]
        state[-1] *= self._gamma[channel] / self._Gamma[channel]

    def deterministic_generator(
            self, time: float, state: NDArray, result: NDArray) -> None:
        if state[-1] == 0: # once mu is zero, remaining traj doesn't matter
            result[:] = np.zeros_like(result)

        alpha = sum(self.jump_rates(time, state)) / 2
        result[:-1] = (-1j * self._H_eff @ state[:-1]) + (alpha * state[:-1])
        result[-1] = 0
        # result[-1] = self._Lambda * state[-1]
        # We keep mu constant during the deterministic evolution and instead
        # multiply with exp(Lambda * t) in the end

    def arguments(self, args: Any) -> None:
        pass


class NonHermitianIC(InitialStateGenerator):
    def __init__(self, state: qt.Qobj, ntraj: int):
        # Assumes state is diagonalizable
        eigenvalues, eigenstates = state.eigenstates()

        frequencies = np.array([np.abs(lamb) for lamb in eigenvalues])
        frequencies /= sum(frequencies)
        weights = [0 if freq == 0 else lamb / freq
                   for lamb, freq in zip(eigenvalues, frequencies)]

        super().__init__(
            zip(eigenstates, frequencies, weights),
            ntraj
        )
