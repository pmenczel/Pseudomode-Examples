__all__ = ['LindbladUnraveling', 'InitialDM']

from .pdprocess import PDProcess
from .multitraj_patch import InitialStateGenerator

import numpy as np
import qutip as qt

from typing import Any
from numpy.typing import NDArray


class LindbladUnraveling(PDProcess):
    def __init__(self, hamiltonian: qt.Qobj,
                 lindblad_ops: list[qt.Qobj], rates: list[float]):
        if len(rates) != len(lindblad_ops):
            raise ValueError()
        self.hamiltonian = hamiltonian
        self.lindblad_ops = lindblad_ops
        self.rates = rates
        
        self._L_data = [L.full() for L in lindblad_ops]
        self._zipped_data = [(g, (L.dag() * L).full())
                             for g, L in zip(rates, lindblad_ops)]
        self._H_eff = (hamiltonian.full() -
                       0.5j * sum(g * LdL for g, LdL in self._zipped_data))
        
        # need the dimensions of a state in the space H acts on
        # best I can come up with is this:
        self.dims = hamiltonian.eigenstates(eigvals=1)[-1][0].dims

    def initial_state_to_array(self, state: qt.Qobj) -> NDArray:
        return state.full().flatten() # note that flatten returns a copy

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
