from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp
import qutip as qt

from typing import Any, Optional
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
    def _argument(self, args: Any) -> None:
        pass


class PDTrajectoryResult(qt.Result):
    pass #TODO


class PDPIntegrator:
    def __init__(self, system: PDProcess, options: dict):
        self.system = system
        self.options = options

        self._is_set = False
        self._current_time: Optional[float] = None
        self._current_state: Optional[NDArray] = None
        self._collapses: Optional[list[tuple[float, int]]] = None
        self._generator: Optional[np.random.Generator] = None

    def set_state(self, time: float, state: NDArray,
                  generator: np.random.Generator) -> None:
        self._current_time = time
        self._current_state = np.append(state, 0)
        self._collapses = []
        self._generator = generator
        self._is_set = True
    
    def get_state(self, copy: bool = True) -> NDArray:
        result = self._current_state[:-1]
        if copy:
            return np.array(result)
        else:
            return result
    
    def _rhs(self, time: float, state: NDArray) -> NDArray:
        result = np.zeros_like(state)
        self.system.deterministic_generator(time, state[:-1], result[:-1])
        result[-1] = sum(self.system.jump_rates(time, state[:-1]))
        return result
    
    def _integration_step(
            self, max_time: float) -> tuple[Optional[int], float, NDArray]:
        minus_log_lambda = -np.log(self._generator.random())
        def event(t, state):
            return np.real(state[-1]) - minus_log_lambda
        event.terminal = True
        event.direction = 1

        integration_result = solve_ivp(
            self._rhs, (self._current_time, max_time), self._current_state,
            method=self.options['method'], t_eval=[max_time], events=[event])
        
        if integration_result.status == 0: # no jump
            final_time = integration_result.t[-1]
            final_state = np.transpose(integration_result.y)[0, :-1]
            return None, final_time, final_state
        else:
            jump_time = integration_result.t_events[0][0]
            jump_state = integration_result.y_events[0][0][:-1]

            weights = self.system.jump_rates(jump_time, jump_state)
            probs = weights / sum(weights)
            jump_channel = self._generator.choice(len(weights), p=probs)

            final_state = self.system.apply_jump(
                jump_time, jump_channel, jump_state)
            return jump_channel, jump_time, final_state

    def integrate(
            self, time: float, copy: bool = False) -> tuple[float, NDArray]:
        # TODO: Copy parameter not supported
        while True:
            jump_channel, final_time, final_state =\
                self._integration_step(time)
            self._current_time = final_time
            self._current_state = final_state
            if jump_channel is None:
                return final_time, final_state
            else:
                self._collapses.append((final_time, jump_channel))

    @property
    def integrator_options(self):
        return {} # TODO

class PDPSolver(qt.MultiTrajSolver):
    """
    Monte-Carlo simulation for piecewise deterministic process specified as
    a `PDProcess` object.
    """
    name = "PDP Solver"
    trajectory_resultclass = PDTrajectoryResult

    # The `method` option will be passed on to scipy's `solve_ivp`.
    # We do not support alternative integrator classes specified by `method`.
    solver_options = {
        **qt.MultiTrajSolver.solver_options,
        'method': 'LSODA',
    }

    def __init__(self, system: PDProcess, *, options: Optional[dict] = None):
        self.system = system
        super().__init__(rhs=None, options=options)

    def _get_integrator(self):
        raise NotImplementedError #TODO
        
    def _initialize_stats(self):
        stats = super()._initialize_stats()
        stats.update({
            "method": self.options["method"],
            "solver": "Piecewise Deterministic Process Solver",
            "system": str(self.system),
        })
        return stats
    
    def _prepare_state(self, state: Any) -> NDArray:
        return self.system.initial_state_to_array(state)
    
    def _restore_state(self, data: NDArray, *, copy: bool = False) -> Any:
        return self.system.array_to_state(data)
    
    def _run_one_traj(self, seed: np.random.SeedSequence,
                      state: NDArray, tlist: list[float],
                      e_ops: list[Any]) -> tuple[np.random.SeedSequence,
                                                 qt.Result]:
        seed, result = super()._run_one_traj(seed, state, tlist, e_ops)
        result.collapse = self._integrator.collapses
        return seed, result

    def _argument(self, args: Any):
        if args is not None:
            self.system._argument(args)

    def _get_integrator(self):
        return PDPIntegrator(self.system, self.options)
    
    def _apply_options(self, _):
        # _apply_options in the base Solver re-initializes integrator if
        # `method` is changed. We don't need that here since there is only
        # one integrator class
        self._integrator.options = self.options
