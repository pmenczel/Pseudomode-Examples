__all__ = ['PDTrajectoryResult', 'PDPIntegrator', 'PDPSolver']

from .processes import PDProcess
from .multitraj_patch import EnhancedMultiTrajSolver

import numpy as np
from scipy.integrate import solve_ivp
import qutip as qt
from qutip.solver.integrator import Integrator

from typing import Any, Iterable, Optional
from numpy.typing import NDArray


class PDTrajectoryResult(qt.Result):
    pass # TODO: store collapses (modify tests to make sure they are stored)
    # Also:
    # Possibly calculate expectation values here based on system.expect instead
    # of using system.array_to_state.
    # However multitrajsolver applies restore_state before passing something
    # into the result object, so array_to_state is already applied.
    # In order to change that, we'd have to make things more complicated,
    # which might not be worth it.
    # (The gain would be not having to do array_to_state if store_states is set
    # to false, which might be more expensive than direct `expect` calls)


class PDPIntegrator(Integrator):
    integrator_options = {
        'scipy_method': 'RK45',
        'first_step': None,
        'max_step': np.inf,
        'min_step': 0,
        'rtol': 1e-3,
        'atol': 1e-6,
    }

    support_time_dependant = True  # spelling like in parent class :(
    supports_blackbox = False
    name = "PDP Integrator"

    def __init__(self, system: PDProcess, options: dict):
        self._current_time: Optional[float] = None
        self._current_state: Optional[NDArray] = None
        self._collapses: Optional[list[tuple[float, int]]] = None
        self._generator: Optional[np.random.Generator] = None
        
        super().__init__(system, options)

    def _prepare(self):
        self._collapses = []

    # Made generator optional to agree with parent class contract
    def set_state(self, time: float, state: NDArray,
                  generator: Optional[np.random.Generator] = None) -> None:
        self._current_time = time
        self._current_state = np.append(state, 0)

        if generator is not None:
            self._generator = generator
        elif self._generator is None:
            self._generator = np.random.default_rng()

        self._is_set = True
    
    def get_state(self, copy: bool = False) -> tuple[float, NDArray]:
        result = self._current_state[:-1]
        if copy:
            return self._current_time, np.copy(result)
        else:
            return self._current_time, result
    
    def _rhs(self, time: float, state: NDArray) -> NDArray:
        result = np.zeros_like(state)
        self.system.deterministic_generator(time, state[:-1], result[:-1])
        result[-1] = sum(self.system.jump_rates(time, state[:-1]))
        return result
    
    def _integration_step(
            self, times: list[float]
            ) -> tuple[Optional[int], float, NDArray, list[NDArray]]:
        minus_log_lambda = -np.log(self._generator.random())
        def event(t, state):
            return np.real(state[-1]) - minus_log_lambda
        event.terminal = True
        event.direction = 1

        integration_result = solve_ivp(
            self._rhs, (self._current_time, times[-1]), self._current_state,
            method=self.options['scipy_method'], t_eval=times,
            events=[event], first_step=self.options['first_step'],
            max_step=self.options['max_step'],
            min_step=self.options['min_step'],
            rtol=self.options['rtol'], atol=self.options['atol'])
        
        if len(integration_result.y) == 0:
            states = []
        else:
            states = np.transpose(integration_result.y)
        
        if integration_result.status == 0: # no jump
            final_time = integration_result.t[-1]
            final_state = states[-1]
            return None, final_time, final_state, states
        else:
            jump_time = integration_result.t_events[0][0]
            jump_state = integration_result.y_events[0][0]

            weights = self.system.jump_rates(jump_time, jump_state)
            probs = weights / sum(weights)
            jump_channel = self._generator.choice(len(weights), p=probs)

            self.system.apply_jump(jump_time, jump_channel, jump_state[:-1])
            return jump_channel, jump_time, jump_state, states

    def integrate(
            self, time: float, copy: bool = False) -> tuple[float, NDArray]:
        while True:
            jump_channel, final_time, final_state, _ =\
                self._integration_step([time])
            self._current_time = final_time
            final_state[-1] = 0
            self._current_state = final_state
            if jump_channel is None:
                break
            else:
                self._collapses.append((final_time, jump_channel))
        return self.get_state(copy=copy)
    
    def run(self, tlist: list[float]) -> Iterable[tuple[float, NDArray]]:
        tlist = tlist[1:]
        while True:
            jump_channel, final_time, final_state, states =\
                self._integration_step(tlist)
            self._current_time = final_time
            final_state[-1] = 0
            self._current_state = final_state
            
            num_states = len(states)
            yield from zip(tlist[:num_states], states[:, :-1])
            tlist = tlist[num_states:]
            
            if jump_channel is None:
                break
            else:
                self._collapses.append((final_time, jump_channel))
            

    @property
    def options(self):
        """PDP Integrator options are `scipy_method`, `first_step`, `min_step`,
        `max_step`, `rtol` and `atol` as documented in
        `scipy.integrate.solve_ivp`."""
        return super().options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)

class PDPSolver(EnhancedMultiTrajSolver):
    """
    Monte-Carlo simulation for piecewise deterministic process specified as
    a `PDProcess` object.
    """
    name = "PDP Solver"
    trajectory_resultclass = PDTrajectoryResult
    _avail_integrators = {}

    solver_options = {
        **EnhancedMultiTrajSolver.solver_options,
        'method': 'PDP',
    }

    def __init__(self, system: PDProcess, *, options: Optional[dict] = None):
        self.system = system
        super().__init__(rhs=system, options=options)
        
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
                      e_ops: Any) -> tuple[np.random.SeedSequence, qt.Result]:
        seed, result = super()._run_one_traj(seed, state, tlist, e_ops)
        result.collapse = self._integrator._collapses
        return seed, result

    def _argument(self, args: Any):
        if args is not None:
            self.system.arguments(args)
    
    @classmethod
    def avail_integrators(cls):
        return cls._avail_integrators

PDPSolver.add_integrator(PDPIntegrator, 'PDP')
