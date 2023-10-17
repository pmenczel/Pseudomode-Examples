__all__ = ['InitialStateGenerator',
           'EnhancedMultiTrajResult',
           'EnhancedMultiTrajSolver']

from qutip.solver.multitraj import MultiTrajSolver, _get_map

import numpy as np
import qutip as qt

from time import time
from operator import itemgetter
from collections import Counter
import bisect

from typing import Any


class InitialStateGenerator:
    def __init__(self, initial_conditions: list[tuple[Any, float, complex]],
                 ntraj: int):
        """
        initial_conditions: a list of possible initial conditions
            Each entry in the list contains three parts:
            * state: the state to be passed to the solver's `run` method
            * frequency: the desired frequency (between 0 and 1) of this
                initial state among the trajectories
            * weight: an additional weight to be added to trajectories
                starting from this initial state
            Note that sum(f for f in frequencies) must be one.

        ntraj: number of trajectories
            There must be at least one trajectory for each state with non-zero
            frequency.

        An InitialStateGenerator object represents a list of states together
        with the number of trajectories starting from the respective states,
        `trajectory_count(n)`, and corrected weights, `corrected_weight(n)`.
        It is guaranteed that the total number of trajectories is `ntraj`:
            sum_n trajectory_count(n) = ntraj.
        It is further guaranteed that for each n,
            weight * frequency = corrected_weight * (trajectory_count / ntraj).
        We try to generate a distribution of trajectory counts that
        approximates the provided frequencies as well as possible under these
        constraints,
            trajectory_count ~ frequency * ntraj.
        """
        self.ntraj = ntraj
        self._states: list[tuple[Any, float, complex, int]] = []
        
        # remove zero-frequency entries
        # also, note down originally requested frequency for each entry
        # and calculate the "target" freq * ntraj
        filtered_ics = [(state, freq, weight, freq * ntraj)
                        for state, freq, weight in initial_conditions
                        if freq > 0]
        if len(filtered_ics) > ntraj:
            raise ValueError("Not enough trajectories "
                             "for mixed initial conditions")

        # The following algorithm is loosely based on
        # https://stackoverflow.com/a/792490
        # We initially round up because each state needs at least
        # one trajectory. We then remove trajectories until the correct
        # total number of trajectories is reached.
        # We remove the trajectory from the state with maximum result / target,
        # but we never remove the only remaining trajectory
        self._states = []
        under_consideration = []
        total_number = 0
        for state, freq, weight, target in filtered_ics:
            result = int(np.ceil(target))
            total_number += result
            # if only one trajectory, can be added to self._states and not
            # considered further
            if result == 1:
                self._states.append((state, freq, weight, result))
                continue

            ratio = result / target
            # under_consideration is kept sorted according to the ratio
            bisect.insort(under_consideration,
                          (state, freq, weight, result, ratio),
                          key=itemgetter(4))
        
        while total_number > ntraj:
            state, freq, weight, result, _ = under_consideration.pop()
            result -= 1
            total_number -= 1

            if result == 1:
                self._states.append((state, freq, weight, result))
                continue

            ratio = result / target
            bisect.insort(under_consideration,
                          (state, freq, weight, result, ratio),
                          key=itemgetter(4))
        
        # Finally we have achieved total_number = ntraj, add all remaining
        # states to self._states
        for state, freq, weight, result, _ in under_consideration:
            self._states.append((state, freq, weight, result))
    
    def nstates(self):
        return len(self._states)
    
    def state(self, n: int) -> Any:
        return self._states[n][0]
    
    def trajectory_count(self, n: int) -> int:
        return self._states[n][3]
    
    def weight(self, n: int) -> complex:
        _, orig_freq, extra_weight, traj_count = self._states[n]
        return extra_weight * orig_freq * self.ntraj / traj_count
    
    def state_numbers(self) -> list[int]:
        counts = Counter({n: self.trajectory_count(n)
                          for n in range(self.nstates())})
        return list(counts.elements())


class EnhancedMultiTrajResult(qt.MultiTrajResult):
    def _weighted_dm(self, state, weight):
        if state is None:
            return state
        return qt.ket2dm(state) * weight

    def add(self, trajectory_info: tuple[np.random.SeedSequence, qt.Result]):
        _, trajectory = trajectory_info
        if not hasattr(trajectory, 'weight'):
            return super().add(trajectory_info)
        weight = trajectory.weight
        
        old_states = trajectory.states
        trajectory.states = [self._weighted_dm(state, weight)
                             for state in old_states]
        old_final_state = trajectory.final_state
        trajectory.final_state = self._weighted_dm(old_final_state, weight)
        old_edata = trajectory.e_data
        trajectory.e_data = {key: np.asarray(data) * weight
                             for key, data in old_edata.items()}
        
        result = super().add(trajectory_info)

        trajectory.states = old_states
        trajectory.final_state = old_final_state
        trajectory.e_data = old_edata
        return result


class EnhancedMultiTrajSolver(MultiTrajSolver):
    resultclass = EnhancedMultiTrajResult

    # Make use of Integrator's `run` method which might be more efficient
    # than repeated calls to its `integrate` method.
    def _integrate_one_traj(self, seed: np.random.SeedSequence,
                            tlist: list[float], result: qt.Result
                            ) -> tuple[np.random.SeedSequence, qt.Result]:
        # Note that integrator.run discards first value of tlist
        for t, state in self._integrator.run(tlist):
            result.add(t, self._restore_state(state, copy=False))
        return seed, result

    # Support for mixed initial state
    # Argument types mostly too complicated for type hints
    # We do not support target tolerance
    def run_mixed(self, state_generator: InitialStateGenerator,
                  tlist: list[float], *,
                  args=None, e_ops=(), timeout=None, seed=None):
        start_time = time()
        self._argument(args)
        stats = self._initialize_stats()
        seeds = self._read_seed(seed, state_generator.ntraj)

        result = self.resultclass(
            e_ops, self.options, solver=self.name, stats=stats
        )

        map_func = _get_map[self.options['map']]
        map_kw = {
            'timeout': timeout,
            'job_timeout': self.options['job_timeout'],
            'num_cpus': self.options['num_cpus'],
        }
        trajectory_infos = list(zip(seeds, state_generator.state_numbers()))
        stats['preparation time'] += time() - start_time
        
        start_time = time()
        map_func(self._run_one_traj_mixed, trajectory_infos,
                 (state_generator, tlist, e_ops),
                 reduce_func=result.add, map_kw=map_kw,
                 progress_bar=self.options["progress_bar"],
                 progress_bar_kwargs=self.options["progress_kwargs"])
        result.stats['run time'] = time() - start_time
        return result

    def _run_one_traj_mixed(
            self,
            trajectory_info: tuple[np.random.SeedSequence, int],
            state_generator: InitialStateGenerator,
            tlist: list[float], e_ops: Any) -> qt.Result:
        seed, state_number = trajectory_info
        state = self._prepare_state(
            state_generator.state(state_number))
        weight = state_generator.weight(state_number)
        
        seed, result = self._run_one_traj(seed, state, tlist, e_ops)
        result.weight = weight
        return seed, result