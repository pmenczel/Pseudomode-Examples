from collections import Counter
import pytest
from pdp import InitialStateGenerator

import numpy as np
import qutip as qt


EPSI = 1e-8


@pytest.mark.parametrize('state', ['A', qt.basis(2, 0)])
@pytest.mark.parametrize('weight', [1, 0.2, 2 + .5j])
@pytest.mark.parametrize('ntraj', [1, 10])
def test_single_state(state, weight, ntraj):
    gen = InitialStateGenerator([(state, 1, weight)], ntraj)

    assert gen.nstates() == 1
    assert gen.state(0) == state
    assert gen.trajectory_count(0) == ntraj
    assert gen.weight(0) == weight
    assert gen.state_numbers() == ([0] * ntraj)


@pytest.mark.parametrize('weight', [1, 0.2, 2 + .5j])
@pytest.mark.parametrize('ntraj', [1, 10])
def test_removes_zero_frequency(weight, ntraj):
    gen = InitialStateGenerator([
        ('A', 1, weight),
        ('B', 0, 1)
    ], ntraj)

    assert gen.nstates() == 1
    assert gen.state(0) == 'A'
    assert gen.trajectory_count(0) == ntraj
    assert gen.weight(0) == weight
    assert gen.state_numbers() == ([0] * ntraj)


@pytest.mark.parametrize('ics', [
    [('A', 1, 1)],
    [('A', 0.1, 1), ('B', 0.1, 3), ('C', 0.1, 0), ('D', 0, 1), ('E', 0.7, 1)],
    [('A', 0.5, 1j), ('B', 0.5, 2 - 1j)],
    [('A', 0.02, 1), ('B', 0.03, 1), ('C', 0.05, 1), ('D', 0.06, 1),
     ('E', 0.07, 1), ('F', 0.08, 1), ('G', 0.09, 1), ('H', 0.1, 1),
     ('I', 0.11, 1), ('J', 0.12, 1), ('K', 0.13, 1), ('L', 0.14, 1)]
])
@pytest.mark.parametrize('ntraj', [0, 1, 10, 420])
def test_constraints_satisfied(ics, ntraj):
    required_states = len(
        [freq for _, freq, _ in ics if freq > 0]
    )
    if required_states > ntraj:
        with pytest.raises(ValueError):
            InitialStateGenerator(ics, ntraj)
        ntraj = required_states
    
    gen = InitialStateGenerator(ics, ntraj)

    nstates = gen.nstates()
    total_traj = sum(gen.trajectory_count(n) for n in range(nstates))
    assert total_traj == ntraj
    assert len(gen.state_numbers()) == ntraj

    for n in range(nstates):
        state = gen.state(n)
        freq, weight = next((freq, weight)
                            for x, freq, weight in ics if state == x)
        assert np.abs(
            freq * weight -
            gen.weight(n) * gen.trajectory_count(n) / ntraj) < EPSI
        

def test_exactly_solvable():
    gen = InitialStateGenerator([('A', 0.25, 1), ('B', 0.75, 1j)], 4)
    assert gen.nstates() == 2
    for n in range(2):
        if gen.state(n) == 'A':
            na = n
            assert gen.trajectory_count(n) == 1
            assert gen.weight(n) == 1
        else:
            nb = n
            assert gen.trajectory_count(n) == 3
            assert gen.weight(n) == 1j
    assert Counter(gen.state_numbers()) == Counter([na, nb, nb, nb])
            
    gen = InitialStateGenerator([('A', 0.25, 1), ('B', 0.75, 1j)], 100)
    assert gen.nstates() == 2
    for n in range(2):
        if gen.state(n) == 'A':
            na = n
            assert gen.trajectory_count(n) == 25
            assert gen.weight(n) == 1
        else:
            nb = n
            assert gen.trajectory_count(n) == 75
            assert gen.weight(n) == 1j
    assert Counter(gen.state_numbers()) == Counter([na] * 25 + [nb] * 75)
