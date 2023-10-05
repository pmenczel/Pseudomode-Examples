import qutip as qt
from pdp import LindbladUnraveling, PDPSolver

def test_constant_evolution():
    H = 0 * qt.sigmaz()
    system = LindbladUnraveling(H, [], [])
    solver = PDPSolver(system)
    assert True # TODO