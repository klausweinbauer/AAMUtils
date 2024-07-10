import numpy as np
import rdkit.Chem.rdmolfiles as rdmolfiles
import pulp as lp

from aamutils.utils import mol_to_graph, print_graph
from aamutils.algorithm.ilp import expand_partial_aam_balanced


def test_e2e():
    mol_G = rdmolfiles.MolFromSmiles("C=[O:1].[O:2]")
    mol_H = rdmolfiles.MolFromSmiles("C([O:2])[O:1]")
    exp_X = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    np.testing.assert_array_equal(X, exp_X)
    assert 4 == v
    assert "Optimal" == status


def test_example1():
    mol_G = rdmolfiles.MolFromSmiles("CCC[Cl:1].[N:2]")
    mol_H = rdmolfiles.MolFromSmiles("CCC[N:2].[Cl:1]")
    exp_X = np.zeros((5, 5))
    exp_X[0, 0] = 1
    exp_X[1, 1] = 1
    exp_X[2, 2] = 1
    exp_X[3, 4] = 1
    exp_X[4, 3] = 1
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    np.testing.assert_array_equal(X, exp_X)
    assert 2 == v
    assert "Optimal" == status


def test_example2_m1():
    mol_G = rdmolfiles.MolFromSmiles("[C:1][C:2]=[C:3][C:4]")
    mol_H = rdmolfiles.MolFromSmiles("[C:1].[C:2]=[C:4][C:3]")
    exp_X = np.zeros((4, 4))
    exp_X[0, 0] = 1
    exp_X[1, 1] = 1
    exp_X[2, 3] = 1
    exp_X[3, 2] = 1
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    assert "Optimal" == status
    assert 8 == v
    np.testing.assert_array_equal(X, exp_X)


def test_example2_m2():
    mol_G = rdmolfiles.MolFromSmiles("CC=CC")
    mol_H = rdmolfiles.MolFromSmiles("C.C=CC")
    exp_X = np.eye(4)
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    assert "Optimal" == status
    assert 2 == v
    np.testing.assert_array_equal(X, exp_X)


def test_1():
    solutions = []
    for ag, ah in [(0, 0), (1, 1), (1, 0), (0, 1), (2, 0), (0, 2), (2, 1), (1, 2)]:
        k = np.max([ag, ah])
        d = lp.LpVariable("d", cat=lp.LpInteger)
        g = lp.LpVariable("g", cat=lp.LpBinary)
        s = lp.LpVariable("s", cat=lp.LpBinary)

        problem = lp.LpProblem("Problem", lp.LpMinimize)

        problem += g + s
        problem += d == ag - ah
        problem += d <= k * g
        problem += -d <= k * s

        status = problem.solve()

        solutions.append(
            "({} -> {}) {} | D: {} G: {} S: {}".format(
                ag, ah, lp.LpStatus[status], d.value(), g.value(), s.value()
            )
        )

    for s in solutions:
        print(s)

    #assert False
