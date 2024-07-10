import numpy as np
import rdkit.Chem.rdmolfiles as rdmolfiles

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
    assert 3 == v
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
    print_graph(G)
    print_graph(H)
    X, status, v = expand_partial_aam_balanced(G, H)
    print(X)
    np.testing.assert_array_equal(X, exp_X)
    assert 3 == v
    assert "Optimal" == status
    assert False
     
def test_example2():
    mol_G = rdmolfiles.MolFromSmiles("[C:1][C:2]=[C:3][C:4]")
    mol_H = rdmolfiles.MolFromSmiles("[C:1].[C:2]=[C:4][C:3]")
    exp_X = np.zeros((4, 4))
    exp_X[0, 0] = 1
    exp_X[1, 1] = 1
    exp_X[2, 2] = 1
    exp_X[3, 3] = 1
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    print_graph(G)
    print_graph(H)
    X, status, v = expand_partial_aam_balanced(G, H)
    assert "Optimal" == status
    print("X")
    print(X)
    np.testing.assert_array_equal(X, exp_X)
    assert False

