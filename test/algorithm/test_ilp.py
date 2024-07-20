import numpy as np
import rdkit.Chem.rdmolfiles as rdmolfiles

from aamutils.utils import mol_to_graph
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
    mol_G = rdmolfiles.MolFromSmiles("[C:1][C:2][C:3][Cl].[N]")
    mol_H = rdmolfiles.MolFromSmiles("[C:1][C:2][C:3][N].[Cl]")
    exp_X = np.zeros((5, 5))
    exp_X[0, 0] = 1
    exp_X[1, 1] = 1
    exp_X[2, 2] = 1
    exp_X[3, 4] = 1
    exp_X[4, 3] = 1
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    assert 4 == v
    assert "Optimal" == status
    np.testing.assert_array_equal(X, exp_X)


def test_example1_without_aam():
    # The tested mapping is not correct. This is to show equal solutions based
    # on the implemented optimization criterion. The correct mapping is in
    # test_example1()
    mol_G = rdmolfiles.MolFromSmiles("CCC[Cl].[N]")
    mol_H = rdmolfiles.MolFromSmiles("CCC[N].[Cl]")
    exp_X = np.zeros((5, 5))
    exp_X[0, 2] = 1
    exp_X[1, 1] = 1
    exp_X[2, 0] = 1
    exp_X[3, 4] = 1
    exp_X[4, 3] = 1
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    assert 4 == v
    assert "Optimal" == status
    np.testing.assert_array_equal(X, exp_X)


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
    assert 6 == v
    np.testing.assert_array_equal(X, exp_X)


def test_example2_m2():
    mol_G = rdmolfiles.MolFromSmiles("CC=CC")
    mol_H = rdmolfiles.MolFromSmiles("C.C=CC")
    exp_X = np.zeros((4, 4))
    exp_X[0, 3] = 1
    exp_X[1, 2] = 1
    exp_X[2, 1] = 1
    exp_X[3, 0] = 1
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    assert "Optimal" == status
    assert 2 == v
    np.testing.assert_array_equal(X, exp_X)


def test_edge_diff_constraint():
    G_smiles = "CCCC"
    H_smiles = "CCCC"
    mol_G = rdmolfiles.MolFromSmiles(G_smiles)
    mol_H = rdmolfiles.MolFromSmiles(H_smiles)
    G = mol_to_graph(mol_G)
    H = mol_to_graph(mol_H)
    X, status, v = expand_partial_aam_balanced(G, H)
    assert v == 0
    assert "Optimal" == status
