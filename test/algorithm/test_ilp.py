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
    assert 3 == v
    assert "Optimal" == status
