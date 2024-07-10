import rdkit.Chem.rdmolfiles as rdmolfiles

from aamutils.utils import mol_to_graph
from aamutils.algorithm.aaming import get_its, get_rc


def test_its():
    g_smiles = "[I:1][I:9].[cH:6]1[cH:7][cH:2][cH:3][cH:4][cH:5]1"
    h_smiles = "[I:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[H+].[I-:9]"
    g_mol = rdmolfiles.MolFromSmiles(g_smiles)
    h_mol = rdmolfiles.MolFromSmiles(h_smiles)
    G = mol_to_graph(g_mol)
    H = mol_to_graph(h_mol)
    ITS = get_its(G, H)
    RC = get_rc(ITS)
    assert 3 == len(RC.nodes)
    assert all(i in RC.nodes for i in [1, 2, 9])


def test_partial_aam_with_unmapped_H():
    g_smiles = "[CH2:1]=[O:2].[CH3:3][NH2:4].[H][H]"
    h_smiles = "[CH3:3][NH:4][CH3:1].[OH2:2]"
    g_mol = rdmolfiles.MolFromSmiles(g_smiles)
    h_mol = rdmolfiles.MolFromSmiles(h_smiles)
    G = mol_to_graph(g_mol)
    H = mol_to_graph(h_mol)
    ITS = get_its(G, H)
    RC = get_rc(ITS)
    assert 4 == len(ITS.nodes)
    assert 3 == len(RC.nodes)
    assert all([i in ITS.nodes for i in [1, 2, 3, 4]])
    assert all([i in RC.nodes for i in [1, 2, 4]])
    assert 2 == len(RC.edges)
    assert (2, None) == RC[1][2]["bond"]
    assert (None, 1) == RC[1][4]["bond"]
