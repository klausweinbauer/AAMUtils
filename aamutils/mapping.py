import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles

from aamutils.utils import graph_to_mol, mol_to_graph, set_aam
from aamutils.algorithm.ilp import expand_partial_aam_balanced


class Mapper:
    def __init__(self, include_H=False):
        self.include_H = include_H

    def _map_rxn(self, rxn_smiles: str, expected_rc):
        g_smiles, h_smiles = rxn_smiles.split(">>")
        g_mol = rdmolfiles.MolFromSmiles(g_smiles)
        h_mol = rdmolfiles.MolFromSmiles(h_smiles)
        if self.include_H:
            g_mol = rdmolops.AddHs(g_mol, 0)
            h_mol = rdmolops.AddHs(h_mol, 0)
        G = mol_to_graph(g_mol)
        H = mol_to_graph(h_mol)

        M, status, e_diff = expand_partial_aam_balanced(G, H, expected_rc=expected_rc)
        if status != "Optimal":
            return None

        set_aam(G, H, M)

        g_mol = graph_to_mol(G)
        h_mol = graph_to_mol(H)

        aam_smiles = "{}>>{}".format(
            rdmolfiles.MolToSmiles(g_mol), rdmolfiles.MolToSmiles(h_mol)
        )
        return aam_smiles

    def get_aam(self, smiles: str | list[str], expected_rc=None) -> str | list[str]:
        return_list = True
        if isinstance(smiles, str):
            smiles = [smiles]
            return_list = False
        if expected_rc is None:
            expected_rc = [None for _ in range(len(smiles))]
        elif not isinstance(expected_rc, list):
            expected_rc = [expected_rc]
        if len(smiles) != len(expected_rc):
            raise ValueError(
                "Argument min_bonds must be of same length ({}) as smiles.".format(
                    len(smiles)
                )
            )
        aam_smiles = []
        for smiles, exp_rc in zip(smiles, expected_rc):
            aam_smiles.append(self._map_rxn(smiles, expected_rc=exp_rc))
        if not return_list:
            return aam_smiles[0]
        return aam_smiles
