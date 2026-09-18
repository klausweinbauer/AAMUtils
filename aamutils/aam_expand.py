import networkx as nx
import rdkit.Chem.rdmolfiles as rdmolfiles
from aamutils.algorithm.ilp import expand_partial_aam_balanced
from aamutils.utils import smiles_to_graph, graph_to_mol, set_aam, mol_to_graph
from typing import Optional
from rdkit.Chem import MolFromSmiles, MolToSmiles


def extend_aam_from_graph(G: nx.Graph, H: nx.Graph) -> str:
    """
    Extends atom-atom mappings (AAM) from two input graphs,
    G (reactants) and H (products),by solving the AAM problem using an
    Integer Linear Programming (ILP) approach and generating
    a reaction SMILES (RSMI) string.

    Parameters:
    - G (nx.Graph): Graph representing the reactants.
    Nodes should contain necessary attributes for AAM.
    - H (nx.Graph): Graph representing the products.
    Nodes should contain necessary attributes for AAM.

    Returns:
    - str: A reaction SMILES string (RSMI) in the format 'reactant>>product'
    with extended atom mappings.
    """
    # Solve the partial AAM problem using ILP and retrieve the mapping matrix
    M, _, _ = expand_partial_aam_balanced(G, H)

    # Apply the AAM matrix to the graphs
    set_aam(G, H, M)

    # Convert the modified graphs back to RDKit molecules
    r_mol = graph_to_mol(G)
    p_mol = graph_to_mol(H)

    # Generate reaction SMILES string from the RDKit molecules
    result_smiles = "{}>>{}".format(
        rdmolfiles.MolToSmiles(
            r_mol, canonical=True, kekuleSmiles=False, allHsExplicit=True
        ),
        rdmolfiles.MolToSmiles(
            p_mol, canonical=True, kekuleSmiles=False, allHsExplicit=True
        ),
    )
    return result_smiles


def extend_aam_from_rsmi(partial_rxn_smiles: str, time_limit: int = 1200) -> str:
    assert isinstance(partial_rxn_smiles, str) and ">>" in partial_rxn_smiles, (
        f"Expected reaction string containing '>>', got: {partial_rxn_smiles!r}"
    )

    r_smi, p_smi = partial_rxn_smiles.split(">>")
    r_mol = MolFromSmiles(r_smi)
    p_mol = MolFromSmiles(p_smi)

    assert r_mol is not None and p_mol is not None, (
        f"Failed to parse molecules: reactant='{r_smi}', product='{p_smi}'"
    )

    assert r_mol.GetNumAtoms() == p_mol.GetNumAtoms(), (
        f"Atom count mismatch: {r_mol.GetNumAtoms()} reactant atoms vs "
        f"{p_mol.GetNumAtoms()} product atoms in '{partial_rxn_smiles}'"
    )

    g_graph = mol_to_graph(r_mol)
    h_graph = mol_to_graph(p_mol)

    mapping_matrix, status, _ = expand_partial_aam_balanced(
        g_graph, h_graph, time_limit=time_limit
    )

    if not status == "Optimal":
        print(
        f"ILP solver did not find an optimal solution (status='{status}') "
        f"for: {partial_rxn_smiles}"
    )

    set_aam(g_graph, h_graph, mapping_matrix)

    for atom in r_mol.GetAtoms():
        atom.SetAtomMapNum(int(g_graph.nodes[atom.GetIdx()]["aam"]))

    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(int(h_graph.nodes[atom.GetIdx()]["aam"]))

    return f"{MolToSmiles(r_mol)}>>{MolToSmiles(p_mol)}"
