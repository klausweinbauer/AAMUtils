import collections
import numpy as np
import networkx as nx
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdDepictor as rdDepictor


def mol_to_graph(mol: Chem.rdchem.Mol) -> nx.Graph:
    bond_order_map = {
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "QUADRUPLE": 4,
        "AROMATIC": 1.5,
    }
    g = nx.Graph()
    for atom in mol.GetAtoms():
        aam = atom.GetAtomMapNum()
        g.add_node(atom.GetIdx(), symbol=atom.GetSymbol(), aam=aam)
    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType()).split(".")[-1]
        bond_order = 1
        if bond_type in bond_order_map.keys():
            bond_order = bond_order_map[bond_type]
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond=bond_order)
    return g


def smiles_to_graph(smiles: str) -> nx.Graph | tuple[nx.Graph, nx.Graph]:
    if ">>" in smiles:
        smiles_token = smiles.split(">>")
        g = mol_to_graph(rdmolfiles.MolFromSmiles(smiles_token[0]))
        h = mol_to_graph(rdmolfiles.MolFromSmiles(smiles_token[1]))
        return g, h
    else:
        return mol_to_graph(rdmolfiles.MolFromSmiles(smiles))


def graph_to_mol(
    G: nx.Graph, symbol_key="symbol", aam_key="aam", bond_type_key="bond"
) -> Chem.rdchem.Mol:
    bond_order_map = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        1.5: Chem.rdchem.BondType.AROMATIC,
    }
    rw_mol = Chem.rdchem.RWMol()
    idx_map = {}
    for n, d in G.nodes(data=True):
        idx = rw_mol.AddAtom(Chem.rdchem.Atom(d[symbol_key]))
        idx_map[n] = idx
        if aam_key in d.keys() and d[aam_key] >= 0:
            rw_mol.GetAtomWithIdx(idx).SetAtomMapNum(d[aam_key])
    for n1, n2, d in G.edges(data=True):
        idx1 = idx_map[n1]
        idx2 = idx_map[n2]
        rw_mol.AddBond(idx1, idx2, bond_order_map[d[bond_type_key]])
    return rw_mol.GetMol()


def get_beta_map(G, H, aam_key="aam"):
    node2aam_G = collections.defaultdict(
        lambda: -1, {n: d[aam_key] for (n, d) in G.nodes(data=True) if d[aam_key] > 0}
    )
    aam2node_H = collections.defaultdict(
        lambda: -1, {d[aam_key]: n for (n, d) in H.nodes(data=True) if d[aam_key] > 0}
    )
    beta_map = []
    for n_G, aam_G in node2aam_G.items():
        if aam_G in aam2node_H and aam2node_H[aam_G] > -1:
            beta_map.append((n_G, aam2node_H[aam_G], aam_G))
    return beta_map


def set_aam(G, H, M, beta_map=None, aam_key="aam"):
    if beta_map is None:
        beta_map = get_beta_map(G, H)
    used_atom_numbers = [aam for _, _, aam in beta_map]

    aam_G = collections.defaultdict(lambda: -1)
    for bi, _, aam in beta_map:
        aam_G[bi] = aam

    next_aam_nr = 1
    for n, d in G.nodes(data=True):
        if aam_G[n] > -1:
            aam_nr = aam_G[n]
        else:
            while next_aam_nr in used_atom_numbers:
                next_aam_nr += 1
            aam_nr = next_aam_nr
            used_atom_numbers.append(aam_nr)
        d[aam_key] = int(aam_nr)
        aam_G[n] = aam_nr

    aam_G = np.array([v for _, v in sorted(aam_G.items(), key=lambda x: x[0])])
    for (_, d), aam in zip(H.nodes(data=True), np.dot(aam_G.T, M)):
        d[aam_key] = int(aam)


def is_valid_rxn_smiles(smiles):
    smiles_token = smiles.split(">>")
    if len(smiles_token) != 2:
        return False
    mol1 = rdmolfiles.MolFromSmiles(smiles_token[0])
    if mol1 is None:
        return False
    mol2 = rdmolfiles.MolFromSmiles(smiles_token[0])
    if mol2 is None:
        return False
    return True


def print_graph(graph):
    print(
        "Graph Nodes: {}".format(
            " ".join(
                [
                    "[{}]{}:{}".format(
                        n[0], n[1].get("symbol", None), n[1].get("aam", None)
                    )
                    for n in graph.nodes(data=True)
                ]
            )
        )
    )
    print(
        "Graph Edges: {}".format(
            " ".join(
                [
                    "[{}]-[{}]:{}".format(n[0], n[1], n[2]["bond"])
                    for n in graph.edges(data=True)
                ]
            )
        )
    )


def its2mol(its: nx.Graph, aam_key="aam", bond_key="bond") -> Chem.rdchem.Mol:
    _its = its.copy()
    for n in _its.nodes:
        _its.nodes[n][aam_key] = n
    for u, v in _its.edges():
        _its[u][v][bond_key] = 1
    return graph_to_mol(_its)


def plot_its(its, ax, bond_key="bond", aam_key="aam", symbol_key="symbol"):
    bond_char = {None: "∅", 1: "—", 2: "=", 3: "≡"}
    mol = its2mol(its, aam_key=aam_key, bond_key=bond_key)

    positions = {}
    conformer = rdDepictor.Compute2DCoords(mol)
    for i, atom in enumerate(mol.GetAtoms()):
        aam = atom.GetAtomMapNum()
        apos = mol.GetConformer(conformer).GetAtomPosition(i)
        positions[aam] = [apos.x, apos.y]

    ax.axis("equal")
    ax.axis("off")

    nx.draw_networkx_edges(its, positions, edge_color="#000000", ax=ax)
    nx.draw_networkx_nodes(its, positions, node_color="#FFFFFF", node_size=500, ax=ax)

    labels = {n: "{}:{}".format(d[symbol_key], n) for n, d in its.nodes(data=True)}
    edge_labels = {}
    for u, v, d in its.edges(data=True):
        bc1 = d[bond_key][0]
        bc2 = d[bond_key][1]
        if bc1 == bc2:
            continue
        if bc1 in bond_char.keys():
            bc1 = bond_char[bc1]
        if bc2 in bond_char.keys():
            bc2 = bond_char[bc2]
        edge_labels[(u, v)] = "({},{})".format(bc1, bc2)

    nx.draw_networkx_labels(its, positions, labels=labels, ax=ax)
    nx.draw_networkx_edge_labels(its, positions, edge_labels=edge_labels, ax=ax)
