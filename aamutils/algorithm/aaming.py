import collections
import networkx as nx


def _add_its_nodes(ITS, G, H, eta, symbol_key):
    eta_G, eta_G_inv, eta_H, eta_H_inv = eta[0], eta[1], eta[2], eta[3]
    for n, d in G.nodes(data=True):
        n_ITS = eta_G[n]
        n_H = eta_H_inv[n_ITS]
        if n_ITS is not None and n_H is not None:
            ITS.add_node(n_ITS, symbol=d[symbol_key], idx_map=(n, n_H))
    for n, d in H.nodes(data=True):
        n_ITS = eta_H[n]
        n_G = eta_G_inv[n_ITS]
        if n_ITS is not None and n_G is not None and n_ITS not in ITS.nodes:
            ITS.add_node(n_ITS, symbol=d[symbol_key], idx_map=(n_G, n))


def _add_its_edges(ITS, G, H, eta, bond_key):
    eta_G, eta_G_inv, eta_H, eta_H_inv = eta[0], eta[1], eta[2], eta[3]
    for n1, n2, d in G.edges(data=True):
        if n1 > n2:
            continue
        e_G = d[bond_key]
        n_ITS1 = eta_G[n1]
        n_ITS2 = eta_G[n2]
        n_H1 = eta_H_inv[n_ITS1]
        n_H2 = eta_H_inv[n_ITS2]
        e_H = None
        if H.has_edge(n_H1, n_H2):
            e_H = H[n_H1][n_H2][bond_key]
        if not ITS.has_edge(n_ITS1, n_ITS2) and n_ITS1 > 0 and n_ITS2 > 0:
            ITS.add_edge(n_ITS1, n_ITS2, bond=(e_G, e_H))

    for n1, n2, d in H.edges(data=True):
        if n1 > n2:
            continue
        e_H = d[bond_key]
        n_ITS1 = eta_H[n1]
        n_ITS2 = eta_H[n2]
        n_G1 = eta_G_inv[n_ITS1]
        n_G2 = eta_G_inv[n_ITS2]
        if n_G1 is None or n_G2 is None:
            continue
        if not G.has_edge(n_G1, n_G2) and n_ITS1 > 0 and n_ITS2 > 0:
            ITS.add_edge(n_ITS1, n_ITS2, bond=(None, e_H))


def get_its(
    G: nx.Graph, H: nx.Graph, aam_key="aam", symbol_key="symbol", bond_key="bond"
) -> nx.Graph:
    """Get the ITS graph of reaction G \u2192 H.

    :param G: Educt molecular graph.
    :param H: Product molecular graph.
    :param aam_key: (optional) The node label that encodes atom indices on the
        molecular graph.
    :param symbol_key: (optional) The node label that encodes atom symbols on
        the molecular graph.
    :param bond_key: (optional) The edge label that encodes bond order on the
        molecular graph.

    :returns: Returns the ITS graph.
    """
    eta_G = collections.defaultdict(lambda: None)
    eta_G_inv = collections.defaultdict(lambda: None)
    eta_H = collections.defaultdict(lambda: None)
    eta_H_inv = collections.defaultdict(lambda: None)
    eta = (eta_G, eta_G_inv, eta_H, eta_H_inv)

    for n, d in G.nodes(data=True):
        if aam_key in d.keys() and d[aam_key] >= 0:
            eta_G[n] = d[aam_key]
            eta_G_inv[d[aam_key]] = n
    for n, d in H.nodes(data=True):
        if aam_key in d.keys() and d[aam_key] >= 0:
            eta_H[n] = d[aam_key]
            eta_H_inv[d[aam_key]] = n

    ITS = nx.Graph()
    _add_its_nodes(ITS, G, H, eta, symbol_key)
    _add_its_edges(ITS, G, H, eta, bond_key)

    return ITS


def get_rc(ITS: nx.Graph, symbol_key="symbol", bond_key="bond") -> nx.Graph:
    """Get the reaction center (RC) graph from an ITS graph.

    :param ITS: The ITS graph to get the RC from.
    :param symbol_key: (optional) The node label that encodes atom symbols on
        the molecular graph.
    :param bond_key: (optional) The edge label that encodes bond order on the
        molecular graph.

    :returns: Returns the reaction center (RC) graph.
    """
    rc = nx.Graph()
    for n1, n2, d in ITS.edges(data=True):
        edge_label = d[bond_key]
        if edge_label[0] != edge_label[1]:
            rc.add_node(n1, **{symbol_key: ITS.nodes[n1][symbol_key]})
            rc.add_node(n2, **{symbol_key: ITS.nodes[n2][symbol_key]})
            rc.add_edge(n1, n2, **{bond_key: edge_label})
    return rc


def is_rc_valid(RC: nx.Graph, symbol_key="symbol") -> bool:
    """Checks if a reaction center (RC) graph is valid. A valid RC does not
    mean it is chemically correct. This function only checks the necessary
    structural requirements for a correct RC. A chemically correct RC is always
    valid but a valid RC is not necessarily chemically correct.

    :param symbol_key: (optional) The node label that encodes atom symbols on
        the molecular graph.

    :returns: Returns ``True`` if the RC graph is valid, ``False`` otherwise.
    """
    for n, d in RC.degree():
        if d < 2:
            return False, "Atom {} {} has degree {}.".format(
                n, RC.nodes[n][symbol_key], d
            )
    return True
