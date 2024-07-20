import numpy as np
import pulp as lp
import networkx as nx

from aamutils.utils import get_beta_map


def _bijection_constraint(problem, X):
    m = int(np.sqrt(len(X.keys())))

    for j in range(m):
        problem += (
            lp.lpSum([X[i, j] for i in range(m)]) == 1,
            "bijection_col_{}".format(j),
        )
    for i in range(m):
        problem += (
            lp.lpSum([X[i, j] for j in range(m)]) == 1,
            "bijection_row_{}".format(i),
        )


def _beta_constraint(problem, X, beta_map):
    for bi, bj, _ in beta_map:
        problem += (X[bi, bj] == 1, "beta_constraint1_{}-{}".format(bi, bj))


def _edge_diff_constraint(problem, X, D, A_G, A_H):
    m = int(np.sqrt(len(D.keys())))
    for i in range(m):
        for j in range(m):
            problem += D[i, j] == lp.lpSum(
                [X[i, k] * A_H[k, j] for k in range(m)]
            ) - lp.lpSum([A_G[i, k] * X[k, j] for k in range(m)])


def _atom_type_constraint(problem, X, G, H, symbol_key="symbol"):
    m = int(np.sqrt(len(X.keys())))
    assert m == len(G.nodes)
    assert m == len(H.nodes)
    for i in range(m):
        for j in range(m):
            g_sym = G.nodes(data=True)[i][symbol_key]
            h_sym = H.nodes(data=True)[j][symbol_key]
            assert isinstance(g_sym, str)
            assert isinstance(h_sym, str)
            if g_sym != h_sym:
                problem += (X[i, j] == 0, "atom_type_constraint_{}-{}".format(i, j))


def _indicator_constraint(problem, D, G, S, k):
    m = int(np.sqrt(len(D.keys())))
    assert m == int(np.sqrt(len(G.keys())))
    assert m == int(np.sqrt(len(S.keys())))

    for i in range(m):
        for j in range(m):
            problem += (
                D[i, j] <= k * G[i, j],
                "g_indicator_constraint_lower_{}-{}".format(i, j),
            )
            problem += (
                -D[i, j] <= k * S[i, j],
                "s_indicator_constraint_lower_{}-{}".format(i, j),
            )


def expand_partial_aam_balanced(
    G: nx.Graph,
    H: nx.Graph,
    beta_map: None | list[tuple[int, int, int]] = None,
    bond_key="bond",
) -> tuple[np.ndarray, str, float]:
    """Function to extend the partial atom-atom map beta of the balanced
    reaction G \u2192 H to a full atom-atom map based on the Minimal Chemical
    Distance (minimize the number of changing bonds).

    :param G: Educt molecular graph.
    :param H: Product molecular graph.
    :param beta_map: (optional) A list of predefined atom-atom mappings. If not
        specified the existing mapping in G and H will be used.
    :param bond_key: (optional) The edge label in G and H which encodes the
        bond order.

    :returns: A 3-tuple. The first element is the mapping matrix X. The second
        element is the status string of the ILP solver (e.g. 'Optimal') and the
        third is the ILP solution value.
    """

    A_G = nx.adjacency_matrix(G, weight=bond_key).todense()
    A_H = nx.adjacency_matrix(H, weight=bond_key).todense()
    k = np.max([np.max(A_G), np.max(A_H)])
    m = len(G.nodes)

    if m != len(H.nodes):
        raise ValueError(
            (
                "Reaction is not balanced. " + "{} reactant atoms and {} product atoms."
            ).format(len(G.nodes), len(H.nodes))
        )

    if beta_map is None:
        beta_map = get_beta_map(G, H)

    # Mapping matrix X
    lp_X = lp.LpVariable.dicts(
        "X",
        [(i, j) for i in range(m) for j in range(m)],
        cat=lp.LpBinary,
    )

    # Difference matrix D
    lp_D = lp.LpVariable.dicts("D", [(i, j) for i in range(m) for j in range(m)])

    # Indicator matrix G and S
    lp_G = lp.LpVariable.dicts(
        "G",
        [(i, j) for i in range(m) for j in range(m)],
        cat=lp.LpBinary,
    )
    lp_S = lp.LpVariable.dicts(
        "S",
        [(i, j) for i in range(m) for j in range(m)],
        cat=lp.LpBinary,
    )

    # Non-zero counter function for D
    lp_f = lp.LpVariable("f", 0, None, lp.LpInteger)

    problem = lp.LpProblem("AAM", lp.LpMinimize)

    problem += (lp_f, "objective")
    problem += (
        lp_f == lp.lpSum([lp_G[i, j] + lp_S[i, j] for i in range(m) for j in range(m)]),
        "objective",
    )

    _bijection_constraint(problem, lp_X)
    _beta_constraint(problem, lp_X, beta_map)
    _atom_type_constraint(problem, lp_X, G, H)
    _edge_diff_constraint(problem, lp_X, lp_D, A_G, A_H)
    _indicator_constraint(problem, lp_D, lp_G, lp_S, k)

    status = problem.solve(lp.PULP_CBC_CMD(logPath=r"solver.log"))

    np_X = np.zeros([m, m], dtype=np.int32)
    for (i, j), v in lp_X.items():
        np_X[i, j] = int(v.value())

    return np_X, lp.LpStatus[status], lp_f.value()
