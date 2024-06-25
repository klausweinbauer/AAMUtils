import numpy as np
import pulp as lp
import networkx as nx


def _bijection_constraint(problem, M, n):
    G_n, H_n = n[0], n[1]
    for i in range(G_n):
        for j in range(H_n):
            problem += M[i, j] >= 0
            problem += M[i, j] <= 1

    for j in range(H_n):
        problem += (
            lp.lpSum([M[i, j] for i in range(G_n)]) == 1,
            "bijection_col_{}".format(j),
        )
    for i in range(G_n):
        problem += (
            lp.lpSum([M[i, j] for j in range(H_n)]) == 1,
            "bijection_row_{}".format(i),
        )


def _beta_constraint(problem, M, n, delta, A_G, A_H, beta_map):
    G_n, H_n = n[0], n[1]
    for bi, bj, _ in beta_map:
        problem += (M[bi, bj] == 1, "beta_constraint1_{}-{}".format(bi, bj))

    for i in range(G_n):
        for j in range(H_n):
            problem += delta[i, j] == A_H[i, j] - (
                lp.lpSum([A_G[i, k] * M[k, j] for k in range(G_n)])
            )


def _atom_type_constraint(problem, M, n, G, H):
    G_n, H_n = n[0], n[1]
    for i in range(G_n):
        for j in range(H_n):
            g_sym = G.nodes(data=True)[i]
            h_sym = H.nodes(data=True)[j]
            if g_sym != h_sym:
                problem += (M[i, j] == 0, "atom_type_constraint_{}-{}".format(i, j))


def _absolute_edge_diff(problem, n, delta, delta_abs):
    G_n, H_n = n[0], n[1]
    for i in range(G_n):
        for j in range(H_n):
            problem += delta_abs[i, j] >= delta[i, j]
            problem += delta_abs[i, j] >= -delta[i, j]


def expand_partial_aam_balanced(G, H, beta_map=[], bond_key="bond"):
    A_G = nx.adjacency_matrix(G, weight=bond_key)
    A_H = nx.adjacency_matrix(H, weight=bond_key)

    G_n = len(G.nodes)
    H_n = len(H.nodes)

    if G_n != H_n:
        raise ValueError(
            (
                "Reaction is not balanced. " + "{} reactant atoms and {} product atoms."
            ).format(len(G.nodes), len(H.nodes))
        )

    M = lp.LpVariable.dicts(
        "M",
        [(i, j) for i in range(G_n) for j in range(H_n)],
        lowBound=0,
        upBound=1,
        cat=lp.LpInteger,
    )
    delta = lp.LpVariable.dicts(
        "delta", [(i, j) for i in range(G_n) for j in range(H_n)]
    )
    delta_abs = lp.LpVariable.dicts(
        "delta_abs", [(i, j) for i in range(G_n) for j in range(H_n)]
    )

    e_diff = lp.LpVariable("e_diff", 0, None, lp.LpInteger)

    problem = lp.LpProblem("AAM", lp.LpMinimize)

    problem += (e_diff, "objective")
    problem += (
        e_diff == lp.lpSum([delta_abs[i, j] for i in range(G_n) for j in range(H_n)]),
        "objective",
    )

    n = (G_n, H_n)
    _bijection_constraint(problem, M, n)
    _beta_constraint(problem, M, n, delta, A_G, A_H, beta_map)
    _atom_type_constraint(problem, M, n, G, H)
    _absolute_edge_diff(problem, n, delta, delta_abs)

    problem.solve(lp.PULP_CBC_CMD(logPath=r"solver.log"))

    np_M = np.zeros([G_n, H_n])
    for (i, j), v in M.items():
        np_M[i, j] = v.value()

    return lp.LpStatus[problem.status], np_M, e_diff.value()
