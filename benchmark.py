import time
import traceback
import copy
import json
import collections
import random
import argparse
import networkx as nx
import numpy as np

import rdkit.Chem.rdmolfiles as rdmolfiles

from aamutils.algorithm.ilp import expand_partial_aam_balanced
from aamutils.algorithm.aaming import get_its, get_rc
from aamutils.utils import mol_to_graph, set_aam


def load_data():
    with open("data/test_dataset.json", "r") as f:
        _data = json.load(f)

    data = []
    rc_size_hist = collections.defaultdict(lambda: 0)
    for i, entry in enumerate(_data):
        if entry["equivalent"] is False:
            continue
        try:
            _entry = copy.deepcopy(entry)
            smiles = _entry["local_mapper"]
            smiles = smiles.split(">>")
            g_mol = rdmolfiles.MolFromSmiles(smiles[0])
            h_mol = rdmolfiles.MolFromSmiles(smiles[1])
            G = mol_to_graph(g_mol)
            H = mol_to_graph(h_mol)
            if len(G.nodes) != len(H.nodes):
                print("Skip because unbalanced")
                continue
            ITS = get_its(G, H)
            RC = get_rc(ITS)
            _entry["G"] = G
            _entry["H"] = H
            _entry["ITS"] = ITS
            _entry["RC"] = RC
            rc_size_hist[len(RC.nodes)] += 1
            data.append(_entry)
        except Exception:
            print("Error at index {}".format(i))

    print("{} of {} are equivalent".format(len(data), len(_data)))
    return data


def run(n, remove_mode, remove_ratio, seed=None):
    if seed is not None:
        random.seed(seed)
    data = load_data()
    testcase_cnt = 0
    success_cnt = 0
    start_time = time.time()
    for i, entry in enumerate(data):
        try:
            remove_cnt = 0

            nodes = []
            if remove_mode == "keep_rc":
                rc_nodes = list(entry["RC"].nodes)
                nodes = list(entry["ITS"].nodes)
                nodes = [n for n in nodes if n not in rc_nodes]
            elif remove_mode == "rc":
                nodes = list(entry["RC"].nodes)
            else:
                raise ValueError()
            samples = random.sample(nodes, int(len(nodes) * remove_ratio))

            for rand_n in samples:
                G_idx, H_idx = nx.get_node_attributes(entry["ITS"], "idx_map")[rand_n]
                remove_cnt += 1
                entry["G"].nodes[G_idx]["aam"] = 0
                entry["H"].nodes[H_idx]["aam"] = 0

            M, status, value = expand_partial_aam_balanced(entry["G"], entry["H"])

            set_aam(entry["G"], entry["H"], M)
            ITS = get_its(entry["G"], entry["H"])
            RC = get_rc(ITS)

            success = nx.is_isomorphic(
                entry["ITS"],
                ITS,
                node_match=lambda n1, n2: n1["symbol"] == n2["symbol"],
                edge_match=lambda e1, e2: e1["bond"] == e2["bond"],
            )

            testcase_cnt += 1
            if success:
                success_cnt += 1

            print(
                (
                    "[{:>6}|{:>4}] {} {:>2} {} | Removed {} ids. "
                    + "RC Nodes: {}->{} Edges: {}->{} | "
                    + "ETA: {}"
                ).format(
                    entry["R-id"],
                    testcase_cnt,
                    status,
                    int(value),
                    "SUCC" if success else "FAIL",
                    remove_cnt,
                    len(entry["RC"].nodes),
                    len(RC.nodes),
                    len(entry["RC"].edges),
                    len(RC.edges),
                    time.strftime(
                        "%H:%M:%S",
                        time.gmtime(
                            int(time.time() - start_time)
                            * ((np.min([len(data), n]) - testcase_cnt) / testcase_cnt)
                        ),
                    ),
                )
            )

            if testcase_cnt == n:
                break
        except Exception as e:
            print("[{}] Error: {}".format(entry["R-id"], e))
            traceback.print_exc()

    print(
        ("Expanding was successful in {:.2%} ({} out of {} testcases).").format(
            success_cnt / testcase_cnt, success_cnt, testcase_cnt
        )
    )


if __name__ == "__main__":
    default_rm_mode = "rc"
    default_rm_ratio = 0.5
    default_n = 1000
    default_seed = None

    parser = argparse.ArgumentParser(
        prog="benchmarks.py",
        description="Script to run the benchmarks from the paper.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=default_n,
        help="The number of test cases. Default: {}".format(default_n),
    )
    parser.add_argument(
        "--remove-mode",
        choices=["rc", "keep_rc"],
        default=default_rm_mode,
        help=(
            "This argument selects from which atoms atom-numbers are removed "
            + "for test case generation. "
            + "(1) 'rc': remove AAM from reaction center "
            + "(use --remove-ratio to set the amount of removed atom-numbers), "
            + "(2) 'keep_rc': remove AAM from all other atoms except the "
            + "reaction center. Default: {}"
        ).format(default_rm_mode),
    )
    parser.add_argument(
        "--remove-ratio",
        type=float,
        default=default_rm_ratio,
        help=(
            "The ratio ([0, 1]) of atom-numbers to remove. "
            + "The ratio specifies the upper bound: a ratio of 0.5 will "
            + "remove 1 atom-number in a reaction center with 3 atoms."
            + " Default: {}"
        ).format(default_rm_ratio),
    )
    parser.add_argument(
        "--seed",
        default=default_seed,
        help=(
            "The seed used to initialize random number generation. " + "Default: {}"
        ).format(default_seed),
    )

    args = parser.parse_args()
    run(args.n, args.remove_mode, args.remove_ratio, args.seed)
