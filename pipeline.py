import time
import traceback
import copy
import json
import collections
import random
import networkx as nx
import numpy as np

import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops

from aamutils.algorithm.ilp import expand_partial_aam_balanced
from aamutils.algorithm.aaming import get_its, get_rc
from aamutils.utils import mol_to_graph, set_aam, graph_to_mol
from fgutils.utils import print_graph

with open("testdata.json", "r") as f:
    _data = json.load(f)

data = []
rc_size_hist = collections.defaultdict(lambda: 0)
for i, entry in enumerate(_data):
    # if len(data) == 100:
    #     break
    if entry["equivalent"] is False:
        continue
    try:
        _entry = copy.deepcopy(entry)
        smiles = _entry["local_mapper"]
        smiles = smiles.split(">>")
        g_mol = rdmolfiles.MolFromSmiles(smiles[0])
        h_mol = rdmolfiles.MolFromSmiles(smiles[1])
        # g_mol = rdmolops.AddHs(g_mol, 1)
        # h_mol = rdmolops.AddHs(h_mol, 1)
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
    except Exception as e:
        # traceback.print_exc()
        # print(smiles)
        print("Error at index {}".format(i))
        # break

print("{} of {} are equivalent".format(len(data), len(_data)))

target_cnt = 1000
remove_cnt_conf = 2
testcase_cnt = 0
success_cnt = 0
start_time = time.time()
for i, entry in enumerate(data):
    try:
        remove_cnt = 0
        nodes = list(entry["RC"].nodes)
        samples = random.sample(nodes, remove_cnt_conf)
        for rand_n in samples:
            G_idx, H_idx = nx.get_node_attributes(entry["ITS"], "idx_map")[rand_n]
            remove_cnt += 1
            entry["G"].nodes[G_idx]["aam"] = 0
            entry["H"].nodes[H_idx]["aam"] = 0
        if remove_cnt != remove_cnt_conf:
            print(
                "[{}] Skip because number of removed aams is invalid ({} != {}).".format(
                    remove_cnt_conf, remove_cnt
                )
            )
            continue

        g_mol = graph_to_mol(entry["G"])
        h_mol = graph_to_mol(entry["H"])

        M, status, value = expand_partial_aam_balanced(entry["G"], entry["H"])

        set_aam(entry["G"], entry["H"], M)
        ITS = get_its(entry["G"], entry["H"])
        RC = get_rc(ITS)

        success = len(entry["RC"].nodes) == len(RC.nodes) and len(
            entry["RC"].edges
        ) == len(RC.edges)

        testcase_cnt += 1
        if success:
            success_cnt += 1

        print(
            (
                "[{:>6}] {} {} | Removed {} aam numbers. "
                + "RC Nodes: {} -> {} Edges: {} -> {} | "
                + "ETA: {}"
            ).format(
                entry["R-id"],
                status,
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
                        * (
                            (np.min([len(data), target_cnt]) - testcase_cnt)
                            / testcase_cnt
                        )
                    ),
                ),
            )
        )

        if testcase_cnt == target_cnt:
            break
    except Exception as e:
        print("[{}] Error: {}".format(entry["R-id"], e))
        traceback.print_exc()

print(
    (
        "Expanding the RC with {} node was successful in {:.2%} "
        + "({} out of {} testcases)."
    ).format(remove_cnt_conf, success_cnt / testcase_cnt, success_cnt, testcase_cnt)
)
