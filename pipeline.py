import traceback
import copy
import json
import collections

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
    # if i != 19:
    #     continue
    if len(data) == 100:
        break
    if entry["equivalent"] is False:
        continue
    try:
        _entry = copy.deepcopy(entry)
        smiles = _entry["local_mapper"]
        smiles = smiles.split(">>")
        g_mol = rdmolfiles.MolFromSmiles(smiles[0])
        h_mol = rdmolfiles.MolFromSmiles(smiles[1])
        g_mol = rdmolops.AddHs(g_mol, 1)
        h_mol = rdmolops.AddHs(h_mol, 1)
        G = mol_to_graph(g_mol)
        H = mol_to_graph(h_mol)
        ITS = get_its(G, H)
        RC = get_rc(ITS)
        _entry["G"] = G
        _entry["H"] = H
        _entry["ITS"] = ITS
        _entry["RC"] = RC
        rc_size_hist[len(RC.nodes)] += 1
        data.append(_entry)
    except Exception as e:
        traceback.print_exc()
        print(smiles)
        print("Error at index {}".format(i))
        break

print("{} of {} are equivalent".format(len(data), len(_data)))

expanded_data = []
output_data = []
for i, entry in enumerate(data):
    try:
        remove_cnt = 0
        # print(entry["G"].nodes(data=True))
        # print(entry["H"].nodes(data=True))
        # print(entry["local_mapper"])
        for n, d in entry["ITS"].nodes(data=True):
            if n not in entry["RC"]:
                remove_cnt += 1
                G_idx, H_idx = d["idx_map"]
                entry["G"].nodes[G_idx]["aam"] = 0
                entry["H"].nodes[H_idx]["aam"] = 0
        if remove_cnt == 0:
            print("[{}] Skip because RC == ITS".format(entry["R-id"]))
            continue

        M, status, value = expand_partial_aam_balanced(entry["G"], entry["H"])

        set_aam(entry["G"], entry["H"], M)
        ITS = get_its(entry["G"], entry["H"])
        RC = get_rc(ITS)

        print(
            "[{}] Removed {} aam numbers. RC {} -> {}".format(
                entry["R-id"], remove_cnt, len(entry["RC"].nodes), len(RC.nodes)
            )
        )

        g_mol = graph_to_mol(entry["G"])
        h_mol = graph_to_mol(entry["H"])

        g_mol = rdmolops.RemoveHs(g_mol, 0)
        h_mol = rdmolops.RemoveHs(h_mol, 0)

        entry["expanded"] = "{}>>{}".format(
            rdmolfiles.MolToSmiles(g_mol), rdmolfiles.MolToSmiles(h_mol)
        )
        expanded_data.append(entry)
        output_data.append(
            {
                "R-id": entry["R-id"],
                "local_mapper": entry["local_mapper"],
                "expanded_aam": entry["expanded"],
            }
        )
        if len(output_data) % 2 == 0:
            with open("expanded.json", "w") as f:
                json.dump(output_data, f, indent=4)
                break
    except Exception as e:
        print("[{}] Error: {}".format(entry["R-id"], e))

print("Expanded aams {}".format(len(expanded_data)))


# ordered_data = sorted(enumerate(data), key=lambda x: (len(x[1]["G"]), len(x[1]["RC"])))
# print(ordered_data[0])
# for i, entry in enumerate(ordered_data):
#     if len(entry["RC"]) > 7:
#         print("Index: ", i)
#         break

# print(ordered_data[i])
# print_graph(ordered_data[i]["G"])
# print_graph(ordered_data[i]["H"])
# print_graph(ordered_data[i]["RC"])
# print(len(ordered_data[i]["G"]))
# print(len(ordered_data[i]["RC"]))
# print(ordered_data[i]["local_mapper"])
