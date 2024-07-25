import os
import json
import logging
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops

from aamutils.utils import mol_to_graph, set_aam, graph_to_mol, is_valid_rxn_smiles
from aamutils.algorithm.ilp import expand_partial_aam_balanced
from aamutils.algorithm.aaming import get_its, get_rc, is_rc_valid

logger = logging.getLogger("aamutils")


def run(args):
    outputfile = args.o
    inputfile = args.inputfile
    if outputfile is None:
        in_path, in_ext = os.path.splitext(inputfile)
        outputfile = "{}_extended.json".format(in_path)
    rxn_smiles = []
    with open(inputfile, "r") as f:
        rxn_smiles = [line.strip() for line in f if is_valid_rxn_smiles(line)]

    results = []
    success_cnt = 0
    for smiles in rxn_smiles:
        result = {"input": smiles}
        try:
            r_smiles, p_smiles = smiles.split(">>")
            r_mol = rdmolfiles.MolFromSmiles(r_smiles)
            p_mol = rdmolfiles.MolFromSmiles(p_smiles)
            r_mol = rdmolops.AddHs(r_mol, 1)
            p_mol = rdmolops.AddHs(p_mol, 1)
            G = mol_to_graph(r_mol)
            H = mol_to_graph(p_mol)

            M, status, e_diff = expand_partial_aam_balanced(G, H)

            set_aam(G, H, M)

            r_mol = graph_to_mol(G)
            p_mol = graph_to_mol(H)

            its = get_its(G, H)
            rc = get_rc(its)
            rc_invalid = not is_rc_valid(rc)
            result_smiles = "{}>>{}".format(
                rdmolfiles.MolToSmiles(r_mol), rdmolfiles.MolToSmiles(p_mol)
            )
            result["expanded_aam"] = result_smiles
            result["ilp_status"] = status
            result["optimization_result"] = e_diff
            result["invalid_reaction_center"] = rc_invalid
            result["reaction_edges"] = len(rc.nodes)
            success_cnt += 1
        except Exception as e:
            result["error"] = str(e)
        finally:
            results.append(result)

    with open(outputfile, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(
        "Expanded {} atom-atom-maps (failed: {})".format(
            success_cnt, len(rxn_smiles) - success_cnt
        )
    )
    logger.info("The result is written to file '{}'.".format(outputfile))


def configure_parser(subparsers):
    parser = subparsers.add_parser(
        "expand", help="Utility to expand a partial atom-atom-maps."
    )
    parser.add_argument(
        "inputfile",
        help=(
            "A file containing reaction smiles with "
            + "partial atom-atom-maps. Reactions should be separated by newlines."
        ),
    )
    parser.add_argument("-o", default=None, help="Path to output file.")

    parser.set_defaults(func=run)
