import os
import logging
import networkx as nx
import argparse

from project.matching import match_graphs, MatchingType
from project.utils import read_graph_from_file, resample_graph, reduce_graphs,\
        visualize_matching, write_swc
from project.toydata import *
import glob
import pdb


# logger = logging.getLogger(__name__)
#
#
# def get_arguments():
#     # parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--gt_fn", type=str, dest="gt_fn", default=None,
#         help="path to ground truth folder"
#     )
#     parser.add_argument(
#         "--pred_fn", type=str, dest="pred_fn", default=None,
#         help="path to predicted folder"
#     )
#
#     parser.add_argument(
#         "--output_folder", type=str, default=None,
#         help="path to output folder"
#     )
#     parser.add_argument(
#         "--debug", action="store_true", default=False,
#         help="flag to set logging to DEBUG"
#     )
#
#     args = parser.parse_args()
#     return args
#
#
# def main():
#     args = get_arguments()
#     if args.debug:
#         logging.basicConfig(level=logging.DEBUG)
#     else:
#         logging.basicConfig(level=logging.INFO)
#
#     # 1) Load graphs with attributes
#     # NOTE: We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
#     if args.gt_fn is not None and args.pred_fn is not None:
#         G = read_graph_from_file(args.gt_fn)
#         print("number of nodes in G:", G.number_of_nodes())
#         H = read_graph_from_file(args.pred_fn)
#         print("number of nodes in H:", H.number_of_nodes())
#     else:
#         G, H = two_trees()
#         G = read_graph_from_file("tests/toygraph_GT.swc")
#         H = read_graph_from_file("tests/toygraph_predicted.swc")
#
#
#     matching_type = MatchingType.Hierarchical
#     matched = match_graphs(
#         source=G,
#         target=H,
#         matching_type=matching_type,
#         matching_dist="fixed",
#         max_distance=5,
#         visualize=False)
#     print(len(matched.keys()), G.number_of_nodes())
#
#     # find all not matched pred nodes with degree 2
#     unmatched_deg2_nodes = []
#     for node in H.nodes():
#         # Check if node is not in the values of the match dict and has degree 2
#         if node not in matched.values() and H.degree(node) == 2:
#             unmatched_deg2_nodes.append(node)
#
#     print(len(unmatched_deg2_nodes))
#
#     for node in unmatched_deg2_nodes:
#         neighbors = list(H.neighbors(node))
#
#         # For undirected graph, also check predecessors if needed
#         if isinstance(H, nx.DiGraph):
#             predecessors = list(H.predecessors(node))
#             successors = list(H.successors(node))
#             if len(predecessors) == 1 and len(successors) == 1:
#                 pred = predecessors[0]
#                 succ = successors[0]
#                 # Add edge from pred -> succ if it doesn't already exist
#                 if not H.has_edge(pred, succ):
#                     H.add_edge(pred, succ)
#                 H.remove_node(node)
#         else:
#             if len(neighbors) == 2:
#                 n1, n2 = neighbors
#                 if not H.has_edge(n1, n2):
#                     H.add_edge(n1, n2)
#                 H.remove_node(node)
#
#     # write to swc
#     outfn = os.path.join(args.output_folder, os.path.basename(args.pred_fn).replace(".swc","_gt_res.swc"))
#     write_swc(H, outfn)
#
#
# if __name__ == "__main__":
#     main()

import os
import logging
import networkx as nx
import argparse
import glob

from project.matching import match_graphs, MatchingType
from project.utils import read_graph_from_file, write_swc
from project.toydata import *

logger = logging.getLogger(__name__)


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_fn", type=str, dest="gt_fn", default=None,
        help="path to ground truth FOLDER containing .swc files"
    )
    parser.add_argument(
        "--pred_fn", type=str, dest="pred_fn", default=None,
        help="path to predicted FOLDER containing .swc files"
    )
    parser.add_argument(
        "--output_folder", type=str, default=None,
        help="path to output folder"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="flag to set logging to DEBUG"
    )
    parser.add_argument(
        "--candidate_selection", type=str, default="query",
        choices=["query", "ball"],
        help="method to select matching candidates"
    )

    args = parser.parse_args()
    return args


def process_pair(gt_path, pred_path, output_folder):
    """Run matching + degree-2 cleanup for a single GT / prediction pair."""
    logger.info(f"Processing GT: {gt_path}")
    logger.info(f"Processing Pred: {pred_path}")

    # 1) Load graphs
    G = read_graph_from_file(gt_path)
    logger.info(f"number of nodes in G (GT): {G.number_of_nodes()}")
    H = read_graph_from_file(pred_path)
    logger.info(f"number of nodes in H (Pred): {H.number_of_nodes()}")

    # 2) Matching
    matching_type = MatchingType.Hierarchical
    matched = match_graphs(
        source=G,
        target=H,
        matching_type=matching_type,
        matching_dist="fixed",
        max_distance=1.5,
        candidate_selection="ball",
        visualize=False,
    )
    logger.info(f"number of matched nodes: {len(matched.keys())} / {G.number_of_nodes()}")

    # 3) Find all not matched pred nodes with degree 2
    unmatched_deg2_nodes = []
    for node in H.nodes():
        # Check if node is not in the values of the match dict and has degree 2
        if node not in matched.values() and H.degree(node) == 2:
            unmatched_deg2_nodes.append(node)

    logger.info(f"Unmatched degree-2 nodes in Pred: {len(unmatched_deg2_nodes)}")

    # 4) Remove degree-2 unmatched nodes by bypassing them
    for node in unmatched_deg2_nodes:
        neighbors = list(H.neighbors(node))

        if isinstance(H, nx.DiGraph):
            predecessors = list(H.predecessors(node))
            successors = list(H.successors(node))
            if len(predecessors) == 1 and len(successors) == 1:
                pred = predecessors[0]
                succ = successors[0]
                # Add edge from pred -> succ if it doesn't already exist
                if not H.has_edge(pred, succ):
                    H.add_edge(pred, succ)
                H.remove_node(node)
        else:
            # undirected graph case
            if len(neighbors) == 2:
                n1, n2 = neighbors
                if not H.has_edge(n1, n2):
                    H.add_edge(n1, n2)
                H.remove_node(node)

    # 5) Write to SWC
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(pred_path).replace(".swc", "")
    outfn = os.path.join(output_folder, f"{base_name}.swc")
    write_swc(H, outfn)
    logger.info(f"Wrote cleaned prediction SWC to: {outfn}")


def main():
    args = get_arguments()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.gt_fn is None or args.pred_fn is None or args.output_folder is None:
        raise ValueError("Please provide --gt_fn (GT folder), --pred_fn (pred folder), and --output_folder")

    gt_dir = args.gt_fn
    pred_dir = args.pred_fn
    out_dir = args.output_folder

    if not os.path.isdir(gt_dir):
        raise NotADirectoryError(f"--gt_fn must be a folder, got: {gt_dir}")
    if not os.path.isdir(pred_dir):
        raise NotADirectoryError(f"--pred_fn must be a folder, got: {pred_dir}")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.swc")))
    logging.info(f"Found {len(gt_files)} GT SWC files in {gt_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Pair by basename: <name>.swc in GT with <name>.swc in Pred
    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base)

        if not os.path.exists(pred_path):
            logger.warning(f"No predicted file for {base} in {pred_dir}, skipping.")
            continue

        try:
            process_pair(gt_path, pred_path, out_dir)
        except Exception as e:
            logger.error(f"Failed on {base}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
