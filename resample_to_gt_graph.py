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


logger = logging.getLogger(__name__)


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_fn", type=str, dest="gt_fn", default=None,
        help="path to ground truth folder"
    )
    parser.add_argument(
        "--pred_fn", type=str, dest="pred_fn", default=None,
        help="path to predicted folder"
    )

    parser.add_argument(
        "--output_folder", type=str, default=None,
        help="path to output folder"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="flag to set logging to DEBUG"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # 1) Load graphs with attributes
    # NOTE: We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
    if args.gt_fn is not None and args.pred_fn is not None:
        G = read_graph_from_file(args.gt_fn)
        print("number of nodes in G:", G.number_of_nodes())
        H = read_graph_from_file(args.pred_fn)
        print("number of nodes in H:", H.number_of_nodes())
    else:
        G, H = two_trees()
        G = read_graph_from_file("tests/toygraph_GT.swc")
        H = read_graph_from_file("tests/toygraph_predicted.swc")


    matching_type = MatchingType.Hierarchical
    matched = match_graphs(
        source=G,
        target=H,
        matching_type=matching_type,
        matching_dist="fixed",
        max_distance=10,
        visualize=False)
    print(len(matched.keys()), G.number_of_nodes())

    # find all not matched pred nodes with degree 2
    unmatched_deg2_nodes = []
    for node in H.nodes():
        # Check if node is not in the values of the match dict and has degree 2
        if node not in matched.values() and H.degree(node) == 2:
            unmatched_deg2_nodes.append(node)

    print(len(unmatched_deg2_nodes))

    for node in unmatched_deg2_nodes:
        neighbors = list(H.neighbors(node))

        # For undirected graph, also check predecessors if needed
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
            if len(neighbors) == 2:
                n1, n2 = neighbors
                if not H.has_edge(n1, n2):
                    H.add_edge(n1, n2)
                H.remove_node(node)

    # write to swc
    outfn = os.path.join(args.output_folder, os.path.basename(args.pred_fn).replace(".swc","_gt_res.swc"))
    write_swc(H, outfn)


if __name__ == "__main__":
    main()
