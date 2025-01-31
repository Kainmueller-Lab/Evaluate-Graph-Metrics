import networkx as nx
import argparse

from project.metrics import evaluate_all_metrics
from project.matching import match_graphs, MatchingType
from project.utils import read_graph_from_file
from project.toydata import *
import glob
import os


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
        "--matching", type=str, default="hierarchical",
        choices=["hierarchical", "nearest", "nearest_with_radius", "greedy",
                 "greedy_with_parent"], #TODO: check if we can unify here
        help="choose matching method to assign pred to gt nodes"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    # 1) Load graphs with attributes
    # NOTE: We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
    if args.gt_fn is not None and args.pred_fn is not None:
        G = read_graph_from_file(args.gt_fn)
        print("number of nodes in G:", G.number_of_nodes())
        H = read_graph_from_file(args.pred_fn)
        print("number of nodes in H:", H.number_of_nodes())
    else:
        G, H = two_trees()
    #assert nx.is_branching(G) and nx.is_branching(H), "Graphs must be branchings"

    # 2) Match graphs
    # Match target graph (e.g. prediction) to source graph (e.g. ground truth)
    # The matching variable is a dictionary that maps nodes in target graph to nodes in source graph.
    # NOTE: This is not symmetric, match(G,H) in general does not equal match(H,G).
    if args.matching == "hierarchical":
        matching_type = MatchingType.Hierarchical
    elif args.matching == "nearest":
        matching_type = MatchingType.Nearest
    else:
        raise NotImplementedError
    matched, FN, FP = match_graphs(source=G, target=H, matching_type=matching_type)

    # 3) Compute metrics wrt to source and matched target graph
    results_dict = evaluate_all_metrics(G, H, matched, FN, FP)

    # 4) Print results
    for name, value in results_dict.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
