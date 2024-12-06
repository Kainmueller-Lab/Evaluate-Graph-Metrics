import networkx as nx

from project.metrics import evaluate_all_metrics
from project.matching import match_graphs, MatchingType
from project.toydata import *

def main():

    # 1) Load graphs with attributes
    # NOTE: We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
    G, H = two_trees()
    assert nx.is_branching(G) and nx.is_branching(H), "Graphs must be branchings"

    # 2) Match graphs
    # Match target graph (e.g. prediction) to source graph (e.g. ground truth)
    # The matching variable is a dictionary that maps nodes in target graph to nodes in source graph.
    # NOTE: This is not symmetric, match(G,H) in general does not equal match(H,G).
    matching = match_graphs(source=G, target=H, matching_type=MatchingType.Greedy)

    # 3) Compute metrics wrt to source and matched target graph
    results_dict = evaluate_all_metrics(G, H, matching)

    # 4) Print results
    for name, value in results_dict.items():
        print(f"{name}: {value}")

if __name__ == "__main__":
    main()
