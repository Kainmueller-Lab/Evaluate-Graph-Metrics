import networkx as nx

from project.metrics import GraphMetrics, graph_metrics_methods
from project.matching import match_graphs, MatchingType
from project.toydata import examples

def main():

    # 1) Load graphs with attributes
    # NOTE: We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
    G, H = examples()
    assert nx.is_branching(G) and nx.is_branching(H), "Graphs must be branching graphs"

    # 2) Match graphs
    # Match target graph (e.g. prediction) to source graph (e.g. ground truth)
    # matching is a dictionary that maps nodes in target graph to nodes in source graph.
    # NOTE: This is not symmetric, match(G,H) in general does not equal match(H,G).
    matching = match_graphs(source=G, target=H, matching_type=MatchingType.Nearest)
    print(matching)


    # 3) Compute metrics wrt to source and matched target graph
    for metric in graph_metrics_methods:
        res = metric(G, H, matching)
        print(f"{metric.__name__}: {res}")


if __name__ == "__main__":
    main()
