import networkx as nx

from src.metrics import GraphMetrics, graph_metrics_methods
from src.matching import match_graphs

def main():

    # 1) Load graphs with attributes
    G = nx.DiGraph()
    edges_G = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (2, 4),
        (4, 5)
    ]
    G.add_edges_from(edges_G)
    H = nx.DiGraph()
    edges_H = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (2, 4),
        #(4, 5)
    ]
    H.add_edges_from(edges_H)

    # 2) Match graphs
    # Match target graph (e.g. prediction) to source graph (e.g. ground truth)
    # NOTE: Is this symmetric? 
    matching = match_graphs(source=G, target=H, matching_type=None)

    # 3) Compute metrics wrt to source and matched target graph
    for metric in graph_metrics_methods:
        res = metric(G, H, matching)
        print(f"{metric.__name__}: {res}")


if __name__ == "__main__":
    main()
