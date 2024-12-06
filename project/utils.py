import networkx as nx

def betti_1_number(G):
    """
    Computes the Betti 1 number (number of independent cycles) of a graph.
    
    Parameters:
    ----------
    G : networkx.Graph
        The input graph (undirected or directed).
    
    Returns:
    -------
    int
        The Betti 1 number of the graph.
    """
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    num_components = nx.number_connected_components(G)
    
    # Betti 1 number formula
    return num_edges - num_nodes + num_components

def compute_edge_metrics(G, H, matching):
    """
    Computes common edges (TP), false splits (FN), and false merges (FP).
    Returns a dictionary with TP, FN, and FP values.
    """
    G_edges = set(G.edges)
    H_edges = set([(matching[e[0]], matching[e[1]]) for e in H.edges])

    false_splits = G_edges - H_edges  # False negatives
    false_merges = H_edges - G_edges  # False positives
    common_edges = G_edges & H_edges  # True positives

    metrics = {
        "TP": len(common_edges),
        "FN": len(false_splits),
        "FP": len(false_merges),
    }
    return metrics