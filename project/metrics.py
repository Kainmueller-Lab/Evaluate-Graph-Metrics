import inspect
import networkx as nx
import numpy as np
# NOTE: Check out documentation of networkx for all available graph algorithms:
# https://networkx.org/documentation/stable/reference/algorithms/index.html

class GraphMetrics:
    
    @staticmethod
    def betti_0_error(G, H, matching=None):
        '''
        Counts the number of connected components in the graph G and H and computes the absolute difference
        '''
        return np.abs(nx.number_connected_components(G.to_undirected()) - nx.number_connected_components(H.to_undirected()))
    
    @staticmethod
    def graph_edit_distance(G, H, matching=None):
        '''
        Returns the graph edit distance between G and H. This is the generic method of networkx and does not use matching information.
        Long runtimes are expected for large graphs.
        '''
        print("Warning: Graph edit distance has long runtimes... Comment it out in this case.")
        return nx.graph_edit_distance(G, H)
    
    @staticmethod
    def false_edges(G, H, matching):
        G_edges = set(G.edges)
        H_edges = set([(matching[e[0]], matching[e[1]]) for e in H.edges])

        false_splits = G_edges - H_edges # E.g. edges present in the GT but not in the prediction.
        false_merges = H_edges - G_edges # E.g. edges present in the prediction but not in the GT.

        return len(false_splits), len(false_merges)
    
    # TODO: Accuracy / Kggle score or (FS + FM ) / Edges


        

    
    # TODO: Tree edit distance implementation.
    # TODO: Street mover distance.
    


graph_metrics_methods = [
    method for name, method in inspect.getmembers(GraphMetrics, predicate=inspect.isfunction)
]