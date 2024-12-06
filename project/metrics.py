import inspect
import networkx as nx
import numpy as np
# NOTE: Check out documentation of networkx for all available graph algorithms:
# https://networkx.org/documentation/stable/reference/algorithms/index.html

from project.utils import betti_1_number, compute_edge_metrics

class GraphMetrics:
    
    @staticmethod
    def betti_0_error(G, H, matching=None):
        '''
        Counts the number of connected components in the graph G and H and computes the absolute difference
        '''
        return np.abs(nx.number_connected_components(G.to_undirected()) - nx.number_connected_components(H.to_undirected()))
    
    @staticmethod
    def betti_1_error(G, H, matching=None):
        '''
        Counts the number of connected components in the graph G and H and computes the absolute difference
        '''
        return np.abs(betti_1_number(G.to_undirected()) - betti_1_number(H.to_undirected()))

    @staticmethod
    def precision(G, H, matching):
        metrics = compute_edge_metrics(G, H, matching)
        TP, FP = metrics["TP"], metrics["FP"]
        return TP / (TP + FP) if TP + FP > 0 else 0

    @staticmethod
    def recall(G, H, matching):
        metrics = compute_edge_metrics(G, H, matching)
        TP, FN = metrics["TP"], metrics["FN"]
        return TP / (TP + FN) if TP + FN > 0 else 0

    @staticmethod
    def f1_score(G, H, matching):
        metrics = compute_edge_metrics(G, H, matching)
        TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]
        return 2 * TP / (2 * TP + FN + FP) if 2 * TP + FN + FP > 0 else 0
        

    # NOTE: This might have long runtime.
    @staticmethod
    def graph_edit_distance(G, H, matching=None):
        '''
        Returns the graph edit distance between G and H. This is the generic method of networkx and does not use matching information.
        Long runtimes are expected for large graphs.
        '''
        if G.number_of_nodes() < 50: # ppply to small graphs only to avoid long runime.
            return nx.graph_edit_distance(G, H)
        else:
            return -1

    
    # TODO: Tree edit distance implementation based on attributes and specific for trees. This should be more efficient than nx.graph_edit_distance.
    # TODO: Street mover distance.
    

    

def evaluate_all_metrics(G, H, matching):
    """
    Automatically evaluates all metrics in the class on the given graphs.
    Returns a dictionary with metric names and their values.
    """
    return {name: method(G, H, matching) for name, method in inspect.getmembers(GraphMetrics, predicate=inspect.isfunction)}