import inspect
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
# NOTE: Check out documentation of networkx for all available graph algorithms:
# https://networkx.org/documentation/stable/reference/algorithms/index.html

from project.utils import betti_1_number, compute_edge_metrics
from project.streetmover_distance import StreetMoverDistance

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
    # TODO: Include tree edit distance implementation based on attributes and specific for trees. This should be more efficient than nx.graph_edit_distance.
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

    
    @staticmethod
    def street_mover_distance(G, H, matching=None):
        '''
        Taken from https://github.com/davide-belli/generative-graph-transformer/blob/master/metrics/streetmover_distance.py
        '''
        # Adjancency matrices
        G_adjacency = nx.to_numpy_array(G) # TODO: Check if G needs to be converted to undirected first.
        H_adjacency = nx.to_numpy_array(H)
        # Nodes
        # NOTE: Nodes are given as a PyTorch tensor of shape (N, 3) containing N 3D coordinates where N is the number of nodes.
        G_coords = nx.get_node_attributes(G, 'coord')
        H_coords = nx.get_node_attributes(H, 'coord')
        G_nodes = torch.from_numpy(np.array(list(G_coords.values()), dtype=np.float32)) # TODO: Check if this is the correct conversion.
        H_nodes = torch.from_numpy(np.array(list(H_coords.values()), dtype=np.float32))

        smd = StreetMoverDistance(eps=0.001, max_iter=1000, reduction='mean') # TODO: Find best hyperparameters
        _, cost = smd.forward(G_adjacency, G_nodes, H_adjacency, H_nodes, n_points=100)
        return cost[0]
    

    

def evaluate_all_metrics(G, H, matching):
    """
    Automatically evaluates all metrics in the class on the given graphs.
    Returns a dictionary with metric names and their values.
    """
    return {name: method(G, H, matching) for name, method in inspect.getmembers(GraphMetrics, predicate=inspect.isfunction)}