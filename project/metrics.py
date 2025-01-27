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
    
    #@staticmethod
    # def betti_0_error(G, H, matching=None):
    #     '''
    #     Counts the number of connected components in the graph G and H and computes the absolute difference
    #     '''
    #     return np.abs(nx.number_connected_components(G.to_undirected()) - nx.number_connected_components(H.to_undirected()))
    #
    # @staticmethod
    # def betti_1_error(G, H, matching=None):
    #     '''
    #     Counts the number of connected components in the graph G and H and computes the absolute difference
    #     '''
    #     return np.abs(betti_1_number(G.to_undirected()) - betti_1_number(H.to_undirected()))
    #
    # @staticmethod
    # def precision(G, H, matching):
    #     metrics = compute_edge_metrics(G, H, matching)
    #     TP, FP, FM, FS, FN = metrics["TP"], metrics["FP"], metrics["FM"], metrics["FS"], metrics["FN"]
    #     print("false_merges:", FM)
    #     print("false_splits:", FS)
    #     print("false_negatives:", FN)
    #     return TP / (TP + FP) if TP + FP > 0 else 0
    #
    # @staticmethod
    # def recall(G, H, matching):
    #     metrics = compute_edge_metrics(G, H, matching)
    #     TP, FN = metrics["TP"], metrics["FN"]
    #     return TP / (TP + FN) if TP + FN > 0 else 0
    #
    # @staticmethod
    # def f1_score(G, H, matching):
    #     metrics = compute_edge_metrics(G, H, matching)
    #     TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]
    #     return 2 * TP / (2 * TP + FN + FP) if 2 * TP + FN + FP > 0 else 0
    #
    #
    # # NOTE: This might have long runtime.
    # # TODO: Include tree edit distance implementation based on attributes and specific for trees. This should be more efficient than nx.graph_edit_distance.
    # @staticmethod
    # def graph_edit_distance(G, H, matching=None):
    #     '''
    #     Returns the graph edit distance between G and H. This is the generic method of networkx and does not use matching information.
    #     Long runtimes are expected for large graphs.
    #     '''
    #     if G.number_of_nodes() < 50: # ppply to small graphs only to avoid long runime.
    #         return nx.graph_edit_distance(G, H)
    #     else:
    #         return -1

    
    @staticmethod
    def street_mover_distance(G, H):
        '''
        Taken from https://github.com/davide-belli/generative-graph-transformer/blob/master/metrics/streetmover_distance.py
        '''
        G = G.to_undirected()
        H = H.to_undirected()
        # Adjancency matrices
        G_adjacency = nx.to_numpy_array(G)
        H_adjacency = nx.to_numpy_array(H)
        # Nodes
        # NOTE: Nodes are given as a PyTorch tensor of shape (N, 3) containing N 3D coordinates where N is the number of nodes.
        G_coords = nx.get_node_attributes(G, 'coord')
        H_coords = nx.get_node_attributes(H, 'coord')
        G_nodes = [torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in G_coords.values()]
        H_nodes = [torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in H_coords.values()]

        smd = StreetMoverDistance(eps=0.001, max_iter=400, reduction=None) # TODO: Find best hyperparameters
        (y_pc, output_pc), cost = smd.forward(G_adjacency, G_nodes, H_adjacency, H_nodes, n_points=70000)

        # Print the number of points in each point cloud
        print(f"Number of points in Graph G: {len(y_pc)}")
        print(f"Number of points in Graph H: {len(output_pc)}")
        # print("Graph G Point Cloud:", y_pc)
        # print("Graph H Point Cloud:", output_pc)
        print("Point Cloud G Shape:", y_pc.shape)
        print("Point Cloud H Shape:", output_pc.shape)
        print("Any NaN in G:", torch.isnan(y_pc).any())
        print("Any NaN in H:", torch.isnan(output_pc).any())
        print("Any Inf in G:", torch.isinf(y_pc).any())
        print("Any Inf in H:", torch.isinf(output_pc).any())

        return cost[0]
    

    

def evaluate_all_metrics(G, H):
    """
    Automatically evaluates all metrics in the class on the given graphs.
    Returns a dictionary with metric names and their values.
    """
    return {name: method(G, H) for name, method in inspect.getmembers(GraphMetrics, predicate=inspect.isfunction)}