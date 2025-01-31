import inspect
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
# NOTE: Check out documentation of networkx for all available graph algorithms:
# https://networkx.org/documentation/stable/reference/algorithms/index.html

from project.utils import betti_0_number, betti_1_number, compute_edge_metrics, compute_node_metrics
from project.streetmover_distance import StreetMoverDistance


class GraphMetrics:
    
    @staticmethod
    def betti_0_error(G, H, matched, FN, FP):
        '''
        Counts the number of connected components in the graph G and H and computes the absolute difference
        '''
        return betti_0_number(G, H)

    @staticmethod
    def betti_1_error(G, H, matched, FN, FP):
        '''
        Counts the number of connected components in the graph G and H and computes the absolute difference
        '''
        return np.abs(betti_1_number(G.to_undirected()) - betti_1_number(H.to_undirected()))

    @staticmethod
    def precision_edgewise(G, H, matched, FN, FP):
        metrics = compute_edge_metrics(G, H, matched, FN, FP, False)
        TP, FP, FM, FS, FN = metrics["TP"], metrics["FP"], metrics["FM"], metrics["FS"], metrics["FN"]
        print("false_merges:", FM)
        print("false_splits:", FS)
        print("false_negatives:", FN)
        return TP / (TP + FP) if TP + FP > 0 else 0

    @staticmethod
    def recall_edgewise(G, H, matched, FN, FP):
        metrics = compute_edge_metrics(G, H, matched, FN, FP, False)
        TP, FN = metrics["TP"], metrics["FN"]
        return TP / (TP + FN) if TP + FN > 0 else 0

    @staticmethod
    def f1_score_edgewise(G, H, matched, FN, FP):
        metrics = compute_edge_metrics(G, H, matched, FN, FP, True)
        TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]
        return 2 * TP / (2 * TP + FN + FP) if 2 * TP + FN + FP > 0 else 0

    @staticmethod
    def precision_nodewise(G, H, matched, FN, FP):
        metrics = compute_node_metrics(G, H, matched, FN, FP)
        TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]

        print("false_negatives_nodewise:", FN)
        print("false_positives_nodewise:", FP)
        return TP / (TP + FP) if TP + FP > 0 else 0

    @staticmethod
    def recall_nodewise(G, H, matched, FN, FP):
        metrics = compute_node_metrics(G, H, matched, FN, FP)
        TP, FN = metrics["TP"], metrics["FN"]
        return TP / (TP + FN) if TP + FN > 0 else 0

    @staticmethod
    def f1_score_nodewise(G, H, matched, FN, FP):
        metrics = compute_node_metrics(G, H, matched, FN, FP)
        TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]
        return 2 * TP / (2 * TP + FN + FP) if 2 * TP + FN + FP > 0 else 0




    # NOTE: This might have long runtime.
    # TODO: Include tree edit distance implementation based on attributes and specific for trees. This should be more efficient than nx.graph_edit_distance.
    @staticmethod
    def graph_edit_distance(G, H, matched, FN, FP):
        '''
        Returns the graph edit distance between G and H. This is the generic method of networkx and does not use matching information.
        Long runtimes are expected for large graphs.
        '''
        if G.number_of_nodes() < 50: # ppply to small graphs only to avoid long runime.
            return nx.graph_edit_distance(G, H)
        else:
            return -1

    
    @staticmethod
    def street_mover_distance(G, H, matched, FN, FP):
        '''
        Taken from https://github.com/davide-belli/generative-graph-transformer/blob/master/metrics/streetmover_distance.py
        '''
        G = G.to_undirected()
        H = H.to_undirected()

        G_adjacency = nx.to_numpy_array(G)
        H_adjacency = nx.to_numpy_array(H)

        G_coords = nx.get_node_attributes(G, 'coord')
        H_coords = nx.get_node_attributes(H, 'coord')

        G_nodes = [torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in G_coords.values()]
        H_nodes = [torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in H_coords.values()]

        G_nodes_tensor = torch.stack(G_nodes)
        H_nodes_tensor = torch.stack(H_nodes)

        G_nodes_normalized = (G_nodes_tensor - G_nodes_tensor.min(dim=0).values) / (
                    G_nodes_tensor.max(dim=0).values - G_nodes_tensor.min(dim=0).values)
        H_nodes_normalized = (H_nodes_tensor - H_nodes_tensor.min(dim=0).values) / (
                    H_nodes_tensor.max(dim=0).values - H_nodes_tensor.min(dim=0).values)

        G_nodes = [coord for coord in G_nodes_normalized]
        H_nodes = [coord for coord in H_nodes_normalized]

        # print(f"Normalized G_nodes range: {G_nodes_normalized.min(dim=0).values}, {G_nodes_normalized.max(dim=0).values}")
        # print(f"Normalized H_nodes range: {H_nodes_normalized.min(dim=0).values}, {H_nodes_normalized.max(dim=0).values}")

        smd = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=None)

        (y_pc, output_pc), cost = smd.forward(G_adjacency, G_nodes, H_adjacency, H_nodes, n_points=100)

        print(f"Number of points in Graph G: {len(y_pc)}")
        print(f"Number of points in Graph H: {len(output_pc)}")
        print("Point Cloud G Shape:", y_pc.shape)
        print("Point Cloud H Shape:", output_pc.shape)
        print("Any NaN in G:", torch.isnan(y_pc).any())
        print("Any NaN in H:", torch.isnan(output_pc).any())
        print("Any Inf in G:", torch.isinf(y_pc).any())
        print("Any Inf in H:", torch.isinf(output_pc).any())

        return cost[0]


def evaluate_all_metrics(G, H, matched, FN, FP):
    """
    Automatically evaluates all metrics in the class on the given graphs.
    Returns a dictionary with metric names and their values.
    """
    return {name: method(G, H, matched, FN, FP) for name, method in inspect.getmembers(GraphMetrics, predicate=inspect.isfunction)}