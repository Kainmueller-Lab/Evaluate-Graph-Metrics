import inspect
import logging
import time
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
# NOTE: Check out documentation of networkx for all available graph algorithms:
# https://networkx.org/documentation/stable/reference/algorithms/index.html

from project.utils import betti_0_number, betti_1_number, \
        compute_edge_metrics, compute_node_metrics
from project.streetmover_distance import StreetMoverDistance


logger = logging.getLogger(__name__)


class GraphMetrics:

    @staticmethod
    def betti_0_error(G, H):
        '''
        Counts the number of connected components in the graph G and H
        and computes the absolute difference.
        '''
        return {"betti_0": betti_0_number(G, H)}

    @staticmethod
    def betti_1_error(G, H):
        '''
        Computes the Betti 1 number (number of independent cycles) of a graph.
        '''
        return {"betti_1": np.abs(
            betti_1_number(G.to_undirected()) - betti_1_number(H.to_undirected()))}

    @staticmethod
    def f1_score_edgewise(G, H, matched, visualize=True):
        return compute_edge_metrics(G, H, matched, visualize=visualize)

    @staticmethod
    def f1_score_nodewise(G, H, matched):
        return compute_node_metrics(G, H, matched)

    # NOTE: This might have long runtime.
    # TODO: Include tree edit distance implementation based on attributes and specific for trees. This should be more efficient than nx.graph_edit_distance.
    @staticmethod
    def graph_edit_distance(G, H, matched):
        '''
        Returns the graph edit distance between G and H. This is the generic method of networkx and does not use matching information.
        Long runtimes are expected for large graphs.
        '''
        if G.number_of_nodes() < 50: # ppply to small graphs only to avoid long runime.
            return {"ted": nx.graph_edit_distance(G, H)}
        else:
            return {"ted": -1}

    @staticmethod
    def street_mover_distance_VF(G, H, normalize=True):
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

        if normalize:
            G_nodes_tensor = torch.stack(G_nodes)
            H_nodes_tensor = torch.stack(H_nodes)

            def min_max(tensor_A, tensor_B):
                # Compute element-wise minimum and maximum across both tensors
                min_coords = torch.min(tensor_A, tensor_B)  # Nx3 minimum coordinates
                max_coords = torch.max(tensor_A, tensor_B)  # Nx3 maximum coordinates
                bbox_min = min_coords.min(dim=0).values
                bbox_max = max_coords.max(dim=0).values
                return bbox_min, bbox_max

            def normalize(tensor, min, max):
                return (tensor - min) / (max - min)
            min, max = min_max(G_nodes_tensor, H_nodes_tensor)

            G_nodes_normalized = normalize(G_nodes_tensor, min, max)
            H_nodes_normalized = normalize(H_nodes_tensor, min, max)
            G_nodes = [coord for coord in G_nodes_normalized]
            H_nodes = [coord for coord in H_nodes_normalized]

            # print(f"Normalized G_nodes range: {G_nodes_normalized.min(dim=0).values}, {G_nodes_normalized.max(dim=0).values}")
            # print(f"Normalized H_nodes range: {H_nodes_normalized.min(dim=0).values}, {H_nodes_normalized.max(dim=0).values}")

        smd = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=None, use_correct_sinkhorn_dist=False)

        (y_pc, output_pc), cost = smd.forward(G_adjacency, G_nodes, H_adjacency, H_nodes, n_points=2000)

        print(f"Number of points in Graph G: {len(y_pc)}")
        print(f"Number of points in Graph H: {len(output_pc)}")
        print("Point Cloud G Shape:", y_pc.shape)
        print("Point Cloud H Shape:", output_pc.shape)
        print("Any NaN in G:", torch.isnan(y_pc).any())
        print("Any NaN in H:", torch.isnan(output_pc).any())
        print("Any Inf in G:", torch.isinf(y_pc).any())
        print("Any Inf in H:", torch.isinf(output_pc).any())

        return {"smd_VF":cost[0]}
    


    @staticmethod
    def street_mover_distance_POT(G, H, normalize=True):
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

        if normalize:
            G_nodes_tensor = torch.stack(G_nodes)
            H_nodes_tensor = torch.stack(H_nodes)

            def min_max(tensor_A, tensor_B):
                # Compute element-wise minimum and maximum across both tensors
                min_coords = torch.min(tensor_A, tensor_B)  # Nx3 minimum coordinates
                max_coords = torch.max(tensor_A, tensor_B)  # Nx3 maximum coordinates
                bbox_min = min_coords.min(dim=0).values
                bbox_max = max_coords.max(dim=0).values
                return bbox_min, bbox_max

            def normalize(tensor, min, max):
                return (tensor - min) / (max - min)
            min, max = min_max(G_nodes_tensor, H_nodes_tensor)

            G_nodes_normalized = normalize(G_nodes_tensor, min, max)
            H_nodes_normalized = normalize(H_nodes_tensor, min, max)
            G_nodes = [coord for coord in G_nodes_normalized]
            H_nodes = [coord for coord in H_nodes_normalized]

            # print(f"Normalized G_nodes range: {G_nodes_normalized.min(dim=0).values}, {G_nodes_normalized.max(dim=0).values}")
            # print(f"Normalized H_nodes range: {H_nodes_normalized.min(dim=0).values}, {H_nodes_normalized.max(dim=0).values}")

        smd = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=None, use_correct_sinkhorn_dist=True)

        (y_pc, output_pc), cost = smd.forward(G_adjacency, G_nodes, H_adjacency, H_nodes, n_points=2000)

        print(f"Number of points in Graph G: {len(y_pc)}")
        print(f"Number of points in Graph H: {len(output_pc)}")
        print("Point Cloud G Shape:", y_pc.shape)
        print("Point Cloud H Shape:", output_pc.shape)
        print("Any NaN in G:", torch.isnan(y_pc).any())
        print("Any NaN in H:", torch.isnan(output_pc).any())
        print("Any Inf in G:", torch.isinf(y_pc).any())
        print("Any Inf in H:", torch.isinf(output_pc).any())

        return {"smd_POT":cost[0]}


def evaluate_all_metrics(G, H, matched, smd=False, visualize=False,
                         G_resampled=None, H_resampled=None):
    """
    Automatically evaluates all metrics in the class on the given graphs.
    Returns a dictionary with metric names and their values.
    """
    logger.info("start metrics computation")
    start_time = time.time()
    metrics_dict = {}

    for name, method in inspect.getmembers(GraphMetrics, predicate=inspect.isfunction):
        if smd == True and name.startswith("street_mover_distance"):
            metrics_dict.update(method(G, H))
        elif smd == False and name.startswith("street_mover_distance"):
            continue
        elif visualize == True and name == "F1_score_edgewise":
            metrics_dict.update(method(G, H, matched, visualize=visualize))
        elif "betti" in name:
            metrics_dict.update(method(G, H))
        else:
            metrics_dict.update(method(G, H, matched))

    logger.info("time for metric computation: %f sec." % (time.time() - start_time))
    return metrics_dict
