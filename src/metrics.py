import inspect
import networkx as nx
import numpy as np
# NOTE: Check out documentation of networkx for all available graph algorithms:
# https://networkx.org/documentation/stable/reference/algorithms/index.html

class GraphMetrics:
    
    @staticmethod
    def betti_0_error(G, H, matching=None):
        return np.abs(nx.number_connected_components(G.to_undirected()) - nx.number_connected_components(H.to_undirected()))
    
    @staticmethod
    def graph_edit_distance(G, H, matching=None):
        return nx.graph_edit_distance(G, H)
    


graph_metrics_methods = [
    method for name, method in inspect.getmembers(GraphMetrics, predicate=inspect.isfunction)
]