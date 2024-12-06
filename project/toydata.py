import networkx as nx
import numpy as np

def examples():
    G = nx.DiGraph()
    nodes_G = [
        (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
        (2, {'coord': np.array([1, 0, 0]), 'radius': 1}),
        (3, {'coord': np.array([0, 1, 0]), 'radius': 1}),
        (4, {'coord': np.array([0, 0, 1]), 'radius': 1}),
        (5, {'coord': np.array([1, 1, 0]), 'radius': 1}),
        (6, {'coord': np.array([0, 1, 1]), 'radius': 1}),
    ]
    edges_G = [
        (1,2),
        (2,3),
        (2,4),
        (3,5),
        (3,6),
    ]
    G.add_nodes_from(nodes_G)
    G.add_edges_from(edges_G)

    H = nx.DiGraph()
    nodes_H = [
        (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
        (2, {'coord': np.array([1, 0, 0]), 'radius': 1}),
        (3, {'coord': np.array([0, 1, 0]), 'radius': 1}),
        (4, {'coord': np.array([0, 0, 1]), 'radius': 1}),
        (5, {'coord': np.array([1, 1, 0]), 'radius': 1}),
        (6, {'coord': np.array([0, 1, 1]), 'radius': 1}),
    ]
    edges_H = [
        (1,2),
        #(2,3),
        (2,4),
        (3,5),
        (3,6),
    ]
    H.add_nodes_from(nodes_H)
    H.add_edges_from(edges_H)

    return G, H

def two_trees():
    G = nx.DiGraph()
    nodes_G = [
        (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
        (2, {'coord': np.array([1, 0, 0]), 'radius': 1}),
        (3, {'coord': np.array([0, 1, 0]), 'radius': 1}),
        (4, {'coord': np.array([0, 0, 1]), 'radius': 1}),
        (5, {'coord': np.array([1, 1, 0]), 'radius': 1}),
        (6, {'coord': np.array([0, 1, 1]), 'radius': 1}),
    ]
    edges_G = [(2,1),
               (2,3),
               (5,4),
               (5,6)]
    G.add_nodes_from(nodes_G)
    G.add_edges_from(edges_G)

    H = nx.DiGraph()
    nodes_H = [
        (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
        (2, {'coord': np.array([1, 0, 0]), 'radius': 1}),
        (3, {'coord': np.array([0, 1, 0]), 'radius': 1}),
        (4, {'coord': np.array([0, 0, 1]), 'radius': 1}),
        (5, {'coord': np.array([1, 1, 0]), 'radius': 1}),
        (6, {'coord': np.array([0, 1, 1]), 'radius': 1}),
    ]
    edges_H = [(2,1), (4,3), (5,6)]
    H.add_nodes_from(nodes_H)
    H.add_edges_from(edges_H)

    return G, H

def smd_graph():
    G = nx.DiGraph()
    nodes_G = [
        (1, {'coord': np.array([0, 0]), 'radius': 1}),
        (2, {'coord': np.array([1, 0]), 'radius': 1}),
        (3, {'coord': np.array([2, 0]), 'radius': 1}),
        (4, {'coord': np.array([3, 0]), 'radius': 1}),
        (5, {'coord': np.array([4, 0]), 'radius': 1}),
    ]
    edges_G = [
        (1,2),
        (2,3),
        (3,4),
        (4,5),
    ]
    G.add_nodes_from(nodes_G)
    G.add_edges_from(edges_G)

    H = nx.DiGraph()
    nodes_H = [
        (1, {'coord': np.array([0, 1]), 'radius': 1}),
        (2, {'coord': np.array([1, 1]), 'radius': 1}),
        (3, {'coord': np.array([2, 1]), 'radius': 1}),
        (4, {'coord': np.array([3, 1]), 'radius': 1}),
        (5, {'coord': np.array([4, 1]), 'radius': 1}),
    ]
    edges_H = [
        (1,2),
        (2,3),
        (3,4),
        (4,5),
    ]
    H.add_nodes_from(nodes_H)
    H.add_edges_from(edges_H)

    return G, H