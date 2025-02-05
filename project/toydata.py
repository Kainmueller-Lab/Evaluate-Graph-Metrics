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


def basic_MS_example():
    G = nx.DiGraph()
    nodes_G = [
        (1, {'coord': np.array([0, 4]), 'radius': 1}),
        (2, {'coord': np.array([0, 3]), 'radius': 1}),
        (3, {'coord': np.array([-1, 2]), 'radius': 1}),
        (4, {'coord': np.array([1, 2]), 'radius': 1}),
        (5, {'coord': np.array([0, 1]), 'radius': 1}),
        (6, {'coord': np.array([0, 0]), 'radius': 1})
    ]
    edges_G = [
        (1,2),
        (2,3),
        (2,4),
        (4,5),
        (5,6)
    ]
    G.add_nodes_from(nodes_G)
    G.add_edges_from(edges_G)

    H = nx.DiGraph()
    nodes_H = [
        (1, {'coord': np.array([0, 4]), 'radius': 1}),
        (2, {'coord': np.array([0, 3]), 'radius': 1}),
        (3, {'coord': np.array([-1, 2]), 'radius': 1}),
        (4, {'coord': np.array([1, 2]), 'radius': 1}),
        (5, {'coord': np.array([0, 1]), 'radius': 1}),
        (6, {'coord': np.array([0, 0]), 'radius': 1})
    ]
    edges_H = [
        (1,2),
        (2,3),
        (2,4),
        (3,5),
        (5,6)
    ]
    H.add_nodes_from(nodes_H)
    H.add_edges_from(edges_H)
    return G, H


def merge_example():
    G = nx.DiGraph()
    nodes_G = [ (idx, {'coord': np.array([0, idx]), 'radius': 1}) for idx in range(7)]
    edges_G = [
        (0,1),
        (0,2),
        (1,3),
        (1,4),
        (2,5),
        (2,6)
    ]
    G.add_nodes_from(nodes_G)
    G.add_edges_from(edges_G)

    H = nx.DiGraph()
    nodes_H = [ (idx, {'coord': np.array([0, idx]), 'radius': 1}) for idx in range(7)]
    edges_H = [
        # Same edges as G
        (0,1),
        (0,2),
        (1,3),
        (1,4),
        (2,5),
        (2,6),
        # Additional edges
        (0,4),
        (0,5),
        (0,6),
        (1,2),
        (1,5),
        (4,5)
    ]
    H.add_nodes_from(nodes_H)
    H.add_edges_from(edges_H)
    return G, H


def split_example_1():
    G = nx.DiGraph()
    nodes_G = [ (idx, {'coord': np.array([0, idx]), 'radius': 1}) for idx in range(7)]
    edges_G = [
        (0,1),
        (0,2),
        (1,3),
        (2,4),
        (3,5),
        (4,6)
    ]
    G.add_nodes_from(nodes_G)
    G.add_edges_from(edges_G)

    H = nx.DiGraph()
    nodes_H = [ (idx, {'coord': np.array([0, idx]), 'radius': 1}) for idx in range(7)]
    edges_H = [
        # Same edges as G, some removed
        (0,1),
        (0,2),
        #(1,3),
        #(2,4),
        (3,5),
        (4,6),
        # Additional edges
        (1,5),
        (1,6)
    ]
    H.add_nodes_from(nodes_H)
    H.add_edges_from(edges_H)
    return G, H


def split_example_2():
    G = nx.DiGraph()
    nodes_G = [ (idx, {'coord': np.array([0, idx]), 'radius': 1}) for idx in range(13)]
    edges_G = [
        (0,1),
        (0,2),
        (1,3),
        (1,4),
        (2,5),
        (2,6),
        (3,7),
        (3,8),
        (4,9),
        (5,10),
        (6,11),
        (6,12),
    ]
    G.add_nodes_from(nodes_G)
    G.add_edges_from(edges_G)

    H = nx.DiGraph()
    nodes_H = [ (idx, {'coord': np.array([0, idx]), 'radius': 1}) for idx in range(13)]
    edges_H = [
        # Same edges as G, some removed
        (0,1),
        (0,2),
        #(1,3),
        #(1,4),
        #(2,5),
        (2,6),
        (3,7),
        (3,8),
        #(4,9),
        (5,10),
        (6,11),
        #(6,12),
        # Additional edges
        (0,8),
        (7,8),
        (8,9),
        (9,10),
        (10,11),
        (11,12),
    ]
    H.add_nodes_from(nodes_H)
    H.add_edges_from(edges_H)
    return G, H


import networkx as nx
import random

def generate_random_tree(n):
    """
    Generate a random tree with n nodes.
    
    The tree is constructed by adding nodes one by one and connecting each new node
    to a random existing node (randomly growing a spanning tree).
    """
    if n < 1:
        raise ValueError("Number of nodes must be at least 1")

    G = nx.DiGraph()
    G.add_node(0)  # Start with a single root node

    for i in range(1, n):
        parent = random.choice(list(G.nodes))  # Pick a random existing node
        G.add_edge(parent, i)  # Connect new node to the chosen parent

    return G


def random_graphs(num_nodes=100):
    coordinates = [10*np.random.randn(3) for i in range(num_nodes)]
    noisy_coords = [coord + 0.2* np.random.randn(3) for coord in coordinates]

    G = generate_random_tree(num_nodes)
    H = generate_random_tree(num_nodes)

    G_node_attrs = {
        node_idx: {
            'coord': coordinates[i],
            'radius': 2 
        }
        for node_idx, i in enumerate(G.nodes)
    }

    H_node_attrs = {
        node_idx: {
            'coord': noisy_coords[i],
            'radius': 2 
        }
        for node_idx, i in enumerate(H.nodes)
    }

    nx.set_node_attributes(G, G_node_attrs)
    nx.set_node_attributes(H, H_node_attrs)

    return G, H

