import unittest
import networkx as nx
import numpy as np

from project.matching import match_graphs, MatchingType

class TestMatching(unittest.TestCase):
    def setUp(self):
        # NOTE:  We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
        # A branching is a directed forest with each node having, at most, one parent. So the maximum in-degree is equal to 1.
        # Nodes without parents are called roots. Nodes without children are called leaves.
        self.G = nx.DiGraph()
        nodes_G = [
            (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0, 0]), 'radius': 1}),
            (3, {'coord': np.array([2.5, 0, 0]), 'radius': 1}),
            (4, {'coord': np.array([4, 0, 0]), 'radius': 1}),
            (5, {'coord': np.array([-10, 0, 0]), 'radius': 1}),
        ]
        edges_G = []
        self.G.add_nodes_from(nodes_G)
        self.G.add_edges_from(edges_G)

        self.H = nx.DiGraph()
        nodes_H = [
            (1, {'coord': np.array([0.1, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2.1, 0, 0]), 'radius': 1}),
            (3, {'coord': np.array([3.9, 0, 0]), 'radius': 1}),
            (4, {'coord': np.array([10, 0, 0]), 'radius': 1}),
        ]
        edges_H = []
        self.H.add_nodes_from(nodes_H)
        self.H.add_edges_from(edges_H)

    def test_nearest(self):
        result = match_graphs(self.G, self.H, MatchingType.Nearest)
        self.assertDictEqual(result, {1: 1, 2: 2, 3: 2, 4: 3, 5: 1})

    def test_nearest_within_radius(self):
        result = match_graphs(self.G, self.H, MatchingType.Nearest_Within_Radius)
        self.assertDictEqual(result, {1: 1, 2: 2, 3: 2, 4: 3})

    def test_greedy(self):
        result = match_graphs(self.G, self.H, MatchingType.Greedy)
        self.assertDictEqual(result, {1: 1, 2: 2, 4: 3})


if __name__ == "__main__":
    unittest.main()
