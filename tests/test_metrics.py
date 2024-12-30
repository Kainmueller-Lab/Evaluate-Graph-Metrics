import unittest
import networkx as nx
import numpy as np

from project.metrics import GraphMetrics
from project.matching import match_graphs, MatchingType
from project.toydata import two_trees, smd_graph

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # NOTE:  We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
        # A branching is a directed forest with each node having, at most, one parent. So the maximum in-degree is equal to 1.
        # Nodes without parents are called roots. Nodes without children are called leaves.
        self.G, self.H = two_trees() 
        self.matching = match_graphs(source=self.G, target=self.H, matching_type=MatchingType.Nearest)


    def test_is_branching(self):
        result = nx.is_branching(self.G)
        self.assertEqual(result, True)

    def test_betti_0_error(self):
        result = GraphMetrics.betti_0_error(self.G, self.H)
        self.assertEqual(result, 1)

    def test_betti_1_error(self):
        result = GraphMetrics.betti_1_error(self.G, self.H)
        self.assertEqual(result, 0)

    def test_graph_edit_distance(self):
        result = GraphMetrics.graph_edit_distance(self.G, self.H)
        self.assertEqual(result, 3.0)

    def test_precision(self):
        result = GraphMetrics.precision(self.G, self.H, self.matching)
        self.assertEqual(result, 2/3)

    def test_recall(self):
        result = GraphMetrics.recall(self.G, self.H, self.matching)
        self.assertEqual(result, 2/4)

    def test_f1_score(self):
        result = GraphMetrics.f1_score(self.G, self.H, self.matching)
        self.assertEqual(result, 2 / (3/2 + 4/2))

    def test_trivial_smd(self):
        result = GraphMetrics.street_mover_distance(self.G, self.G)
        self.assertAlmostEqual(result, 0.0, delta=0.001)

    def test_translated_smd(self):
        G, H = smd_graph()
        result = GraphMetrics.street_mover_distance(G, H)
        self.assertAlmostEqual(result, 1.0, delta=0.001)


if __name__ == "__main__":
    unittest.main()
