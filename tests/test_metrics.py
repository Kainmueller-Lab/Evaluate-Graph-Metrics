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


from project.utils import compute_edge_metrics
from project.toydata import basic_MS_example, merge_example, split_example_1, split_example_2

class TestMergesAndSplits(unittest.TestCase):
    def test_basic(self):
        gt, pred = basic_MS_example()
        matching = match_graphs(source=gt, target=pred, matching_type=MatchingType.Nearest)
        metrics = compute_edge_metrics(gt, pred, matching, visualize=False)
        false_merges = metrics["FM"]
        false_splits = metrics["FS"]
        self.assertEqual(false_merges, 1)
        self.assertEqual(false_splits, 1)
        

    def test_false_merges(self):
        gt, pred = merge_example()
        matching = match_graphs(source=gt, target=pred, matching_type=MatchingType.Nearest)
        metrics = compute_edge_metrics(gt, pred, matching, visualize=False)
        false_merges = metrics["FM"]
        false_splits = metrics["FS"]
        num_FP = metrics["FP_edges"]
        num_FN = metrics["FN_edges"]
        self.assertEqual(num_FP, 6)
        self.assertEqual(false_merges, 3)
        self.assertEqual(num_FN, 0)
        self.assertEqual(false_splits, 0)

    def test_false_splits(self):
        gt, pred = split_example_1()
        matching = match_graphs(source=gt, target=pred, matching_type=MatchingType.Nearest)
        metrics = compute_edge_metrics(gt, pred, matching, visualize=False)
        false_merges = metrics["FM"]
        false_splits = metrics["FS"]
        num_FP = metrics["FP_edges"]
        num_FN = metrics["FN_edges"]
        self.assertEqual(num_FP, 2)
        self.assertEqual(false_merges, 1)
        self.assertEqual(num_FN, 2)
        self.assertEqual(false_splits, 1)

        gt, pred = split_example_2()
        matching = match_graphs(source=gt, target=pred, matching_type=MatchingType.Nearest)
        metrics = compute_edge_metrics(gt, pred, matching, visualize=False)
        false_merges = metrics["FM"]
        false_splits = metrics["FS"]
        num_FP = metrics["FP_edges"]
        num_FN = metrics["FN_edges"]
        self.assertEqual(num_FP, 6)
        self.assertEqual(false_merges, 5)
        self.assertEqual(num_FN, 5)
        self.assertEqual(false_splits, 4)


if __name__ == "__main__":
    unittest.main()
