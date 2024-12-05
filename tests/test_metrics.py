import unittest
import networkx as nx

from src.metrics import GraphMetrics

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.G = nx.DiGraph()
        edges_G = [
            (1,2),
            (2,3),
            (3,4),
            (1,5),
            (2,5),
            (3,6),
            (4,6)
        ]
        self.G.add_edges_from(edges_G)
        self.H = nx.DiGraph()
        edges_H = [
            (1,2),
            #(2,3),
            (3,4),
            (1,5),
            (2,5),
            (3,6),
            (4,6)
        ]
        self.H.add_edges_from(edges_H)

    def test_betti_0_error(self):
        result = GraphMetrics.betti_0_error(self.G, self.H)
        self.assertEqual(result, 1)

    def test_graph_edit_distance(self):
        result = GraphMetrics.graph_edit_distance(self.G, self.H)
        self.assertEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
