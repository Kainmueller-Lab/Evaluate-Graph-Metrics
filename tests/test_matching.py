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
        


# class TestOneToOne(unittest.TestCase):
#     def test_basic(self):
#         self.G = nx.DiGraph()
#         nodes_G = [
#             (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
#             (2, {'coord': np.array([1, 0, 0]), 'radius': 1}),
#             (3, {'coord': np.array([0, 1, 0]), 'radius': 1}),
#         ]
#         edges_G = []
#         self.G.add_nodes_from(nodes_G)
#         self.G.add_edges_from(edges_G)
#
#         self.H = nx.DiGraph()
#         nodes_H = [
#             (1, {'coord': np.array([0.5, 0.5, 0]), 'radius': 1}),
#             (2, {'coord': np.array([0.51, 0.51, 0.1]), 'radius': 1}),
#         ]
#         edges_H = []
#         self.H.add_nodes_from(nodes_H)
#         self.H.add_edges_from(edges_H)
#
#         result = match_graphs(self.G, self.H, MatchingType.One_to_One)
#         self.assertDictEqual(result, {1: 1, 2: 2})
#
#     def test_basic_2(self):
#         N = 6
#         self.G = nx.DiGraph()
#         nodes_G = [(i, {'coord': np.array([i, 0]), 'radius': 1}) for i in range(N)]
#         edges_G = []
#         self.G.add_nodes_from(nodes_G)
#         self.G.add_edges_from(edges_G)
#
#         self.H = nx.DiGraph()
#         nodes_H = [(i, {'coord': np.array([i+0.5, 0]), 'radius': 1}) for i in range(N-1)]
#         edges_H = []
#         self.H.add_nodes_from(nodes_H)
#         self.H.add_edges_from(edges_H)
#
#         result = match_graphs(self.G, self.H, MatchingType.One_to_One)
#         self.assertDictEqual(result, {i : i for i in range(N-1)})
#
#     # TODO: adapt test to radius threshold
    # def test_out_of_reach(self):
    #     self.G = nx.DiGraph()
    #     nodes_G = [
    #         (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
    #     ]
    #     edges_G = []
    #     self.G.add_nodes_from(nodes_G)
    #     self.G.add_edges_from(edges_G)

    #     self.H = nx.DiGraph()
    #     nodes_H = [
    #         (1, {'coord': np.array([2, 0, 0]), 'radius': 1}),
    #     ]
    #     edges_H = []
    #     self.H.add_nodes_from(nodes_H)
    #     self.H.add_edges_from(edges_H)

    #     result = match_graphs(self.G, self.H, MatchingType.One_to_One)
    #     self.assertDictEqual(result, {})



class TestHierarchicalMatching(unittest.TestCase):
    def test_basic(self):
        '''
        G consists of an edge (2,1). Within 1 lie three nodes of H:
        1' has a parent (2') which has same position as 2,
        3' has no parent
        4' has a parent (5') which is far from 2.

        The resulting matching should map 1 to 1' and their parents 2 to 2'.
        '''
        self.G = nx.DiGraph()
        nodes_G = [
            (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0, 0]), 'radius': 1}),
        ]
        edges_G = [
            (2,1)
        ]
        self.G.add_nodes_from(nodes_G)
        self.G.add_edges_from(edges_G)

        self.H = nx.DiGraph()
        nodes_H = [
            (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0, 0]), 'radius': 1}),
            (3, {'coord': np.array([0, 0.1, 0]), 'radius': 1}),
            (4, {'coord': np.array([0, -0.1, 0]), 'radius': 1}),
            (5, {'coord': np.array([10, 0, 0]), 'radius': 1}),
        ]
        edges_H = [
            (3,2),
            (3,5),
            (2,1),
            (5,4),
        ]
        self.H.add_nodes_from(nodes_H)
        self.H.add_edges_from(edges_H)

        result = match_graphs(self.G, self.H, MatchingType.Hierarchical)
        self.assertDictEqual(result, {1: 1, 2: 2})

    
    def test_basic(self):
        '''
        G consists of an edge (2,1) where 2 is parent of 1. 
        Within 1 lie three nodes of H:
        1' (with some offset position from 1) has a parent (2') which has same position as 2,
        3' (same position as 1) has no parent
        4' (same position as 1) has a parent (5') which is far from 2.

        The resulting matching should map 1 to 1' and their parents 2 to 2',
        even though 3' and 4' are closer to 1, because 1' has fitting parent.
        '''
        self.G = nx.DiGraph()
        nodes_G = [
            (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0, 0]), 'radius': 1}),
        ]
        edges_G = [
            (2,1)
        ]
        self.G.add_nodes_from(nodes_G)
        self.G.add_edges_from(edges_G)

        self.H = nx.DiGraph()
        nodes_H = [
            (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0.2, 0]), 'radius': 1}),
            (3, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (4, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (5, {'coord': np.array([10, 0, 0]), 'radius': 1}),
        ]
        edges_H = [
            (3,2),
            (3,5),
            (2,1),
            (5,4),
        ]
        self.H.add_nodes_from(nodes_H)
        self.H.add_edges_from(edges_H)

        result = match_graphs(self.G, self.H, MatchingType.Hierarchical)
        self.assertDictEqual(result, {1: 1, 2: 2})

    def test_basic_2(self):
        self.G = nx.DiGraph()
        nodes_G = [
            (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0, 0]), 'radius': 1}),
            (3, {'coord': np.array([4, 0, 0]), 'radius': 1}),
        ]
        edges_G = [
            (1,2),
            (2,3),
        ]
        self.G.add_nodes_from(nodes_G)
        self.G.add_edges_from(edges_G)

        self.H = nx.DiGraph()
        nodes_H = [
            # Similar to G
            (1, {'coord': np.array([0, 0.2, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0.2, 0]), 'radius': 1}),
            (3, {'coord': np.array([4, 0.2, 0]), 'radius': 1}),
            # Just the two children (2,3) of G
            (4, {'coord': np.array([2, 0, 0]), 'radius': 1}),
            (5, {'coord': np.array([4, 0, 0]), 'radius': 1}),
            # Just the node 3 of G with a parent far out
            (6, {'coord': np.array([2, 0, 10]), 'radius': 1}),
            (7, {'coord': np.array([4, 0, 0]), 'radius': 1}),
            # The edge (2,3) of G where 2 has a parent far out
            (8, {'coord': np.array([0, 0, 10]), 'radius': 1}),
            (9, {'coord': np.array([2, 0, 0]), 'radius': 1}),
            (10, {'coord': np.array([4, 0, 0]), 'radius': 1}),
        ]
        edges_H = [
            (1,2),
            (2,3),
            (4,5),
            (6,7),
            (8,9),
            (9,10),
        ]
        self.H.add_nodes_from(nodes_H)
        self.H.add_edges_from(edges_H)

        result = match_graphs(self.G, self.H, MatchingType.Hierarchical)
        self.assertDictEqual(result, {1: 1, 2: 2, 3: 3})

class TestHierarchicalMatching(unittest.TestCase):
    def test_hierarchical_basic(self):

        G = nx.DiGraph()
        nodes_G = [
            (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0, 0]), 'radius': 1}),
        ]
        edges_G = [(2, 1)]
        G.add_nodes_from(nodes_G)
        G.add_edges_from(edges_G)


        H = nx.Graph()
        nodes_H = [
            (10, {'coord': np.array([0.1, 0, 0]), 'radius': 1}),
            (20, {'coord': np.array([2.1, 0, 0]), 'radius': 1}),
            (30, {'coord': np.array([0.2, 0.1, 0]), 'radius': 1}),
        ]
        edges_H = [(20, 10), (20, 30)]
        H.add_nodes_from(nodes_H)
        H.add_edges_from(edges_H)

        result = match_graphs(G, H, MatchingType.Hierarchical)


        self.assertIn(1, result)
        self.assertIn(2, result)


        self.assertEqual(result[2], 20)


        self.assertIn(result[1], [10, 30])

if __name__ == "__main__":
    unittest.main()
