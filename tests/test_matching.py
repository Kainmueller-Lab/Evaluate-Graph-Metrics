import unittest
import networkx as nx
import numpy as np

from project.matching import match_graphs, MatchingType, match_hungarian, match_hierarchical

import logging
logging.basicConfig(level=logging.INFO)



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
    
        result = match_hierarchical(self.G, self.H)
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
    
        result = match_hierarchical(self.G, self.H)
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

        self.H = nx.Graph()
        nodes_H = [
            # Similar to G
            (1, {'coord': np.array([0, 0.35, 0]), 'radius': 1}),
            (2, {'coord': np.array([2, 0.35, 0]), 'radius': 1}),
            (3, {'coord': np.array([4, 0.35, 0]), 'radius': 1}),
            # Just the two children (2,3) of G
            (4, {'coord': np.array([2, 0.3, 0]), 'radius': 1}),
            (5, {'coord': np.array([4, 0.3, 0]), 'radius': 1}),
            # Just the node 3 of G with a parent far out
            (6, {'coord': np.array([2, 0, 10]), 'radius': 1}),
            (7, {'coord': np.array([4, 0.2, 0]), 'radius': 1}),
            # The edge (2,3) of G where 2 has a parent far out
            (8, {'coord': np.array([0, 0, 10]), 'radius': 1}),
            (9, {'coord': np.array([2, 0.25, 0]), 'radius': 1}),
            (10, {'coord': np.array([4, 0.25, 0]), 'radius': 1}),
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

        result = match_hierarchical(self.G, self.H)
        self.assertDictEqual(result, {1: 1, 2: 2, 3: 3})

    # def test_hierarchical_basic(self):
    #
    #     G = nx.DiGraph()
    #     nodes_G = [
    #         (1, {'coord': np.array([0, 0, 0]), 'radius': 1}),
    #         (2, {'coord': np.array([2, 0, 0]), 'radius': 1}),
    #     ]
    #     edges_G = [(2, 1)]
    #     G.add_nodes_from(nodes_G)
    #     G.add_edges_from(edges_G)
    #
    #
    #     H = nx.Graph()
    #     nodes_H = [
    #         (10, {'coord': np.array([0.1, 0, 0]), 'radius': 1}),
    #         (20, {'coord': np.array([2.1, 0, 0]), 'radius': 1}),
    #         (30, {'coord': np.array([0.2, 0.1, 0]), 'radius': 1}),
    #     ]
    #     edges_H = [(20, 10), (20, 30)]
    #     H.add_nodes_from(nodes_H)
    #     H.add_edges_from(edges_H)
    #
    #     result = match_graphs(G, H, MatchingType.Hierarchical)
    #
    #
    #     self.assertIn(1, result)
    #     self.assertIn(2, result)
    #
    #
    #     self.assertEqual(result[2], 20)
    #
    #
    #     self.assertIn(result[1], [10, 30])



class TestHungarianMatching(unittest.TestCase):
    def test_simple_match_graphs(self):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()

        # toy example
        for i, c in enumerate([(0, 0, 0), (1, 1, 1), (2, 0, 0)]):
            g1.add_node(i, x=c[0], y=c[1], z=c[2])
        for j, c in enumerate([(0.1, 0, 0), (1.2, 1.1, 1), (5, 5, 5)]):
            g2.add_node(j, x=c[0], y=c[1], z=c[2])

        result = match_graphs(g1, g2, MatchingType.Hungarian)
        self.assertDictEqual(result, {0: 0, 1: 1, 2: 2})


    def test_simple_match_hungarian(self):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()

        # toy example
        for i, c in enumerate([(0, 0, 0), (1, 1, 1), (2, 0, 0)]):
            g1.add_node(i, x=c[0], y=c[1], z=c[2])
        for j, c in enumerate([(0.1, 0, 0), (1.2, 1.1, 1), (5, 5, 5)]):
            g2.add_node(j, x=c[0], y=c[1], z=c[2])

        result = match_hungarian(g1, g2, unmatch_penalty=1e3)
        self.assertDictEqual(result, {0: 0, 1: 1, 2: 2})

    
    def test_match_hungarian_with_low_unmatch_penalty(self):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()

        # toy example
        for i, c in enumerate([(0, 0, 0), (1, 1, 1), (2, 0, 0)]):
            g1.add_node(i, x=c[0], y=c[1], z=c[2])
        for j, c in enumerate([(0.1, 0, 0), (1.2, 1.1, 1), (5, 5, 5)]):
            g2.add_node(j, x=c[0], y=c[1], z=c[2])

        result = match_hungarian(g1, g2, unmatch_penalty=3)
        self.assertDictEqual(result, {0: 0, 1: 1, 2: None})

    def test_unmatch_penalty(self):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()

        # toy example
        for i, c in enumerate([(0, 0, 0), (1, 0, 0), (2, 0, 0)]):
            g1.add_node(i, x=c[0], y=c[1], z=c[2])
        for i, c in enumerate([(0, 0, 5), (1, 0, 5), (2, 0, 5)]):
            g2.add_node(i, x=c[0], y=c[1], z=c[2])

        result = match_hungarian(g1, g2, unmatch_penalty=4.9)
        self.assertDictEqual(result, {0: None, 1: None, 2: None})

        result = match_hungarian(g1, g2, unmatch_penalty=5.1)
        self.assertDictEqual(result, {0: 0, 1: 1, 2: 2})


    def test_example(self):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()

        # Toy example where in g1 there are intermediate nodes (corresp. to subsampled nodes in GT) and g2
        # has only the key nodes. Ideally, the matching should assign the corresponding key notes,
        # and leave the intermediate nodes unmatched.
        node_list  = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
        intermediate_node_list = [(0, 0, 0), (0, 0, 0.2), (0, 0, 0.4), (0, 0, 0.6), (0, 0, 0.8),
                                  (0, 0, 1), (0, 0, 1.2), (0, 0, 1.4), (0, 0, 1.6), (0, 0, 1.8),
                                  (0, 0, 2), (0, 0, 2.2), (0, 0, 2.4), (0, 0, 2.6), (0, 0, 2.8),
                                  (0, 0, 3)]

        for i, c in enumerate(intermediate_node_list):
            g1.add_node(i, x=c[0], y=c[1], z=c[2])
        
        for i, c in enumerate(node_list):
            g2.add_node(i, x=c[0], y=c[1], z=c[2])

        result = match_hungarian(g1, g2, unmatch_penalty=0.15)
        true_result = {0: 0, 1: None, 2: None, 3: None, 4: None,
                       5: 1, 6: None, 7: None, 8: None, 9: None,
                       10: 2, 11: None, 12: None, 13: None, 14: None,
                       15: 3}
        self.assertDictEqual(result, true_result)
    
    # TODO: Add more elaborate example

if __name__ == "__main__":
    unittest.main()
