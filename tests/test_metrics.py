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
        self.assertEqual(result["betti_0"].astype(int), 1)

    def test_betti_1_error(self):
        result = GraphMetrics.betti_1_error(self.G, self.H)
        self.assertEqual(result["betti_1"].astype(int), 0)

    def test_graph_edit_distance(self):
        result = GraphMetrics.graph_edit_distance(self.G, self.H, None)
        self.assertEqual(result["ted"], 3.0)

    # # TODO: Make teste for edge and node values
    # def test_precision(self):
    #     result = GraphMetrics.f1_score_nodewise(self.G, self.H, self.matching)
    #     self.assertEqual(result["precision_nodes"], 2/3)

    # def test_recall(self):
    #     result = GraphMetrics.f1_score_nodewise(self.G, self.H, self.matching)
    #     self.assertEqual(result["recall_nodes"], 2/4)

    # def test_f1_score(self):
    #     result = GraphMetrics.f1_score_nodewise(self.G, self.H, self.matching)
    #     self.assertEqual(result["f1_nodes"], 2 / (3/2 + 4/2))

from project.streetmover_distance import SinkhornDistance_POT
class TestSinkhornDistance(unittest.TestCase):
    def test_trivial(self):
        N = 100 # Number nodes

        # Generate 3D point cloud A
        np.random.seed(22)
        A = np.random.rand(N, 3)

        # NOTE: Sometimes, if coordinates are between 0 and 1 the distance between itdentical point clouds has high error.
        sinkhorn_dist = SinkhornDistance_POT(regularization=0.05)
        result, _, _ = sinkhorn_dist(A,A)
        self.assertAlmostEqual(result, 0.0, delta=0.1)

        A = A*100 
        sinkhorn_dist = SinkhornDistance_POT(regularization=0.05)
        result, _, _ = sinkhorn_dist(A,A)
        self.assertAlmostEqual(result, 0.0, delta=0.01)

    def test_translation(self):
        N = 100 # Number nodes
        T = 20 # Translation distance

        # Generate two 3D point clouds A and A + t
        A = np.random.rand(N, 3)
        B = [coord + np.array([T,0,0]) for coord in A]

        sinkhorn_dist = SinkhornDistance_POT(regularization=0.05)
        result, _, _ = sinkhorn_dist(A,B)
        self.assertAlmostEqual(result, T, delta=0.01)

    def test_scale(self):
        N = 100 # Number nodes
        T = 20 # Translation distance
        S = 0.3

        # Generate two 3D point clouds A and A + t
        A = np.random.rand(N, 3)
        B = [coord + np.array([T,0,0]) for coord in A]
        # Scale them
        SA = [coord * S for coord in A]
        SB = [coord * S for coord in B]

        sinkhorn_dist = SinkhornDistance_POT(regularization=0.05)
        result, _, _ = sinkhorn_dist(SA,SB)
        self.assertAlmostEqual(result, S*T, delta=0.01)


class TestSMD(unittest.TestCase):
        def test_trivial_smd(self):
            G, _ = two_trees() 
            result = GraphMetrics.street_mover_distance_POT(G, G, normalize=False)
            # NOTE: Strangely the error between identical graphs is quite high. Unclear yet why. - ck
            self.assertAlmostEqual(result["smd_POT"], 0.0, delta=0.4)

        def test_translated_smd(self):
            G, H = smd_graph()
            result = GraphMetrics.street_mover_distance_POT(G, H, normalize=False)
            self.assertAlmostEqual(result["smd_POT"], 1.0, delta=0.1)

        # def test_smd_normalization(self):
        #     N = 5
        #     G = nx.DiGraph()
        #     nodes_G = [(idx, {'coord': np.array([idx, 0]), 'radius': 1}) for idx in range(N)]
        #     edges_G = [(idx, idx+1) for idx in range(N-1)]
        #     G.add_nodes_from(nodes_G)
        #     G.add_edges_from(edges_G)

        #     H = nx.DiGraph()
        #     nodes_H = [(idx, {'coord': np.array([idx, 1]), 'radius': 1}) for idx in range(N)]
        #     edges_H = [(idx, idx+1) for idx in range(N-1)]
        #     H.add_nodes_from(nodes_H)
        #     H.add_edges_from(edges_H)

        #     result = GraphMetrics.street_mover_distance(G, H, normalize=False)
        #     self.assertAlmostEqual(result["smd"], 1.0, delta=0.001)
        #     # result = GraphMetrics.street_mover_distance(G, H, normalize=True)
        #     # self.assertAlmostEqual(result["smd"], 1.0, delta=0.001)
            
        def test_smd_random_translate(self):
            N = 9
            G = nx.DiGraph()
            nodes_G = [(idx, {'coord': np.array([idx, 0]), 'radius': 1}) for idx in range(N)]
            edges_G = [(idx, idx+1) for idx in range(N-1)]
            G.add_nodes_from(nodes_G)
            G.add_edges_from(edges_G)

            H = nx.DiGraph()
            nodes_H = [(idx, {'coord': np.array([idx, 1]), 'radius': 1}) for idx in range(N)]
            edges_H = [(idx, idx+1) for idx in range(N-1)]
            H.add_nodes_from(nodes_H)
            H.add_edges_from(edges_H)

            def translate_graph(t):
                H = nx.DiGraph()
                nodes_H = [(idx, {'coord': np.array([idx, t]), 'radius': 1}) for idx in range(N)]
                edges_H = [(idx, idx+1) for idx in range(N-1)]
                H.add_nodes_from(nodes_H)
                H.add_edges_from(edges_H)
                return H

            t = 5 # TODO: This gets instable and 0 is returned when t is too large eg 20
            result = GraphMetrics.street_mover_distance_POT(G, translate_graph(t), normalize=False)
            self.assertAlmostEqual(result["smd_POT"], t, delta=0.1)
            # result = GraphMetrics.street_mover_distance(G, H, normalize=True)
            # self.assertAlmostEqual(result["smd"], 1.0, delta=0.001)


        def test_smd_random_scale(self):
            N = 10 # Number nodes
            T = 5 # Translation distance
            S = 0.3 # Scaling factor

            #coords = [np.random.randint(0, 10, 2) for _ in range(N)]
            coords = [np.array([idx, 0]) for idx in range(N)]
            translated_coords = [coord + np.array([T,0]) for coord in coords]

            # Two translated graphs
            G = nx.DiGraph()
            nodes_G = [(idx, {'coord': coords[idx], 'radius': 1}) for idx in range(N)]
            edges_G = [(idx, idx+1) for idx in range(N-1)]
            G.add_nodes_from(nodes_G)
            G.add_edges_from(edges_G)
            H = nx.DiGraph()
            nodes_H = [(idx, {'coord': translated_coords[idx], 'radius': 1}) for idx in range(N)]
            edges_H = [(idx, idx+1) for idx in range(N-1)]
            H.add_nodes_from(nodes_H)
            H.add_edges_from(edges_H)

            # Graphs G and H but scaled
            scaled_coords_G = [coord * S for coord in coords]
            scaled_coords_H = [coord * S for coord in translated_coords]
            A = nx.DiGraph()
            nodes_A = [(idx, {'coord': scaled_coords_G[idx], 'radius': 1}) for idx in range(N)]
            edges_A = [(idx, idx+1) for idx in range(N-1)]
            A.add_nodes_from(nodes_A)
            A.add_edges_from(edges_A)
            B = nx.DiGraph()
            nodes_B = [(idx, {'coord': scaled_coords_H[idx], 'radius': 1}) for idx in range(N)]
            edges_B = [(idx, idx+1) for idx in range(N-1)]
            B.add_nodes_from(nodes_B)
            B.add_edges_from(edges_B)

            result = GraphMetrics.street_mover_distance_POT(G, H, normalize=False)
            self.assertAlmostEqual(result["smd_POT"], T, delta=0.01)
            original_smd = result["smd_POT"]

            result = GraphMetrics.street_mover_distance_POT(A, B, normalize=False)
            self.assertAlmostEqual(result["smd_POT"], S*T, delta=0.4)



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
