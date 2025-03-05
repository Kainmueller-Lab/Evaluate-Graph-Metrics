import unittest
import networkx as nx
import numpy as np
import torch

from project.metrics import GraphMetrics, evaluate_all_metrics
from project.matching import match_graphs, MatchingType
from project.toydata import two_trees, smd_graph
from project.utils import read_graph_from_file, resample_graph, reduce_graphs,\
        visualize_matching

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

    # # TODO: Make tests for edge and node values
    def test_nodes(self):
        result = GraphMetrics.f1_score_nodewise(self.G, self.H, self.matching)
        self.assertEqual(result["precision_nodes"], 1)
        self.assertEqual(result["recall_nodes"], 1)
        self.assertEqual(result["f1_nodes"], 1)

    def test_edges(self):
        result = GraphMetrics.f1_score_edgewise(self.G, self.H, self.matching)
        self.assertEqual(result["precision_edges"], 2/3)
        self.assertEqual(result["recall_edges"], 2/4)
        self.assertEqual(result["f1_edges"], 2 / (3/2 + 4/2))



from project.streetmover_distance import SinkhornDistance_POT
class TestSinkhornDistance(unittest.TestCase):
    def test_trivial(self):
        N = 100 # Number nodes

        # Generate 3D point cloud A
        np.random.seed(22)
        A = np.random.rand(N, 3)

        # NOTE: Sometimes, if coordinates are between 0 and 1 the distance between itdentical point clouds has high error.
        sinkhorn_dist = SinkhornDistance_POT(regularization=1)
        result, _, _ = sinkhorn_dist(A,A)
        self.assertAlmostEqual(result, 0.0, delta=0.01)

        # Scale point cloud larger
        A = A*100 
        sinkhorn_dist = SinkhornDistance_POT(regularization=1)
        result, _, _ = sinkhorn_dist(A,A)
        self.assertAlmostEqual(result, 0.0, delta=0.01)

    def test_translation(self):
        N = 100 # Number nodes
        T = 20 # Translation distance

        # Generate two 3D point clouds A and A + t
        A = np.random.rand(N, 3)
        B = [coord + np.array([T,0,0]) for coord in A]

        sinkhorn_dist = SinkhornDistance_POT(regularization=1)
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

        sinkhorn_dist = SinkhornDistance_POT(regularization=1)
        result, _, _ = sinkhorn_dist(SA,SB)
        self.assertAlmostEqual(result, S*T, delta=0.01)

    def test_toydata(self):
        G = read_graph_from_file("tests/42_BP_GT.swc")
        H = read_graph_from_file("tests/42_BP.swc")

        G = G.to_undirected()
        H = H.to_undirected()

        G_coords = nx.get_node_attributes(G, 'coord')
        H_coords = nx.get_node_attributes(H, 'coord')

        G_nodes = [torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in G_coords.values()]
        H_nodes = [torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in H_coords.values()]


        sinkhorn_dist = SinkhornDistance_POT(regularization=1)
        result, _, _ = sinkhorn_dist(G_nodes,H_nodes)
        self.assertAlmostEqual(21.0847, result, delta=0.0001)

        scale = 0.1
        scaled_G_nodes = [scale * torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in G_coords.values()]
        scaled_H_nodes = [scale * torch.from_numpy(np.array(arr, dtype=np.float32)) for arr in H_coords.values()]
        result, _, _ = sinkhorn_dist(scaled_G_nodes,scaled_H_nodes)
        self.assertAlmostEqual(2.1084, result, delta=0.0001)



class TestSMD(unittest.TestCase):
        def test_trivial_smd(self):
            G, _ = two_trees() 
            result = GraphMetrics.street_mover_distance(G, G)
            self.assertAlmostEqual(result["smd_correct"], 0.0, delta=0.001)

        def test_translated_smd(self):
            G, H = smd_graph()
            result = GraphMetrics.street_mover_distance(G, H)
            self.assertAlmostEqual(result["smd_correct"], 1.0, delta=0.001)
            
        def test_smd_random_translate(self):
            N = 9
            G = nx.DiGraph()
            nodes_G = [(idx, {'coord': np.array([idx, 0]), 'radius': 1}) for idx in range(N)]
            edges_G = [(idx, idx+1) for idx in range(N-1)]
            G.add_nodes_from(nodes_G)
            G.add_edges_from(edges_G)

            def G_translated_by(t):
                H = nx.DiGraph()
                nodes_H = [(idx, {'coord': np.array([idx, t]), 'radius': 1}) for idx in range(N)]
                edges_H = [(idx, idx+1) for idx in range(N-1)]
                H.add_nodes_from(nodes_H)
                H.add_edges_from(edges_H)
                return H

            t = 20
            result = GraphMetrics.street_mover_distance(G, G_translated_by(t), normalize=False)
            self.assertAlmostEqual(result["smd_correct"], t, delta=0.001)


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

            result = GraphMetrics.street_mover_distance(G, H, normalize=False)
            self.assertAlmostEqual(result["smd_correct"], T, delta=0.001)

            result = GraphMetrics.street_mover_distance(A, B, normalize=False)
            self.assertAlmostEqual(result["smd_correct"], S*T, delta=0.001)



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



class TestToyData(unittest.TestCase):
    def test_main(self):
        # 1) Load graphs
        G = read_graph_from_file("tests/42_BP_GT.swc")
        H = read_graph_from_file("tests/42_BP.swc")
        assert nx.is_branching(G) and nx.is_branching(H), "Graphs must be branchings"

        # 2) No resampling

        # 3) Match graphs
        matching_type = MatchingType.Nearest

        matched = match_graphs(
            source=G,
            target=H,
            matching_type=matching_type,
            visualize=False)

        # 4) Reduce graph to have only branching and end points and their matched counterparts
        G_reduced, H_reduced, matched_reduced = reduce_graphs(
            G, H, matched)

        # 5) Compute metrics wrt to source and matched target graph
        results_dict = evaluate_all_metrics(
            G_reduced, H_reduced, matched, smd=True, visualize=False,
            G_resampled=None,
            H_resampled=None
        )
        self.assertAlmostEqual(results_dict["smd_faulty"], 0.0004, delta=0.01)
        self.assertAlmostEqual(results_dict["smd_correct"], 0.2483, delta=0.01)

        # Not normalized values:
        # self.assertAlmostEqual(results_dict["smd_faulty"], 95949.25, delta=0.001)
        # self.assertAlmostEqual(results_dict["smd_correct"], 77.465, delta=0.001)
    
    def test_toygraph(self):
        # 1) Load graphs
        G = read_graph_from_file("tests/toygraph_GT.swc")
        H = read_graph_from_file("tests/toygraph_predicted.swc")
        assert nx.is_branching(G) and nx.is_branching(H), "Graphs must be branchings"

        # 2) No resampling

        # 3) Match graphs
        matching_type = MatchingType.One_to_One

        matched = match_graphs(
            source=G,
            target=H,
            matching_type=matching_type,
            visualize=False)

        # 4) Reduce graph to have only branching and end points and their matched counterparts
        G_reduced, H_reduced, matched_reduced = reduce_graphs(
            G, H, matched)

        # 5) Compute metrics wrt to source and matched target graph
        results_dict = evaluate_all_metrics(
            G_reduced, H_reduced, matched, smd=True, visualize=False,
            G_resampled=None,
            H_resampled=None
        )
        print(results_dict)
        self.assertAlmostEqual(results_dict["smd_correct"], 0.0214, delta=0.001)
        self.assertEqual(results_dict["FM"], 1)



if __name__ == "__main__":
    unittest.main()
