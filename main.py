import logging
import networkx as nx
import argparse

from project.metrics import evaluate_all_metrics
from project.matching import match_graphs, MatchingType
from project.utils import read_graph_from_file, resample_graph, reduce_graphs,\
        visualize_matching
from project.toydata import *
import glob
import os


logger = logging.getLogger(__name__)


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_fn", type=str, dest="gt_fn", default=None,
        help="path to ground truth file"
    )
    parser.add_argument(
        "--pred_fn", type=str, dest="pred_fn", default=None,
        help="path to predicted file"
    )
    parser.add_argument(
        "--gt_dir", type=str, dest="gt_dir", default=None,
        help="path to ground truth folder"
    )
    parser.add_argument(
        "--pred_dir", type=str, dest="pred_dir", default=None,
        help="path to predicted folder"
    )

    parser.add_argument(
        "--output_folder", type=str, default=None,
        help="path to output folder"
    )
    parser.add_argument(
        "--matching", type=str, default="one_to_one",
        choices=["hierarchical", "nearest", "nearest_with_radius", "greedy",
                 "greedy_with_parent", "one_to_one"], #TODO: check if we can unify here
        help="choose matching method to assign pred to gt nodes"
    )
    parser.add_argument(
        "--matching_dist", type=str, default="fixed",
        choices=["radius", "fixed"],
        help="specify if matching distance should be based on gt radius or fixed distance"
    )
    parser.add_argument(
        "--max_distance", type=int, default=10,
        help="maximal distance to match gt and pred points"
    )
    parser.add_argument(
        "--resample", action="store_true", default=False,
        help="flag to resample gt and pred skeletons"
    )
    # maybe --no-resample as flag, as same resample should be default?
    parser.add_argument(
        "--resample_stepsize", type=int, default=2,
        help="stepsize for resampling gt and pred graph for node matching"
    )
    parser.add_argument(
        "--reduce_graph", action="store_true", default=False,
        help="flag to only keep branching points and end points and their matched counterparts"
    )
    parser.add_argument(
        "--smd", action="store_true", default=False,
        help="flag to compute wasserstein distance (smd)"
    )
    parser.add_argument(
        "--visualize", action="store_true", default=False,
        help="flag to visualize intermediate steps"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="flag to set logging to DEBUG"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # 1) Load graphs with attributes
    # NOTE: We currently expect all graphs to be Branchings, see https://networkx.org/documentation/stable/reference/algorithms/tree.html
    if args.gt_fn is not None and args.pred_fn is not None:
        Gs = [read_graph_from_file(args.gt_fn)]
        Hs = [read_graph_from_file(args.pred_fn)]
    elif args.gt_dir is not None and args.pred_dir is not None:
        gt_fls = sorted(os.listdir(args.gt_dir))
        pred_fls = sorted(os.listdir(args.pred_dir))
        logger.debug(f"gt files: {gt_fls}, pred files: {pred_fls}")
        assert len(gt_fls) > 0 and gt_fls == pred_fls, "lists of gt and pred files have to be identical and non-empty!"
        Gs = [read_graph_from_file(os.path.join(args.gt_dir, gt)) for gt in gt_fls]
        Hs = [read_graph_from_file(os.path.join(args.pred_dir, pred)) for pred in pred_fls]
    else:
        G, H = two_trees()
        Gs = [read_graph_from_file("tests/toygraph_GT.swc")]
        Hs = [read_graph_from_file("tests/toygraph_predicted.swc")]
    #assert nx.is_branching(G) and nx.is_branching(H), "Graphs must be branchings"
    logger.debug(f"number of nodes in G: {[G.number_of_nodes() for G in Gs]}")
    logger.debug(f"number of edges in G: {[G.number_of_edges() for G in Gs]}")
    logger.debug(f"number of nodes in H: {[H.number_of_nodes() for H in Hs]}")
    logger.debug(f"number of edges in H: {[H.number_of_edges() for H in Hs]}")

    # 2) Resample both graphs to have same spacing -> TODO: copy to matching?
    if args.resample:
        Gs = [resample_graph(G, stepsize=args.resample_stepsize,
                             visualize=args.visualize) for G in Gs]
        Hs = [resample_graph(H, stepsize=args.resample_stepsize,
                             visualize=args.visualize) for H in Hs]
        logger.info(
            "number of nodes in G resampled: %s",
            [G.number_of_nodes() for G in Gs])
        logger.info(
            "number of nodes in H resampled: %s",
            [H.number_of_nodes() for H in Hs])

    # 3) Match graphs
    # Match target graph (e.g. prediction) to source graph (e.g. ground truth)
    # The matching variable is a dictionary that maps nodes in target graph to
    # nodes in source graph.
    # NOTE: This is not symmetric, match(G,H) in general does not equal match(H,G).
    if args.matching == "hierarchical":
        matching_type = MatchingType.Hierarchical
    elif args.matching == "one_to_one":
        matching_type = MatchingType.One_to_One
    elif args.matching == "nearest":
        matching_type = MatchingType.Nearest
    else:
        raise NotImplementedError

    matcheds = [
        match_graphs(
            source=G,
            target=H,
            matching_type=matching_type,
            matching_dist=args.matching_dist,
            max_distance=args.max_distance,
            visualize=args.visualize)
        for G, H in zip(Gs, Hs)
    ]
    logger.info(
            "number of matched nodes: %s",
            [len(matched) for matched in matcheds])
    logger.info(
            "number of unmatched nodes: %s",
            [(G.number_of_nodes() - len(matched), H.number_of_nodes() - len(matched))
             for G, H, matched in zip(Gs, Hs, matcheds)])

    # 4) Reduce graph to have only branching and end points and their matched counterparts
    if args.reduce_graph:
        Gs, Hs, matcheds = zip(*[reduce_graphs(G, H, matched) for G, H, m in zip(G, H, matcheds)])
        if args.visualize:
            visualize_matching(G[0], H[0], matched[0])

    # 5) Compute metrics wrt to source and matched target graph
    results_dict = evaluate_all_metrics(
        args,
        Gs, Hs, matcheds,
        smd=args.smd,
        visualize=args.visualize,
    )

    # 6) Print results / TODO: save as json or csv file
    for name, value in results_dict.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
