import logging
import sys

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment

import time

from enum import Enum
from .utils import label_connected_components, visualize_matching


logger = logging.getLogger(__name__)


class MatchingType(Enum):
    Nearest = 1
    Nearest_Within_Radius = 2
    Greedy = 3
    Greedy_with_parent = 4
    Hierarchical = 5
    One_to_One = 6 # Matches to nearest candidate within radius and removes it from candidate list.
    Hungarian = 7
    # TODO: Graph matching solution? Could improve this, but unclear if efficient.


def match_graphs(source, target, matching_type, matching_dist="fixed",
                 max_distance=10, visualize=False, candidate_selection="query"):
    """
    Matches nodes from the source graph to nodes in the target graph based on the specified matching strategy.

    Parameters:
    ----------
    source : networkx.Graph
        The source graph whose nodes need to be matched.
        Nodes must have positional attributes ('coord') for distance-based matching.
    target : networkx.Graph
        The target graph to which source nodes will be matched.
        Nodes must have positional attributes ('coord') for distance-based matching.
    matching_type : MatchingType
        The type of matching to be performed. Supported types are:
        - MatchingType.Nearest: Matches each source node to the nearest target node. Many-to-one matching allowed. All source nodes are matched.
        - MatchingType.Nearest_Within_Radius: Matches each source node to the nearest target node within a radius. Many-to-one matching allowed. Source nodes may not be matched.
        - MatchingType.Greedy: Matches each source node to the nearest target node within its radius in a greedy fashion. One-to-one. If no target node is within the radius, the source node is not matched.
        - MatchingType.Hungarian: Uses the Hungarian algorithm for optimal one-to-at-most-one matching based on distance.
        
    Returns:
    -------
    match_dict : dict
        A dictionary representing the matching result. Keys are source graph node IDs, and values are target graph node IDs.
        If a source node is not matched (e.g., in Greedy), it will not appear in the dictionary.
    """
    assert matching_type in MatchingType, f"Unknown matching type: {matching_type}"
    logger.info("start node matching (%s)" % matching_type)
    if matching_type == MatchingType.Nearest:
        # Matches a source node to the closest target node. Can be many-to-one. Every source node is matched.
        return match_nearest(source=source, target=target)
    if matching_type == MatchingType.Nearest_Within_Radius:
        # Matches a source node to the closest target node within a radius. Can be many-to-one. Source nodes may not be matched.
        return match_nearest_within_radius(source=source, target=target)
    if matching_type == MatchingType.Greedy:
        # Matches a source node to closest target node in a greedy fashion. Is one-to-one. Every source node is matched.
        return match_greedy(source=source, target=target)
    if matching_type == MatchingType.Greedy_with_parent:
        return match_greedy_parent(source=source, target=target)
    # if matching_type == MatchingType.Hierarchical:
    #     return match_hierarchical(source=source, target=target,
    #                               visualize=visualize)
    # if matching_type == MatchingType.One_to_One:
    #     return match_one_to_one(
    #         source=source, target=target, matching_dist=matching_dist,
    #         max_distance=max_distance, visualize=visualize)
    if matching_type == MatchingType.Hierarchical:
        return match_hierarchical(
            source=source,
            target=target,
            visualize=visualize,
            candidate_selection=candidate_selection,
            max_distance=max_distance
        )

    if matching_type == MatchingType.One_to_One:
        return match_one_to_one(
            source=source,
            target=target,
            matching_dist=matching_dist,
            max_distance=max_distance,
            visualize=visualize,
            candidate_selection=candidate_selection
        )
    if matching_type == MatchingType.Hungarian:
        return match_hungarian(source=source, target=target, unmatch_penalty=max_distance)
    return {}




def match_nearest(source, target):
    '''
    Matches a source node to the closest target node. Can be many-to-one. Every source node is matched.
    '''
    match_dict = {}
    source_positions = nx.get_node_attributes(source, 'coord')
    target_positions = nx.get_node_attributes(target, 'coord')
    if not source_positions or not target_positions:
        raise ValueError("Both source and target graphs must have 'coord' attributes for nodes.")

    target_nodes = list(target_positions.keys())  # Target node list
    target_coords = np.array([target_positions[node] for node in target_nodes])  # Nx2 or Nx3 matrix of coordinates


    target_kdtree = KDTree(target_coords)

    # Find the closest target node
    for source_node, source_pos in source_positions.items():
        dist, idx = target_kdtree.query(source_pos)
        closest_target_node = target_nodes[idx]
        match_dict[source_node] = closest_target_node

    return match_dict


def match_nearest_within_radius(source, target):
    '''
    Matches a source node to the closest target node. Can be many-to-one. Every source node is matched.
    '''
    match_dict = {}
    source_positions = nx.get_node_attributes(source, 'coord')
    source_radii = nx.get_node_attributes(source, 'radius')
    target_positions = nx.get_node_attributes(target, 'coord')
    if not source_positions or not target_positions:
        raise ValueError("Both source and target graphs must have 'coord' attributes for nodes.")

    for source_node, source_pos in source_positions.items():
        # Find the closest target node
        closest_target_node = min(
            target_positions.keys(),
            key=lambda target_node: np.linalg.norm(np.array(source_pos) - np.array(target_positions[target_node]))
        )
        if np.linalg.norm(np.array(source_pos) - np.array(target_positions[closest_target_node])) <= source_radii[source_node]:
            match_dict[source_node] = closest_target_node

    return match_dict


# def match_one_to_one(source, target, matching_dist="fixed",
#                      max_distance=5, visualize=False):
#     """
#     Matches source nodes to the nearest available target node within a given radius.
#     Ensures a one-to-one matching: each target is used at most once.
#     """
#     match_dict = {}
#     source_positions = nx.get_node_attributes(source, 'coord')
#     source_nodes = list(source_positions.keys())
#     source_coords = np.array([source_positions[node] for node in source_nodes])
#     source_radii = nx.get_node_attributes(source, 'radius')
#     source_radii = np.array([source_radii[k] for k in source_radii.keys()])
#
#     target_positions = nx.get_node_attributes(target, 'coord')
#     target_nodes = list(target_positions.keys())  # Target node list
#     target_coords = np.array([target_positions[node] for node in target_nodes])  # Nx2 or Nx3 matrix of coordinates
#     target_kdtree = KDTree(target_coords)
#
#     match_dict = {}  # Store matched (source_idx, target_idx)
#     used_targets = []  # Keep track of assigned target indices
#     used_sources = []
#     dists = []
#     source_ids = []
#     target_ids = []
#
#     # get distance between all gt-pred pairs
#     if matching_dist == "fixed":
#         k_neighbors = target_kdtree.query_ball_point(
#             source_coords, [max_distance], return_length=True)
#         kd_dists, kd_indices = target_kdtree.query(
#             source_coords, k=max(k_neighbors), distance_upper_bound=max_distance)
#     elif matching_dist == "radius":
#         k_neighbors = target_kdtree.query_ball_point(
#             source_coords, source_radii, return_length=True)
#         kd_dists, kd_indices = target_kdtree.query(
#             source_coords, k=max(k_neighbors))
#     else:
#         raise NotImplementedError
#
#     for i, (c_dists, c_ids) in enumerate(zip(kd_dists, kd_indices)):
#         if type(c_dists) in [float, np.float32, np.float64]:
#             c_dists = [c_dists]
#             c_ids = [c_ids]
#         for c_dist, c_id in zip(c_dists, c_ids):
#             if not np.isinf(c_dist):
#                 dists.append(c_dist)
#                 source_ids.append(i)
#                 target_ids.append(c_id)
#
#     sort_idx = np.argsort(dists)
#     # sort dists and id lists in inreasing order
#     dists = np.array(dists)[sort_idx]
#     source_ids = np.array(source_ids)[sort_idx]
#     target_ids = np.array(target_ids)[sort_idx]
#
#     # iterate through all gt-pred pairs
#     for c_dist, c_source, c_target in zip(dists, source_ids, target_ids):
#         c_source_nx = source_nodes[c_source]
#         c_target_nx = target_nodes[c_target]
#         # if nodes haven't been matched yet, match them
#         if c_source_nx not in used_sources and c_target_nx not in used_targets:
#             match_dict[c_source_nx] = c_target_nx
#             used_targets.append(c_target_nx)
#             used_sources.append(c_source_nx)
#
#     # if visualize:
#     #     visualize_matching(source, target, match_dict)
#
#     return match_dict
#
# def match_one_to_one_query(source, target, matching_dist="fixed",
#                      max_distance=10, visualize=False, candidate_selection="query"):
#     """
#     Matches source nodes to the nearest available target node within a given radius.
#     Ensures a one-to-one matching: each target is used at most once.
#     """
#     match_dict = {}
#     source_positions = nx.get_node_attributes(source, 'coord')
#     source_nodes = list(source_positions.keys())
#     source_coords = np.array([source_positions[node] for node in source_nodes])
#     source_radii = nx.get_node_attributes(source, 'radius')
#     source_radii = np.array([source_radii[k] for k in source_radii.keys()])
#
#     target_positions = nx.get_node_attributes(target, 'coord')
#     target_nodes = list(target_positions.keys())
#     target_coords = np.array([target_positions[node] for node in target_nodes])
#     target_kdtree = KDTree(target_coords)
#
#     used_targets = set()
#     used_sources = set()
#
#     dists = []
#     source_ids = []
#     target_ids = []
#
#     if matching_dist == "fixed":
#         # Get only 1 nearest neighbor within max_distance
#         kd_dists, kd_indices = target_kdtree.query(source_coords, k=1, distance_upper_bound=max_distance)
#     elif matching_dist == "radius":
#         kd_dists, kd_indices = target_kdtree.query(source_coords, k=1, distance_upper_bound=source_radii)
#     else:
#         raise NotImplementedError
#
#     # Collect valid (not inf) distances
#     for i, (dist, idx) in enumerate(zip(kd_dists, kd_indices)):
#         if not np.isinf(dist):
#             dists.append(dist)
#             source_ids.append(i)
#             target_ids.append(idx)
#
#     sort_idx = np.argsort(dists)
#     dists = np.array(dists)[sort_idx]
#     source_ids = np.array(source_ids)[sort_idx]
#     target_ids = np.array(target_ids)[sort_idx]
#
#     for dist, src_idx, tgt_idx in zip(dists, source_ids, target_ids):
#         src_node = source_nodes[src_idx]
#         tgt_node = target_nodes[tgt_idx]
#         if src_node not in used_sources and tgt_node not in used_targets:
#             match_dict[src_node] = tgt_node
#             used_sources.add(src_node)
#             used_targets.add(tgt_node)
#
#     # if visualize:
#     #     visualize_matching(source, target, match_dict)
#
#     return match_dict

def match_one_to_one(source, target, matching_dist="fixed",
                     max_distance=10, visualize=False, candidate_selection="query"):
    """
    Matches source nodes to the nearest available target node within a given radius.
    Ensures a one-to-one matching: each target is used at most once.
    """
    match_dict = {}
    source_positions = nx.get_node_attributes(source, 'coord')
    source_nodes = list(source_positions.keys())
    source_coords = np.array([source_positions[node] for node in source_nodes])

    source_radii_dict = nx.get_node_attributes(source, 'radius')
    source_radii = np.array([source_radii_dict[node] for node in source_nodes])

    target_positions = nx.get_node_attributes(target, 'coord')
    target_nodes = list(target_positions.keys())
    target_coords = np.array([target_positions[node] for node in target_nodes])
    target_kdtree = KDTree(target_coords)

    used_targets = set()
    used_sources = set()

    dists = []
    source_ids = []
    target_ids = []

    if candidate_selection == "query":

        if matching_dist == "fixed":
            kd_dists, kd_indices = target_kdtree.query(
                source_coords, k=1, distance_upper_bound=max_distance
            )
        elif matching_dist == "radius":
            kd_dists, kd_indices = target_kdtree.query(
                source_coords, k=1, distance_upper_bound=source_radii
            )
        else:
            raise NotImplementedError

        for i, (dist, idx) in enumerate(zip(kd_dists, kd_indices)):
            if not np.isinf(dist):
                dists.append(dist)
                source_ids.append(i)
                target_ids.append(idx)

    elif candidate_selection == "ball":
        if matching_dist == "fixed":
            k_neighbors = target_kdtree.query_ball_point(
                source_coords, [max_distance], return_length=True
            )
            if max(k_neighbors) == 0:
                return {}
            kd_dists, kd_indices = target_kdtree.query(
                source_coords, k=max(k_neighbors), distance_upper_bound=max_distance
            )
        elif matching_dist == "radius":
            k_neighbors = target_kdtree.query_ball_point(
                source_coords, source_radii, return_length=True
            )
            if max(k_neighbors) == 0:
                return {}
            kd_dists, kd_indices = target_kdtree.query(
                source_coords, k=max(k_neighbors)
            )
        else:
            raise NotImplementedError

        for i, (c_dists, c_ids) in enumerate(zip(kd_dists, kd_indices)):
            if isinstance(c_dists, (float, np.float32, np.float64)):
                c_dists = [c_dists]
                c_ids = [c_ids]

            for c_dist, c_id in zip(c_dists, c_ids):
                if not np.isinf(c_dist):
                    if matching_dist == "radius" and c_dist > source_radii[i]:
                        continue
                    dists.append(c_dist)
                    source_ids.append(i)
                    target_ids.append(c_id)
    else:
        raise ValueError(f"Unknown candidate_selection: {candidate_selection}")

    sort_idx = np.argsort(dists)
    dists = np.array(dists)[sort_idx]
    source_ids = np.array(source_ids)[sort_idx]
    target_ids = np.array(target_ids)[sort_idx]

    for dist, src_idx, tgt_idx in zip(dists, source_ids, target_ids):
        src_node = source_nodes[src_idx]
        tgt_node = target_nodes[tgt_idx]
        if src_node not in used_sources and tgt_node not in used_targets:
            match_dict[src_node] = tgt_node
            used_sources.add(src_node)
            used_targets.add(tgt_node)

    return match_dict

def match_greedy(source, target):
    '''
    Matches a source node to closest target node within its radius in a greedy fashion.
    '''
    match_dict = {}
    source_positions = nx.get_node_attributes(source, 'coord')
    source_radii = nx.get_node_attributes(source, 'radius')
    target_positions = nx.get_node_attributes(target, 'coord')

    if not source_positions or not target_positions:
        raise ValueError("Both source and target graphs must have 'coord' attributes for nodes.")

    # Convert target positions to a mutable set for greedy one-to-one matching
    unmatched_targets = set(target_positions.keys())

    for source_node, source_pos in source_positions.items():
        if not unmatched_targets:
            continue

        # Find the closest unmatched target node
        closest_target_node = min(
            unmatched_targets,
            key=lambda target_node: np.linalg.norm(np.array(source_pos) - np.array(target_positions[target_node]))
        )
        if np.linalg.norm(np.array(source_pos) - np.array(target_positions[closest_target_node])) <= source_radii[source_node]:
            match_dict[source_node] = closest_target_node
            unmatched_targets.remove(closest_target_node)  # Mark the target as matched

    return match_dict


def match_greedy_parent(source, target):
    """
    Matches nodes in the source graph to nodes in the target graph based on spherical proximity
    and parent relationship constraints. Also prints unmatched nodes in both graphs.

    """
    match_dict = {}


    source_positions = nx.get_node_attributes(source, 'coord')
    source_radii = nx.get_node_attributes(source, 'radius')
    target_positions = nx.get_node_attributes(target, 'coord')
    target_radii = nx.get_node_attributes(target, 'radius')

    matched_targets = set()

    for source_node, source_pos in source_positions.items():
        source_radius = source_radii[source_node]
        proximity_candidates = []

        # Find all candidates in the spherical proximity of the source node
        for target_node, target_pos in target_positions.items():
            dist = np.linalg.norm(np.array(source_pos) - np.array(target_pos))
            if dist <= source_radius:
                proximity_candidates.append(target_node)


        best_candidate = None
        minimal_d = float('inf')

        for candidate in proximity_candidates:
            candidate_pos = target_positions[candidate]
            candidate_radius = target_radii[candidate]


            d_pos = np.linalg.norm(np.array(source_pos) - np.array(candidate_pos))
            d_radius = abs(source_radius - candidate_radius)
            d = d_pos + d_radius  # Assuming equal weights


            source_parent = next(source.predecessors(source_node), None) if source_node in source else None
            candidate_parent = next(target.predecessors(candidate), None) if candidate in target else None

            if source_parent and candidate_parent:

                source_parent_pos = source_positions.get(source_parent, None)
                candidate_parent_pos = target_positions.get(candidate_parent, None)
                source_parent_radius = source_radii.get(source_parent, 0)
                candidate_parent_radius = target_radii.get(candidate_parent, 0)

                if source_parent_pos is not None and candidate_parent_pos is not None:
                    d_parent_pos = np.linalg.norm(np.array(source_parent_pos) - np.array(candidate_parent_pos))
                    d_parent_radius = abs(source_parent_radius - candidate_parent_radius)
                    d_parent = d_parent_pos + d_parent_radius
                    d += d_parent


            if d < minimal_d:
                minimal_d = d
                best_candidate = candidate


        if best_candidate:
            match_dict[source_node] = best_candidate
            matched_targets.add(best_candidate)


    unmatched_source = set(source.nodes) - set(match_dict.keys())
    unmatched_target = set(target.nodes) - matched_targets


    print("Unmatched nodes in source graph:", len(unmatched_source))
    print("Unmatched nodes in target graph:", len(unmatched_target))
    print("matched targets:", len(matched_targets))

    return match_dict, unmatched_source, unmatched_target


def check_parent_tree_label(graph, node, pred_matched_labels):
    parent_id = list(graph.predecessors(node))[0]
    parent_label = None
    if parent_id in pred_matched_labels:
        parent_label = pred_matched_labels[parent_id]
    else:
        # check also for grandparents if parent is not assigned
        for i in range(3):
            ancester_id = list(graph.predecessors(parent_id))
            if len(ancester_id) > 0:
                ancester_id = ancester_id[0]
            else:
                break
            if ancester_id in pred_matched_labels:
                parent_label = pred_matched_labels[ancester_id]
                break
            else:
                parent_id = ancester_id
    return parent_label


def match_hierarchical_old(source, target, visualize=False):
    start_time = time.time()
    match_dict = {}
    pred_matched_labels = {}
    # resample graph to have nodes along the segments

    # find gt roots
    gt_roots = [n for n, d in source.in_degree() if d==0]
    pred_roots = [n for n, d in target.in_degree() if d==0]

    # label connected components (each tree one label)
    source, gt_labels = label_connected_components(source)
    target, pred_labels = label_connected_components(target)
    # get positions
    source_positions = nx.get_node_attributes(source, 'coord')
    target_positions = nx.get_node_attributes(target, 'coord')
    if not source_positions or not target_positions:
        raise ValueError("Both source and target graphs must have 'coord' attributes for nodes.")
    # build kd trees
    pred_kdtree = KDTree(np.array(list(target_positions.values())))
    gt_kdtree = KDTree(np.array(list(source_positions.values())))

    # iterate through each gt tree
    for root in gt_roots:
        clabel = gt_labels[root]
        cnode = root
        next_nodes = []
        # traverse through tree
        while True:
            dists, candidates = pred_kdtree.query(
                source_positions[cnode], p=2,
                distance_upper_bound=10
            )
            # distance_upper_bound = source.nodes[cnode]["radius"]
            # TODO: add different options here: radius, parameter for fixed width
            if np.isinf(dists):
                if source.out_degree(cnode) > 0:
                    next_nodes += list(source.successors(cnode))
                if len(next_nodes) > 0:
                    cnode = next_nodes.pop()
                    continue
                else:
                    break
            if type(candidates) in [int, np.int64]:
                candidates = [candidates]
            if type(dists) == float:
                dists = [dists]
            candidates = np.array(target.nodes)[candidates]
            if len(candidates) > 1:
                print("dists, candidates: ", dists, candidates)

            # get semantic (root, segment, branching, end) and parent label
            candidate_semantic = []
            candidate_parent_label = []
            for pnode in candidates:
                if target.in_degree(pnode) == 0:
                    candidate_semantic.append("root")
                    candidate_parent_label.append(None)
                else:
                    if target.out_degree(pnode) == 0:
                        candidate_semantic.append("end")
                    elif target.out_degree(pnode) == 1:
                        candidate_semantic.append("segment")
                    else:
                        candidate_semantic.append("branching")
                    candidate_parent_label.append(
                        check_parent_tree_label(target, pnode,
                                                pred_matched_labels))
            candidate_semantic = np.array(candidate_semantic)
            candidate_parent_label = np.array(candidate_parent_label)

            # match prediction nodes to gt nodes
            match = False
            if source.in_degree(cnode) == 0:
                if np.any(candidate_semantic == "root"):
                    # match two roots with each other
                    pnode = candidates[candidate_semantic=="root"][0]
                    match = True
                elif np.any(candidate_parent_label == None):
                    # match node with other semantic, but with unmatched parent
                    pnode = candidates[candidate_parent_label == None][0]
                    match = True
                else:
                    print("NOT MATCHED. Check if other gt tree close by")
            else:
                if source.out_degree(cnode) == 0:
                    cnode_semantic = "end"
                elif source.out_degree(cnode) == 1:
                    cnode_semantic = "segment"
                else:
                    cnode_semantic = "branching"
                if np.any(np.logical_and(candidate_semantic == cnode_semantic,
                                         candidate_parent_label == clabel)):
                    # same semantic and parent
                    pnode = candidates[np.logical_and(
                        candidate_semantic == cnode_semantic,
                        candidate_parent_label == clabel)][0]
                    match = True
                elif np.any(candidate_parent_label == clabel):
                    # same parent
                    pnode = candidates[candidate_parent_label == clabel][0]
                    match = True
                elif np.any(np.logical_and(candidate_parent_label == None,
                                      candidate_semantic == cnode_semantic)):
                    # same semantic, but parent None
                    pnode = candidates[np.logical_and(
                        candidate_parent_label == None,
                        candidate_semantic == cnode_semantic)][0]
                    match = True
                elif np.any(candidate_parent_label == None):
                    # parent None
                    pnode = candidates[candidate_parent_label == None][0]
                    match = True
                else:
                    # candidate parents are already matched to other tree
                    # check if node of other tree is close by, if not match
                    for candidate, c_parent_label in zip(candidates, candidate_parent_label):
                        # TODO: put query in def
                        gt_dists, gt_candidates = gt_kdtree.query(
                            target_positions[candidate], p=2,
                            distance_upper_bound=10
                        )
                        if type(gt_candidates) in [int, np.int64]:
                            gt_candidates = [gt_candidates]
                        gt_candidate_labels = [source.nodes[
                            np.array(source.nodes)[gtc]]["tree_label"] for gtc in gt_candidates]
                        if np.any(np.array(gt_candidate_labels) == c_parent_label):
                            continue
                        else:
                            match = True
                            pnode = candidate
                            break
                    if match == False:
                        logger.debug("for gt node %i with label %i, no matching candidate: %s, %s" % (
                            cnode, clabel, candidates, candidate_parent_label))

            if match:
                match_dict[cnode] = pnode
                pred_matched_labels[pnode] = source.nodes[
                    cnode]["tree_label"]

            if source.out_degree(cnode) > 0:
                next_nodes += list(source.successors(cnode))
            if len(next_nodes) > 0:
                cnode = next_nodes.pop()
            else:
                break
    # TODO: move up to "main matching def"
    if visualize:
        visualize_matching(source, target, match_dict)

    logger.info("time for node matching: %f sec." % (time.time() - start_time))

    return match_dict



def get_candidates_query(kdtree, position, all_nodes, matched_nodes, k=1, distance_upper_bound=10):
    dists, candidates = kdtree.query(position, k=k, p=2, distance_upper_bound=distance_upper_bound)

    if isinstance(candidates, (int, np.integer)):
        candidates = [candidates]
    if isinstance(dists, float):
        dists = [dists]

    candidates = [
        all_nodes[idx]
        for dist, idx in zip(dists, candidates)
        if not np.isinf(dist) and idx < len(all_nodes)
    ]
    unmatched_candidates = [node for node in candidates if node not in matched_nodes]
    return unmatched_candidates


def get_candidates_ball(kdtree, position, all_nodes, matched_nodes, radius=1.5):
    candidates = kdtree.query_ball_point(position, r=radius, p=2)

    if isinstance(candidates, (int, np.integer)):
        candidates = [candidates]

    candidates = [all_nodes[idx] for idx in candidates]
    unmatched_candidates = [node for node in candidates if node not in matched_nodes]
    target_debug_pos = np.array([97.37256, 77.50172, 111.5607])
    if np.allclose(position, target_debug_pos, atol=1e-4):
        print(f"\n[DEBUG] Query at {position} (approx match to target)")
        print(f"  Raw candidates: {candidates}")
        print(f"  Unmatched candidates: {unmatched_candidates}")
    # open("debug.txt", "a").write(
    #     f"Query at {position} → raw: {candidates}, unmatched: {unmatched_candidates}\n"
    # )
    return unmatched_candidates


def find_neighbor_tree_label(graph, node, pred_matched_labels, semantic_map, max_semantic_hops=5):
    """
    BFS over the graph, counting semantic hops only through 'branch' or 'end' nodes.
    Returns the first encountered matched label within max_semantic_hops.
    """
    visited = set()
    queue = [(node, 0)]

    while queue:
        current, semantic_hops = queue.pop(0)

        if semantic_hops > max_semantic_hops:
            break

        if current in visited:
            continue
        visited.add(current)

        if current != node and current in pred_matched_labels:
            # open("debug.txt", "a").write(
            #     f"Returning label '{pred_matched_labels[current]}' from neighbor {current} (start node: {node})\n")

            return pred_matched_labels[current]

        for neighbor in graph.neighbors(current):
            #open("debug.txt", "a").write(f"Current node: {current}, Neighbors: {list(graph.neighbors(current))}\n")
            if neighbor in visited:
                continue

            if semantic_map.get(neighbor) in {"branch", "end"}:
                queue.append((neighbor, semantic_hops + 1))
            else:
                queue.append((neighbor, semantic_hops))

    return None



def match_hierarchical(source, target, visualize=True, candidate_selection="query", max_distance=10):
    start_time = time.time()
    match_dict = {}
    pred_matched_labels = {}
    pred_matched_nodes = set()
    gt_matched_nodes = set()

    target = target.to_undirected()

    # Find GT (source) roots – source is directed
    gt_roots = [n for n, d in source.in_degree() if d == 0]
    print(f"gt_roots{gt_roots}")

    # Label connected components (trees)
    source, gt_labels = label_connected_components(source)
    #target, pred_labels = label_connected_components(target)

    all_target_nodes = list(target.nodes)

    # Get positions of nodes
    source_positions = nx.get_node_attributes(source, 'coord')
    target_positions = nx.get_node_attributes(target, 'coord')

    if not source_positions or not target_positions:
        raise ValueError("Both source and target graphs must have 'coord' attributes for nodes.")

    # Build KD-Trees for spatial search
    pred_kdtree = KDTree(np.array(list(target_positions.values())))
    gt_kdtree = KDTree(np.array(list(source_positions.values())))

    # make it deterministic by startig at root with min distance to its nearest candidate
    gt_root_distances = []
    for root in gt_roots:
        if candidate_selection == "query":
            unmatched_candidates = get_candidates_query(
                pred_kdtree, source_positions[root], all_target_nodes, set(),
                distance_upper_bound=max_distance
            )
        elif candidate_selection == "ball":
            unmatched_candidates = get_candidates_ball(
                pred_kdtree, source_positions[root], all_target_nodes, set(),
                radius=max_distance
            )
        else:
            raise ValueError(f"Unknown candidate_selection: {candidate_selection}")
        if unmatched_candidates:
            distances = [
                np.linalg.norm(np.array(target_positions[n]) - np.array(source_positions[root]))
                for n in unmatched_candidates
            ]
            min_distance = min(distances)
        else:
            min_distance = float("inf")

        gt_root_distances.append((root, min_distance))

    # Sort by ascending distance
    gt_roots = [r for r, _ in sorted(gt_root_distances, key=lambda x: x[1])]
    print(f"gt_roots{gt_roots}")

    # Assign semantics to GT nodes
    gt_semantics = {}
    for node in source.nodes:
        in_deg = source.in_degree(node)
        out_deg = source.out_degree(node)
        if in_deg == 0:
            gt_semantics[node] = "root"
        elif out_deg == 0:
            gt_semantics[node] = "end"
        elif in_deg == 1 and out_deg == 1:
            gt_semantics[node] = "segment"
        elif out_deg > 1:
            gt_semantics[node] = "branch"
        else:
            gt_semantics[node] = "other"

    target_semantics = {}
    semantic_map = {}
    for node in target.nodes:
        deg = target.degree(node)
        if deg == 1:
            target_semantics[node] = "end"
        elif deg == 2:
            target_semantics[node] = "segment"
        else:
            target_semantics[node] = "branch"



    # Iterate over each GT root
    for root in gt_roots:
        clabel = gt_labels[root]
        print(f'root position{source_positions[root]}')

        ### STEP 1: Match the root node

        if candidate_selection == "query":
            unmatched_candidates = get_candidates_query(
                pred_kdtree, source_positions[root], all_target_nodes, pred_matched_nodes,
                distance_upper_bound=max_distance
            )
        elif candidate_selection == "ball":
            unmatched_candidates = get_candidates_ball(
                pred_kdtree, source_positions[root], all_target_nodes, pred_matched_nodes,
                radius=max_distance
            )
        else:
            raise ValueError(f"Unknown candidate_selection: {candidate_selection}")

        target_debug_pos = np.array([97.37256, 77.50172, 111.5607])
        if np.allclose(source_positions[root], target_debug_pos, atol=1e-4):
            print(f"[DEBUG] Step 1 — Matching GT root node {root} at {source_positions[root]}")

        if unmatched_candidates:
            endpoint_candidates = [n for n in unmatched_candidates if target_semantics[n] == "end"]

            if endpoint_candidates:
                root_coord = np.array(source_positions[root])
                distances = {
                    n: np.linalg.norm(np.array(target_positions[n]) - root_coord)
                    for n in endpoint_candidates
                }
                best_match = min(distances, key=distances.get)
                match_dict[root] = best_match
                pred_matched_labels[best_match] = clabel
                pred_matched_nodes.add(best_match)
                gt_matched_nodes.add(root)
            # else:
            #     #continue
            #     print(f"No endpoint match found for GT root {root}")

        ### STEP 2: Traverse tree and match branch points only
        next_nodes = list(source.successors(root))

        while next_nodes:
            cnode = next_nodes.pop()
            next_nodes += list(source.successors(cnode))

            if gt_semantics.get(cnode) == "segment":
                continue

            cnode_semantic = gt_semantics.get(cnode)
            clabel = gt_labels[cnode]
            if candidate_selection == "query":
                unmatched_candidates = get_candidates_query(
                    pred_kdtree, source_positions[cnode], all_target_nodes, pred_matched_nodes,
                    distance_upper_bound=max_distance
                )
            elif candidate_selection == "ball":
                unmatched_candidates = get_candidates_ball(
                    pred_kdtree, source_positions[cnode], all_target_nodes, pred_matched_nodes,
                    radius=max_distance
                )
            else:
                raise ValueError(f"Unknown candidate_selection: {candidate_selection}")

            if unmatched_candidates:
                #semantic_map = {}
                neighbor_label_map = {}
                for node in unmatched_candidates:
                    neighbor_label = find_neighbor_tree_label(target, node, pred_matched_labels, target_semantics)

                    neighbor_label_map[node] = neighbor_label


                match_both = []
                match_neighbor_only = []
                match_semantic_only = []

                for node in unmatched_candidates:
                    semantic = target_semantics[node]
                    neighbor_label = neighbor_label_map[node]

                    if semantic == cnode_semantic and neighbor_label == clabel:
                        match_both.append(node)
                    elif semantic != cnode_semantic and neighbor_label == clabel:
                        match_neighbor_only.append(node)
                    elif semantic == cnode_semantic and neighbor_label is None:
                        match_semantic_only.append(node)

                target_debug_pos = np.array([97.37256, 77.50172, 111.5607])
                if np.allclose(source_positions[cnode], target_debug_pos, atol=1e-4):
                    print(f"[DEBUG] Step 2 — Matching GT branch node {cnode} at {source_positions[cnode]}")
                    print(f"  match_both: {match_both}")
                    print(f"  match_neighbor_only: {match_neighbor_only}")
                    print(f"  match_semantic_only: {match_semantic_only}")


                selected = None
                if match_both:
                    selected = min(
                        match_both,
                        key=lambda n: np.linalg.norm(np.array(target_positions[n]) - np.array(source_positions[cnode]))
                    )
                elif match_neighbor_only:
                    selected = min(
                        match_neighbor_only,
                        key=lambda n: np.linalg.norm(np.array(target_positions[n]) - np.array(source_positions[cnode]))
                    )
                elif match_semantic_only:
                    selected = min(
                        match_semantic_only,
                        key=lambda n: np.linalg.norm(np.array(target_positions[n]) - np.array(source_positions[cnode]))
                    )
                if selected is not None:
                    match_dict[cnode] = selected
                    pred_matched_labels[selected] = clabel
                    pred_matched_nodes.add(selected)
                    gt_matched_nodes.add(cnode)
                # else:
                #     #continue
                #     print(f"No valid match for GT branch node {cnode}")


    ### STEP 3: Match unmatched target branch/end nodes to GT ###
    for pnode in target.nodes:
        if pnode in pred_matched_nodes:
            continue  # already matched

        if target_semantics.get(pnode) == "segment":
            continue

        # Step b: Try to find neighbor label from matched nodes
        neighbor_label = find_neighbor_tree_label(target, pnode, pred_matched_labels, target_semantics)
        if neighbor_label is None:
            continue

        # Step c: Query nearby GT nodes
        all_gt_nodes = list(source.nodes)
        if candidate_selection == "query":
            unmatched_gt_candidates = get_candidates_query(
                gt_kdtree, target_positions[pnode], all_gt_nodes, gt_matched_nodes,
                distance_upper_bound=max_distance
            )
        elif candidate_selection == "ball":
            unmatched_gt_candidates = get_candidates_ball(
                gt_kdtree, target_positions[pnode], all_gt_nodes, gt_matched_nodes,
                radius=max_distance
            )
        else:
            raise ValueError(f"Unknown candidate_selection: {candidate_selection}")

        if not unmatched_gt_candidates:
            continue

        # Step d: Build maps
        match_both = []
        match_neighbor_only = []
        match_semantic_only = []

        for gnode in unmatched_gt_candidates:
            g_semantic = gt_semantics.get(gnode)
            g_label = gt_labels[gnode]

            if g_semantic == target_semantics[pnode] and g_label == neighbor_label:
                match_both.append(gnode)
            elif g_semantic != target_semantics[pnode] and g_label == neighbor_label:
                match_neighbor_only.append(gnode)
            elif g_semantic == target_semantics[pnode] and g_label is None:
                match_semantic_only.append(gnode)

        target_debug_pos = np.array([97.37256, 77.50172, 111.5607])
        if np.allclose(target_positions[pnode], target_debug_pos, atol=1e-4):
            print(f"  match_both: {match_both}")
            print(f"  match_neighbor_only: {match_neighbor_only}")
            print(f"  match_semantic_only: {match_semantic_only}")

        # Step e: Choose the best match
        selected = None
        if match_both:
            selected = min(
                match_both,
                key=lambda n: np.linalg.norm(np.array(source_positions[n]) - np.array(target_positions[pnode]))
            )
        elif match_neighbor_only:
            selected = min(
                match_neighbor_only,
                key=lambda n: np.linalg.norm(np.array(source_positions[n]) - np.array(target_positions[pnode]))
            )
        elif match_semantic_only:
            selected = min(
                match_semantic_only,
                key=lambda n: np.linalg.norm(np.array(source_positions[n]) - np.array(target_positions[pnode]))
            )

        # Step f: Register the match
        if selected is not None:
            match_dict[selected] = pnode  # flipped direction
            pred_matched_labels[pnode] = gt_labels[selected]
            pred_matched_nodes.add(pnode)
            gt_matched_nodes.add(selected)
        # else:
        #     #continue
        #     print(f"No valid GT match for predicted node {pnode}")


    ### STEP 4: Match remaining unmatched GT nodes to unmatched prediction nodes ###
    for gnode in source.nodes:
        if gnode in gt_matched_nodes:
            continue

        cnode_semantic = gt_semantics.get(gnode)
        clabel = gt_labels[gnode]
        if candidate_selection == "query":
            unmatched_pred_candidates = get_candidates_query(
                pred_kdtree, source_positions[gnode], all_target_nodes, pred_matched_nodes,
                distance_upper_bound=max_distance
            )
        elif candidate_selection == "ball":
            unmatched_pred_candidates = get_candidates_ball(
                pred_kdtree, source_positions[gnode], all_target_nodes, pred_matched_nodes,
                radius=max_distance
            )
        else:
            raise ValueError(f"Unknown candidate_selection: {candidate_selection}")

        if not unmatched_pred_candidates:
            continue

        #semantic_map = {}
        neighbor_label_map = {}
        for node in unmatched_pred_candidates:
            neighbor_label = find_neighbor_tree_label(target, node, pred_matched_labels, target_semantics)
            neighbor_label_map[node] = neighbor_label

        match_both = []
        match_neighbor_only = []
        match_semantic_only = []

        for node in unmatched_pred_candidates:
            semantic = target_semantics[node]
            neighbor_label = neighbor_label_map[node]

            if semantic == cnode_semantic and neighbor_label == clabel:
                match_both.append(node)
            elif semantic != cnode_semantic and neighbor_label == clabel:
                match_neighbor_only.append(node)
            elif semantic == cnode_semantic and neighbor_label is None:
                match_semantic_only.append(node)

        target_debug_pos = np.array([97.37256, 77.50172, 111.5607])
        if np.allclose(source_positions[gnode], target_debug_pos, atol=1e-4):
            print(f"  match_both: {match_both}")
            print(f"  match_neighbor_only: {match_neighbor_only}")
            print(f"  match_semantic_only: {match_semantic_only}")

        selected = None
        if match_both:
            selected = min(
                match_both,
                key=lambda n: np.linalg.norm(np.array(target_positions[n]) - np.array(source_positions[gnode]))
            )
        elif match_neighbor_only:
            selected = min(
                match_neighbor_only,
                key=lambda n: np.linalg.norm(np.array(target_positions[n]) - np.array(source_positions[gnode]))
            )
        elif match_semantic_only:
            selected = min(
                match_semantic_only,
                key=lambda n: np.linalg.norm(np.array(target_positions[n]) - np.array(source_positions[gnode]))
            )

        if selected is not None:
            match_dict[gnode] = selected
            pred_matched_labels[selected] = clabel
            pred_matched_nodes.add(selected)
            gt_matched_nodes.add(gnode)
        # else:
        #     #continue
        #     print(f"No match found in STEP 4 for GT node {gnode}")


    #match_dict_one_to_one = match_one_to_one(source, target)

    # if visualize:
    #     visualize_matching(source, target, match_dict, match_dict_one_to_one)

    logger.info("Time for node matching: %.2f sec" % (time.time() - start_time))
    return match_dict


def match_hungarian(source, target, unmatch_penalty, visualize=False):
    """
    Perform Hungarian node matching between two directed graphs (source, target)
    using 3D node coordinates. Allows non-assignment via dummy nodes.
    NOTE: in linajea https://www.nature.com/articles/s41587-022-01427-7 they do hungarian matching on edges

    Parameters
    ----------
    source :
        Ground-truth graph
    target :
        Predicted graph
    unmatch_penalty : float, optional
        Cost assigned to leaving a node unmatched. Larger -> fewer unmatches.
        NOTE: Set this to the threshold distance for matching.

    Returns
    -------
    dict
        Mapping {sourcet_node: target_node or None}
    """
    source_positions = nx.get_node_attributes(source, 'coord')
    source_nodes = list(source_positions.keys())
    source_coords = np.array([source_positions[node] for node in source_nodes])

    target_positions = nx.get_node_attributes(target, 'coord')
    target_nodes = list(target_positions.keys())
    target_coords = np.array([target_positions[node] for node in target_nodes])

    n, m = len(source_nodes), len(target_nodes)
    cost = np.linalg.norm(source_coords[:, None, :] - target_coords[None, :, :], axis=2)

    # Pad cost matrix to allow non-assignment
    size = n+m # NOTE: This adds additional dummy nodes to allow all nodes to be unmatched
    padded_cost = np.full((size, size), unmatch_penalty, dtype=float)
    padded_cost[:n, :m] = cost

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(padded_cost)

    # Construct mapping
    match_dict = {}
    for i, j in zip(row_ind, col_ind):
        if i < n:
            if j < m and padded_cost[i, j] < unmatch_penalty:
                match_dict[source_nodes[i]] = target_nodes[j]
            # NOTE: We do not explicitly store unmatched nodes as None
            # else:
            #     mapping[source_nodes[i]] = None

    # if visualize:
    #     visualize_matching(source, target, match_dict)

    return match_dict





