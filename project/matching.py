import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from enum import Enum

from .utils import label_connected_components, visualize_matching


class MatchingType(Enum):
    Nearest = 1
    Nearest_Within_Radius = 2
    Greedy = 3
    Greedy_with_parent = 4
    Hierarchical = 5
    # TODO: Graph matching solution? Could improve this, but unclear if efficient.


def match_graphs(source, target, matching_type):
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

    Returns:
    -------
    match_dict : dict
        A dictionary representing the matching result. Keys are source graph node IDs, and values are target graph node IDs.
        If a source node is not matched (e.g., in Greedy), it will not appear in the dictionary.
    """
    assert matching_type in MatchingType, f"Unknown matching type: {matching_type}"

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
    if matching_type == MatchingType.Hierarchical:
        return match_hierarchical(source=source, target=target)
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

    for source_node, source_pos in source_positions.items():
        # Find the closest target node
        closest_target_node = min(
            target_positions.keys(),
            key=lambda target_node: np.linalg.norm(np.array(source_pos) - np.array(target_positions[target_node]))
        )
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


def match_hierarchical(source, target):
    
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
    # build pred kd tree
    pred_kdtree = KDTree(np.array(list(target_positions.values())))

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
            # TODO: nearest neighbors should also work if no radius is given 
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
            candidates = np.array(candidates)
            candidates = candidates + 1
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
                    parent_id = list(target.predecessors(pnode))[0]
                    if parent_id in pred_matched_labels:
                        candidate_parent_label.append(pred_matched_labels[parent_id])
                    else:
                        for i in range(3):
                            ancester_id = list(target.predecessors(parent_id))
                            if len(ancester_id) > 0:
                                ancester_id = ancester_id[0]
                            else:
                                break
                            if ancester_id in pred_matched_labels:
                                candidate_parent_label.append(pred_matched_labels[ancester_id])
                                break
                            else:
                                parent_id = ancester_id
                            i += 1
                        if len(candidate_semantic) > len(candidate_parent_label):
                            candidate_parent_label.append(None)
                            
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
                    pnode = candidates[np.logical_and(
                        candidate_parent_label == None,
                        candidate_semantic == cnode_semantic)][0]
                    match = True
                elif np.any(candidate_parent_label == None):
                    pnode = candidates[candidate_parent_label == None][0]
                    match = True
                else:
                    print("NOT MATCHED, please check")

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
    unmatched_source = set(source.nodes)-set(match_dict.keys())
    unmatched_target = set(target.nodes) - set(match_dict.values())
    visualize_matching(source, target, match_dict)

    return match_dict, unmatched_source, unmatched_target

