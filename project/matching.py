import networkx as nx
import numpy as np

from enum import Enum

class MatchingType(Enum):
    Nearest = 1
    Nearest_Within_Radius = 2
    Greedy = 3
    Greedy_with_parent = 4
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

import networkx as nx
import numpy as np

def match_greedy_parent(source, target):
    """
    Matches nodes in the source graph to nodes in the target graph based on spherical proximity
    and parent relationship constraints. Also prints unmatched nodes in both graphs.

    Parameters:
    ----------
    source : networkx.Graph
        The source graph with 'coord' and 'radius' node attributes.
    target : networkx.Graph
        The target graph with 'coord' and 'radius' node attributes.

    Returns:
    -------
    match_dict : dict
        A dictionary mapping nodes in the source graph to nodes in the target graph.
    unmatched_source : set
        A set of nodes in the source graph that could not be matched.
    unmatched_target : set
        A set of nodes in the target graph that were not matched.
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
