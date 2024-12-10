import networkx as nx
import numpy as np

from enum import Enum

class MatchingType(Enum):
    Nearest = 1
    Nearest_Within_Radius = 2
    Greedy = 3
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
            break  # Stop if there are no unmatched target nodes left

        # Find the closest unmatched target node
        closest_target_node = min(
            unmatched_targets,
            key=lambda target_node: np.linalg.norm(np.array(source_pos) - np.array(target_positions[target_node]))
        )
        if np.linalg.norm(np.array(source_pos) - np.array(target_positions[closest_target_node])) <= source_radii[source_node]:
            match_dict[source_node] = closest_target_node
            unmatched_targets.remove(closest_target_node)  # Mark the target as matched

    return match_dict
