import networkx as nx
import numpy as np

from enum import Enum

class MatchingType(Enum):
    Nearest = 1
    Greedy = 2
    Within_Radius = 3


def match_graphs(source, target, matching_type):
    """
    Matches nodes from the source graph to nodes in the target graph based on the specified matching strategy.

    Parameters:
    ----------
    source : networkx.Graph
        The source graph whose nodes need to be matched.
        Nodes must have positional attributes ('x', 'y', 'z') for distance-based matching.
    target : networkx.Graph
        The target graph to which source nodes will be matched.
        Nodes must have positional attributes ('x', 'y', 'z') for distance-based matching.
    matching_type : MatchingType
        The type of matching to be performed. Supported types are:
        - MatchingType.Nearest: Matches each source node to the nearest target node. Many-to-one matching allowed.
        - MatchingType.Greedy: Matches each source node to the nearest target node in a greedy fashion. One-to-one matching.
        - MatchingType.Within_Radius: Matches each source node to the nearest target node within a specified radius.
          Many-to-one matching is allowed, but some source nodes may remain unmatched.

    Returns:
    -------
    match_dict : dict
        A dictionary representing the matching result. Keys are source graph node IDs, and values are target graph node IDs.
        If a source node is not matched (e.g., in Within_Radius), it will not appear in the dictionary.
    """
    assert matching_type in MatchingType, f"Unknown matching type: {matching_type}"

    match_dict = {}
    if matching_type == MatchingType.Nearest:
        # Matches a source node to the closest target node. Can be many-to-one. Every source node is matched.
        source_positions = nx.get_node_attributes(source, 'coord')
        target_positions = nx.get_node_attributes(target, 'coord')
        if not source_positions or not target_positions:
            raise ValueError("Both source and target graphs must have 'pos' attributes for nodes.")

        for source_node, source_pos in source_positions.items():
            # Find the closest target node
            closest_target_node = min(
                target_positions.keys(),
                key=lambda target_node: np.linalg.norm(np.array(source_pos) - np.array(target_positions[target_node]))
            )
            match_dict[source_node] = closest_target_node
    if matching_type == MatchingType.Greedy:
        # Matches a source node to closest target node in a greedy fashion. Is one-to-one. Every source node is matched.
        # TODO
        pass
    if matching_type == MatchingType.Within_Radius:
        # Matches a source node to closest target node within a radius. Can be many-to-one. Source nodes may not be matched.
        # TODO
        pass
    return match_dict

