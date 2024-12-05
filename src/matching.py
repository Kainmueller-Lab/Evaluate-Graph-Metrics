import networkx as nx

from enum import Enum

class MatchingType(Enum):
    Nearest = 1
    Greedy = 2
    WITHIN_RADIUS = 3

def match_graphs(source, target, matching_type):
    match_dict = {}

    if matching_type == MatchingType.Nearest:
        pass
    
    return match_dict

