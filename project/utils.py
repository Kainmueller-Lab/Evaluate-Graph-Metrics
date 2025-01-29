import networkx as nx
from networkx.generators.atlas import graph_atlas
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def betti_1_number(G):
    """
    Computes the Betti 1 number (number of independent cycles) of a graph.
    
    Parameters:
    ----------
    G : networkx.Graph
        The input graph (undirected or directed).
    
    Returns:
    -------
    int
        The Betti 1 number of the graph.
    """
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    num_components = nx.number_connected_components(G)
    
    # Betti 1 number formula
    return num_edges - num_nodes + num_components


def compute_edge_metrics(G, H, matching):
    """
    Computes common edges (TP), false negatives (FN), and false positives (FP).
    Returns a dictionary with TP, FN, and FP values.
    """
    G_edges = set(G.edges)
    H_edges = set([(matching[e[0]], matching[e[1]]) for e in H.edges if e[0] in matching and e[1] in matching])

    false_negatives = G_edges - H_edges  # False negatives
    false_positives = H_edges - G_edges  # False positives
    common_edges = G_edges & H_edges  # True positives

    false_merges = set()
    for fp_edge in false_positives:
        u, v =fp_edge
        if u in G and v in G and not (nx.has_path(G, source=u, target=v) or nx.has_path(G, source=v, target=u)):
            false_merges.add(fp_edge)

    H_copy = H.copy()
    H_copy.remove_edges_from(false_merges)
    components_before = nx.number_weakly_connected_components(H)
    components_after = nx.number_weakly_connected_components(H_copy)
    false_splits = components_after - components_before

    #print("false_merges:", len(false_merges))

    metrics = {
        "TP": len(common_edges),
        "FN": len(false_negatives),
        "FP": len(false_positives),
        "FM": len(false_merges),
        "FS": false_splits
    }
    return metrics


def create_graph_from_swc(fn, scale_factor=None, offset=None):
    """
        Reads swc file and converts it to target graph structure.

        Parameters:
        ----------
        str
            The path to the input swc file.

        Returns:
        -------
        G : networkx.Graph
            The undirected input graph.
        """
    # create graph
    graph = nx.Graph()
    # open swc file
    f = open(fn, "r")
    if scale_factor is None:
        scale_factor = [1, 1, 1]
    # create graph node for each swc line
    for line in f:
        line = line.replace("\n", "")
        if line == "":
            continue
        if line[0] == "#":
            continue
        if line[0] == " ":
            continue
        split = line.split(" ")
        node_id = int(split[0])
        node_type = int(float(split[1]))
        x = float(split[2])*scale_factor[0]
        y = float(split[3])*scale_factor[1]
        z = float(split[4])*scale_factor[2]
        if offset is not None:
            x = x + offset[0]
            y = y + offset[1]
            z = z + offset[2]

        radius = float(split[5])*scale_factor[2]
        parent_id = int(split[6])

        graph.add_node(
            node_id,
            coord = np.array([x, y, z]),
            #parent_id=parent_id,
            #node_type=node_type,
            radius=radius
        )
        if parent_id != -1:
            graph.add_edge(node_id, parent_id)


    f.close()
    return graph


def create_directed_graph(graph, root_type=None, roots=None):
    directed_graph = None
    if root_type == "manual":
        raise NotImplementedError
    elif root_type == "largest_radius":
        # convert graph to directed_graph and remove all edges
        #directed_graph = nx.DiGraph()
        directed_graph = graph.to_directed()
        to_remove = list(directed_graph.edges)
        directed_graph.remove_edges_from(to_remove)

        # take for each connected component the end point with largest radius as root
        connected_components = list(nx.connected_components(graph))
        
        # iterate through all connected components
        for cc in connected_components:
            cc_graph = graph.subgraph(cc)
            if len(cc_graph.nodes) == 1:
                continue
            
            # get end nodes
            degree = cc_graph.degree()
            terminal_idx = []
            for k, item in degree:
                if item == 1:
                    terminal_idx.append(k)
            
            # get radius of end end nodes
            radius = nx.get_node_attributes(cc_graph, 'radius')
            terminal_radius = np.array([radius[k] for k in terminal_idx])
            max_rad_idx = np.argmax(terminal_radius)
            root_idx = terminal_idx[max_rad_idx]
            max_rad = terminal_radius[max_rad_idx]

            # starting at node with largest radius and traverse through
            # connected component, add edges to directed graph
            visited = []
            stack = [root_idx]
            while len(stack) > 0:
                c_node = stack.pop()
                neighbors = cc_graph.neighbors(c_node)
                for n in neighbors:
                    if n not in visited:
                        stack.append(n)
                        directed_graph.add_edge(c_node, n)
                visited.append(c_node)
    else:
        raise NotImplementedError
    return directed_graph


def read_graph_from_file(filename):
    graph = None
    if filename.endswith(".swc"):
        graph = create_graph_from_swc(filename)
    else:
        raise NotImplementedError
    # convert unordered graph to ordered graph here?
    # take endpoint with largest radius as root
    graph = create_directed_graph(graph, root_type="largest_radius")
    return graph


def label_connected_components(graph):
    
    # get connected components
    connected_components = list(
        nx.connected_components(graph.to_undirected(as_view=True)))
    lbl = 0
    # iterate through all connected components and assign tree label
    for cc in connected_components:
        cc_graph = graph.subgraph(cc)
        for n in cc_graph.nodes:
            graph.nodes[n]["tree_label"] = lbl
        lbl += 1
    node_labels = nx.get_node_attributes(graph, "tree_label")
         
    return graph, node_labels


def visualize_matching(gt_graph, pred_graph, matched_dict, title="Node Matching"):
    gt_pos = np.array(list(nx.get_node_attributes(gt_graph, 'coord').values()))
    pred_pos = np.array(list(nx.get_node_attributes(
        pred_graph, 'coord').values()))
    gt_matches = ["TP" if n in matched_dict.keys() else "FN" for n in gt_graph.nodes]
    pred_matches = ["TP" if n in matched_dict.values() else "FP" for n in pred_graph.nodes]

    data = {
        "x": list(gt_pos[:,0]) + list(pred_pos[:,0]),
        "y": list(gt_pos[:, 1]) + list(pred_pos[:,1]),
        "z": list(gt_pos[:,2]) + list(pred_pos[:,2]),
        "graph": ["gt",] * len(gt_pos) + ["pred",] * len(pred_pos),
        "match": gt_matches + pred_matches
    }
    df = pd.DataFrame(data=data)
    print(df)

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="match", symbol="graph",
                       size_max=4)
    fig.update_traces(marker={'size': 2})
    fig.show()

