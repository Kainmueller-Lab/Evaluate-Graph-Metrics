import sys
import logging

import networkx as nx
from networkx.generators.atlas import graph_atlas
import numpy as np
import napari
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdb

logger = logging.getLogger(__name__)


def betti_0_number(G, H):
    return np.abs(nx.number_connected_components(G.to_undirected()) - nx.number_connected_components(H.to_undirected()))


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


def compute_node_metrics(G, H, matched):
    G_nodes_matched = set(n for n in G.nodes if n in matched.keys())
    H_nodes_matched = set(n for n in H.nodes if n in matched.values())

    FN = set(G.nodes) - G_nodes_matched  # False Negatives: Nodes in GT that are not matched.
    FP = set(H.nodes) - H_nodes_matched  # False Positives: Nodes in Pred that are not matched.

    num_TP = len(G_nodes_matched)
    num_FN = len(FN)
    num_FP = len(FP)

    precision = num_TP / (num_TP + num_FP) if (num_TP + num_FP) > 0 else 0
    recall = num_TP / (num_TP + num_FN) if (num_TP + num_FN) > 0 else 0
    f1 = (2 * num_TP) / (2 * num_TP + num_FN + num_FP) if (2 * num_TP + num_FN + num_FP) > 0 else 0

    metrics = {
        "TP_nodes": num_TP,
        "FN_nodes": num_FN,
        "FP_nodes": num_FP,
        "precision_nodes": precision,
        "recall_nodes": recall,
        "f1_nodes": f1
    }
    return metrics


import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px


def visualization(gt_graph, pred_graph, false_merges, false_splits, matched_dict, title="Edge Visualization"):
    edge_data = []

    color_mapping = {
        "GT": "blue",
        "Pred": "black",
        "Matched": "green",
        "False Merge": "red"
    }

    for node1, node2 in gt_graph.edges:
        pos1, pos2 = gt_graph.nodes[node1]['coord'], gt_graph.nodes[node2]['coord']
        edge_data.append({"x": pos1[0], "y": pos1[1], "z": pos1[2], "type": "GT"})
        edge_data.append({"x": pos2[0], "y": pos2[1], "z": pos2[2], "type": "GT"})
        edge_data.append({"x": None, "y": None, "z": None, "type": "GT"})

    for node1, node2 in pred_graph.edges:
        pos1, pos2 = pred_graph.nodes[node1]['coord'], pred_graph.nodes[node2]['coord']
        edge_data.append({"x": pos1[0], "y": pos1[1], "z": pos1[2], "type": "Pred"})
        edge_data.append({"x": pos2[0], "y": pos2[1], "z": pos2[2], "type": "Pred"})
        edge_data.append({"x": None, "y": None, "z": None, "type": "Pred"})

    for gt_node, pred_node in matched_dict.items():
        pos_gt, pos_pred = gt_graph.nodes[gt_node]['coord'], pred_graph.nodes[pred_node]['coord']
        edge_data.append({"x": pos_gt[0], "y": pos_gt[1], "z": pos_gt[2], "type": "Matched"})
        edge_data.append({"x": pos_pred[0], "y": pos_pred[1], "z": pos_pred[2], "type": "Matched"})
        edge_data.append({"x": None, "y": None, "z": None, "type": "Matched"})

    for fm in false_merges:
        pos1, pos2 = pred_graph.nodes[fm[0]]['coord'], pred_graph.nodes[fm[1]]['coord']
        edge_data.append({"x": pos1[0], "y": pos1[1], "z": pos1[2], "type": "False Merge"})
        edge_data.append({"x": pos2[0], "y": pos2[1], "z": pos2[2], "type": "False Merge"})
        edge_data.append({"x": None, "y": None, "z": None, "type": "False Merge"})

    df = pd.DataFrame(edge_data)

    fig = px.line_3d(df, x="x", y="y", z="z", color="type", title=title,
                     color_discrete_map=color_mapping)
    fig.show()


def compute_edge_metrics(G, H, matched, visualize=True):
    """
    Computes common edges (TP), false negatives (FN), and false positives (FP).
    Returns a dictionary with TP, FN, and FP values.
    """
    # True Positives are edges in GT that are matched to an existing edge in Pred.
    TP = [(v, w) for (v, w) in G.edges if
          v in matched.keys() and w in matched.keys() and H.has_edge(matched[v], matched[w])]
    # False Negatives are edges in GT that are either (1) matched to a missing edge in Pred or (2) are unmatched (i.e. at least one node is unmatched).
    G_edges_matched_but_H_edge_missing = [(v, w) for (v, w) in G.edges if
                                          v in matched.keys() and w in matched.keys() and not H.has_edge(matched[v],
                                                                                                         matched[w])]
    G_edges_unmatched = [(v, w) for (v, w) in G.edges if v not in matched.keys() or w not in matched.keys()]
    FN = G_edges_matched_but_H_edge_missing + G_edges_unmatched
    # False Positives are edges in Pred that are either (1) matched to a missing edge in GT or (2) are unmatched (i.e. at least one node is unmatched).
    match_in_G = lambda x: next((k for k, v in matched.items() if v == x), None)
    H_edges_matched_but_G_edge_missing = [(y, z) for (y, z) in H.edges if
                                          y in matched.values() and z in matched.values() and not G.has_edge(
                                              match_in_G(y), match_in_G(z))]
    H_edges_unmatched = [(y, z) for (y, z) in H.edges if y not in matched.values() or z not in matched.values()]
    FP = H_edges_matched_but_G_edge_missing + H_edges_unmatched

    num_FN = len(FN)
    #print(FN)
    num_FP = len(FP)
    num_TP = len(TP)

    G_components = {frozenset(comp) for comp in nx.weakly_connected_components(G)}
    false_merges_intra_component = set()
    false_merges_inter_component = set()
    for fp_edge in FP:
        u, v = fp_edge
        x = next((s for s, t in matched.items() if t == u), None)
        y = next((s for s, t in matched.items() if t == v), None)
        if H.has_edge(u, v):
            # if H.degree(u) > 1 and H.degree(v) > 1:
            if G.has_node(x) and G.has_node(y):
                # if G.degree(x) > 1 and G.degree(y) > 1:
                #
                if not (nx.has_path(G, source=x, target=y) or nx.has_path(G, source=y, target=x)):
                    same_component = any(x in comp and y in comp for comp in G_components)
                    if same_component:
                        false_merges_intra_component.add(fp_edge)
                    else:
                        false_merges_inter_component.add(fp_edge)

    total_false_merges = false_merges_intra_component | false_merges_inter_component
    # print(total_false_merges)
    # print(len(total_false_merges))
    # print(G.edges)
    # print(H.edges)
    # print(matched)
    H_copy = H.copy()
    H_copy.remove_edges_from(total_false_merges)
    false_splits = np.abs(nx.number_weakly_connected_components(G) - nx.number_weakly_connected_components(H_copy))

    if visualize == True:
        visualization(G, H, total_false_merges, false_splits, matched)

    # print("false_merges:", len(false_merges))

    precision = num_TP / float(num_TP + num_FP) if num_TP + num_FP > 0 else 0
    recall = num_TP / float(num_TP + num_FN) if num_TP + num_FN > 0 else 0
    f1 = 2 * num_TP / float(2 * num_TP + num_FN + num_FP) \
        if 2 * num_TP + num_FN + num_FP > 0 else 0

    metrics = {
        "TP_edges": num_TP,
        "FN_edges": num_FN,
        "FP_edges": num_FP,
        "FM": len(total_false_merges),
        "relative_FM": len(total_false_merges) / len(H.edges()),
        "FS": false_splits,
        "relative_FS": false_splits / len(G.edges()),
        "precision_edges": precision,
        "recall_edges": recall,
        "f1_edges": f1
    }
    return metrics


def compute_edge_metrics_old(G, H, matched, visualize=True):
    """
    Computes common edges (TP), false negatives (FN), and false positives (FP).
    Returns a dictionary with TP, FN, and FP values.
    """

    G_edges_matched = set([(e[0], e[1]) for e in G.edges if e[0] in matched and e[1] in matched])
    FN = G.edges() - G_edges_matched

    H_nodes_matched = np.unique(list(matched.values()))

    H_edges_matched = set([(e[0], e[1]) for e in H.edges \
                           if e[0] in H_nodes_matched and e[1] in H_nodes_matched])

    FP = H.edges() - H_edges_matched
    logger.debug("len matched edges for G and H: %i / %i" % (
        len(G_edges_matched), len(H_edges_matched)))

    num_FN = len(FN)
    #print(FN)
    num_FP = len(FP)
    num_TP = len(G_edges_matched)

    G_components = {frozenset(comp) for comp in nx.weakly_connected_components(G)}
    false_merges_intra_component = set()
    false_merges_inter_component = set()
    for fp_edge in FP:
        u, v = fp_edge
        x = next((s for s, t in matched.items() if t == u), None)
        y = next((s for s, t in matched.items() if t == v), None)
        if H.has_edge(u, v):
            # if H.degree(u) > 1 and H.degree(v) > 1:
            if G.has_node(x) and G.has_node(y):
                # if G.degree(x) > 1 and G.degree(y) > 1:
                #
                if not (nx.has_path(G, source=x, target=y) or nx.has_path(G, source=y, target=x)):
                    same_component = any(x in comp and y in comp for comp in G_components)
                    if same_component:
                        false_merges_intra_component.add(fp_edge)
                    else:
                        false_merges_inter_component.add(fp_edge)

    total_false_merges = false_merges_intra_component | false_merges_inter_component
    # print(total_false_merges)
    # print(len(total_false_merges))
    # print(G.edges)
    # print(H.edges)
    # print(matched)
    H_copy = H.copy()
    H_copy.remove_edges_from(total_false_merges)
    false_splits = np.abs(nx.number_weakly_connected_components(G) - nx.number_weakly_connected_components(H_copy))

    # if visualize == True:
    #     visualization(G, H, total_false_merges, false_splits, matched)

    # print("false_merges:", len(false_merges))

    precision = num_TP / float(num_TP + num_FP) if num_TP + num_FP > 0 else 0
    recall = num_TP / float(num_TP + num_FN) if num_TP + num_FN > 0 else 0
    f1 = 2 * num_TP / float(2 * num_TP + num_FN + num_FP) \
        if 2 * num_TP + num_FN + num_FP > 0 else 0

    metrics = {
        "TP_edges": num_TP,
        "FN_edges": num_FN,
        "FP_edges": num_FP,
        "FM": len(total_false_merges),
        "FS": false_splits,
        "precision_edges": precision,
        "recall_edges": recall,
        "f1_edges": f1
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
    graph = nx.DiGraph()
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
        x = float(split[2]) * scale_factor[0]
        y = float(split[3]) * scale_factor[1]
        z = float(split[4]) * scale_factor[2]
        if offset is not None:
            x = x + offset[0]
            y = y + offset[1]
            z = z + offset[2]

        radius = float(split[5]) * scale_factor[2]
        parent_id = int(split[6])

        graph.add_node(
            node_id,
            coord=np.array([x, y, z]),
            # parent_id=parent_id,
            # node_type=node_type,
            radius=radius
        )
        if parent_id != -1:
            graph.add_edge(parent_id, node_id)

    f.close()
    #print(graph.edges())
    return graph


def create_directed_graph(graph, root_type=None, roots=None):
    directed_graph = None
    if root_type == "manual":
        raise NotImplementedError
    elif root_type == "largest_radius":
        # convert graph to directed_graph and remove all edges
        # directed_graph = nx.DiGraph()
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
                if item >= 0:
                    terminal_idx.append(k)

            # get radius of end end nodes
            radius = nx.get_node_attributes(cc_graph, 'radius')
            terminal_radius = np.array([radius[k] for k in terminal_idx])
            max_rad_idx = np.argmax(terminal_radius)
            root_idx = terminal_idx[max_rad_idx]
            #print(root_idx)
            max_rad = terminal_radius[max_rad_idx]

            # bfs_tree = nx.bfs_tree(cc_graph, root_idx)
            # directed_graph = nx.compose(directed_graph, bfs_tree)

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

    #print(directed_graph.edges())
    return directed_graph


def read_graph_from_file(filename):
    graph = None
    if filename.endswith(".swc"):
        graph = create_graph_from_swc(filename)
        # print(graph.edges())
    else:
        raise NotImplementedError
    # convert unordered graph to ordered graph here?
    # take endpoint with largest radius as root
    #graph = create_directed_graph(graph, root_type="largest_radius")

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


def visualize_matching(gt_graph, pred_graph, matched_dict, match_dict_one_to_one=None, title="Matching Differences Only"):
    fig = go.Figure()


    edge_x_gt, edge_y_gt, edge_z_gt = [], [], []
    for node1, node2 in gt_graph.edges:
        pos1 = gt_graph.nodes[node1]['coord']
        pos2 = gt_graph.nodes[node2]['coord']
        edge_x_gt.extend([pos1[0], pos2[0], None])
        edge_y_gt.extend([pos1[1], pos2[1], None])
        edge_z_gt.extend([pos1[2], pos2[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x_gt, y=edge_y_gt, z=edge_z_gt,
        mode='lines',
        line=dict(color='blue', width=2),
        hoverinfo='none',
        name='GT Edges'
    ))


    edge_x_pred, edge_y_pred, edge_z_pred = [], [], []
    for node1, node2 in pred_graph.edges:
        pos1 = pred_graph.nodes[node1]['coord']
        pos2 = pred_graph.nodes[node2]['coord']
        edge_x_pred.extend([pos1[0], pos2[0], None])
        edge_y_pred.extend([pos1[1], pos2[1], None])
        edge_z_pred.extend([pos1[2], pos2[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x_pred, y=edge_y_pred, z=edge_z_pred,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none',
        name='Pred Edges'
    ))


    if match_dict_one_to_one is not None:
        # Extra FPs: in matched_dict but not in match_dict_one_to_one
        edge_x_fp_extra, edge_y_fp_extra, edge_z_fp_extra = [], [], []
        unmatched_preds = set(matched_dict.values()) - set(match_dict_one_to_one.values())

        for gt_node, pred_node in matched_dict.items():
            if pred_node in unmatched_preds:
                pos_gt = gt_graph.nodes[gt_node]['coord']
                pos_pred = pred_graph.nodes[pred_node]['coord']
                edge_x_fp_extra.extend([pos_gt[0], pos_pred[0], None])
                edge_y_fp_extra.extend([pos_gt[1], pos_pred[1], None])
                edge_z_fp_extra.extend([pos_gt[2], pos_pred[2], None])

        fig.add_trace(go.Scatter3d(
            x=edge_x_fp_extra, y=edge_y_fp_extra, z=edge_z_fp_extra,
            mode='lines',
            line=dict(color='green', width=3),
            hoverinfo='none',
            name='Extra FPs (not in one-to-one)'
        ))

        # Extra FNs: in matched_dict but not in match_dict_one_to_one
        edge_x_fn_extra, edge_y_fn_extra, edge_z_fn_extra = [], [], []
        unmatched_gts = set(matched_dict.keys()) - set(match_dict_one_to_one.keys())

        for gt_node, pred_node in matched_dict.items():
            if gt_node in unmatched_gts:
                pos_gt = gt_graph.nodes[gt_node]['coord']
                pos_pred = pred_graph.nodes[pred_node]['coord']
                edge_x_fn_extra.extend([pos_gt[0], pos_pred[0], None])
                edge_y_fn_extra.extend([pos_gt[1], pos_pred[1], None])
                edge_z_fn_extra.extend([pos_gt[2], pos_pred[2], None])

        fig.add_trace(go.Scatter3d(
            x=edge_x_fn_extra, y=edge_y_fn_extra, z=edge_z_fn_extra,
            mode='lines',
            line=dict(color='orange', width=3),
            hoverinfo='none',
            name='Extra FNs (not in one-to-one)'
        ))

    fig.update_layout(title=title)
    fig.show()



def visualize_matching_old(gt_graph, pred_graph, matched_dict, title="Node Matching"):
    gt_pos = np.array(list(nx.get_node_attributes(gt_graph, 'coord').values()))
    pred_pos = np.array(list(nx.get_node_attributes(
        pred_graph, 'coord').values()))
    gt_matches = ["TP" if n in matched_dict.keys() else "FN" for n in gt_graph.nodes]
    pred_matches = ["TP" if n in matched_dict.values() else "FP" for n in pred_graph.nodes]

    data = {
        "x": list(gt_pos[:, 0]) + list(pred_pos[:, 0]),
        "y": list(gt_pos[:, 1]) + list(pred_pos[:, 1]),
        "z": list(gt_pos[:, 2]) + list(pred_pos[:, 2]),
        "graph": ["gt", ] * len(gt_pos) + ["pred", ] * len(pred_pos),
        "match": gt_matches + pred_matches
    }
    df = pd.DataFrame(data=data)

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="match", symbol="graph",
                        size_max=4)
    fig.update_traces(marker={'size': 4})

    edge_x_gt, edge_y_gt, edge_z_gt = [], [], []
    edge_x_pred, edge_y_pred, edge_z_pred = [], [], []
    edge_x_match, edge_y_match, edge_z_match = [], [], []

    for node1, node2 in gt_graph.edges:
        pos1 = gt_graph.nodes[node1]['coord']
        pos2 = gt_graph.nodes[node2]['coord']

        edge_x_gt.extend([pos1[0], pos2[0], None])
        edge_y_gt.extend([pos1[1], pos2[1], None])
        edge_z_gt.extend([pos1[2], pos2[2], None])

    for node1, node2 in pred_graph.edges:
        pos1 = pred_graph.nodes[node1]['coord']
        pos2 = pred_graph.nodes[node2]['coord']

        edge_x_pred.extend([pos1[0], pos2[0], None])
        edge_y_pred.extend([pos1[1], pos2[1], None])
        edge_z_pred.extend([pos1[2], pos2[2], None])

    for gt_node, pred_node in matched_dict.items():
        pos_gt = gt_graph.nodes[gt_node]['coord']
        pos_pred = pred_graph.nodes[pred_node]['coord']

        edge_x_match.extend([pos_gt[0], pos_pred[0], None])
        edge_y_match.extend([pos_gt[1], pos_pred[1], None])
        edge_z_match.extend([pos_gt[2], pos_pred[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x_gt, y=edge_y_gt, z=edge_z_gt,
        mode='lines',
        line=dict(color='blue', width=2),
        hoverinfo='none',
        name='GT Edges'
    ))

    fig.add_trace(go.Scatter3d(
        x=edge_x_pred, y=edge_y_pred, z=edge_z_pred,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none',
        name='Pred Edges'
    ))

    fig.add_trace(go.Scatter3d(
        x=edge_x_match, y=edge_y_match, z=edge_z_match,
        mode='lines',
        line=dict(color='red', width=3),
        hoverinfo='none',
        name='Matched Edges'
    ))

    fig.show()


def visualize_graph(graph):
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in graph.edges():
        x0, y0, z0 = graph.nodes[edge[0]]['coord']
        x1, y1, z1 = graph.nodes[edge[1]]['coord']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_z = []
    for node in graph.nodes():
        x, y, z = graph.nodes[node]['coord']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=2,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.show()


def update_segment(graph, old_segment_ids, new_segment_pos):
    # update one segment (from one root/branching point to another branching
    # point/ end point with resampled points
    npoints = len(graph.nodes)
    succ = old_segment_ids[0]
    cid = np.max(graph.nodes) + 1
    cradius = graph.nodes[succ]["radius"]  # TODO: interpolate radius
    # insert new nodes and edges
    for pos in new_segment_pos:
        graph.add_node(cid, coord=pos, radius=cradius)
        graph.add_edge(succ, cid)
        succ = cid
        cid += 1
    # add last edge to original segment end point
    graph.add_edge(succ, old_segment_ids[-1])

    # remove nodes and edges from resampled segment
    for cid in old_segment_ids[1:-1]:
        graph.remove_node(cid)
    if len(new_segment_pos) > 0:
        if graph.has_edge(old_segment_ids[0], old_segment_ids[-1]):
            graph.remove_edge(old_segment_ids[0], old_segment_ids[-1])

    return graph


# adapted from https://rdrr.io/cran/nat/src/R/neuron.R#sym-resample_segment
def resample_segment(seg, stepsize):
    if seg.shape[0] < 2:
        return None
    seg_xyz = seg[:, :3]
    seg_len = np.sum(np.sqrt(np.sum(np.diff(seg_xyz, axis=0) ** 2, axis=1)))
    if seg_len <= stepsize:
        return None

    internal_points = np.arange(stepsize, seg_len, stepsize)
    # if last resample point is close to end point of segment, discard it
    if abs(internal_points[-1] - seg_len) < stepsize:
        internal_points = internal_points[:-1]
        if len(internal_points) == 0:
            return None

    cumlength = np.insert(np.cumsum(np.sqrt(
        np.sum(np.diff(seg_xyz, axis=0) ** 2, axis=1))), 0, 0)

    seg_new = np.empty((len(internal_points), seg.shape[1]))
    for i in range(seg.shape[1]):
        if np.all(np.isfinite(seg[:, i])):
            seg_new[:, i] = np.interp(internal_points, cumlength, seg[:, i])
        else:
            seg_new[:, i] = np.nan
    return seg_new


def resample_graph(graph, stepsize, visualize=False):
    logger.info("resampling graph with stepsize %i" % stepsize)
    # TODO: check given in-between points in interpolation
    # TODO: interpolate radius
    graph_resampled = graph.copy()
    if visualize:
        visualize_graph(graph)

    # get roots
    roots = [n for n, d in graph.in_degree() if d == 0]
    positions = nx.get_node_attributes(graph, 'coord')

    # traverse tree
    for root in roots:
        start = root
        next_starts = []
        while graph.out_degree(start) > 0:
            childs = list(graph.successors(start))
            for child in childs:
                # find complete segment up to the next branching point
                segment_points = [start]
                segment_pos = [positions[start]]
                cnode = child
                while graph.out_degree(cnode) == 1:
                    segment_points.append(cnode)
                    segment_pos.append(positions[cnode])
                    cnode = list(graph.successors(cnode))[0]
                segment_points.append(cnode)
                segment_pos.append(positions[cnode])
                # resample segment
                segment_resampled = resample_segment(
                    np.array(segment_pos), stepsize)
                # add to resampled graph
                if segment_resampled is not None:
                    graph_resampled = update_segment(
                        graph_resampled, segment_points, segment_resampled)
                # if previous graph had smaller resampling rate, we need to delete points
                if segment_resampled is None and len(segment_points) > 2:
                    graph_resampled = update_segment(
                        graph_resampled, segment_points, [])

                # if segment end point is branching point, add it as next start
                if graph.out_degree(cnode) > 1:
                    next_starts.append(cnode)
            if len(next_starts) > 0:
                start = next_starts.pop()
            else:
                break
    if visualize:
        visualize_graph(graph_resampled)
    # maybe original and resampled in one visualization?

    return graph_resampled


def remove_segment_points(graph, roots, check_against, matched, gt=False):
    graph_reduced = graph.copy()
    # traverse tree
    for root in roots:
        start = root
        next_starts = []
        while graph.out_degree(start) > 0:
            childs = list(graph.successors(start))
            for child in childs:
                # find complete segment up to the next branching point
                segment_points = [start]
                cnode = child
                while graph.out_degree(cnode) == 1:
                    segment_points.append(cnode)
                    cnode = list(graph.successors(cnode))[0]
                segment_points.append(cnode)
                segment_points = np.array(segment_points)

                # check which nodes to remove
                rm_nodes = []
                for cn in segment_points[1:]:
                    if graph.out_degree(cn) == 1:
                        if cn in matched:
                            if type(matched[cn]) in [int, np.int32, np.int64]:
                                mnodes = np.array([matched[cn]])
                            else:
                                mnodes = np.array(matched[cn])
                            if np.any(np.isin(mnodes, check_against.nodes)):
                                mnodes_out_edges = np.array(
                                    [check_against.out_degree(mn) for mn in mnodes])
                                if np.all(mnodes_out_edges == 1):
                                    rm_nodes.append(cn)
                        else:
                            rm_nodes.append(cn)

                # remove segment points
                if len(rm_nodes) > 0:
                    for rm in rm_nodes:
                        graph_reduced.remove_node(rm)
                    keep_nodes = segment_points[
                        np.isin(segment_points, rm_nodes, invert=True)]
                    for i, cn in enumerate(keep_nodes[1:]):
                        graph_reduced.add_edge(keep_nodes[i], keep_nodes[i + 1])

                # if segment end point is branching point, add it as next start
                if graph.out_degree(cnode) > 1:
                    next_starts.append(cnode)
            if len(next_starts) > 0:
                start = next_starts.pop()
            else:
                break
    return graph_reduced


def reduce_graphs(gt, pred, matched, visualize=False):
    # reduce graph: remove points along segments
    # as long as they are not matched to a branching point
    logger.info("removing points along segments in graph")
    logger.debug("len gt/pred: %i/%i" % (len(gt.nodes), len(pred.nodes)))

    # get roots
    gt_roots = [n for n, d in gt.in_degree() if d == 0]
    pred_roots = [n for n, d in pred.in_degree() if d == 0]

    # hack to cleanup matching dict in case of FM and FS
    matched_pred_ids, pred_cnt = np.unique(list(matched.values()), return_counts=True)
    n_matched_pred = matched_pred_ids[pred_cnt > 1]
    for nm in n_matched_pred:
        nm_out = pred.out_degree(nm)
        if nm_out == 1:
            continue
        n_matched_gt = [k for k, v in matched.items() if v == nm]
        n_matched_gt_out = [gt.out_degree(k) for k in n_matched_gt]
        # print(n_matched_gt, n_matched_gt_out, pred.out_degree(nm))
        if np.any(n_matched_gt_out == nm_out):
            # remove all in between nodes
            for ngt, ngtout in zip(n_matched_gt, n_matched_gt_out):
                if ngtout == 1:
                    del matched[ngt]

    gt_reduced = remove_segment_points(gt, gt_roots, pred, matched, gt=True)
    # turn matching dict to have pred ids as keys
    matched_pred = {}
    for k, v in matched.items():
        if v in matched_pred:
            matched_pred[v].append(k)
        else:
            matched_pred[v] = [k]
    # matched_pred = {v: k for k, v in matched.items()} # heads up, it is not 1:1 matched
    pred_reduced = remove_segment_points(pred, pred_roots, gt, matched_pred)
    logger.debug("len gt/pred reduced: %i/%i" % (
        len(gt_reduced.nodes), len(pred_reduced.nodes)))

    # update matched
    to_remove = []
    for k, v in matched.items():
        if k not in gt_reduced.nodes or v not in pred_reduced.nodes:
            to_remove.append(k)
    for rm in to_remove:
        if k in matched:
            del matched[k]
    print("before passing to main: ", len(gt_reduced.nodes), len(pred_reduced.nodes))

    return gt_reduced, pred_reduced, matched
