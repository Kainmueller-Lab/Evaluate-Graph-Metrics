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


def compute_node_metrics(G, H, matched, FN, FP):
    G_nodes_matched = set(n for n in G.nodes if n in matched)
    H_nodes_matched = set(matched[n] for n in H.nodes if n in matched)
    # unmatched H.nodes are removed need to include the unmatched BP
    num_TP = len(G_nodes_matched)
    num_FN = len(FN)
    num_FP = len(FP)
    # need to include the unmatched bp targets or H.nodes

    precision = num_TP / float(num_TP + num_FP) if num_TP + num_FP > 0 else 0
    recall = num_TP / float(num_TP + num_FN) if num_TP + num_FN > 0 else 0
    f1 = 2 * num_TP / float(2 * num_TP + num_FN + num_FP) if \
            2 * num_TP + num_FN + num_FP > 0 else 0

    metrics = {
        "TP_nodes": num_TP,
        "FN_nodes": num_FN,
        "FP_nodes": num_FP,
        "precision_nodes": precision,
        "recall_nodes": recall,
        "f1_nodes": f1
    }
    return metrics


def visualization(G, H, false_merges, false_splits, matched):
    # code structure is ready just need to add false splits and otner errors if needed. False merge is already included
    G_nodes = G.nodes()
    H_nodes = H.nodes()

    G_edges = []
    H_edges = []
    FM = []  # False merges
    FS = []  # False splits


    for edge in G.edges():
        node1, node2 = edge
        node1_coord = G_nodes[node1]['coord']
        node2_coord = G_nodes[node2]['coord']
        G_edges.append([node1_coord.tolist(), node2_coord.tolist()])


    for edge in H.edges():
        node1, node2 = edge
        node1_coord = H_nodes[node1]['coord']
        node2_coord = H_nodes[node2]['coord']
        H_edges.append([node1_coord.tolist(), node2_coord.tolist()])


    for edge in false_merges:
        node1, node2 = edge
        node1_coord = H_nodes[node1]['coord']
        node2_coord = H_nodes[node2]['coord']
        FM.append([node1_coord.tolist(), node2_coord.tolist()])


    fig = go.Figure()


    for i, edge in enumerate(G_edges):
        fig.add_trace(go.Scatter3d(
            x=[edge[0][0], edge[1][0]],
            y=[edge[0][1], edge[1][1]],
            z=[edge[0][2], edge[1][2]],
            mode='lines',
            line=dict(color='green', width=3),
            name="G_edges" if i == 0 else None,
            showlegend=(i == 0)
        ))


    for i, edge in enumerate(H_edges):
        fig.add_trace(go.Scatter3d(
            x=[edge[0][0], edge[1][0]],
            y=[edge[0][1], edge[1][1]],
            z=[edge[0][2], edge[1][2]],
            mode='lines',
            line=dict(color='blue', width=3),
            name="H_edges" if i == 0 else None,
            showlegend=(i == 0)
        ))


    for i, edge in enumerate(FM):
        fig.add_trace(go.Scatter3d(
            x=[edge[0][0], edge[1][0]],
            y=[edge[0][1], edge[1][1]],
            z=[edge[0][2], edge[1][2]],
            mode='lines',
            line=dict(color='red', width=5),
            name="False merge" if i == 0 else None,
            showlegend=(i == 0)
        ))


    fig.update_layout(
        title="Graph Visualization with Plotly",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def compute_edge_metrics(G, H, matched, FN, FP, visualize=False):
    """
    Computes common edges (TP), false negatives (FN), and false positives (FP).
    Returns a dictionary with TP, FN, and FP values.
    """
    G_edges_matched = set([(e[0], e[1]) for e in G.edges if e[0] in matched and e[1] in matched])
    H_nodes_matched = np.unique(list(matched.values()))
    H_edges_matched = set([(e[0], e[1]) for e in H.edges \
                   if e[0] in H_nodes_matched and e[1] in H_nodes_matched])
    logger.debug("len matched edges for G and H: %i / %i" % (
        len(G_edges_matched), len(H_edges_matched)))

    num_FN = len(set(G.edges) - G_edges_matched)  # False negatives
    false_positives = set(H.edges) - H_edges_matched  # False positives
    num_FP = len(false_positives)
    num_TP = len(G_edges_matched)

    G_components = {frozenset(comp) for comp in nx.weakly_connected_components(G)}
    false_merges_intra_component = set()
    false_merges_inter_component = set()
    for fp_edge in false_positives:
        u, v = fp_edge
        if G.has_node(u) and G.has_node(v):
            #
            if not (nx.has_path(G, source=u, target=v) or nx.has_path(G, source=v, target=u)):
                same_component = any(u in comp and v in comp for comp in G_components)
                if same_component:
                    false_merges_intra_component.add(fp_edge)
                else:
                    false_merges_inter_component.add(fp_edge)

    total_false_merges = false_merges_intra_component | false_merges_inter_component
    # print(len(total_false_merges))
    # print(matched)
    H_copy = H.copy()
    H_copy.remove_edges_from(false_merges_intra_component)
    components_before = nx.number_weakly_connected_components(H)
    components_after = nx.number_weakly_connected_components(H_copy)
    false_splits = components_after - components_before
    beta1= betti_0_number(G, H)
    total_false_splits = false_splits + beta1

    if visualize == True:
        visualization(G, H, total_false_merges, total_false_splits, matched)

    #print("false_merges:", len(false_merges))

    precision = num_TP / float(num_TP + num_FP) if num_TP + num_FP > 0 else 0
    recall = num_TP / float(num_TP + num_FN) if num_TP + num_FN > 0 else 0
    f1 = 2 * num_TP / float(2 * num_TP + num_FN + num_FP) \
            if 2 * num_TP + num_FN + num_FP > 0 else 0

    metrics = {
        "TP_edges": num_TP,
        "FN_edges": num_FN,
        "FP_edges": num_FP,
        "FM": len(total_false_merges),
        "FS": total_false_splits,
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

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="match", symbol="graph",
                       size_max=4)
    fig.update_traces(marker={'size': 2})
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
    cradius = graph.nodes[succ]["radius"] # TODO: interpolate radius
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
    if graph.has_edge(old_segment_ids[0], old_segment_ids[-1]):
        graph.remove_edge(old_segment_ids[0], old_segment_ids[-1])

    return graph


# adapted from https://rdrr.io/cran/nat/src/R/neuron.R#sym-resample_segment
def resample_segment(seg, stepsize):
    if seg.shape[0] < 2:
        return None
    seg_xyz = seg[:, :3]
    l = np.sum(np.sqrt(np.sum(np.diff(seg_xyz, axis=0) ** 2, axis=1)))
    if l <= stepsize:
        return None

    internal_points = np.arange(stepsize, l, stepsize)
    if internal_points[-1] == l:
        internal_points = internal_points[:-1]

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
    roots = [n for n, d in graph.in_degree() if d==0]
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

                # if segment end point is branching point, add it as next start
                if graph.out_degree(cnode) > 1:
                    next_starts.append(cnode)
            if len(next_starts) > 0:
                start = next_starts.pop()
            else:
                break
    if visualize:
        visualize_graph(graph_resampled)

    return graph_resampled


def remove_segment_points(graph, roots, check_against, matched):
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
                    segment_pos.append(positions[cnode])
                    cnode = list(graph.successors(cnode))[0]
                segment_points.append(cnode)

                # check which nodes to remove
                rm_nodes = []
                for cn in segment_points:
                    if graph.out_degree(cn) == 1:
                        if check_against.out_degree(matched[cn]) == 1:
                            rm_nodes.append(cn)

                # remove segment points
                if len(rm_nodes) > 0:
                    # remove points

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
    logger.info("resampling graph with stepsize %i" % stepsize)

    # get roots
    gt_roots = [n for n, d in gt.in_degree() if d==0]
    pred_roots = [n for n, d in pred.in_degree() if d==0]

    gt_reduced = remove_segment_points(gt, gt_roots, pred, matched)
    # turn matching dict to have pred ids as keys
    matched_pred = {v: k for k, v in matched.items()}
    pred_reduced = remove_segment_points(pred, pred_roots, gt, matched_pred)

    if visualize:
        visualize_graph(gt_reduced)


