import math
import torch
import torch.nn as nn

import numpy as np
import ot

import matplotlib.pyplot as plt
import plotly.graph_objects as go

plt.switch_backend('agg')



class StreetMoverDistance(nn.Module):
    r"""
        Given two planar graphs approximate them with two 2D point clouds equidistantly sampled over the edges.
        Then, compute the sinkhorn distance approximation between the two point clouds using the implementation publicly
        available from Daniel Daza.
        
        Return the sampled point clouds, the sinkhorn distance, and the coupling and cost matrices.
        If one of the two graphs is empty, replace its point cloud with a point cloud of 100 points lying in the origin
        of axis (0, 0).
    """
    
    def __init__(self, eps, max_iter, reduction='mean', use_correct_sinkhorn_dist=True):
        super(StreetMoverDistance, self).__init__()
        # NOTE: The below SinkhornDistance implementation is faulty (see smd_analysis notebook).
        # We use the implementation from POT package instead.
        self.sinkhorn_distance = SinkhornDistance_POT() if use_correct_sinkhorn_dist else SinkhornDistance(eps=eps, max_iter=max_iter, reduction=reduction)

    def forward(self, y_A, y_nodes, output_A, output_nodes, n_points=2000):
        y_pc = self.get_point_cloud(y_A, y_nodes, n_points)
        output_pc = self.get_point_cloud(output_A, output_nodes, n_points)
        sink_dist, P, C = self.sinkhorn_distance(y_pc, output_pc)
        return (y_pc, output_pc), (sink_dist, P, C)

    import plotly.graph_objects as go
    import torch

    def get_point_cloud(self, A, nodes, n_points, visualize=False, title="Generated Point Cloud"):
        n_divisions = n_points - 1 + 0.01
        total_len = get_cumulative_distance(A, nodes)
        step = total_len / n_divisions
        #step=1
        points = []
        next_step = 0.
        used_len = 0.

        for i in range(A.shape[0]):
            for j in range(i):
                if A[i, j] == 1.:
                    next_step, used, pts = get_points(next_step, step, nodes[j].clone(), nodes[i].clone())
                    if torch.isnan(torch.tensor(pts)).any():
                        print(f"nan detected between edge {j} and {i}")
                    used_len += used
                    points += pts
                    last_node = nodes[i].clone()

        # Handle edge cases where fewer points are generated
        if 0 < len(points) < n_points:
            while len(points) < n_points:
                points.append((last_node[0].item(), last_node[1].item()))

        # Handle empty graphs
        if len(points) == 0:
            points = torch.zeros((100, 2))  # Adjust to 3D for visualization

        points = torch.FloatTensor(points)
        #points = torch.stack(nodes)

        # Optional Plotly visualization
        if visualize:
            self.visualize_point_cloud_plotly(points, title)

        return points

    def visualize_point_cloud_plotly(self, points, title="Generated Point Cloud"):
        if points.shape[1] == 3:  # 3D visualization
            fig = go.Figure(data=go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.8),
            ))
            fig.update_layout(scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis"),
                title=title)
        else:  # 2D visualization
            fig = go.Figure(data=go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.8),
            ))
            fig.update_layout(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                title=title
            )

        fig.show()


def get_cumulative_distance(A, nodes):
    tot = 0.
    for i in range(A.shape[0]):
        for j in range(i):
            # print(i, j)
            if A[i, j] == 1.:
                # print(nodes[i], nodes[j])
                tot += euclidean_distance(nodes[i], nodes[j])
    return tot


def get_points(next_step, step, a, b):
    if torch.isnan(a).any() or torch.isnan(b).any():
        print(f"NaN in node coordinates: a={a}, b={b}")
        return next_step, 0, []
    if torch.isinf(a).any() or torch.isinf(b).any():
        print(f"Inf in node coordinates: a={a}, b={b}")
        return next_step, 0, []

    l = euclidean_distance(a, b)

    if l < 1e-6:
        print("skipping edge")
        return next_step, 0, []
    m = ((b[1] - a[1]) / (b[0] - a[0])).item() if b[0] != a[0] else 0.0
    sign_x = -1 if b[0] < a[0] else 1  # going backwards or forward
    sign_y = -1 if b[1] < a[1] else 1  # going backwards or forward
    pts = []
    used = 0
    while next_step <= l:
        used += next_step
        l -= next_step
        dx = sign_x * next_step / math.sqrt(1 + m ** 2)
        dy = m * dx if abs(dx) > 1e-06 else sign_y * next_step
        a[0] += dx
        a[1] += dy
        pts.append((a[0].item(), a[1].item()))
        next_step = step
    next_step = step - l
    return next_step, used, pts


def euclidean_distance(a, b):
    return math.sqrt((a - b).pow(2).sum().item())



class SinkhornDistance_POT(nn.Module):
    def __init__(self, regularization=1): # TODO: Describe and fix default regul parameter
        super(SinkhornDistance_POT, self).__init__()
        self.regularization = regularization
    
    def forward(self, A, B):
        def to_array(coords):
            if isinstance(coords, torch.Tensor):
                return coords.cpu().numpy()  # Convert tensor to NumPy safely
            return np.array([coord for coord in coords], dtype=np.float32)  # Ensure consistent NumPy array
        
        A = to_array(A)
        B = to_array(B)

        # Compute pairwise distance matrix
        M = ot.dist(A, B, metric='euclidean')
        # Uniform weights (assume each point contributes equally)
        w_A = np.ones(len(A)) / len(A)
        w_B = np.ones(len(B)) / len(B)
        # Compute Sinkhorn Distance (Regularized Optimal Transport)
        S_dist = ot.emd2(w_A, w_B, M, self.regularization)
        return S_dist, None, None
        

# From https://github.com/dfdazac/wassdistance
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
    
    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()
        
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1
        
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            
            actual_nits += 1
            if err.item() < thresh:
                break
        
        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        
        return cost, pi, C
    
    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
    
    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C
    
    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def show_assignments(a, b, P, title=""):
    norm_P = P / P.max()
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            plt.arrow(a[i, 0], a[i, 1], b[j, 0] - a[i, 0], b[j, 1] - a[i, 1],
                      alpha=norm_P[i, j].item() / 2)
    plt.scatter(a[:, 0], a[:, 1], label="target")
    plt.scatter(b[:, 0], b[:, 1], label="prediction")
    plt.axis('off')
    plt.legend(prop={'size': 20})
    plt.title(f'StreetMover: {title[:6]}', fontsize=25)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./sinkhorn/wass{title}.png")
    plt.clf()