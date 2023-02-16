import os
from typing import Optional
from matplotlib import rc
import torch
import numpy as np
import seaborn as sns
import plotly.express as px

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch_geometric
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go

from gpn.utils import Prediction


def visualize_graph_embeddings(
        data: torch_geometric.data.Data, 
        x: torch.Tensor, 
        labels: torch.Tensor,
        embedding: Optional[np.array] = None,
        colorscale: str = 'YlGnBu', 
        save_image: bool = False, 
        base_path: str = '', 
        fig_title: Optional[str] = None,
        cmax: Optional[float] = None, 
        cmin: Optional[float] = None, 
        colorbar: bool = True, 
        showlegend: bool = False,
        legend_items: Optional[dict] = None,
        return_embedding: bool = False):
    
    
    # colorscale options
    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |

    if embedding is None:
        tsne = TSNE(n_components=2)
        latent_embedding = tsne.fit_transform(x.detach().numpy())
    else:
        latent_embedding = embedding

    labels = labels.detach().numpy()

    G = to_networkx(data)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = latent_embedding[edge[0]]
        x1, y1 = latent_embedding[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        showlegend=False,
        line=dict(width=0.05, color='black'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_labels = []
    node_text = []

    for node in G.nodes():
        x, y = latent_embedding[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(labels[node].item()) 
        node_text.append(f'{labels[node]}')

    if isinstance(legend_items, dict):
        node_trace = []

        colors = px.colors.qualitative.D3
        colors = colors[:4] + [colors[5]] + colors[8:10]

        for k, v in legend_items.items():
            tmp = go.Scatter(
                x=np.array(node_x)[np.array(labels) == k],
                y=np.array(node_y)[np.array(labels) == k],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    color=colors[k],
                    size=[min(d[1] * 0.75, 20) for d in G.degree],
                    line=dict(width=1, color='black')),
                text=v,
                legendgroup=v,
                name=v,
                showlegend=True)

            node_trace.append(tmp)

        traces = [edge_trace, *node_trace]

    else:
        cb = dict(
            thickness=15,
            tickfont=dict(size=24, color='black'),
            xanchor='left',
            titleside='right'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=colorbar,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                reversescale=False,
                color=[],
                size=[min(d[1] * 0.75, 20) for d in G.degree],
                colorbar=cb,
                line=dict(width=1, color='black')),
            showlegend=False
        )

        node_trace.marker.color = node_labels
        node_trace.text = node_text

        traces = [edge_trace, node_trace]

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            hovermode=None,
            width=1200 if not showlegend else 1400,
            height=800,
            showlegend=showlegend,
            legend=None if not showlegend else {
                'itemsizing': 'constant',
                'orientation': 'v',
                'y': 0.25,
                'font': {'size': 24}},
            margin=dict(b=5, l=5, r=5, t=5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(255,255,255,1.0)',
            plot_bgcolor='rgba(0,0,0,0)')
    )

    if save_image:
        base_path = os.path.join(base_path, 'graphs')

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if fig_title is None:
            image_path = os.path.join(base_path, 'graph.png')

        else:
            image_path = os.path.join(base_path, f'{fig_title}.png')

        fig.write_image(image_path)

    else:
        fig.show()

    if return_embedding:
        return latent_embedding
