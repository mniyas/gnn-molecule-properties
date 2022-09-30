import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from torch_geometric.datasets import QM9

from qm_property_predictor.molecular_property_inference import MoleculeProperty

st.set_page_config(page_title="Molecular Property Prediction with GNN", layout="wide")


# Layout for plotting functions
axis = dict(
    showbackground=False,
    showticklabels=False,
    showgrid=False,
    zeroline=False,
    title="",
)

layout = dict(
    showlegend=True,
    scene=dict(
        aspectmode="data",
        xaxis=dict(
            **axis,
        ),
        yaxis=dict(
            **axis,
        ),
        zaxis=dict(
            **axis,
        ),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
)


@st.cache()
def get_molecules_names():
    path = osp.join(osp.dirname(__file__), "data", "qm9-labels.csv")
    df = pd.read_csv(path).head(100)
    return df.mol_id + "(" + df.smiles + ")"


@st.cache(allow_output_mutation=True)
def get_inference_model():
    model = MoleculeProperty()
    return model


def visualise_molecule(graph):
    """Plot a molecule based on a graph from QM9"""
    pos = graph.pos.clone()
    edge_index = graph.edge_index

    pos = (pos - pos.mean(0)) / pos.std(0)

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    atom_type = (1 + graph.x[:, :5].argmax(-1)) * 10

    pos = pos[edge_index]
    data = [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=atom_type, color=atom_type),
            showlegend=True,
        )
    ]
    for i in range(edge_index.size(-1)):
        line_data = pos[:, i, :]
        data.append(
            go.Scatter3d(
                x=line_data[:, 0],
                y=line_data[:, 1],
                z=line_data[:, 2],
                mode="lines",
                line=dict(
                    color="grey",
                    width=3,
                ),
                showlegend=False,
            )
        )

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)


st.title("Molecular property prediction with GNN")
st.markdown(
    """Molecular properties of QM9 are estimated using DimeNet GNN model.
       This app demonstrates properties predicted for a small number of molecules.
       The results show good agreement with experimental data and demonstrate the potential of this approach for predicting molecular properties."""
)
with st.sidebar:
    st.selectbox("Select Molecule", options=get_molecules_names())

model = get_inference_model()
dataset = model.access_data()
random_idx = np.random.randint(len(dataset))
graph = dataset[random_idx]
visualise_molecule(graph)
predictions = model.predict(graph)
df = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))
columns = st.columns(6)
for col, pred in zip(columns, predictions[:6]):
    col.metric(label=pred[0], value="%0.2f" % pred[1])

columns = st.columns(6)
for col, pred in zip(columns, predictions[6:]):
    col.metric(label=pred[0], value="%0.2f" % pred[1])
