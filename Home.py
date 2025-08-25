import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Assume TensorVal is already defined
# from tensor_engine import TensorVal
from engine import *

import plotly.graph_objects as go
import networkx as nx

def fmt_shape(x):
    """Compact label showing shape or scalar"""
    if isinstance(x, np.ndarray):
        if x.shape == () or x.size == 1:
            return "scalar"
        return f"array{list(x.shape)}"
    elif np.isscalar(x):  # ✅ correct scalar check
        return "scalar"
    else:
        return str(type(x).__name__)

def fmt_full(x):
    """Full content for hover tooltip"""
    if isinstance(x, np.ndarray):
        return repr(x)
    return str(x)

def trace_graph(root):
    """Trace back from root and collect tensor/op nodes and edges"""
    nodes, edges = [], []
    seen = {}

    def add_tensor(v):
        if id(v) not in seen:
            label = fmt_shape(v.data)
            hover = f"<b>data:</b> {fmt_full(v.data)}<br><b>grad:</b> {fmt_full(v.grad)}"
            nodes.append({
                "id": f"T{id(v)}",
                "label": label,
                "type": "tensor",
                "hover": hover
            })
            seen[id(v)] = v

            if v._op:
                op_id = f"O{id(v)}"
                nodes.append({"id": op_id, "label": v._op, "type": "op", "hover": v._op})
                edges.append((op_id, f"T{id(v)}"))  # op → tensor
                for child in v._prev:
                    add_tensor(child)
                    edges.append((f"T{id(child)}", op_id))  # tensor → op

    add_tensor(root)
    return nodes, edges


def plot_graph(root):
    nodes, edges = trace_graph(root)

    # Build networkx graph
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], label=n["label"], type=n["type"])
    for u, v in edges:
        G.add_edge(u, v)

    # Use graphviz layout
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G)

    # Edges (with arrows)
    edge_traces = []
    for u, v in G.edges():
        edge_traces.append(go.Scatter(
            x=[pos[u][0], pos[v][0]],
            y=[-pos[u][1], -pos[v][1]],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none"
        ))

    # Nodes
    node_x, node_y, node_text, node_color, node_shape = [], [], [], [], []
    for n in nodes:
        node_x.append(pos[n["id"]][0])
        node_y.append(-pos[n["id"]][1])
        node_text.append(n["label"])
        if n["type"] == "tensor":
            node_color.append("lightblue")
            node_shape.append("circle")
        else:  # op
            node_color.append("orange")
            node_shape.append("square")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n["label"] for n in nodes],
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=60,
            line_width=2,
            symbol=node_shape
        )
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig


# ===================================================
# Streamlit UI
# ===================================================

st.title("Interactive Tensor Graph with Backprop 🔥")

user_code = st.text_area("Write your code here:", height=200)

if st.button("Run Forward"):
    local_vars = {"TensorVal": TensorVal, "np": np}
    try:
        exec(user_code, {}, local_vars)
        root = None
        for v in local_vars.values():
            if isinstance(v, TensorVal):
                root = v
        if root is None:
            st.error("No TensorVal found!")
        else:
            st.session_state["root"] = root
            st.success("Forward pass done ✅")
            nodes, edges = trace_graph(root)
            st.plotly_chart(plot_graph(nodes, edges), use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Run Backward"):
    if "root" not in st.session_state:
        st.error("Run forward first!")
    else:
        try:
            root = st.session_state["root"]
            root.zero_grad()
            root.backward()
            st.success("Backward done ✅")
            nodes, edges = trace_graph(root)
            st.plotly_chart(plot_graph(nodes, edges), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")