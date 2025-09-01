import streamlit as st
import numpy as np

# Example autodiff class (replace with your actual class)
from engine import *


# Initialize session state to store created tensors
if "tensors" not in st.session_state:
    st.session_state.tensors = []

st.title("Automatic Differentiation Playground")

st.subheader("Create a New Tensor")

# Input fields for tensor creation
default_label = f"T{len(st.session_state.tensors)}"
label = st.text_input("Tensor Label", value=default_label)
shape = st.text_input("Tensor Shape (comma separated)", value="2,2")

# Button to create tensor
if st.button("Create Tensor"):
    # Check uniqueness of label
    existing_labels = [t.label for t in st.session_state.tensors]
    if label in existing_labels:
        st.error(f"A tensor with label '{label}' already exists! Choose a unique label.")
    else:
        try:
            shape_tuple = tuple(map(int, shape.split(",")))
            data = np.random.randn(*shape_tuple)  # Random init
            tensor = TensorVal(data, label=label)
            st.session_state.tensors.append(tensor)
            st.success(f"Created {tensor}")
        except Exception as e:
            st.error(f"Error: {e}")

st.subheader("Current Tensors")
for idx, t in enumerate(st.session_state.tensors):
    st.write(f"**{idx}:** {t}")

if st.button("Clear All Tensors"):
    st.session_state["tensors"].clear()
    st.success("All tensors cleared!")