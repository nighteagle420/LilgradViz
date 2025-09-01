import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import graphviz as gv
import ast
# Import your custom tensor/autodiff class here
# from your_autodiff_module import Tensor, backward
from engine import *

st.set_page_config(layout="wide")
st.title("Automatic Differentiation with Visualization")

# Initialize session state variables
if "tensors" not in st.session_state:
    st.session_state.tensors = {}  # Stores all tensors by their label
if "operation_history" not in st.session_state:
    st.session_state.operation_history = []  # Stores a list of dictionaries for operations
# if "graph" not in st.session_state:
#     st.session_state.graph = None
if "final_tensor_label" not in st.session_state:
    st.session_state.final_tensor_label = None
if "gradient_data" not in st.session_state:
    st.session_state.gradient_data = None
# if "backprop_step" not in st.session_state:
#     st.session_state.backprop_step = -1 # -1: not run, 0: initial, >0: steps
    
st.sidebar.header("1. Create Tensors")

tensor_label = st.sidebar.text_input("Tensor Label (e.g., 'x', 'y')", value=f"T{len(st.session_state.tensors)}")
tensor_value_str = st.sidebar.text_area("Tensor Value in numpy format", value="1.0")

if st.sidebar.button("Add Tensor"):
    if tensor_label not in st.session_state.tensors:
        # Create a new Tensor instance from your class
        try:
            # Convert string to Python object (list of lists)
            matrix_list = ast.literal_eval(tensor_value_str)
            matrix = np.array(matrix_list)      
        except Exception as e:
            st.error(f"Invalid input: {e}")
        
        new_tensor = TensorVal(matrix, label=tensor_label)
        st.session_state.tensors[tensor_label] = new_tensor
        st.sidebar.success(f"Tensor '{tensor_label}' with value {matrix} added!")
    else:
        st.sidebar.error("Tensor with this label already exists. Choose a new one.")
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.header("2. Define Operations")

available_tensors = list(st.session_state.tensors.keys())

if len(available_tensors) >= 2:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        input1_label = st.selectbox("Input 1", available_tensors, key="op_input_1")
    with col2:
        input2_label = st.selectbox("Input 2", available_tensors, key="op_input_2")
    
    # 2) Ask the label of new tensor created during an operation
    output_tensor_label = st.sidebar.text_input("Output Tensor Label", value=f"T{len(available_tensors)}")
    
    # Simple addition operation
    if st.sidebar.button("Add (+)"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor + input2_tensor
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "+", 
                "inputs": [input1_label, input2_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Added operation: {input1_label} + {input2_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()

def draw_graph():
    graph = gv.Digraph('G', engine='dot')
    graph.attr(rankdir='TB')
    
    tensor_nodes = set()
    op_nodes = set()

    # Create nodes and edges
    for op_details in st.session_state.operation_history:
        op_id = f"op_{op_details['op']}_{'_'.join(op_details['inputs'])}"
        
        # Add operation node
        graph.node(op_id, op_details['op'], shape='circle', style='filled', fillcolor='lightgreen')
        op_nodes.add(op_id)
        
        # Add input tensor nodes and edges
        for input_label in op_details['inputs']:
            if input_label not in tensor_nodes:
                tensor = st.session_state.tensors[input_label]
                label_text = f"<{input_label}<br/>Value: {tensor.data}<br/>Grad: {tensor.grad}>"
                graph.node(input_label, label=label_text, shape='box', style='filled', fillcolor='skyblue')
                tensor_nodes.add(input_label)
            graph.edge(input_label, op_id)
            
        # Add output tensor node and edge
        output_label = op_details['output']
        if output_label not in tensor_nodes:
            tensor = st.session_state.tensors[output_label]
            label_text = f"<{output_label}<br/>Value: {tensor.data}<br/>Grad: {tensor.grad}>"
            graph.node(output_label, label=label_text, shape='box', style='filled', fillcolor='skyblue')
            tensor_nodes.add(output_label)
        graph.edge(op_id, output_label)

    st.graphviz_chart(graph,use_container_width=True)

# def draw_backprop_graph(current_step):
#     """
#     Draws the computation graph and highlights the backpropagation
#     steps up to the current_step.
#     """
#     graph = gv.Digraph('G', engine='dot')
#     graph.attr(rankdir='TB', splines='spline')

#     tensor_nodes = set()

#     # Pass 1: Draw all nodes and forward edges
#     for op_details in st.session_state.operation_history:
#         op_id = f"op_{op_details['op']}_{op_details['output']}"
        
#         graph.node(op_id, op_details['op'], shape='circle', style='filled', fillcolor='lightgreen')
        
#         for input_label in op_details['inputs']:
#             if input_label not in tensor_nodes:
#                 tensor = st.session_state.tensors[input_label]
#                 label_text = f"<{input_label}<br/>Value: {tensor.data}<br/>Grad: {tensor.grad}>"
#                 graph.node(input_label, label=label_text, shape='box', style='filled', fillcolor='skyblue')
#                 tensor_nodes.add(input_label)
#             graph.edge(input_label, op_id)
            
#         output_label = op_details['output']
#         if output_label not in tensor_nodes:
#             tensor = st.session_state.tensors[output_label]
#             label_text = f"<{output_label}<br/>Value: {tensor.data}<br/>Grad: {tensor.grad}>"
#             graph.node(output_label, label=label_text, shape='box', style='filled', fillcolor='skyblue')
#             tensor_nodes.add(output_label)
#         graph.edge(op_id, output_label)

#     # Pass 2: Draw red backward edges for completed backprop steps
#     if current_step > 0:
#         reversed_history = st.session_state.operation_history[::-1]
        
#         for i in range(current_step):
#             op_details = reversed_history[i]
#             output_label = op_details['output']
#             op_id = f"op_{op_details['op']}_{op_details['output']}"
            
#             for input_label in op_details['inputs']:
#                 graph.edge(output_label, input_label, color='red', style='dashed', dir='back', penwidth='2.0')

#     st.graphviz_chart(graph, use_container_width=True)


if st.session_state.operation_history:
    draw_graph()
    
if st.session_state.final_tensor_label:
    st.markdown("---")
    st.header("3. Run Backpropagation")
    if st.button("Calculate Gradients"):
        final_tensor = st.session_state.tensors[st.session_state.final_tensor_label]
        final_tensor.backward()
        
        st.success("Backpropagation complete!")
        
        # st.subheader("Final Gradients")
        st.session_state.gradient_data = [
            {"Tensor": label, "Value": str(tensor.data), "Gradient": str(tensor.grad)}
            for label, tensor in st.session_state.tensors.items()
        ]
        # st.session_state.backprop_step = 0
        st.rerun()
    if(st.session_state.gradient_data):
        st.table(st.session_state.gradient_data) 
        
# if st.session_state.backprop_step >= 0:
#     with st.container(border=True):
#             st.header("Backpropagation Animation 🧠")
            
#             max_steps = len(st.session_state.operation_history)
            
#             # Display current step info
#             if st.session_state.backprop_step == 0:
#                 st.write("Initial state. Gradients are calculated. Click 'Next' to see the flow.")
#             else:
#                 reversed_history = st.session_state.operation_history[::-1]
#                 op_info = reversed_history[st.session_state.backprop_step - 1]
#                 st.markdown(f"**Step {st.session_state.backprop_step}/{max_steps}**: Gradient flows from `{op_info['output']}` back to `{', '.join(op_info['inputs'])}`.")
            
#             # Draw the animated graph
#             draw_backprop_graph(st.session_state.backprop_step)
            
#             # Animation controls
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("Next Step ➡️", disabled=(st.session_state.backprop_step >= max_steps)):
#                     st.session_state.backprop_step += 1
#                     st.rerun()
#             with col2:
#                 if st.button("Reset Animation 🔄"):
#                     st.session_state.backprop_step = 0
#                     st.rerun()                 
        
        
    
st.sidebar.markdown("---")
st.sidebar.header("4. Reset")
if st.sidebar.button("Reset App"):
    st.session_state.tensors = {}
    st.session_state.operation_history = []
    st.session_state.final_tensor_label = None
    st.session_state.gradient_data = None
    # st.session_state.backprop_step = -1
    st.rerun()