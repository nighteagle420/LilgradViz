import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import graphviz as gv
import ast
# Import your custom tensor/autodiff class here
# from your_autodiff_module import Tensor, backward
from engine import *

st.set_page_config(layout="wide")
st.title("LilGradViz - Automatic Differentiation with Visualization")

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
        # st.rerun()
        st.sidebar.success(f"Tensor '{tensor_label}' with value {matrix} added!")
    else:
        st.sidebar.error("Tensor with this label already exists. Choose a new one.")





available_tensors = list(st.session_state.tensors.keys())
# ----------------------------------UNARY-------------------------------------------
if len(available_tensors) >= 1:
    st.sidebar.markdown("---")
    st.sidebar.header("2a. Define Operations(unary)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        input1_label = st.selectbox("Input", available_tensors, key="op_input_11")
    with col2:
        output_tensor_label = st.sidebar.text_input("Output Label", value="z")
    
    # 2) Ask the label of new tensor created during an operation
    
    col1,col2 = st.sidebar.columns(2)
    # Simple addition operation
    if col1.button("Sum"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            # input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor.sum()
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "sum", 
                "inputs": [input1_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Sum operation: {input1_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()
            
    if col2.button("Relu"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            # input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor.relu()
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "relu", 
                "inputs": [input1_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Sum operation: {input1_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()
            
    col1,col2 = st.sidebar.columns(2)
    # Simple addition operation
    if col1.button("Log"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            # input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor.log()
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "log", 
                "inputs": [input1_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Sum operation: {input1_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()
            
    if col2.button("Tanh"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            # input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor.tanh()
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "tanh", 
                "inputs": [input1_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Sum operation: {input1_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()


#--------------------------------BINARY-----------------------------------------------
if len(available_tensors) >= 2:
    st.sidebar.markdown("---")
    st.sidebar.header("2b. Define Operations(Binary)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        input1_label = st.selectbox("Input 1", available_tensors, key="op_input_21")
    with col2:
        input2_label = st.selectbox("Input 2", available_tensors, key="op_input_22")
    
    # 2) Ask the label of new tensor created during an operation
    output_tensor_label = st.sidebar.text_input("Output Tensor Label", value="z")
    col1,col2 = st.sidebar.columns(2)
    # Simple addition operation
    if col1.button("Add (+)"):
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
            
    if col2.button("Subtract (-)"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor - input2_tensor
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "-", 
                "inputs": [input1_label, input2_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Subtracted operation: {input1_label} - {input2_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()
            
    col1,col2 = st.sidebar.columns(2)
    # Simple addition operation
    if col1.button("Multiply (*)"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor * input2_tensor
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "*", 
                "inputs": [input1_label, input2_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Multiply operation: {input1_label} * {input2_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()
            
    if col2.button("Divide (/)"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor / input2_tensor
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "/", 
                "inputs": [input1_label, input2_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Divided operation: {input1_label} - {input2_label} = {output_tensor_label}")
            # A function to draw the graph
            st.rerun()
            
    if st.sidebar.button("Matmul (@)"):
        if not output_tensor_label:
            st.sidebar.error("Please provide a label for the output tensor.")
        elif output_tensor_label in st.session_state.tensors:
            st.sidebar.error("Output label already exists. Choose a new one.")
        else:
            input1_tensor = st.session_state.tensors[input1_label]
            input2_tensor = st.session_state.tensors[input2_label]
            
            # Create a new Tensor instance
            new_tensor = input1_tensor @ input2_tensor
            
            st.session_state.tensors[output_tensor_label] = new_tensor
            st.session_state.operation_history.append({
                "op": "@", 
                "inputs": [input1_label, input2_label], 
                "output": output_tensor_label
            })
            st.session_state.final_tensor_label = output_tensor_label
            st.sidebar.success(f"Matmul operation: {input1_label} @ {input2_label} = {output_tensor_label}")
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
    if st.session_state.gradients_data is not None:
        st.table(st.session_state.gradients_data)
    


if st.session_state.operation_history:
    draw_graph()
    
if st.session_state.final_tensor_label:
    st.sidebar.markdown("---")
    st.sidebar.header("3. Run Backpropagation/Clear Grads")
    col1,col2 = st.sidebar.columns(2)
    if col1.button("Calculate Gradients"):
        final_tensor = st.session_state.tensors[st.session_state.final_tensor_label]
        try:
            
            final_tensor.backward()
            
            # st.success("Backpropagation complete!")
        
            st.subheader("Final Gradients")
            st.session_state.gradients_data = [
                {"Tensor": label, "Value": str(tensor.data), "Gradient": str(tensor.grad)}
                for label, tensor in st.session_state.tensors.items()
            ]
            st.rerun()
            # st.table(gradients_data)
        except Exception as e:
            st.error(f"{e}")
        
        
    if col2.button("Clear Grads"):
        final_tensor = st.session_state.tensors[st.session_state.final_tensor_label]
        final_tensor.zero_grad()
        st.session_state.gradients_data = None
        st.rerun()
        
    
st.sidebar.markdown("---")
st.sidebar.header("4. Maintainance and Controls")
if st.sidebar.button("Reset App"):
    st.session_state.tensors = {}
    st.session_state.operation_history = []
    st.session_state.final_tensor_label = None
    st.session_state.gradient_data = None
    st.rerun()