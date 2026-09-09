import onnx

# Load the model
model = onnx.load("output/hola_elmo/hola_elmo.onnx")

# Print details for all inputs
for input in model.graph.input:
    print(f"Input Name: {input.name}")
    
    # Get shape dimensions
    shape = [dim.dim_value if dim.dim_value > 0 else "Dynamic" for dim in input.type.tensor_type.shape.dim]
    print(f"Input Shape: {shape}")
    
    # Get data type (1 = Float32, 3 = Int8, etc.)
    print(f"Data Type Tensor Enum: {input.type.tensor_type.elem_type}\n")
