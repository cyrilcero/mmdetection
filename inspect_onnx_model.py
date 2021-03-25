import onnx

# Load the ONNX model
model = onnx.load("/home/cyril.cero/mmdetection/publish/pth_onnx.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))