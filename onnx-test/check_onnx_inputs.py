import onnx
import onnxruntime as ort

# 加载 ONNX 模型
onnx_model_path = "/home/stone/Desktop/AnyFont/IconShop/onnx2triton/iconshop_1_folded_modified.onnx"
onnx_model = onnx.load(onnx_model_path)

# 打印模型输入定义列表
print("ONNX 模型的输入定义:")
for input in onnx_model.graph.input:
    print(f"Name: {input.name}")
    print(f"Type: {input.type}")
    print(f"Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

for output in onnx_model.graph.output:
    print(f"Name: {output.name}")
    print(f"Type: {output.type}")
    print(f"Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")


# 使用 onnxruntime 加载模型并打印输入定义
try:
    ort_session = ort.InferenceSession(onnx_model_path)
    print("\nONNX Runtime 模型的输入定义:")
    for input in ort_session.get_inputs():
        print(f"Name: {input.name}")
        print(f"Type: {input.type}")
        print(f"Shape: {input.shape}")
except Exception as e:
    print(f"Error loading model with onnxruntime: {e}")

# 验证模型以确保其正确性
try:
    onnx.checker.check_model(onnx_model)
    print("The model is valid.")
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")
