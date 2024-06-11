import onnx
from onnx import shape_inference

model = onnx.load("/home/stone/Desktop/AnyFont/IconShop/onnx2triton/iconshop_1.onnx")
# # model = shape_inference.infer_shapes(model)
# # print(onnx.helper.printable_graph(model.graph))


# for output_tensor in model.graph.output:
#     for dim in output_tensor.type.tensor_type.shape.dim:
#         if dim.dim_value == 562:
#             dim.dim_param = 'len'


# onnx.save(model, "/home/stone/Desktop/AnyFont/IconShop/onnx2triton/iconshop_1_folded_modified.onnx")

def count_slice_nodes(model):
    # 加载 ONNX 模型
    # model = onnx.load(onnx_model_path)
    
    # 获取模型中的图
    graph = model.graph
    
    # 统计 Slice 节点的数量
    slice_count = sum(1 for node in graph.node if node.op_type == 'Slice')
    
    return slice_count

# 使用示例：将 'your_model.onnx' 替换为你的 ONNX 模型文件路径
# onnx_model_path = 'your_model.onnx'
slice_count = count_slice_nodes(model)

print(f"The number of 'Slice' nodes in the model: {slice_count}")
