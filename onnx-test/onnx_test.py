import torch
import onnxruntime
from transformers import AutoTokenizer

onnx_path = '/home/stone/Desktop/AnyFont/IconShop/iconshop_1.onnx'

# 使用 ONNX Runtime 进行推理
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-12_H-512_A-8')

text = 'calendar'

# tokenize text input
encoded_dict = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length= 50,
    add_special_tokens=True,
    return_token_type_ids=False,  # for RoBERTa
)
tokenized_text = encoded_dict["input_ids"].squeeze()
tokenized_text = tokenized_text.repeat(4, 1).to(device)

# 定义示例输入
pixel_seq = torch.tensor([[3], [3], [3], [3]], device=device)
xy_seq = torch.tensor([[[3, 3]], [[3, 3]], [[3, 3]], [[3, 3]]], device=device)
text = tokenized_text.cpu().numpy()

# 准备输入数据
ort_inputs = {
    'pixel_seq': pixel_seq.cpu().numpy(),
    'xy_seq': xy_seq.cpu().numpy(),
    'text': text
}

# 运行推理
outputs = ort_session.run(['output'], ort_inputs)

# 输出结果
print('推理结果:', outputs[0])