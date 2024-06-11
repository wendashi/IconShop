import torch

# # 加载权重文件
# bin_name = 'scheduler'

# optimizer_state = torch.load(f'/home/stone/Desktop/AnyFont/IconShop/proj_log/FIGR_SVG/epoch_100/{bin_name}.bin', map_location='cpu')

# # 打开一个文本文件以写入
# with open(f'{bin_name}_contents.txt', 'w') as f:
#     # 遍历每个权重张量
#     # f.write(f'{optimizer_state.keys()}\n')

#     for key in optimizer_state.keys():
#     # 写入权重张量的名称和形状到文件
#         f.write(f'{key}: {optimizer_state[key]}\n')

# print(f"权重张量的名称和形状已保存到 {bin_name}_contents.txt 文件中。")

import pickle

# 定义一个 persistent_load 函数，用于加载被引用的对象
# def persistent_load(persid):
#     raise pickle.UnpicklingError("Could not find persistent object")

# # 加载 pickle 文件
# with open("random_states_0.pkl", "rb") as f:
#     random_states = pickle.load(f, encoding='latin1', fix_imports=True)



f = open('random_states_0.pkl','rb')
data = torch.load(f,map_location='cpu')#可使用cpu或gpu

# 打开一个文本文件以写入
with open("random_states.txt", "w") as f:
    # 将内容写入到文件中
    for key in data.keys():
    # 写入权重张量的名称和形状到文件
        f.write(f'{key}: {data[key]}\n')

