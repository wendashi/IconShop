import torch

def test_sum(a, b):
    return a + b

# 注释参数为 int
scripted_fn = torch.jit.script(test_sum, example_inputs=[(3, 4)])

print(type(scripted_fn))  # torch.jit.ScriptFunction

# 以 Python 代码查看编译后的图
print(scripted_fn.code)

# 使用 TorchScript 解释器调用函数
scripted_fn(20, 100)

# torch.jit.script 本身、版本啥的都没问题

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

scripted_gate = torch.jit.script(MyDecisionGate())