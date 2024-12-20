import torch
import torch
import torch.nn.functional as F
x = torch.arange(0,100).reshape(-1)
print(x)
print(x.shape)
x = x.le(10)
sum = x.sum()
print(sum)