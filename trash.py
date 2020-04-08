import torch
import torch.nn as nn
from model import DCGenerator,DCDiscriminator

G = DCGenerator(input_dim=20,image_size=64)
D = DCDiscriminator(image_size=64)
input = torch.randn(1,20,1,1)
output = G(input)
print("changemodel")
output = D(output)
print(nn.Sigmoid()(output))
