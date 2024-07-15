import torch
import torch.nn as  nn
from src.core_arrakis.activation_cache import register_hooks

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    @register_hooks    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
sm = SimpleModule()
input_tensor = torch.randn(1, 10)
print(sm(input_tensor))