# Residual Decomposer

## Introduction

Residual Decomposer is a tool designed for analyzing the residual connections within transformer models. It focuses on decomposing the residuals into their constituent components, allowing for a detailed examination of how information is preserved and transformed across layers.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.residual_decomposer
   :members:
   :undoc-members:
   :show-inheritance:
```

- `mlp_basis(layer_idx)`: Identifies the basis components of the Multi-Layer Perceptron (MLP) within a specified layer. This method is crucial for understanding the foundational elements that contribute to the layer's output.

- `decompose_residual(input_ids, layer_idx, top_k=10)`: Decomposes the residual connections at a specified layer into their top `k` components. This method provides insights into the most significant contributors to the residual signal, offering a deeper understanding of information flow and transformation within the model.

## Example Usage

```python
# imports to run this example
import torch
from arrakis.src.core_arrakis.activation_cache import *
from arrakis.src.bench.base_bench import BaseInterpretabilityBench

config = HookedAutoConfig(name="llama") # keep default values for other args
model = HookedAutoModel(config)

input_ids = torch.randint(0, 50256, (1, 50)) # generate some random tokens(replace with your ids)

# Derive from BaseInterpretabilityBench
class MIExperiment(BaseInterpretabilityBench):
   def __init__(self, model, save_dir="experiments"):
      super().__init__(model, save_dir)

exp = MIExperiment(model) # create an `exp` object.

@exp.use_tools("decomposer") # the tool name to be used.
def test_residual_decomposer(input_ids, layer_idx, decomposer): # same as tool name, extra arg is passed.
   rd = decomposer.decompose_residual(input_ids, layer_idx)
   return rd

# Driver code, call the function based on whatever arguments you want!
test_residual_decomposer(input_ids, 0) # one such example. Change as needed!
```

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An essential resource for those interested in the mathematical and theoretical foundations of transformer models. It offers a comprehensive exploration of residual decomposition, making it an invaluable resource for researchers and practitioners looking to deepen their understanding of transformer model dynamics.