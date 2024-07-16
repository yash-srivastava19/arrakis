# Read Write Heads

## Introduction

ReadWriteHeads is a tool designed for identifying and analyzing the read and write heads within transformer models. It focuses on distinguishing between heads that predominantly "write" information to the context and those that "read" from it, facilitating a deeper understanding of the model's information processing dynamics.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.read_write_heads
   :members:
   :undoc-members:
   :show-inheritance:
```


- `identify_write_heads(layer_idx, threshold=0.8)`: Identifies the heads within a specified layer that act primarily as write heads, based on a threshold for their contribution to the overall context.

- `identify_read_heads(layer_idx, dim_idx, threshold=0.8)`: Identifies the heads within a specified layer that act primarily as read heads, focusing on their ability to extract information from the context.

- `trace_information_flow(input_ids, src_token, dst_token, n_layers=2)`: Traces the flow of information from a source token to a destination token across a specified number of layers, highlighting the roles of read and write heads in this process.

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

@exp.use_tools("write_read") # the tool name to be used.
def test_read_write(src_idx, write_read):
   write_heads = write_read.identify_write_heads(0)  # Example layer
   read_heads = write_read.identify_read_heads(1, dim_idx=src_idx)  # Example layer
   return {
      "write_heads": write_heads,
      "read_heads": read_heads
   }

# Driver code, call the function based on whatever arguments you want!
test_read_write(0) # one such example. Change as needed!
```

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An essential resource for those interested in the mathematical and theoretical foundations of transformer models. It provides a comprehensive exploration of the roles of read and write heads in information processing, making it an invaluable resource for researchers and practitioners looking to deepen their understanding of transformer model dynamics.