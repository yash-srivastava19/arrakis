# Logit Attribution

## Introduction

Logit Attributor is a specialized tool designed for attributing the logits of a transformer model's output to specific input tokens. This tool is crucial for understanding how different parts of the input contribute to the model's final decision, making it invaluable for interpretability and analysis of transformer models.

## Methods

```{eval-rst}  
.. automodule:: core_arrakis.logit_attribution
   :members:
   :undoc-members:
   :show-inheritance:
```

- `logit_attribution(input_ids, target_idx)`: This method calculates the contribution of each input token to the logit of a target index. It is essential for dissecting the model's decision-making process and understanding which tokens have the most significant impact on the output.

- `track_token_circulation(input_ids, target_token, n_layers=5)`: Tracks the circulation of a target token's influence across a specified number of layers. This method provides insights into how information about a specific token is propagated through the model, affecting the final logits.

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

@exp.use_tools("logit") # the tool name to be used.
def test_logit_attributor(input_ids, target_idx, n_layers, logit): # same as tool name, extra arg is passed.
   direct_moves = logit.track_token_circulation(input_ids, target_idx, n_layers)
   logit_attribution = logit.logit_attribution(input_ids, target_idx)
   return {
      "direct_moves": direct_moves,
      "logit_attribution": logit_attribution
   }

# Driver code, call the function based on whatever arguments you want!
test_logit_attributor(input_ids, 0, 2) # one such example. Change as needed!
```

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An invaluable resource for those looking to delve deeper into the mathematical and theoretical aspects of transformer models. It offers a comprehensive exploration of logit attribution and token circulation, providing a solid foundation for researchers and practitioners looking to apply these techniques.