# LogitLens

## Introduction

LogitLens is a tool designed for in-depth analysis of the contributions of different layers and tokens to the logits of a transformer model's output. It extends the capabilities of the Knowledge Prober by focusing on layer-specific attributions and providing a detailed view of how input tokens influence the model's predictions at various depths.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.logit_lens
   :members:
   :undoc-members:
   :show-inheritance:
```

- `layer_attribution(input_ids, target_idx)`: Calculates the contribution of each layer to the logit of a target index. This method is crucial for understanding the role of different layers in the model's decision-making process.

- `logit_lens(input_ids, target_idx, layer_idx)`: Provides a focused analysis of how tokens contribute to the logits at a specific layer. It is essential for dissecting the model's internal mechanisms and understanding the influence of token representations at various stages.

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

@exp.use_tools("logit_lens") # the tool name to be used.
def test_logit_lens(input_ids, target_idx, layer_idx, logit_lens): # same as tool name, extra arg is passed.
   ll = logit_lens.logit_lens(input_ids, target_idx, layer_idx)
   la = logit_lens.layer_attributions(input_ids, target_idx)
   return {
      "logit_lens": ll,
      "layer_attributions": la
   }

# Driver code, call the function based on whatever arguments you want!
test_logit_lens(input_ids, 0, 0) # one such example. Change as needed!
```

## Resources

- [Interpreting GPT - The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

- [Understanding SAE features with Logit Lens](https://www.lesswrong.com/posts/qykrYY6rXXM7EEs8Q/understanding-sae-features-with-the-logit-lens)