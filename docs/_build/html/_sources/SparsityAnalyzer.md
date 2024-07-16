# Sparsity Analyzer

## Introduction

Sparsity Analyzer is a tool designed for analyzing the sparsity of activations within transformer models. It provides methods for computing the sparsity of activations, pruning the network based on sparsity thresholds, and identifying neurons that exhibit polysemy across different contexts.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.sparsity_analyzer
   :members:
   :undoc-members:
   :show-inheritance:
```

- `compute_activation_sparsity(id_list, layer)`: Computes the sparsity of activations for a given list of IDs at a specified layer. This method is essential for understanding the distribution of activation values and identifying sparse activations.

- `prune_network(sparsity_threshold=0.9)`: Prunes the network to remove connections or neurons with activation sparsity above a specified threshold. This method is useful for optimizing the model by reducing the complexity of the network.

- `find_polysemantic_neurons(layer, threshold=3)`: Identifies neurons within a specified layer that exhibit polysemy, based on a threshold for the diversity of their activation patterns. This method is crucial for understanding the multifunctionality of neurons in processing different types of information.

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

@exp.use_tools("sparsity") # the tool name to be used.
def test_sparsity_analyzer(input_ids, layer="layers.1.self_attn.hook_result_post",  sparsity=None): # same as tool name, extra arg is passed.
   act_sparsity = sparsity.compute_activation_sparsity([input_ids], layer)
   find_ps_neu = sparsity.find_polysemantic_neurons(3)
   prune_network = sparsity.prune_network(0.9)
   return {
      "act_sparsity": act_sparsity,
      "find_ps_neu": find_ps_neu,
      "prune_network": prune_network
   }

# Driver code, call the function based on whatever arguments you want!
test_sparsity_analyzer(input_ids, layer="layers.0.self_attn.hook_result_post") # one such example. Change as needed!
```

## Resources

- [An intuitive explanation of Sparse Auroencoders for MI](https://www.lesswrong.com/posts/CJPqwXoFtgkKPRay8/an-intuitive-explanation-of-sparse-autoencoders-for)