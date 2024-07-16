# Superposition Disentangler

## Introduction

Superposition Disentangler is a tool designed for analyzing and disentangling the superposition of concepts within the activations of transformer models. It focuses on identifying distinct conceptual components within the activations and provides methods for projecting, reconstructing, and interpolating these components.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.superposition_disentangler
   :members:
   :undoc-members:
   :show-inheritance:
```

- `compute_activation_covariance(ids_list, layer)`: Computes the covariance matrix of activations for a given list of IDs at a specified layer. This method is essential for understanding the relationships between different activations and identifying superposed concepts.

- `disentangle_superposition(cov_matrix, n_concepts=10)`: Disentangles the superposition of concepts within the activations by extracting the top `n` conceptual components based on the covariance matrix. This method is crucial for isolating and analyzing individual concepts within the model's activations.

- `project_activations(activations, concepts)`: Projects the activations onto the identified conceptual components, allowing for the analysis of the contribution of each concept to the activations.

- `reconstruct_activations(projections, concepts)`: Reconstructs the activations from their projections on the conceptual components. This method is useful for understanding how the superposed concepts combine to form the original activations.

- `interpolate_concepts(concept1, concept2, alpha)`: Interpolates between two concepts to explore the continuous space of conceptual representations within the model. This method is valuable for examining the transitions between different concepts and understanding their relationships.

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

@exp.use_tools("superposition") # the tool name to be used.
def test_superposition_disentangler(input_ids, layer_idx, superposition): # same as tool name, extra arg is passed.
   act_covariance = superposition.compute_activation_covariance([input_ids], layer_idx)
   dis_supoerpos = superposition.disentangle_superposition(act_covariance)
   return {
      "act_covariance": act_covariance,
      "dis_supoerpos": dis_supoerpos
   }

# Driver code, call the function based on whatever arguments you want!
test_superposition_disentangler(input_ids, 'layers.0.self_attn.hook_result_post') # one such example. Change as needed!
```

## Resources

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)