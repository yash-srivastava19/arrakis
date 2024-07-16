# Knowledge Prober

## Introduction

Knowledge Prober is a tool designed for analyzing the polysemy of tokens and probing transformer models for concept-specific neuron activations. It enables the extraction of polysematic scores by evaluating the nearest neighbors in a specific layer of the model, and it facilitates the probing of the model on a concept neuron to obtain relevant scores. This tool is particularly useful for understanding the nuanced ways in which transformer models encode and differentiate between various meanings of the same word or concept.

## Methods

```{eval-rst}  
.. automodule:: core_arrakis.knowledge_prober
   :members:
   :undoc-members:
   :show-inheritance:
```

- `polysemantic_score(layer_idx, k_neighbors=50)`: Calculates the polysematic score for tokens by examining their k nearest neighbors in the embedding space of a specified layer. This score helps in understanding the diversity of contexts in which a token is used across different parts of the dataset.

- `knowledge_probe(ids, concept_neuron)`: Probes the model for activations related to a specific concept neuron, given a set of input IDs. This method is useful for identifying how strongly a particular concept is represented within the model's internal representations.

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

@exp.use_tools("k_prober") # the tool name to be used.
def test_knowledge_prober(ids, layer_idx, neuron_idx, k_prober): # same as tool name, one extra arg is passed.
   pscore = k_prober.polysemantic_score(layer_idx, 2)
   true_false = k_prober.knowledge_probe([(ids, ids)], (layer_idx, neuron_idx))
   return {
      "k_probe": true_false,
      "polysemantic_score": pscore
   }

# Driver code, call the function based on whatever arguments you want!
test_knowledge_prober(input_ids, 0, 0) # one such example. Change as needed!
```

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An essential resource for those interested in the mathematical and theoretical foundations of transformer models. It provides a comprehensive exploration of the principles behind knowledge probing and polysematic score analysis, making it an invaluable resource for researchers and practitioners looking to deepen their understanding of how transformer models encode and process knowledge.