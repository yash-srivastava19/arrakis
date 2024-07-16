# Knowledge Graph Extractor

## Introduction

The Knowledge Graph Extractor is a sophisticated tool designed for extracting relational sub-spaces from the interactions between query and key vectors (Q-K interaction) within transformer models. By projecting token embeddings onto these sub-spaces, it facilitates the identification of various types of relations among tokens. This tool is invaluable for tasks that require a deep understanding of the relationships within text, such as semantic analysis, information retrieval, and knowledge graph construction.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.kg_extractor
   :members:
   :undoc-members:
   :show-inheritance:
```

- `extract_relational_subspaces(layer_idx, top_k=3)`: Extracts the top `k` relational sub-spaces from a specified layer. These sub-spaces are derived from the Q-K interactions and represent distinct relational patterns that can be observed in the data.

- `project_token_embeddings(token_ids, subspaces)`: Projects the embeddings of specified tokens onto the extracted relational sub-spaces. This projection helps in visualizing and understanding the positioning of tokens within the relational context defined by the sub-spaces.

- `identify_relation_types(token_ids, subspaces, threshold=0.7)`: Identifies the types of relations between tokens by analyzing their projections in the relational sub-spaces. Relations are determined based on a similarity threshold, allowing for the classification of token pairs into different relational categories.

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

@exp.use_tools("kg_extractor") # the tool name to be used
def test_kg_extractor(token_ids, kg_extractor): # # same as tool name, one extra arg is passed.
   top_subspaces = kg_extractor.extract_relational_subspaces(0)
   irt = kg_extractor.identify_relation_types(token_ids, top_subspaces)
   return {
      "top_subspaces": top_subspaces,
      "identify_relation_types": irt
   }

# Driver code, call the function based on whatever arguments you want!
test_kg_extractor(input_ids) # one such example. Change as needed!
```

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): This resource offers a comprehensive exploration of the mathematical and theoretical foundations of transformer models. It provides valuable insights into the mechanisms of Q-K interactions and the principles behind the extraction and analysis of relational sub-spaces, making it an essential read for anyone looking to deepen their understanding of knowledge graph extraction techniques.