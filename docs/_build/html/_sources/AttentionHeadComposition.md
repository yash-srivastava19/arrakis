# Attention Head Composition

## Introduction
The concept of Attention Head Composition is a pivotal aspect of understanding the inner workings of transformer models. Originating from the Transformer Circuits Thread, this framework sheds light on the intricate operations within transformers, with a special focus on the composition of attention heads.

Attention heads are fundamental components of transformer models, responsible for capturing dependencies and relationships within the input data. 
Examining the attention weights and patterns of the individual attentions heads, we ca dedeuce what parts of the input the model is focusing on to make a particular prediction.

Induction heads, which are a product of attention head composition, play a crucial role in enabling transformers to perform in-context learning algorithms. These algorithms allow the model to generalize from the input data, making inferences and predictions that extend beyond the immediate context. The composition of attention heads can occur in three distinct manners, each corresponding to a different component of the attention mechanism: keys, queries, and values. This tripartite composition framework allows for a versatile and dynamic approach to modeling complex data relationships.

## API Reference
```{eval-rst}  
.. automodule:: core_arrakis.attention_head_comp
   :members:
   :undoc-members:
   :show-inheritance:
```

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

@exp.use_tools("composer") # tool name to be used.
def test_attention_composer(head1, head2, alpha, composer): # same as tool name, one extra arg is passed.
   hc = composer.head_composition(head1, head2, alpha)
   app = composer.attention_path_patching(input_ids, head1, head2)
   return {
      "head_composition": hc,
      "attention_path_patching": app
   }

# Driver code, call the function based on whatever arguments you want!
test_attention_composer((0, 0), (0, 1), 0.5) # one such example. Change as needed!
```

## Resources
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

- [Thoughts on Formalizing Composition](https://www.lesswrong.com/posts/vaHxk9jSfmGbzvztT/thoughts-on-formalizing-composition)