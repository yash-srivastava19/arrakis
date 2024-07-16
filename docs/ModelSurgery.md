# Model Surgery

## Introduction

Model Surgery is a Python class designed for performing temporary modifications on transformer models. It allows for the deletion, permutation, and replacement of layers within the model to facilitate experimentation and analysis. This tool is particularly useful for researchers and practitioners looking to understand the inner workings of transformer models or to enhance their performance.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.model_surgery
   :members:
   :undoc-members:
   :show-inheritance:
```

- `get_final_norm(self)` : Retrieves the final layer normalization module from the model.
- `get_transformer_layers(self)` : Identifies and returns the path to the transformer layers within the model.
- `delete_layers(self, indices)` : Temporarily deletes specified layers from the model.
- `permute_layers(self, indices)` : Temporarily permutes the layers of the model according to the specified order.
- `replace_layers(self, indices, replacements)` : Temporarily replaces specified layers with new ones.

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

@exp.use_tools("surgery") # the name of the tool to be used.
def test_model_surgery(indicies, replacements, surgery): # same name as tool name, extra argument is passed.
   model = surgery.get_model()
   print("Before surgery: \n\n",model)

   with surgery.replace_layers(indicies, replacements):
       print("After surgery: \n\n",model)
        # print(surgery.permute_layers([0, 1]))
   with surgery.permute_layers([0, 1]):
       print("After surgery: \n\n",model)

    with surgery.delete_layers([0, 1]):
       print("After surgery: \n\n",model)

# Driver code, call the function based on whatever arguments you want!
test_model_surgery([0, 1], [model.model_attrs.get_mlp_out(0), model.model_attrs.get_mlp_out(1)]) # one such example. Change as needed!
```
## Resources

- [Tuned Lens - Eliciting Latent Prediction from Transformers](https://arxiv.org/pdf/2303.08112)
