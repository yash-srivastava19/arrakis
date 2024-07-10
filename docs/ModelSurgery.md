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
from pre_trained_models_hf import model
from model_surgery import ModelSurgery

ms = ModelSurgery(model)

# Temporarily delete layers 0 and 1
with ms.delete_layers([0, 1]):
        # Perform operations with the modified model
        pass

# Temporarily permute layers 0 and 1
with ms.permute_layers([1, 0]):
        # Perform operations with the modified model
        pass

# Temporarily replace layers 0 and 1 with new linear layers
with ms.replace_layers([0, 1], [torch.nn.Linear(128, 128), torch.nn.Linear(128, 128)]):
        # Perform operations with the modified model
        pass
```
## Resources

- [Tuned Lens - Eliciting Latent Prediction from Transformers](https://arxiv.org/pdf/2303.08112)
