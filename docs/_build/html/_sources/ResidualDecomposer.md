# Residual Decomposer

## Introduction

Residual Decomposer is a tool designed for analyzing the residual connections within transformer models. It focuses on decomposing the residuals into their constituent components, allowing for a detailed examination of how information is preserved and transformed across layers.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.residual_decomposer
   :members:
   :undoc-members:
   :show-inheritance:
```

- `mlp_basis(layer_idx)`: Identifies the basis components of the Multi-Layer Perceptron (MLP) within a specified layer. This method is crucial for understanding the foundational elements that contribute to the layer's output.

- `decompose_residual(input_ids, layer_idx, top_k=10)`: Decomposes the residual connections at a specified layer into their top `k` components. This method provides insights into the most significant contributors to the residual signal, offering a deeper understanding of information flow and transformation within the model.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage the Residual Decomposer for their specific needs.*

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An essential resource for those interested in the mathematical and theoretical foundations of transformer models. It offers a comprehensive exploration of residual decomposition, making it an invaluable resource for researchers and practitioners looking to deepen their understanding of transformer model dynamics.