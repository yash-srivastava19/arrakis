# Residual Tools

## Introduction

Residual Tools is a suite designed to analyze and interpret the role of residual connections within transformer models. It provides methods for decomposing residuals to understand their contribution to the model's output, tracking the movement of information across layers, and attributing features to specific inputs.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.residual_tools
   :members:
   :undoc-members:
   :show-inheritance:
```

- `residual_decomposition(activations, layer_idx)`: Decomposes the activations at a specified layer to analyze the contribution of residual connections. This method is essential for understanding how residuals influence the layer's output.

- `residual_movement(input_ids, layer_idx1, layer_idx2)`: Tracks the movement of information through residual connections between two layers. It helps in understanding how information is propagated and transformed across the model.

- `feature_attribution(input_ids, target_ids)`: Attributes the model's output features to specific input tokens. This method is crucial for interpreting the model's decision-making process and understanding how input information contributes to the final output.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage Residual Tools for their specific needs.*

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An essential resource for those interested in the mathematical and theoretical foundations of transformer models. It offers a comprehensive exploration of residual connections, making it an invaluable resource for researchers and practitioners looking to deepen their understanding of transformer model dynamics.