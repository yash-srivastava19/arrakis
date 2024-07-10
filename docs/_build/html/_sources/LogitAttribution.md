# Logit Attribution

## Introduction

Logit Attributor is a specialized tool designed for attributing the logits of a transformer model's output to specific input tokens. This tool is crucial for understanding how different parts of the input contribute to the model's final decision, making it invaluable for interpretability and analysis of transformer models.

## Methods

```{eval-rst}  
.. automodule:: core_arrakis.logit_attribution
   :members:
   :undoc-members:
   :show-inheritance:
```

- `logit_attribution(input_ids, target_idx)`: This method calculates the contribution of each input token to the logit of a target index. It is essential for dissecting the model's decision-making process and understanding which tokens have the most significant impact on the output.

- `track_token_circulation(input_ids, target_token, n_layers=5)`: Tracks the circulation of a target token's influence across a specified number of layers. This method provides insights into how information about a specific token is propagated through the model, affecting the final logits.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage the Logit Attributor for their specific needs.*

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An invaluable resource for those looking to delve deeper into the mathematical and theoretical aspects of transformer models. It offers a comprehensive exploration of logit attribution and token circulation, providing a solid foundation for researchers and practitioners looking to apply these techniques.