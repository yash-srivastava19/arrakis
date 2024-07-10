# LogitLens

## Introduction

LogitLens is a tool designed for in-depth analysis of the contributions of different layers and tokens to the logits of a transformer model's output. It extends the capabilities of the Knowledge Prober by focusing on layer-specific attributions and providing a detailed view of how input tokens influence the model's predictions at various depths.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.logit_lens
   :members:
   :undoc-members:
   :show-inheritance:
```

- `layer_attribution(input_ids, target_idx)`: Calculates the contribution of each layer to the logit of a target index. This method is crucial for understanding the role of different layers in the model's decision-making process.

- `logit_lens(input_ids, target_idx, layer_idx)`: Provides a focused analysis of how tokens contribute to the logits at a specific layer. It is essential for dissecting the model's internal mechanisms and understanding the influence of token representations at various stages.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage LogitLens for their specific needs.*

## Resources

- [Interpreting GPT - The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

- [Understanding SAE features with Logit Lens](https://www.lesswrong.com/posts/qykrYY6rXXM7EEs8Q/understanding-sae-features-with-the-logit-lens)