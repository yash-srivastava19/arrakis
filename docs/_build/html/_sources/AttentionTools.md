# Attention Tools

## Introduction
Attention Tools is a comprehensive framework designed to facilitate the understanding, analysis, and manipulation of attention patterns within transformer models. It encompasses a variety of methods aimed at analyzing, visualizing, and altering the attention mechanisms that are central to the functionality of these models.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.attention_tools
   :members:
   :undoc-members:
   :show-inheritance:
```

- `attention_patterns(input_ids, layer_idx, head_idx)`: This method allows for the extraction and visualization of attention patterns between tokens in a specific layer and head of the transformer model. It is instrumental in understanding how different parts of the input sequence attend to each other.

- `top_attended_ids(input_ids, layer_idx, head_idx, num_tokens=5)`: Identifies and returns the IDs of the top `num_tokens` that receive the highest attention in a specified layer and head. This method is useful for pinpointing key tokens that play a significant role in the model's decision-making process.

- `ablate_head(layer_idx, head_idx)`: Temporarily disables or "ablates" a specific attention head within a layer. This method is useful for assessing the impact of individual heads on the model's overall performance and understanding.

- `merge_heads(layer_idx, head_idx1, head_idx2, alpha=0.5)`: Combines the attention patterns of two heads within the same layer, weighted by `alpha`. This can reveal how different attention mechanisms complement each other and contribute to the model's ability to process information.

## Example Usage

## Resources
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
