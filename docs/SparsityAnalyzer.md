# Sparsity Analyzer

## Introduction

Sparsity Analyzer is a tool designed for analyzing the sparsity of activations within transformer models. It provides methods for computing the sparsity of activations, pruning the network based on sparsity thresholds, and identifying neurons that exhibit polysemy across different contexts.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.sparsity_analyzer
   :members:
   :undoc-members:
   :show-inheritance:
```

- `compute_activation_sparsity(id_list, layer)`: Computes the sparsity of activations for a given list of IDs at a specified layer. This method is essential for understanding the distribution of activation values and identifying sparse activations.

- `prune_network(sparsity_threshold=0.9)`: Prunes the network to remove connections or neurons with activation sparsity above a specified threshold. This method is useful for optimizing the model by reducing the complexity of the network.

- `find_polysemantic_neurons(layer, threshold=3)`: Identifies neurons within a specified layer that exhibit polysemy, based on a threshold for the diversity of their activation patterns. This method is crucial for understanding the multifunctionality of neurons in processing different types of information.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage the Sparsity Analyzer for their specific needs.*

## Resources

- [An intuitive explanation of Sparse Auroencoders for MI](https://www.lesswrong.com/posts/CJPqwXoFtgkKPRay8/an-intuitive-explanation-of-sparse-autoencoders-for)