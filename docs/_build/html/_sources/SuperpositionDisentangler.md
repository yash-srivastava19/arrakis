# Superposition Disentangler

## Introduction

Superposition Disentangler is a tool designed for analyzing and disentangling the superposition of concepts within the activations of transformer models. It focuses on identifying distinct conceptual components within the activations and provides methods for projecting, reconstructing, and interpolating these components.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.superposition_disentangler
   :members:
   :undoc-members:
   :show-inheritance:
```

- `compute_activation_covariance(ids_list, layer)`: Computes the covariance matrix of activations for a given list of IDs at a specified layer. This method is essential for understanding the relationships between different activations and identifying superposed concepts.

- `disentangle_superposition(cov_matrix, n_concepts=10)`: Disentangles the superposition of concepts within the activations by extracting the top `n` conceptual components based on the covariance matrix. This method is crucial for isolating and analyzing individual concepts within the model's activations.

- `project_activations(activations, concepts)`: Projects the activations onto the identified conceptual components, allowing for the analysis of the contribution of each concept to the activations.

- `reconstruct_activations(projections, concepts)`: Reconstructs the activations from their projections on the conceptual components. This method is useful for understanding how the superposed concepts combine to form the original activations.

- `interpolate_concepts(concept1, concept2, alpha)`: Interpolates between two concepts to explore the continuous space of conceptual representations within the model. This method is valuable for examining the transitions between different concepts and understanding their relationships.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage the Superposition Disentangler for their specific needs.

## Resources

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)