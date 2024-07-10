# Knowledge Prober

## Introduction

Knowledge Prober is a tool designed for analyzing the polysemy of tokens and probing transformer models for concept-specific neuron activations. It enables the extraction of polysematic scores by evaluating the nearest neighbors in a specific layer of the model, and it facilitates the probing of the model on a concept neuron to obtain relevant scores. This tool is particularly useful for understanding the nuanced ways in which transformer models encode and differentiate between various meanings of the same word or concept.

## Methods

```{eval-rst}  
.. automodule:: core_arrakis.knowledge_prober
   :members:
   :undoc-members:
   :show-inheritance:
```

- `polysemantic_score(layer_idx, k_neighbors=50)`: Calculates the polysematic score for tokens by examining their k nearest neighbors in the embedding space of a specified layer. This score helps in understanding the diversity of contexts in which a token is used across different parts of the dataset.

- `knowledge_probe(ids, concept_neuron)`: Probes the model for activations related to a specific concept neuron, given a set of input IDs. This method is useful for identifying how strongly a particular concept is represented within the model's internal representations.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage the Knowledge Prober for their specific needs.*

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): An essential resource for those interested in the mathematical and theoretical foundations of transformer models. It provides a comprehensive exploration of the principles behind knowledge probing and polysematic score analysis, making it an invaluable resource for researchers and practitioners looking to deepen their understanding of how transformer models encode and process knowledge.