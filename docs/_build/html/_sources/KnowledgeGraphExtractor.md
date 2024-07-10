# Knowledge Graph Extractor

## Introduction

The Knowledge Graph Extractor is a sophisticated tool designed for extracting relational sub-spaces from the interactions between query and key vectors (Q-K interaction) within transformer models. By projecting token embeddings onto these sub-spaces, it facilitates the identification of various types of relations among tokens. This tool is invaluable for tasks that require a deep understanding of the relationships within text, such as semantic analysis, information retrieval, and knowledge graph construction.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.kg_extractor
   :members:
   :undoc-members:
   :show-inheritance:
```

- `extract_relational_subspaces(layer_idx, top_k=3)`: Extracts the top `k` relational sub-spaces from a specified layer. These sub-spaces are derived from the Q-K interactions and represent distinct relational patterns that can be observed in the data.

- `project_token_embeddings(token_ids, subspaces)`: Projects the embeddings of specified tokens onto the extracted relational sub-spaces. This projection helps in visualizing and understanding the positioning of tokens within the relational context defined by the sub-spaces.

- `identify_relation_types(token_ids, subspaces, threshold=0.7)`: Identifies the types of relations between tokens by analyzing their projections in the relational sub-spaces. Relations are determined based on a similarity threshold, allowing for the classification of token pairs into different relational categories.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage the Knowledge Graph Extractor for their specific needs.*

## Resources

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): This resource offers a comprehensive exploration of the mathematical and theoretical foundations of transformer models. It provides valuable insights into the mechanisms of Q-K interactions and the principles behind the extraction and analysis of relational sub-spaces, making it an essential read for anyone looking to deepen their understanding of knowledge graph extraction techniques.