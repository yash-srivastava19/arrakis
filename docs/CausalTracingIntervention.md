# Causal Tracing Intervention

## Introduction

Causal Tracing Intervention is a methodology designed to dissect and understand the causal mechanisms within transformer models. The core idea behind this approach focuses on tracing the influence of specific tokens or neurons on the model's output, allowing for a granular analysis of how input features affect predictions. It is particularly useful for performing targeted intervention on activations and observe the impoact on the final output.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.causal_tracing_intervention
   :members:
   :undoc-members:
   :show-inheritance:
```

- `trace_token_probability(input_ids, target_idx)`: This method calculates the probability distribution of a target token given a sequence of input IDs. It is useful for understanding how changes in the input sequence affect the prediction of a specific token, providing insights into the model's attention and prediction mechanisms.

- `intervene_on_neuron(input_ids, layer_idx, neuron_idx, scale=0)`: Allows for the intervention on a specific neuron within a given layer of the transformer model. By adjusting the neuron's activation (scaling it by a specified factor), this method can reveal the neuron's causal impact on the model's output. It is a powerful tool for causal analysis and understanding the role of individual neurons in the model's computations.

## Example Usage

*This section will provide practical examples and code snippets demonstrating how to utilize the methods listed above. It will illustrate their application in real-world scenarios, helping users to effectively leverage the Causal Tracing Intervention methodology.*

## Resources

- [Causal Scrubbing - A method for rigorously testing interpretability hypotheses](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)

- [Towards Vision Langauge Mechanistic Interpretability: A Causal Tracing Tool for BLIP](https://openaccess.thecvf.com/content/ICCV2023W/CLVL/papers/Palit_Towards_Vision-Language_Mechanistic_Interpretability_A_Causal_Tracing_Tool_for_BLIP_ICCVW_2023_paper.pdf)

- [Causal Tracing - Neel Nanda](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)