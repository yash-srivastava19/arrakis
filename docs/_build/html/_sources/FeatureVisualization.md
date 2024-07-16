# Feature Visualization

## Introduction

Feature Visualization is a methodology designed to dissect and understand the causal mechanisms within transformer models. It focuses on tracing the influence of specific tokens or neurons on the model's output, allowing for a granular analysis of how input features affect predictions. This approach is particularly useful for performing targeted intervention on activations to observe the impact on the final output.

## API Reference

```{eval-rst}  
.. automodule:: core_arrakis.feature_viz
   :members:
   :undoc-members:
   :show-inheritance:
```

## Example Usage

```python
# imports to run this example
import torch
from arrakis.src.core_arrakis.activation_cache import *
from arrakis.src.bench.base_bench import BaseInterpretabilityBench

config = HookedAutoConfig(name="llama") # keep default values for other args
model = HookedAutoModel(config)

input_ids = torch.randint(0, 50256, (1, 50)) # generate some random tokens(replace with your ids)

# Derive from BaseInterpretabilityBench
class MIExperiment(BaseInterpretabilityBench):
   def __init__(self, model, save_dir="experiments"):
      super().__init__(model, save_dir)

exp = MIExperiment(model) # create an `exp` object.

@exp.use_tools("feature_viz") # the tool name to be used.
def test_feature_viz(feature_viz): # same as tool name, extra arg is passed.
    top_k = feature_viz.activate_neuron([input_ids],"layers.1.self_attn.q_proj.hook_result_pre", 0, top_k=10)
    interpolations = feature_viz.neuron_interpolation(input_ids, "model.layers.1", 0, 1, num_interpolations=10, max_new_tokens=10)
    return {"top_k": top_k, "interpolations": interpolations}

# Driver code, call the function based on whatever arguments you want!
test_feature_viz() # one such example. Change as needed!
```

## Resources

- [Feature Visualization - Distill Pub](https://distill.pub/2017/feature-visualization/)

- [OpenAI Microscope](https://microscope.openai.com/models)

- [Lucid, a feature visualization library for images](https://github.com/tensorflow/lucid/)