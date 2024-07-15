from arrakis.src.bench.base_bench import BaseInterpretabilityBench
from arrakis.src.graph.base_graph import *
from arrakis.src.core_arrakis.activation_cache import *
from arrakis.src.core_arrakis.feature_viz import *

# Step1: Create a HookedAutoConfig and HookedAutoModel
config = HookedAutoConfig(name="llama", vocab_size=50256, hidden_size=8, intermediate_size=2, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4)
model = HookedAutoModel(config)

input_ids = torch.randint(0, 50256, (1, 10)) # Random input_ids

# Step2: Create an experiment
class AttentionVisulazierExperiment(BaseInterpretabilityBench):
    def __init__(self, model, save_dir="experiments"):
        super().__init__(model, save_dir)
        self.tools.update({"feature_viz": FeatureViz(model)})

# Step3: Set the plotting library
exp = AttentionVisulazierExperiment(model)
exp.set_plotting_lib(MatplotlibWrapper)


# Step4: Test the feature_viz(or any tool you want!) tool
@exp.use_tools("feature_viz")
def test_feature_viz(feature_viz):
    top_k = feature_viz.activate_neuron([input_ids],"layers.1.self_attn.q_proj.hook_result_pre", 0, top_k=10)
    interpolations = feature_viz.neuron_interpolation(input_ids, "model.layers.1", 0, 1, num_interpolations=10, max_new_tokens=10)
    return {"top_k": top_k, "interpolations": interpolations}

print(test_feature_viz()["interpolations"])