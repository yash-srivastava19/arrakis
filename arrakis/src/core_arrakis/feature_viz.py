# Test this.
import copy
import torch
import torch.nn as  nn
from .base_interpret import BaseInterpretabilityTool
class FeatureViz(BaseInterpretabilityTool):
    """Visulizing what neurons or features are sensitive to in a model."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model
    
    def activate_neuron(self, input_ids, layer_name, neuron_idx, top_k=10):
        """Find inputs that most activate a given neuron in a given layer of the model."""
        max_activations = []
        for ids in input_ids:
            _, cache = self.model(ids)
            activations = cache[layer_name][:, :, neuron_idx]
            max_act, max_idx = torch.max(activations, dim=1)
            max_activations.append(max_act)
        
        return sorted(max_activations, reverse=True)[:top_k]
    
    def neuron_interpolation(self, input_ids, layer_name, neuron_idx1, neuron_idx2, num_interpolations=10, max_new_tokens=10):
        """Interpolate between two neurons to understand their sematic space."""
        alphas = torch.linspace(0, 1, num_interpolations)
        interpolations = []

        for al in alphas:
            model_copy = copy.deepcopy(self.model) # we can use a context manager as well.
            layer = dict(model_copy.named_modules())[layer_name]

            if isinstance(layer, nn.Linear):
                layer.weight.data[:, neuron_idx1] = (1 - al) * self.model.state_dict()[f"{layer_name}.weight"][:, neuron_idx1] + al * self.model.state_dict()[f"{layer_name}.weight"][:, neuron_idx2]
        
            output_ids = model_copy(input_ids)
            interpolations.append(output_ids)
        
        return interpolations