import torch
from .base_interpret import BaseInterpretabilityTool
class CausalTracer(BaseInterpretabilityTool):
    """ locating and Editing Factored Cognition."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    # Works on GPT2 style models.
    def trace_token_probability(self, input_ids, target_idx):
        """Traces the probability of the target token through the layers."""
        _, cache = self.model(input_ids)
        probs = []

        for layer_idx in range(len(self.model.model.h)):  # h is for gpt style models.
            resid = cache[f"blocks.{layer_idx}.hook_resid_post"].clone().requires_grad_(True)
            logits = cache['logits'].clone().requires_grad_(True)

            prob = torch.softmax(logits, dim=-1)[0, target_idx]
            probs.append(prob.item())

            # Compute gradient for causal tracing
            prob.backward()
            grad = resid.grad.norm(dim=-1).squeeze().cpu()
            yield prob.item(), grad

        probs.append(torch.softmax(logits, dim=-1)[0, target_idx].item())
        yield probs[-1], None

    # Works on GPT2 style models.
    def intervene_on_neuron(self, input_ids, layer_idx, neuron_idx, scale=0):
        """Intervenes on a neuron in the model."""
        def hook_fn(module, input, output):
            output[:, neuron_idx] = scale * output[:, neuron_idx]
            return output

        handle = self.model.model_attrs.get_mlp(layer_idx).register_forward_hook(hook_fn)
        output = self.model.model(input_ids)
        handle.remove()
        return output
