# Update: Everything works.
import torch
import torch.nn as nn
from .base_interpret import BaseInterpretabilityTool

class SparsityAnalyzer(BaseInterpretabilityTool):
    """Analyzes the sparsity of the network."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    # Works(Tested with GPT2)
    def compute_activation_sparsity(self, id_list, layer):
        """Computes the sparsity of the activations of the layer."""
        total_neurons = 0
        active_neurons = 0
        for input_ids in id_list:
            _, cache = self.model(input_ids)
            acts = cache[layer].squeeze() > 0
            total_neurons += acts.numel()
            active_neurons += acts.sum().item()
        return 1 - (active_neurons / total_neurons)

    # Works.
    def prune_network(self, sparsity_threshold=0.9):
        """Prunes the network based on the sparsity threshold."""
        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.LayerNorm):  # I don't think this is making sense to me.
                W = module.weight.data
                # print(self.model.model_attrs.get_lin_ff_type())
                if self.model.model_attrs.get_lin_ff_type() in name:  # Don't prune the unembedding matrix. Check with our notation.
                    continue
                row_norms = torch.norm(W, dim=0)
                mask = row_norms > torch.quantile(row_norms, sparsity_threshold)
                module.weight.data = W * mask.unsqueeze(0)

    #Works.
    def find_polysemantic_neurons(self, threshold=1):
        """Finds the polysemantic neurons in the network."""

        # By deafualt, we are testing for MLP layer.
        # print(self.model.model.state_dict().keys())
        # input()
        W = self.model.model_attrs.get_mlp_out(0).weight # Let's see if it makes sense :(
        # W = self.model.model.state_dict()[layer]
        U, S, V = torch.svd(W)
        S_norm = S / S.sum()
        polysemantic = []
        for i in range(V.shape[1]):
            if torch.sum(S_norm[:i+1]) < 0.5 and S_norm[i] > 0.1:
                polysemantic.append(i)
        return polysemantic if len(polysemantic) >= threshold else []
