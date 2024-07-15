# Update: Works as intended.
import torch
from .base_interpret import BaseInterpretabilityTool 

class KnowledgeProber(BaseInterpretabilityTool):
    """Probes the model for knowledge."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model
    
    #Works 
    def polysemantic_score(self, layer_idx, k_neighbors=5):
        """Computes the polysemantic score of the neurons in the layer."""
        # W = self.model.model.h[layer_idx].mlp.c_fc.weight        # This is for gpt style models.
        W = self.model.model_attrs.get_mlp_in(layer_idx).weight #Check whether this is in weight or out weight.
        
        norms = torch.norm(W, dim=0)
        W_norm = W / norms
        
        similarity = W_norm.T @ W_norm
        _, topk = torch.topk(similarity, k_neighbors + 1, dim=1)
        
        scores = []
        for i in range(W.shape[1]):
            neighbors = W_norm[:, topk[i, 1:]]  # Exclude self
            centrality = torch.norm(neighbors.mean(dim=1))
            dispersion = torch.norm(neighbors - neighbors.mean(dim=1, keepdim=True), dim=0).mean()
            scores.append(centrality / dispersion)
        
        return torch.tensor(scores)
    
    # Works.
    def knowledge_probe(self, ids, concept_neuron):
        """Probes the model for knowledge."""
        true_scores = []
        false_scores = []
        
        layer_idx, neuron_idx = concept_neuron
        
        for true_ids, false_ids in ids:
            _, true_cache = self.model(true_ids)  # This is a caveat.
            _, false_cache = self.model(false_ids)

            block_type = self.model.model_attrs.get_block_type() 
            true_act = true_cache[f"{block_type}.{layer_idx}.mlp.hook_result_post"][0, neuron_idx]
            false_act = false_cache[f"{block_type}.{layer_idx}.mlp.hook_result_post"][0, neuron_idx]
            
            true_scores.append(true_act.item())
            false_scores.append(false_act.item())
        
        return true_scores, false_scores
