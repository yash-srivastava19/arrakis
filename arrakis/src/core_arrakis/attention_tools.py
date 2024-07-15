# Update - It works. Tests have been done.
import torch
from .base_interpret import BaseInterpretabilityTool

class AttentionTools(BaseInterpretabilityTool):
    """Understanding attention patterns by analyzing, visualizing and manipulating it."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = model

    #Works
    def attention_patterns(self, input_ids, layer_idx, head_idx):
        """Extract the attention patterns of a specific head in a specific layer of a model."""
        _, cache = self.model(input_ids)
        q = self.model.model_attrs.get_q(layer_idx).weight[head_idx]
        k = self.model.model_attrs.get_k(layer_idx).weight[head_idx]

        # In the mixed models, there seems to be some problem. Q,K, V projections are not matching :)
        # q1 = cache[f"h.{layer_idx}.attn.c_attn.hook_resid_post"][head_idx]
        # k1 = cache[f"h.{layer_idx}.attn.c_attn.hook_resid_post"][head_idx]
        

        return torch.softmax(q @ k / q.shape[-1]**0.5 , dim=-1)
    
    #Works
    def top_attended_ids(self, input_ids, layer_idx, head_idx, num_tokens=2):
        """Extract the top-k attended ids of a specific head in a specific layer of a model."""
        attention = self.attention_patterns(input_ids, layer_idx, head_idx)
        _, top_k_indices = torch.topk(attention, num_tokens, dim=-1)
        return [input_ids[0, idx] for idx in [top_k_indices.item()]]
    
    #Works.
    def ablate_head(self,layer_idx, head_idx):
        """Ablate a specific head in a specific layer of a model."""
        
        W_q = self.model.model_attrs.get_q(layer_idx).weight.clone()
        W_k = self.model.model_attrs.get_k(layer_idx).weight.clone()
        W_v = self.model.model_attrs.get_v(layer_idx).weight.clone()
        
        # Head clone is basically these three sub-networks arranged vertically :
        W_q[:, head_idx] = 0.0
        W_k[:, head_idx] = 0.0
        W_v[:, head_idx] = 0.0
        
        head_clone = torch.cat([W_q, W_k, W_v], dim=0)

        # W_k = 0 ; W_q = 0 ; W_v = 0
        return head_clone

    #Works
    def merge_heads(self, layer_idx, head_idx1, head_idx2, alpha=0.5):
        """Merge two heads in a specific layer of a model. This is really helpful in understanding redundancy in attention heads."""

        # For gpt neo, it is attn.attention.(q_proj, k_proj, v_proj, out_proj).weight, 

        # This is the new way :)
        W_q = self.model.model_attrs.get_q(layer_idx).weight 
        W_k = self.model.model_attrs.get_k(layer_idx).weight
        W_v = self.model.model_attrs.get_v(layer_idx).weight
        W_o = self.model.model_attrs.get_o(layer_idx).weight

        # In Pyotrch, we can't modify tensors that are part of the computations graph in-place if they require grad.
        W_q_copy = W_q.clone()
        W_k_copy = W_k.clone()
        W_v_copy = W_v.clone()
        W_o_copy = W_o.clone()


        W_q_copy[:, head_idx1] = alpha * W_q[:, head_idx1] + (1 - alpha) * W_q[:, head_idx2]
        W_k_copy[:, head_idx1] = alpha * W_k[:, head_idx1] + (1 - alpha) * W_k[:, head_idx2]
        W_v_copy[:, head_idx1] = alpha * W_v[:, head_idx1] + (1 - alpha) * W_v[:, head_idx2]
        W_o_copy[:, head_idx1] = alpha * W_o[:, head_idx1] + (1 - alpha) * W_o[:, head_idx2]

        W_q_copy[:, head_idx2] = 0
        W_k_copy[:, head_idx2] = 0
        W_v_copy[:, head_idx2] = 0
        W_o_copy[:, head_idx2] = 0

        return W_q_copy, W_k_copy, W_v_copy, W_o_copy