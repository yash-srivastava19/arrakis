import torch 
from .base_interpret import BaseInterpretabilityTool
class ResidualDecomposer(BaseInterpretabilityTool):
    """Decomposes the residual stream into interpretable components from MLP and attention."""
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = model 
    
    # Works.
    def mlp_basis(self, layer_idx):
        """Computes the basis of the MLP layer."""
        W_in = self.model.model_attrs.get_mlp_in(layer_idx).weight
        W_out = self.model.model_attrs.get_mlp_out(layer_idx).weight
        # print(W_in.shape, W_out.shape)
        U, S, V = torch.svd(W_out @ W_in) 
        # print(U.shape, S.shape, V.shape) 

        d_mlp = max(W_in.shape)  # Let's see.
        # print(U[:, :d_mlp].shape, V[:d_mlp].shape) 
        return U[:, :d_mlp], V[:d_mlp]  # REFACTOR: this is d_mlp

    # Breaks for models that have same Q,K,V projection. 
    def decompose_residual(self, input_ids, layer_idx, top_k=1):
        """Decomposes the residual stream into interpretable components from MLP and attention."""
        _, cache = self.model(input_ids)
        # Attention decomposition. It should break here.
        W_q = self.model.model_attrs.get_q(layer_idx).weight
        W_k = self.model.model_attrs.get_k(layer_idx).weight
        W_v = self.model.model_attrs.get_v(layer_idx).weight
        W_o = self.model.model_attrs.get_o(layer_idx).weight

        # print(W_q.shape, W_k.shape, W_v.shape, W_o.shape)
        block_type = self.model.model_attrs.get_block_type()
        
        resid = cache[f"{block_type}.{layer_idx}.hook_resid_post"]
        # print(resid.shape)

        U_mlp, V_mlp = self.mlp_basis(layer_idx)
        mlp_comps = (resid @ V_mlp) @ U_mlp  # let it break here, for GPT models till we get a fix.
        
        scores = torch.einsum("...qd, ...kd -> qk", W_q, W_k) / W_q.shape[-1]**0.5
        
        attn_pattern = torch.softmax(scores, dim=-1)
        # attn_pattern = torch.softmax(Q @ K.transpose(-2, -1) / Q.shape[-1]**0.5, dim=-1)
        # print(attn_pattern.shape, W_v.shape, W_o.shape)

        attn_comps = (attn_pattern @ W_v) @ W_o

        # Top components
        mlp_norms = torch.norm(mlp_comps, dim=0)
        attn_norms = torch.norm(attn_comps.reshape(attn_comps.shape[0], -1), dim=0)

        top_mlp = torch.topk(mlp_norms, top_k)
        top_attn = torch.topk(attn_norms, top_k)

        return {
            "mlp": {"components": mlp_comps, "top_k": (top_mlp.values, top_mlp.indices)},
            "attn": {"components": attn_comps, "top_k": (top_attn.values, top_attn.indices)},
        }