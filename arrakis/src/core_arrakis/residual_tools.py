import torch
from .base_interpret import BaseInterpretabilityTool

class ResidualTools(BaseInterpretabilityTool):
    """The residual stream in transformers carries information throughout the network."""
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = model 

    #Works.
    def residual_decomposition(self, input_ids, layer_idx):
        """Break down the residual stream to see where info comes from."""
        # See this whether the names are correct or not.
        _, cache = self.model(input_ids)
        block_type = self.model.model_attrs.get_block_type() 
        residual = cache[f"{block_type}.{layer_idx}.hook_resid_pre"]
        attn_out = cache[f"{block_type}.{layer_idx}.attn.hook_result"]
        mlp_out  = cache[f"{block_type}.{layer_idx}.mlp.hook_result"]

        return {
            "from_prev": residual-attn_out-mlp_out,
            "from_attn": attn_out,
            "from_mlp": mlp_out
        }
    
    #Works.
    def residual_movement(self, input_ids, layer_idx1, layer_idx2):
        """ Measure how much the residual stream changes between layers."""
        _, cache = self.model(input_ids)
        block_type = self.model.model_attrs.get_block_type()
        residual1 = cache[f"{block_type}.{layer_idx1}.hook_resid_post"]
        residual2 = cache[f"{block_type}.{layer_idx2}.hook_resid_pre"]
        
        movement = torch.norm(residual2 - residual1, dim=-1) / torch.norm(residual1, dim=-1)
        return movement.mean().item()
    
    # Logits part is not working.
    def feature_attribution(self, input_ids, target_ids):
        """Attribute the final prediction back through the residual stream."""
        _, cache = self.model(input_ids)
        block_type = self.model.model_attrs.get_block_type()
        logits = cache["logits"]  # BREAK.
        target_logits = logits[0, target_ids]

        residuals = [cache[f"{block_type}.{i}.hook_resid_post"] for i in range(len(self.model.model_attrs.get_block()))]
        attributions = []

        for resid in residuals:
            resid.requires_grad = True
            logits = self.model.model_attrs.get_lin_ff()(resid) # BREAK.
            # logits = self.model.model_attrs.get_embed()(logits) # BREAK.

            target_logit = logits[0, target_ids]
            target_logit.mean().backward() # This fixes it, but I'm not sure how correct it is.

            attributions.append(torch.norm(resid.grad, dim=-1))
        
        return attributions