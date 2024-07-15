import torch
from .base_interpret import BaseInterpretabilityTool

class LogitAttributor(BaseInterpretabilityTool):
    """Attribution of the logit to the input tokens."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model
    
    # Works as of now, but need more testing. GPT test passed.
    def logit_attribution(self, input_ids, target_idx):
        """Computes the attribution of the target logit to the input tokens. Returns the attributions."""
        _, cache = self.model(input_ids)
        logits = cache["logits"]
        target_logit = logits[0, target_idx]
        target_logit.requires_grad = True

        attributions = {}

        # print(list(self.model.named_parameters()))
        # input()

        for name, param in self.model.named_parameters():
            # We will make some arrangments for the final layers as well in the ModelAttributes class.
            if "W_u" in name or "ln_f" in name or ".norm." in name:  # Focus on final layer. 
                param.requires_grad = True
                attributions[name ] = 0.0
        
        target_logit.backward(retain_graph=True)

        for name, param in self.model.named_parameters():
            if "W_u" in name or "ln_f" in name or ".norm." in name:
                if param.grad is not None:
                    attributions[name] += (param * param.grad).sum().item()
                if param.grad is not None:
                    param.grad.zero_()  # Clear the gradients for the next computation

        return attributions
    
    # Works as of now, but need more testing. GPT test passed.
    def track_token_circulation(self, input_ids, target_idx, n_layers=2):
        """Tracks the token circulation through the layers. Returns the direct moves and the total moves."""
        total_moves = 0
        direct_moves = {i: 0 for i in range(n_layers)}
        
        _, cache = self.model(input_ids)
        block_type = self.model.model_attrs.get_block_type()
        
        for i in range(1, n_layers):
            prev_resid = cache[f"{block_type}.{i-1}.hook_resid_post"][0, target_idx]
            curr_attn = cache[f"{block_type}.{i}.attn.hook_result_post"][0, target_idx]
            curr_mlp = cache[f"{block_type}.{i}.mlp.hook_result_post"][0, target_idx]
            curr_resid = cache[f"{block_type}.{i}.hook_resid_post"][0, target_idx]
            
            move_dist = torch.norm(curr_resid - prev_resid)
            total_moves += move_dist.item()
            
            direct_moves[i] += torch.norm(curr_attn) + torch.norm(curr_mlp)
        
        # Normalize direct moves
        for i in direct_moves:
            direct_moves[i] /= total_moves
        
        return direct_moves
