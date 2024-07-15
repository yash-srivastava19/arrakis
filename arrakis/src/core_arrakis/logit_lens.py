# Layer attribution and layer attribution. Till now, working for GPT2.
from .base_interpret import BaseInterpretabilityTool

class LogitLens(BaseInterpretabilityTool):
    """Analyzes the logit lens of the model."""
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = model 
    
    def layer_attributions(self, input_ids, target_idx):
        """Computes the attributions of the layers."""
        _, cache = self.model(input_ids)
        logits = []
        block_type = self.model.model_attrs.get_block_type()

        for i in range(len(self.model.model_attrs.get_block())):
            block_logit = cache[f"{block_type}.{i}.hook_resid_post"] @ self.model.model_attrs.get_lin_ff().weight
            logits.append(block_logit)

        final_logit = cache[f"{block_type}.{len(self.model.model_attrs.get_block())-1}.hook_resid_post"] @ self.model.model_attrs.get_lin_ff().weight
        logits.append(final_logit)

        attributions = [logit[0, target_idx].item() for logit in logits]
        return attributions

    # Same issue. Need to see what are W_u and such.
    def logit_lens(self, input_ids, target_idx, layer_idx):
        """Computes the logit lens of the model."""
        #compares what a layer things the logit should be vs the actual impact.
        _, cache = self.model(input_ids)
        block_type = self.model.model_attrs.get_block_type()
        #direct logit
        direct_logit = cache["logits"]

        # indirect logit
        residual_stack = cache[f"{block_type}.{layer_idx}.hook_resid_post"]
        # print(residual_stack.shape)
        # input()
        # res = residual_stack.clone()
        # print(len(self.model.model_attrs.get_block()))

        # Models that are not working are llama, gemma, phi3(position_ids issue), and bloom(more args in block forward call)
        for i in range(layer_idx+1, len(self.model.model_attrs.get_block()) -1):
            
            block = self.model.model_attrs.get_block()[i]
            
            if type(residual_stack) == tuple:
                # This too is a hacky fix. There are some problems with the implementation.
                continue
            
            #Here, bloom breaks as it requires an alibi and attention mask parameter. Will see, till now bloom is dropped.
            import torch
            if self.model.name in ["llama", "gemma", "phi3"]: 
                # This is such a hacky fix. I tried understanding the hf code, but it seems same to me. The position_ids should be optionalt, but somehow it is necessary for llama, gemma, and phi3.
                position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                residual_stack = block(residual_stack, position_ids=position_ids) # this is the bitch line.
            else:
                residual_stack = block(residual_stack)

        # print(residual_stack)
        # input()
        indirect_logit = residual_stack[0] @ self.model.model_attrs.get_lin_ff().weight

        return direct_logit[0, target_idx].item(), indirect_logit[0, target_idx].item()