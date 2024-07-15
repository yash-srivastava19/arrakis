# Update, this is working perfectly.
import torch
from .base_interpret import BaseInterpretabilityTool

class AttentionComposer(BaseInterpretabilityTool):
    """Class to interpret attention heads."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    #Works
    def head_composition(self, head1, head2, alpha=0.5):
        """Composes two heads."""
        layer1, idx1 = head1
        layer2, idx2 = head2
        
        Q1 = self.model.model_attrs.get_q(layer1).weight[:, idx1]
        K1 = self.model.model_attrs.get_k(layer1).weight[:, idx1]
        V1 = self.model.model_attrs.get_v(layer1).weight[:, idx1]
        O1 = self.model.model_attrs.get_o(layer1).weight[:, idx1]

        Q2 = self.model.model_attrs.get_q(layer2).weight[:, idx2]
        K2 = self.model.model_attrs.get_k(layer2).weight[:, idx2]
        V2 = self.model.model_attrs.get_v(layer2).weight[:, idx2]
        O2 = self.model.model_attrs.get_o(layer2).weight[:, idx2]
        

        # Composite head: attend to what both heads attend to
        Q = (Q1 + Q2) / 2
        K = (K1 + K2) / 2
        V = (V1 + V2) / 2
        O = alpha * O1 + (1 - alpha) * O2 # This is basically "blending" the output. We saw this in Computer Vision!

        return (Q, K, V, O)

    #Works
    def apply_composed_head(self, input_ids, Q, K, V, O):
        """Applies the composed head to the input."""
        _, cache = self.model(input_ids)
        query = cache["resid_pre"] @ Q
        key = cache["resid_pre"] @ K
        value = cache["resid_pre"] @ V
        
        # attn = torch.softmax(query @ key.T / query.shape[-1]**0.5, dim=-1)
        attn = torch.einsum("...qd, ...kd -> qk", query, key) / query.shape[-1]**0.5
        
        result = (attn @ value) @ O.T
        
        return result

    # Works. Works.
    def attention_path_patching(self, input_ids, src_head, dest_head):
        """Patches the attention path from a source head to a destination head."""
        src_layer, src_idx = src_head
        dest_layer, dest_idx = dest_head
        
        def patch_fn(module, input, output):
            src_q = self.model.model_attrs.get_q(src_layer).weight[:, src_idx]
            src_k = self.model.model_attrs.get_k(src_layer).weight[:, src_idx]
            src_v = self.model.model_attrs.get_v(src_layer).weight[:, src_idx]
            if input:
                q = input[0] @ src_q
                k = input[0] @ src_k
                v = input[0] @ src_v
                
                # attn = torch.softmax(q @ k.T / q.shape[-1]**0.5, dim=-1)
                attn = torch.einsum("...qd, ...kd -> qk", q, k) / q.shape[-1]**0.5
                patched = (attn @ v).unsqueeze(-1)
                
                output[0][:, :, dest_idx] = patched.squeeze(-1)  # Patch the output tensor. output[0] is the output tensor.
                return output
        
        handle = self.model.model_attrs.get_attn(dest_layer).register_forward_hook(patch_fn) # this means input and output path will be there.
        result, _ = self.model(input_ids)
        handle.remove()
        return result