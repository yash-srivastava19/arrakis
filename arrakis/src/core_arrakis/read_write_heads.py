import torch
from .base_interpret import BaseInterpretabilityTool

class WriteReadAnalyzer(BaseInterpretabilityTool):
    """Analyzes the read and write heads of the model."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model
    
    #Works
    def identify_write_heads(self, layer_idx, threshold=0.8):
        """Identifies the write heads of the layer. Returns the indices of the heads."""
        W_v = self.model.model_attrs.get_v(layer_idx).weight
        W_o = self.model.model_attrs.get_o(layer_idx).weight

        write_strengths = torch.norm(W_o, dim=0)
        top_k = int(threshold * W_o.shape[1])
        _, indices = torch.topk(write_strengths, top_k)
        
        return [(layer_idx, idx.item()) for idx in indices]
    
    # Works.
    def identify_read_heads(self, layer_idx, dim_idx, threshold=0.8):
        """Identifies the read heads of the layer. Returns the indices of the heads."""
        W_q = self.model.model_attrs.get_q(layer_idx).weight
        W_k = self.model.model_attrs.get_k(layer_idx).weight

        q_aligned = torch.cosine_similarity(W_q[:, dim_idx], W_k.transpose(-2, -1))  # Check if this is correct.
        top_k = int(threshold * q_aligned.shape[0])
        _, indices = torch.topk(q_aligned, top_k)
        
        return [(layer_idx, idx.item()) for idx in indices]
    
    # Works.
    def trace_information_flow(self, input_ids, src_token, dst_token, n_layers=2):
        """Traces the information flow from the source token to the destination token. Returns the flow at each layer."""
        src_idx = input_ids[0].tolist().index(src_token)
        dst_idx = input_ids[0].tolist().index(dst_token)
        
        flow = []
        _, cache = self.model(input_ids)
        
        for i in range(n_layers):
            W_q = self.model.model_attrs.get_q().weight
            W_k = self.model.model_attrs.get_k().weight
            W_v = self.model.model_attrs.get_v().weight
            
            Q = cache[f"blocks.{i}.hook_resid_pre"] @ W_q
            K = cache[f"blocks.{i}.hook_resid_pre"] @ W_k
            V = cache[f"blocks.{i}.hook_resid_pre"] @ W_v
            
            attn = torch.softmax(Q @ K.transpose(-2, -1) / Q.shape[-1]**0.5, dim=-1)
            flow.append(attn[dst_idx, src_idx])
        
        return flow
