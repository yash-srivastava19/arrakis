# Update: Works. But, we need to test it more.
import torch
from .base_interpret import BaseInterpretabilityTool
class KnowledgeGraphExtractor(BaseInterpretabilityTool):
    """Extracting relational subspaces from Q-K interaction,Projecting token embeddings onto these subspaces, Indentifying types of relations""" 
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = model 
    
    # Works.
    def extract_relational_subspaces(self, layer_idx, top_k=3):
        """Extracts the top-k relational subspaces from the Q-K interaction matrix of the layer."""
        # attn_block = self.model.model_attrs.get_attn(layer_idx)
        W_q = self.model.model_attrs.get_q(layer_idx).weight
        W_k = self.model.model_attrs.get_k(layer_idx).weight
        # W_k = self.model.blocks[layer_idx].attn.W_k
        
        W_QK = W_q @ W_k.T

        U, S, V = torch.svd(W_QK)  # We can use our FactoredMatrix class here.
        top_subspaces = V[:, :top_k].T 
        return top_subspaces

    # Works
    def project_token_embeddings(self, token_ids, subspaces):
        """Projects the token embeddings onto the given subspaces."""
        embeddings = self.model.model_attrs.get_embed()(token_ids) # It should be self.model.embed afaik.
        return {i: (embeddings @ subspace.T).squeeze() for i, subspace in enumerate(subspaces)}
    
    # Works, but more tests needed.
    def identify_relation_types(self, token_ids, subspaces, threshold=0.3):
        """Identifies the types of relations based on the projected embeddings."""
        projections = self.project_token_embeddings(token_ids, subspaces)
        relations = {i: [] for i in range(len(subspaces))}

        for i, proj in projections.items():
            for j, val in enumerate(proj):
                if (val > threshold).all() :   # See this refactor. if all of the vals is bigger than threshold, we append. 
                    relations[i].append(token_ids[j])
        
        return relations