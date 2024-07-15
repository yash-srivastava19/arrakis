# Update: Works but needs to be tested.
import torch
from .base_interpret import BaseInterpretabilityTool

class SuperpositionDisentangler(BaseInterpretabilityTool):
    """Disentangles the superposition of the activations."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    # Works, but needs to be tested.
    def compute_activation_covariance(self, ids_list, layer):
        """Computes the covariance matrix of the activations."""
        activations = []
        for input_ids in ids_list:
            _, cache = self.model(input_ids)
            activations.append(cache[layer].squeeze())
        activations = torch.stack(activations)
        return torch.cov(activations.transpose(-2, -1)[:,:-1].reshape(activations.shape[1], -1)) # I'm not sure whether this is correct or not.

    # Works, but needs to be tested.
    def disentangle_superposition(self, cov_matrix, n_concepts=10):
        """Disentangles the superposition of the activations."""
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        idx = torch.argsort(eigenvalues, descending=True)
        return eigenvectors[:, idx[:n_concepts]]

    # The rest need to be udnerstood and tested.
    def project_activations(self, activations, concepts):
        """Projects the activations onto the concepts."""
        return activations @ concepts

    def reconstruct_activations(self, projections, concepts):
        """Reconstructs the activations from the projections."""
        return projections @ concepts.T

    def interpolate_concepts(self, concept1, concept2, alpha):
        """Interpolates between two concepts."""
        return (1 - alpha) * concept1 + alpha * concept2