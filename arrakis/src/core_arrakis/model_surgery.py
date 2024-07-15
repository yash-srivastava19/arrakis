# This, introduced by EleutherAI, is a more robust version of LogitLens.
from contextlib import contextmanager
from .base_interpret import BaseInterpretabilityTool
# This is a really hacky class as of now, so test it more, and if it works, then include it.
class ModelSurgery(BaseInterpretabilityTool):
    """Performs surgery on the model. Replaces, deletes, or permutes layers(in context)."""
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def get_model(self):
        return self.model
    
    @contextmanager
    def replace_layers(self, indices, replacements):
        """Replaces the specified layers with the given replacements(in context)."""
        # Access layers
        layers = self.model.model_attrs.get_block()
        # Replace specified layers with new ones
        for index, replacement in zip(indices, replacements):
            layers[index] = replacement
        # Update the model's layers
        self.model.model_attrs.set_block(layers)
        try:
            yield self.model
        finally:
            # Restore the original layers
            for index in indices:
                layers[index] = None
            self.model.model_attrs.set_block(layers)

    @contextmanager
    def delete_layers(self, indices):
        """Deletes the specified layers(in context)."""
        # Access layers
        layers = self.model.model_attrs.get_block()
        # Delete specified layers
        for index in sorted(indices, reverse=True):
            del layers[index]
        # Update the model's layers
        self.model.model_attrs.set_block(layers)
        try:
            yield self.model
        finally:
            # Restore the deleted layers
            for index in sorted(indices):
                layers.insert(index, None)
            self.model.model_attrs.set_block(layers)

    @contextmanager
    def permute_layers(self, indices):
        """Permutes the layers based on the given indices(in context)."""
        # Access layers
        layers = self.model.model_attrs.get_block()
        # Permute the layers based on the given indices
        permuted_layers = [layers[i] for i in indices]
        # Update the model's layers
        self.model.model_attrs.set_block(layers)
        try:
            yield self.model
        finally:
            # Restore the original layer order
            self.model.model_attrs.set_block(layers)