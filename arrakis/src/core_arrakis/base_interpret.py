from abc import ABC, abstractmethod

class BaseInterpretabilityTool(ABC):
    """Base class for interpretability tools."""
    def __init__(self, model):
        self.model = model