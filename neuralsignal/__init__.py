# NeuralSignal Package
__version__ = "0.1.0"


from neuralsignal.sdk.client import NeuralSignal
from neuralsignal.sdk.results import BatchDetectionResult, DetectionResult

__all__ = ["NeuralSignal", "DetectionResult", "BatchDetectionResult"]
