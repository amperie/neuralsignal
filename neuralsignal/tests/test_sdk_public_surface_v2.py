import json

from neuralsignal.sdk.client import NeuralSignal
from neuralsignal.sdk.results import BatchDetectionResult, DetectionResult


def test_detection_result_serializes_flagged_state():
    result = DetectionResult(scores={"sabotage": 0.8}, flags={"sabotage": True})

    assert result.flagged is True
    assert json.loads(result.to_json())["flagged"] is True


def test_neuralsignal_evaluate_uses_indirect_evaluator():
    def evaluator(examples, detectors):
        assert detectors == ["sabotage"]
        assert examples == [{"input": "in", "output": "out"}]
        return [{"scores": {"sabotage": 0.7}, "metadata": {"judge": "fixture"}}]

    ns = NeuralSignal(detectors=["sabotage"], threshold=0.5, evaluator=evaluator)
    result = ns.evaluate("in", "out")

    assert result.flagged is True
    assert result.scores == {"sabotage": 0.7}
    assert result.metadata == {"judge": "fixture"}


def test_neuralsignal_evaluate_batch_returns_batch_result():
    ns = NeuralSignal(evaluator=lambda examples, detectors: [{"scores": {"x": 0.1}} for _ in examples])

    result = ns.evaluate_batch([{"input": "a", "output": "b"}, {"input": "c", "output": "d"}])

    assert isinstance(result, BatchDetectionResult)
    assert [item.flagged for item in result.results] == [False, False]



def test_root_package_exports_public_sdk():
    from neuralsignal import BatchDetectionResult, DetectionResult, NeuralSignal

    assert NeuralSignal.__name__ == "NeuralSignal"
    assert DetectionResult.__name__ == "DetectionResult"
    assert BatchDetectionResult.__name__ == "BatchDetectionResult"
