"""Public automation API for NeuralSignal dataset workflows."""

from neuralsignal.automation.dataset_automation_core import (
    create_dataset,
    create_s1_model,
    get_config,
    run_automation,
    run_data_collection,
    run_experiment,
)

__all__ = [
    "create_dataset",
    "create_s1_model",
    "get_config",
    "run_automation",
    "run_data_collection",
    "run_experiment",
]

