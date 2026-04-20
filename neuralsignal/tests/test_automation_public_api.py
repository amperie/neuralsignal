from neuralsignal.automation import get_config, run_experiment


def test_automation_exports_public_api():
    assert callable(run_experiment)


def test_get_config_loads_packaged_default():
    cfg = get_config()

    assert isinstance(cfg, dict)
    assert "run_data_collection" in cfg
