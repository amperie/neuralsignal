from neuralsignal.remote.runpod import RunPodJob, build_pod_payload, redacted


def test_build_pod_payload_encodes_config_and_excludes_local_api_key():
    job = RunPodJob(
        run_id="run-1",
        config={"dataset": {"name": "fixture"}},
        manifest={
            "runpod": {"image": "image:latest", "gpu_count": 1, "volume_gb": 75},
            "env": {"HF_HOME": "/workspace/cache"},
        },
    )

    payload = build_pod_payload(job, secrets={"HF_TOKEN": "hf_x", "AWS_SECRET_ACCESS_KEY": "secret"})

    assert payload["name"] == "neuralsignal-run-1"
    assert payload["imageName"] == "image:latest"
    assert payload["env"]["NEURALSIGNAL_RUN_ID"] == "run-1"
    assert payload["dockerStartCmd"] == ["--run-id", "run-1"]
    assert "NEURALSIGNAL_FEATURE_CONFIG_B64" in payload["env"]
    assert "RUNPOD_API_KEY" not in payload["env"]


def test_redacted_hides_secret_values():
    payload = {"env": {"HF_TOKEN": "hf_x", "AWS_SECRET_ACCESS_KEY": "secret", "SAFE": "ok"}}

    result = redacted(payload)

    assert result["env"]["HF_TOKEN"] == "<redacted>"
    assert result["env"]["AWS_SECRET_ACCESS_KEY"] == "<redacted>"
    assert result["env"]["SAFE"] == "ok"
