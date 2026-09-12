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

    payload = build_pod_payload(job, secrets={"HF_TOKEN": "hf_x", "AWS_SECRET_ACCESS_KEY": "secret", "RUNPOD_KEY": "local", "RUNPOD_API_KEY": "local"})

    assert payload["name"] == "neuralsignal-run-1"
    assert payload["imageName"] == "image:latest"
    assert payload["env"]["NEURALSIGNAL_RUN_ID"] == "run-1"
    assert payload["dockerStartCmd"] == ["--run-id", "run-1"]
    assert "NEURALSIGNAL_FEATURE_CONFIG_B64" in payload["env"]
    assert "RUNPOD_API_KEY" not in payload["env"]
    assert "RUNPOD_KEY" not in payload["env"]


def test_redacted_hides_secret_values():
    payload = {"env": {"HF_TOKEN": "hf_x", "AWS_SECRET_ACCESS_KEY": "secret", "SAFE": "ok"}}

    result = redacted(payload)

    assert result["env"]["HF_TOKEN"] == "<redacted>"
    assert result["env"]["AWS_SECRET_ACCESS_KEY"] == "<redacted>"
    assert result["env"]["SAFE"] == "ok"


def test_api_uses_pods_rest_endpoint_and_accepts_empty_delete(monkeypatch):
    from neuralsignal.remote import runpod

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b""

    def urlopen(request, timeout):
        assert request.full_url == "https://rest.runpod.io/v1/pods/pod-1"
        assert request.method == "DELETE"
        assert timeout == 60
        return Response()

    monkeypatch.setattr(runpod.urllib.request, "urlopen", urlopen)
    assert runpod.terminate("pod-1", token="test") == {}


def test_payload_maps_gpu_and_registry_settings():
    payload = build_pod_payload(RunPodJob("run", {}, {"runpod": {
        "image": "private:tag", "cloud_type": "COMMUNITY",
        "gpu_type_ids": ["NVIDIA GeForce RTX 4090"],
        "container_registry_auth_id": "auth-id",
    }}))
    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert payload["cloudType"] == "COMMUNITY"
    assert payload["containerRegistryAuthId"] == "auth-id"
