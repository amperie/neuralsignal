import logging

import pytest

from neuralsignal.remote.lifecycle import PodProgress, _wait_for_bundle
from neuralsignal.remote.runpod import ssh_command, get_pod


@pytest.mark.parametrize("pod, expected", [
    ({}, None),
    ({"publicIp": "203.0.113.1", "portMappings": {}}, None),
    ({"publicIp": "203.0.113.1", "portMappings": {"22": 12345}}, "ssh root@203.0.113.1 -p 12345"),
    ({"publicIp": "invalid;command", "portMappings": {"22": 22}}, None),
    ({"publicIp": "203.0.113.1", "portMappings": {"22": 99999}}, None),
])
def test_ssh_command_uses_assigned_endpoint(pod, expected):
    assert ssh_command(pod) == expected


def test_progress_logs_status_changes_and_prints_ssh_once(capsys, caplog):
    class Api:
        pod = {"desiredStatus": "RUNNING"}
        def get_pod(self, pod_id):
            assert pod_id == "pod-1"
            return self.pod
    api = Api()
    progress = PodProgress(api, "pod-1")
    with caplog.at_level(logging.INFO):
        progress()
        assert "Pod SSH" not in capsys.readouterr().out
        api.pod.update(publicIp="203.0.113.1", portMappings={"22": 12345})
        progress()
        progress()
    output = capsys.readouterr().out
    assert output.count("ssh root@203.0.113.1 -p 12345") == 1
    assert "tmux attach -t ns" in output
    assert caplog.text.count("status=RUNNING") == 1


def test_status_failure_is_retried_without_logging_response_secrets(caplog):
    class Api:
        fails = True
        def get_pod(self, pod_id):
            if self.fails:
                raise RuntimeError("SECRET_SERVER_RESPONSE")
            return {"desiredStatus": "RUNNING"}
    api = Api()
    progress = PodProgress(api, "pod-1")
    with caplog.at_level(logging.INFO):
        progress()
        progress()
        api.fails = False
        progress()
    assert "SECRET_SERVER_RESPONSE" not in caplog.text
    assert caplog.text.count("Pod status lookup failed") == 1
    assert "lookup recovered" in caplog.text


def test_wait_reports_progress_and_preserves_timeout(monkeypatch):
    from neuralsignal.remote import lifecycle
    times = iter([0, 2])
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(lifecycle, "exists", lambda *args: False)
    calls = []
    with pytest.raises(TimeoutError):
        _wait_for_bundle(None, "s3://test/run/bundle.zip", 0, 1, monitor=lambda: calls.append(True))
    assert calls == [True]


def test_get_pod_reads_expected_endpoint(monkeypatch):
    from neuralsignal.remote import runpod
    def api(method, path, token):
        assert (method, path, token) == ("GET", "/pods/pod-1", "test-token")
        return {"id": "pod-1"}
    monkeypatch.setattr(runpod, "api", api)
    assert get_pod("pod-1", "test-token") == {"id": "pod-1"}
