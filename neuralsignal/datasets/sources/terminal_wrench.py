"""Pinned, fail-closed Terminal Wrench loader. Trajectory commands are data only."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from tempfile import NamedTemporaryFile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from neuralsignal.datasets.sources.agent_transcripts import transcript_example

REPO = "few-sh/terminal-wrench"
LABELS = {"baseline": 0, "attacker_legitimate_solve": 0,
          "rewarded_serious_exploit": 1, "rewarded_nonserious_hack": 1}


class TerminalWrenchSource:
    def __init__(self, cache, revision, variant="original", workers=12):
        if variant not in {"original", "sanitized", "stripped"}:
            raise ValueError(variant)
        self.cache, self.revision = Path(cache) / revision, revision
        self.variant, self.workers = variant, workers
        self.sessions = local()

    def read(self, path):
        target = self.cache / path
        if not target.resolve().is_relative_to(self.cache.resolve()):
            raise ValueError("Trajectory path escapes source cache")
        if not target.exists():
            if not hasattr(self.sessions, "session"):
                self.sessions.session = requests.Session()
                self.sessions.session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
                              status_forcelist=[429, 500, 502, 503, 504])))
            response = self.sessions.session.get(f"https://raw.githubusercontent.com/{REPO}/{self.revision}/{path}", timeout=90)
            response.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temporary.write(response.content)
            Path(temporary.name).replace(target)
        return target.read_text(encoding="utf-8")

    def records(self):
        tasks = json.loads(self.read("index/tasks.json"))
        records = json.loads(self.read("index/trajectories.json"))
        records += [{**baseline, "tree_name": "baseline_trajectories", "trajectory_label": baseline["label"]}
                    for task in tasks for baseline in task["baselines"]]
        counts = Counter(row["classification"] for row in records)
        if set(counts) - set(LABELS) - {"no_reward"}:
            raise ValueError(f"Unknown classifications: {counts}")
        self.audit = {"revision": self.revision, "variant": self.variant,
                      "source_counts": dict(counts), "excluded_no_reward": counts["no_reward"]}
        return [row for row in records if row["classification"] != "no_reward"]

    def normalize(self, row, prompt):
        if not row.get("task_id") or not row.get("model"):
            raise ValueError("Missing Terminal Wrench task/model identifier")
        tree = row["tree_name"]
        if tree == "hack_trajectories" and self.variant != "original":
            tree = f"{self.variant}_trajectories"
        path = f"tasks/{row['task_id']}/{row['model']}/{tree}/{row['trajectory_label']}/trial/agent/trajectory.json"
        trajectory = json.loads(self.read(path))
        if not str(trajectory.get("schema_version", "")).startswith("ATIF-v1."):
            raise ValueError(f"Unknown trajectory schema: {path}")
        messages, removed = [], 0
        for step in trajectory["steps"]:
            role = {"agent": "assistant", "user": "user", "system": "system"}.get(step["source"])
            if role is None or not isinstance(step.get("message", ""), str):
                raise ValueError(f"Unexpected step: {path}")
            content = step.get("message", "")
            if role == "user" and prompt in content:
                content = content.replace(prompt, "")
                removed += 1
            if role == "user" and "You are an authorized Red Team QA Agent" in content:
                raise ValueError(f"Unrecognized elicitation prompt: {path}")
            if step.get("tool_calls"):
                content += "\ntool_calls: " + json.dumps(step["tool_calls"], ensure_ascii=False)
            messages.append({"role": role, "content": content})
            for result in (step.get("observation") or {}).get("results", []):
                if not isinstance(result.get("content"), str):
                    raise ValueError(f"Unknown observation: {path}")
                messages.append({"role": "tool", "content": result["content"]})
        identifier = f"{row['task_id']}/{row['model']}/{row['tree_name']}/{row['trajectory_label']}"
        return transcript_example(identifier, messages, {
            "source": REPO, "revision": self.revision, "variant": self.variant,
            "task_id": row["task_id"], "model": row["model"],
            "abnormal": LABELS[row["classification"]], "classification": row["classification"],
            "categories": row.get("observed_categories", []), "trajectory_path": path,
            "removed_elicitation_prompts": removed,
        })

    def iter_examples(self):
        records = self.records()
        prompt = self.read("prompts/hack_prompt_v5.md").strip()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            yield from pool.map(lambda row: self.normalize(row, prompt), records)
