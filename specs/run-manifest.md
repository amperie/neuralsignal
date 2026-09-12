# Run manifest

[RunManifest](../neuralsignal/storage/manifests.py) is written as `manifest.json`
inside a local run directory and the remote ZIP bundle. The worker does not
publish a separate manifest object during collection.

Current fields:

```json
{
  "schema_version": "run_manifest.v1",
  "run_id": "example-run",
  "state": "completed",
  "dataset": {"source": "malt", "name": "metr-evals/malt-public", "split": "public"},
  "features": {"schema_version": "features.v1", "materialized_sets": []},
  "rows": {"expected": null, "written": 8, "uploaded": 0},
  "shards": [
    {"index": 0, "path": "features/part-00000.parquet", "rows": 8,
     "sha256": "CHECKSUM", "state": "written", "bytes": 1234}
  ]
}
```

Values above illustrate the shape; a real checksum and byte count come from the
file. Remote `materialized_sets` holds copied config entries, including disabled
ones. The local collection CLI currently omits that list.

## State and counters

The dataclass defaults to `created`; the remote worker starts at `running`.
Local and remote collection mark `completed` after successful collection.
Exceptions do not reliably persist a final `failed` manifest. There is no pod
lifecycle state machine in this file.

`written` sums shard rows. `expected` stays null and `uploaded` stays zero in the
bundle workflow; zero uploaded rows does not mean bundle upload failed. Shards
stay `written`. Completion alone is not evidence that an older image produced
usable features; inspect rows and files.

## Reader checks

Sync/training reject an explicitly non-completed run; missing state is accepted
for older manifests. Sync accepts shard states written/uploaded/completed and
rejects other states. Training verifies listed paths, checksums and per-shard row
counts. It does not check all schema/version fields or shard state values.

No code SHA, image digest, pod ID, model config, timestamps, failure payload,
config hash, or prompt hash is automatically recorded. Remote resume and
config-identity enforcement are absent. Use a new run ID and directory for each
collection and record experiment provenance separately.
