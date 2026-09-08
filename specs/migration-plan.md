# Migration Plan

## Goal

Move to v2 without an unbounded rewrite. The first milestone is a working
indirect-only feature collection and local S1 training flow.

## Keep Initially

- existing judge model instrumentation;
- existing collector hooks;
- existing feature-set implementations;
- existing S1 model wrapper where useful;
- existing dataset code only as reference.

## Replace First

- global mutable config singleton;
- scan-first dataset creation;
- public direct-mode stubs;
- backend abstraction in the SDK path;
- batch padding behavior.

## First Usable Milestone

```text
local dataset examples -> local feature collection -> parquet features ->
local S1 training -> local MLflow model artifact
```

No RunPod required for the first milestone.

## Second Milestone

```text
local CLI -> RunPod feature collection -> remote S3 -> local sync ->
local MinIO curated dataset -> local S1 training -> local MLflow
```

## Deletion Candidates

After replacements exist:

- direct-mode public API;
- old dataset creator paths that require scan persistence;
- backend code used only for raw scan storage;
- unused automation YAMLs tied to old flow.

## Cutover Criteria

v2 becomes the default when:

- SDK indirect evaluation works from the new public API;
- feature datasets are reproducible and manifest-backed;
- padding regression test passes;
- one MALT import path works;
- one S1 model trains locally and logs to MLflow;
- remote feature collection can complete and sync on one small RunPod job.
