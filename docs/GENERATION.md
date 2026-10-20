# Video generation and registration

The core evaluation accepts existing selected videos from any generator. The provider interface also supports generating candidates without tying the repository to local storage layouts or credentials files. Original machine-specific batch runners are not copied into this public package.

## Provider interface

Implement an importable Python function with this signature:

```python
def generate(*, task, model, output_dir, candidate_index, parameters):
    # Submit/poll/download through your provider's documented API.
    # Read credentials from your environment, not from parameters or a source file.
    # Save the actual generated video under output_dir.
    return output_dir / "candidate.mp4"
```

`task` includes only `task_id`, `task_type`, `prompt`, `task_version`, and absolute local reference-image paths. It deliberately excludes `rubric`. Use no image input for T2V, first+last for FLT, first only for KFT. `parameters` must contain non-secret generation settings; never API tokens. `candidate_index` is zero-based, while persisted manifest candidate indices are one-based.

`examples/provider_adapter.py` is an explicit interface template and raises `NotImplementedError` until connected to a real provider. No native Seedance/Kling/Veo/etc. API implementation is claimed by this template. Keep provider-specific request schemas and SDK dependencies in your adapter. Confirm that the chosen model actually supports the task mode rather than silently replacing FLT with generic I2V.

```bash
python -m raven_eval.generate \
  --task-type flt \
  --model YOUR_MODEL --model-version EXACT_API_VERSION \
  --provider your_adapter:generate \
  --parameters configs/generation.example.json \
  --output-dir runs/YOUR_MODEL/flt
```

The default is three candidates per task. The command writes a candidate JSONL manifest as each candidate completes. Use a new run directory to avoid overwriting evidence. `--dry-run` validates the task selection and previews the candidate count without importing or calling the adapter. It does not claim that a stub adapter or real remote service works.

## Best-of-three selection

Use the `raven_eval.select` command shown in the README. Keep candidate-selection evidence separate from the final judge evidence. The release's selection strategy is described in [PROTOCOL.md](PROTOCOL.md#generation-and-candidate-selection).

For each selected I2V output, verify first-frame/endpoint adherence and record the acceptance rule. The separate pooled-pixel 0.98 gate reported in the study is not automatically reproduced by the candidate selector; its exact pooling definition needs to be fixed before making that claim. A reference-input mismatch is not fixed by renaming a video.

## Existing videos

Create one JSONL row per selected model/task output:

```json
{"task_type":"t2v","task_id":"T2V_L1_D1_K1_0006","model":"my-generator","model_version":"exact-version","video_path":"runs/videos/my-generator/duck.mp4","selected":true}
```

`video_path` can be absolute at runtime or relative to the repository root. Public manifests should use relative paths. Exactly one selected row per task/model is allowed. `selected:false` candidate rows are ignored by the comparison builder. Keep distinct versions/modes as distinct model IDs and document aliases before ranking. A numeric count of models is not inferred from API platforms.

The comparison builder creates edges among the models actually supplied per task. This does not itself certify complete benchmark coverage: review the applicability registry and required tasks before publishing a model as complete. Missing outputs and incompatible tasks require different statuses.

## Provider records

Keep API credentials out of Git. Record non-secret generation parameters, substitutions for unsupported durations/resolutions, candidate IDs, and immutable provider versions where available. Use a new protocol/run record if a hosted alias changes its backend. `runs/` is ignored by Git so generated files, local paths, and evaluation logs are not accidentally added to the source repository.
