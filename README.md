# RAVEN-Eval

**Rubric-Guided Automatic Evaluation for AI Video Generation Models Based on LMM Preference Judgement**

[中文说明](README_zh-CN.md) · [Implementation protocol](docs/PROTOCOL.md) · [Task data](docs/DATA.md) · [Generation adapters](docs/GENERATION.md) · [Published results](results/published)

RAVEN-Eval evaluates fine-grained video-generation capabilities with task-specific rubrics, anonymous pairwise LMM judgments, order-swap verification, and confidence-aware Davidson ranking. It covers text-to-video generation, first–last-frame completion, and keyframe expansion.

![RAVEN-Eval teaser](docs/assets/teaser.png)

## Benchmark

| Task | Input | Focus | Tasks |
| --- | --- | --- | ---: |
| T2V: text-to-video | Text prompt | Objects, spatial details, interactions, domain knowledge, reasoning | 150 |
| FLT: first–last-frame completion | Prompt + first and last reference frames | Endpoint adherence and plausible intermediate events | 50 |
| KFT: keyframe expansion | Prompt + starting reference frame | Initial-state adherence and plausible subsequent evolution | 50 |

The reported study evaluates 20 generators and 13 LMM judges. Generator results cover 20 models for T2V/KFT and 14 for FLT. **Evaluation completed: July 31, 2026.**

This release includes a **250-task text snapshot, 150 I2V reference images, portable evaluation code, and published reference leaderboards**. The snapshot's T2V IDs and labels are not an asserted mapping to the final paper split. Raw final-paper comparisons and the complete generated-video collection are not bundled; the published scores therefore cannot be exactly reconstructed from this release alone. See [data versioning](docs/DATA.md).

## Main method

![RAVEN-Eval main framework](docs/assets/framework.png)

1. Construct discriminative tasks and priority-aware rubrics for semantic completion and detail control.
2. Generate three candidates per model/task and select one using a fixed candidate-selection judge. For I2V, verify the reference-frame constraints.
3. Uniformly sample **8 chronological frames** per video. Supply the prompt, rubric, and applicable reference frames; conceal generator identities.
4. Use three fixed judges to evaluate **A/B and B/A**, giving six requests per comparison edge. Resolve consistent wins/losses and stable ties; retain uncertainty separately.
5. Fit one confidence-aware, tie-aware Davidson model per task. Aggregate centered skills using **mean − population standard deviation** and convert to Elo-style scores.
6. Estimate 95% intervals by resampling complete model-pair judgment groups within tasks and refitting the models.

The study's judge ensemble is GPT-5.4, Claude Sonnet 4.6, and Gemini 3.1 Flash-Lite; the selection judge is separate. This implementation accepts **Responses-compatible judge endpoints**. Native Anthropic/Gemini endpoints require a compatible gateway or an adapter; a model name alone does not provide API compatibility.

## Environment requirements

- Python **3.10+**; Linux, macOS, or Windows through WSL. The judge uses POSIX file locks.
- `ffmpeg` and `ffprobe` available on `PATH`.
- NumPy, SciPy, and certifi, installed by the package.
- No local GPU is required for API-based generation/judging or CPU ranking. Hosted APIs require their own credentials and access.

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv

# From the repository root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ffmpeg -version
ffprobe -version
```

On macOS install FFmpeg with `brew install ffmpeg`. All examples below run from the repository root. This is a source/data repository; use the editable installation so the task data and guidelines remain available alongside the code.

## Quick start: no API calls

```bash
python -m raven_eval.demo --output-dir runs/demo
python -m unittest discover -s tests -v
python tools/validate_release.py
```

The demo creates three small synthetic videos, constructs task comparison files, inserts **synthetic** preferences, and runs Davidson estimation plus a small bootstrap. Its output is a software smoke test, not a benchmark result.

## Run an evaluation

### 1. Generate or register videos

Connect a generator through the [provider interface](docs/GENERATION.md), or register existing selected outputs in a JSONL manifest. Generator calls receive the **public prompt and reference images**, never the rubric.

```bash
# Preview three-candidate generation without importing/calling the provider.
python -m raven_eval.generate \
  --task-type t2v --task-id T2V_L1_D1_K1_0006 \
  --model my-generator --model-version my-version \
  --provider examples.provider_adapter:generate \
  --output-dir runs/my-generator --dry-run
```

`examples/provider_adapter.py` defines the interface and must be connected to your model's API before a real run. It is not a bundled vendor implementation. For existing videos, follow [selected_videos.example.jsonl](configs/selected_videos.example.jsonl); its example paths must be replaced with actual files.

For automatic best-of-three selection after generation:

```bash
# Set LMM_API_KEY in the environment, and set an accessible selector model/endpoint.
export SELECTOR_MODEL="your-selection-judge"
export JUDGE_API_URL="https://your-gateway.example/v1/responses"
python -m raven_eval.select \
  --candidates runs/my-generator/candidates.jsonl \
  --model "$SELECTOR_MODEL" --api-url "$JUDGE_API_URL" \
  --output runs/my-generator/selected.jsonl
```

The selector uses bidirectional candidate round-robin comparisons. Its deterministic tie rule and reference-check requirements are documented in [the protocol](docs/PROTOCOL.md). Concatenate the selected manifests from all applicable models into `runs/selected_videos.jsonl`.

### 2. Construct comparison pairs

```bash
python -m raven_eval.pairs \
  --videos runs/selected_videos.jsonl --output-dir runs/pairs --dry-run
python -m raven_eval.pairs \
  --videos runs/selected_videos.jsonl --output-dir runs/pairs
```

For a registered global library of **at most 25 models**, use complete task-wise comparison graphs. **Starting with model 26**, use selected anchors. Supply `--library-model-count` if a run contains only part of the global library; do not infer global size from one category. Above 25, the builder requires a frozen ten-anchor set per task type and explicit new-model IDs. [Anchor selection and insertion](docs/PROTOCOL.md#incremental-model-insertion) describes this project policy.

### 3. Run anonymous judges

Set `JUDGE_MODEL` to the first accessible judge model identifier, `JUDGE_API_URL` to its Responses-compatible endpoint, and `LMM_API_KEY` through your environment or secret manager. The optional `.env.example` documents variable names; it is not automatically loaded.

```bash
# Read-only preview; no credentials, API calls, frame extraction, or output writes.
python -m raven_eval.judge \
  --task-types t2v --model "$JUDGE_MODEL" --api-url "$JUDGE_API_URL" \
  --output-base judge_1 --dry-run

# Real evaluation: set LMM_API_KEY in the environment first.
python -m raven_eval.judge \
  --task-types t2v --model "$JUDGE_MODEL" --api-url "$JUDGE_API_URL" \
  --output-base judge_1
```

Repeat with the other fixed judges using distinct `judge_2` and `judge_3` output bases. Use `--task-types first_last` for FLT or `--task-types keyframe` for KFT. For different judge credentials use `--api-key-env ENVIRONMENT_VARIABLE_NAME`.

The default is anonymous 8-frame evaluation. `--show-model-names` explicitly reveals identities. `--sampling 1fps` is a separately tracked legacy protocol; do not mix it with the 8-frame results. Changing the endpoint's underlying model version also requires a new output base and run record.

### 4. Rank models and compute confidence intervals

```bash
python -m raven_eval.ranking \
  --input-dir runs/pairs/t2v \
  --judge-base judge_1 --judge-base judge_2 --judge-base judge_3 \
  --bootstrap-rounds 200 --bootstrap-pairs 2000 --seed 20260631 \
  --output-dir runs/results/t2v
```

Run the same command separately for `flt` and `kft`. For Overall, pass all three `--input-dir` arguments in one run and write to a separate output directory. Overall pools the applicable task-level skills; it does **not** average three category Elo scores or ranks. Unsupported tasks are excluded, never assigned zero. Do not reclassify generation failures as unsupported tasks.

Outputs: `leaderboard.csv`, `leaderboard.json`, and `task_scores.csv`. Newly computed intervals use `ci_low`/`ci_high`; published reference tables retain their original `±` display. A quick run may use `--bootstrap-rounds 0`, which intentionally leaves CI fields empty.

## Scoring

For a pair and `M` judges, let `c` be the number of order-consistent judgments. Its reliability is `λ = c/M`. A confident result contributes one unit to its win/loss/tie outcome. An uncertain result contributes `0.5` to each win direction and zero to ties. The likelihood averages over pairs and judges; skill regularization is normalized by the task's model count.

```text
P(A wins) = exp(s_A) / Z
P(B wins) = exp(s_B) / Z
P(tie)    = exp(gamma + (s_A + s_B)/2) / Z
Z         = exp(s_A) + exp(s_B) + exp(gamma + (s_A + s_B)/2)

score_i = mean(task_skill_i) - std(task_skill_i, ddof=0)
Elo_i   = 1000 + 400 / ln(10) * score_i
```

The implementation documents explicit choices where the main equation and supplementary implementation notes differ. See [optimizer, bootstrap, and reproduction details](docs/PROTOCOL.md).

## Results and examples

| Category | First-ranked model | Score |
| --- | --- | ---: |
| T2V | Seedance 2 | 840.7 |
| FLT | Seedance 2 | 890.5 |
| KFT | Seedance 2 | 1094.3 |
| Overall | Seedance 2 | 904.1 |

The full reference tables are in [results/published](results/published). They are a fixed result snapshot; fresh runs belong under `runs/`.

![Task distribution](docs/assets/dataset_statistics.png)

[Example videos and reference images](docs/EXAMPLES.md) illustrate the rubber duck, steamed buns, and hot iron ball / balloons cases.

## Repository structure

```text
raven_eval/          generation orchestration, candidate selection, pair construction,
                    anonymous judging, Davidson ranking, anchor selection, offline demo
data/tasks/         150 T2V + 50 FLT + 50 KFT task JSON files
data/references/    150 I2V input images
guidelines/         task-specific label-only judge guidelines
configs/            non-secret manifest examples
results/published/  fixed published category and Overall tables
docs/assets/        teaser, main framework, task figures, example media
tests/              scoring, graph-policy, anonymization and pipeline checks
tools/              release validation
```

## Citation and release scope

```bibtex
@article{jia2026raveneval,
  title={RAVEN-Eval: Rubric-Guided Automatic Evaluation for AI Video Generation Models Based on LMM Preference Judgement},
  author={Jia, Ziheng and Qian, Jiaying and Zhang, Zicheng and Zhu, Xiaorong and Gao, Lancheng and Min, Xiongkuo},
  journal={arXiv preprint arXiv:2608.09111},
  year={2026}
}
```

See [CITATION.cff](CITATION.cff), [data provenance](docs/DATA.md), and [license status](LICENSE_NOTICE.md). This source release contains no credentials, local account paths, or private evaluation logs.
