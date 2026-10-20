# Implementation protocol

## Scope and versions

The executable release uses anonymous, rubric-guided pairwise evaluation with eight uniform video samples. The included task-text snapshot is versioned separately from the final reported benchmark. Published reference tables are not recalculated from the snapshot.

Each run must record the exact generator and judge API identifiers, accessible backend versions when available, generation settings, task version, candidate-selection rule, endpoint/gateway version, guideline hashes, and sampling policy. An API alias does not uniquely identify immutable weights. Freeze the judge ensemble across a comparison graph.

## Generation and candidate selection

Generate three candidates per supported model/task. Do not send the rubric to the generator. The release's `raven_eval.select` compares all three candidate pairs in both orders with a fixed selector, applies primary semantic/detail criteria first, and chooses the candidate with the most resolved wins. Unresolved comparisons give both candidates 0.5; exact score ties choose the lowest candidate index. These are explicit implementation choices for an executable selection procedure; they are not claimed to reproduce an undocumented historical tie rule.

The reported selection judge is GPT-5.4-mini. Keep selection evidence separate from final generator-comparison evidence. The candidate selector compares required references, but does not implement a separate pixel-similarity gate. The reported I2V threshold of 0.98 requires a frozen pooling/similarity definition; automatic reproduction of that gate and historical candidate pools is outside this release. Verify and record reference-frame compliance before marking a selected video eligible. Record missing/failed candidates and retry; do not silently substitute a different model or version.

Generation adapters return existing downloaded video paths. Preserve candidate indices, actual API versions, request parameters, and any unsupported-setting substitutions. Audio is ignored by the judge; the reported study used silent videos. Mark genuinely incompatible tasks in a separate applicability registry before evaluation. A failed API call is not an incompatibility.

## Judging

The default sampler places eight timestamps from 0 to `max(last decoded frame timestamp − 0.001 seconds, 0)`, inclusive, using equal spacing. The small endpoint margin avoids seeking past the final frame through decimal rounding; container duration alone is not a safe endpoint for low/variable-frame-rate videos. Images preserve native frame dimensions. Very short clips can produce near-identical samples. FLT includes the two reference endpoint images; KFT includes the first reference image; T2V has no image reference.

Judge labels are exactly `Better`, `Worse`, or `Comparable`, relative to the first displayed video. The prompt uses positional aliases Model A/B. Image bytes are sent as data URIs; local filenames and generator metadata are not sent. Swapping the videos does not reveal the generator identities.

Each selected judge evaluates both directions. Use distinct output bases, and supply those exact bases to the ranker. The ranker requires all selected judges and both valid directions for every included edge; missing results fail rather than becoming ties. The judge records protocol hashes and refuses incompatible cached outputs under the same base. `--skip-existing-base` is an explicit legacy override; do not use it to mix protocols in a new benchmark run.

The API client targets Responses-compatible endpoints. Provider-native chat, Anthropic messages, and Gemini generation endpoints require adapters. Credentials are read from named environment variables.

## Confidence-aware Davidson estimation

After mapping B/A back to A/B:

| Forward | Reverse | Resolved A/B mass | Confident |
| --- | --- | --- | --- |
| Better | Worse | (1, 0, 0) | yes |
| Worse | Better | (0, 1, 0) | yes |
| Comparable | Comparable | (0, 0, 1) | yes |
| other complete combination | | (0.5, 0.5, 0) | no |

For each pair, average the outcome mass over the fixed `M` judges, then multiply by the fraction of confident judges. Divide the task likelihood by the number of included pairs. Fit the negative log likelihood plus:

```text
alpha / N * sum(skill_i^2) + beta * gamma^2 + gauge * mean(skill)^2
alpha = 1e-3, beta = 1e-4, gauge = 10
L-BFGS-B: maxiter=1000, ftol=1e-11, gtol=1e-6
initial skills=0, initial gamma=0
```

The release follows the model-count-normalized skill penalty in the main equation. The supplementary hyperparameter table describes an unnormalized penalty, so the two should not be treated as numerically identical. A maximum iteration budget is not a requirement to execute every iteration. Optimization failures stop the run. Scores are centered after fitting.

The point estimate requires the positive-reliability comparison graph to be connected. Regularization does not turn disconnected evidence into identified relative ranks. Edges whose judges are all uncertain have zero reliability and do not connect the effective graph.

## Aggregation and Overall

For each model, aggregate its centered skill across the applicable tasks using mean minus population standard deviation (`ddof=0`). Use the same statistic for categories and Overall, then transform to Elo-style scores. Different category Elo scores are not averaged. With all 250 tasks applicable, pooling preserves the 150:50:50 task-count weighting. Unsupported tasks do not enter as zero scores; disclose task coverage when denominators differ.

The `votes` field in a newly computed result counts complete raw directional judgments involving a model (`2 × M` per included pair). It is not the sum of reliability weights. Published Votes retain the original reported values and should not be inferred from this release's example graph.

## Bootstrap confidence intervals

Use 200 rounds and 2,000 sampled model pairs per round, with seed integer `20260631`. The seed is an identifier, not a calendar date. Allocate the pair budget as evenly as possible among the input tasks; randomly assign remainder slots to different tasks each round. Sample edges with replacement within each task. Retain the complete set of judges and directions for a sampled pair; never resample individual judge responses.

Every replicate refits each task's Davidson model, recomputes mean minus standard deviation, and applies the Elo transform. Duplicate edge draws are represented by multiplicities. The 2.5th and 97.5th percentiles give interval endpoints. The point estimate uses the original graph.

The 2,000-pair budget applies to each category run. An Overall run uses 2,000 pairs spread across all supplied tasks. These are explicit budget-scope choices because the report does not uniquely settle the scope. A supplementary environment table uses a different cached-score/bootstrap description; this release follows the 200-round, pair-resampling-and-refit definition, not cached task-score resampling.

Bootstrap retains every original task node even if resampling removes all of its incident edges. Such disconnected replicates are regularized and included, with counts reported in `leaderboard.json`. Their intervals are conditional on this declared policy and the frozen graph/anchors; the policy can affect sparse-graph uncertainty. No failed optimizer runs are silently discarded. `--bootstrap-rounds 0` computes only point estimates.

## Incremental model insertion

**The 25-model switch is this project's deployment policy.** It is not a threshold experimentally established by the paper.

- Global registered library size after insertion ≤25: retain a complete graph among all applicable models per task. Reuse historical edges only when protocol and versions match.
- Size >25: retain the complete historical base and add new-model/anchor edges. No new/new edges are required by default. The pair builder emits insertion edges; score them, merge them with historical evidence without duplicating model pairs, and rerun the joint fit. Do not rank only the insertion edges as if they contained the complete old graph.

Choose anchors independently by task type. Sort old models by the joint category score and split into five near-equal tiers. Within each tier select two models with the largest reliability statistic:

```text
rho_i,m = SRCC(task_rank_i_from_judge_m, task_rank_i_from_joint_judges)
q_i = mean_m(rho_i,m) - tau * std_m(rho_i,m)
tau = 1  # explicit release convention
```

`raven_eval.anchors` consumes joint and single-judge `task_scores.csv` files with identical task/model coverage. At least three task observations per model are required. Undefined correlations are ineligible; if any tier lacks two eligible anchors, no anchor set is emitted. Freeze the output JSON as a versioned artifact.

```bash
python -m raven_eval.anchors \
  --task-type t2v --joint-scores runs/joint/task_scores.csv \
  --judge-scores runs/judge_1/task_scores.csv \
  --judge-scores runs/judge_2/task_scores.csv \
  --judge-scores runs/judge_3/task_scores.csv \
  --output runs/anchors_v1.json

python -m raven_eval.pairs --videos runs/selected_videos.jsonl \
  --library-model-count 26 --anchors runs/anchors_v1.json \
  --new-model new-model-id --output-dir runs/insertion_pairs
```

Run held-out insertion validation before adopting an anchor set, particularly for FLT, where the reported anchor experiment did not establish the same validation. Periodically compare with complete-graph recalibration; a changed anchor set needs a new version.

## Reproduction boundary

This release provides an executable, explicit implementation of the documented equations and deployment rules. Exact recovery of the published numeric tables still needs the final benchmark ID mapping, original selected generations, full judge responses, and frozen historical API snapshots. The task-text snapshot and synthetic demo do not provide those missing records. Candidate-selection tie rules, missing reference similarity implementation, regularization normalization, and bootstrap scope are documented choices rather than hidden assumptions.
