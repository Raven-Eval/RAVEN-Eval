"""Confidence-aware per-task Davidson MLE and stratified pair bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


LABELS = {"better": "win", "worse": "loss", "comparable": "tie", "similar": "tie",
          "更好": "win", "更差": "loss", "相近": "tie", "相似": "tie"}


@dataclass
class Task:
    task_id: str
    models: list[str]
    pairs: list[tuple[str, str]]
    weights: np.ndarray  # rows: pair-level reliability times averaged judge outcome mass
    confident: list[int]
    judge_count: int


def label(value):
    if isinstance(value, dict):
        value = value.get("output_text", value.get("label", ""))
    return LABELS.get(str(value).strip().casefold())


def merge(forward, reverse):
    """Reverse labels are relative to B/A, then mapped back to A/B."""
    if (forward, reverse) == ("win", "loss"):
        return np.array([1., 0., 0.]), True
    if (forward, reverse) == ("loss", "win"):
        return np.array([0., 1., 0.]), True
    if (forward, reverse) == ("tie", "tie"):
        return np.array([0., 0., 1.]), True
    return np.array([.5, .5, 0.]), False


def load_task(path, bases):
    data = json.loads(path.read_text(encoding="utf-8"))
    models, pairs, weights, confident = set(), [], [], []
    seen = set()
    for triple in data["pairwise_comparison_triples"]:
        a, b = (triple[f"generated_video_{side}"]["model"] for side in ("a", "b"))
        if a == b:
            raise ValueError(f"Self comparison in {path.name}")
        canonical = tuple(sorted((a, b)))
        if canonical in seen:
            raise ValueError(f"Duplicate model pair in {path.name}: {canonical}")
        seen.add(canonical)
        mass, count = np.zeros(3), 0
        for base in bases:
            forward = label(triple.get(base + "_evaluation"))
            reverse = label(triple.get(base + "_reversed_evaluation"))
            if forward is None or reverse is None:
                raise ValueError(f"Missing/invalid {base} forward/reverse result in {path.name}, pair {canonical}")
            outcome, stable = merge(forward, reverse)
            mass += outcome
            count += int(stable)
        reliability = count / len(bases)
        models.update((a, b)); pairs.append((a, b))
        weights.append(reliability * mass / len(bases)); confident.append(count)
    if not pairs:
        raise ValueError(f"No comparisons in {path.name}")
    return Task(str(data.get("task_id", path.stem)), sorted(models), pairs,
                np.asarray(weights), confident, len(bases))


def connected(task, multiplicity=None):
    graph = {m: set() for m in task.models}
    counts = np.ones(len(task.pairs)) if multiplicity is None else multiplicity
    for (a, b), w, n in zip(task.pairs, task.weights, counts):
        if n > 0 and w.sum() > 0:
            graph[a].add(b); graph[b].add(a)
    visited, stack = set(), [task.models[0]]
    while stack:
        model = stack.pop()
        if model not in visited:
            visited.add(model); stack.extend(graph[model] - visited)
    return len(visited) == len(task.models)


def objective(params, indices, mass, pair_count, alpha=1e-3, beta=1e-4, gauge=10.):
    n = len(params) - 1
    s, gamma = params[:n], params[-1]
    i, j = indices[:, 0], indices[:, 1]
    logits = np.column_stack((s[i], s[j], gamma + .5 * (s[i] + s[j])))
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    probabilities = np.exp(log_probs)
    residual = mass.sum(axis=1, keepdims=True) * probabilities - mass
    grad_s = np.zeros(n)
    np.add.at(grad_s, i, residual[:, 0] + .5 * residual[:, 2])
    np.add.at(grad_s, j, residual[:, 1] + .5 * residual[:, 2])
    mean = s.mean()
    loss = -(mass * log_probs).sum() / pair_count + alpha / n * (s @ s) + beta * gamma**2 + gauge * mean**2
    grad_s = grad_s / pair_count + 2 * alpha / n * s + 2 * gauge * mean / n
    grad_gamma = residual[:, 2].sum() / pair_count + 2 * beta * gamma
    return float(loss), np.r_[grad_s, grad_gamma]


def fit(task, multiplicity=None):
    """Keep all original nodes even when a bootstrap replicate drops their edges."""
    counts = np.ones(len(task.pairs)) if multiplicity is None else np.asarray(multiplicity)
    index = {m: i for i, m in enumerate(task.models)}
    indices = np.array([(index[a], index[b]) for a, b in task.pairs], dtype=int)
    result = minimize(objective, np.zeros(len(task.models) + 1),
                      args=(indices, task.weights * counts[:, None], float(counts.sum())),
                      method="L-BFGS-B", jac=True,
                      options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-6})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Optimization failed for {task.task_id}: {result.message}")
    skills = result.x[:-1] - result.x[:-1].mean()
    return dict(zip(task.models, map(float, skills))), float(result.x[-1])


def aggregate(fits):
    values = defaultdict(list)
    for scores in fits:
        for model, score in scores.items():
            values[model].append(score)
    rows = []
    for model, scores in values.items():
        mean, std = float(np.mean(scores)), float(np.std(scores, ddof=0))
        score = mean - std
        rows.append({"model": model, "score": score, "elo": 1000 + 400 / math.log(10) * score,
                     "mean": mean, "std": std, "task_count": len(scores)})
    rows.sort(key=lambda r: (-r["score"], r["model"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def bootstrap(tasks, rounds, budget, seed):
    if budget < len(tasks):
        raise ValueError("Bootstrap pair budget must cover every task at least once")
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    disconnected = Counter()
    for iteration in range(rounds):
        allocation = np.full(len(tasks), budget // len(tasks), dtype=int)
        allocation[rng.choice(len(tasks), budget % len(tasks), replace=False)] += 1
        fits = []
        for task, n in zip(tasks, allocation):
            multiplicity = rng.multinomial(int(n), np.full(len(task.pairs), 1 / len(task.pairs)))
            if not connected(task, multiplicity):
                disconnected[task.task_id] += 1
            fits.append(fit(task, multiplicity)[0])
        for row in aggregate(fits):
            samples[row["model"]].append(row["elo"])
        if (iteration + 1) % 10 == 0:
            print(f"bootstrap {iteration + 1}/{rounds}", flush=True)
    return {m: list(map(float, np.percentile(v, [2.5, 97.5]))) for m, v in samples.items()}, dict(disconnected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--judge-base", action="append", required=True, help="Repeat once per fixed judge; exactly named fields only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=0)
    parser.add_argument("--bootstrap-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260631)
    args = parser.parse_args(argv)
    if args.bootstrap_rounds < 0 or args.bootstrap_pairs < 1 or len(set(args.judge_base)) != len(args.judge_base):
        parser.error("Invalid bootstrap counts or duplicate judge bases")
    paths = sorted({p.resolve() for d in args.input_dir for p in d.glob("*_pairwise_triples.json")})
    if not paths:
        parser.error("No pairwise task JSON files")
    tasks = [load_task(p, args.judge_base) for p in paths]
    if len({t.task_id for t in tasks}) != len(tasks):
        parser.error("Duplicate task IDs across inputs")
    for task in tasks:
        if not connected(task):
            parser.error(f"Effective comparison graph disconnected for {task.task_id}; complete/recheck judgments")
    fitted = [fit(t) for t in tasks]
    rows = aggregate([f[0] for f in fitted])
    intervals, disconnected = bootstrap(tasks, args.bootstrap_rounds, args.bootstrap_pairs, args.seed) if args.bootstrap_rounds else ({}, {})
    votes = Counter()
    for task in tasks:
        for a, b in task.pairs:
            votes[a] += 2 * task.judge_count; votes[b] += 2 * task.judge_count
    for row in rows:
        row['ci_low'], row['ci_high'] = intervals.get(row['model'], [None, None])
        row['votes'] = votes[row['model']]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ['rank', 'model', 'elo', 'ci_low', 'ci_high', 'votes', 'score', 'mean', 'std', 'task_count']
    with (args.output_dir/'leaderboard.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir/'task_scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['task_id', 'model', 'skill', 'log_eta'])
        for task, (scores, gamma) in zip(tasks, fitted):
            writer.writerows((task.task_id, model, score, gamma) for model, score in scores.items())
    metadata = {'method': 'confidence-aware per-task Davidson; mean_minus_std; Elo',
                'judge_bases': args.judge_base, 'task_count': len(tasks), 'leaderboard': rows,
                'normalization': 'likelihood / pair_count; judge masses / M; skill L2 / model_count',
                'optimizer': {'maxiter':1000,'ftol':1e-11,'gtol':1e-6,'alpha':1e-3,'beta':1e-4,'gauge':10},
                'bootstrap': {'rounds':args.bootstrap_rounds,'pairs_per_round':args.bootstrap_pairs,'seed':args.seed,
                              'unit':'model pair within task, keeping all judge directions',
                              'disconnected_replicates_by_task':disconnected,
                              'node_policy':'original task node sets retained; regularized fits; all replicates included'},
                'point_graphs_connected':True,'source_files':[p.name for p in paths]}
    (args.output_dir/'leaderboard.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(f"Ranked {len(rows)} models over {len(tasks)} tasks; output: {args.output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
