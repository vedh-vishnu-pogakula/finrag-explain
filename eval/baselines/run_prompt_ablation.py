"""
Generation-configuration ablation for B1.

CLAUDE.md fixes the generation model and prompt as *experimental constants* across B1/B2/B3,
which means there is exactly one legitimate moment to choose them: before B2 exists. This
script is that moment. It runs the same questions, the same retriever and the same index
through several generation configurations, so the choice is made on evidence and the evidence
survives as a table.

The arms are designed to be *separable*. "We changed the prompt and accuracy went up" is not
a result; it does not say which change did the work, and a reviewer will ask. So:

    direct-0shot   the original v1 baseline, byte-identical prompt   -> the reference point
    direct-3shot   + few-shot exemplars, still answering directly    -> isolates exemplars
    pot-0shot      + program-of-thought, no exemplars                -> isolates PoT
    pot-3shot      + both                                            -> the interaction

A fifth arm varies the model instead of the prompt, which separates "better prompting" from
"bigger model" -- two explanations that a single improved number cannot distinguish.

Every arm costs nothing: local open-weights inference on MPS/CUDA/CPU, per the project's
zero-budget constraint. Runtime, not money, is the reason it defaults to a 50-question slice.
That slice is a *prefix* of the 250-question evaluation subsample (subsampling shuffles once
then slices, so the sets are nested) -- the ablation is therefore run on a subset of the same
questions the final number is reported on, not on an unrelated draw.

    python eval/baselines/run_prompt_ablation.py --dataset finqa --limit 50
    python eval/baselines/run_prompt_ablation.py --dataset finqa --arms pot-3shot,pot-3shot-3b

Results land in eval/results/ablation_<dataset>_<split>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "eval" / "baselines" / "run_b1_rag.py"

# (name, extra CLI flags). The tag keeps each arm's report and per-question checkpoint in its
# own file, so a resumed or re-run arm can never contaminate another one's records.
ARMS = {
    "direct-0shot":   ["--no-program", "--n-shot", "0"],
    "direct-3shot":   ["--no-program", "--n-shot", "3"],
    "pot-0shot":      ["--n-shot", "0"],
    "pot-3shot":      ["--n-shot", "3"],
}
DEFAULT_ARMS = list(ARMS)


def run_arm(name: str, dataset: str, split: str, limit: int, fresh: bool,
            model: str | None = None, load_in_4bit: bool = False,
            tag_prefix: str = "", no_4bit: bool = False) -> dict | None:
    """Run one arm as a subprocess.

    Model and quantization are applied to *every* arm rather than being arms themselves, so
    the same four-way prompt comparison can be replicated at a second model size. A prompt
    finding measured only at 1.5B is a claim about 1.5B; showing the same ordering at 7B is
    what turns it into a claim about the prompt.
    """
    tag = f"_ABL{tag_prefix}-{name}"
    cmd = [sys.executable, str(RUNNER), "--dataset", dataset, "--split", split,
           "--limit", str(limit), "--tag", tag, *ARMS[name]]
    if model:
        cmd += ["--model", model]
    if load_in_4bit:
        cmd.append("--load-in-4bit")
    if no_4bit:
        cmd.append("--no-4bit")
    if fresh:
        cmd.append("--fresh")
    print(f"\n{'=' * 70}\n[ablation] {name}: {' '.join(ARMS[name]) or '(defaults)'}\n{'=' * 70}")
    t0 = time.time()
    # Streamed, not captured: a 50-question local run is slow enough that silent output looks
    # like a hang, and the runner already prints its own progress every 10 questions.
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[ablation] {name} FAILED (exit {proc.returncode}) -- skipping this arm")
        return None
    report_path = ROOT / "eval" / "results" / f"b1_rag_{dataset}_{split}{tag}.json"  # noqa: E501
    if not report_path.exists():
        print(f"[ablation] {name}: no report at {report_path}")
        return None
    report = json.loads(report_path.read_text())
    report["_arm"] = name
    report["_runtime_s"] = round(time.time() - t0, 1)
    return report


def row_for(report: dict) -> dict:
    a = report.get("answers", {})
    cfg = report.get("config", {})
    fm = report.get("failure_modes", {})
    return {
        "arm": report["_arm"],
        "model": cfg.get("generation_model", "?").split("/")[-1],
        "prompt": cfg.get("prompt_version", "?"),
        "numeric_accuracy": a.get("numeric_accuracy"),
        "span_f1": a.get("span_token_f1"),
        "overall_em": a.get("overall_exact_match"),
        "program_rate": report.get("program_rate"),
        "unparseable": report.get("unparseable_json_rate"),
        "declined": report.get("declined_rate"),
        "citation_precision": report.get("citation_precision"),
        "n_generation_failures": fm.get("generation", 0),
        "n_retrieval_failures": fm.get("retrieval", 0),
        "runtime_s": report["_runtime_s"],
    }


def to_markdown(rows: list[dict], dataset: str, split: str, limit: int) -> str:
    cols = [("arm", "Arm"), ("model", "Model"), ("numeric_accuracy", "Numeric acc."),
            ("span_f1", "Span F1"), ("program_rate", "Program rate"),
            ("unparseable", "Unparseable"), ("declined", "Declined"),
            ("citation_precision", "Citation prec."), ("runtime_s", "Runtime (s)")]
    out = [f"### B1 generation ablation -- {dataset}/{split}, n={limit}", "",
           "Retriever, index, top-k and question set are identical across arms; only the",
           "generation configuration varies. `Program rate` is the share of answers produced",
           "by executing a model-written expression rather than by the model stating a value.",
           ""]
    out.append("| " + " | ".join(label for _, label in cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        out.append("| " + " | ".join(
            "n/a" if r.get(key) is None else str(r.get(key)) for key, _ in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="B1 generation-configuration ablation")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=50,
                    help="questions per arm (a prefix of the 250-question eval subsample)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help=f"comma-separated subset of: {', '.join(ARMS)}")
    ap.add_argument("--fresh", action="store_true", help="ignore existing arm checkpoints")
    ap.add_argument("--model", default=None,
                    help="apply this model to every arm (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit NF4 for every arm (CUDA only; needed for 7B on a free T4)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="override generation.load_in_4bit=true for a laptop run; pair with "
                         "a smaller --model")
    ap.add_argument("--tag-prefix", default="",
                    help="distinguishes result files between model sizes, e.g. --tag-prefix 7b")
    args = ap.parse_args()

    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        sys.exit(f"[ablation] unknown arm(s) {unknown}; choose from {list(ARMS)}")

    print(f"[ablation] {len(names)} arms x {args.limit} questions on "
          f"{args.dataset}/{args.split}. Local inference only -- this costs nothing to run.")

    prefix = f"-{args.tag_prefix}" if args.tag_prefix else ""
    reports = [r for r in (run_arm(n, args.dataset, args.split, args.limit, args.fresh,
                                   args.model, args.load_in_4bit, prefix, args.no_4bit)
                           for n in names) if r]
    if not reports:
        sys.exit("[ablation] every arm failed; nothing to summarize")

    rows = [row_for(r) for r in reports]
    out_dir = ROOT / "eval" / "results"
    suffix = f"_{args.tag_prefix}" if args.tag_prefix else ""
    stem = out_dir / f"ablation_{args.dataset}_{args.split}{suffix}"
    stem.with_suffix(".json").write_text(json.dumps(
        {"dataset": args.dataset, "split": args.split, "limit": args.limit, "rows": rows},
        indent=2))
    markdown = to_markdown(rows, args.dataset, args.split, args.limit)
    stem.with_suffix(".md").write_text(markdown)

    print("\n" + markdown)
    best = max(rows, key=lambda r: r["numeric_accuracy"] or 0)
    print(f"[ablation] best numeric accuracy: {best['arm']} = {best['numeric_accuracy']}")
    print(f"[ablation] wrote {stem}.json and {stem}.md")


if __name__ == "__main__":
    main()
