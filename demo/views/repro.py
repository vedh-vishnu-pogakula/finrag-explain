"""Reproducibility -- every number → file → command, plus what it cost (nothing)."""
from __future__ import annotations

import json

import streamlit as st

from common import RESULTS, ROOT

ROWS = [
    ("Retrieval baseline", "b1_retrieval_{finqa,tatqa}_dev.json", "python eval/baselines/run_b1_retrieval.py --dataset finqa"),
    ("B1 answers (frozen 7B config)", "b1_rag_{finqa,tatqa}_dev.json", "python eval/baselines/run_b1_rag.py --dataset finqa   # Colab T4; --rescore recomputes verdicts with no GPU"),
    ("Generation ablation", "ablation_finqa_dev{,_7b}.json", "python eval/baselines/run_prompt_ablation.py --dataset finqa --limit 50"),
    ("Retrieval attribution (C1)", "attribution_*_dev_{score,ranking}.json", "python eval/baselines/run_attribution.py --dataset finqa"),
    ("Evidence grounding", "grounding_{finqa,tatqa}_dev.json", "python eval/baselines/run_grounding.py --dataset finqa"),
    ("B2 faithfulness per verifier", "b2_{finqa,tatqa}_dev{,_LLMVER,_LLMVER-ALT}.json", "python eval/baselines/run_b2.py --dataset finqa --phase decompose|verify [--verifier llm --model ...]"),
    ("Verifier cross-check", "verifier_crosscheck_{finqa,tatqa}_dev.json", "python eval/baselines/run_verifier_crosscheck.py --dataset finqa --limit 50"),
    ("Perturbation audit (C2)", "perturbation_{finqa,tatqa}_dev{,_LLMVER}.json", "python eval/baselines/run_perturbation.py --dataset finqa [--verifier llm --limit 80]"),
    ("Seed variance", "perturbation_variance.json", "python eval/baselines/variance_across_seeds.py"),
    ("Master verifier table", "verifier_comparison.json", "python eval/baselines/compare_verifiers.py"),
    ("Failure-case dataset (C2)", "failure_cases_{finqa,tatqa}_dev.jsonl, failure_cases_summary.json", "python eval/baselines/export_failure_cases.py"),
    ("B1/B2/B3 + selective accuracy", "b3_comparison.{json,md}", "python eval/baselines/build_b3_table.py"),
    ("Paper figures", "paper/figures/fig01..14", "python eval/figures/make_figures.py"),
]


def render() -> None:
    st.markdown('<div class="ar-eyebrow">Reproducibility</div>', unsafe_allow_html=True)
    st.markdown("Every number in the paper and in this demo is read from a file under "
                "`eval/results/`; every file is produced by one command. Per-question checkpoints "
                "make each run resumable, and the 189-test suite runs offline.")
    st.dataframe([{"artefact": a, "file(s) in eval/results/": f, "regenerate with": c} for a, f, c in ROWS],
                 width="stretch", hide_index=True, height=520)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Fixed configuration (experimental constants)")
        st.markdown(
            "- Embeddings `BAAI/bge-small-en-v1.5`, 256-token chunks, top-k 5, per-document scope\n"
            "- Generator `Qwen/Qwen2.5-7B-Instruct`, NF4 4-bit + double quantisation, fp16 compute, "
            "greedy decoding, program-of-thought, 3 exemplars in the system message\n"
            "- Grounding NLI `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, thresholds 0.5\n"
            "- Faithfulness verifier `cross-encoder/nli-deberta-v3-base`, threshold 0.5; judge "
            "decomposition budget 1024 tokens\n"
            "- Attribution: 64 Shapley permutations (exact ≤ 10 units), 256 surrogate samples, RBO depth 10\n"
            "- Subsample 250 per dataset, seeded and nested; permutation tests 20,000 resamples; seeds 0–3")
        st.markdown("#### Cost")
        st.markdown("**₹0.** Generation, decomposition and LLM verification on a free Colab Tesla T4; "
                    "everything else on a laptop CPU. A billing guard strips every paid-provider "
                    "credential before any RAGAS call and substitutes a local judge, so the "
                    "OpenAI default can never be reached.")
    with c2:
        st.markdown("#### Environment that produced the numbers")
        env = RESULTS / "colab_env.json"
        if env.exists():
            e = json.loads(env.read_text())
            st.markdown(f"Colab: `{e.get('gpu')}` · Python {e.get('python')}")
            st.dataframe([{"package": k, "version": v} for k, v in e.get("packages", {}).items()],
                         width="stretch", hide_index=True, height=330)
        req = ROOT / "requirements.txt"
        if req.exists():
            with st.expander("requirements.txt (laptop, pinned)"):
                st.code(req.read_text(), language="text")
    st.markdown("#### What could not be run, and why")
    st.markdown(
        "- **RAGAS's `FaithfulnesswithHHEM`**: Vectara HHEM's custom modelling code predates "
        "`transformers` 5.x and fails to load under the pinned version (`AttributeError: "
        "'HHEMv2ForSequenceClassification' object has no attribute 'all_tied_weights_keys'`). "
        "A standard NLI cross-encoder stands in; the cross-check measures the consequence.\n"
        "- **An independent non-Qwen 7B verifier**: Falcon3-7B verified at 95 s/question and "
        "OLMo-2-7B at 74 s/question on the free T4 (≈ 10 s for Qwen and Phi) — 10–13 GPU-hours "
        "for 500 questions, beyond a free session. Disclosed in the paper's Limitations.")
