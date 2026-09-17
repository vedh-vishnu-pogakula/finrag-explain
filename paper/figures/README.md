# Paper figures

Regenerate all of them with one command; nothing here is hand-drawn or hand-typed:

```bash
python eval/figures/make_figures.py          # -> paper/figures/fig??_*.{png,pdf}
```

Every figure reads a result file under `eval/results/` and prints that path in its own
footer. If a result file changes (e.g. the Falcon3 verifier rows from Colab cell 10g land),
re-run the command and the affected figures update; a figure whose source is missing is
skipped, never drawn from memory. Verifier colours are fixed (NLI = blue, Phi-3.5 = orange,
Qwen2.5-7B = aqua, Falcon3 = yellow) so a verifier looks the same in every figure.

| # | file | shows | source |
|---|---|---|---|
| 1 | `fig01_generation_ablation` | prompt form × exemplars at 1.5B and 7B; program-of-thought dominant at both sizes | `ablation_finqa_dev{,_7b}.json` |
| 2 | `fig02_retrieval_recall` | recall@k / hit@k / all-gold@k, k = 1,3,5,10 — retrieval is not the bottleneck | `b1_retrieval_*_dev.json` |
| 3 | `fig03_failure_modes` | where B1 loses: generation ≫ retrieval | `b1_rag_*_dev.json` |
| 4 | `fig04_attribution_lift` | comprehensiveness lift over random units, three estimators, both datasets + ranking mode | `attribution_*_dev_{score,ranking}.json` |
| 5 | `fig05_attribution_mass_by_kind` | attribution mass by unit kind; Shapley puts 3.0% on stopwords vs occlusion 11.3% | `attribution_finqa_dev_score.json` |
| 6 | `fig06_grounding_by_outcome` | groundedness and grounds-to-gold by outcome | `grounding_*_dev.json` |
| 7 | `fig07_verifier_master` | **the thesis figure**: separation / specificity / provenance per verifier, with significance | `verifier_comparison.json` |
| 8 | `fig08_faithfulness_correct_vs_wrong` | mean faithfulness of correct vs wrong answers per verifier (NLI inverted on FinQA; Phi highest mean, no gap) | `checkpoints/b2_*_dev*.jsonl` |
| 9 | `fig09_restatement_mechanism` | faithfulness when the answer copies a figure vs computes one — the mechanism | `failure_cases_*_dev.jsonl` |
| 10 | `fig10_perturbation_audit` | control / operand-sentence removed / random removed, per verifier — Contribution 2 | `perturbation_*_dev{,_LLMVER}.json` |
| 11 | `fig11_specificity_across_seeds` | specificity per seed with mean ± sd; deterministic arms byte-identical | `perturbation_variance.json` |
| 12 | `fig12_selective_accuracy` | accuracy among verified answers vs coverage for B1 / B2 / B3 flags — B3's practical value | `b3_comparison.json` |
| 13 | `fig13_failure_case_labels` | the labeled failure-case dataset at a glance | `failure_cases_summary.json` |
| 14 | `fig14_verifier_crosscheck` | 2×2 agreement between NLI and LLM verdicts, statement-level κ | `verifier_crosscheck_*_dev.json` |

Suggested placement: 7, 10, 12 are the results section's load-bearing figures; 8, 9, 14 support
the mechanism; 1–6 belong to the system/baseline sections; 11 and 13 to robustness and the
dataset deliverable.
