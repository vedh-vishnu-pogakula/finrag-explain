# Explainable Financial RAG — Project Memory

This file is read automatically by Claude Code at the start of every session in this repo.
It exists so you don't have to re-explain the project each time you open a new terminal/session.
## Working Rules (Strict — Deadline Critical)

### 1. No Hallucination
- Never assume file contents, APIs, library behavior, or project structure. Always read/check the actual file or source before referencing it.
- If unsure about something, say "I don't know / need to verify" instead of guessing.
- Never invent function names, imports, config keys, or file paths — verify they exist first.
- If a dependency/version isn't confirmed, check package.json / requirements.txt / lockfile before using it.

### 2. Plan Before Code — Always
- For any non-trivial task: first output a short plan (what will change, which files, why).
- Wait for explicit confirmation ("go" / "approved" / "yes") before writing code.
- Do NOT jump directly into code generation on the first response.
- No plan changes mid-way unless something factually breaks the current plan — if it does, flag it clearly and explain why before pivoting.

### 3. Review Before Finalizing
- After planning and before final code: do a quick self-check — does this match the stated goal, existing codebase conventions, and constraints?
- Call out risks/assumptions explicitly rather than silently proceeding.

### 4. Token & Time Discipline
- Be concise. No repeating the question, no restating obvious context, no filler explanations.
- No over-explaining basic concepts unless explicitly asked.
- Answers should be direct: plan → confirmation → code/execution. No unnecessary preamble.

### 5. Stability Over Exploration
- Don't propose alternate approaches unless asked or unless the current plan is provably broken.
- Stick to the agreed plan/architecture unless new info requires a change — and justify any change briefly.

### 6. When Uncertain
- Ask a single clarifying question instead of guessing and building on a wrong assumption.
## Project

**Title:** Explainable Financial RAG — Retrieval Attribution & Faithfulness-Verified Explanations
for Financial Question Answering
**Type:** B.E. CS/AI-ML final-year major project, CBIT Hyderabad, 8-month timeline
**Status (2026-09-16, resumed after a 20-day gap; hard deadline ~2026-09-23):** Months 1–5
complete, Month 6 mostly done, Months 7–8 compressed into the final week — see "Deadline
week" below. 164/164 tests pass offline; `requirements.txt` pinned; spaCy `en_core_web_sm`
installed. The Colab cells 10e/10f (alt judge, LLM-verifier perturbation) were **never
successfully run** — no `*_LLMVER-ALT` or `perturbation_*_LLMVER` file exists in
`eval/results/`. That is the only remaining GPU work.

Month 2: both loaders done and tested.

Month 3: baseline B1 end to end. Retrieval FinQA Recall@5 0.850 / MRR 0.783, TAT-QA 0.843 /
0.851 — **retrieval is not the bottleneck**, 4–5% of failures are retrieval failures.
**Frozen generation config: `Qwen2.5-7B-Instruct` 4-bit, program-of-thought, 3 exemplars**,
run on Colab's free T4. B1 on 250 questions/dataset: FinQA execution accuracy **0.450**,
TAT-QA numeric **0.541** / span F1 **0.745**; citation precision **0.787 / 0.768**; structured
output parsed 250/250. Against the original B1 (1.5B, direct prompt, display-string scoring)
that is FinQA 0.04 → 0.45. Three separable causes, all in
`eval/results/ablation_finqa_dev{,_7b}.md`: program-of-thought (dominant, +0.14 at 7B), three
fixed exemplars (+0.02), and scoring against FinQA's official `exe_ans` (+0.026).

Month 4: Contribution 1 — `src/attribution/`. All methods beat a random-unit baseline by
~0.48–0.50 normalized comprehensiveness; on the ranking value function Shapley's lift (0.451)
is 44% above occlusion's (0.314), and Shapley puts 3.0% of mass on stopwords vs occlusion's
11.3%.

Month 5: `src/grounding/` (sentence/table-row segmentation, DeBERTa-MNLI entailment,
operand-provenance grounding for numeric answers) and `src/faithfulness/` (`ragas_local.py`
local judge + tested billing guard; `staged.py` decomposition cached, verification
repeatable). Grounding: FinQA 0.757 groundedness / 68.8% grounds-to-gold, TAT-QA 0.664 /
58.0%. B2 across four verifiers — **see the finding below**.

Month 6 (in progress): perturbation audit built and run with the NLI verifier
(`src/faithfulness/perturb.py`, `eval/baselines/run_perturbation.py`). Specificity **+0.087**
(FinQA, p=0.005) and **+0.228** (TAT-QA, p<0.0001) — the metric *passes* the audit while
failing to separate correct from incorrect answers, because the operand-supplying sentence is
also the lexically-overlapping one. **Perturbation sensitivity is necessary but not sufficient
for metric validity** — that is the methodological contribution. Still pending: the same audit
under the LLM verifier (Colab cell 10f), a second judge family (cell 10e), and the labeled
failure-case dataset. `python eval/baselines/compare_verifiers.py` regenerates the master table.

## Deadline week (2026-09-16 → ~2026-09-23) — what is left, in priority order

Everything below is zero-GPU except item 1, which is one Colab run the user starts first and
that runs in the background while the rest is built locally.

1. **Colab 10e + 10f** (user; ~2–3 h T4). Before Run all: *File → Revert to saved version*,
   confirm cell 4 prints `Notebook version 2026-09-16.1 matches the repo`, confirm cell 12's
   manifest shows every file OK. Cell 13 zips and downloads the 8 files: `b2_{finqa,tatqa}_dev_LLMVER-ALT.{json,jsonl}`,
   `perturbation_{finqa,tatqa}_dev_LLMVER.{json,jsonl}` → `eval/results/` and
   `eval/results/checkpoints/` (`unzip -o ~/Downloads/finrag_artifacts_10e_10f.zip -d .`). Then `python eval/baselines/compare_verifiers.py` fills the two
   `pending` LLM-specificity cells and adds the alt-judge rows.
2. **Labeled failure-case dataset** (Contribution 2's stated deliverable) — export from the
   perturbation + B2 files already on disk: per (question, verifier) the cases where the metric
   moved the wrong way (targeted drop ≤ random drop) or scored a wrong answer as faithful /
   a correct grounded answer as unfaithful, with the mechanism label. No model calls.
3. **B3 table** — B3 is the composition of results already computed on the same 250 questions
   (B1 answer + grounding + faithfulness-across-verifiers + perturbation specificity). One
   script joins them by `qa_id` into the B1/B2/B3 comparison the brief promised. No GPU.
4. **Streamlit demo** — replay mode over the 250 checkpointed questions (retrieval + attribution
   live, since they are embedding-only and cheap; generation / grounding / faithfulness shown
   from the checkpoints). Live 7B generation is not possible on the laptop; do not fake it.
5. **Variance** — generation and both verifiers are greedy/deterministic by design, so
   "run 3×" is a no-op; the honest variance figure is the random-removal arm re-drawn under
   3 seeds (local NLI, free). Report it as such.
6. **Paper / report** — `paper/` holds only the literature review + §5.5 draft. Results
   sections come from the README tables; every number must trace to a file in `eval/results/`.
7. Optional if time remains: ~50-statement human spot-check CSV for the user to label.

## Hard constraint: zero budget

**This project must cost ₹0.** No paid APIs, no metered tokens, no billing account — anywhere,
at any month. It's an unfunded college project, so "it only costs a few dollars" is blocked,
not a tradeoff.

- Generation runs on a **local open-weights model** (`configs/config.yaml:
  generation.provider: local`, Qwen2.5-1.5B-Instruct) via `transformers`, on Apple MPS or
  Colab's free T4. `provider: api` exists but is opt-in, prints a cost warning, and is
  unreachable from the default config.
- **The real trap is Month 6:** RAGAS defaults to an OpenAI judge LLM and will bill silently
  the first time it's called. It must be constructed with an explicit local/free judge.
- Everything else is already free and local: bge-small embeddings, FAISS, spaCy, both datasets.
- Free-tier hosted APIs (Gemini/Groq/OpenRouter) are an acceptable fallback only if local
  inference proves too slow — but open weights are the stronger reproducibility claim in a
  viva, so prefer local.

## Month 5 finding that reframes the paper — read before writing anything up

Measured, on the frozen 7B config, 250 questions per dataset. **A reported faithfulness score
is not interpretable without naming the verifier that produced it.**

- Same answers, same contexts, three independent NLI verifiers: FinQA mean faithfulness
  **0.163 / 0.237 / 0.383**. Adding RAGAS's own LLM verifier gives 0.34 (FinQA) and 0.80
  (TAT-QA) against the NLI verifier's 0.24 / 0.37.
- RAGAS presents its LLM verifier (`Faithfulness`) and its model verifier
  (`FaithfulnesswithHHEM`) as interchangeable. Statement-level agreement between an NLI
  verifier and the LLM one is **at or below chance**: Cohen's kappa **-0.147** (FinQA),
  **+0.157** (TAT-QA), n=50 each.
- Validated against a *deterministic* reference — operand provenance from the grounding layer,
  which involves no model at all. If every figure an answer consumed is provably in the
  retrieved evidence, the answer is grounded by construction. Separation between grounded and
  ungrounded answers: **-0.023, +0.027, +0.103** across the three NLI verifiers. Essentially
  none.
- Mechanism, both datasets, p<0.001: faithfulness tracks whether the answer *restates* a
  figure present in the evidence (FinQA 0.476 vs 0.195; TAT-QA 0.502 vs 0.154), not whether it
  is grounded. NLI cannot verify arithmetic, so a correct computed answer looks unsupported
  while a wrong copied one looks supported.

**RESOLVED by the full 250-question LLM-verifier run.** RAGAS's *default* LLM verifier works;
the cheap model-based verifier is the one that breaks. Separation between correct and
incorrect answers:

| verifier | FinQA correct / wrong | sep | TAT-QA correct / wrong | sep |
|---|---|---|---|---|
| NLI cross-encoder | 0.186 / 0.284 | **-0.098 (inverted)** | 0.457 / 0.311 | +0.146 |
| **LLM (RAGAS default)** | **0.513 / 0.256** | **+0.257** | **0.818 / 0.589** | **+0.229** |

**Do not state this as "RAGAS faithfulness is broken" — that was an earlier, wrong reading.**
The defensible claim is a *domain-specific counterexample to the field's own prescription*:
RAGBench and ARES both argue for replacing RAGAS's LLM judge with cheaper fine-tuned models,
and RAGAS itself ships `FaithfulnesswithHHEM` for cost reasons. On numerical financial QA
that substitution **inverts the metric**. The mechanism is measured: entailment models cannot
verify arithmetic, so faithfulness collapses to lexical restatement (p<0.001, both datasets).

That makes the LLM judge the one to use here and the cheap verifier a trap — the opposite of
the general-domain recommendation, which is precisely what makes it worth publishing.

**Consequence for B2:** report faithfulness as a *table across verifiers*, never as one
number. A single B2 faithfulness figure would contradict the paper's own thesis.

## The two contributions (do not scope-creep beyond these)

1. **Retrieval Attribution** — quantify which query tokens/entities/numeric values drove a
   retrieved chunk's score. Perturb the query (occlusion / local-surrogate), re-score against the
   retriever's similarity function. Anchor methodology on RankingSHAP/Rank-LIME — NOT a naive
   off-the-shelf SHAP/LIME call. This is training-free / perturbation-based, never
   fine-tuning-based — do not let generated text (code comments, paper drafts, README, etc.)
   describe it as fine-tuning.
2. **Faithfulness-of-Faithfulness Testing** — run RAGAS, then perturb/remove the evidence it
   claims is important, and check whether the RAGAS score moves in the expected direction.
   Produces a labeled dataset of RAGAS failure cases.

A supporting **evidence-grounding** layer (sentence-level answer↔context matching) feeds both
contributions but is not itself a novelty claim.

## Base paper / closest related work

**Wording that must be fixed before submission:** this project does **not** extend the base
paper's architecture. MC-Dropout uncertainty-penalised retrieval was never implemented. What
it does is address two gaps that paper leaves open (no attribution trail, no faithfulness-
metric testing). Say "addresses limitations of", never "extends" or "builds on" — claiming to
extend an architecture that appears nowhere in the code is exactly what unravels in a viva.

- **Base paper (closest architecture, NOT extended in code):** Bayesian RAG — Ngartera,
  *Frontiers in Artificial Intelligence*, 2025/2026. MC-Dropout uncertainty-penalized retrieval
  scoring on SEC 10-K filings. No source-attribution trail, no faithfulness-metric testing —
  those are the two gaps this project addresses. Nadarajah & Koina are co-authors.
- **Closest peer-reviewed competitor to differentiate from:** FinRAG-12B (Katerenchuk, Duboue &
  Evanini, ACL 2026 Industry Track) — production banking RAG system with document-level citation
  tags and a static faithfulness sub-score. Proprietary fine-tune, not something to reproduce in
  code — cite it, don't build on it.

## Datasets (v1 scope — do not add SEC filings/other corpora without re-confirming scope)

- **FinQA** (Chen et al., 2021) — numerical reasoning over S&P 500 earnings reports; has gold
  supporting-fact indices (`gold_inds`) — use these directly for Precision@k/Recall@k, don't
  invent a new labeling scheme. Loader done: `src/ingestion/finqa_loader.py`.
- **TAT-QA** (Zhu et al., 2021) — hybrid tabular/textual financial QA. Loader done:
  `src/ingestion/tatqa_loader.py`, tested against the real dev set. Table-evidence
  `gold_chunk_ids` are a documented best-effort heuristic (no gold row index exists in the
  source data) — don't treat them as ground truth in eval code without re-reading
  `_table_gold_ids`'s docstring.

## Baselines

- B1 — Plain RAG (retriever + generator, no explanation layer)
- B2 — RAG + RAGAS only
- B3 — Full system (B2 + attribution + evidence grounding + perturbation-validated faithfulness)

## Tech stack

- Python 3.10+, pandas/NumPy, spaCy (sentence segmentation, financial NER)
- Embeddings: sentence-transformers (BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2)
- Retrieval: FAISS, single retriever only in v1 (no reranker, no BM25 hybrid — stretch goal only)
- Generation: one fixed instruction-tuned open-weights LLM run locally (Qwen2.5-1.5B-Instruct
  via `transformers`), evidence-bound prompt — see the zero-budget constraint above
- Faithfulness: RAGAS (`ragas` package) + custom perturbation harness
- Demo: Streamlit (shell exists at `demo/streamlit_app.py`, cached resource loading already
  wired, not yet connected to a real pipeline); experiment tracking: MLflow or structured CSV/JSON

## Shared internal schema

Every dataset loader must output `Chunk` and `Question` from `src/ingestion/schema.py` — see
that file's docstring. Downstream code (retrieval, attribution, grounding) only ever imports
from `schema.py`, never from `finqa_loader.py`/`tatqa_loader.py` directly, so it stays
dataset-agnostic even though the two raw formats are quite different.

## Repo structure (Section 7 of brief)

```
finrag-explain/
  data/{raw,processed,sample}/   # raw+processed are gitignored; sample/ has tiny checked-in fixtures
  src/{ingestion,retrieval,attribution,grounding,generation,faithfulness}/
  eval/{baselines,metrics,results}/
  demo/streamlit_app.py
  notebooks/
  configs/config.yaml
  paper/
  scripts/download_data.sh
  tests/
  requirements.txt
  README.md
```

## Engineering guardrails — read before writing code in these areas

- **Attribution (Month 4):** never re-run the full RAG pipeline per perturbation. Pre-compute
  chunk embeddings once, batch-embed query perturbations, score via a single matrix multiply
  against cached embeddings. If you catch yourself calling the generator LLM inside a
  perturbation loop — stop, that's the compute-exhausting anti-pattern. **The hook for this
  already exists:** `Retriever.score_query_variants(variants, doc_id)` in
  `src/retrieval/retriever.py` does exactly one embedding pass and one matmul, returning an
  (n_variants × n_candidate_chunks) score matrix plus the candidate chunks in column order.
  Attribution code should call that, never `Retriever.retrieve` in a loop. **Now built on:**
  `src/attribution/perturbation.py::PerturbationScorer` adds mask-level caching, and every
  estimator collects all the coalitions it needs *before* scoring any of them — so a Shapley
  run is one batch, not one per permutation (60–75% cache hit rate in practice). Explaining k
  chunks costs the same as explaining one: coalitions depend only on the query, and one matmul
  scores every chunk.
- **Attribution evaluation:** there is no gold explanation to score against, so never claim
  "correct attribution". Faithfulness only — comprehensiveness and sufficiency — and **always
  against a random-unit baseline**, because removing any 20% of a query lowers the score
  somewhat and a bare comprehensiveness number proves nothing. Report the lift over random.
  Negative sufficiency is a real property of dense retrieval (dropping the rest of the query
  can score higher than the full query), not a bug to fix.
- **Retrieval scope:** per-document is the default and is what the gold labels were written
  for — FinQA `gold_inds` and TAT-QA `rel_paragraphs` index into one filing, and chunk_ids are
  only unique within a doc_id (every FinQA doc has a `text_3`). Anything keyed on chunk_id
  alone, without doc_id, is a bug. `retrieval.scope: corpus` exists for a harder, more
  deployment-like setting but the labels don't match it.
- **Generation:** the frozen configuration is `local_model` + `use_program` + `n_shot` +
  `load_in_4bit` **together** (`configs/config.yaml`), across all baselines — changing any one
  invalidates B1/B2/B3 comparability, so they are experimental constants, not knobs. Decoding
  is greedy (`do_sample=False`): a baseline that answers differently on every run can't be
  compared, and sampling noise would be indistinguishable from the Month 6 perturbation
  effects. A small local model can't be schema-constrained the way a hosted API can, so
  `_extract_json` digs the JSON out of prose/code-fenced output and records
  `unparseable_json` when there isn't any — that rate is a finding to report, not an error to
  suppress.
- **Program-of-thought is not optional, and not a trick.** The model returns the arithmetic in
  `answer_expression`; `src/generation/calculator.py` executes it. Asking a small model to do
  financial arithmetic mentally scored 0.04 on FinQA — 24% of answers copied a number with no
  arithmetic attempted, 19% were unparseable because the model wrote out correct working like
  `"$135.02 - $148.92 = -$13.90"`. FinQA ships a gold `program` field precisely because its
  authors expected systems to emit programs. **Never** replace the executor with "just ask the
  model to compute it", and never let `evaluate()` become an `eval()` — it walks an AST with an
  operator whitelist because the expression is untrusted model output.
- **Few-shot exemplars go in the system prompt, never as chat turns.** As user/assistant turns
  they are structurally identical to the real question, and the model answered a live question
  with `-30584 / 8920 * 100` where `8920` existed only inside an exemplar. Exactly one message
  may carry evidence. If you add an exemplar, re-check this.
- **Only execute `answer_expression` when it computes something** (`is_program`). It is often
  just the answer restated, and executing `"4.35%"` rewrites a correct answer as `0.0435`.
- **Prompt/model changes require the ablation, not a before/after pair.**
  `eval/baselines/run_prompt_ablation.py` runs four separable arms (direct/pot × 0/3-shot),
  and it has been run at **two** model sizes on purpose: an effect measured only at 1.5B is a
  claim about 1.5B. FinQA execution accuracy 0.04/0.06/0.08/0.12 at 1.5B and
  0.34/0.36/0.48/0.50 at 7B — same ordering, program-of-thought dominant at both. A single
  improved number can't distinguish which change did the work, and a reviewer will ask.
  Re-run the grid at the frozen model size whenever the prompt changes.
- **Heavy generation runs go to Colab's free T4** (`notebooks/run_b1_colab.ipynb`), not the
  laptop — an hour of sustained local inference overheats an M3 and the machine throttles.
  The repo lives in Drive so checkpoints survive Colab disconnects. `bitsandbytes`/`accelerate`
  are CUDA-only and live in `requirements-colab.txt`, deliberately outside the pinned local
  `requirements.txt`; the notebook records resolved versions to `eval/results/colab_env.json`,
  because a number from an unrecorded environment isn't reproducible.
- **RAGAS (Month 6):** **it defaults to an OpenAI judge LLM and will bill silently** — under
  the zero-budget constraint it must be constructed with an explicit local/free judge model
  before the first call, not after seeing a charge. **Judge *quality* is a second, separate
  risk, and it threatens Contribution 2 directly:** the claim is "we detect when RAGAS's
  faithfulness score fails to move as it should", which is unsupportable if the judge itself
  is too weak to score faithfulness — the failures then belong to the judge, not to the
  metric. A 1.5B judge is not adequate. Decide this before writing B2 code; the ₹0 options are
  a free-tier hosted judge, a 7B judge on Colab, or an NLI entailment model (e.g. DeBERTa-MNLI)
  which is local, free, fast and purpose-built for the entailment step. Then throttle
  concurrency
  (`RunConfig(max_workers=2, timeout=180, max_retries=5)`) to avoid 429 death spirals. Run
  each perturbed case 3x, report mean/variance, not a single score — treat high variance as a
  finding, not noise.
- **FinQA table linearization:** map header→value explicitly per row (don't flatten tables into
  one string — this silently kills numeric retrieval accuracy). Already implemented this way in
  `finqa_loader.py::_linearize_table_row` — follow the same pattern for TAT-QA, adjusted for its
  multi-row headers. Enforce a hard token limit using the actual embedding model's tokenizer,
  not word/char counts.
- **TAT-QA multi-row headers:** don't assume `table[0]` is a single clean header row the way
  FinQA's is — inspect ~20 real examples first (see `tatqa_loader.py` docstring).
- **Numeric answer scoring:** normalize numeric strings (strip %, commas, currency; fix precision)
  before exact-match scoring — "5.2%" vs "5.2" vs "0.052" are the same answer. TAT-QA also has a
  `scale` field (thousand/million/percent) that changes what a bare number means — don't drop it.
- **FinQA is scored on `exe_ans`, not `answer`.** `answer` is a rounded display string;
  `exe_ans` is the executed gold program and is what FinQA's official execution-accuracy
  metric uses. Scoring against the display string marks a model wrong for being *more* precise
  than the annotation (computed -6.8528 fails a 1% tolerance against "-7%") and cost 0.026 on
  FinQA. Two traps that came with it: `answer` is present-but-empty on 12/883 dev questions so
  `.get("answer", fallback)` does **not** fire (test the value, not the key); and the ×100
  percent relation must be gated on `Question.answer_is_percent`, never applied universally,
  or an answer wrong by 100× scores correct.
- **A metric change must never require re-running a model.** Generation is the expensive half
  and its outputs are checkpointed, so `run_b1_rag.py --rescore` recomputes verdicts from
  saved predictions — no GPU, no cost. Use it after touching anything in `eval/metrics/`, and
  re-score *every* affected results file (both datasets and all ablation arms) so no table is
  left mixing two scoring definitions.
- **RAGAS does not install correctly on its own.** `ragas==0.4.3` imports
  `langchain_community.chat_models.vertexai` at module scope but declares
  `langchain-community` with no upper bound, so a fresh resolve picks 0.4.x — where that
  module is gone — and `import ragas` fails outright. `requirements.txt` therefore pins
  `langchain-community==0.3.31`, the newest release that still ships it. Don't remove that
  pin; langchain-community is sunset upstream, so nothing will fix it for us.
- **Faithfulness is two stages with different inputs, and that is what makes Month 6
  affordable.** `decompose` reads only (question, answer) → statements, via the judge LLM;
  `verify` reads (statements, context) → score, via local HHEM. Contribution 2 perturbs the
  *context*, so decomposition is invariant across every perturbation of a question.
  `src/faithfulness/staged.py` computes it once, caches it to disk, and re-runs only the local
  verifier. Calling `ragas.evaluate()` inside the perturbation loop instead would be thousands
  of LLM calls recomputing a result that cannot change — the same anti-pattern the attribution
  guardrail forbids. Cache entries store the answer they came from; `cache_is_stale()` catches
  the silent-corruption case where B1 is re-run and answers change.
- **Three model families, kept deliberately separate.** Generator = Qwen2.5-7B; grounding NLI =
  DeBERTa-MNLI; RAGAS verification = Vectara HHEM. The generator must not grade its own
  answers, and grounding must not share a model with RAGAS — Contribution 2 perturbs the
  evidence *grounding* says matters and checks whether *RAGAS* reacts, which is circular if
  both run the same weights. Only statement decomposition sits on the generator's model,
  because it makes no correctness judgement.
- **Numeric answers are grounded on their operands, not their value.** Measured, not assumed:
  NLI scores ~0.09 entailment on a correct FinQA answer, because "the rental of 2009 is 19"
  does not entail "the change was -42.4" — that step is arithmetic, which no NLI model
  performs. Program-of-thought already names the figures consumed, so grounding checks their
  provenance in the evidence (deterministic, no model, and it yields the exact
  `(chunk_id, sentence_index)` Month 6 needs to perturb). NLI is kept for genuine text spans,
  which is what TAT-QA's 93 span questions need. Don't "fix" the low NLI numbers by lowering
  the threshold — that would be tuning the instrument to hide that it is the wrong one.
- **A Colab run must prove it did the work.** Cost three hours once: cells were added to
  `notebooks/run_b1_colab.ipynb` and pushed, but Colab serves the notebook copy the browser
  cached at open time. "Runtime -> Run all" then executed the *old* cell list, every cell
  succeeded, and nothing new was written -- indistinguishable from success. Two guards now
  make that impossible and must not be removed: `NOTEBOOK_VERSION` is stamped in the first
  code cell and compared against the repo copy right after `git reset --hard` (mismatch =
  hard stop with the revert instruction), and the closing **artifact manifest** cell lists
  every expected output file and names the cell to re-run for anything absent. **Bump
  `NOTEBOOK_VERSION` whenever a cell is added or changed**, and tell the user to do
  *File -> Revert to saved version* before Run all -- every time, not just the first.
- **Never let an expensive run report success without an artifact.** Same principle as the
  manifest, applied everywhere: a script that writes nothing should exit non-zero or say so
  loudly. Hours of free-tier GPU are cheap to spend and impossible to get back.
- **Checkpoint everything.** Free-tier Colab/session disconnects mid-run are expected for the
  Month 6 evaluation loop — write results to disk after every N questions, resume from last
  checkpoint, never re-run a full loop from scratch after an interruption.
- **Dependency versions are pinned** in `requirements.txt` (done 2026-08-11, from the
  environment that produced the Month 3/4 results). Keep them pinned — these libraries change
  APIs often enough to break reproducibility by Month 7. If you bump one, re-run the affected
  eval and update the numbers in README.md in the same change; a pinned file plus stale
  numbers is worse than neither.

## Working conventions

- Two separate loader modules for FinQA and TAT-QA (`finqa_loader.py`, `tatqa_loader.py`) — don't
  force one shared parser, the schemas differ enough that it's not worth it. They share only
  `schema.py`'s output dataclasses.
- Attribution/perturbation evaluation runs on a fixed subsample (200-300 questions per dataset,
  `configs/config.yaml: evaluation.subsample_size`), not the full test set — this is a
  scope/sanity cap, not (if architected per the guardrails above) a compute-survival requirement.
- Separate retrieval/grounding failures from generation/arithmetic failures when logging
  faithfulness results — a right-evidence-wrong-math case is not a grounding failure.
