# Explainable Financial RAG

Retrieval attribution and faithfulness-verified explanations for financial question answering.
B.E. CS/AI-ML final-year major project, CBIT Hyderabad.

See `CLAUDE.md` for the full project context (contributions, base paper, datasets, guardrails) —
Claude Code reads it automatically at the start of every session in this repo.

## Status

- [x] Month 1 — Literature review, scope locked
- [x] Month 2 — Data pipeline
  - [x] FinQA loader (`src/ingestion/finqa_loader.py`) — tested, produces `chunks.jsonl` +
        `questions.jsonl` with gold evidence IDs matched to FinQA's own `gold_inds`
  - [x] TAT-QA loader (`src/ingestion/tatqa_loader.py`) — tested against the real dev set;
        handles multi-row headers and section-label rows (see module docstring for the
        header/data-row split heuristic and its known limitations)
- [x] Month 3 — Baseline RAG end-to-end (B1)
  - [x] Retrieval (`src/retrieval/`) — bge-small embeddings with tokenizer-enforced chunk
        limits, cached chunk-embedding matrix, exact per-document search plus FAISS flat-IP
        corpus search
  - [x] Generation (`src/generation/`) — evidence-bound prompt, structured answer + citations,
        program-of-thought with a sandboxed arithmetic executor (`calculator.py`)
  - [x] Generation-configuration ablation (`eval/baselines/run_prompt_ablation.py`) — four
        separable arms, replicated at two model sizes; FinQA 0.34 → 0.50 at 7B
  - [x] Metrics (`eval/metrics/`) — Precision/Recall/hit/MRR@k; FinQA execution accuracy
        against `exe_ans`, and numeric normalization that respects TAT-QA's `scale`
  - [x] Baseline runners (`eval/baselines/`) — checkpointed, resumable, fixed subsample
- [x] Month 4 — Retrieval attribution (Contribution 1)
  - [x] Query segmentation into attributable units (numbers / entities / terms / stopwords)
  - [x] Batched, cached perturbation engine — one embedding pass per distinct coalition
  - [x] Three estimators: occlusion, RankingSHAP-anchored Shapley, Rank-LIME-anchored surrogate
  - [x] Two value functions: per-chunk score, and rank-biased-overlap over the whole ordering
  - [x] Faithfulness evaluation (comprehensiveness / sufficiency vs a random baseline)
- [~] Month 5 — Evidence grounding + RAGAS integration (B2)
  - [x] Grounding (`src/grounding/`) — sentence/table-row segmentation, local DeBERTa-MNLI
        entailment, operand-provenance grounding for numeric answers
  - [x] RAGAS wired to a local judge (`src/faithfulness/ragas_local.py`) with a tested guard
        against its OpenAI default; two-stage caching (`staged.py`) so Month 6 perturbations
        cost no LLM calls
  - [ ] B2 runner + B2 numbers
- [ ] Month 6 — Faithfulness perturbation testing (Contribution 2)
- [ ] Month 7 — Full evaluation (B1/B2/B3) + Streamlit demo + human study
- [ ] Month 8 — Paper write-up, workshop submission

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm    # NER for attribution unit grouping
```

`requirements.txt` is pinned to the exact versions that produced the results below. The spaCy
model is optional — attribution falls back to a capitalization heuristic without it — but the
reported entity numbers assume it is installed.

## Get the data

```bash
bash scripts/download_data.sh      # downloads FinQA + TAT-QA into data/raw/ (gitignored)
```

Small 8-example fixtures for fast, offline testing already live in `data/sample/` (checked
into git) — you don't need the full download to run the tests below.

## Run the loaders

```bash
cd src/ingestion
python finqa_loader.py --input ../../data/raw/finqa/dev.json --split dev --out-dir ../../data/processed
python tatqa_loader.py --input ../../data/raw/tatqa/dev.json --split dev --out-dir ../../data/processed
```

Writes `data/processed/{finqa,tatqa}_dev_chunks.jsonl` and `_questions.jsonl`. Each question's
`gold_chunk_ids` line up with `chunk_id`s from the same `doc_id`:
- FinQA: verified exactly against the dataset's own `gold_inds` (chunk text for `table_3` in doc
  `V/2008/page_17.pdf-1` matches FinQA's gold string word-for-word).
- TAT-QA: paragraph evidence (`para_N`) comes directly from `rel_paragraphs`; table evidence has
  no gold row index in the source data, so `gold_chunk_ids` for table-sourced answers is a
  best-effort heuristic (row's linearized text contains the answer string verbatim) — see
  `_table_gold_ids` in `tatqa_loader.py`, and don't mistake it for a ground-truth label later.

## Run the B1 baseline

Retrieval only — ~40s per dataset including the first embedding pass:

```bash
python eval/baselines/run_b1_retrieval.py --dataset finqa --split dev
python eval/baselines/run_b1_retrieval.py --dataset tatqa --split dev
```

End to end (retrieval → generation → scoring). **This costs nothing** — generation runs on a
local open-weights model (see below). It checkpoints every 10 questions, so an interrupted run
resumes rather than restarting:

```bash
python eval/baselines/run_b1_rag.py --dataset finqa --split dev --dry-run          # prompt only
python eval/baselines/run_b1_rag.py --dataset finqa --split dev --generator stub   # instant, offline
```

`configs/config.yaml` holds the **frozen** generation configuration — `Qwen2.5-7B-Instruct`
in 4-bit — because that is what the reported numbers come from. `bitsandbytes` is CUDA-only,
so a laptop spot check overrides both explicitly rather than editing the frozen config:

```bash
python eval/baselines/run_b1_rag.py --dataset finqa --split dev --limit 25 \
    --no-4bit --model Qwen/Qwen2.5-1.5B-Instruct
```

The full 250-question passes belong on Colab's free T4, not on a laptop — see *Heavy runs*
below. A `--limit 25` spot check is the intended laptop workflow.

## Cost: this project runs on ₹0

No paid API sits anywhere in the default path, at any month:

| Component | What it uses | Cost |
|---|---|---|
| Embeddings | `bge-small-en-v1.5`, downloaded once, runs locally | free |
| Vector search | FAISS (CPU) | free |
| Generation | `Qwen2.5-7B-Instruct` (4-bit) via `transformers`, on Colab's free T4 | free |
| Arithmetic | executed in Python by `src/generation/calculator.py`, not by an LLM | free |
| Datasets | FinQA + TAT-QA from GitHub | free |
| Compute | Colab's free tier for generation; your laptop for everything else | free |

The first run downloads model weights (~5 GB, one time). `generation.provider: api` exists in
[configs/config.yaml](configs/config.yaml) as an opt-in path and prints a cost warning, but
nothing in the default configuration can reach it.

**Watch out in Month 6:** RAGAS defaults to an OpenAI judge model and will bill silently. It
must be constructed with an explicit local/free judge before its first call.

Summary reports land in `eval/results/` and **are versioned** — they are cited here and in the
paper. The bulky per-question records in `eval/results/checkpoints/` are gitignored; copy them
out of Drive when a failure analysis needs them.

### B1 retrieval results (dev, 250-question fixed subsample, bge-small-en-v1.5, top-k=5)

| Dataset | Recall@5 | Full-recall@5 | Hit@5 | MRR | Scored |
|---------|---------:|--------------:|------:|----:|-------:|
| FinQA   | 0.850    | 0.728         | 0.956 | 0.783 | 250/250 |
| TAT-QA  | 0.843    | 0.741         | 0.942 | 0.851 | 243/250 |

Recall — not Precision — is the headline: questions have 1–4 gold facts out of 30–60 chunks, so
P@5 is capped around 0.4 even for a perfect retriever. Read the TAT-QA row with the
`_table_gold_ids` heuristic caveat in mind, and note the ~0.5% of FinQA questions whose gold
evidence is the table header row (see `finqa_loader.py`'s docstring) and is unreachable by
construction.

### B1 generation ablation — why the prompt looks the way it does

The first version of B1 asked the model to read a table, choose the right figures, do the
arithmetic mentally, and report a bare number. It scored **0.04** on FinQA. The per-question
records said why: 24% of answers were a number copied out of the evidence with no arithmetic
attempted, and another 19% were *unparseable* because the model had written out its working —

```
"$135.02 - $148.92 = -$13.90"      # the correct computation, discarded by the metric
```

Two changes follow from that, and the ablation separates them. **Program-of-thought**: the
model returns the calculation in `answer_expression` and
[`src/generation/calculator.py`](src/generation/calculator.py) executes it exactly, so the one
thing a small model is reliably bad at is no longer its job. This is FinQA's own formulation —
the dataset ships a gold `program` field — not a workaround. **Few-shot exemplars**: three
hand-written examples, fixed across datasets and models.

FinQA dev, 50 questions, execution accuracy. Identical retriever and question set across
arms; the grid was run twice, at two model sizes, so the finding is about the prompt rather
than about one model:

| Arm | 1.5B | 7B |
|---|---:|---:|
| `direct-0shot` <sub>(the original v1 prompt)</sub> | 0.04 | 0.34 |
| `direct-3shot` <sub>(exemplars only)</sub> | 0.06 | 0.36 |
| `pot-0shot` <sub>(program-of-thought only)</sub> | 0.08 | 0.48 |
| **`pot-3shot`** <sub>(both)</sub> | **0.12** | **0.50** |

**The ordering replicates at both scales, and program-of-thought is the dominant term.** At
7B it is worth +0.14 on its own against few-shot's +0.02; at 1.5B the two are closer (+0.04
and +0.02) and combine super-additively (+0.08 against a +0.06 sum), because at that size the
model needs the exemplars before it can use the program channel at all. Reporting only the
combined number would leave a reviewer unable to tell which half did the work — hence four
arms at two scales rather than a before/after pair.

Reproduce with:

```bash
python eval/baselines/run_prompt_ablation.py --dataset finqa --limit 50 \
    --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --tag-prefix 7b
```

Two implementation notes that cost real debugging time and are worth not rediscovering:

- Exemplars live in the **system prompt**, not in user/assistant chat turns. As chat turns they
  are structurally identical to the real question, and the 1.5B model answered a live question
  with `-30584 / 8920 * 100` — where `8920` appears nowhere except inside an exemplar. Exactly
  one message may carry evidence.
- `answer_expression` is only executed when it actually *computes* something. It is often just
  the answer restated, and executing `"4.35%"` would rewrite a correct answer as `0.0435`.

### Scoring: FinQA execution accuracy, not the display string

FinQA ships two gold answers and they are not interchangeable. `answer` is a display string,
rounded for presentation; `exe_ans` is the executed value of the gold program. **FinQA's own
metric is execution accuracy against `exe_ans`**, and scoring against the display string marks
a model wrong for being more precise than the annotation:

```
pred = -6.8528    answer = "-7%"    exe_ans = -0.06853     ← scored wrong against "-7%"
pred = 10.745     answer = "11%"    exe_ans =  0.10745     ← scored wrong against "11%"
```

A 1% relative tolerance cannot absorb `6.36` against `6`. Switching to execution accuracy is
worth **+0.026** on FinQA and, more importantly, makes these numbers comparable to published
FinQA results. Two supporting fixes came with it:

- `answer` is present-but-empty on 12 of 883 dev questions (1.4%), and `.get("answer", fallback)`
  only fires on an *absent* key — so those were scored against `""`, unscoreable, and silently
  bucketed as span questions. That is what dragged FinQA's span F1 to 0.04 on a dataset that
  is essentially all numeric.
- FinQA stores percentages as fractions while the prompt asks for percent form, so a factor of
  100 is a unit convention. It is applied **only** to questions whose gold is a percentage
  (`%` in the display string, or a `divide` in the gold program when the string is empty).
  Applying it everywhere would score an answer wrong by 100× as correct.

Changing a metric must never require re-running a model, so `run_b1_rag.py --rescore`
recomputes verdicts from saved predictions — no GPU, no cost.

### B1 end-to-end results

`Qwen2.5-7B-Instruct` 4-bit, `pot-3shot`, 250 questions per dataset, Colab free T4:

| Dataset | Answer accuracy | Span EM | Span F1 | Recall@5 | Citation prec. | Parsed |
|---|---:|---:|---:|---:|---:|---:|
| FinQA <sub>(execution accuracy)</sub> | **0.450** | — | — | 0.850 | 0.787 | 250/250 |
| TAT-QA <sub>(numeric)</sub> | **0.541** | 0.366 | 0.745 | 0.843 | 0.768 | 250/250 |

Against the original B1 (`Qwen2.5-1.5B-Instruct`, direct prompt, display-string scoring):
FinQA **0.04 → 0.45**, TAT-QA numeric **0.204 → 0.541**, TAT-QA span F1 **0.521 → 0.745**.

For context, the FinQA paper reports general-crowd humans at 50.7% and their trained
FinQANet (RoBERTa-large) at 61.2% — both working from gold evidence. This system reads through
a retriever whose recall@5 is 0.85, so its ceiling is ~0.85 and it captures roughly half of
what is reachable.

**Citation precision is the number that matters most to this project**, and it moved 0.530 →
0.787. Both contributions rest on the model's evidence citations meaning something; near 0.53
they were close to noise, at 0.79 the Month 5 grounding layer has real signal to work with.

**Retrieval is not the bottleneck** — 4–5% of failures are retrieval failures. The
evidence-finding half works, which is what the two contributions depend on.

### What still fails

| Bucket | FinQA | TAT-QA |
|---|---:|---:|
| Correct | 45.2% | 47.6% |
| Right operands present, **combined wrongly** | 30.0% | 12.4% |
| **No expression emitted** (read a value instead of computing) | 10.0% | 22.8% |
| Declined despite having the evidence | 4.0% | 10.4% |
| Operand not in the evidence (hallucinated) | 6.8% | 1.6% |
| Retrieval miss | 4.0% | 5.2% |

**The arithmetic problem is solved.** Hallucinated operands are 7% on FinQA and 2% on TAT-QA,
and structured output parsed on 250/250. What remains on FinQA is *column selection* — the
model writes `(22 - 19) / 22` when the question named 2007 and 2009, because one linearized
table row carries every year (`...of 2009 is 19 ; ...of 2008 is 22 ; ...of 2007 is 33`).

**Abstention is new and worth reporting.** The 1.5B model declined on 0 questions; the 7B model
declines on 4% (FinQA) and 10.4% (TAT-QA), and on TAT-QA 27 of its 28 declines are on questions
whose gold evidence *was* retrieved. For financial QA, preferring "insufficient evidence" to a
fabricated figure is the right failure mode — but a 10% decline rate on answerable questions is
also the largest single pool of recoverable answers left in B1.

### Heavy runs go to Colab's free T4

Generation is the only part of this project that needs a real GPU, and a laptop running an
hour of sustained inference gets hot enough to throttle. [
`notebooks/run_b1_colab.ipynb`](notebooks/run_b1_colab.ipynb) runs the ablation and both
250-question passes on Colab's free tier at ₹0:

- the notebook clones from GitHub into Google Drive and pulls on every later run, so the Colab
  copy always matches the laptop and checkpoints survive the disconnects free Colab is prone to
- every run resumes from its last checkpoint — reconnect, re-run all cells, nothing recomputed
- `requirements-colab.txt` adds only `bitsandbytes` + `accelerate` (CUDA-only, which is why
  they are not in the pinned local `requirements.txt`), and the resolved versions are written
  to `eval/results/colab_env.json` so a Colab number is always traceable to its environment

## Run evidence grounding (Month 5)

Sentence-level answer↔evidence matching. **Runs on a laptop in seconds and costs nothing** —
generation is already checkpointed, so grounding is a post-process over the saved B1 records,
exactly like `--rescore`:

```bash
python eval/baselines/run_grounding.py --dataset finqa --split dev
python eval/baselines/run_grounding.py --dataset tatqa --split dev --backend stub  # offline
```

FinQA: 250 questions in **15.7s**. TAT-QA: **35.5s**. No GPU.

### Numeric answers are grounded on operands, not on the final value

This is the design decision worth defending, and it was forced by measurement rather than
chosen. Running NLI entailment on the answer *value* scores ~0.09 even when the evidence is
exactly right, because `the contingent rental of 2009 is 19` genuinely does not entail
`the change was -42.4` — getting between them is arithmetic, which no NLI model performs.
Reporting those as ungrounded would measure the instrument, not the system.

Program-of-thought already names the figures the answer consumed, so each operand becomes a
claim verified by **provenance**: is this number in the retrieved evidence, and in which
sentence. Deterministic, needs no model, and it yields the exact `(chunk_id, sentence_index)`
that Month 6 perturbs. NLI is kept for genuine text spans, where entailment *is* the right
instrument — which is what TAT-QA's 93 span questions need.

### Grounding results (250 questions per dataset, DeBERTa-MNLI + operand provenance)

| Dataset | Groundedness | Supported | Contradicted | Grounds to gold | Numeric claims |
|---|---:|---:|---:|---:|---:|
| FinQA | 0.757 | 0.745 | 0.088 | 68.8% | 93% |
| TAT-QA | 0.664 | 0.686 | 0.180 | 58.0% | 54% |

**The split by failure mode is the result**, because it separates the two things CLAUDE.md
asks to keep apart:

| | FinQA supported / grounds-to-gold | TAT-QA supported / grounds-to-gold |
|---|---:|---:|
| Answer correct | 0.875 / **0.876** | 0.836 / **0.781** |
| Generation failure | 0.730 / **0.624** | 0.677 / **0.531** |
| Retrieval failure | 0.200 / 0.000 | 0.444 / 0.000 |
| Declined | 0.000 / 0.000 | 0.115 / 0.038 |

Read the middle row. **Wrong answers still ground to the correct evidence 62% (FinQA) and 53%
(TAT-QA) of the time** — the system found and used the right facts and then did the arithmetic
wrong. That is a fundamentally different failure from not finding the evidence at all, which
the bottom row shows scoring 0.000 as it should. A single accuracy number cannot distinguish
those two; this table can.

Grounding also disagrees with the model's own citations (`grounded~cited` 0.663 / 0.596): a
model can cite `[1]` while its answer is actually supported only by `[3]`. That gap is a
finding to report in B3, not an error to reconcile.

**Contradiction counts only for claims that found no support.** Retrieved evidence holds many
facts, and an NLI model reads `revenue 2018 was 440.7` as contradicting a claim about 2019 —
so a raw maximum across sentences fires on ordinary multi-year tables. It reported 44% of
TAT-QA answers as contradicted, which said nothing about faithfulness; gated properly it is
0.180, and it means what the name says.

## Run attribution (Contribution 1)

Free, and never touches the generator — it only re-scores query variants against chunk
embeddings computed once at index build time.

```bash
python eval/baselines/run_attribution.py --dataset finqa --split dev --limit 100
python eval/baselines/run_attribution.py --dataset finqa --split dev --mode ranking
python eval/baselines/run_attribution.py --dataset finqa --split dev --embedder hash  # instant
```

Re-running without `--fresh` reloads the checkpoint and re-aggregates the report without
recomputing anything.

### Attribution results (dev, `bge-small-en-v1.5`, top-20% of units removed/kept)

Comprehensiveness ↑ (removing the top-weighted units should collapse the score), sufficiency ↓
(those units alone should recover it), both normalized by the query's full attributable range.
**Lift** is comprehensiveness minus the same measurement with randomly chosen units — the
number that actually shows the explanation is doing work.

| Run | Method | Comp ↑ | Random | Lift ↑ | Suff ↓ |
|---|---|---:|---:|---:|---:|
| FinQA, score | occlusion | 0.566 | 0.074 | 0.492 | 0.023 |
| FinQA, score | shapley | 0.578 | 0.074 | 0.504 | −0.018 |
| FinQA, score | surrogate | 0.580 | 0.074 | 0.506 | −0.012 |
| TAT-QA, score | occlusion | 0.593 | 0.118 | 0.474 | 0.102 |
| TAT-QA, score | shapley | 0.592 | 0.118 | 0.473 | 0.065 |
| TAT-QA, score | surrogate | 0.602 | 0.118 | 0.483 | 0.072 |
| FinQA, **ranking** | occlusion | 0.493 | 0.179 | 0.314 | 0.423 |
| FinQA, **ranking** | shapley | 0.630 | 0.179 | **0.451** | 0.314 |
| FinQA, **ranking** | surrogate | 0.567 | 0.179 | 0.387 | 0.333 |

Two findings worth writing up:

1. **Explaining the ranking is where the method choice matters.** On per-chunk scores all
   three methods are within 0.015 of each other. On the ranking-level value function, Shapley's
   lift is 44% higher than occlusion's (0.451 vs 0.314) — occlusion measures each unit only in
   the presence of all the others, so it misses the interactions that determine an ordering.
   This is the empirical case for the RankingSHAP anchoring rather than a leave-one-out
   shortcut.
2. **Shapley is much better at ignoring function words.** Share of attribution mass landing on
   stopwords: 3.0% (Shapley) vs 11.3% (occlusion) on FinQA, 2.3% vs 11.2% on TAT-QA. Content
   terms take 72–78%, numeric values 14–16%, named entities 3–8%.

A negative sufficiency is real, not a bug: keeping only the top-weighted units sometimes scores
*higher* than the full query, because the discarded words were diluting the embedding.

## Tests

```bash
pytest tests/ -v          # 148 tests, offline, ~16s
pytest tests/ -m "not slow"   # skips the one test that loads the real embedding model
```

Covers both loaders against checked-in fixtures, the retrieval layer (chunk-ID collision
isolation across documents, search matching the score matrix exactly, index round-trip, the
embedding-model mismatch guard), numeric answer normalization, and the generator's failure
paths (refusal, truncated JSON, out-of-range citations, API exceptions) via a fake client — no
network, no API key.

The calculator is tested as untrusted-input handling, not just as arithmetic: `evaluate()` is
asserted to reject `__import__`, attribute access, comprehensions, names, division by zero and
`9**9**9`. It walks an AST with an operator whitelist, so there is no `eval()` for model output
to reach. Two regression tests pin the bugs that cost the most to find — exemplar figures
leaking into real answers, and a percent literal being mistaken for a calculation.

## Next session

Months 3–4 are done and B1's generation configuration is now frozen: `Qwen2.5-7B-Instruct` in
4-bit, program-of-thought, three exemplars. **That configuration is an experimental constant** —
B2 and B3 must be produced under exactly the same one, or the three baselines cannot be
compared.

Month 5's grounding half is done and reported above. What remains is the B2 runner and its
numbers, and two things about it are already settled:

**The judge.** RAGAS's own factory signature is `llm_factory(model, provider="openai")` — the
billed provider is the default *in code*, and `evaluate()` without an explicit `llm=`
constructs it and charges on the first call. `src/faithfulness/ragas_local.py` supplies a
local judge and three tested guards. Judge *quality* is the separate risk: a faithfulness
metric scored by a weak judge cannot support a claim about when that metric fails, so
verification runs on Vectara HHEM — an independent NLI model — while only the mechanical
statement-decomposition step sits on the local LLM. Three model families stay separate on
purpose: generator (Qwen), grounding (DeBERTa-MNLI), RAGAS verification (HHEM). Grounding
must not share weights with RAGAS, or Contribution 2 would be testing a model against itself.

**The compute.** Faithfulness decomposes `(question, answer)` with the LLM and verifies
`(statements, context)` with HHEM. Contribution 2 perturbs the *context*, so decomposition is
invariant across perturbations — `src/faithfulness/staged.py` computes it once, caches it, and
re-runs only the local verifier. That is the difference between ~4,500 LLM calls (about a day
of free-tier GPU) and 500 one-time calls with a perturbation loop that costs nothing. Same
guardrail as attribution: never re-run the invariant half inside a perturbation loop.

Known rough edges, in the order they'd matter:

- `tatqa_loader.py::_merge_header_cells` can drop a wide title spanning multiple year columns
  ("Years Ended September 30," sometimes attaches to only one of three year columns). Values
  are still correct; only header context is thinned. Revisit if TAT-QA attribution looks off.
- TAT-QA table-evidence gold IDs remain a documented heuristic (now derivation-operand based,
  96.9% coverage of dev). Fine for retrieval eval; do not treat as ground truth for
  attribution correctness.
- `requirements.txt` is still unpinned. The environment is now confirmed working end to end,
  so pinning it with `pip freeze` is overdue per the Month 2 guardrail.
