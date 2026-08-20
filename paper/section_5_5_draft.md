# §5.5 Model-Based Faithfulness Verification

*Draft for insertion after the current p14 (end of §5.4), before the current §5.5.
Renumber the existing §5.5 to §5.6 and update the roadmap on p3.*

*Citation numbers are placeholders — renumber into the review's existing scheme. Verify every
bibliographic detail against the original venue before submission; do not take them from here.*

---

Sections 5.2–5.4 converge on a single recommendation: replace RAGAS's zero-/few-shot LLM judge
with something cheaper and better calibrated. RAGBench [7] demonstrates a fine-tuned RoBERTa
outperforming LLM judges on RAG evaluation; ARES [8] reports its fine-tuned, calibrated judges
beating RAGAS by an average of ~59 points in context-relevance accuracy; FaithJudge [14] argues
that static LLM judging is a non-stationary problem requiring continual recalibration. The
implied prescription is consistent across all three, and the tooling has followed it: RAGAS
itself now ships `FaithfulnesswithHHEM`, a drop-in variant that replaces the LLM verification
step with a local classifier, presented to users as an efficiency choice rather than a
semantic one.

That substitution draws on an established line of work in *model-based factual-consistency
verification*, which this review has not yet covered and which the remainder of this section
addresses.

**Natural-language-inference verifiers.** The dominant formulation treats the retrieved context
as a premise and each generated claim as a hypothesis, scoring factual consistency as textual
entailment. FactCC [A] (Kryscinski et al., EMNLP 2020) trains a BERT-based classifier on
synthetically corrupted summaries and established the framing for abstractive summarization.
SummaC [B] (Laban et al., TACL 2022) revisits NLI models for inconsistency detection and shows
that applying entailment at *sentence* granularity rather than document granularity recovers
most of the performance earlier work had attributed to model capacity, yielding a lightweight
detector competitive with far larger systems. AlignScore [C] (Zha et al., ACL 2023) generalises
the objective into a single alignment function trained across 4.7M examples from seven tasks,
and reports state-of-the-art factual-consistency results at a fraction of LLM-judge cost.
QAFactEval [D] (Fabbri et al., NAACL 2022) takes a question-generation/question-answering route
to the same target, and notably finds QA-based and entailment-based metrics to be
complementary rather than redundant.

**Deployed verifiers.** Vectara's Hallucination Evaluation Model (HHEM) [E] packages this
approach as a production artifact — a compact cross-encoder returning a per-claim consistency
probability — and is the model RAGAS's `FaithfulnesswithHHEM` invokes. Its appeal is precisely
that it removes the LLM from the scoring loop: deterministic, inexpensive, and fast enough to
run inside an evaluation sweep.

**The shared evaluation basis, and its limit.** These verifiers are validated almost
exclusively on summarization and general-domain question answering — SummEval, FRANK, AggreFact,
XSumFaith and comparable corpora. In every one of those settings the claim under test is a
*restatement* of information present in the premise: the verifier's task is to decide whether
the summary sentence says what the source document already said. Entailment is the correct
formal relation for that task, and the benchmarks are constructed so that it is.

Numerically-derived answers break that assumption. When a financial QA system reports that a
liability "decreased by 42.4%", the retrieved evidence contains the two year-end figures and
nothing else; the answer stands in an *arithmetic* relation to the premise, not an entailment
relation. A verifier trained to recognise restatement has no mechanism for discharging that
inference, and the failure is asymmetric in a way that matters: an answer that copies a figure
verbatim out of the evidence is trivially entailed, while an answer that correctly computes a
figure not present in the evidence is not. Nothing in the literature surveyed above evaluates
these verifiers on answers of the second kind.

This is the gap the present project's second contribution occupies. Section 7 reports that on
FinQA — where the overwhelming majority of answers are computed rather than extracted — the
substitution the field recommends removes the metric's ability to discriminate correct answers
from incorrect ones altogether, while RAGAS's original LLM verifier retains it at p < 0.001.
The result does not contradict RAGBench or ARES on their own terms; it identifies a domain in
which their prescription does not transfer, and gives the mechanism for why.

---

## References to add

- [A] Kryscinski, McCann, Xiong & Socher. *Evaluating the Factual Consistency of Abstractive
  Text Summarization.* EMNLP 2020.
- [B] Laban, Schnabel, Bennett & Hearst. *SummaC: Re-Visiting NLI-based Models for
  Inconsistency Detection in Summarization.* TACL 2022.
- [C] Zha, Yang, Li & Hu. *AlignScore: Evaluating Factual Consistency with a Unified Alignment
  Function.* ACL 2023.
- [D] Fabbri, Wu, Liu & Xiong. *QAFactEval: Improved QA-Based Factual Consistency Evaluation
  for Summarization.* NAACL 2022.
- [E] Vectara. *HHEM: Hallucination Evaluation Model.* Model release (`vectara/
  hallucination_evaluation_model`); cite the model card and note the version evaluated.
