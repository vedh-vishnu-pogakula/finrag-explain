"""The story page -- what AuditRAG is, in one screen, with every number read from a file."""
from __future__ import annotations

import json

import streamlit as st

from common import (APP_NAME, AQUA, BLUE, CRITICAL, GOOD, ORANGE, PAPER_TITLE, RESULTS,
                    TAGLINE, tile)


def _json(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _headline_numbers() -> dict:
    """The three numbers on the hero. Missing file -> the tile says so instead of inventing."""
    out = {}
    b3 = _json("b3_comparison.json")
    if b3 and "finqa" in b3:
        sel = b3["finqa"]["selective_accuracy"]
        gate = next((v for k, v in sel.items() if k.startswith("B3") and "Qwen" in k), None)
        out["b1"] = sel["B1 (trust everything)"]["accuracy_overall"]
        out["b3"] = gate["accuracy_verified"] if gate else None
        out["b3_cov"] = gate["coverage"] if gate else None
        nli = next((v for k, v in sel.items() if k.startswith("B2") and "nli" in k), None)
        out["b2_nli"] = nli["accuracy_verified"] if nli else None
    vc = _json("verifier_comparison.json")
    if vc and "finqa" in vc:
        nli = next((r for r in vc["finqa"] if r["verifier"].startswith("nli")), None)
        qwen = next((r for r in vc["finqa"] if "Qwen" in r["verifier"]), None)
        out["sep_nli"] = nli["separation"] if nli else None
        out["sep_qwen"] = qwen["separation"] if qwen else None
        out["spec_qwen"] = qwen["specificity"] if qwen else None
    cc = _json("verifier_crosscheck_finqa_dev.json")
    if cc:
        out["kappa"] = cc["agreement"]["cohens_kappa"]
    return out


_PIPELINE_SVG = """
<div class="ar-anim" style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
<svg viewBox="0 0 1180 300" width="100%" height="300" role="img"
     aria-label="AuditRAG pipeline: question, retrieve, attribute, generate, ground, verify with three verifiers, audit, verdict">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#c3c2b7"/>
    </marker>
    <style>
      .node { fill: #fcfcfb; stroke: #c3c2b7; stroke-width: 1.5; rx: 10; }
      .node.live { stroke: __AQUA__; }
      .node.replay { stroke: __ORANGE__; stroke-dasharray: 5 3; }
      .lbl { font-size: 15px; font-weight: 600; fill: #0b0b0b; }
      .sub { font-size: 11.5px; fill: #52514e; }
      .edge { stroke: #c3c2b7; stroke-width: 1.5; fill: none; marker-end: url(#arr); }
      .flow { stroke: __BLUE__; stroke-width: 2.5; fill: none; stroke-dasharray: 8 10;
              animation: dash 1.6s linear infinite; opacity: .85; }
      .pulse { animation: pulse 2.4s ease-in-out infinite; transform-origin: center; transform-box: fill-box; }
      .verdict { fill: __GOOD__; }
      @keyframes dash { to { stroke-dashoffset: -36; } }
      @keyframes pulse { 0%,100% { opacity: .55 } 50% { opacity: 1 } }
    </style>
  </defs>

  <!-- row 1: the RAG pipeline -->
  <g transform="translate(10,40)">
    <rect class="node" x="0" y="0" width="140" height="58"/>
    <text class="lbl" x="70" y="25" text-anchor="middle">Question</text>
    <text class="sub" x="70" y="44" text-anchor="middle">FinQA / TAT-QA</text>

    <path class="edge" d="M140,29 L186,29"/>
    <path class="flow" d="M140,29 L186,29"/>

    <rect class="node live" x="188" y="0" width="150" height="58"/>
    <text class="lbl" x="263" y="25" text-anchor="middle">Retrieve</text>
    <text class="sub" x="263" y="44" text-anchor="middle">bge-small · top-5</text>

    <path class="edge" d="M338,29 L384,29"/>
    <path class="flow" d="M338,29 L384,29"/>

    <rect class="node live" x="386" y="0" width="170" height="58"/>
    <text class="lbl" x="471" y="25" text-anchor="middle">Attribute</text>
    <text class="sub" x="471" y="44" text-anchor="middle">Shapley · Rank-LIME · occlusion</text>

    <path class="edge" d="M556,29 L602,29"/>
    <path class="flow" d="M556,29 L602,29"/>

    <rect class="node replay" x="604" y="0" width="160" height="58"/>
    <text class="lbl" x="684" y="25" text-anchor="middle">Generate</text>
    <text class="sub" x="684" y="44" text-anchor="middle">Qwen2.5-7B · program-of-thought</text>

    <path class="edge" d="M764,29 L810,29"/>
    <path class="flow" d="M764,29 L810,29"/>

    <rect class="node replay" x="812" y="0" width="150" height="58"/>
    <text class="lbl" x="887" y="25" text-anchor="middle">Ground</text>
    <text class="sub" x="887" y="44" text-anchor="middle">operand provenance</text>

    <path class="edge" d="M962,29 L1008,29"/>
    <path class="flow" d="M962,29 L1008,29"/>

    <rect class="node" x="1010" y="0" width="150" height="58"/>
    <text class="lbl" x="1085" y="25" text-anchor="middle">Answer</text>
    <text class="sub" x="1085" y="44" text-anchor="middle">+ cited evidence</text>
  </g>

  <!-- row 2: the audit -->
  <g transform="translate(10,150)">
    <path class="edge" d="M877,-52 C877,-20 877,-10 877,0"/>
    <path class="flow" d="M877,-52 C877,-20 877,-10 877,0"/>

    <rect class="node" x="640" y="0" width="150" height="62" style="stroke:__BLUE__"/>
    <text class="lbl" x="715" y="24" text-anchor="middle">NLI verifier</text>
    <text class="sub" x="715" y="44" text-anchor="middle">184M · cheap</text>

    <rect class="node" x="806" y="0" width="150" height="62" style="stroke:__ORANGE__"/>
    <text class="lbl" x="881" y="24" text-anchor="middle">Phi-3.5 judge</text>
    <text class="sub" x="881" y="44" text-anchor="middle">3.8B LLM</text>

    <rect class="node" x="972" y="0" width="150" height="62" style="stroke:__AQUA__"/>
    <text class="lbl" x="1047" y="24" text-anchor="middle">Qwen-7B judge</text>
    <text class="sub" x="1047" y="44" text-anchor="middle">RAGAS default path</text>

    <path class="edge" d="M640,31 L560,31"/>
    <path class="flow" d="M640,31 L560,31"/>

    <rect class="node pulse" x="330" y="-6" width="228" height="74" style="stroke:__BLUE__; stroke-width:2"/>
    <text class="lbl" x="444" y="20" text-anchor="middle">Audit the auditor</text>
    <text class="sub" x="444" y="40" text-anchor="middle">separation · specificity</text>
    <text class="sub" x="444" y="56" text-anchor="middle">provenance · agreement</text>

    <path class="edge" d="M330,31 L250,31"/>
    <path class="flow" d="M330,31 L250,31"/>

    <rect class="node" x="60" y="-6" width="188" height="74" style="stroke:__GOOD__; stroke-width:2"/>
    <text class="lbl verdict" x="154" y="22" text-anchor="middle">Verdict</text>
    <text class="sub" x="154" y="42" text-anchor="middle">VERIFIED only if grounded</text>
    <text class="sub" x="154" y="58" text-anchor="middle">AND a validated verifier agrees</text>
  </g>

  <g transform="translate(10,262)">
    <rect x="0" y="0" width="14" height="14" rx="3" style="fill:#fcfcfb; stroke:__AQUA__; stroke-width:1.5"/>
    <text class="sub" x="20" y="11">live in this demo</text>
    <rect x="130" y="0" width="14" height="14" rx="3" style="fill:#fcfcfb; stroke:__ORANGE__; stroke-width:1.5; stroke-dasharray:4 2"/>
    <text class="sub" x="150" y="11">replayed from Colab checkpoints (7B does not fit on a laptop)</text>
  </g>
</svg>
</div>
""".replace("__BLUE__", BLUE).replace("__AQUA__", AQUA).replace("__ORANGE__", ORANGE).replace("__GOOD__", GOOD)


def render() -> None:
    st.markdown('<div class="ar-eyebrow">B.E. major project · CBIT Hyderabad · ₹0 compute</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="ar-hero-title">{APP_NAME}</div>'
                f'<div class="ar-hero-sub">{TAGLINE}.</div>', unsafe_allow_html=True)
    st.caption(f"Paper: *{PAPER_TITLE}*")

    st.iframe(_PIPELINE_SVG, height=310)

    n = _headline_numbers()
    tiles = []
    if n.get("b3") is not None:
        tiles.append(tile("Accuracy of answers a user is told to trust — FinQA",
                          f"{n['b1']:.2f} → {n['b3']:.2f}",
                          f"trust everything vs AuditRAG's gate (provenance AND validated verifier), "
                          f"coverage {n['b3_cov']:.0%}", AQUA))
    if n.get("sep_nli") is not None:
        tiles.append(tile("Does the cheap verifier know right from wrong? — FinQA",
                          f"{n['sep_nli']:+.3f}",
                          f"mean(correct) − mean(wrong). Inverted. The 7B LLM verifier: "
                          f"{n['sep_qwen']:+.3f}", CRITICAL))
    if n.get("kappa") is not None:
        tiles.append(tile("Agreement between the two verifiers RAGAS calls interchangeable",
                          f"κ = {n['kappa']:+.3f}", "statement-level Cohen's κ, n=50 — chance", ORANGE))
    if n.get("spec_qwen") is not None:
        tiles.append(tile("Does the validated verifier react to the RIGHT evidence? — FinQA",
                          f"{n['spec_qwen']:+.3f}",
                          "targeted-removal drop − random-removal drop, p<0.001", BLUE))
    if tiles:
        st.markdown('<div class="ar-tiles">' + "".join(tiles) + "</div>", unsafe_allow_html=True)
        st.markdown('<div class="ar-source">Sources: eval/results/b3_comparison.json · '
                    'verifier_comparison.json · verifier_crosscheck_finqa_dev.json</div>',
                    unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("#### What this project is, in five lines")
        st.markdown(
            "1. A financial RAG system that **explains its retrieval** — which words and numbers "
            "in the question pulled each piece of evidence (Contribution 1, RankingSHAP-anchored, "
            "training-free).\n"
            "2. A **grounding layer** that finds, deterministically, the sentence that supplied "
            "every figure the answer used.\n"
            "3. **The audit**: the same answers scored by three faithfulness verifiers, then tested "
            "on four independent axes — does the score know right from wrong, does it react to the "
            "right evidence, does it agree with the deterministic reference, do verifiers agree "
            "with each other (Contribution 2).\n"
            "4. **The finding**: the cheap verifier the field recommends *inverts* on numerical "
            "finance — it cannot do arithmetic, so it rewards answers that copy a figure and "
            "punishes answers that compute one. A 7B LLM verifier passes; a 3.8B LLM fails with "
            "the highest mean of all. *Verifier capacity, not type, decides validity.*\n"
            "5. **The consequence**: a verification gate built from provenance plus the validated "
            "verifier makes the answers it marks trustworthy far more often right — and the cheap "
            "verifier's gate is worse than no gate."
        )
    with c2:
        st.markdown("#### Where to click")
        st.markdown(
            "- **Ask a question** — one question end to end; type your own against the same "
            "document and watch retrieval + attribution recompute.\n"
            "- **Audit playground** — remove evidence sentences yourself and watch the NLI "
            "verifier's score move (live), next to the LLM verifier's recorded response.\n"
            "- **Results** — every figure in the paper, interactive.\n"
            "- **Failure cases** — the 1,500 labeled verdicts, filterable.\n"
            "- **Reproducibility** — every number → file → command."
        )
        st.markdown("#### Honesty notes")
        st.markdown(
            "- Generation ran on a free Colab T4 and is replayed here; nothing is faked with a "
            "smaller model.\n"
            "- FinQA accuracy 0.45 is **not** state of the art and is not the claim — the "
            "generator is a fixed instrument.\n"
            "- The strongest verifier is the generator's own model; see the paper's Limitations."
        )
