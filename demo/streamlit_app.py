"""
AuditRAG -- the demo.

    streamlit run demo/streamlit_app.py

What is live and what is replayed, stated up front because the distinction is the point:

  * **Retrieval, attribution and the NLI verifier run live** on the laptop. They are
    embedding-only or a small cross-encoder, so a Shapley explanation or a re-verification
    costs about a second. Type any question against a document, or remove evidence sentences
    in the audit playground, and they recompute.
  * **Generation and the LLM verifiers are replayed** from the checkpoints in
    `eval/results/checkpoints/`. Generation is Qwen2.5-7B in 4-bit on Colab's T4 -- it does
    not fit on the laptop and the project has no budget for an API. Faking a live 7B answer
    with a smaller model would show numbers that are not the paper's numbers, so the demo
    does not.

Every number on screen traces to a file a reviewer can open; the pages name their sources.
Pages live in `demo/views/`, shared loaders and the theme in `demo/common.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "views"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import APP_NAME, inject_css  # noqa: E402

st.set_page_config(page_title=APP_NAME, page_icon="🧾", layout="wide")
inject_css()

import ask  # noqa: E402
import dashboard  # noqa: E402
import failures  # noqa: E402
import playground  # noqa: E402
import repro  # noqa: E402
import story  # noqa: E402

pages = [
    st.Page(story.render, title="AuditRAG", icon="🧾", default=True),
    st.Page(ask.render, title="Ask a question", icon="❓", url_path="ask"),
    st.Page(playground.render, title="Audit playground", icon="🧪", url_path="audit"),
    st.Page(dashboard.render, title="Results", icon="📊", url_path="results"),
    st.Page(failures.render, title="Failure cases", icon="🔍", url_path="failures"),
    st.Page(repro.render, title="Reproducibility", icon="🔁", url_path="repro"),
]
st.navigation(pages).run()
