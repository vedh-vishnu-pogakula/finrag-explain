"""Retrieval package (Month 3 / baseline B1).

    from retriever import Retriever          # after putting src/retrieval on sys.path
    r = Retriever.from_config()
    hits = r.retrieve("what was the change in net revenue?", doc_id="V/2008/page_17.pdf-1")

Modules are imported by filename (matching the existing src/ingestion convention) rather than
as a package, so this file stays intentionally empty of imports -- pulling Embedder in here
would drag sentence-transformers into every `import` of anything in the package.
"""
