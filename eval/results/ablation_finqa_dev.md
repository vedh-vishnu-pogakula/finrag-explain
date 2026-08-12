### B1 generation ablation -- finqa/dev, n=50

Retriever, index, top-k and question set are identical across arms; only the
generation configuration varies. `Program rate` is the share of answers produced
by executing a model-written expression rather than by the model stating a value.

| Arm | Model | Numeric acc. | Span F1 | Program rate | Unparseable | Declined | Citation prec. | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| direct-0shot | Qwen2.5-1.5B-Instruct | 0.0408 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5102 | 0.3 |
| direct-3shot | Qwen2.5-1.5B-Instruct | 0.0612 | 0.0 | 0.0 | 0.02 | 0.0 | 0.4796 | 0.1 |
| pot-0shot | Qwen2.5-1.5B-Instruct | 0.0612 | 0.0 | 0.78 | 0.0 | 0.0 | 0.52 | 0.1 |
| pot-3shot | Qwen2.5-1.5B-Instruct | 0.102 | 0.0 | 0.82 | 0.0 | 0.0 | 0.5 | 0.1 |
