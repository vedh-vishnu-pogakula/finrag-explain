### B1 generation ablation -- finqa/dev, n=50

Retriever, index, top-k and question set are identical across arms; only the
generation configuration varies. `Program rate` is the share of answers produced
by executing a model-written expression rather than by the model stating a value.

| Arm | Model | Numeric acc. | Span F1 | Program rate | Unparseable | Declined | Citation prec. | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| direct-0shot | Qwen2.5-7B-Instruct | 0.34 | n/a | 0.0 | 0.0 | 0.06 | 0.6939 | 0.1 |
| direct-3shot | Qwen2.5-7B-Instruct | 0.36 | n/a | 0.0 | 0.0 | 0.04 | 0.67 | 0.1 |
| pot-0shot | Qwen2.5-7B-Instruct | 0.48 | n/a | 0.86 | 0.0 | 0.06 | 0.76 | 0.1 |
| pot-3shot | Qwen2.5-7B-Instruct | 0.5 | n/a | 0.78 | 0.0 | 0.08 | 0.74 | 0.1 |
