# Model-Collapse-in-Tabular-Data-Generation

We investigate model collapse in tabular data generation, inspired by recent findings
of collapse in large language models trained on synthetic text. Using a highly imbal-
anced click-through-rate dataset (roughly 1.5% positives), we evaluate three generative
paradigms—CTGAN, CVAE, and the autoregressive LLM-based GReaT—under an iter-
ative retraining regime where each synthesizer is fine-tuned on a 50/50 mixture of real and
self-generated samples. Across all models, we observe consistent forms of collapse, including
overinflation of minority-class modes and systematic distortion of feature–feature correlation
structure.
CTGAN variants oversample the minority class and fail to preserve global relationships,
though CTGAN-on-CTGAN training can still improve downstream predictive performance.
CVAE V1 underperforms real-data baselines, while CVAE V2 improves downstream accuracy
but introduces spurious correlations between unrelated features. The LLM-based generator
hallucinates categorical and label values, producing the lowest fidelity among all methods
despite stable downstream performance across iterations.
Our results show that while synthetic-on-synthetic training can enhance downstream
models, it simultaneously degrades statistical fidelity, indicating that tabular generative
models are vulnerable to collapse effects similar to those seen in LLMs. This highlights the
need for collapse-aware evaluation protocols when using synthetic data for machine-learning
pipelines.
