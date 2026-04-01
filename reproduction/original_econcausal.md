# Original EconCausal Source

This artifact does not redistribute the original 10,490-row EconCausal corpus.

Use the official external release for the base benchmark:

- Paper landing page: <https://arxiv.org/abs/2510.07231>
- Benchmark repository referenced by that page: <https://github.com/econaikaist/econcausal-benchmark/tree/main>

This artifact starts from either:

1. the original EconCausal release downloaded from the external source above, or
2. the derived paper data already included under `main_results/` and `icl_experiment/`.

Use the original source when reproducing:

- full-corpus ideology classification
- any step that requires the non-redistributed 10,490-row benchmark file

Use the included derived data when reproducing:

- main-results reruns on the ideology-sensitive subset
- official ICL-experiment reruns for this paper
- curated paper analyses based on the included subset files
