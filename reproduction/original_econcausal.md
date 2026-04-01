# Original EconCausal Source

This artifact does not redistribute the original 10,490-row EconCausal corpus.

Use the official external release for the base benchmark:

- Paper landing page: <https://huggingface.co/papers/2510.07231>
- Benchmark repository referenced by that page: <https://github.com/econaikaist/econcausal-benchmark>

This artifact starts from either:

1. the original EconCausal release downloaded from the external source above, or
2. the derived paper data already included under `data_derived/`.

Use the original source when reproducing:

- full-corpus ideology classification
- full metadata normalization from original triplets
- any step that requires the non-redistributed 10,490-row benchmark file

Use the included derived data when reproducing:

- Task 1 ideology-subset evaluation
- official Task 2 reruns for this paper
- curated paper analyses based on the included subset files

