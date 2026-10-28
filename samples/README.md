# Sample videos

Self-recorded experiment videos used for development, the demo and the accuracy benchmark.
All videos in this folder are recorded by the author and licensed **CC BY 4.0**.

Naming: `freefall_<NN>.mp4`, `pendulum_<LLL>[_<amp>].mp4` (L in cm), `trap_<what>.mp4`.
Keep each file under 20 MB (trim and use 960 px). Use Git LFS if the total gets large.

Ground truth goes in `ground_truth.json` (schema: `docs/plan/03-data-contracts.md` §9;
example: `ground_truth.example.json`). Recording guide: `docs/recording-guide.md`.
