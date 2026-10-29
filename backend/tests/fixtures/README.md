Recorded API responses for golden tests (docs/plan/08 §3).

- `media/`: synthetic test video (60 fps, 5 s, true g = 9.81 m/s², 460 px = 1 m) with
  ground truth (`*.gt.json`) and its first key frame.
- `sam/*.sam.json.gz`: compact SAM 3.1 records captured from the live API in M1
  (`scripts/spikes/`), replayed offline by `tests/fakes.py::FakeSam`:
  60 fps 5 s, 240 fps 3 s, 2 s hold before release (60 fps), and the image-mode ruler/ball
  on key frame 0. M3 golden tests check g within 0.5 % of 9.81 on these.
- `media/pendulum_60fps_12s.*` and `sam/pendulum_*.sam.json.gz` (M4): synthetic pendulum
  (L = 0.60 m, θ0 = 15°, γ = 0.02 s⁻¹, exact nonlinear ODE, rendered by
  `scripts/spikes/make_test_media.py`). Its SAM records are **synthetic** (`"synthetic": true`,
  made by `tests/synth_sam.py` from ground truth with 1 px dilation), not live recordings;
  replace them with live ones once recorded.
