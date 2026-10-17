# LoRA Identity Isolation port

Goal: opt-in functional Fizgig mechanism port for Krea 2, preserving this fork's existing training paths; no real-data training or experiments.

- [x] Verify reference source/revision/license and record source-to-requirement map and deviations.
- [x] Audit existing DOP, captions/caches, bucket sampling, optimizer windows, successful-update clock, resume, and UI.
- [x] Add validated explicit component settings, boundary-safe subject transforms, and synthetic caption tests.
- [x] Port pinned-asset compositor with explicit preparation, content-addressed complete caches, and local fixture tests.
- [x] Integrate dataset mapping, approximate repeat balancing/exposure reports, and category-homogeneous updates without changing run steps.
- [x] Extend existing DOP with multi-trigger conditioning and successful-update-window cadence; reuse transient optimizer LR windows and checkpoint state.
- [x] Add UI controls and backend-powered previews after backend semantics settle.
- [x] Run focused tests/static checks and an integrated synthetic smoke; document actual evidence and GPU/model limitations.
- [x] Deliver usage/configuration, manual validation, experiment plan, and logical task-only commits.

Compatibility decisions: source code wins over prose; explicit isolation overrides initialize presets; Off never disables legacy DOP. No automatic stranger/model downloads. Exact composite LR scaling requires homogeneous updates, never loss weighting. Additional cache integrity and update-window semantics are documented extensions. Real-data effectiveness remains unproven.
