# Course engineering rules

This repository is a Chinese zero-to-practice LLM systems curriculum.

- Explain prerequisites, intuition, rigorous mechanism, a hand-computable example, runnable code, expected results, debugging, graduated exercises, reference solutions, and a mastery gate.
- Never present a toy implementation as production-equivalent. State every simplification.
- Prefer standard-library and NumPy CPU labs. Optional PyTorch/CUDA/Triton labs must check dependencies, document exact prerequisites and bounded hardware use, and skip clearly when unavailable.
- All paths and commands run from repository root; prefer `python -m labs.<track>.<module>`.
- Every track owns its own books/, labs/, and test filename prefix; do not edit other tracks without coordination.
- Use `unittest`; keep offline tests free of API calls, GPU requirements, downloads, absolute paths, secrets, and nondeterministic timing assertions.
- Reference real primary sources; record retrieval dates and pinned commit IDs when verified. Never invent versions, URLs, measurements, or model architecture details.
- Core content and reference implementations must be complete. Deliberate exercises may contain TODO only in explicitly named starter files with solutions and acceptance tests.
- Protect company resources: GPU labs default to one device and bounded workloads; multi-GPU experiments are opt-in. No privileged containers, broad cleanup commands, unauthenticated public listeners, or automatic API spend.
- Public repository: never include personal memory, company endpoints, keys, private PDFs, or raw private data.
- A successful test is evidence only for its tested environment; distinguish local CPU verified, compile-only, optional GPU/API unexecuted.
- Keep durable progress in docs/BUILD_STATUS.md and update coverage and validation evidence at release.
