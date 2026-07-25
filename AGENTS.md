# AGENTS.md instructions for burn_pc

## Role In The Stack

- `burn_pc` provides backend-generic predictive-coding tensor and inference
  primitives for Burn.
- Keep the crate independent of Dragon models, datasets, P2P transport,
  application orchestration, and deployment concerns.
- Downstream crates choose the latent state, local energy, inference schedule,
  and integration with gradient-based or forward-only optimization.

## Local Tooling

- Use the rustup stable Cargo and rustc binaries directly.
- Set `RUSTC` to the matching rustup rustc when invoking Cargo.
- Avoid `/snap/bin/cargo`.

## Design Principles

- Keep inference updates on the selected Burn backend. Host synchronization is
  allowed only in explicitly named diagnostics helpers.
- Make numerical assumptions and synchronization points visible in public APIs.
- Prefer small composable functions and typed metrics over owning a training
  loop or model lifecycle.
- Preserve deterministic backend-independent tests for update direction,
  clipping, decay, and invalid configuration.

## Validation

- Run `cargo fmt --all`.
- Run `cargo test --all-targets`.
- Run `cargo clippy --all-targets -- -D warnings`.
