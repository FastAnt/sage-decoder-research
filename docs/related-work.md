# Related work

* **Weight-Space Programming** — R. Takehi, J. Teoh, A. Zhang, O. Khattab, 2026.
  <https://rikiyat.github.io/blog/weight-space-programming/> ·
  logs <https://github.com/RikiyaT/lm_understand-logs>.
  The protocol we followed: one `codex exec` turn per step, tools disabled, a Python kernel over
  the weights, a bounded budget of forward passes, edits made from weight statistics rather than
  gradients. Their headline results (identifying a game from weights, repairing a PPO policy,
  removing a looping direction from Qwen3.5-0.8B) are the regime where weight editing works.
  Our calibration reproduces the "repair a broken component" case on an audio codec.
* **SAGE** — a compact semantic-acoustic VAE for 44.1 kHz stereo music (104.6 M params, 16-channel
  latent at 86 Hz). <https://github.com/francescobrigante/SAGE>
* **SAME / Stable Audio 3 autoencoder** — the sibling codec, 10.8 Hz latent; we measured it first
  and moved to SAGE because of the higher latent rate.
* **RAVE** — real-time audio VAEs; our first phase used a raspberry-style decoder per stem.
* **Demucs** — source separation, used to split the first track into stems.
