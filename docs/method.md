# Method

## Pipeline

1. **Substrate.** SAGE (`SAGE_FTe992.ckpt`, 104.6 M params) encodes a 44.1 kHz stereo track into a
   16-channel latent at 86.1 frames/s. The encoder is frozen for the whole study; only the decoder
   (52.32 M params, 10 Swin-V2 blocks) is edited.
2. **Track.** *Stumblin' In* (238.9 s). Latent: 16 × 20 580 frames = 322 KiB in int8 (13.8 kbps).
3. **Harness.** A Python kernel exposes the decoder's tensors, a structural API and budgeted forward
   passes. The model (GPT-6 Astra, `codex exec`, reasoning effort `xhigh`) gets one turn per step and
   must answer with a single Python block, which the harness executes in a restricted namespace.
4. **Objective.** Maximise decode speed (measured in GFLOP per second of audio and in wall-clock
   decode time) subject to a quality constraint measured as log-mel distance on three 10 s windows.

## Kernel API (round 1: structure)

```
list_blocks()                  -> [{stage, block, dim, params_m, removed}]
remove_block(stage, block)     structural edit, ~0 bytes to store
restore_blocks()
list_params() / get / set      tensor access (explicit values cost 2 B each)
score()                        mel_db, gflop_per_s, params_m, speedup_vs_base, used
```

## Kernel API (round 2: rule patches)

```
scale_tensor(name, coeff)          W *= coeff                      4 B per coefficient
scale_singular(name, coeffs, r)    W += sum_k (c_k-1) sigma_k u_k v_k^T   4 B each
svd(name, rank)                    top singular values (the model may inspect them)
```

The *rule* form matters: a coefficient that scales a tensor or a singular direction is stored as a
float, while the direction itself is recomputed from the base weights at load time. This keeps the
shipped delta at hundreds of bytes instead of megabytes.

## Metric

`log-mel distance = mean |log10(mel(ref)) - log10(mel(est))|` over 128 mel bands (2048 FFT,
512 hop), computed on the full track and on three fixed windows.

**Relative floor.** The mel power is floored at `1e-4 · max_power` per signal. An absolute floor
(`+1e-8`) makes near-silent bins dominate: identical audio differing only by int16 quantisation
scored 0.2866 vs 0.3665. With the relative floor the same audio scores a stable 0.1374.

## Budgets

| run | turns | evaluations | patch budget |
|---|---:|---:|---|
| structural pruning | 10 | 30 | none (structure only) |
| rule patch | 12 | 40 | 300 kB (used 0.6 kB) |
| calibration (injected defect) | 6 | 30 | 20 000 scalars |

## Reproducing the numbers

`scripts/recompute.py` loads the codec once, applies the plan and the deltas, and writes
`data/metrics.json`. `scripts/make_figures.py` renders the two curves. The architecture diagram is
`scripts/draw_architecture.py` (written by Codex, then run locally).
