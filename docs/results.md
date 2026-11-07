# Results (numbers)

![Where the error lives: log-mel error of the pruned decoder over the whole track](figures/error_map.png)

The error is diffuse: 0.2–0.3 dB in every mel band across the whole track, with a mild excess
above 19 kHz and an edge artefact in the first second. There is no single broken component to
switch off, which is why pure weight editing does not improve it.

All values from `data/metrics.json` (single evaluation session, fixed metric).

## Structural pruning

| variant | params (M) | GFLOP/s | GPU decode (s) | log-mel |
|---|---:|---:|---:|---:|
| full decoder | 52.32 | 28.83 | 2.62 | 0.1374 |
| −1 block (stage 0) | 38.65 | 25.67 | 1.59 | 0.2180 |
| −2 blocks (stage 0) | 24.99 | 22.51 | 1.51 | 1.1682 |
| −2 + 2 (stage 1) | 18.14 | 17.63 | 1.29 | 1.0647 |
| −2 + 4 (stage 1) | 11.29 | 12.75 | 1.14 | 0.6219 |
| −2 + 6 (stage 1) | 4.44 | 7.86 | 0.96 | 0.6851 |
| −2 + 6 + 1 (stage 2) | 3.58 | 5.33 | 0.74 | 1.0530 |
| **Astra plan (5 blocks)** | **27.52** | **15.82** | **1.10** | **0.2072** |

Astra's plan is deliberately not the smallest: it keeps the two 13.7 M blocks of stage 0 that
dominate the compute, and removes blocks from stage 1/2 where damage per FLOP is cheapest.

## Delta frontier (pruned decoder + per-track patch)

| patch | delta | log-mel |
|---|---:|---:|
| none | 0 | 0.2072 |
| Astra rule patch (14 gains + 4 singular) | 0.6 kB | 0.2063 |
| low-rank-1 | 118.8 kB | 0.1696 |
| low-rank-2 | 232.0 kB | 0.1689 |
| low-rank-4 | 458.8 kB | 0.1525 |
| full fine-tune delta | 53.7 MB | 0.1465 |
| reference: full decoder | 0 | 0.1374 |

## Calibration (injected defect, pre-fix metric)

| state | log-mel |
|---|---:|
| clean | 0.2589 |
| postnet ×2.5 | 0.2831 |
| after Astra repair | 0.2596 |

The injected defect was created by scaling `postnet.net.0.weight` and `postnet.net.2.weight` by
2.5. Astra's first hypothesis was "excessive gain in the first postnet convolution", its second
"reversed polarity of the final postnet weight"; the saved patch restored 97 % of the damage.
This table predates the metric fix and is being re-measured.

## Timing

| | full | pruned |
|---|---:|---:|
| GPU decode, 239 s track | 2.62 s | 1.10–1.59 s |
| CPU decode (1 thread) | 56.7 s | 44.4 s |
| RTF on CPU | 4.2× | 5.4× |
