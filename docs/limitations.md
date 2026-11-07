# Limitations and threats to validity

1. **Quality is not solved.** The pruned decoder is audibly worse. A 122–238 kB per-track delta
   closes ~60–75 % of the measured gap; this has not been confirmed by a blind listening test.
2. **Metric proxy.** Log-mel ignores phase and transients. PESQ through torchaudio is broken on
   this stack (1.2 on identical signals), so no perceptual metric is in the loop yet.
3. **Metric bug history.** An absolute mel floor made earlier numbers unstable (±0.08 dB from
   int16 quantisation). The calibration table still uses the pre-fix metric.
4. **One track, one codec.** The pruning plan is track-specific: the same removed blocks cost
   +0.12 log-mel on a different song, i.e. it does not generalise.
5. **CPU speedup is small (1.28×).** Decode time is dominated by iSTFT/patch packing, not by the
   transformer blocks, so the mobile benefit must be measured on the device (LiteRT export not yet
   done).
6. **The LLM patch was optimised against the buggy metric** (0.6 kB, no measurable effect after the
   fix). It must be re-run with the corrected objective.
7. **Budget dependence.** 30–40 evaluations and 10–12 turns is small; a larger budget may find
   better plans, but the qualitative findings (block removal is cheap, quality is expensive) are
   unlikely to flip.
