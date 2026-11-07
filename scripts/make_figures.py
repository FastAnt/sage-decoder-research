import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = json.loads((Path(__file__).resolve().parents[1] / "data/metrics.json").read_text())
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
C_BASE, C_PRUNE, C_DELTA = "#4C72B0", "#C44E52", "#55A868"

# 1. removal curve (scatter: the relation is NOT monotonic)
curve = DATA["removal_curve"]
base = DATA["base"]
pts = [(base["params_m"], base["mel_db"], "full decoder")] + \
      [(c["params_m"], c["mel_db"], c["tag"]) for c in curve]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=55, color=C_PRUNE, label="blocks removed, no repair")
for x, y, l in pts:
    ax.annotate(l, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=7)
ax.scatter([DATA["astra_plan"]["params_m"]], [DATA["astra_plan"]["mel_db"]], s=180, marker="*",
           color="#8172B2", zorder=5, label="Astra plan: 5 blocks removed")
ax.scatter([base["params_m"]], [base["mel_db"]], s=120, marker="s", color=C_BASE, zorder=5,
           label="full decoder")
ax.set_xlabel("decoder parameters (M)"); ax.set_ylabel("log-mel distance, dB (lower is better)")
ax.set_title("Structural pruning: size vs quality (one track)")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "pruning_curve.png", dpi=150); plt.close(fig)

# 2. delta frontier
fr = DATA["delta_frontier"]
xs = [0] + [f["delta_kib"] for f in fr]
ys = [DATA["astra_plan"]["mel_db"]] + [f["mel_db"] for f in fr]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, ys, "o-", color=C_DELTA)
ax.axhline(base["mel_db"], ls="--", color=C_BASE, label=f"full decoder {base['mel_db']:.3f}")
for x, y, f in zip(xs[1:], ys[1:], fr):
    ax.annotate(f["tag"].replace("_", " "), (x, y), textcoords="offset points", xytext=(6, -12), fontsize=7)
ax.scatter([0.6], [DATA["astra_patch"]["mel_db"]], marker="x", s=90, color="#8172B2",
           label="Astra 0.6 kB rule patch")
ax.set_xscale("log")
ax.set_xlabel("per-track weight delta, kB (log)"); ax.set_ylabel("log-mel distance, dB")
ax.set_title("How many bytes buy back the quality? (pruned decoder + low-rank delta)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "delta_frontier.png", dpi=150); plt.close(fig)

# 3. calibration (injected defect) - numbers from the session before the metric fix
fig, ax = plt.subplots(figsize=(6.5, 4.2))
vals = [0.2589, 0.2831, 0.2596]
names = ["clean decoder", "injected defect\n(postnet gain x2.5)", "after Astra repair"]
cols = [C_BASE, C_PRUNE, C_DELTA]
bars = ax.bar(names, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylabel("log-mel distance, dB"); ax.set_title("Calibration: can Astra localise a real fault?")
ax.set_ylim(0, 0.32); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=150); plt.close(fig)
print("figures ok")
