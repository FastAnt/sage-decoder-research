"""Recompute every number used in the repo with the fixed log-mel metric (relative floor)."""
import json, sys, time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

REF = Path("/home/mercedes/dev/decoder_refactor")
sys.path.insert(0, str(REF))
from src import audio, sage_tools as ST
from src.sage_prune_verify import apply_plan
from src.sage_patch_verify import apply_patch

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
wav, _ = audio.load(REF / "data_stumblin" / "original.wav")
lat = torch.load(REF / "output_stumblin" / "latent_fp32.pt", map_location="cpu").cuda()
PLAN = REF / "experiments_sage/prune/runs/prune_v1/prune_plan.pt"

c = ST.load_sage("cuda")                       # load ONCE (repeated loads break matmul precision)
dec = c.model.decoder
BASE_W = {n: p.detach().clone() for n, p in dec.named_parameters()}
BASE_BLOCKS = [list(s.blocks) for s in dec.stages]


def reset():
    with torch.no_grad():
        for n, p in dec.named_parameters():
            p.copy_(BASE_W[n])
    for i, s in enumerate(dec.stages):
        s.blocks = torch.nn.ModuleList(BASE_BLOCKS[i])


def mel(rec):
    return round(ST.mel_distance(wav, rec[..., : wav.shape[-1]])["mel_db"], 4)


def measure(tag, do_flops=True):
    t = time.time()
    with torch.no_grad():
        rec = c.decode(lat)[0][..., : wav.shape[-1]]
    dt = round(time.time() - t, 2)
    fl = None
    if do_flops:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            c.decode(lat)
        fl = round(fc.get_total_flops() / 1e9 / (wav.shape[-1] / 44100), 2)
    return {"tag": tag, "params_m": round(sum(p.numel() for p in dec.parameters()) / 1e6, 2),
            "gflop_per_s": fl, "decode_s_gpu": dt, "mel_db": mel(rec)}


res = {}
reset(); res["base"] = measure("base")

curve = []
for tag, spec in [("-1 stage0", {0: 1}), ("-2 stage0", {0: 2}), ("-2s0 -2s1", {0: 2, 1: 2}),
                  ("-2s0 -4s1", {0: 2, 1: 4}), ("-2s0 -6s1", {0: 2, 1: 6}),
                  ("-2s0 -6s1 -1s2", {0: 2, 1: 6, 2: 1})]:
    reset()
    for i, s in enumerate(dec.stages):
        s.blocks = torch.nn.ModuleList(BASE_BLOCKS[i][: len(BASE_BLOCKS[i]) - spec.get(i, 0)])
    curve.append(measure(tag))
res["removal_curve"] = curve

reset()
plan = torch.load(PLAN, map_location="cpu", weights_only=False)
apply_plan(c, plan)
res["astra_plan"] = measure("astra_plan")

FT = REF / "experiments_sage/prune/ft"
w_p = torch.load(FT / "weights_pruned.pt", map_location="cpu")
w_f = torch.load(FT / "weights_ft.pt", map_location="cpu")
delta = {n: (w_f[n].float() - w_p[n].float()) for n in w_p}
named = dict(dec.named_parameters())
frontier = []
for r, bytes_est in ((1, 121678), (2, 237600), (4, 469810)):
    reset(); apply_plan(c, plan)
    approx = {}
    for n, d in delta.items():
        if d.dim() >= 2:
            mat = d.reshape(d.shape[0], -1)
            u, s, v = torch.linalg.svd(mat, full_matrices=False)
            rr = min(r, s.numel())
            approx[n] = w_p[n].float() + ((u[:, :rr] * s[:rr]) @ v[:rr]).reshape(d.shape)
        else:
            approx[n] = w_f[n]
    with torch.no_grad():
        for n, v in approx.items():
            named[n].copy_(v.to(named[n].device))
    row = measure(f"lowrank{r}"); row["delta_kib"] = round(bytes_est / 1024, 1)
    frontier.append(row)
reset(); apply_plan(c, plan)
with torch.no_grad():
    for n, v in w_f.items():
        named[n].copy_(v.to(named[n].device))
row = measure("full_finetune_delta")
row["delta_kib"] = round(sum(v.numel() for v in delta.values()) * 2 / 1024)
frontier.append(row)
res["delta_frontier"] = frontier

reset(); apply_plan(c, plan)
payload = torch.load(REF / "experiments_sage/prune2/runs/restore_v1/patch_plan.pt",
                     map_location="cpu", weights_only=False)
apply_patch(c, payload)
res["astra_patch"] = measure("astra_patch"); res["astra_patch"]["delta_kib"] = 0.6

reset(); res["calib_clean"] = measure("clean", do_flops=False)
defect = torch.load(REF / "experiments_sage/baseline/defect.pt", map_location="cpu", weights_only=False)
with torch.no_grad():
    for e in defect["edits"]:
        named[e["name"]].copy_(e["value"].float().to(named[e["name"]].device))
res["calib_defect"] = measure("defect", do_flops=False)
repair = torch.load(REF / "experiments_sage/runs/calib_v1/delta.pt", map_location="cpu", weights_only=False)
with torch.no_grad():
    for e in repair["edits"]:
        named[e["name"]].copy_(e["value"].float().to(named[e["name"]].device))
res["calib_repaired"] = measure("repaired", do_flops=False)

(OUT / "metrics.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
