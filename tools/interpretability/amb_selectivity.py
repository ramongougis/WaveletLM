"""amb_selectivity.py — is the AMB doing content-addressed recall, or a running mean?

Loads a trained run, forwards a batch of real WT-103 val text, and reports per AMB
layer the statistics that settle the "running mean vs selective read" question the
2026-07-24/25 feature-map work raised (see plans/associative_memory_bypass.md):

  s              learned sharpness exp(log_s)   [softplus_s only] -- higher = sharper
  unrel<q,k>     mean off-diagonal cosine of the read scores. ~0.86 = softplus_l2's
                 running-mean floor; toward 0 = selective.
  w-entropy      read-weight entropy / log(context). ~1.0 = uniform (running mean);
                 lower = the read concentrates on few tokens (content-addressed).
  |b*amb|/|x|    fraction of the residual the AMB injects (0 = inert / beta-collapsed).
  effrank(S)     participation ratio of the d x d state S_T (capacity actually used;
                 << d means slots are superposed -> interference + illegibility).

Mirrors _assoc_retrieve's math (kept in sync by hand -- it's a diagnostic, not core).

Usage:  python tools/interpretability/amb_selectivity.py --run logs/<run_dir> [--batch 8]
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from model import WaveletLM               # noqa: E402
from train import load_and_encode_dataset  # noqa: E402


class _L:
    def log(self, m): print(m, flush=True)


def phi(z, fmap, log_s):
    if fmap == "elu1":        return F.elu(z) + 1.0
    if fmap == "relu2":       return F.relu(z).square()
    if fmap == "relu_l2":     return F.normalize(F.relu(z), dim=-1, eps=1e-6)
    if fmap == "relu2_l2":    return F.normalize(F.relu(z).square(), dim=-1, eps=1e-6)
    if fmap == "softplus_l2": return F.normalize(F.softplus(z), dim=-1, eps=1e-6)
    if fmap == "softplus_s":  return F.normalize(F.softplus(log_s.exp() * z), dim=-1, eps=1e-6)
    raise ValueError(fmap)


def load_model(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg["compile"] = False
    model = WaveletLM(cfg.get("vocab_size", 50257), cfg, device=device).to(device)
    ckpt_path = next((os.path.join(run_dir, n) for n in
                      ["best_model.pt", "last_checkpoint.pt", "model.pt"]
                      if os.path.exists(os.path.join(run_dir, n))), None)
    if ckpt_path is None:
        print(f"[warn] no checkpoint in {run_dir}; using INIT weights"); return model, cfg
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}   # strip torch.compile prefix
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt_path}  (missing {len(miss)}, unexpected {len(unexp)})")
    return model, cfg


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    model, cfg = load_model(args.run, dev); model.eval()
    T = cfg.get("block_size", 256)
    _, va, _, _, _ = load_and_encode_dataset(
        {"dataset": cfg.get("dataset", "wikitext-103"), "tokenizer": "auto", "block_size": T}, _L())
    g = torch.Generator().manual_seed(0)
    s0 = torch.randint(0, len(va) - T - 1, (args.batch,), generator=g)
    x = torch.stack([va[i:i + T] for i in s0]).to(dev).long()

    blocks = [b for b in model.layers if getattr(b, "associative_bypass_enabled", False)]
    xin, inj = {}, {}
    for i, b in enumerate(model.layers):
        if not getattr(b, "associative_bypass_enabled", False):
            continue
        b.register_forward_pre_hook(lambda m, a, i=i: xin.__setitem__(i, a[0].detach()))
        b.assoc_out.register_forward_hook(lambda m, a, o, i=i: inj.__setitem__(i, o.detach()))
    model(x)

    causal = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool))
    tpos = torch.arange(1, T + 1, device=dev).float()                 # context length per row
    print(f"\n{'layer':>5} {'s':>6} {'unrel<q,k>':>11} {'w-entropy':>10} {'|b*amb|/|x|':>12} {'effrank(S)':>11}")
    for i, b in enumerate(model.layers):
        if i not in xin:
            continue
        fmap = b.assoc_feature_map; log_s = getattr(b, "assoc_log_s", None)
        s_val = log_s.exp().item() if (fmap == "softplus_s" and log_s is not None) else float("nan")
        x0 = b.assoc_ln(xin[i].float())
        q = phi(b.assoc_q(x0), fmap, log_s); k = phi(b.assoc_k(x0), fmap, log_s)
        v = b.assoc_v(x0)
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = torch.einsum("btd,bsd->bts", qn, kn)                    # (B,T,T) cosine
        off = sim.masked_select(~torch.eye(T, device=dev, dtype=torch.bool)).mean().item()
        score = torch.einsum("btd,bsd->bts", q, k).clamp_min(0) * causal   # read weights (causal)
        prob = score / score.sum(-1, keepdim=True).clamp_min(1e-9)
        ent = -(prob * (prob + 1e-12).log()).sum(-1)                 # (B,T)
        went = (ent[:, 1:] / tpos[1:].log().clamp_min(1e-6)).mean().item()
        amb = b.assoc_beta * inj[i]
        ratio = (amb.norm(dim=-1) / xin[i].float().norm(dim=-1).clamp_min(1e-6)).mean().item()
        S = torch.einsum("btd,bte->bde", k, v)                        # (B,d,d) = S_T
        sv = torch.linalg.svdvals(S.float())
        effrank = (sv.sum(-1) ** 2 / (sv ** 2).sum(-1).clamp_min(1e-12)).mean().item()
        print(f"{i:>5} {s_val:>6.2f} {off:>11.3f} {went:>10.3f} {ratio:>12.4f} {effrank:>11.2f}")
    print(f"\nread of the table: unrel<q,k>~0.86 & w-entropy~1.0 & |b*amb|/|x|~0 => inert running "
          f"mean (softplus_l2 failure). Selective+used => low unrel, w-entropy<1, ratio>0, effrank>1.")


if __name__ == "__main__":
    main()
