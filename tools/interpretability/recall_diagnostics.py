"""recall_diagnostics.py — Track A: WHERE does in-context recall break?

Finding 7 established that WaveletLM does no in-context retrieval (induction lift
~0). This localizes *why*, on an existing checkpoint, with **no training** — the
read-vs-write question from plans/associative_memory_bypass.md.

Method (forward-only). Build matched induction/control sequences:

    induced : [filler] KEY VALUE [filler] KEY(query)   -> should predict VALUE
    control : [filler] K'   V'   [filler] KEY(query)   -> VALUE not bound here

At every layer L we read the residual at the QUERY position and measure its
cosine with the VALUE token's embedding direction. Because the LM head is TIED,
that embedding direction *is* the VALUE logit direction, so this asks: "how much
of the signal needed to predict VALUE has reached the query position by layer L?"
The induced-minus-control gap isolates the in-context binding's contribution;
a random-token direction gives the chance floor.

Reading the result:
  * gap ~0 (~chance) at every layer  -> the binding never reaches the query:
        a WRITE / ROUTING failure. The model does not carry k->v to where the
        prediction happens. => build the associative-memory bypass (the write
        mechanism); the readout is not the problem.
  * gap rises at some layer but the final induction lift stays ~0 -> the signal
        arrives but is not surfaced: a READ failure (the readout can't address
        content). => the bypass must also fix addressing, not just storage.

Runs in ~1-2 min on a laptop GPU/CPU. Usage:
  python tools/interpretability/recall_diagnostics.py \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --trials 256
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

try:  # Windows consoles default to cp950/cp1252 and choke on non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from model import WaveletLM, get_tokenizer  # noqa: E402
from train import load_and_encode_dataset    # noqa: E402


class _L:
    def log(self, m):
        print(m, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--trials", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--gap", type=int, default=64)
    p.add_argument("--filler_dataset", default="wikitext-103")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    enc = get_tokenizer(cfg)
    V, T = enc.vocab_size, int(cfg["block_size"])

    model = WaveletLM(V, cfg, device=device).to(device)
    ck = torch.load(os.path.join(args.run_dir, "best_model.pt"), map_location=device)
    if isinstance(ck, dict) and "model_state" in ck:
        ck = ck["model_state"]
    model.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v
                           for k, v in ck.items()}, strict=True)
    model.eval()
    base = getattr(model, "_orig_mod", model)
    if hasattr(base, "reset_semantic_state"):
        base.reset_semantic_state()
    base.decompose_bypass_cross_window = False
    layers = base.layers
    n_layers = len(layers)
    emb = base.token_embedding.weight            # (V, C), tied to the head

    data_cfg = dict(cfg); data_cfg["dataset"] = args.filler_dataset
    data_cfg["tokenizer"] = "auto"
    _, val_data, test_data, _, _ = load_and_encode_dataset(data_cfg, _L())
    filler = test_data if len(test_data) > len(val_data) else val_data

    pos_pair = T - args.gap - 3
    val_pos = pos_pair + 1                       # the demonstration VALUE position
    seqs_i, seqs_c, keys, values, randoms = [], [], [], [], []
    for _ in range(args.trials):
        start = torch.randint(0, len(filler) - T - 1, (1,)).item()
        s = filler[start:start + T].clone().to(torch.long)
        key, val = torch.randint(1000, 40000, (2,)).tolist()
        k2, v2 = torch.randint(1000, 40000, (2,)).tolist()
        rnd = torch.randint(1000, 40000, (1,)).item()   # unrelated direction
        i_, c_ = s.clone(), s.clone()
        i_[pos_pair], i_[val_pos] = key, val
        c_[pos_pair], c_[val_pos] = k2, v2
        i_[-1] = c_[-1] = key
        seqs_i.append(i_); seqs_c.append(c_)
        keys.append(key); values.append(val); randoms.append(rnd)
    X_i = torch.stack(seqs_i); X_c = torch.stack(seqs_c)
    Yk = torch.tensor(keys); Yv = torch.tensor(values); Yr = torch.tensor(randoms)

    # Capture the residual at TWO positions per layer, in one forward pass:
    #   SOURCE  = the demonstration VALUE position (val_pos): does the binding
    #     form here? We ask whether it encodes its predecessor KEY — the write
    #     signature (a value that "knows it followed KEY" is retrievable by KEY).
    #   QUERY   = the final position (-1): does the binding PROPAGATE to where
    #     VALUE must be predicted? We ask whether it carries VALUE.
    captured = {}

    def mk_hook(idx):
        def hook(_m, _in, out):
            x = out[0] if isinstance(out, tuple) else out
            captured[idx] = torch.stack(
                [x[:, val_pos, :], x[:, -1, :]], dim=1).detach().float()    # (B, 2, C)
        return hook

    handles = [layers[i].register_forward_hook(mk_hook(i)) for i in range(n_layers)]

    @torch.no_grad()
    def run(X, key_dirs, val_dirs, rnd_dirs):
        # per layer: SOURCE cos(VALUEpos, KEY) [write], QUERY cos(query, VALUE)
        # [propagation], QUERY cos(query, random) [chance]; + final logP(VALUE).
        src_k = torch.zeros(n_layers); qry_v = torch.zeros(n_layers)
        qry_r = torch.zeros(n_layers); lp = torch.zeros(len(X)); nseen = 0
        for i in range(0, len(X), args.batch):
            xb = X[i:i + args.batch].to(device); b = xb.size(0)
            kb = F.normalize(emb[key_dirs[i:i + b].to(device)].float(), dim=-1)
            vb = F.normalize(emb[val_dirs[i:i + b].to(device)].float(), dim=-1)
            rb = F.normalize(emb[rnd_dirs[i:i + b].to(device)].float(), dim=-1)
            logits, _ = model(xb, targets=None)
            lp[i:i + b] = F.log_softmax(logits[:, -1, :].float(), -1)[
                torch.arange(b), val_dirs[i:i + b].to(device)].cpu()
            for L in range(n_layers):
                src = F.normalize(captured[L][:, 0, :], dim=-1)   # VALUE position
                qry = F.normalize(captured[L][:, 1, :], dim=-1)   # query position
                src_k[L] += (src * kb).sum(-1).sum().cpu()
                qry_v[L] += (qry * vb).sum(-1).sum().cpu()
                qry_r[L] += (qry * rb).sum(-1).sum().cpu()
            nseen += b
        return src_k / nseen, qry_v / nseen, qry_r / nseen, lp

    sk_i, qv_i, qr_i, lp_i = run(X_i, Yk, Yv, Yr)
    sk_c, qv_c, qr_c, lp_c = run(X_c, Yk, Yv, Yr)
    for h in handles:
        h.remove()

    lift = (lp_i - lp_c).mean().item()
    print(f"\n[recall-diagnostics] {args.run_dir}  ({args.trials} trials, gap={args.gap}, "
          f"{n_layers} layers)")
    print(f"  baseline induction lift (final logP VALUE, induced-control): "
          f"{lift:+.3f} nats  (Finding 7: ~0 expected)\n")
    print("  layer | SOURCE cos(VALUEpos,KEY) i/c/delta | QUERY cos(query,VALUE) i/c/delta | chance")
    for L in range(n_layers):
        sg = (sk_i[L] - sk_c[L]).item(); qg = (qv_i[L] - qv_c[L]).item()
        print(f"   {L:>3}  | {sk_i[L]:+.4f}/{sk_c[L]:+.4f}/{sg:+.4f}"
              f"  | {qv_i[L]:+.4f}/{qv_c[L]:+.4f}/{qg:+.4f}  | {qr_i[L]:+.4f}")

    src_gap = max((sk_i[L] - sk_c[L]).item() for L in range(n_layers))
    qry_gap = max((qv_i[L] - qv_c[L]).item() for L in range(n_layers))
    floor = abs(qr_i).mean().item()
    thr = max(0.02, 2 * floor)
    src_on, qry_on = src_gap > thr, qry_gap > thr
    print(f"\n  chance-scale ~{floor:.4f}; a signal counts as PRESENT above delta {thr:.4f}")
    print(f"  SOURCE write  (VALUEpos encodes KEY):    max delta {src_gap:+.4f} -> "
          f"{'PRESENT' if src_on else 'absent'}")
    print(f"  QUERY propagation (query carries VALUE): max delta {qry_gap:+.4f} -> "
          f"{'PRESENT' if qry_on else 'absent'}")
    print(f"  final readout lift: {lift:+.3f} nats -> {'recall' if lift > 0.3 else 'no recall'}")
    print("  VERDICT:")
    if not src_on and not qry_on:
        print("   binding NEVER FORMS -- write fails at the source (no KEY->VALUE association).")
    elif src_on and not qry_on:
        print("   binding FORMS AT SOURCE but FAILS TO PROPAGATE -- routing/retrieval is the break.")
    elif src_on and qry_on and lift <= 0.3:
        print("   binding forms and propagates but ISN'T SURFACED -- a READ failure at the output.")
    else:
        print("   recall appears functional (unexpected given Finding 7 -- re-check the probe).")


if __name__ == "__main__":
    main()
