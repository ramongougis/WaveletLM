"""induction_probe.py — does a trained WaveletLM do in-context copying?

The capability that gates attention-free architectures as modern-LLM
components is ASSOCIATIVE RECALL: having seen "KEY VALUE" earlier in the
context, predict VALUE when KEY reappears (Olsson et al.'s induction
behaviour; the Mamba paper's Induction Heads / Selective Copying tasks).
Attention and selective SSMs do this; linear time-invariant SSMs famously
cannot, and WaveletLM's mixing is structurally LTI-like (fixed dilation
ladder, position-uniform predict/update nets) — so this is the honest test.

Protocol (works on ANY trained checkpoint, no training required):
  induced : [natural filler] KEY VALUE [natural filler] KEY -> ?
  control : [natural filler] K'  V'    [natural filler] KEY -> ?
Both sequences are token-identical except for the demonstrated pair, so the
difference in log P(VALUE) at the final position isolates the in-context
copy. Filler is real test-set text (keeps the model in-distribution); KEY /
VALUE are random mid-vocabulary ids with no prior association.

Reported:
  lift  = mean logP(VALUE | induced) - mean logP(VALUE | control), in nats.
          ~0     => no in-context copying (LTI-like failure mode)
          >> 0   => induction behaviour present
  acc   = fraction of trials where VALUE is the argmax next token (induced).
  rank  = median rank of VALUE among 50,257 candidates, both conditions.

Caveat: a null result here means the model did not LEARN induction from
WikiText/Pile pretraining; it is suggestive but not proof that the
architecture CANNOT (that requires the synthetic-task training arm).

Usage:
  python tools/interpretability/induction_probe.py \
      --run_dir logs/wikitext-103_2026-07-15_10-53-46 --trials 256
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

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
    p.add_argument("--gap", type=int, default=64,
                   help="tokens between the demonstrated pair and the query")
    p.add_argument("--filler_dataset", default=None,
                   help="Dataset supplying the filler text (default: the run's own). "
                        "Set the SAME value across checkpoints for a controlled "
                        "comparison; also avoids loading huge streamed caches.")
    p.add_argument("--hf_model", default=None,
                   help="POSITIVE CONTROL: run the identical probe on a "
                        "HuggingFace causal LM (e.g. 'gpt2'). --run_dir still "
                        "supplies block_size/filler/protocol, so only the model "
                        "differs. Requires the same GPT-2 BPE vocabulary.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    enc = get_tokenizer(cfg)
    V, T = enc.vocab_size, int(cfg["block_size"])

    if args.hf_model:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.hf_model).to(device)
        model.eval()
        label = f"{args.hf_model} (HF control)"

        def _fwd(xb):
            return model(xb).logits
    else:
        model = WaveletLM(V, cfg, device=device).to(device)
        ckpt = torch.load(os.path.join(args.run_dir, "best_model.pt"),
                          map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        model.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v
                               for k, v in ckpt.items()}, strict=True)
        model.eval()
        # Match the benchmark's eval conditions: no cross-window state leakage.
        base = getattr(model, "_orig_mod", model)
        if hasattr(base, "reset_semantic_state"):
            base.reset_semantic_state()
        base.decompose_bypass_cross_window = False
        label = args.run_dir

        def _fwd(xb):
            return model(xb, targets=None)[0]

    # Filler text: by default the run's own eval split (in-distribution); with
    # --filler_dataset, a fixed corpus shared across checkpoints (controlled).
    data_cfg = dict(cfg)
    if args.filler_dataset:
        data_cfg["dataset"] = args.filler_dataset
        data_cfg["tokenizer"] = "auto"
    _, val_data, test_data, _, _ = load_and_encode_dataset(data_cfg, _L())
    filler = test_data if len(test_data) > len(val_data) else val_data

    # Build trials. Layout inside one T-token window:
    #   [filler ...] KEY VALUE [filler x gap] KEY   -> predict VALUE
    pos_pair = T - args.gap - 3          # index of KEY in the demonstration
    seqs_i, seqs_c, values = [], [], []
    for _ in range(args.trials):
        start = torch.randint(0, len(filler) - T - 1, (1,)).item()
        base_seq = filler[start:start + T].clone().to(torch.long)
        key, val = torch.randint(1000, 40000, (2,)).tolist()
        k2, v2 = torch.randint(1000, 40000, (2,)).tolist()
        induced, control = base_seq.clone(), base_seq.clone()
        induced[pos_pair], induced[pos_pair + 1] = key, val
        control[pos_pair], control[pos_pair + 1] = k2, v2
        induced[-1] = control[-1] = key          # the query
        seqs_i.append(induced); seqs_c.append(control); values.append(val)
    X_i = torch.stack(seqs_i); X_c = torch.stack(seqs_c)
    Y = torch.tensor(values)

    @torch.no_grad()
    def final_logprobs(X):
        out = []
        for i in range(0, len(X), args.batch):
            logits = _fwd(X[i:i + args.batch].to(device))
            out.append(F.log_softmax(logits[:, -1, :].float(), dim=-1).cpu())
        return torch.cat(out)

    lp_i, lp_c = final_logprobs(X_i), final_logprobs(X_c)
    idx = torch.arange(len(Y))
    v_i, v_c = lp_i[idx, Y], lp_c[idx, Y]
    rank_i = (lp_i > v_i.unsqueeze(1)).sum(1) + 1
    rank_c = (lp_c > v_c.unsqueeze(1)).sum(1) + 1
    acc = (lp_i.argmax(-1) == Y).float().mean().item()
    top10 = (rank_i <= 10).float().mean().item()

    print(f"\n[induction] {label}  ({args.trials} trials, gap={args.gap}, "
          f"pair at position {pos_pair}/{T})")
    print(f"  logP(VALUE) induced : {v_i.mean():+.3f} nats")
    print(f"  logP(VALUE) control : {v_c.mean():+.3f} nats")
    print(f"  INDUCTION LIFT      : {(v_i - v_c).mean():+.3f} nats "
          f"(>>0 = in-context copying present; ~0 = absent)")
    print(f"  median rank of VALUE: induced {rank_i.median().item():,} / "
          f"control {rank_c.median().item():,}  (of {V:,})")
    print(f"  VALUE is argmax     : {100*acc:.1f}%   in top-10: {100*top10:.1f}%")
    print(f"  (uniform-chance logP would be {-torch.log(torch.tensor(float(V))):.3f})")


if __name__ == "__main__":
    main()
