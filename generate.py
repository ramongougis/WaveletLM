# Copyright 2025-2026 Ramon Gougis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# EXARCH - Explicit Attentionless Reasoning with Causal Harmonics
# generate.py

import argparse
import math
import time
import json
import os
from collections import Counter

import torch
import torch.nn.functional as F
import tiktoken

from model import ExarchLM, MultiNodeExarchLM, causal_haar_decompose, quantize_model


# ==============================================================================
# UTILITIES
# ==============================================================================

def strip_compiled_prefix(state_dict):
    """Remove _orig_mod. prefix from torch.compile state dicts."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k[10:]] = v
        else:
            out[k] = v
    return out


def load_checkpoint(model, ckpt_path):
    """Load checkpoint with _orig_mod prefix handling."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = strip_compiled_prefix(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Filter benign missing/unexpected keys
    real_missing = [k for k in missing if not k.startswith("fht.")]
    real_unexpected = [k for k in unexpected if not k.endswith(".suppress_concept_vec")
                       and not k.endswith(".amplify_scale_vec")]

    if real_missing:
        print(f"[WARNING] Missing keys: {real_missing}")
    if real_unexpected:
        print(f"[WARNING] Unexpected keys: {real_unexpected}")


# ==============================================================================
# SAMPLING UTILITIES
# ==============================================================================

def apply_repetition_penalty(logits, generated_ids, penalty):
    """Apply repetition penalty to logits based on tokens already generated."""
    if penalty == 1.0:
        return logits
    for batch_idx in range(logits.size(0)):
        unique_tokens = generated_ids[batch_idx].unique()
        for token_id in unique_tokens:
            if logits[batch_idx, token_id] > 0:
                logits[batch_idx, token_id] /= penalty
            else:
                logits[batch_idx, token_id] *= penalty
    return logits


def top_p_filtering(logits, top_p):
    """Apply nucleus (top-p) sampling by zeroing out tokens outside the top-p mass."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


# ==============================================================================
# QUALITY METRICS
# ==============================================================================

def compute_quality_metrics(token_ids, log_probs, entropies):
    """Compute quality metrics for a generated sequence."""
    metrics = {}

    if log_probs:
        metrics['mean_log_prob'] = sum(log_probs) / len(log_probs)
        metrics['sum_log_prob'] = sum(log_probs)
    if entropies:
        metrics['mean_entropy'] = sum(entropies) / len(entropies)

    # Distinct-n
    for n in (1, 2, 3):
        if len(token_ids) >= n:
            ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
            metrics[f'distinct_{n}'] = len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    # Repetition-4
    if len(token_ids) >= 4:
        fourgrams = [tuple(token_ids[i:i+4]) for i in range(len(token_ids) - 3)]
        counts = Counter(fourgrams)
        repeated = set()
        for i, fg in enumerate(fourgrams):
            if counts[fg] > 1:
                for j in range(4):
                    repeated.add(i + j)
        metrics['repetition_4'] = len(repeated) / len(token_ids)

    metrics['num_tokens'] = len(token_ids)
    return metrics


def format_metrics(metrics):
    """Format a metrics dict as a readable string."""
    parts = []
    if 'mean_log_prob' in metrics:
        parts.append(f"MeanLogP={metrics['mean_log_prob']:.4f}")
    if 'mean_entropy' in metrics:
        parts.append(f"MeanH={metrics['mean_entropy']:.2f}")
    for n in (1, 2, 3):
        key = f'distinct_{n}'
        if key in metrics:
            parts.append(f"D{n}={metrics[key]:.3f}")
    if 'repetition_4' in metrics:
        parts.append(f"Rep4={metrics['repetition_4']:.3f}")
    if 'wavelet_drift_mean' in metrics:
        parts.append(f"WavDrift={metrics['wavelet_drift_mean']:.4f}")
    return " | ".join(parts)


# ==============================================================================
# LOOKAHEAD RERANKING
# ==============================================================================

@torch.no_grad()
def _score_lookahead(model, idx, candidate_token, depth, context_len, use_amp, amp_dtype):
    """Score a candidate token by doing a greedy rollout of `depth` steps."""
    device = idx.device
    idx_extended = torch.cat([idx, candidate_token.view(1, 1)], dim=1)
    cumulative_log_prob = 0.0

    for _ in range(depth):
        idx_cond = idx_extended[:, -context_len:]
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp and device.type == 'cuda'):
            logits, _ = model(idx_cond, targets=None)
        next_logits = logits[:, -1, :].float()
        log_probs = torch.log_softmax(next_logits, dim=-1)
        best_token = next_logits.argmax(dim=-1)
        cumulative_log_prob += log_probs[0, best_token[0]].item()
        idx_extended = torch.cat([idx_extended, best_token.unsqueeze(0)], dim=1)

    return cumulative_log_prob


# ==============================================================================
# CORE GENERATION
# ==============================================================================

@torch.no_grad()
def generate_one(
    model,
    enc,
    input_tensor,
    *,
    temperature: float = 1.0,
    num_tokens: int = 512,
    context_len: int = 512,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    return_stats: bool = False,
    entropy_adaptive: bool = False,
    entropy_temp_min: float = 0.3,
    entropy_temp_max: float = 1.5,
    lookahead_k: int = 0,
    lookahead_depth: int = 5,
    wavelet_coherence: bool = False,
    wavelet_coherence_weight: float = 0.5,
    wavelet_coherence_window: int = 128,
    wavelet_coherence_levels: int = 4,
    fwpkm_inference_updates: bool = False,
    fwpkm_update_lr: float = 0.01,
    fwpkm_chunk_size: int = 64,
):
    """Generate tokens one at a time with optional inference strategies."""
    idx = input_tensor.clone()
    device = idx.device
    prompt_len = idx.size(1)

    # Reset semantic state
    if hasattr(model, 'reset_semantic_state'):
        model.reset_semantic_state()

    # Reset FwPKM fast weights at start of generation
    if fwpkm_inference_updates and hasattr(model, 'reset_fast_weights'):
        model.reset_fast_weights()

    # Wavelet coherence: compute prompt's topic fingerprint
    prompt_fingerprint = None
    coherence_drift_scores = []
    _wavelet_emb_fn = getattr(model, 'token_embedding', None)
    if wavelet_coherence and _wavelet_emb_fn is not None:
        prompt_emb = _wavelet_emb_fn(idx)
        if prompt_emb.size(1) >= (1 << wavelet_coherence_levels):
            approx, _ = causal_haar_decompose(prompt_emb, wavelet_coherence_levels)
            prompt_fingerprint = approx.mean(dim=1)
            prompt_fingerprint = F.normalize(prompt_fingerprint, dim=-1)

    log_probs_list = []
    entropies_list = []

    for i in range(num_tokens):
        idx_cond = idx[:, -context_len:]

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp and device.type == 'cuda'):
            logits, _ = model(idx_cond, targets=None)
        next_logits = logits[:, -1, :].float()

        # Periodically clear cache
        if i % 128 == 127 and device.type == 'cuda':
            torch.cuda.empty_cache()

        # Repetition penalty
        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, idx, repetition_penalty)

        # Entropy tracking
        if return_stats or entropy_adaptive:
            raw_probs = torch.softmax(next_logits, dim=-1)
            log_raw = torch.log(raw_probs + 1e-12)
            entropy = -(raw_probs * log_raw).sum(dim=-1)
            if return_stats:
                entropies_list.append(entropy[0].item())

        # Temperature (possibly adaptive)
        if entropy_adaptive:
            vocab_size = next_logits.size(-1)
            max_entropy = math.log(vocab_size)
            h_norm = entropy[0].item() / max_entropy
            adaptive_temp = entropy_temp_max - h_norm * (entropy_temp_max - entropy_temp_min)
            next_logits /= max(1e-8, adaptive_temp)
        else:
            next_logits /= max(1e-8, temperature)

        # Wavelet coherence: tighten sampling when topic drifts
        effective_top_p = top_p
        if prompt_fingerprint is not None and i > 0 and i % 16 == 0:
            window_start = max(0, idx.size(1) - wavelet_coherence_window)
            recent_ids = idx[:, window_start:]
            recent_emb = _wavelet_emb_fn(recent_ids)
            if recent_emb.size(1) >= (1 << wavelet_coherence_levels):
                approx, _ = causal_haar_decompose(recent_emb, wavelet_coherence_levels)
                current_fp = F.normalize(approx.mean(dim=1), dim=-1)
                cos_sim = (prompt_fingerprint * current_fp).sum(dim=-1).item()
                drift = 1.0 - cos_sim
                coherence_drift_scores.append(drift)
                drift_penalty = drift * wavelet_coherence_weight
                effective_top_p = max(0.3, top_p - drift_penalty)

        # Top-p (nucleus) filtering
        if effective_top_p < 1.0:
            next_logits = top_p_filtering(next_logits, effective_top_p)

        # Lookahead reranking
        if lookahead_k > 0:
            topk_logits, topk_indices = torch.topk(
                next_logits[0], k=min(lookahead_k, next_logits.size(-1)))

            saved_state = None
            if hasattr(model, '_persistent_semantic_state') and model._persistent_semantic_state is not None:
                saved_state = model._persistent_semantic_state.clone()
            saved_count = getattr(model, '_persistent_token_count', 0)

            best_score = float('-inf')
            best_candidate = topk_indices[0]
            for cand_idx in range(topk_indices.size(0)):
                candidate = topk_indices[cand_idx]
                if saved_state is not None:
                    model._persistent_semantic_state = saved_state.clone()
                    model._persistent_token_count = saved_count
                score = _score_lookahead(
                    model, idx, candidate, lookahead_depth,
                    context_len, use_amp, amp_dtype)
                cand_log_prob = torch.log_softmax(next_logits, dim=-1)[0, candidate].item()
                score += cand_log_prob
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if saved_state is not None:
                model._persistent_semantic_state = saved_state
                model._persistent_token_count = saved_count

            idx_next = best_candidate.view(1, 1)
            if return_stats:
                log_prob = torch.log_softmax(next_logits, dim=-1)
                log_probs_list.append(log_prob[0, best_candidate].item())
        else:
            probs = torch.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if return_stats:
                log_prob = torch.log_softmax(next_logits, dim=-1)
                log_probs_list.append(log_prob[0, idx_next[0, 0]].item())

        idx = torch.cat((idx, idx_next), dim=1)

        # FwPKM: update fast weights after each chunk
        if (fwpkm_inference_updates and hasattr(model, 'update_fast_weights')
                and (i + 1) % fwpkm_chunk_size == 0
                and idx.size(1) > fwpkm_chunk_size):
            chunk_end = idx.size(1)
            chunk_start = chunk_end - fwpkm_chunk_size
            chunk_ids = idx[:, chunk_start:chunk_end]
            model.update_fast_weights(chunk_ids, lr=fwpkm_update_lr)

    text = enc.decode(idx[0].tolist())
    generated_ids = idx[0, prompt_len:].tolist()

    if return_stats:
        stats = {
            'log_probs': log_probs_list,
            'entropies': entropies_list,
            'token_ids': generated_ids,
        }
        if coherence_drift_scores:
            stats['wavelet_drift'] = coherence_drift_scores
            stats['wavelet_drift_mean'] = sum(coherence_drift_scores) / len(coherence_drift_scores)
            stats['wavelet_drift_max'] = max(coherence_drift_scores)
        return text, generated_ids, stats
    return text


# ==============================================================================
# BEST-OF-N GENERATION
# ==============================================================================

def generate_best_of_n(model, enc, input_tensor, n, seed, **gen_kwargs):
    """Generate N completions and select by highest mean log-prob."""
    gen_kwargs['return_stats'] = True
    candidates = []

    for j in range(n):
        torch.manual_seed(seed + j)
        if input_tensor.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed + j)

        txt, token_ids, stats = generate_one(model, enc, input_tensor, **gen_kwargs)
        metrics = compute_quality_metrics(token_ids, stats['log_probs'], stats['entropies'])
        if 'wavelet_drift_mean' in stats:
            metrics['wavelet_drift_mean'] = stats['wavelet_drift_mean']
            metrics['wavelet_drift_max'] = stats['wavelet_drift_max']
        candidates.append((txt, token_ids, stats, metrics))

    best_idx = max(range(len(candidates)),
                   key=lambda j: candidates[j][3].get('mean_log_prob', float('-inf')))
    txt, token_ids, stats, metrics = candidates[best_idx]

    print(f"[Best-of-{n}] Selected candidate {best_idx+1}/{n}")
    for j, (_, _, _, m) in enumerate(candidates):
        marker = " <-- SELECTED" if j == best_idx else ""
        print(f"  Candidate {j+1}: MeanLogP={m.get('mean_log_prob', 0):.4f}{marker}")

    return txt, metrics


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser("EXARCH text generation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint (e.g., logs/run_dir/best_model.pt)")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (default: same dir as checkpoint)")
    parser.add_argument("--prompt", default="The history of",
                        help="Generation prompt")
    parser.add_argument("--num_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")

    # Inference strategies
    parser.add_argument("--strategies", action="store_true",
                        help="Enable all inference strategies (entropy_adaptive, lookahead, wavelet_coherence, best_of_n=5, metrics)")
    parser.add_argument("--best_of_n", type=int, default=1)
    parser.add_argument("--entropy_adaptive", action="store_true")
    parser.add_argument("--entropy_temp_min", type=float, default=0.3)
    parser.add_argument("--entropy_temp_max", type=float, default=1.5)
    parser.add_argument("--lookahead_k", type=int, default=0)
    parser.add_argument("--lookahead_depth", type=int, default=5)
    parser.add_argument("--wavelet_coherence", action="store_true")
    parser.add_argument("--wavelet_coherence_weight", type=float, default=0.5)
    parser.add_argument("--wavelet_coherence_window", type=int, default=128)
    parser.add_argument("--wavelet_coherence_levels", type=int, default=4)
    parser.add_argument("--fwpkm_inference_updates", action="store_true",
                        help="Enable FwPKM fast-weight updates during generation")
    parser.add_argument("--fwpkm_update_lr", type=float, default=0.01)
    parser.add_argument("--fwpkm_chunk_size", type=int, default=64)
    parser.add_argument("--metrics", action="store_true",
                        help="Print quality metrics")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of completions to generate")
    args = parser.parse_args()

    # --strategies enables inference strategies matching EXARCH-research defaults
    if args.strategies:
        args.entropy_adaptive = True
        args.entropy_temp_max = 0.9
        args.top_p = 0.85
        args.repetition_penalty = 1.2
        args.metrics = True

    # Resolve device
    device = torch.device(args.device if torch.cuda.is_available() or args.device != 'cuda' else 'cpu')
    print(f"Device: {device}")

    # Load config
    if args.config is None:
        run_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(run_dir, "config.json")
    else:
        run_dir = os.path.dirname(args.checkpoint)
        config_path = args.config

    # Log to both stdout and generations.txt
    gen_file_path = os.path.join(run_dir, "generations.txt")
    gen_file = open(gen_file_path, 'a', encoding='utf-8')

    def log(msg=""):
        print(msg)
        gen_file.write(msg + "\n")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Precision
    use_amp = config.get('use_amp', True)
    amp_dtype_str = config.get('amp_dtype', 'fp16')
    amp_dtype = torch.float16 if amp_dtype_str == 'fp16' else torch.bfloat16

    if config.get('allow_tf32', True) and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    if config.get('multinodal_enabled', False):
        model = MultiNodeExarchLM(vocab_size, config, device=device)
    else:
        model = ExarchLM(vocab_size, config, device=device)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    model.eval()

    if config.get('quantize_enabled', False):
        q_stats = quantize_model(model, config)
        print(f"[Quantization] Original: {q_stats['original_mib']:.1f} MiB -> "
              f"Quantized: {q_stats['quantized_mib']:.1f} MiB "
              f"({q_stats['compression_ratio']:.2f}x)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.2f}M parameters")

    # Encode prompt
    prompt_ids = enc.encode(args.prompt)
    if not prompt_ids:
        print("[WARNING] Empty prompt, using [0]")
        prompt_ids = [0]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    context_len = config.get('block_size', 512)

    gen_kwargs = dict(
        temperature=args.temperature,
        num_tokens=args.num_tokens,
        context_len=context_len,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        entropy_adaptive=args.entropy_adaptive,
        entropy_temp_min=args.entropy_temp_min,
        entropy_temp_max=args.entropy_temp_max,
        lookahead_k=args.lookahead_k,
        lookahead_depth=args.lookahead_depth,
        wavelet_coherence=args.wavelet_coherence,
        wavelet_coherence_weight=args.wavelet_coherence_weight,
        wavelet_coherence_window=args.wavelet_coherence_window,
        wavelet_coherence_levels=args.wavelet_coherence_levels,
        fwpkm_inference_updates=args.fwpkm_inference_updates,
        fwpkm_update_lr=args.fwpkm_update_lr,
        fwpkm_chunk_size=args.fwpkm_chunk_size,
    )

    gen_file.write("\n" + "=" * 60 + "\n")
    gen_file.write(f"Generation run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    gen_file.write(f"Checkpoint: {args.checkpoint}\n")
    gen_file.write("=" * 60 + "\n\n")

    log(f"Prompt: {args.prompt}")
    log(f"Generating {args.num_tokens} tokens (temp={args.temperature}, top_p={args.top_p})...")
    log("-" * 60)

    for sample_idx in range(args.n):
        if args.n > 1:
            log(f"\n--- Sample {sample_idx + 1}/{args.n} ---")

        t0 = time.time()

        if args.best_of_n > 1:
            txt, metrics = generate_best_of_n(
                model, enc, input_tensor, args.best_of_n,
                args.seed + sample_idx * 100, **gen_kwargs)
        elif args.metrics:
            txt, token_ids, stats = generate_one(
                model, enc, input_tensor, return_stats=True, **gen_kwargs)
            metrics = compute_quality_metrics(token_ids, stats['log_probs'], stats['entropies'])
            if 'wavelet_drift_mean' in stats:
                metrics['wavelet_drift_mean'] = stats['wavelet_drift_mean']
                metrics['wavelet_drift_max'] = stats['wavelet_drift_max']
        else:
            txt = generate_one(model, enc, input_tensor, **gen_kwargs)
            metrics = None

        elapsed = time.time() - t0

        log(txt)
        log("-" * 60)
        log(f"Time: {elapsed:.2f}s ({args.num_tokens/elapsed:.1f} tok/s)")

        if metrics:
            log(f"Metrics: {format_metrics(metrics)}")

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        log(f"\nPeak GPU memory: {peak_mem:.0f} MiB")

    gen_file.close()
    print(f"Saved to {gen_file_path}")


if __name__ == "__main__":
    main()
