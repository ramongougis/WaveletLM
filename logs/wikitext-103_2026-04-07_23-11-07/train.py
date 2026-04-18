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

# WaveletLM
# train.py

import os
import math
import json
import time
import random
import shutil
import argparse
import datetime

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

from model import (
    set_seed, WaveletLM, MultiNodeWaveletLM,
    Logger, parameter_breakdown, quantize_model,
)


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_and_encode_dataset(config, logger):
    """Load dataset via HuggingFace and encode with GPT-2 tokenizer (tiktoken).

    Returns:
        train_data, val_data, test_data: torch.long tensors of token IDs
        enc: tiktoken encoding object
    """
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab  # 50257

    dataset_name = config.get("dataset", "wikitext-103")

    # Check for cached encoded tensors
    cache_dir = ".cache"
    cache_path = os.path.join(cache_dir, f"{dataset_name}_gpt2.pt")

    if os.path.exists(cache_path):
        logger.log(f"[Dataset] Loading {dataset_name} from cache...")
        cached = torch.load(cache_path, weights_only=False)
        train_data = cached['train'].long()
        val_data = cached['val'].long()
        test_data = cached['test'].long()
        test_bytes = cached.get('test_bytes', None)
        if test_bytes is None:
            # Old cache without byte count — compute from decoding
            test_bytes = len(enc.decode(test_data.tolist()).encode("utf-8"))
        bytes_per_token = test_bytes / len(test_data)
        logger.log(f"[Dataset] {dataset_name}: train={len(train_data):,}, "
                   f"val={len(val_data):,}, test={len(test_data):,} tokens (cached)")
        logger.log(f"[Dataset] Test: {test_bytes:,} bytes, {bytes_per_token:.4f} bytes/token")
        logger.log(f"[Dataset] Vocab size: {vocab_size}")
        return train_data, val_data, test_data, enc, bytes_per_token

    # Encode from scratch
    if dataset_name == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    elif dataset_name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    else:
        ds = load_dataset(dataset_name)

    def encode_split(split_data):
        text = "\n\n".join(split_data["text"])
        num_bytes = len(text.encode("utf-8"))
        tokens = enc.encode(text, allowed_special=set())
        return torch.tensor(tokens, dtype=torch.long), num_bytes

    logger.log(f"[Dataset] Loading {dataset_name}...")
    train_data, _ = encode_split(ds["train"])
    val_data, _ = encode_split(ds["validation"])
    test_data, test_bytes = encode_split(ds["test"])

    # Cache for future runs (stored as int32 to save space; cast back on load)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'train': train_data.int(), 'val': val_data.int(), 'test': test_data.int(),
                'test_bytes': test_bytes}, cache_path)
    logger.log(f"[Dataset] Cached to {cache_path}")

    # Remove HuggingFace dataset cache now that we have our own tensor cache
    del ds
    try:
        import shutil as _shutil
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
        if os.path.exists(hf_cache):
            for entry in os.listdir(hf_cache):
                if dataset_name.replace("-", "_") in entry.lower():
                    _shutil.rmtree(os.path.join(hf_cache, entry), ignore_errors=True)
                    logger.log(f"[Dataset] Removed HF cache: {entry}")
    except Exception:
        pass

    bytes_per_token = test_bytes / len(test_data)
    logger.log(f"[Dataset] {dataset_name}: train={len(train_data):,}, "
               f"val={len(val_data):,}, test={len(test_data):,} tokens")
    logger.log(f"[Dataset] Test: {test_bytes:,} bytes, {bytes_per_token:.4f} bytes/token")
    logger.log(f"[Dataset] Vocab size: {vocab_size}")

    return train_data, val_data, test_data, enc, bytes_per_token


# ==============================================================================
# LEARNING RATE SCHEDULE
# ==============================================================================

def get_lr(step, config, total_steps, warmup_steps):
    """Linear warmup + cosine decay to min_lr."""
    if step < warmup_steps:
        return config['lr'] * step / warmup_steps

    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(decay_ratio * math.pi))
    return config['min_lr'] + coeff * (config['lr'] - config['min_lr'])


# ==============================================================================
# BATCH GENERATION
# ==============================================================================

def make_get_batch(train_data, val_data, config, device):
    """Create get_batch closure with cached arange."""
    T = config['block_size']
    _arange_T = torch.arange(T, device=device)[None, :]

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        bs = config['micro_batch_size']
        max_start = len(data) - T - 1
        ix = torch.randint(0, max_start, (bs,), device=device)
        offsets = ix[:, None] + _arange_T
        x = data[offsets]
        y = data[offsets + 1]
        return x, y

    return get_batch


# ==============================================================================
# LOSS ESTIMATION
# ==============================================================================

@torch.no_grad()
def estimate_loss(model, get_batch, config, device, use_amp, amp_dtype):
    """Estimate train and val loss over eval_interval batches each."""
    eval_batches = config['eval_interval']
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                _, loss = model(X, Y)
            losses[k] = loss.detach()
        out[split] = float(losses.mean())
    model.train()
    return out


# ==============================================================================
# BENCHMARKS
# ==============================================================================

@torch.no_grad()
def evaluate_full_validation(model, eval_data, config, logger, device, use_amp, amp_dtype):
    """Evaluate over entire test set using non-overlapping windows."""
    model.eval()

    # Reset semantic state
    base_model = getattr(model, "_orig_mod", model)
    if hasattr(base_model, 'reset_semantic_state'):
        base_model.reset_semantic_state()

    T = config['block_size']
    batch_size = config['micro_batch_size']
    eval_len = len(eval_data)
    num_windows = (eval_len - 1) // T

    if num_windows == 0:
        logger.log("[WARN] Test data too small for evaluation")
        model.train()
        return None

    total_loss = 0.0
    total_tokens = 0
    num_batches = (num_windows + batch_size - 1) // batch_size

    logger.log(f"\n[BENCHMARK] Non-overlapping: {num_windows} windows of length {T}")

    pbar = tqdm(range(num_batches), desc="Non-overlapping Benchmark")
    for batch_idx in pbar:
        start_window = batch_idx * batch_size
        end_window = min(start_window + batch_size, num_windows)
        current_bs = end_window - start_window

        if current_bs < batch_size and batch_idx == num_batches - 1:
            continue  # skip incomplete final batch

        x_list, y_list = [], []
        for w in range(start_window, end_window):
            offset = w * T
            x_list.append(eval_data[offset:offset + T])
            y_list.append(eval_data[offset + 1:offset + T + 1])

        X = torch.stack(x_list).to(device)
        Y = torch.stack(y_list).to(device)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(X, targets=None)

        B, seq_len, V = logits.shape
        loss_per_token = F.cross_entropy(
            logits.view(-1, V), Y.view(-1).long(), reduction='none')
        total_loss += loss_per_token.sum().item()
        total_tokens += B * seq_len

    model.train()

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    bits_per_token = avg_loss / math.log(2)

    return {
        'perplexity': perplexity,
        'bits_per_token': bits_per_token,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
    }


@torch.no_grad()
def evaluate_sliding_window(model, eval_data, config, logger, device, use_amp, amp_dtype,
                             stride=None):
    """Evaluate using sliding window with overlap (HuggingFace methodology)."""
    model.eval()

    base_model = getattr(model, "_orig_mod", model)
    if hasattr(base_model, 'reset_semantic_state'):
        base_model.reset_semantic_state()

    T = config['block_size']
    batch_size = config['micro_batch_size']
    if stride is None:
        stride = T // 2

    eval_len = len(eval_data)
    min_context = T - stride

    # Build windows
    prev_end_loc = 0
    windows = []
    for begin_loc in range(0, eval_len - 1, stride):
        end_loc = min(begin_loc + T, eval_len - 1)
        if end_loc - begin_loc < T:
            break
        trg_len = end_loc - prev_end_loc
        windows.append((begin_loc, trg_len))
        prev_end_loc = end_loc
        if end_loc >= eval_len - 1:
            break

    logger.log(f"\n[SLIDING WINDOW] block_size={T}, stride={stride}, "
               f"min_context={min_context}, {len(windows)} windows")

    total_loss = 0.0
    total_scored_tokens = 0
    num_batches = (len(windows) + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), desc="Sliding Window Benchmark")
    for batch_idx in pbar:
        start_w = batch_idx * batch_size
        end_w = min(start_w + batch_size, len(windows))
        current_bs = end_w - start_w

        if current_bs < batch_size and batch_idx == num_batches - 1 and num_batches > 1:
            continue

        x_list, y_list, trg_lens = [], [], []
        for w_idx in range(start_w, end_w):
            begin_loc, trg_len = windows[w_idx]
            x_list.append(eval_data[begin_loc:begin_loc + T])
            y_list.append(eval_data[begin_loc + 1:begin_loc + T + 1])
            trg_lens.append(trg_len)

        X = torch.stack(x_list).to(device)
        Y = torch.stack(y_list).to(device)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(X, targets=None)

        B, seq_len, V = logits.shape
        for b in range(B):
            trg_len = trg_lens[b]
            logits_scored = logits[b, -trg_len:, :]
            targets_scored = Y[b, -trg_len:]
            loss_per_token = F.cross_entropy(
                logits_scored, targets_scored.long(), reduction='none')
            total_loss += loss_per_token.sum().item()
            total_scored_tokens += loss_per_token.numel()

    model.train()

    if total_scored_tokens == 0:
        logger.log("[SLIDING WINDOW] No tokens scored.")
        return None

    avg_loss = total_loss / total_scored_tokens
    perplexity = math.exp(avg_loss)
    bits_per_token = avg_loss / math.log(2)

    return {
        'perplexity': perplexity,
        'bits_per_token': bits_per_token,
        'avg_loss': avg_loss,
        'total_scored_tokens': total_scored_tokens,
        'num_windows': len(windows),
        'stride': stride,
        'min_context': min_context,
    }


# ==============================================================================
# CHECKPOINT SAVE
# ==============================================================================

def save_with_retry(state_dict, path, retries=3):
    """Save checkpoint with retry logic for filesystem issues."""
    for attempt in range(retries):
        try:
            torch.save(state_dict, path)
            return
        except (OSError, RuntimeError) as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                raise


# ==============================================================================
# MAIN TRAINING
# ==============================================================================

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found!")
    with open(args.config, 'r') as f:
        config = json.load(f)

    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Precision
    use_amp = config.get('use_amp', True)
    amp_dtype_str = config.get('amp_dtype', 'fp16')
    amp_dtype = torch.float16 if amp_dtype_str == 'fp16' else torch.bfloat16

    if config.get('allow_tf32', True) and device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Benchmark-only mode: skip training, load checkpoint, run benchmarks + generation
    benchmark_only = config.get('benchmark_only', False)
    benchmark_run_dir = config.get('benchmark_run_dir', '')
    dataset_name = config.get("dataset", "wikitext-103")

    if benchmark_only:
        if not benchmark_run_dir:
            raise ValueError("benchmark_only=true requires benchmark_run_dir to be set")
        # Load config from the run directory
        run_config_path = os.path.join(benchmark_run_dir, "config.json")
        if os.path.exists(run_config_path):
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
            for k in ['C', 'layers', 'levels', 'low_rank', 'mlp_expansion', 'mlp_layers',
                       'pkm_enabled', 'pkm_num_keys', 'pkm_top_k', 'pkm_heads',
                       'fwpkm_enabled', 'fwpkm_num_keys', 'fwpkm_top_k', 'fwpkm_heads',
                       'wavelet_mode', 'shared_lifting_weights', 'lifting_linear_only',
                       'lifting_hidden_mult', 'lifting_init', 'lifting_dropout',
                       'use_mixer_gate', 'mixer_gate_activation', 'decompose_bypass',
                       'decompose_bypass_cross_window', 'learned_residual', 'skip_proj_out',
                       'stochastic_depth_rate', 'tie_embedding_to_lm_head',
                       'multinodal_enabled', 'multinodal_num_cells', 'multinodal_cell_dim',
                       'multinodal_seeds', 'multinodal_cross_cell_gating',
                       'multinodal_cross_cell_gate_interval', 'multinodal_features_per_cell',
                       'multinodal_bagged_eps']:
                if k in run_config:
                    config[k] = run_config[k]
        log_dir = benchmark_run_dir
        logger = Logger(log_dir, filename="benchmark.txt")
        logger.log(f"=== BENCHMARK ONLY MODE ===")
        logger.log(f"Run directory: {benchmark_run_dir}")
    else:
        # Create run directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(config.get("out_dir", "logs"), f"{dataset_name}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        # Backup config and source
        with open(os.path.join(log_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        try:
            shutil.copy("model.py", os.path.join(log_dir, "model.py"))
            shutil.copy("train.py", os.path.join(log_dir, "train.py"))
        except Exception:
            pass

        logger = Logger(log_dir)

    # Log config
    logger.log(f"Starting WaveletLM {'Benchmark' if benchmark_only else 'Training'} on {dataset_name.upper()}")
    logger.log(f"Device: {device}, AMP: {use_amp} ({amp_dtype_str})")
    logger.log("")
    max_len = max(len(k) for k in config.keys() if not k.startswith("__"))
    for k, v in config.items():
        if not k.startswith("__"):
            logger.log(f"\t{k.ljust(max_len + 1)}: {v}")
    logger.log("")

    # Load dataset
    train_data, val_data, test_data, enc, bytes_per_token = load_and_encode_dataset(config, logger)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    vocab_size = enc.n_vocab

    # Compute training schedule
    T = config['block_size']
    effective_batch = config['micro_batch_size'] * config.get('grad_accum', 1)
    steps_per_epoch = len(train_data) // (T * effective_batch)
    total_steps = config['epochs'] * steps_per_epoch
    warmup_fraction = config.get('warmup_fraction', 0.3)
    warmup_steps = int(warmup_fraction * total_steps)

    if not benchmark_only:
        logger.log(f"[Schedule] {steps_per_epoch} steps/epoch, {total_steps} total steps")
        logger.log(f"[Schedule] Warmup: {warmup_steps} steps ({warmup_fraction*100:.0f}%)")

    # Build model
    if config.get('multinodal_enabled', False):
        model = MultiNodeWaveletLM(vocab_size, config, device=device)
    else:
        model = WaveletLM(vocab_size, config, device=device)
    model = model.to(device)

    logger.log(f"[Run Folder] {log_dir}")
    total_params, trainable_params = parameter_breakdown(model, config)
    logger.log(f"[Model] {total_params/1e6:.2f}M parameters ({trainable_params/1e6:.2f}M trainable)")

    train_peak_mem = None

    if not benchmark_only:
        # Compile
        if config.get('compile', True) and device == 'cuda':
            logger.log("[Compile] torch.compile enabled")
            model = torch.compile(model)

        # Optimizer
        optimizer_name = config.get('optimizer', 'Adagrad')
        if optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(
                model.parameters(), lr=config['lr'],
                eps=config.get('optimizer_eps', 2e-13))
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config['lr'],
                eps=config.get('optimizer_eps', 1e-8))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        logger.log(f"[Optimizer] {optimizer_name}, lr={config['lr']}, eps={config.get('optimizer_eps')}")

        # GradScaler for fp16 only
        use_scaler = use_amp and amp_dtype_str == 'fp16'
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        # Training state
        get_batch = make_get_batch(train_data, val_data, config, device)
        best_val_loss = float('inf')
        best_epoch = 0
        global_step = 0
        epoch_times = []
        overall_start_time = time.time()

        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # =====================================================================
        # TRAINING LOOP
        # =====================================================================

        for epoch in range(config['epochs']):
            epoch_start = time.time()
            logger.log(f"\n=== EPOCH {epoch+1}/{config['epochs']} ===")

            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
            for step in pbar:
                lr = get_lr(global_step, config, total_steps, warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0

                for micro in range(config.get('grad_accum', 1)):
                    xb, yb = get_batch('train')

                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        _, loss = model(xb, yb)
                        loss = loss / config.get('grad_accum', 1)

                    loss_accum += loss.detach().item()

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Gradient clipping
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                global_step += 1

                pbar.set_postfix({'loss': f"{loss_accum:.4f}", 'lr': f"{lr:.2e}"})

                # Eval
                if global_step % config['eval_interval'] == 0:
                    losses = estimate_loss(model, get_batch, config, device, use_amp, amp_dtype)
                    pbar.clear()
                    logger.log(
                        f"Step {global_step}: "
                        f"train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}\t\t"
                        f"lr={lr:.2e}"
                    )

                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        best_epoch = epoch + 1

                        skip_warmup = config.get('skip_warmup_saves', True)
                        if skip_warmup and global_step < warmup_steps:
                            logger.log(f"New Best Val: {best_val_loss:.4f} (skipping save during warmup)")
                        else:
                            logger.log(f"New Best Val: {best_val_loss:.4f}. Saving...")
                            try:
                                state_dict = model._orig_mod.state_dict()
                            except AttributeError:
                                state_dict = model.state_dict()
                            save_with_retry(state_dict, os.path.join(log_dir, "best_model.pt"))

            # End of epoch
            epoch_duration = time.time() - epoch_start
            epoch_times.append(epoch_duration)

            losses = estimate_loss(model, get_batch, config, device, use_amp, amp_dtype)
            logger.log(f"Epoch {epoch+1} done. Time: {epoch_duration:.1f}s ({epoch_duration/3600:.2f}h)")
            logger.log(f"  Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")

            total_elapsed = time.time() - overall_start_time
            logger.log(f"  Cumulative time: {total_elapsed/3600:.2f}h")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                best_epoch = epoch + 1
                logger.log(f"  New Best Val Loss! Saving model...")
                try:
                    state_dict = model._orig_mod.state_dict()
                except AttributeError:
                    state_dict = model.state_dict()
                save_with_retry(state_dict, os.path.join(log_dir, "best_model.pt"))

        # =====================================================================
        # TRAINING COMPLETE
        # =====================================================================

        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        logger.log("\n=== TRAINING COMPLETE ===")
        logger.log(f"Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
        logger.log(f"Avg Epoch Time: {avg_epoch_time:.1f}s")
        logger.log(f"Total Time: {time.time() - overall_start_time:.1f}s")

        # Record training VRAM before teardown
        if device == 'cuda':
            train_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.log(f"\nTraining Peak VRAM: {train_peak_mem:.0f} MiB")

        # =====================================================================
        # TEARDOWN: free all training state to get clean VRAM for inference
        # =====================================================================

        del model, optimizer, scaler
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # =========================================================================
    # BENCHMARKS & GENERATION (runs after training or in benchmark_only mode)
    # =========================================================================

    # Rebuild model from config for inference
    logger.log("\nLoading best model for benchmarks...")
    best_model_path = os.path.join(log_dir, "best_model.pt")

    if config.get('multinodal_enabled', False):
        model = MultiNodeWaveletLM(vocab_size, config, device=device).to(device)
    else:
        model = WaveletLM(vocab_size, config, device=device).to(device)

    try:
        ckpt = torch.load(best_model_path, map_location=device)
        new_state_dict = {}
        for k, v in ckpt.items():
            new_key = k[10:] if k.startswith("_orig_mod.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.log("Best model loaded.")
    except Exception as e:
        logger.log(f"Could not load best model: {e}")

    model.eval()

    # Non-overlapping benchmark
    results_full = evaluate_full_validation(
        model, test_data, config, logger, device, use_amp, amp_dtype)
    if results_full:
        bpb_full = results_full['bits_per_token'] / bytes_per_token
        logger.log(f"\n[BENCHMARK - Non-overlapping]")
        logger.log(f"  Perplexity: {results_full['perplexity']:.4f}")
        logger.log(f"  BPT: {results_full['bits_per_token']:.4f}")
        logger.log(f"  BPB: {bpb_full:.4f}")
        logger.log(f"  Avg Loss: {results_full['avg_loss']:.4f}")

    # Sliding window benchmark
    results_sw = evaluate_sliding_window(
        model, test_data, config, logger, device, use_amp, amp_dtype)
    if results_sw:
        bpb_sw = results_sw['bits_per_token'] / bytes_per_token
        logger.log(f"\n[BENCHMARK - Sliding Window]")
        logger.log(f"  Perplexity: {results_sw['perplexity']:.4f}")
        logger.log(f"  BPT: {results_sw['bits_per_token']:.4f}")
        logger.log(f"  BPB: {bpb_sw:.4f}")
        logger.log(f"  Avg Loss: {results_sw['avg_loss']:.4f}")
        logger.log(f"  Stride: {results_sw['stride']}, Min Context: {results_sw['min_context']}")

    # Tear down benchmark model entirely and rebuild fresh for generation
    # This ensures inference VRAM matches generate.py (clean CUDA state)
    del model, results_full, results_sw
    import gc; gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Rebuild model from scratch (identical to generate.py's loading path)
    if config.get('multinodal_enabled', False):
        model = MultiNodeWaveletLM(vocab_size, config, device=device).to(device)
    else:
        model = WaveletLM(vocab_size, config, device=device).to(device)
    ckpt = torch.load(best_model_path, map_location=device)
    new_state_dict = {}
    for k, v in ckpt.items():
        new_key = k[10:] if k.startswith("_orig_mod.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    del ckpt, new_state_dict
    model.eval()

    if config.get('quantize_enabled', False):
        q_stats = quantize_model(model, config)
        logger.log(f"\n[Quantization] Original: {q_stats['original_mib']:.1f} MiB -> "
                   f"Quantized: {q_stats['quantized_mib']:.1f} MiB "
                   f"({q_stats['compression_ratio']:.2f}x)")

    if device == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inference_baseline_mem = torch.cuda.memory_allocated() / (1024 ** 2)

    # Generate samples using generate.py functions
    from generate import generate_one, generate_best_of_n, compute_quality_metrics, format_metrics

    prompt = config.get('generation_prompt', 'The history of')
    prompt_ids = enc.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    num_tokens = config.get('num_new_tokens', 512)
    block_size = config['block_size']

    base_kwargs = dict(
        temperature=config.get('temperature', 1.0),
        num_tokens=num_tokens,
        context_len=block_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    # Standard generation (no strategies)
    logger.log("\n=== GENERATION — Standard ===")
    logger.log(f"Prompt: {prompt}")
    if hasattr(model, 'reset_semantic_state'):
        model.reset_semantic_state()
    txt = generate_one(model, enc, input_tensor, **base_kwargs)
    logger.log(f"Generated:\n{txt}\n")

    # Generation with all strategies
    logger.log("=== GENERATION — Strategies ===")
    logger.log(f"Prompt: {prompt}")
    if hasattr(model, 'reset_semantic_state'):
        model.reset_semantic_state()
    strategies_kwargs = dict(
        **base_kwargs,
        top_p=0.85,
        repetition_penalty=1.2,
        entropy_adaptive=True,
        entropy_temp_max=0.9,
    )
    txt, token_ids, stats = generate_one(
        model, enc, input_tensor, return_stats=True, **strategies_kwargs)
    metrics = compute_quality_metrics(token_ids, stats['log_probs'], stats['entropies'])
    logger.log(f"Generated:\n{txt}\n")
    logger.log(f"Metrics: {format_metrics(metrics)}")

    if device == 'cuda':
        inference_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        inference_act_mem = inference_peak_mem - inference_baseline_mem
        logger.log(f"Inference VRAM (approx): {inference_peak_mem:.0f} MiB (model={inference_baseline_mem:.0f} + activations={inference_act_mem:.0f}) — use generate.py for accurate measurement")
        if train_peak_mem is not None:
            logger.log(f"Training Peak VRAM: {train_peak_mem:.0f} MiB")

    logger.close()
    print(f"\nRun directory: {log_dir}")


if __name__ == "__main__":
    train()
