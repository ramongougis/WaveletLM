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

# Silence HuggingFace per-file download progress bars before importing datasets.
# Streaming PG-19 emits one bar per book file (~28K bars total), which spams
# the log without adding useful information. Set before import so the env vars
# are picked up at module-load time.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

# Defensive belt-and-suspenders: also call the programmatic disable in case
# the env vars are read after this module imports (older datasets API).
try:
    import datasets
    datasets.disable_progress_bars()
except (ImportError, AttributeError):
    try:
        datasets.disable_progress_bar()  # singular, older API
    except (ImportError, AttributeError, NameError):
        pass

from model import (
    set_seed, WaveletLM, MultiNodeWaveletLM,
    Logger, parameter_breakdown, quantize_model,
    get_tokenizer, resolve_tokenizer_name, SP_PG19_MODEL_PATH,
)


# ==============================================================================
# SENTENCEPIECE TRAINING (one-time setup for PG-19)
# ==============================================================================

def train_pg19_sentencepiece(cache_dir=".cache", logger=None):
    """Train a 32K-vocab SentencePiece BPE model on the PG-19 train split.

    Called automatically on first PG-19 run if the cached model is missing.
    Output: .cache/pg19_sp32k.model. Single-threaded SP training; sample-capped
    to keep wall time reasonable on first setup.

    Implementation notes:
    - PG-19 entries are full books (often millions of chars). We split each
      book into paragraph-sized chunks at "\n\n" boundaries before writing,
      so SentencePiece sees many short "sentences" rather than a handful of
      multi-million-char lines (which would be truncated by max_sentence_length
      to ~4 KB each, leaving the BPE algorithm with insufficient training data).
    - sample_path is cleaned up in a finally block so a failed SP training
      does not leave a 2 GB orphan.
    - Any partial .model file from a prior failed run is deleted up front so
      the existence check at the top of this function is reliable.
    """
    import sentencepiece as spm

    log = (lambda s: logger.log(s)) if logger is not None else print
    model_prefix = os.path.join(cache_dir, "pg19_sp32k")
    model_path = model_prefix + ".model"
    vocab_path = model_prefix + ".vocab"
    if os.path.exists(model_path):
        return model_path

    # Clear any partial outputs from a previous failed run
    for stale in (model_path, vocab_path):
        if os.path.exists(stale):
            os.remove(stale)

    os.makedirs(cache_dir, exist_ok=True)
    log("[SentencePiece] Training 32K BPE on PG-19 train split (one-time setup).")

    sample_path = os.path.join(cache_dir, "_pg19_sp_train_sample.txt")
    sample_cap_bytes = 2 * 1024 * 1024 * 1024  # 2 GB cap; ample coverage for 32K vocab

    log(f"[SentencePiece] Streaming PG-19 train text (cap {sample_cap_bytes/1e9:.1f} GB)...")
    written = 0
    n_books = 0
    n_chunks = 0
    pbar = tqdm(total=sample_cap_bytes, unit="B", unit_scale=True,
                desc="Streaming PG-19 → SP corpus", mininterval=1.0)
    try:
        with open(sample_path, 'w', encoding='utf-8') as fout:
            # emozilla/pg19 is a parquet mirror of DeepMind's PG-19 (identical
            # text, much faster download). Avoids trust_remote_code and the
            # ~28K-file sequential per-file download.
            ds = load_dataset("emozilla/pg19", split="train", streaming=True)
            for example in ds:
                n_books += 1
                # Split book into paragraph-like chunks so SP gets many short
                # "sentences" rather than one multi-million-char line.
                paragraphs = example["text"].split("\n\n")
                for para in paragraphs:
                    para = para.replace("\n", " ").strip()
                    if len(para) < 20:
                        continue  # skip page numbers, headers, fragments
                    line = para + "\n"
                    fout.write(line)
                    n = len(line.encode("utf-8"))
                    written += n
                    n_chunks += 1
                    pbar.update(n)
                    if written >= sample_cap_bytes:
                        break
                pbar.set_postfix(books=n_books, chunks=n_chunks)
                if written >= sample_cap_bytes:
                    break
        pbar.close()
        log(f"[SentencePiece] Wrote {written/1e9:.2f} GB of training text "
            f"({n_books} books, {n_chunks:,} chunks).")

        log("[SentencePiece] Training BPE model (this may take 5–30 minutes)...")
        spm.SentencePieceTrainer.train(
            input=sample_path,
            model_prefix=model_prefix,
            vocab_size=32000,
            model_type="bpe",
            character_coverage=0.9995,
            normalization_rule_name="nmt_nfkc",
            max_sentence_length=8192,  # raised from 4192 default; chunks comfortably fit
            shuffle_input_sentence=True,
            num_threads=8,
        )
        log(f"[SentencePiece] Trained model saved to {model_path}")
    finally:
        # Always clean up the 2 GB intermediate, even on failure.
        if os.path.exists(sample_path):
            os.remove(sample_path)

    return model_path


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_and_encode_dataset(config, logger):
    """Load dataset via HuggingFace and encode with the auto-selected tokenizer.

    Tokenizer is chosen from config["tokenizer"] (defaults to "auto" which
    routes by dataset). Cache is namespaced by tokenizer so swapping tokenizers
    doesn't reuse stale token IDs.

    Returns:
        train_data, val_data, test_data: torch.long tensors of token IDs
        enc: Tokenizer wrapper (provides .encode, .decode, .vocab_size)
        bytes_per_token: float for BPB calculations
    """
    dataset_name = config.get("dataset", "wikitext-103")
    tokenizer_name = resolve_tokenizer_name(config)

    # Train SP model on first PG-19 run if not yet cached.
    if tokenizer_name == "sentencepiece-pg19-32k" and not os.path.exists(SP_PG19_MODEL_PATH):
        train_pg19_sentencepiece(cache_dir=".cache", logger=logger)

    enc = get_tokenizer(config)
    vocab_size = enc.vocab_size

    logger.log(f"[Tokenizer] {enc.display_name()}")

    # Check for cached encoded tensors (namespaced by tokenizer)
    cache_dir = ".cache"
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{tokenizer_name}.pt")

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

    # Encode from scratch. We map "pg19" to a parquet-based community mirror
    # (emozilla/pg19) instead of the canonical script-based pg19 loader, since
    # the canonical loader downloads ~28K files sequentially at ~4 MB/s, while
    # the parquet mirror downloads in parallel via hf_transfer (10–60× faster
    # for the same ~11 GB). Underlying book text is identical to DeepMind's
    # PG-19 release. trust_remote_code is only needed for the canonical script
    # path; safe for the loaders below.
    if dataset_name == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    elif dataset_name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    elif dataset_name == "pg19":
        ds = load_dataset("emozilla/pg19")
    else:
        ds = load_dataset(dataset_name, trust_remote_code=True)

    def encode_split(split_data, split_name=""):
        # Per-example streaming encode. The previous "\n\n".join(...) approach
        # materialized the entire split as a single Python string before
        # encoding, which OOM'd at PG-19 scale (~11 GB joined string → ~22 GB
        # bytes object → ~56 GB Python list of int objects → 100+ GB peak RAM).
        # Per-example processing keeps peak memory bounded by the largest
        # single example (a few MB for PG-19's largest books), at the cost of
        # one tokenizer call per example. Token-count and byte-count semantics
        # match the original "\n\n".join behavior exactly: the inter-example
        # "\n\n" separator is encoded once and inserted between each pair of
        # examples, and its 2 bytes are added to the byte total.
        #
        # Byte-count caveat preserved from earlier WT-103 verification: none
        # of "\n", "\n\n", or "" joins reproduce Gemini's asserted canonical
        # wiki.test.raw byte count; "\n\n" is the closest (delta ~18K bytes
        # at 1.3 MB scale) and is kept as the self-consistent choice.
        sep_text = "\n\n"
        sep_bytes_len = len(sep_text.encode("utf-8"))
        sep_tokens = enc.encode(sep_text)
        sep_tensor = torch.tensor(sep_tokens, dtype=torch.long)

        chunks = []
        total_bytes = 0
        n_examples = len(split_data)
        desc = f"Encoding {split_name}" if split_name else "Encoding"
        iterator = tqdm(split_data, total=n_examples, desc=desc, unit="ex",
                        mininterval=1.0, smoothing=0.05)
        for i, example in enumerate(iterator):
            example_text = example["text"]
            if i > 0:
                chunks.append(sep_tensor)
                total_bytes += sep_bytes_len
            total_bytes += len(example_text.encode("utf-8"))
            example_tokens = enc.encode(example_text)
            chunks.append(torch.tensor(example_tokens, dtype=torch.long))

        if chunks:
            logger.log(f"[Dataset] Concatenating {len(chunks):,} {split_name} chunks "
                       f"({sum(c.numel() for c in chunks):,} tokens) into final tensor...")
            tokens_tensor = torch.cat(chunks)
        else:
            tokens_tensor = torch.empty(0, dtype=torch.long)
        return tokens_tensor, total_bytes

    logger.log(f"[Dataset] Loading {dataset_name}...")
    logger.log(f"[Dataset] Encoding train split (this may take a while for large datasets)...")
    train_data, _ = encode_split(ds["train"], split_name="train")
    logger.log(f"[Dataset] Train: {len(train_data):,} tokens.")
    logger.log(f"[Dataset] Encoding validation split...")
    val_data, _ = encode_split(ds["validation"], split_name="val")
    logger.log(f"[Dataset] Val: {len(val_data):,} tokens.")
    logger.log(f"[Dataset] Encoding test split...")
    test_data, test_bytes = encode_split(ds["test"], split_name="test")
    logger.log(f"[Dataset] Test: {len(test_data):,} tokens, {test_bytes:,} bytes.")

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
    """Create get_batch closure with cached arange.

    Datasets stay on CPU; per-batch slices are transferred to GPU. Keeping
    multi-GB datasets (e.g. PG-19 ~16 GB tokenized) on CPU is essential —
    moving the full encoded train tensor onto a 32 GB GPU consumes half the
    VRAM and triggers OOM at the first optimizer step. Per-batch transfers
    are tiny (MBS × T × 8 bytes ≈ 16 KB at MBS=8, T=256) and nearly free.
    """
    T = config['block_size']
    _arange_T_cpu = torch.arange(T)[None, :]

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        bs = config['micro_batch_size']
        max_start = len(data) - T - 1
        ix = torch.randint(0, max_start, (bs,))  # CPU
        offsets = ix[:, None] + _arange_T_cpu     # CPU
        x = data[offsets].to(device, non_blocking=True)
        y = data[offsets + 1].to(device, non_blocking=True)
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
# LOG SUMMARY (CSV upsert)
# ==============================================================================

def upsert_log_summary(run_folder, bpb_sliding, ppl_sliding, params,
                       csv_path=os.path.join("logs", "log_summary.csv")):
    """Insert or update a row in logs/log_summary.csv for this run.

    Row columns: folder, bpb_sliding, ppl_sliding, params.
    - If the run_folder already exists, updates its values in place.
    - If not present, appends a new row.
    - Creates the CSV with a header if it doesn't yet exist.
    - Uses a read-modify-rewrite pattern (safe; the file is tiny).

    Fires for both training runs (benchmark_only=false, end-of-training
    benchmark) and benchmark-only runs (rebench). Tolerant of missing values
    — pass None for any field that's unavailable and it's written as empty.
    """
    import csv
    folder = os.path.basename(run_folder.rstrip(os.sep).rstrip('/'))
    fieldnames = ['folder', 'bpb_sliding', 'ppl_sliding', 'params']

    def _fmt(v, spec=None):
        if v is None:
            return ""
        if spec is None:
            return str(v)
        return format(v, spec)

    new_row = {
        'folder': folder,
        'bpb_sliding': _fmt(bpb_sliding, '.4f'),
        'ppl_sliding': _fmt(ppl_sliding, '.4f'),
        'params': f"{params:,}" if params is not None else "",
    }

    # Read existing rows (if any)
    rows = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append({k: r.get(k, '') for k in fieldnames})
        except Exception:
            rows = []  # corrupt file → overwrite

    # Upsert with merge semantics: if the new value is empty but the existing
    # row has a value, preserve it. Prevents a partial/failed benchmark from
    # silently wiping historical numbers for that folder.
    updated = False
    for i, r in enumerate(rows):
        if r.get('folder') == folder:
            merged = {k: r.get(k, '') for k in fieldnames}
            for k, v in new_row.items():
                if v or not merged.get(k):
                    merged[k] = v
            rows[i] = merged
            updated = True
            break
    if not updated:
        rows.append(new_row)

    # Rewrite
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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

    # Disable cross-window decompose-bypass state during eval to prevent
    # disjoint-history injection (MBS>1) and sliding-window causality leaks
    # (MBS=1). Restored before return.
    _prev_cwb = getattr(base_model, 'decompose_bypass_cross_window', False)
    base_model.decompose_bypass_cross_window = False

    # Disable fast-weight inference updates during benchmark. If left on, FwPKM
    # mutates its value_deltas buffer during every forward pass, making BPB
    # non-deterministic and contaminating state across benchmark runs.
    _prev_fwpkm_upd = []
    for layer in base_model.layers if hasattr(base_model, 'layers') else []:
        if hasattr(layer, 'fwpkm') and getattr(layer.fwpkm, 'inference_updates', False):
            _prev_fwpkm_upd.append((layer, layer.fwpkm.inference_updates))
            layer.fwpkm.inference_updates = False

    T = config['block_size']
    batch_size = config['micro_batch_size']
    eval_len = len(eval_data)
    num_windows = (eval_len - 1) // T

    def _restore_flags():
        base_model.decompose_bypass_cross_window = _prev_cwb
        for layer, v in _prev_fwpkm_upd:
            layer.fwpkm.inference_updates = v

    if num_windows == 0:
        logger.log("[WARN] Test data too small for evaluation")
        _restore_flags()
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

    # Restore the flags so subsequent generation uses their configured values
    _restore_flags()
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

    # Disable cross-window decompose-bypass state during eval. Sliding windows
    # overlap by design, so state passed from a later window's end into an
    # earlier window's beginning is a causality leak. Restored before return.
    _prev_cwb = getattr(base_model, 'decompose_bypass_cross_window', False)
    base_model.decompose_bypass_cross_window = False

    # Disable fast-weight inference updates during benchmark (same rationale
    # as evaluate_full_validation).
    _prev_fwpkm_upd = []
    for layer in base_model.layers if hasattr(base_model, 'layers') else []:
        if hasattr(layer, 'fwpkm') and getattr(layer.fwpkm, 'inference_updates', False):
            _prev_fwpkm_upd.append((layer, layer.fwpkm.inference_updates))
            layer.fwpkm.inference_updates = False

    def _restore_flags():
        base_model.decompose_bypass_cross_window = _prev_cwb
        for layer, v in _prev_fwpkm_upd:
            layer.fwpkm.inference_updates = v

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

    # Restore the flags so subsequent generation uses their configured values
    _restore_flags()
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

def save_with_retry(state_dict, path, retries=3, tokenizer_name=None):
    """Save checkpoint with retry logic for filesystem issues.

    If tokenizer_name is provided, the saved object is a wrapper dict
    {'model_state': ..., 'tokenizer_name': ..., 'sentencepiece_model': ...}.
    For SentencePiece tokenizers, the raw `.model` file bytes are embedded in
    the checkpoint so a user downloading only best_model.pt from HF can run
    generate.py out of the box with no extra files. The field is None for
    GPT-2 checkpoints. Bare state_dict is used when tokenizer_name is None
    (legacy format).
    """
    if tokenizer_name is not None:
        sp_bytes = None
        if tokenizer_name == "sentencepiece-pg19-32k":
            try:
                with open(SP_PG19_MODEL_PATH, "rb") as f:
                    sp_bytes = f.read()
            except FileNotFoundError:
                # Training should never hit this (SP model is trained at startup),
                # but fail softly if so — the checkpoint is still usable locally.
                pass
        payload = {
            "model_state": state_dict,
            "tokenizer_name": tokenizer_name,
            "sentencepiece_model": sp_bytes,
        }
    else:
        payload = state_dict

    for attempt in range(retries):
        try:
            torch.save(payload, path)
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
                       'block_size',
                       'pkm_enabled', 'pkm_num_keys', 'pkm_top_k', 'pkm_heads',
                       'fwpkm_enabled', 'fwpkm_num_keys', 'fwpkm_top_k', 'fwpkm_heads',
                       'fwpkm_inference_updates', 'fwpkm_update_lr', 'fwpkm_chunk_size',
                       'wavelet_mode', 'shared_lifting_weights', 'lifting_linear_only',
                       'lifting_hidden_mult', 'lifting_init', 'lifting_dropout',
                       'untied_reconstruction', 'multi_basis_lifting', 'multi_basis_inits',
                       'cross_scale_gating', 'per_scale_mixer_widths',
                       'wavelet_crawl', 'wavelet_crawl_k',
                       'use_mixer_gate', 'mixer_gate_activation',
                       'mixer_depth', 'mixer_depth_stabilizers', 'mixer_depth_residuals',
                       'decompose_bypass', 'decompose_bypass_cross_window', 'decompose_bypass_ema',
                       'learned_residual', 'skip_proj_out', 'looped_blocks', 'looped_blocks_count',
                       'stochastic_depth_rate', 'tie_embedding_to_lm_head',
                       'per_layer_embedding', 'loop_iterations',
                       'stable_parametrization', 'stab_spectral_norm', 'stab_ff_scaling',
                       'stab_embed_scaling', 'stab_proj_out_scaling',
                       'stab_mixer_eps_scaling', 'stab_lifting_level_scaling',
                       'multinodal_enabled', 'multinodal_num_cells', 'multinodal_cell_dim',
                       'multinodal_seeds', 'multinodal_combination',
                       'multinodal_cross_cell_gating',
                       'multinodal_cross_cell_gate_interval', 'multinodal_features_per_cell',
                       'multinodal_bagged_eps']:
                if k in run_config:
                    config[k] = run_config[k]

            # For arch keys missing from the run's config (i.e., the key
            # didn't exist when the run was trained), force them to their
            # feature-off defaults rather than leaking the current root
            # config's value into a model that was trained without them.
            # per_scale_mixer_widths is handled below as a special case
            # because its required length depends on this run's `levels`.
            ARCH_DEFAULTS = {
                'cross_scale_gating': False,
                'wavelet_crawl': False,
                'wavelet_crawl_k': 3,
                'decompose_bypass_ema': False,
                'untied_reconstruction': False,
                'multi_basis_lifting': False,
                'multi_basis_inits': ['haar', 'random'],
                'looped_blocks': False,
                'looped_blocks_count': 8,
                'per_layer_embedding': False,
                'shared_lifting_weights': False,
                'stable_parametrization': False,
                'stab_spectral_norm': False,
                'stab_ff_scaling': False,
                'stab_embed_scaling': False,
                'stab_proj_out_scaling': False,
                'stab_mixer_eps_scaling': False,
                'stab_lifting_level_scaling': False,
                'mixer_depth': 1,
                'mixer_depth_stabilizers': False,
                'mixer_depth_residuals': False,
                'skip_proj_out': False,
                'fwpkm_inference_updates': False,
                'fwpkm_update_lr': 0.01,
                'fwpkm_chunk_size': 64,
                'lifting_linear_only': False,
                'lifting_hidden_mult': 1,
                'lifting_init': 'haar',
                'lifting_dropout': 0.0,
                'use_mixer_gate': True,
                'mixer_gate_activation': 'silu',
                'multinodal_combination': 'average',
                'loop_iterations': 1,
            }
            for k, default in ARCH_DEFAULTS.items():
                if k not in run_config:
                    config[k] = default

            # Special case: per_scale_mixer_widths absent means the run was
            # trained *before* the feature existed. Set to None (NOT all-1.0),
            # because None and all-1.0 produce DIFFERENT model structures:
            #   - None → scale_mixers[s] is a GatedSpectralMixer directly
            #     (state dict keys: scale_mixers.s.mixer.weight)
            #   - all-1.0 → scale_mixers[s] is a PerScaleMixer wrapper
            #     (state dict keys: scale_mixers.s.mixer.mixer.weight)
            # Using all-1.0 caused checkpoint keys to silently fail to load
            # under strict=False, leaving mixer weights randomly initialized.
            # Symptoms were gibberish generation + inflated BPB on older runs.
            if 'per_scale_mixer_widths' not in run_config:
                config['per_scale_mixer_widths'] = None
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
    # Datasets stay on CPU; per-batch slices move to GPU in get_batch / eval
    # functions. Moving the full encoded train tensor to GPU consumes ~16 GB
    # for PG-19 alone (8 bytes/token × 2B tokens) and triggers OOM at the
    # first optimizer step on a 32 GB 5090. Per-batch transfer cost is
    # negligible (~16 KB/batch).
    vocab_size = enc.vocab_size

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
    total_params, trainable_params = parameter_breakdown(model, config, logger=logger)
    logger.log(f"[Model] {total_params/1e6:.2f}M parameters ({trainable_params/1e6:.2f}M trainable)")

    train_peak_mem = None

    if not benchmark_only:
        # Compile
        if config.get('compile', True) and device == 'cuda':
            logger.log("[Compile] torch.compile enabled")
            model = torch.compile(model)

        # Optimizer
        optimizer_name = config.get('optimizer', 'Adagrad')
        weight_decay = config.get('weight_decay', 0.0)
        if optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(
                model.parameters(), lr=config['lr'],
                eps=config.get('optimizer_eps', 2e-13),
                weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config['lr'],
                eps=config.get('optimizer_eps', 1e-8),
                weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        wd_str = f", weight_decay={weight_decay}" if weight_decay > 0 else ""
        logger.log(f"[Optimizer] {optimizer_name}, lr={config['lr']}, eps={config.get('optimizer_eps')}{wd_str}")

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
                            save_with_retry(state_dict, os.path.join(log_dir, "best_model.pt"),
                                            tokenizer_name=enc.name)

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
                save_with_retry(state_dict, os.path.join(log_dir, "best_model.pt"),
                                tokenizer_name=enc.name)

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
        # Unwrap the save_with_retry payload format. Bare state_dict (legacy)
        # is used as-is; the wrapped format embeds tokenizer metadata alongside
        # the weights and must have its 'model_state' key extracted before load.
        # Iterating the wrapper directly with strict=False silently produces a
        # randomly-initialized model — see PG-19 run 2026-04-25 for the exact
        # failure mode (loss ≈ ln(vocab), gibberish generation).
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        new_state_dict = {}
        for k, v in ckpt.items():
            new_key = k[10:] if k.startswith("_orig_mod.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.log("Best model loaded.")
    except Exception as e:
        logger.log(f"Could not load best model: {e}")

    model.eval()

    # Apply PTQ before benchmark so BPB reflects the quantized model
    if config.get('quantize_enabled', False):
        q_stats = quantize_model(model, config)
        logger.log(f"\n[Quantization] Original: {q_stats['original_mib']:.1f} MiB -> "
                   f"Quantized: {q_stats['quantized_mib']:.1f} MiB "
                   f"({q_stats['compression_ratio']:.2f}x)")

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
    bpb_sw = None
    if results_sw:
        bpb_sw = results_sw['bits_per_token'] / bytes_per_token
        logger.log(f"\n[BENCHMARK - Sliding Window]")
        logger.log(f"  Perplexity: {results_sw['perplexity']:.4f}")
        logger.log(f"  BPT: {results_sw['bits_per_token']:.4f}")
        logger.log(f"  BPB: {bpb_sw:.4f}")
        logger.log(f"  Avg Loss: {results_sw['avg_loss']:.4f}")
        logger.log(f"  Stride: {results_sw['stride']}, Min Context: {results_sw['min_context']}")

    # Upsert this run's entry in logs/log_summary.csv. Fires for both
    # benchmark_only=true (rebench) and benchmark_only=false (end of training).
    try:
        upsert_log_summary(
            run_folder=log_dir,
            bpb_sliding=bpb_sw,
            ppl_sliding=(results_sw['perplexity'] if results_sw else None),
            params=total_params,
        )
        logger.log(f"[log_summary.csv] Updated row for {os.path.basename(log_dir)}")
    except Exception as e:
        logger.log(f"[log_summary.csv] Update skipped: {e}")

    # Log mixer depth stabilizer values if enabled
    if config.get('mixer_depth_stabilizers', False) and config.get('mixer_depth', 1) > 1:
        logger.log(f"\n[Mixer Depth Stabilizers]")
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'mixer_depth_alphas'):
                for d in range(len(layer.mixer_depth_alphas)):
                    a = layer.mixer_depth_alphas[d].item()
                    b = layer.mixer_depth_betas[d].item()
                    logger.log(f"  Layer {i}, depth {d}: alpha={a:.4f}, beta={b:.4f}")

    # Release benchmark intermediates, reuse the same (possibly quantized) model
    # for generation. Reset CUDA peak stats so inference VRAM reflects generation
    # only, not benchmark overhead.
    del results_full, results_sw
    import gc; gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inference_baseline_mem = torch.cuda.memory_allocated() / (1024 ** 2)

    # Critical: restore eval mode before generation. The evaluate_* functions
    # above exit with model.train() (to match training-loop conventions), but
    # generation must run with dropout disabled or sampled logits will be noisy.
    model.eval()

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
