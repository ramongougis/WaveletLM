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
import sys
import math
import json
import time
import random
import shutil
import argparse
import datetime
import subprocess

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

def _load_and_encode_pile_streaming(config, enc, logger, cache_dir, tokenizer_name):
    """Stream a token-capped Pile-mirror subset and encode to tensors.

    Unlike the other datasets (downloaded in full), the Pile mirror is
    ~800 GB — this path streams documents over HTTP and stops at the token
    cap, so disk/RAM cost is set by the cap, not the corpus. The cache is
    namespaced by the cap (pile-<N>M_<tokenizer>.pt) so a small smoke-test
    subset can never be mistaken for a full training subset on cache-hit.

    Deterministic INTERLEAVED split carving (i.i.d. across the consumed
    stream): non-empty document i goes to val if i % pile_holdout_stride == 1
    (until pile_val_docs), to test if the residue is 2 or 3 (until
    pile_test_docs), else to train, which accumulates until
    dataset_max_tokens. With stride 6000 and a ~3M-doc stream, holdouts
    spread uniformly over the WHOLE consumed corpus. Same keys -> same
    subset on any machine (stream order is the mirror's file order).
    History (2026-07-17): the original head-of-stream carving (first N docs
    -> val) produced a val slice ~1.2 nats easier than the mixture — a
    non-i.i.d. sample of whatever component leads the first shard. F1 was
    stopped ~2.5h in and restarted under this scheme.
    NOTE: even the interleaved holdout is NOT the canonical Pile val/test —
    sufficient for the fresh-token-twin experiment and self-consistent
    tracking; the "toe-to-toe" Pile campaign (README Release Pipeline) must
    evaluate against the canonical test file when it runs.

    Documents are independent (unlike WikiText's continuous article
    stream), so they are joined with <|endoftext|>, the GPT-2-family
    pretraining convention. The EOT token contributes 0 bytes to the test
    byte count (it has no text representation), slightly deflating
    bytes/token vs a separator-counting scheme — self-consistent and
    documented here.

    Memory: chunks are held int32 throughout (a 4.8B-token subset peaks
    ~2x19 GB at the final cat, ~58 GB with the .long() view at return —
    pod-sized, not laptop-sized; smoke tests use small caps).
    """
    hf_id = config.get("pile_hf_id", "monology/pile-uncopyrighted")
    max_train_tokens = int(config.get("dataset_max_tokens", 5_000_000_000))
    n_val_docs = int(config.get("pile_val_docs", 500))
    n_test_docs = int(config.get("pile_test_docs", 1000))
    stride = int(config.get("pile_holdout_stride", 6000))

    cache_path = os.path.join(
        cache_dir, f"pile-{max_train_tokens}tok-s{stride}_{tokenizer_name}.pt")

    if os.path.exists(cache_path):
        logger.log(f"[Dataset] Loading pile subset from cache ({cache_path})...")
        cached = torch.load(cache_path, weights_only=False)
        train_data = cached['train'].long()
        val_data = cached['val'].long()
        test_data = cached['test'].long()
        test_bytes = cached['test_bytes']
        bytes_per_token = test_bytes / max(1, len(test_data))
        logger.log(f"[Dataset] pile: train={len(train_data):,}, "
                   f"val={len(val_data):,}, test={len(test_data):,} tokens (cached)")
        logger.log(f"[Dataset] Test: {test_bytes:,} bytes, {bytes_per_token:.4f} bytes/token")
        logger.log(f"[Dataset] Vocab size: {enc.vocab_size}")
        return train_data, val_data, test_data, enc, bytes_per_token

    logger.log(f"[Dataset] Streaming {hf_id}: cap={max_train_tokens:,} train tokens, "
               f"val={n_val_docs} docs, test={n_test_docs} docs "
               f"(interleaved holdout, stride={stride})")

    eot = enc.encode("<|endoftext|>", allow_special=True)
    if len(eot) != 1:
        raise ValueError(f"Expected a single EOT token id, got {eot}")

    ds = load_dataset(hf_id, split="train", streaming=True)

    val_chunks, test_chunks, train_chunks = [], [], []
    test_bytes = 0
    train_tokens = 0
    doc_idx = 0
    next_log = 100_000_000

    n_val = n_test = 0
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        chunk = torch.tensor(enc.encode(text) + eot, dtype=torch.int32)
        residue = doc_idx % stride
        if residue == 1 and n_val < n_val_docs:
            val_chunks.append(chunk)
            n_val += 1
        elif residue in (2, 3) and n_test < n_test_docs:
            test_chunks.append(chunk)
            test_bytes += len(text.encode("utf-8"))
            n_test += 1
        else:
            train_chunks.append(chunk)
            train_tokens += chunk.numel()
            if train_tokens >= next_log:
                logger.log(f"[Dataset] ... {train_tokens:,}/{max_train_tokens:,} "
                           f"train tokens ({doc_idx:,} docs; val {n_val}, test {n_test})")
                next_log += 100_000_000
            if train_tokens >= max_train_tokens:
                break
        doc_idx += 1

    train_data = torch.cat(train_chunks) if train_chunks else torch.empty(0, dtype=torch.int32)
    val_data = torch.cat(val_chunks) if val_chunks else torch.empty(0, dtype=torch.int32)
    test_data = torch.cat(test_chunks) if test_chunks else torch.empty(0, dtype=torch.int32)
    del train_chunks, val_chunks, test_chunks
    train_data = train_data[:max_train_tokens].clone()  # exact cap: "1 epoch" == the cap

    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'train': train_data, 'val': val_data, 'test': test_data,
                'test_bytes': test_bytes}, cache_path)
    logger.log(f"[Dataset] Cached to {cache_path}")

    bytes_per_token = test_bytes / max(1, len(test_data))
    logger.log(f"[Dataset] pile: train={len(train_data):,}, "
               f"val={len(val_data):,}, test={len(test_data):,} tokens")
    logger.log(f"[Dataset] Test: {test_bytes:,} bytes, {bytes_per_token:.4f} bytes/token")
    logger.log(f"[Dataset] Vocab size: {enc.vocab_size}")
    return train_data.long(), val_data.long(), test_data.long(), enc, bytes_per_token


# Canonical word-level normalization units for cross-paper PPL comparison.
# Per-token perplexity depends on the tokenizer and is NOT comparable across
# vocabularies; the field's canonical WT-103/PG-19 numbers are word-level:
# PPL_word = exp(total NLL / canonical word count) — tokenizer-invariant
# (2026-07-20 normalization audit; see the README '‡'/'§' table footnotes).
# wikitext: canonical count = whitespace tokens + one <eos> per line
#   (test 245,569 = 241,211 words + 4,358 lines; matches Merity et al. and
#   the Megatron-LM evaluation convention, arXiv:1909.08053).
# pg19: Rae et al. (arXiv:1911.05507) Sec. 4.2 defines the metric; test
#   n_words = 6,966,499.
# BPB requires no such correction (byte normalization is tokenizer-immune).
CANONICAL_TEST_WORDS = {
    "wikitext-103": 245_569,
    "wikitext-2": 245_569,  # shares the WT-103 validation/test sets
    "pg19": 6_966_499,
}


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

    # The Pile is stream-only (too large to download); it manages its own
    # cap-namespaced cache inside the helper and returns early.
    if dataset_name == "pile":
        return _load_and_encode_pile_streaming(config, enc, logger,
                                               cache_dir=".cache",
                                               tokenizer_name=tokenizer_name)

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

    Two sampling modes:
    - Random (default): each block starts at a uniformly random corpus
      position. Standard nanoGPT-style. ~63% coverage at 1 epoch from
      `1 - exp(-1)`; ~99.3% at 5 epochs.
    - Sequential (`sequential_blocks=true`): blocks proceed in corpus order
      with stride = block_size, no shuffling, no overlap. Each token visited
      exactly once per epoch as a target. Two flavors automatically selected
      from the existing MBS / GA config:
        - MBS=8, GA=1 ("Rainman"): 8 parallel streams advancing through the
          corpus together. Each stream starts at corpus position k * (N/8).
        - MBS=1, GA=8 ("pure sequential"): one stream, each grad-accum
          substep produces the next consecutive block before optimizer.step().
      Validation always uses random sampling regardless of mode (sequential
      val would deterministically resample the same blocks each eval call).

    BBCE (`bbce_enabled=true`): each batch reads `block_size_compressed`
    consecutive tokens per element instead of `block_size`. Targets `y` are
    only the last `block_size/2` next-token positions of that window (the
    uncompressed half is the loss-bearing region; the compressed half is
    context). The model's BBCEPreprocessor handles mean-pooling the
    compressed half. Currently only supported with random sampling
    (sequential+BBCE+caching is Phase 2).
    """
    T = config['block_size']
    _arange_T_cpu = torch.arange(T)[None, :]
    bs = config['micro_batch_size']
    use_sequential = config.get('sequential_blocks', False)
    use_bbce = config.get('bbce_enabled', False)

    if use_bbce:
        if use_sequential:
            raise ValueError(
                "bbce_enabled=true with sequential_blocks=true is not yet "
                "supported (Phase 2). Use random sampling for the initial BBCE "
                "sweep."
            )
        Tc = int(config['block_size_compressed'])
        if Tc < T:
            raise ValueError(
                f"block_size_compressed ({Tc}) must be >= block_size ({T})"
            )
        num_uncompressed = T // 2  # targets are only for the last T/2 positions
        # x window: [start, start + Tc), tokens 0..Tc-1
        # y window: [start + Tc - num_uncompressed, start + Tc)
        # i.e. the next-token target for x[Tc - num_uncompressed - 1] is
        # at corpus position start + Tc - num_uncompressed; this lines up
        # with the last num_uncompressed positions of the x window when
        # shifted by 1 (next-token prediction).
        _arange_Tc_cpu = torch.arange(Tc)[None, :]
        _arange_y_cpu = torch.arange(
            Tc - num_uncompressed, Tc
        )[None, :]  # offsets for the last num_uncompressed positions

    # Persistent train-position counters for sequential mode. Mutated in-place
    # by the closure on each call; modulo wraps at end of corpus.
    train_max_start = len(train_data) - T - 1
    if use_sequential:
        if bs == 1:
            # Pure sequential: one counter
            train_positions = [0]
        else:
            # Rainman: bs streams strided across the corpus
            stride = train_max_start // bs
            train_positions = [k * stride for k in range(bs)]
    else:
        train_positions = None

    def get_batch(split):
        if use_bbce:
            # For BBCE val, fall back to train_data when val_data is shorter
            # than block_size_compressed (typical: val=251K, bc=1M+). The
            # resulting "val" loss is then a train-loss proxy on independent
            # random samples — still useful for tracking best_val during
            # training. Test-set BPB is skipped separately in train.py.
            if split == 'val' and len(val_data) - Tc - 1 < 0:
                data = train_data
            else:
                data = train_data if split == 'train' else val_data

            max_start = len(data) - Tc - 1
            ix = torch.randint(0, max_start, (bs,))  # CPU
            x_offsets = ix[:, None] + _arange_Tc_cpu  # (bs, Tc)
            y_offsets = ix[:, None] + _arange_y_cpu + 1  # (bs, num_uncompressed)
            x = data[x_offsets].to(device, non_blocking=True)  # (bs, Tc) raw tokens
            y = data[y_offsets].to(device, non_blocking=True)  # (bs, num_uncompressed)
            return x, y

        data = train_data if split == 'train' else val_data

        max_start = len(data) - T - 1

        if use_sequential and split == 'train':
            ix = torch.tensor(train_positions, dtype=torch.long)
            # Advance positions by T, wrapping at max_start so a multi-epoch
            # run continues seamlessly past the first full pass.
            for i in range(len(train_positions)):
                train_positions[i] = (train_positions[i] + T) % max_start
        else:
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
    # Force MBS=1 for benchmarks. The benchmark path runs in eager mode (no
    # torch.compile() — see the `if not benchmark_only:` guard around the
    # compile call), so per-batch activation memory is much higher than at
    # training time even at the same nominal MBS. With T=16384 and L=2,
    # a benchmark at MBS=8 OOMs on a 32 GiB GPU even though training at the
    # same MBS fit comfortably (because compile fused away most of the
    # intermediate-tensor materialization). MBS=1 is safe at any T/depth and
    # only adds ~30 seconds across the 17 + 34 = 51 total benchmark windows.
    batch_size = 1
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
        # NOTE: per-token losses are kept in fp16 (cheap, well within range)
        # but cast to fp32 BEFORE .sum(). Without the fp32 cast, summing a
        # fp16 loss tensor of length T overflows fp16 (max 65504) when T is
        # large and per-token loss is non-trivial — e.g. T=16384 × 4.5 nats
        # ≈ 73,500 → inf BPB despite finite per-token losses and finite
        # logits. We cast the small loss tensor instead of the giant logits
        # tensor to avoid OOM at large MBS × T (logits in fp32 would be 24+
        # GiB at MBS=8, T=16384, V=50257).
        loss_per_token = F.cross_entropy(
            logits.view(-1, V), Y.view(-1).long(), reduction='none')
        total_loss += loss_per_token.float().sum().item()
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
    # Force MBS=1 for benchmarks — see companion comment in
    # evaluate_full_validation. Same eager-mode + activation-memory rationale.
    batch_size = 1
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
            # See companion comment in evaluate_full_validation: cast the
            # SMALL per-token loss tensor (not the large logits tensor) to
            # fp32 before .sum() so the accumulator can't overflow fp16.
            loss_per_token = F.cross_entropy(
                logits_scored, targets_scored.long(), reduction='none')
            total_loss += loss_per_token.float().sum().item()
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


@torch.no_grad()
def evaluate_bbce(model, eval_data, config, logger, device, use_amp, amp_dtype,
                  max_windows=None, pad_id=0):
    """BBCE-aware test benchmark.

    Each window scores `block_size/2` consecutive next-token predictions in
    eval_data (the supervised half of BBCE's bisected output). Context is
    `block_size_compressed` tokens ending immediately before the supervised
    region; when the available real context is shorter than
    `block_size_compressed` (typical for the larger BBCE configs where the
    test set is smaller than the context window), the missing prefix is
    left-padded with `pad_id`. Every supervised position in eval_data thus
    gets scored, regardless of whether the model's full long context fits
    inside the test set — the configs that need padding are clearly logged
    so the comparison is honest.

    Coverage is non-overlapping at the supervised level: target windows are
    placed at corpus positions stride=block_size/2 apart. By default
    (max_windows=None) every natural window is scored — same convention as
    the non-BBCE sliding-window benchmark which covers the whole test set.
    If max_windows is set to a positive integer and there are more natural
    windows, they're uniformly subsampled across the full test set (not
    front-truncated) so the benchmark stays statistically representative.

    Returns the same dict shape as evaluate_full_validation / sliding for
    consistent downstream logging.
    """
    model.eval()
    base_model = getattr(model, "_orig_mod", model)
    if hasattr(base_model, 'reset_semantic_state'):
        base_model.reset_semantic_state()

    # Disable cross-window state and FwPKM inference updates during eval,
    # same rationale as evaluate_full_validation / evaluate_sliding_window.
    _prev_cwb = getattr(base_model, 'decompose_bypass_cross_window', False)
    base_model.decompose_bypass_cross_window = False
    _prev_fwpkm_upd = []
    for layer in base_model.layers if hasattr(base_model, 'layers') else []:
        if hasattr(layer, 'fwpkm') and getattr(layer.fwpkm, 'inference_updates', False):
            _prev_fwpkm_upd.append((layer, layer.fwpkm.inference_updates))
            layer.fwpkm.inference_updates = False

    def _restore_flags():
        base_model.decompose_bypass_cross_window = _prev_cwb
        for layer, v in _prev_fwpkm_upd:
            layer.fwpkm.inference_updates = v

    from tools.bbce import pad_idx_for_bbce

    bs = int(config['block_size'])
    bsc = int(config['block_size_compressed'])
    half = bs // 2
    eval_len = len(eval_data)

    # Window math, matched to the training data loader (make_get_batch BBCE
    # branch) so the supervised slice means the same thing here and there:
    #   training: x = data[start : start + Tc],
    #             y = data[start + Tc - half + 1 : start + Tc + 1]
    # Let ctx_end = start + Tc (exclusive). Then x covers corpus positions
    # [ctx_end - Tc, ctx_end) and y covers corpus positions
    # [ctx_end - half + 1, ctx_end + 1) — i.e. next-token predictions for
    # the LAST `half` input positions, offset by 1.
    # We parameterize by `target_start = ctx_end - half`, so:
    #   x covers [target_start + half - Tc, target_start + half), left-padded
    #     to length Tc if target_start + half - Tc < 0
    #   y covers [target_start + 1, target_start + half + 1).
    # Stride = half for non-overlapping supervised coverage. Constraint:
    # target_start + half + 1 <= eval_len → target_start <= eval_len - half - 1.
    max_target_start = eval_len - half - 1
    if max_target_start < 0:
        logger.log("[BBCE BENCHMARK] eval_data shorter than block_size/2 + 1; cannot benchmark.")
        _restore_flags()
        model.train()
        return None

    all_target_starts = list(range(0, max_target_start + 1, half))
    num_natural = len(all_target_starts)

    # Default behavior: score every natural window (full test-set coverage,
    # matching the non-BBCE sliding-window benchmark). If max_windows is set
    # to a positive integer below the natural count, uniformly subsample
    # across the test set (representative, not front-truncated).
    if max_windows is not None and max_windows > 0 and num_natural > max_windows:
        step = num_natural / max_windows
        idxs = [int(i * step) for i in range(max_windows)]
        target_starts = [all_target_starts[i] for i in idxs]
    else:
        target_starts = all_target_starts

    # How many windows will need left-padding? Padding kicks in whenever
    # target_start + half - bsc < 0 → target_start < bsc - half. For larger
    # bsc relative to the test set, this can be every window.
    n_padded = sum(1 for s in target_starts if (s + half - bsc) < 0)

    cap_str = "uncapped" if (max_windows is None or max_windows <= 0) else f"max={max_windows}"
    logger.log(
        f"\n[BBCE BENCHMARK] block_size={bs}, block_size_compressed={bsc}, "
        f"stride={half}, {len(target_starts)} of {num_natural} natural windows "
        f"({cap_str}), {n_padded} require left-padding"
    )

    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(target_starts, desc="BBCE Benchmark")
    for target_start in pbar:
        ctx_end = target_start + half  # exclusive (corresponds to training's ix + Tc)
        ctx_start = ctx_end - bsc
        real_ctx_start = max(0, ctx_start)
        real_ctx = eval_data[real_ctx_start:ctx_end].unsqueeze(0)  # (1, T_real)
        x = pad_idx_for_bbce(real_ctx.long(), bsc, pad_id=pad_id)
        # Next-token targets, offset by 1 from the context's trailing half:
        # logits[:, -half:, :][k] predicts data[ctx_end - half + k + 1] =
        # data[target_start + 1 + k].
        y = eval_data[target_start + 1:target_start + half + 1].unsqueeze(0).long()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(x, targets=None)  # (1, bs, V)

        logits_scored = logits[:, -half:, :]
        V = logits_scored.size(-1)
        loss_per_token = F.cross_entropy(
            logits_scored.reshape(-1, V), y.view(-1), reduction='none')
        total_loss += loss_per_token.float().sum().item()
        total_tokens += y.numel()

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
        'total_scored_tokens': total_tokens,
        'num_windows': len(target_starts),
        'num_natural_windows': num_natural,
        'num_padded_windows': n_padded,
        'stride': half,
        'block_size_compressed': bsc,
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


def save_resume_checkpoint(path, model, optimizers, scaler, *, epoch, step_in_epoch,
                           global_step, best_val_loss, best_epoch, epoch_times,
                           elapsed_seconds, steps_per_epoch, total_steps, retries=3):
    """Full-state training checkpoint for crash/preemption recovery.

    Written to `last_checkpoint.pt` (overwritten in place) every epoch and,
    optionally, every `checkpoint_interval_steps` global steps. Contains
    everything needed for an exact continuation: model weights, optimizer
    state (Adagrad accumulators!), GradScaler state, step/epoch counters,
    best-val tracking, wall-clock elapsed, and all RNG streams. Batches come
    from `torch.randint` on the CPU generator, so restoring RNG state
    reproduces the remaining data order exactly — a resumed run is
    step-for-step identical to one that never stopped (modulo GPU op
    nondeterminism).

    The write is atomic (tmp file + os.replace): a crash mid-write leaves the
    previous valid checkpoint intact rather than a truncated file.
    `epoch`/`step_in_epoch` name the NEXT position to run: an epoch-end save
    stores (epoch+1, 0); a mid-epoch save stores (epoch, step+1).
    """
    base = getattr(model, '_orig_mod', model)
    payload = {
        'resume_format_version': 1,
        'model_state': base.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scaler_state': scaler.state_dict(),
        'epoch': epoch,
        'step_in_epoch': step_in_epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'epoch_times': list(epoch_times),
        'elapsed_seconds': elapsed_seconds,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'rng_torch_cpu': torch.get_rng_state(),
        'rng_torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'rng_python': random.getstate(),
    }
    tmp_path = path + '.tmp'
    for attempt in range(retries):
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError):
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
    parser.add_argument('--mixer_recurrence_inference_steps', '--infer_n', type=int, default=None,
                        help='Eval/benchmark-only: run this many outer mixer-recurrence '
                             'steps instead of the trained count (train deep, infer shallow). '
                             'Clamped to [1, trained]. Omit to use the trained depth.')
    parser.add_argument('--mixer_transform', type=str, default=None,
                        choices=['fwht', 'identity', 'dht', 'dct', 'learned_butterfly'],
                        help='Per-scale mixer transform for the ablation: fwht (default), '
                             'identity (no transform), dht (Hartley), dct, or learned_butterfly '
                             '(learned orthogonal). Injected into the run config and persisted '
                             'to the saved config.json.')
    parser.add_argument('--eval_block_size', type=int, default=None,
                        help='Length-generalization eval ONLY (benchmark_only mode): evaluate the '
                             'trained model at this block size instead of its trained one. No effect '
                             'during training, and a no-op when equal to the trained size — the '
                             'existing benchmark path is byte-for-byte unchanged. Architecture '
                             '(levels/widths) stays from the saved config; only the eval window '
                             'changes. Requires bbce_enabled=false.')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found!")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Mixer-transform ablation override (training mode): inject before model
    # build so it's used and persisted to the saved run config.json. In
    # benchmark_only mode the run's saved config is the source of truth, so the
    # transform is read from there and this CLI flag is unnecessary.
    if args.mixer_transform is not None:
        config['mixer_transform'] = args.mixer_transform

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config.get('allow_tf32', True) and device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Benchmark-only mode: skip training, load checkpoint, run benchmarks + generation
    benchmark_only = config.get('benchmark_only', False)
    benchmark_run_dir = config.get('benchmark_run_dir', '')
    # Resume mode: continue a crashed/preempted run in place from its
    # last_checkpoint.pt. Same source-of-truth rule as benchmark_only: the
    # run's saved config.json wholesale-replaces the root config, except the
    # pointer key itself. Mutually exclusive with benchmark_only.
    resume_run_dir = config.get('resume_run_dir', '')
    if benchmark_only and resume_run_dir:
        raise ValueError("resume_run_dir and benchmark_only are mutually exclusive")
    dataset_name = config.get("dataset", "wikitext-103")

    if benchmark_only:
        if not benchmark_run_dir:
            raise ValueError("benchmark_only=true requires benchmark_run_dir to be set")
        # Rule: in benchmark_only mode, the run's saved config.json is the
        # source of truth for EVERY config key except the two pointer-keys
        # `benchmark_only` and `benchmark_run_dir` (which by necessity come
        # from the root config — they're how the user tells train.py to
        # enter benchmark mode and which directory to point at). Anything
        # else getting pulled from root would risk silently mismatching the
        # forward-pass conditions of the training run we're benchmarking
        # (different AMP dtype, different seed for generation, etc.).
        run_config_path = os.path.join(benchmark_run_dir, "config.json")
        if os.path.exists(run_config_path):
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
            # Wholesale replace: the run's config becomes the new config,
            # except we re-pin the pointer-keys so the rest of train.py
            # still knows it's in benchmark mode.
            config = dict(run_config)
            config['benchmark_only'] = True
            config['benchmark_run_dir'] = benchmark_run_dir

            # For arch keys missing from the run's config (i.e., the key
            # didn't exist when the run was trained), force them to their
            # feature-off defaults rather than leaving them undefined for
            # the model constructor. per_scale_mixer_widths is handled
            # below as a special case because its required length depends
            # on this run's `levels`.
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
                if k not in config:
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

        # Length-generalization eval (opt-in, additive — does NOT touch the default path).
        # When --eval_block_size is given AND differs from the trained block_size, evaluate
        # the *unchanged* model at the new window: only the eval window changes; every
        # architecture key stays from the checkpoint's saved config. WaveletLM has no
        # positional embedding (a-trous levels are length-independent) and, with bbce off,
        # no block_size-dependent buffer, so a model trained at one block size runs at a
        # larger one with no mismatch. No-op when the arg is absent or equals the trained
        # size, so the existing benchmark stays byte-for-byte unchanged.
        if args.eval_block_size is not None and int(args.eval_block_size) != int(config['block_size']):
            if config.get('bbce_enabled', False):
                raise ValueError("--eval_block_size (length-gen eval) is not supported with bbce_enabled=true")
            _trained_bs = int(config['block_size'])
            config['block_size'] = int(args.eval_block_size)
            logger.log(f"[LENGTH-GEN] Eval block_size overridden: trained={_trained_bs} -> "
                       f"eval={config['block_size']} (architecture unchanged; sliding-window BPB "
                       f"uses the new window).")
        logger.log(f"Run directory: {benchmark_run_dir}")
        # Inference-time mixer-recurrence depth override (eval-only). Injected as
        # a NEW config key on top of the run's own config — it never modifies a
        # trained parameter, only tells the eval forward to run fewer outer
        # recurrence steps ("train deep, infer shallow"). See
        # model.py WaveletLMBlock.mixer_recurrence_inference_steps.
        if args.mixer_recurrence_inference_steps is not None:
            config['mixer_recurrence_inference_steps'] = int(args.mixer_recurrence_inference_steps)
            logger.log(f"[Override] mixer_recurrence_inference_steps = "
                       f"{config['mixer_recurrence_inference_steps']} "
                       f"(eval-only recurrence-depth truncation)")
    elif resume_run_dir:
        # Resume an interrupted run in place. The run's saved config.json is
        # the source of truth for every key except the pointer (same rule as
        # benchmark_only) — anything else pulled from root would silently
        # change the training conditions mid-run. The run dir, config.json,
        # and source backups are left untouched; the logger appends.
        resume_ckpt_path = os.path.join(resume_run_dir, "last_checkpoint.pt")
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(
                f"resume_run_dir set but no last_checkpoint.pt in {resume_run_dir} "
                f"(the run predates resume support, or never completed an epoch / "
                f"checkpoint interval)")
        run_config_path = os.path.join(resume_run_dir, "config.json")
        if not os.path.exists(run_config_path):
            raise FileNotFoundError(f"No config.json in {resume_run_dir}")
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        config = dict(run_config)
        config['resume_run_dir'] = resume_run_dir

        log_dir = resume_run_dir
        logger = Logger(log_dir, append=True)
        logger.log("")
        logger.log("=== RESUME MODE ===")
        logger.log(f"Run directory: {resume_run_dir}")
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

    # Seed and AMP must be read AFTER any benchmark_only config replacement,
    # so they use the run's saved values (not whatever the root config has).
    set_seed(config['seed'])
    use_amp = config.get('use_amp', True)
    amp_dtype_str = config.get('amp_dtype', 'fp16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bf16' else torch.float16
    # Refresh dataset_name in case the run config overrode it
    dataset_name = config.get("dataset", "wikitext-103")

    # Log config
    logger.log(f"Starting WaveletLM {'Benchmark' if benchmark_only else 'Training'} on {dataset_name.upper()}")
    logger.log(f"Device: {device}, AMP: {use_amp} ({amp_dtype_str})")
    logger.log("")
    max_len = max(len(k) for k in config.keys() if not k.startswith("__"))
    for k, v in config.items():
        if not k.startswith("__"):
            logger.log(f"\t{k.ljust(max_len + 1)}: {v}")
    logger.log("")

    if device == 'cuda':
        try:
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            ).strip()
            if gpu_name:
                logger.log(f"[GPU] {gpu_name}")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Load dataset
    train_data, val_data, test_data, enc, bytes_per_token = load_and_encode_dataset(config, logger)
    # Datasets stay on CPU; per-batch slices move to GPU in get_batch / eval
    # functions. Moving the full encoded train tensor to GPU consumes ~16 GB
    # for PG-19 alone (8 bytes/token × 2B tokens) and triggers OOM at the
    # first optimizer step on a 32 GB 5090. Per-batch transfer cost is
    # negligible (~16 KB/batch).
    vocab_size = enc.vocab_size

    # Compute training schedule. The stride per step is the number of *new
    # corpus tokens* the step consumes for supervision — i.e. the number of
    # loss-bearing (next-token) positions per micro-batch sample. Without
    # BBCE every position in a block_size-sized window is supervised, so the
    # stride equals block_size. Under BBCE only the uncompressed half is
    # supervised (the compressed half is reused context, an architectural
    # add-on), so the stride is block_size/2. Using the supervised stride
    # keeps "1 epoch" defined as one full pass over the corpus's supervised
    # positions, identically across BBCE and non-BBCE — and per-epoch
    # supervised-token coverage is the full corpus in both cases.
    T = config['block_size']
    if config.get('bbce_enabled', False):
        active_stride = T // 2
    else:
        active_stride = T
    effective_batch = config['micro_batch_size'] * config.get('grad_accum', 1)
    steps_per_epoch = len(train_data) // (active_stride * effective_batch)
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
        # Frozen-wavelet transfer (Release Pipeline item): import a trained
        # shared lifting (incl. crawl dilation logits — they live in the same
        # module) from a donor checkpoint, optionally freezing it. Donor must
        # share the lifting geometry (C, levels/schedule, K, hidden_mult);
        # strict load enforces this. Key matching is done by suffix against
        # the live module's own state_dict so it is robust to whether the
        # donor registered the module as shared_lifting or layers.N.lifting_wavelet.
        lifting_src = config.get('lifting_import_checkpoint', '')
        if lifting_src:
            donor = torch.load(lifting_src, map_location=device)
            if isinstance(donor, dict) and 'model_state' in donor:
                donor = donor['model_state']
            donor = {(k[10:] if k.startswith('_orig_mod.') else k): v
                     for k, v in donor.items()}
            lw = model.layers[0].lifting_wavelet
            picked = {}
            for kk in lw.state_dict():
                cands = [v for k, v in donor.items()
                         if k.endswith('lifting_wavelet.' + kk)
                         or k.endswith('shared_lifting.' + kk)]
                if not cands:
                    raise KeyError(f"[Lifting import] donor lacks lifting key: {kk}")
                picked[kk] = cands[0]
            lw.load_state_dict(picked, strict=True)
            n_imp = sum(v.numel() for v in picked.values())
            logger.log(f"[Lifting] Imported {n_imp/1e6:.2f}M lifting params "
                       f"from {lifting_src}")
            if config.get('lifting_freeze', False):
                for p in lw.parameters():
                    p.requires_grad = False
                logger.log("[Lifting] FROZEN (requires_grad=False; optimizer "
                           "skips grad-less params)")

        # Associative-memory-bypass ADAPTER (plans/associative_memory_bypass.md):
        # load a pretrained checkpoint that has NO assoc_* params into this
        # bypass-equipped model. The AMB modules keep their fresh identity-init
        # (zero-init out_proj -> the model starts EXACTLY as the donor), then the
        # whole model (or just the AMB, if freeze_core) fine-tunes to install
        # in-context retrieval. The native path is simply training from scratch
        # with associative_bypass_enabled=true; this is the cheaper "install into
        # an existing checkpoint" path.
        amb_src = config.get('associative_bypass_adapter_checkpoint', '')
        if amb_src:
            if not config.get('associative_bypass_enabled', False):
                raise ValueError("associative_bypass_adapter_checkpoint requires "
                                 "associative_bypass_enabled=true")
            donor = torch.load(amb_src, map_location=device)
            if isinstance(donor, dict) and 'model_state' in donor:
                donor = donor['model_state']
            donor = {(k[10:] if k.startswith('_orig_mod.') else k): v
                     for k, v in donor.items()}
            _base = getattr(model, '_orig_mod', model)
            missing, unexpected = _base.load_state_dict(donor, strict=False)
            # The donor may ONLY lack the fresh AMB tensors; anything else missing,
            # or any unexpected key, means a config/architecture mismatch -> fail loud.
            bad_missing = [k for k in missing if '.assoc_' not in k]
            if bad_missing or unexpected:
                raise KeyError(
                    f"[AMB adapter] checkpoint mismatch beyond the AMB modules: "
                    f"unexpected={list(unexpected)[:6]}; non-AMB missing={bad_missing[:6]}")
            logger.log(f"[AMB adapter] loaded donor {amb_src}; {len(missing)} fresh "
                       f"AMB tensors kept at identity-init (model starts == donor).")
            if config.get('associative_bypass_adapter_freeze_core', False):
                n_train = n_frozen = 0
                for name, p in _base.named_parameters():
                    if '.assoc_' in name:
                        n_train += p.numel()
                    else:
                        p.requires_grad = False
                        n_frozen += p.numel()
                logger.log(f"[AMB adapter] CORE FROZEN: training only "
                           f"{n_train/1e6:.3f}M AMB params ({n_frozen/1e6:.2f}M frozen).")

        # Compile
        if config.get('compile', True) and device == 'cuda':
            compile_mode = config.get('compile_mode', 'default')
            logger.log(f"[Compile] torch.compile enabled (mode={compile_mode})")
            model = torch.compile(model, mode=compile_mode)

        # Optimizer
        optimizer_name = config.get('optimizer', 'Adagrad')
        weight_decay = config.get('weight_decay', 0.0)
        if optimizer_name == 'Adagrad':
            adagrad_initial_accum = config.get('optimizer_initial_accumulator_value', 0)
            adagrad_lr_decay = config.get('optimizer_lr_decay', 0)
            optimizers = [torch.optim.Adagrad(
                model.parameters(), lr=config['lr'],
                eps=config.get('optimizer_eps', 2e-13),
                weight_decay=weight_decay,
                initial_accumulator_value=adagrad_initial_accum,
                lr_decay=adagrad_lr_decay)]
            wd_str = f", weight_decay={weight_decay}" if weight_decay > 0 else ""
            logger.log(f"[Optimizer] {optimizer_name}, lr={config['lr']}, eps={config.get('optimizer_eps')}, "
                       f"initial_accumulator_value={adagrad_initial_accum}, lr_decay={adagrad_lr_decay}{wd_str}")
        elif optimizer_name == 'AdamW':
            adamw_betas = tuple(config.get('optimizer_betas', [0.9, 0.999]))
            adamw_amsgrad = config.get('optimizer_amsgrad', False)
            optimizers = [torch.optim.AdamW(
                model.parameters(), lr=config['lr'],
                betas=adamw_betas,
                eps=config.get('optimizer_eps', 1e-8),
                weight_decay=weight_decay,
                amsgrad=adamw_amsgrad)]
            wd_str = f", weight_decay={weight_decay}" if weight_decay > 0 else ""
            logger.log(f"[Optimizer] {optimizer_name}, lr={config['lr']}, betas={adamw_betas}, eps={config.get('optimizer_eps')}, amsgrad={adamw_amsgrad}{wd_str}")
        elif optimizer_name == 'Muon':
            # Hybrid Muon + AdamW. Per torch.optim.Muon docs, Muon strictly
            # requires 2D parameters; biases / norms / embeddings / LM head /
            # 3D+ tensors (e.g. FwPKM product-key sub-keys of shape
            # [1, num_subkeys, C/2]) should use AdamW. Split params accordingly.
            muon_params, adamw_params = [], []
            muon_names, adamw_names = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                lname = name.lower()
                is_embedding = 'embedding' in lname or 'lm_head' in lname
                if p.ndim == 2 and not is_embedding:
                    muon_params.append(p)
                    muon_names.append(name)
                else:
                    adamw_params.append(p)
                    adamw_names.append(name)
            muon_optim = torch.optim.Muon(
                muon_params,
                lr=config['lr'],
                weight_decay=config.get('muon_weight_decay', 0.1),
                momentum=config.get('muon_momentum', 0.95),
                nesterov=config.get('muon_nesterov', True),
                ns_coefficients=tuple(config.get('muon_ns_coefficients', [3.4445, -4.775, 2.0315])),
                eps=config.get('muon_eps', 1e-7),
                ns_steps=config.get('muon_ns_steps', 5),
                adjust_lr_fn=config.get('muon_adjust_lr_fn', None),
            )
            adamw_optim = torch.optim.AdamW(
                adamw_params,
                lr=config.get('muon_adamw_lr', config['lr']),
                weight_decay=config.get('muon_adamw_weight_decay', config.get('muon_weight_decay', 0.1)),
                eps=config.get('muon_adamw_eps', 1e-8),
            )
            optimizers = [muon_optim, adamw_optim]
            logger.log(
                f"[Optimizer] Muon+AdamW hybrid: "
                f"Muon on {len(muon_params)} 2D hidden tensors "
                f"(lr={config['lr']}, wd={config.get('muon_weight_decay', 0.1)}, "
                f"momentum={config.get('muon_momentum', 0.95)}, eps={config.get('muon_eps', 1e-7)}, "
                f"ns_steps={config.get('muon_ns_steps', 5)}), "
                f"AdamW on {len(adamw_params)} other tensors "
                f"(lr={config.get('muon_adamw_lr', config['lr'])}, "
                f"wd={config.get('muon_adamw_weight_decay', config.get('muon_weight_decay', 0.1))})"
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # GradScaler for fp16 only
        use_scaler = use_amp and amp_dtype_str != 'bf16'
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        # Training state
        get_batch = make_get_batch(train_data, val_data, config, device)
        best_val_loss = float('inf')
        best_epoch = 0
        global_step = 0
        epoch_times = []
        overall_start_time = time.time()
        checkpoint_interval = int(config.get('checkpoint_interval_steps', 0))

        # Resume: restore the full training state saved by
        # save_resume_checkpoint. Loaded to CPU first — RNG state tensors must
        # be CPU ByteTensors, and optimizer/model load_state_dict move their
        # tensors to the param device themselves.
        start_epoch = 0
        start_step_in_epoch = 0
        if resume_run_dir:
            rc = torch.load(os.path.join(log_dir, "last_checkpoint.pt"),
                            map_location='cpu')
            if rc.get('resume_format_version') != 1:
                raise ValueError(f"Unknown resume checkpoint format: "
                                 f"{rc.get('resume_format_version')}")
            if rc['total_steps'] != total_steps or rc['steps_per_epoch'] != steps_per_epoch:
                raise ValueError(
                    f"[Resume] Schedule mismatch: checkpoint has "
                    f"{rc['steps_per_epoch']} steps/epoch, {rc['total_steps']} total; "
                    f"recomputed {steps_per_epoch}/{total_steps}. Dataset or batch "
                    f"config changed — resuming would corrupt the LR schedule.")
            _base = getattr(model, '_orig_mod', model)
            _base.load_state_dict(rc['model_state'], strict=True)
            for opt, opt_state in zip(optimizers, rc['optimizer_states']):
                opt.load_state_dict(opt_state)
            scaler.load_state_dict(rc['scaler_state'])
            best_val_loss = rc['best_val_loss']
            best_epoch = rc['best_epoch']
            global_step = rc['global_step']
            epoch_times = list(rc['epoch_times'])
            overall_start_time = time.time() - rc['elapsed_seconds']
            start_epoch = rc['epoch']
            start_step_in_epoch = rc['step_in_epoch']
            torch.set_rng_state(rc['rng_torch_cpu'])
            if rc.get('rng_torch_cuda') is not None and device == 'cuda':
                torch.cuda.set_rng_state_all(rc['rng_torch_cuda'])
            random.setstate(rc['rng_python'])
            if config.get('sequential_blocks', False):
                logger.log("[Resume] WARNING: sequential_blocks position state is not "
                           "checkpointed; positions restart from the stride grid.")
            logger.log(f"[Resume] Restored: epoch {start_epoch + 1}, "
                       f"step-in-epoch {start_step_in_epoch}, global step {global_step}, "
                       f"best val {best_val_loss:.4f} (epoch {best_epoch}), "
                       f"elapsed {rc['elapsed_seconds']/3600:.2f}h. "
                       f"(A resumed partial epoch logs a shorter epoch time.)")

        # Truncated BPTT across windows (decompose-bypass cross-window state).
        # Only meaningful in sequential mode — random batching links unrelated
        # windows, so BPTT through them is noise. Span = grad_accum consecutive
        # windows (the accumulation cycle); graph is retained across the span
        # and truncated at each optimizer step.
        _base_model = model.module if hasattr(model, 'module') else model
        bptt_enabled = config.get('decompose_bypass_bptt', False)
        if bptt_enabled and not config.get('sequential_blocks', False):
            logger.log("[BPTT] decompose_bypass_bptt=true requires sequential_blocks=true "
                       "— disabling BPTT (random windows are unrelated).")
            bptt_enabled = False
        _bptt_span_cfg = config.get('decompose_bypass_bptt_span', 0)
        if bptt_enabled and _bptt_span_cfg not in (0, config.get('grad_accum', 1)):
            logger.log(f"[BPTT] decompose_bypass_bptt_span={_bptt_span_cfg} unsupported in v1 "
                       f"(span is tied to grad_accum={config.get('grad_accum', 1)}); using grad_accum.")
        if bptt_enabled:
            logger.log(f"[BPTT] Truncated BPTT enabled: span = grad_accum "
                       f"({config.get('grad_accum', 1)} windows x block_size).")

        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # =====================================================================
        # TRAINING LOOP
        # =====================================================================

        for epoch in range(start_epoch, config['epochs']):
            epoch_start = time.time()
            logger.log(f"\n=== EPOCH {epoch+1}/{config['epochs']} ===")

            first_step = start_step_in_epoch if epoch == start_epoch else 0
            pbar = tqdm(range(first_step, steps_per_epoch), desc=f"Epoch {epoch+1}",
                        initial=first_step, total=steps_per_epoch)
            for step in pbar:
                lr = get_lr(global_step, config, total_steps, warmup_steps)
                for opt in optimizers:
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                loss_accum = 0.0
                ga = config.get('grad_accum', 1)

                if bptt_enabled:
                    # Truncated BPTT: retain the cross-window state's graph
                    # across the grad_accum span, sum the (scaled) losses, and
                    # backward once so gradient flows back through prior windows.
                    # Then truncate (detach) at the optimizer-step boundary.
                    # Holds ga windows' activations at once (memory ~ga x).
                    _base_model._semantic_keep_graph = True
                    span_losses = []
                    for micro in range(ga):
                        xb, yb = get_batch('train')
                        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                            _, loss = model(xb, yb)
                            loss = loss / ga
                        span_losses.append(loss)
                        loss_accum += loss.detach().item()
                    total_loss = sum(span_losses)
                    if scaler.is_enabled():
                        scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()
                    # Truncate the graph immediately after the span's backward.
                    _base_model._semantic_keep_graph = False
                    if _base_model._persistent_semantic_state is not None:
                        _base_model._persistent_semantic_state = \
                            _base_model._persistent_semantic_state.detach()
                else:
                    for micro in range(ga):
                        xb, yb = get_batch('train')

                        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                            _, loss = model(xb, yb)
                            loss = loss / ga

                        loss_accum += loss.detach().item()

                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                # Gradient clipping
                if scaler.is_enabled():
                    for opt in optimizers:
                        scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))

                if scaler.is_enabled():
                    for opt in optimizers:
                        scaler.step(opt)
                    scaler.update()
                else:
                    for opt in optimizers:
                        opt.step()

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

                    # NaN guard. A diverged run is unrecoverable: every later step
                    # spreads the NaN, and the periodic resume-checkpoint OVERWRITES
                    # last_checkpoint.pt with all-NaN weights — destroying the last
                    # good snapshot, which is exactly the artifact needed to diagnose
                    # the blowup. (Measured 2026-07-25: the softplus_s AMB run NaN'd
                    # at ~step 10.1K and kept training to 20K, leaving 2034/2054
                    # tensors non-finite incl. token_embedding — no evidence left, and
                    # ~2.5h of pod time spent producing it.) Abort instead: the last
                    # checkpoint stays pre-NaN, and run_ablation continues the queue.
                    if not (math.isfinite(losses['train']) and math.isfinite(losses['val'])):
                        logger.log(
                            f"[NaN GUARD] Non-finite loss at step {global_step} "
                            f"(train={losses['train']}, val={losses['val']}, lr={lr:.2e}). "
                            f"Aborting; last_checkpoint.pt preserved pre-NaN for diagnosis."
                        )
                        if config.get('abort_on_nan', True):
                            sys.exit(2)

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

                # Periodic mid-epoch resume checkpoint (opt-in via
                # checkpoint_interval_steps; essential for runs whose epochs
                # are hours long — bounds crash loss to the interval).
                # ...but never overwrite a good checkpoint with diverged weights, in
                # case the NaN lands between evals (checkpoint_interval need not be a
                # multiple of eval_interval).
                if (checkpoint_interval and global_step % checkpoint_interval == 0
                        and math.isfinite(loss_accum)):
                    save_resume_checkpoint(
                        os.path.join(log_dir, "last_checkpoint.pt"),
                        model, optimizers, scaler,
                        epoch=epoch, step_in_epoch=step + 1,
                        global_step=global_step,
                        best_val_loss=best_val_loss, best_epoch=best_epoch,
                        epoch_times=epoch_times,
                        elapsed_seconds=time.time() - overall_start_time,
                        steps_per_epoch=steps_per_epoch, total_steps=total_steps)

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

            # Epoch-end resume checkpoint — always written, so every run is
            # resumable with at most one epoch of loss even when
            # checkpoint_interval_steps is 0.
            save_resume_checkpoint(
                os.path.join(log_dir, "last_checkpoint.pt"),
                model, optimizers, scaler,
                epoch=epoch + 1, step_in_epoch=0,
                global_step=global_step,
                best_val_loss=best_val_loss, best_epoch=best_epoch,
                epoch_times=epoch_times,
                elapsed_seconds=time.time() - overall_start_time,
                steps_per_epoch=steps_per_epoch, total_steps=total_steps)

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

        del model, optimizers, scaler
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # =========================================================================
    # BENCHMARKS & GENERATION (runs after training or in benchmark_only mode)
    # =========================================================================

    # Free training-time GPU state before the inference rebuild + reload below.
    # A large model that nearly fills VRAM during training otherwise OOMs the
    # checkpoint reload — the `except` clause then swallows the error and the
    # benchmark silently runs on an UNLOADED (garbage) model, reporting a
    # meaningless BPB/PPL (observed: L=4 full-cap at 30.6/32 GB on the 5090, PPL
    # 61k vs val 3.40). Dropping the training model + optimizer accumulators frees
    # ~optimizer-state-sized VRAM for a clean rebuild. No-op in benchmark_only
    # mode (optimizer was never created; the redundant model rebuild is harmless).
    model = None
    optimizer = None
    import gc
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

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

    # BBCE benchmark: BBCE-trained models require (B, block_size_compressed)
    # input. evaluate_bbce slides bs/2-stride supervised windows over the test
    # set, left-padding the context when bs_compressed > test_len. Reported as
    # results_full so downstream logging/aggregation paths see a single number;
    # results_sw is set to None because BBCE has only one natural eval style
    # (supervised-half scoring) — there is no second "sliding" benchmark to run.
    if config.get('bbce_enabled', False):
        # Default uncapped: score every natural window, matching the non-BBCE
        # sliding-window benchmark's full test-set coverage. The config key
        # `bbce_benchmark_max_windows` may set a positive integer to cap and
        # uniformly subsample (representative, not front-truncated) — useful
        # for very long bc on huge test sets if benchmark time becomes
        # binding. Defaults to None (uncapped).
        max_windows = config.get('bbce_benchmark_max_windows', None)
        if max_windows is not None:
            max_windows = int(max_windows)
        results_full = evaluate_bbce(
            model, test_data, config, logger, device, use_amp, amp_dtype,
            max_windows=max_windows,
        )
        results_sw = None
        bpb_full = None
        bpb_sw = None
        if results_full:
            bpb_full = results_full['bits_per_token'] / bytes_per_token
            logger.log(f"\n[BBCE BENCHMARK Results]")
            logger.log(f"  Perplexity: {results_full['perplexity']:.4f}")
            logger.log(f"  BPT: {results_full['bits_per_token']:.4f}")
            logger.log(f"  BPB: {bpb_full:.4f}")
            logger.log(f"  Avg Loss: {results_full['avg_loss']:.4f}")
            logger.log(f"  Windows: {results_full['num_windows']} "
                       f"(of {results_full['num_natural_windows']} natural, "
                       f"{results_full['num_padded_windows']} padded)")
    else:
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

    # Word-level (canonical-unit) perplexities, logged per run so every log
    # carries the cross-paper-comparable numbers alongside the per-token ones.
    if results_full or results_sw:
        n_tok = len(test_data)
        measured_words = len(enc.decode(test_data.tolist()).split())
        canon = CANONICAL_TEST_WORDS.get(dataset_name)
        logger.log(f"\n[BENCHMARK - Word-level normalization]")
        logger.log(f"  Test tokens: {n_tok:,} ({enc.name}); "
                   f"measured whitespace words: {measured_words:,}")
        if canon:
            ratio = n_tok / canon
            logger.log(f"  Canonical words: {canon:,} -> tokens/word: {ratio:.4f}")
            if results_full:
                logger.log(f"  Non-overlapping word-level PPL: "
                           f"{math.exp(results_full['avg_loss'] * ratio):.2f}")
            if results_sw:
                logger.log(f"  Sliding word-level PPL: "
                           f"{math.exp(results_sw['avg_loss'] * ratio):.2f}")
        else:
            logger.log(f"  No canonical word count for {dataset_name!r}; "
                       f"tokens/measured-word: {n_tok / max(1, measured_words):.4f} "
                       f"(informational — BPB is the comparable metric here)")

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

    # BBCE forward requires (B, block_size_compressed) input. generate_one
    # pads short idx_cond up to bs_compressed internally; here we just set
    # context_len so the rolling window keeps the right number of recent tokens.
    if config.get('bbce_enabled', False):
        gen_context_len = int(config['block_size_compressed'])
    else:
        gen_context_len = block_size

    base_kwargs = dict(
        temperature=config.get('temperature', 1.0),
        num_tokens=num_tokens,
        context_len=gen_context_len,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    # Standard generation (no strategies)
    logger.log("\n=== GENERATION — Standard ===")
    logger.log(f"Prompt: {prompt}")
    if hasattr(model, 'reset_semantic_state'):
        model.reset_semantic_state()
    std_txt = generate_one(model, enc, input_tensor, **base_kwargs)
    logger.log(f"Generated:\n{std_txt}\n")

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
    strat_txt, token_ids, stats = generate_one(
        model, enc, input_tensor, return_stats=True, **strategies_kwargs)
    metrics = compute_quality_metrics(token_ids, stats['log_probs'], stats['entropies'])
    logger.log(f"Generated:\n{strat_txt}\n")
    logger.log(f"Metrics: {format_metrics(metrics)}")

    # Always write the in-process generations to generations.txt so the file
    # exists regardless of launch path. Previously only generate.py (via runs.sh's
    # run_inference_vram_latest) created it, so a manual `python train.py` launch
    # produced no generations.txt at all. runs.sh's fresh-process generate.py pass
    # still appends its accurate "Peak GPU memory" line on top of this when it runs.
    try:
        gen_path = os.path.join(log_dir, "generations.txt")
        with open(gen_path, "a", encoding="utf-8") as gf:
            gf.write("=== IN-PROCESS GENERATION (train.py) ===\n")
            gf.write(f"Prompt: {prompt}\n\n")
            gf.write(f"--- Standard ---\n{std_txt}\n\n")
            gf.write(f"--- Strategies ---\n{strat_txt}\n")
            gf.write(f"Metrics: {format_metrics(metrics)}\n\n")
    except Exception as e:
        logger.log(f"[train.py] Could not write generations.txt: {e}")

    if device == 'cuda':
        # Note: train.py used to log an "Inference VRAM (approx)" estimate
        # here, but it consistently over-reported by 3-4× because the post-
        # benchmark process still holds optimizer accumulators, gradient
        # buffers, AMP master weights, and torch.compile graph state that a
        # fresh `generate.py` invocation does not. Removed in favor of the
        # accurate generate.py measurement (see generations.txt "Peak GPU
        # memory" line for any saved checkpoint).
        logger.log(
            "Inference VRAM: not measured here. Run `python generate.py "
            "--checkpoint <run>/best_model.pt` and read the 'Peak GPU memory' "
            "line from generations.txt for an accurate fresh-process number.")
        if train_peak_mem is not None:
            logger.log(f"Training Peak VRAM: {train_peak_mem:.0f} MiB")

    logger.close()
    print(f"\nRun directory: {log_dir}")


if __name__ == "__main__":
    train()
