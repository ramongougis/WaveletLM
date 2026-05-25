"""
Step 1 — Parse WikiText-103 with SpaCy.

Loads WikiText-103-raw-v1 via HuggingFace datasets (same cache WaveletLM uses;
no separate download required). Groups the per-line HF examples back into
full articles on " = Title = " boundaries, runs SpaCy dependency parsing in
batches, and writes a JSONL cache of the fields needed downstream:
    - sentence boundaries and per-token POS / lemma
    - noun-phrase chunks (text, root lemma, token span)

VRAM: none. SpaCy is pinned to CPU via spacy.require_cpu().
RAM:  ~700 MB SpaCy model + one PARSE_CHUNK worth of docs in flight.
"""

import json
import sys
import time
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


def _iter_documents(hf_split):
    """
    Yield (doc_index, article_text) by grouping the per-line HF examples.
    WT103-raw examples have a "text" field; top-level article headings look
    like " = Title = \n" (exactly one pair of " = " markers, not " = = ").
    """
    buffer, doc_idx = [], 0
    for example in hf_split:
        line = example["text"]
        is_top_heading = (
            line.startswith(" = ")
            and not line.startswith(" = = ")
            and line.strip().endswith("=")
            and line.count(" = ") == 2  # exactly one level deep
        )
        if is_top_heading and buffer:
            text = "".join(buffer).strip()
            if text:
                yield doc_idx, text
            doc_idx += 1
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        text = "".join(buffer).strip()
        if text:
            yield doc_idx, text


def _doc_to_record(spacy_doc) -> dict:
    sentences = []
    for sent in spacy_doc.sents:
        sentences.append({
            "start": sent.start,
            "end":   sent.end,
            "tokens": [
                {"text": t.text, "lemma": t.lemma_.lower(), "pos": t.pos_}
                for t in sent
            ],
        })
    chunks = []
    for chunk in spacy_doc.noun_chunks:
        chunks.append({
            "text":       chunk.text.lower(),
            "root_lemma": chunk.root.lemma_.lower(),
            "start":      chunk.start,
            "end":        chunk.end,
        })
    return {"sentences": sentences, "chunks": chunks}


def run(split: str = "train") -> None:
    dest = C.PARSE_CACHE_TRAIN if split == "train" else C.PARSE_CACHE_TEST

    if dest.exists():
        print(f"[01_parse] Cache exists: {dest}  (delete to re-parse)")
        return

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[01_parse] Loading HuggingFace dataset '{C.HF_CONFIG}' split='{split}' …")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    hf_split_name = "validation" if split == "valid" else split
    ds = load_dataset(C.HF_DATASET, C.HF_CONFIG, split=hf_split_name)
    print(f"  {len(ds):,} raw examples loaded.")

    print(f"[01_parse] Loading SpaCy model '{C.SPACY_MODEL}' on CPU …")
    try:
        spacy.require_cpu()
        nlp = spacy.load(C.SPACY_MODEL, disable=["ner"])
    except OSError:
        print(f"  SpaCy model not found. Run:\n    python -m spacy download {C.SPACY_MODEL}")
        sys.exit(1)

    t0 = time.time()
    docs_written = 0
    batch_texts, batch_indices = [], []

    def _flush(fh, texts, indices):
        for _, parsed in zip(indices, nlp.pipe(texts, batch_size=C.SPACY_BATCH)):
            fh.write(json.dumps(_doc_to_record(parsed), ensure_ascii=False) + "\n")

    with open(dest, "w", encoding="utf-8") as fh:
        for idx, text in _iter_documents(ds):
            batch_texts.append(text)
            batch_indices.append(idx)
            if len(batch_texts) >= C.PARSE_CHUNK:
                _flush(fh, batch_texts, batch_indices)
                docs_written += len(batch_texts)
                print(f"  {docs_written:,} docs parsed  ({time.time()-t0:.0f}s)", flush=True)
                batch_texts, batch_indices = [], []

        if batch_texts:
            _flush(fh, batch_texts, batch_indices)
            docs_written += len(batch_texts)

    print(f"[01_parse] Done — {docs_written:,} docs → {dest}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    run(split)
