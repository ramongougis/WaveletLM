"""
Step 1 — Parse WikiText-103 with SpaCy.

Reads the raw training (and test) text, splits into documents on WT103 heading
markers, runs SpaCy dependency parsing in batches, and writes a JSONL cache of
the information needed by downstream steps:
    - sentence boundaries (token offsets within document)
    - noun-phrase chunks (text, root lemma, token span)
    - per-token POS and lemma (for PMI counting and n-gram extraction)

VRAM: none. SpaCy is pinned to CPU via spacy.require_cpu().
RAM:  ~700 MB for the SpaCy model + one PARSE_CHUNK worth of docs in flight.
      The full parse is never held in memory — each chunk is flushed to JSONL.
"""

import json
import sys
import time
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


def _iter_documents(path: Path):
    """Yield (doc_index, raw_text) for each WT103 article."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw corpus not found: {path}\n"
            "Set WT103_TRAIN / WT103_TEST env vars or adjust config.py."
        )
    buffer, doc_idx = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            # WT103 article boundaries: lines starting with " = " (top-level heading)
            if line.startswith(" = ") and not line.startswith(" = = ") and buffer:
                yield doc_idx, "".join(buffer).strip()
                doc_idx += 1
                buffer = [line]
            else:
                buffer.append(line)
    if buffer:
        yield doc_idx, "".join(buffer).strip()


def _doc_to_record(spacy_doc) -> dict:
    """Extract the fields we need from a parsed SpaCy doc."""
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
    src  = C.TRAIN_RAW if split == "train" else C.TEST_RAW
    dest = C.PARSE_CACHE_TRAIN if split == "train" else C.PARSE_CACHE_TEST

    if dest.exists():
        print(f"[01_parse] Cache already exists: {dest}  (delete to re-parse)")
        return

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[01_parse] Loading SpaCy model '{C.SPACY_MODEL}' on CPU …")
    try:
        spacy.require_cpu()
        nlp = spacy.load(C.SPACY_MODEL, disable=["ner"])
    except OSError:
        print(
            f"  SpaCy model not found. Run:\n"
            f"    python -m spacy download {C.SPACY_MODEL}"
        )
        sys.exit(1)

    t0 = time.time()
    docs_written = 0
    batch_texts, batch_indices = [], []

    def _flush(fh, texts, indices):
        for _, parsed in zip(indices, nlp.pipe(texts, batch_size=C.SPACY_BATCH)):
            record = _doc_to_record(parsed)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(dest, "w", encoding="utf-8") as fh:
        for idx, text in _iter_documents(src):
            batch_texts.append(text)
            batch_indices.append(idx)

            if len(batch_texts) >= C.PARSE_CHUNK:
                _flush(fh, batch_texts, batch_indices)
                docs_written += len(batch_texts)
                elapsed = time.time() - t0
                print(f"  {docs_written:,} docs parsed  ({elapsed:.0f}s)", flush=True)
                batch_texts, batch_indices = [], []

        if batch_texts:
            _flush(fh, batch_texts, batch_indices)
            docs_written += len(batch_texts)

    elapsed = time.time() - t0
    print(f"[01_parse] Done — {docs_written:,} docs → {dest}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    run(split)
