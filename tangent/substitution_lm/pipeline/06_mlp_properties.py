"""
Step 6 — MLP property classifier (Phase 2).

Linear probe (default) over spaCy en_core_web_lg word vectors (300-d) that
imputes ConceptNet-style property entries for the ~90.9% of nodes the raw
ConceptNet lookup misses. ConceptNet entries always override MLP predictions
where present (Reiter-style exception override on top of an MLP default).

Two noise-control levers, both configurable:
    MLP_PROPERTY_COVERAGE_PCT : per-relation top-N% target word cutoff by
                                cumulative frequency — features without
                                enough training labels are excluded outright.
    MLP_PROPERTY_CONFIDENCE   : sigmoid threshold at inference — predictions
                                below threshold are "unknown" and contribute
                                nothing to compatibility().

Architecture:
    MLP_PROPERTY_LAYERS = 0  → Linear(300, K) — fully interpretable per-target
                              weight vector (probing-classifier baseline).
    MLP_PROPERTY_LAYERS ≥ 1  → ReLU MLP for compositional features.

Outputs:
    MLP_PROPERTY_PATH        — merged properties dict (ConceptNet ∪ imputed)
    MLP_PROPERTY_MODEL_PATH  — trained state dict + target schema
"""

import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import config as C


def _vec_for_phrase(nlp, phrase: str):
    """Mean-pool spaCy vectors over whitespace-split tokens of a phrase."""
    vecs = []
    for tok in phrase.split():
        lex = nlp.vocab[tok]
        if lex.has_vector and lex.vector_norm > 0:
            vecs.append(lex.vector)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def _compute_target_schema(properties: dict, coverage_pct: int) -> dict:
    """Per-relation top-N% target-word cutoff by cumulative frequency.

    Per-relation rather than global so sparse relations (e.g. NotCapableOf)
    aren't drowned out by dense ones (RelatedTo).
    """
    pair_counts: dict[str, Counter] = defaultdict(Counter)
    for props in properties.values():
        for relation, targets in props.items():
            for target in targets:
                pair_counts[relation][target] += 1

    schema = {}
    for relation, counts in pair_counts.items():
        total = sum(counts.values())
        cutoff = total * coverage_pct / 100.0
        kept, cumulative = [], 0
        for target, count in counts.most_common():
            kept.append(target)
            cumulative += count
            if cumulative >= cutoff:
                break
        schema[relation] = kept
    return schema


def _build_dataset(properties, id2ngram, target_to_idx, nlp):
    """Build (X, Y_indices) for nodes with ConceptNet data AND a spaCy vector.

    Y is stored sparsely as a list of positive-index lists per sample to keep
    memory tractable at large K (a dense (120K, 76K) float32 matrix is 36 GB).
    """
    X_list, Y_indices = [], []
    skipped = 0
    for nid, props in properties.items():
        word = id2ngram.get(nid)
        if word is None:
            continue
        vec = _vec_for_phrase(nlp, word)
        if vec is None:
            skipped += 1
            continue
        indices = []
        for relation, target_list in props.items():
            for target in target_list:
                idx = target_to_idx.get((relation, target))
                if idx is not None:
                    indices.append(idx)
        X_list.append(vec)
        Y_indices.append(indices)
    X = np.asarray(X_list, dtype=np.float32)
    return X, Y_indices, skipped


def _make_model(input_dim: int, output_dim: int, hidden: int, n_hidden: int):
    import torch.nn as nn
    if n_hidden == 0:
        return nn.Linear(input_dim, output_dim)
    layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, output_dim))
    return nn.Sequential(*layers)


def _compute_pos_weight(Y_indices, K, n_samples):
    """Per-class neg/pos ratio capped at 100 — drives BCEWithLogitsLoss."""
    pos = np.zeros(K, dtype=np.float32)
    for indices in Y_indices:
        for i in indices:
            pos[i] += 1
    neg = n_samples - pos
    pw = neg / (pos + 1.0)
    return np.minimum(pw, 100.0)


class _SparseLabelDataset:
    """Yields (x, y_dense) per sample; y materialised just-in-time."""

    def __init__(self, X, Y_indices, K):
        import torch
        self.X = torch.from_numpy(X)
        self.Y_indices = Y_indices
        self.K = K

    def __len__(self):
        return len(self.Y_indices)

    def __getitem__(self, idx):
        import torch
        y = torch.zeros(self.K, dtype=torch.float32)
        for i in self.Y_indices[idx]:
            y[i] = 1.0
        return self.X[idx], y


def _train(model, X, Y_indices, K, epochs=20, batch_size=256, lr=1e-3, val_pct=10):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(seed=1337)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * val_pct / 100)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_tr, X_va = X[train_idx], X[val_idx]
    Y_tr = [Y_indices[i] for i in train_idx]
    Y_va = [Y_indices[i] for i in val_idx]

    pos_weight = torch.from_numpy(_compute_pos_weight(Y_tr, K, len(Y_tr)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        _SparseLabelDataset(X_tr, Y_tr, K), batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        _SparseLabelDataset(X_va, Y_va, K), batch_size=batch_size, shuffle=False,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total, n_seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = total / max(n_seen, 1)

        model.eval()
        total_val, n_val_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                loss = criterion(model(xb), yb)
                total_val += loss.item() * xb.size(0)
                n_val_seen += xb.size(0)
        val_loss = total_val / max(n_val_seen, 1)

        print(f"  Epoch {epoch:2d}: train {train_loss:.4f}  val {val_loss:.4f}", flush=True)


def _impute(model, properties, id2ngram, targets, nlp, conf, batch_size=512):
    """Predict properties for nodes NOT in ConceptNet; apply confidence threshold."""
    import torch

    uncovered = []
    for nid, word in id2ngram.items():
        if nid in properties:
            continue
        vec = _vec_for_phrase(nlp, word)
        if vec is None:
            continue
        uncovered.append((nid, vec))

    print(f"  Uncovered nodes with vectors: {len(uncovered):,}")

    model.eval()
    imputed = {}
    n_pred_entries = 0
    with torch.no_grad():
        for i in range(0, len(uncovered), batch_size):
            batch = uncovered[i:i + batch_size]
            nids = [b[0] for b in batch]
            vecs = np.asarray([b[1] for b in batch], dtype=np.float32)
            probs = torch.sigmoid(model(torch.from_numpy(vecs))).numpy()
            for nid, p_vec in zip(nids, probs):
                node_props = defaultdict(list)
                for (relation, target), p in zip(targets, p_vec):
                    if p > conf:
                        node_props[relation].append(target)
                if node_props:
                    imputed[nid] = dict(node_props)
                    n_pred_entries += sum(len(v) for v in node_props.values())
    avg = n_pred_entries / max(len(imputed), 1)
    print(f"  Avg entries per imputed node: {avg:.1f}")
    return imputed


def run() -> None:
    if not C.MLP_PROPERTIES_ENABLED:
        print("[06_mlp] MLP_PROPERTIES_ENABLED=False — skipping.")
        return

    if C.MLP_PROPERTY_PATH.exists():
        print(f"[06_mlp] Output exists: {C.MLP_PROPERTY_PATH}  (delete to rebuild)")
        return

    for dep in (C.NODE_TABLE_PATH, C.PROPERTIES_PATH):
        if not dep.exists():
            print(f"[06_mlp] Missing: {dep} — run earlier steps first.")
            sys.exit(1)

    t0 = time.time()
    print("[06_mlp] Loading node table + ConceptNet properties …")
    with open(C.NODE_TABLE_PATH, "rb") as f:
        table = pickle.load(f)
    with open(C.PROPERTIES_PATH, "rb") as f:
        properties = pickle.load(f)

    id2ngram = table["id2ngram"]
    n_nodes = len(id2ngram)
    cn_pct = 100.0 * len(properties) / max(n_nodes, 1)
    print(f"  Nodes: {n_nodes:,}   ConceptNet coverage: {len(properties):,} ({cn_pct:.1f}%)")

    print(f"[06_mlp] Loading spaCy ({C.SPACY_MODEL}) vectors …")
    import spacy
    spacy.require_cpu()
    nlp = spacy.load(
        C.SPACY_MODEL,
        disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"],
    )

    print(f"[06_mlp] Computing target schema (top {C.MLP_PROPERTY_COVERAGE_PCT}% per relation) …")
    schema = _compute_target_schema(properties, C.MLP_PROPERTY_COVERAGE_PCT)
    targets = [(rel, t) for rel in sorted(schema) for t in schema[rel]]
    target_to_idx = {t: i for i, t in enumerate(targets)}
    K = len(targets)
    for rel in sorted(schema):
        print(f"  {rel:18s}: {len(schema[rel]):,} target words")
    print(f"  Total output dim K = {K:,}")

    print("[06_mlp] Building training set …")
    X, Y_indices, skipped = _build_dataset(properties, id2ngram, target_to_idx, nlp)
    n_pos = sum(len(idx) for idx in Y_indices)
    pos_rate = 100.0 * n_pos / max(X.shape[0] * K, 1)
    print(f"  Training nodes: {X.shape[0]:,}   skipped (no vector): {skipped:,}")
    print(f"  Positive labels: {n_pos:,}   label rate: {pos_rate:.4f}%")

    layers = C.MLP_PROPERTY_LAYERS
    label = "linear probe" if layers == 0 else f"{layers}-hidden-layer MLP"
    print(f"[06_mlp] Training {label} …")
    model = _make_model(C.MLP_PROPERTY_EMBED_DIM, K, C.MLP_PROPERTY_HIDDEN_DIM, layers)
    import torch
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    _train(model, X, Y_indices, K)

    print(f"[06_mlp] Saving model + schema → {C.MLP_PROPERTY_MODEL_PATH}")
    torch.save({
        "state_dict": model.state_dict(),
        "targets":    targets,
        "layers":     layers,
        "hidden":     C.MLP_PROPERTY_HIDDEN_DIM,
        "input_dim":  C.MLP_PROPERTY_EMBED_DIM,
    }, C.MLP_PROPERTY_MODEL_PATH)

    print(f"[06_mlp] Imputing properties for uncovered nodes (σ > {C.MLP_PROPERTY_CONFIDENCE}) …")
    imputed = _impute(model, properties, id2ngram, targets, nlp, C.MLP_PROPERTY_CONFIDENCE)
    imp_pct = 100.0 * len(imputed) / max(n_nodes, 1)
    print(f"  Imputed: {len(imputed):,} nodes ({imp_pct:.1f}% of total)")

    # ConceptNet wins on conflict; imputed and ConceptNet are disjoint by construction
    # (impute skips nodes already in properties).
    merged = {**imputed, **properties}
    total_pct = 100.0 * len(merged) / max(n_nodes, 1)
    print(f"  Combined coverage: {len(merged):,} nodes ({total_pct:.1f}%)")

    with open(C.MLP_PROPERTY_PATH, "wb") as f:
        pickle.dump(merged, f)

    elapsed = time.time() - t0
    print(f"[06_mlp] Done → {C.MLP_PROPERTY_PATH}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    run()
