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
from tqdm import tqdm

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
    bar = tqdm(properties.items(), total=len(properties),
               desc="  Vectorising nodes", unit="node",
               dynamic_ncols=True, smoothing=0.05)
    for nid, props in bar:
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


def _train(model, X, Y_indices, K, epochs=5, batch_size=1024, lr=1e-3, val_pct=10):
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
        bar = tqdm(train_loader, desc=f"  Epoch {epoch:2d}/{epochs} train",
                   leave=False, dynamic_ncols=True, smoothing=0.05)
        for xb, yb in bar:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            bar.set_postfix(loss=f"{total / n_seen:.4f}")
        train_loss = total / max(n_seen, 1)

        model.eval()
        total_val, n_val_seen = 0.0, 0
        vbar = tqdm(val_loader, desc=f"  Epoch {epoch:2d}/{epochs} val  ",
                    leave=False, dynamic_ncols=True, smoothing=0.05)
        with torch.no_grad():
            for xb, yb in vbar:
                loss = criterion(model(xb), yb)
                total_val += loss.item() * xb.size(0)
                n_val_seen += xb.size(0)
        val_loss = total_val / max(n_val_seen, 1)

        print(f"  Epoch {epoch:2d}: train {train_loss:.4f}  val {val_loss:.4f}", flush=True)


def _impute(model, properties, id2ngram, targets, nlp, conf, max_per_node=0, batch_size=1024):
    """Predict properties for nodes NOT in ConceptNet; apply confidence threshold."""
    import torch

    uncovered = []
    bar = tqdm(id2ngram.items(), total=len(id2ngram),
               desc="  Vectorising uncovered", unit="node",
               dynamic_ncols=True, smoothing=0.05)
    for nid, word in bar:
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
    n_batches = (len(uncovered) + batch_size - 1) // batch_size
    ibar = tqdm(range(0, len(uncovered), batch_size), total=n_batches,
                desc="  Predicting", unit="batch",
                dynamic_ncols=True, smoothing=0.05)
    with torch.no_grad():
        for i in ibar:
            batch = uncovered[i:i + batch_size]
            nids = [b[0] for b in batch]
            vecs = np.asarray([b[1] for b in batch], dtype=np.float32)
            probs = torch.sigmoid(model(torch.from_numpy(vecs))).numpy()

            n_batch = probs.shape[0]
            if max_per_node and max_per_node > 0:
                # Top-K per node via argpartition (O(K) in C), then filter by
                # confidence. Avoids the per-node Python sort over all
                # super-threshold targets — the linear probe routinely fires
                # thousands of positives per node, and sorting them all was
                # the per-batch bottleneck.
                top_k_idx = np.argpartition(-probs, max_per_node - 1, axis=1)[:, :max_per_node]
                row_idx = np.arange(n_batch)[:, None]
                top_k_probs = probs[row_idx, top_k_idx]
                keep_mask = top_k_probs > conf
                for node_i in range(n_batch):
                    kept = top_k_idx[node_i][keep_mask[node_i]]
                    if len(kept) == 0:
                        continue
                    node_props = defaultdict(list)
                    for ti in kept:
                        relation, target = targets[ti]
                        node_props[relation].append(target)
                    imputed[nids[node_i]] = dict(node_props)
                    n_pred_entries += len(kept)
            else:
                # No cap — find all positives via argwhere
                pos_idx = np.argwhere(probs > conf)
                per_node: dict[int, list[int]] = defaultdict(list)
                for node_i, target_i in pos_idx:
                    per_node[int(node_i)].append(int(target_i))
                for node_i, target_indices in per_node.items():
                    node_props = defaultdict(list)
                    for ti in target_indices:
                        relation, target = targets[ti]
                        node_props[relation].append(target)
                    imputed[nids[node_i]] = dict(node_props)
                    n_pred_entries += len(target_indices)

            ibar.set_postfix(imputed=f"{len(imputed):,}")
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

    import torch

    layers = C.MLP_PROPERTY_LAYERS
    label = "linear probe" if layers == 0 else f"{layers}-hidden-layer MLP"
    model = _make_model(C.MLP_PROPERTY_EMBED_DIM, K, C.MLP_PROPERTY_HIDDEN_DIM, layers)
    n_params = sum(p.numel() for p in model.parameters())

    if C.MLP_PROPERTY_MODEL_PATH.exists():
        print(f"[06_mlp] Loading existing checkpoint → {C.MLP_PROPERTY_MODEL_PATH}")
        ckpt = torch.load(C.MLP_PROPERTY_MODEL_PATH, weights_only=False)
        if ckpt.get("targets") != targets or ckpt.get("layers") != layers:
            print("  Schema / architecture mismatch — re-training from scratch")
            print("  (delete mlp_property_classifier.pt to silence this and proceed cleanly)")
            sys.exit(1)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded {n_params:,} parameters — skipping training")
    else:
        print("[06_mlp] Building training set …")
        X, Y_indices, skipped = _build_dataset(properties, id2ngram, target_to_idx, nlp)
        n_pos = sum(len(idx) for idx in Y_indices)
        pos_rate = 100.0 * n_pos / max(X.shape[0] * K, 1)
        print(f"  Training nodes: {X.shape[0]:,}   skipped (no vector): {skipped:,}")
        print(f"  Positive labels: {n_pos:,}   label rate: {pos_rate:.4f}%")

        print(f"[06_mlp] Training {label} …")
        print(f"  Parameters: {n_params:,}")
        print(f"  Epochs: {C.MLP_PROPERTY_EPOCHS}   Batch size: {C.MLP_PROPERTY_BATCH_SIZE}")
        _train(model, X, Y_indices, K,
               epochs=C.MLP_PROPERTY_EPOCHS,
               batch_size=C.MLP_PROPERTY_BATCH_SIZE)

        print(f"[06_mlp] Saving model + schema → {C.MLP_PROPERTY_MODEL_PATH}")
        torch.save({
            "state_dict": model.state_dict(),
            "targets":    targets,
            "layers":     layers,
            "hidden":     C.MLP_PROPERTY_HIDDEN_DIM,
            "input_dim":  C.MLP_PROPERTY_EMBED_DIM,
        }, C.MLP_PROPERTY_MODEL_PATH)

    print(f"[06_mlp] Imputing properties for uncovered nodes "
          f"(σ > {C.MLP_PROPERTY_CONFIDENCE}, ≤ {C.MLP_PROPERTY_MAX_PER_NODE}/node) …")
    imputed = _impute(model, properties, id2ngram, targets, nlp,
                      C.MLP_PROPERTY_CONFIDENCE,
                      max_per_node=C.MLP_PROPERTY_MAX_PER_NODE,
                      batch_size=C.MLP_PROPERTY_BATCH_SIZE)
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
