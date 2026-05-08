#!/bin/bash
# Active queue (post-low_rank-sweep). All runs use the L=1 / levels=7 / bs=16384
# baseline; canonical config.json is NEVER modified by this script.
#
# Settled wins so far:
#   - low_rank=16 (R1):                BPB 1.2342, -0.0019 vs reference 1.2361
#   - per_scale_mixer_widths=[0.5×4, 0.25×4] (W2):
#                                      BPB 1.2437, +0.0076 vs default 1.2361, -39% mixer
#   - lifting_diaglowrank=False (A1 shelved at +0.0499)
#
# Cancelled:
#   - low_rank=32 (R1.5):  NaN at step 2250, peak lr=1.00e-2
#   - low_rank=64 (R1.75): cancelled (further into unstable region)
#
# Queue:
#   1. DBD     (already complete)     1ep   decompose_bypass=false + low_rank=16
#   2. E5_5ep  (already complete)     5ep   per_scale_mixer_widths=[1.5×4, 0.5×4]
#   3. R1_5ep  (already complete)     5ep   low_rank=16                              (winner confirmation)
#   4. M1..M4                         1ep   off-diagonal magnitude-pruned masking sweep (0.1/1/5/10%)
#   5. M1r..M4r                       1ep   off-diagonal random controls at matched densities
#   6. Structural-prior alternatives  1ep   triangular / block-diagonal / banded / Monarch families
#                                           (different KIND of structured sparsity than magnitude pruning)
#
# Option B: all off-diagonal sweeps (sections 4, 5, 6) use the unified
# `lifting_offdiag_structure` flag. Legacy `lifting_offdiag_mask` family is
# deprecated. Each run requires `model.py` to support `lifting_offdiag_structure`
# plus its per-structure parameters; verify by inspecting the "Shared lifting"
# line in the param breakdown before reading BPB.
#
# After everything completes, a new combined 5-epoch run will fold the surviving
# winners into a new post-parameter-reduction relative baseline.

set -euo pipefail

# Temp config file used for all train.py --config invocations. Auto-deleted
# on any exit path. Canonical config.json is NEVER modified by this script.
TMP_CFG=$(mktemp -t exarch_run_XXXXXX.json)
trap 'rm -f "$TMP_CFG"' EXIT

build_run_config() {
    python -c "
import json, sys
cfg = json.load(open('config.json'))
for patch in sys.argv[1:]:
    cfg.update(json.loads(patch))
json.dump(cfg, open('$TMP_CFG', 'w'), indent=4)
" "$@"
}

git_commit_push() {
    local MSG="$1"
    if ! git diff --quiet config.json 2>/dev/null; then
        echo "[runs.sh] WARNING: config.json was unexpectedly modified."
        echo "[runs.sh] Reverting config.json before commit (canonical preserved)."
        git checkout -- config.json
    fi
    git add . || true
    git commit --no-edit -m "${MSG}" || true
    git pull --no-edit || true
    git push || true
}

# 1-epoch sweep base (L=1 / levels=7 / bs=16384, post-combined-reduction).
BASE_PATCH_1EP='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 1,
    "mlp_expansion": 10,
    "pkm_enabled": false,
    "fwpkm_num_keys": 8281,
    "tie_embedding_to_lm_head": true,
    "micro_batch_size": 1,
    "grad_accum": 1,
    "block_size": 16384,
    "levels": 7,
    "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": false,
    "lifting_level_sharing": false,
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

# 5-epoch confirmation base (same as 1-epoch except epochs=5).
BASE_PATCH_5EP='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 5,
    "mlp_expansion": 10,
    "pkm_enabled": false,
    "fwpkm_num_keys": 8281,
    "tie_embedding_to_lm_head": true,
    "micro_batch_size": 1,
    "grad_accum": 1,
    "block_size": 16384,
    "levels": 7,
    "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": false,
    "lifting_level_sharing": false,
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

run_ablation() {
    local LABEL="$1"
    local BASE_JSON="$2"
    local OVERRIDE_JSON="$3"
    local COMMIT_MSG="$4"

    echo ""
    echo "============================================================"
    echo "=== Ablation: ${LABEL}"
    echo "============================================================"

    build_run_config "$BASE_JSON" "$OVERRIDE_JSON"
    python train.py --config "$TMP_CFG"
    git_commit_push "${COMMIT_MSG}"
}

# ---- 1. DBD: Decompose Bypass Disablement at 1ep with low_rank=16 -----------
# Validates Future Plans #7 while the 1-epoch baseline is fresh. Boolean
# ablation at L=1/E=1 previously found both flags within ±0.0015 BPB of
# baseline (within noise); this run confirms at L=7 / bs=16384 / 1ep with the
# new low_rank=16 winner. Compare BPB sliding directly against R1's 1.2342.
# run_ablation "DBD decompose_bypass=false + low_rank=16" \
#     "$BASE_PATCH_1EP" \
#     '{"low_rank": 16, "decompose_bypass": false, "decompose_bypass_cross_window": false}' \
#     "DBD: decompose_bypass=false + low_rank=16 (1ep, L=7)"

# # ---- 2. E5 confirmation: per_scale_mixer_widths=[1.5×4, 0.5×4] @ 5ep ---------
# # 5-epoch confirmation of the coarse-expansion config (E5 at 1ep was
# # +0.0063 BPB at +24% params — small delta, but the 5-epoch behavior is
# # what determines whether expansion survives). Headline reference at 5ep is
# # 1.0974; compare BPB delta directly against this baseline.
# run_ablation "E5_5ep per_scale_mixer_widths=[1.5x4,0.5x4] (5 epochs)" \
#     "$BASE_PATCH_5EP" \
#     '{"per_scale_mixer_widths": [1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5]}' \
#     "E5_5ep: per_scale_mixer_widths=[1.5x4,0.5x4] (5ep, L=7)"

# ---- 3. R1 confirmation: low_rank=16 @ 5ep ----------------------------------
# 5-epoch confirmation of the 1-epoch R1 winner. Compared against the headline
# 5-epoch reference of 1.0974; if it beats or ties, low_rank=16 becomes the
# new default. Compare BPB sliding directly against the 1.0974 baseline;
# winner is whatever lands cleanly below 1.0974.
# run_ablation "R1_5ep low_rank=16 (5 epochs)" \
#     "$BASE_PATCH_5EP" \
#     '{"low_rank": 16}' \
#     "R1_5ep: low_rank=16 (5ep, L=7)"

# ---- 4. Wavelet off-diagonal magnitude-pruned masking sweep (1ep each) ------
#
# !!! WARNING: BOOTSTRAPPING DEPENDENCY !!!
# magnitude_topk requires a pre-trained reference checkpoint matching the SAME
# architecture (same C, levels, low_rank) — the mask is computed from that
# checkpoint's lifting weight magnitudes per Linear. New scales (different
# C / levels) and new architectures (e.g., semantic embedding, multi-transform,
# B200 scale-up) CANNOT use magnitude_topk without first training an
# uncompressed reference at the new config — a 2-stage training requirement.
#
# Architecturally portable alternatives that need no reference checkpoint:
#   - random_topk (section 5): same density-vs-recovery curve, random placement
#   - upper_triangular / lower_triangular / block_diagonal / banded / monarch
#     (section 6): purely mathematical structures
# If the M{n}r random controls in section 5 match magnitude_topk performance,
# random becomes the preferred default for any future scale-up. If magnitude
# meaningfully outperforms random, magnitude_topk is reserved for fine-tuning
# a known architecture and other approaches must serve as scale-up defaults.
#
# This sweep loads from logs/wikitext-103_2026-05-03_02-13-07/best_model.pt
# (the L=1 / levels=7 / 5-epoch winner). Always uses lifting_diaglowrank=false
# (the structural mask path includes the diagonal by construction). low_rank=16
# carried forward. Compare BPB delta directly against reference 1.2361.
OFFDIAG_MAG_BASE='{"low_rank": 16, "lifting_diaglowrank": false, "lifting_offdiag_structure": "magnitude_topk", "lifting_offdiag_mask_checkpoint": "logs/wikitext-103_2026-05-03_02-13-07/best_model.pt"}'

# run_ablation "M1 off-diagonal magnitude 0.1%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_MAG_BASE'''); b['lifting_offdiag_density']=0.001; print(json.dumps(b))")" \
#     "M1: off-diagonal magnitude-pruned 0.1% (1ep, L=7)"

# run_ablation "M2 off-diagonal magnitude 1%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_MAG_BASE'''); b['lifting_offdiag_density']=0.01; print(json.dumps(b))")" \
#     "M2: off-diagonal magnitude-pruned 1% (1ep, L=7)"

# run_ablation "M3 off-diagonal magnitude 5%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_MAG_BASE'''); b['lifting_offdiag_density']=0.05; print(json.dumps(b))")" \
#     "M3: off-diagonal magnitude-pruned 5% (1ep, L=7)"

# run_ablation "M4 off-diagonal magnitude 10%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_MAG_BASE'''); b['lifting_offdiag_density']=0.10; print(json.dumps(b))")" \
#     "M4: off-diagonal magnitude-pruned 10% (1ep, L=7)"

# ---- 5. Random off-diagonal masking controls (1ep each) ---------------------
# Same densities as M1/M2/M3/M4 but with a random binary mask (deterministic
# under fixed seed) instead of magnitude-ranked. Tests whether placement
# matters at matched density — Lottery Ticket / RIGL pattern would have
# magnitude beat random by more than the noise floor at higher densities and
# converge at low density. If magnitude and random tie everywhere, density
# alone is what matters and the cheaper random construction wins.
OFFDIAG_RAND_BASE='{"low_rank": 16, "lifting_diaglowrank": false, "lifting_offdiag_structure": "random_topk", "lifting_offdiag_mask_seed": 1337}'

# run_ablation "M1r off-diagonal random 0.1%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_RAND_BASE'''); b['lifting_offdiag_density']=0.001; print(json.dumps(b))")" \
#     "M1r: off-diagonal random 0.1% (1ep, L=7)"

# run_ablation "M2r off-diagonal random 1%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_RAND_BASE'''); b['lifting_offdiag_density']=0.01; print(json.dumps(b))")" \
#     "M2r: off-diagonal random 1% (1ep, L=7)"

# run_ablation "M3r off-diagonal random 5%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_RAND_BASE'''); b['lifting_offdiag_density']=0.05; print(json.dumps(b))")" \
#     "M3r: off-diagonal random 5% (1ep, L=7)"

# run_ablation "M4r off-diagonal random 10%" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$OFFDIAG_RAND_BASE'''); b['lifting_offdiag_density']=0.10; print(json.dumps(b))")" \
#     "M4r: off-diagonal random 10% (1ep, L=7)"

# # ---- 6. Structural-prior alternatives for lifting matrices (1ep each) -------
# Tests structured-sparsity *constraints* on the lifting predict/update Linear(C, C)
# matrices, rather than mask-selection of an already-trained reference's positions
# (which is what M1-M4 did). The two are different in kind: magnitude pruning
# selects from the unconstrained optimum, while these constraints force gradient
# descent to find a DIFFERENT optimum that respects the structure. Distinct
# outcomes are possible despite the M-series nulls.
#
# All variants: low_rank=16 (M-series baseline), lifting_diaglowrank=False
# (these structures include the diagonal by construction), lifting_offdiag_mask=False.
# Mutually exclusive with the existing diaglowrank/offdiag_mask paths;
# model.py needs to support `lifting_offdiag_structure` and the per-structure
# parameters before these runs will execute correctly.
#
# Compare BPB delta directly against reference 1.2361, alongside lifting
# parameter count for per-parameter efficiency comparisons.
#
# Per-matrix parameter counts (Linear(2048, 2048) = 4.19M baseline):
#   T_upper / T_lower         : 2.10M  (50% reduction)
#   BD64  block-diagonal      : 131K   (97% reduction, ~A1-equivalent compression)
#   BD256 block-diagonal      : 524K   (87% reduction)
#   BAND64 banded             : 264K   (94% reduction)
#   BAND256 banded            : 1.05M  (75% reduction)
#   MON32 Monarch (2 factors) : 262K   (94% reduction)
#   MON64 Monarch (2 factors) : 131K   (97% reduction, ~A1-equivalent)
STRUCT_BASE='{"low_rank": 16, "lifting_diaglowrank": false, "lifting_offdiag_mask": false}'

# Triangular: 50% density, channel-ordering imposed (each output sees only a half-cone of inputs).
# # Single variant per direction; the upper-vs-lower comparison is itself the ablation.
# run_ablation "T_upper upper triangular" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='upper_triangular'; print(json.dumps(b))")" \
#     "T_upper: lifting upper triangular (1ep, L=7)"

# run_ablation "T_lower lower triangular" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='lower_triangular'; print(json.dumps(b))")" \
#     "T_lower: lifting lower triangular (1ep, L=7)"

# # Block-diagonal: channel-permutation invariant (no arbitrary ordering),
# # each output sees only the b channels in its own block.
# # Two variants bracketing the A1-equivalent (97%) and moderate (87%) compression regimes.
# run_ablation "BD64 block-diagonal block_size=64" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='block_diagonal'; b['lifting_block_size']=64; print(json.dumps(b))")" \
#     "BD64: lifting block-diagonal blocks=32 of 64x64 (1ep, L=7)"

# run_ablation "BD256 block-diagonal block_size=256" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='block_diagonal'; b['lifting_block_size']=256; print(json.dumps(b))")" \
#     "BD256: lifting block-diagonal blocks=8 of 256x256 (1ep, L=7)"

# # Block-diagonal follow-ups (uncomment after BD256 confirms the per-density curve):
# # BD128 fills the gap between BD64 (3% density, 47.7% recovery) and BD256 (~12.5% density);
# # BD512 anchors the upper end (~25% density, expected ~92-96% recovery near reference).
# # Block-diagonal is the leading portable structural prior; these two runs trace the BPB-vs-density
# # curve in the high-recovery regime where the production default likely lives.
# # run_ablation "BD128 block-diagonal block_size=128" \
# #     "$BASE_PATCH_1EP" \
# #     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='block_diagonal'; b['lifting_block_size']=128; print(json.dumps(b))")" \
# #     "BD128: lifting block-diagonal blocks=16 of 128x128 (1ep, L=7)"
# #
# # run_ablation "BD512 block-diagonal block_size=512" \
# #     "$BASE_PATCH_1EP" \
# #     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='block_diagonal'; b['lifting_block_size']=512; print(json.dumps(b))")" \
# #     "BD512: lifting block-diagonal blocks=4 of 512x512 (1ep, L=7)"

# # Banded: locality structure on the channel axis (1D-conv-like).
# # Two variants: tight bandwidth (94%) and wide bandwidth (75%).
# run_ablation "BAND64 banded bandwidth=64" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='banded'; b['lifting_band_width']=64; print(json.dumps(b))")" \
#     "BAND64: lifting banded bandwidth=64 (1ep, L=7)"

# run_ablation "BAND256 banded bandwidth=256" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='banded'; b['lifting_band_width']=256; print(json.dumps(b))")" \
#     "BAND256: lifting banded bandwidth=256 (1ep, L=7)"

# # Monarch (Dao et al., 2022, https://arxiv.org/abs/2204.00595): products of two
# # block-diagonal factors with permutations between, giving full-matrix expressivity
# # at sublinear params. Two variants at differing block counts (sqrt(C)≈45 for C=2048;
# # common power-of-2 choices are 32 and 64).
# run_ablation "MON32 Monarch nblocks=32" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='monarch'; b['lifting_monarch_blocks']=32; print(json.dumps(b))")" \
#     "MON32: lifting Monarch nblocks=32 (1ep, levels=7)"

# run_ablation "MON64 Monarch nblocks=64" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='monarch'; b['lifting_monarch_blocks']=64; print(json.dumps(b))")" \
#     "MON64: lifting Monarch nblocks=64 (1ep, L=7)"

# run_ablation "MON128 Monarch nblocks=128" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='monarch'; b['lifting_monarch_blocks']=128; print(json.dumps(b))")" \
#     "MON32: lifting Monarch nblocks=128 (1ep, levels=7)"
    
# run_ablation "MON256 Monarch nblocks=256" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='monarch'; b['lifting_monarch_blocks']=256; print(json.dumps(b))")" \
#     "MON32: lifting Monarch nblocks=256 (1ep, levels=7)"

# More runs to complete ablation matrices: BD128, BD512, BAND32, and BAND128.
# These four fill in the matched-param-count pairs that the existing data leaves
# in a checker pattern. After these runs, every BD/BAND comparison can be made at
# nearly identical lifting params:
#   BD64 (3.73M)   <-> BAND32  (3.76M)
#   BD128 (7.40M)  <-> BAND64  (7.34M, done)
#   BD256 (14.74M, done) <-> BAND128 (14.33M)
#   BD512 (29.42M) <-> BAND256 (27.63M, done)

# run_ablation "BD128 block-diagonal block_size=128" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='block_diagonal'; b['lifting_block_size']=128; print(json.dumps(b))")" \
#     "BD128: lifting block-diagonal blocks=16 of 128x128 (1ep, L=7)"

# run_ablation "BD512 block-diagonal block_size=512" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='block_diagonal'; b['lifting_block_size']=512; print(json.dumps(b))")" \
#     "BD512: lifting block-diagonal blocks=4 of 512x512 (1ep, L=7)"

# run_ablation "BAND32 banded bandwidth=32" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='banded'; b['lifting_band_width']=32; print(json.dumps(b))")" \
#     "BAND32: lifting banded bandwidth=32 (1ep, L=7)"

# run_ablation "BAND128 banded bandwidth=128" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$STRUCT_BASE'''); b['lifting_offdiag_structure']='banded'; b['lifting_band_width']=128; print(json.dumps(b))")" \
#     "BAND128: lifting banded bandwidth=128 (1ep, L=7)"


# ---- 7. Sparse (p, q) phantom-token embedding sweep -------------------------
# Tests sparsifying the 102.93M token embedding via the (p, q) striding scheme
# from plans/new_compression_ideas.md. Token embedding is the largest non-cascade
# block in the model; ~90% reduction at d=0.10 saves ~92.6M params, taking the
# whole model from 393M to ~300M before any lifting compression and ~210M when
# stacked with BAND256 (the leading portable lifting variant from section 6).
#
# Three modes:
#   - smallest_q:  minimal phantom rows; near-stride-p behavior
#   - structural:  q closest to sqrt(C); square macrocells (default)
#   - budget:      largest q under phantom-row budget
#
# At d=0.10 with C=2048, N=50257:
#   - smallest_q -> (p=18, q=2, N'=50258, phantom=1)
#   - structural -> (p=12, q=8, N'=50264, phantom=7)
#
# Compare 1-epoch BPB delta against the L=1/levels=7/bs=16384 references:
#   - M0/A1 floor (diagonal only):       1.2860
#   - 1ep reference (uncompressed E5W2): 1.2361
# A working sparse embedding scheme should land within ~0.005 BPB of 1.2361
# at d=0.10 if the model is robust to embedding sparsity (i.e., inter-token
# semantics learnable through the resulting feature-overlap graph).
PQEMB_BASE_1EP_PATCH='{"sparse_pq_embedding_enabled": true, "sparse_pq_embedding_density": 0.1}'

run_ablation "PQ_EMB10_smallest_q (p=18,q=2,d=0.10,smallest_q)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_mode']='smallest_q'; print(json.dumps(b))")" \
    "PQ_EMB10_smallest_q: sparse embedding d=0.10 mode=smallest_q (1ep, levels=7)"

run_ablation "PQ_EMB10_structural (p=12,q=8,d=0.10,structural)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_mode']='structural'; print(json.dumps(b))")" \
    "PQ_EMB10_structural: sparse embedding d=0.10 mode=structural (1ep, levels=7)"

# Density sweep at structural mode (after d=0.10 lands; uncomment to run):
# run_ablation "PQ_EMB05_structural (d=0.05)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_density']=0.05; b['sparse_pq_embedding_mode']='structural'; print(json.dumps(b))")" \
#     "PQ_EMB05_structural: sparse embedding d=0.05 mode=structural (1ep, L=7)"
#
# run_ablation "PQ_EMB20_structural (d=0.20)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_density']=0.20; b['sparse_pq_embedding_mode']='structural'; print(json.dumps(b))")" \
#     "PQ_EMB20_structural: sparse embedding d=0.20 mode=structural (1ep, L=7)"


# ---- 8. MLP structural compression sweep ------------------------------------
# Tests sparsifying the MLP weight matrices (W1: (E·C, C), W2: (C, E·C)) via
# three structural variants:
#   - banded:         tiled per-(C,C)-block bilateral band of width W
#   - block_diagonal: tiled per-(C,C)-block diagonal of size b
#   - pq_strided:     1D walk over flattened (out·in) with alternating p,q
#                     steps; q | C and p ∤ C, p ∤ E·C
#
# At C=2048 / E=10 (1-epoch sweep), MLP is 83.91M of 281M total. ~10% density
# would compress that to ~8.4M, taking the model from 281M to ~205M before
# stacking with the lifting + embedding compression. Four density points to
# locate the recovery floor: 25%, 12.5%, 6.25%, 3.125%.
#
# Compare 1-epoch BPB delta against the L=1 / levels=7 / bs=16384 reference
# 1.2361 alongside per-param efficiency (pp recovered / M params).
MLP_BASE_1EP_PATCH='{}'  # MLP-compression flags layered per-run below

# 25% density: BAND W=256 (per-block density 25.05%), BD b=512 (25.0%), (p,q) d=0.25 (only valid is p=6, q=2)
run_ablation "MLP_BAND25 (W=256)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":256}')")" \
    "MLP_BAND25: MLP banded W=256 per (C,C) block, ~25% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_BD25 (b=512)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":512}')")" \
    "MLP_BD25: MLP block_diagonal b=512 per (C,C) block, ~25% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_PQ25 (d=0.25 -> p=6,q=2)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.25,\"mlp_pq_mode\":\"structural\"}')")" \
    "MLP_PQ25: MLP (p,q) striding at d=0.25 structural (p=6,q=2) (1ep, levels=7)"

# 12.5% density: BAND W=128, BD b=256, (p,q) d=0.125 -> p=12, q=4
run_ablation "MLP_BAND125 (W=128)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":128}')")" \
    "MLP_BAND125: MLP banded W=128 per (C,C) block, ~12.5% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_BD125 (b=256)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":256}')")" \
    "MLP_BD125: MLP block_diagonal b=256 per (C,C) block, 12.5% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_PQ125 (d=0.125 -> p=12,q=4)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.125,\"mlp_pq_mode\":\"structural\"}')")" \
    "MLP_PQ125: MLP (p,q) striding at d=0.125 structural (p=12,q=4) (1ep, levels=7)"

# 6.25% density: BAND W=64, BD b=128, (p,q) d=0.0625 -> p=24, q=8
run_ablation "MLP_BAND0625 (W=64)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":64}')")" \
    "MLP_BAND0625: MLP banded W=64 per (C,C) block, ~6.25% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_BD0625 (b=128)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":128}')")" \
    "MLP_BD0625: MLP block_diagonal b=128 per (C,C) block, 6.25% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_PQ0625 (d=0.0625 -> p=24,q=8)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.0625,\"mlp_pq_mode\":\"structural\"}')")" \
    "MLP_PQ0625: MLP (p,q) striding at d=0.0625 structural (p=24,q=8) (1ep, levels=7)"

# 3.125% density: BAND W=32, BD b=64, (p,q) d=0.03125 -> p=48, q=16
run_ablation "MLP_BAND03125 (W=32)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":32}')")" \
    "MLP_BAND03125: MLP banded W=32 per (C,C) block, ~3.17% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_BD03125 (b=64)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":64}')")" \
    "MLP_BD03125: MLP block_diagonal b=64 per (C,C) block, 3.125% W1+W2 density (1ep, levels=7)"

run_ablation "MLP_PQ03125 (d=0.03125 -> p=48,q=16)" \
    "$BASE_PATCH_1EP" \
    "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.03125,\"mlp_pq_mode\":\"structural\"}')")" \
    "MLP_PQ03125: MLP (p,q) striding at d=0.03125 structural (p=48,q=16) (1ep, levels=7)"

# ---- 9. (PLACEHOLDER) 5-epoch confirmation of the chosen top-k winner -------
# After the 1-epoch sweep in sections 4-6 completes, the best-performing
# configuration goes to a 5-epoch confirmation run against the L=1 / levels=7
# 5-epoch headline (logs/wikitext-103_2026-05-03_02-13-07/log.txt, BPB 1.0974).
#
# Currently leading 1-epoch candidates:
#   - M4 (magnitude_topk, density=0.10): BPB 1.2438 (+0.0096 vs 1ep ref 1.2361)
#     -- but bootstrapping-dependent; only valid if a same-architecture
#        reference checkpoint exists.
#   - Whichever random_topk / structural variant matches or beats M4 at
#     comparable lifting params -- see runs.md tables for the final winner.
#
# Compare 5-epoch BPB delta directly against the 5-epoch headline 1.0974.
# Strongest result: BPB cleanly below 1.0974, indicating compression doesn't
# only preserve performance at scale but improves it (e.g., via reduced
# overfitting to the lifting cascade's full param footprint).
#
# Uncomment and fill in the chosen winner config before launching:
#
# WINNER_BASE='{"low_rank": 16, "lifting_diaglowrank": false, "lifting_offdiag_structure": "<chosen>", "lifting_offdiag_density": <chosen>, "lifting_offdiag_mask_checkpoint": "logs/wikitext-103_2026-05-03_02-13-07/best_model.pt"}'
# run_ablation "C5ep <chosen> winner @ 5 epochs" \
#     "$BASE_PATCH_5EP" \
#     "$WINNER_BASE" \
#     "C5ep: <chosen> winner (5ep, L=7) -- top-k 5-epoch confirmation"

echo ""
echo "============================================================"
echo "=== Queue complete."
echo "===   1) DBD     (already complete)"
echo "===   2) E5_5ep  (already complete)"
echo "===   3) R1_5ep  (already complete)"
echo "===   4) M1..M4 magnitude-pruned off-diagonal — compare BPB delta vs 1.2361"
echo "===   5) M1r..M4r random controls at matched densities — compare BPB delta vs 1.2361"
echo "===   6) Structural priors: T_upper/lower, BD64/256, BAND64/256, MON32/64"
echo "===      — compare BPB delta vs 1.2361 alongside per-param efficiency"
echo "===   7) Sparse (p, q) phantom-token embedding sweep at d=0.10"
echo "===      smallest_q (p=18,q=2) and structural (p=12,q=8) — compare BPB delta vs 1.2361"
echo "===   8) MLP structural compression: BAND/BD/PQ at densities 25%, 12.5%, 6.25%, 3.125%"
echo "===      12 runs (4 densities × 3 structures) — compare BPB delta vs 1.2361 alongside per-param efficiency"
echo "===   9) 5-epoch confirmation of chosen winner — compare BPB delta vs 1.0974"
echo "===      (placeholder; uncomment after sections 4-8 complete and a winner is chosen)"
echo "==="
echo "=== Next: combine surviving winners into a new 5-epoch baseline."
echo "=== Implementation gating: sections 4, 5, and 6 all require model.py to"
echo "===   support the unified lifting_offdiag_structure flag (Option B) and"
echo "===   its per-structure parameters. Verify each variant's 'Shared lifting'"
echo "===   line moves in the expected direction before reading BPB."
echo "============================================================"
