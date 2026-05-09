#!/bin/bash
# Active queue (post-low_rank-sweep). All runs use the L=1 / levels=7 / bs=16384
# baseline; canonical config.json is NEVER modified by this script.
#
# Settled wins so far:
#   - low_rank=16 (LR16):              BPB 1.2342, -0.0019 vs reference 1.2361
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
#   3. LR16_5ep  (already complete)     5ep   low_rank=16                              (winner confirmation)
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

# `set -uo pipefail` (without `-e`) — we want strict undefined-variable and
# pipeline-failure detection, but NOT auto-exit on non-zero return. The
# previous `set -e` was killing the script after train.py exits, even with
# `|| echo "..."` guards in place — likely because torch.compile's atexit
# subprocess cleanup is sending a signal that bash interprets unusually.
# Removing `-e` lets the queue continue past any single-run failure; we still
# get a script-wide stop on truly fatal issues (unset variables, broken
# pipelines, etc.).
set -uo pipefail

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

# 1-epoch sweep base — Baseline 3 (B3) defaults.
# B3 = Test 1 + wavelet_crawl=False. Architecturally minimal change from
# Test 1: drop the deprecated wavelet_crawl convolutional component, keep
# everything else. T-lower is intentionally NOT included — it provided no
# real storage/VRAM savings (mask buffer overhead actually slightly INCREASED
# train VRAM, +80 MiB), and Test 1's matched-budget BPB / best val won
# decisively over the bs=16384 NB stack. Mask-based "compression" is no
# longer a production direction; T-lower remains an opt-in stability tool
# (NaN remediation only).
#
# Architecturally:
#   - Test 1 throughput regime: bs=256, MBS=8 (~58,500 steps/epoch vs the
#     bs=16384 stack's ~7,300 at matched epoch budget)
#   - Test 1 structural choices: levels=5, low_rank=4, R0 mixer pattern
#     [1.0×3, 0.5×3] (W2 mixer contraction dropped — per L=9 R0+T-lower
#     finding that W2 costs ~0.0100 BPB at depth)
#   - Test 1 reductions: mlp_expansion=10, pkm_enabled=False,
#     fwpkm_num_keys=8281, tie_embedding_to_lm_head=True
#   - Lifting: lifting_offdiag_structure="none" (dense, unmasked — Test 1
#     default)
#   - Cleanup: wavelet_crawl=False (deprecated convolutional component)
# Historical pre-B3 runs (DBD, M1-M4, BAND/BD/MON sweeps, prior CB/NB-stack
# runs, NB at bs=16384, the earlier T-lower-flavored B3 runs from
# 2026-05-09 19:28 / 21:06) overrode these with their own per-run patches;
# new runs inherit the redefined B3 by default.
BASE_PATCH_1EP='{
    "dataset": "wikitext-103",
    "layers": 1,
    "epochs": 1,
    "mlp_expansion": 10,
    "pkm_enabled": false,
    "fwpkm_num_keys": 8281,
    "tie_embedding_to_lm_head": true,
    "micro_batch_size": 8,
    "grad_accum": 1,
    "block_size": 256,
    "levels": 5,
    "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": false,
    "lifting_level_sharing": false,
    "low_rank": 4,
    "lifting_offdiag_structure": "none",
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
    "micro_batch_size": 8,
    "grad_accum": 1,
    "block_size": 256,
    "levels": 5,
    "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
    "wavelet_crawl": false,
    "lifting_diaglowrank": false,
    "lifting_level_sharing": false,
    "low_rank": 4,
    "lifting_offdiag_structure": "none",
    "lr": 0.01,
    "min_lr": 0.0002,
    "eval_interval": 250
}'

run_inference_vram_latest() {
    # Locate the most recent run folder by mtime and invoke generate.py on its
    # best checkpoint to get fresh-process inference VRAM measurements. Two
    # passes are run: one standard (naive sampling), one with --strategies
    # enabled. Both append to the run's generations.txt with their own
    # Peak GPU memory line. The strategies-mode pass is also a useful canary
    # for diagnosing strategies-only generation issues (e.g. the levels=9
    # strategies-mode anomaly observed on the NB stack).
    local LATEST_RUN
    LATEST_RUN=$(ls -td logs/wikitext-103_*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "[runs.sh] Skipping inference VRAM measurement (no log dir found)"
        return
    fi
    LATEST_RUN="${LATEST_RUN%/}"
    if [ ! -f "$LATEST_RUN/best_model.pt" ]; then
        echo "[runs.sh] Skipping inference VRAM measurement (no best_model.pt at $LATEST_RUN)"
        return
    fi
    echo ""
    echo "=== Measuring inference VRAM (fresh process, standard) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" || \
        echo "[runs.sh] generate.py (standard) exited non-zero; continuing"
    echo ""
    echo "=== Measuring inference VRAM (fresh process, --strategies) for ${LATEST_RUN}"
    python generate.py --checkpoint "$LATEST_RUN/best_model.pt" --strategies || \
        echo "[runs.sh] generate.py --strategies exited non-zero; continuing"
}

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
    # Diagnostic: capture train.py's actual exit code so we can SEE what's
    # happening when the queue halts unexpectedly. With `set -e` removed at
    # script top, a non-zero exit no longer halts the queue, but we still
    # want it visible in the log.
    python train.py --config "$TMP_CFG"
    local TRAIN_EXIT=$?
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "[runs.sh] train.py exited with code $TRAIN_EXIT; continuing to next ablation"
    fi
    run_inference_vram_latest
    git_commit_push "${COMMIT_MSG}"
}

# ---- 1. DBD: Decompose Bypass Disablement at 1ep with low_rank=16 -----------
# Validates Future Plans #7 while the 1-epoch baseline is fresh. Boolean
# ablation at L=1/E=1 previously found both flags within ±0.0015 BPB of
# baseline (within noise); this run confirms at L=7 / bs=16384 / 1ep with the
# new low_rank=16 winner. Compare BPB sliding directly against LR16's 1.2342.
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

# ---- 3. LR16 confirmation: low_rank=16 @ 5ep --------------------------------
# 5-epoch confirmation of the 1-epoch LR16 winner. Compared against the headline
# 5-epoch reference of 1.0974; if it beats or ties, low_rank=16 becomes the
# new default. Compare BPB sliding directly against the 1.0974 baseline;
# winner is whatever lands cleanly below 1.0974.
# run_ablation "LR16_5ep low_rank=16 (5 epochs)" \
#     "$BASE_PATCH_5EP" \
#     '{"low_rank": 16}' \
#     "LR16_5ep: low_rank=16 (5ep, L=7)"

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
#     "MON64: lifting Monarch nblocks=64 (1ep, levels=7)"

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


# ---- 7. Combined-reductions baseline (W2 mixer + low_rank=4 + BAND 128) -----
# Merges the wins banked so far into a single combined-reductions baseline:
#   - W2 per-scale mixer widths   ([0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
#   - low_rank=4
#   - BAND 128 lifting structural compression
# Establishes the new reference BPB that subsequent compression sweeps
# (embedding, MLP, mixer Linear, FwPKM) compare against — not the old 1.2361.
#
# Composability prediction (linear sum-of-individual-costs):
#   1.2342 (LR16 ref) + (1.2437 - 1.2342)  W2  + (1.2508 - 1.2342) BAND128
#   = 1.2342 + 0.0095 + 0.0166
#   = 1.2603 (linear-cost prediction)
# At 70-80% composability (per SparseGPT-line literature): ~1.2540 - 1.2560.
# Empirical ground truth from this run determines the new baseline.
#
# Subsequent compression sweeps use this BPB as their +Δ reference.
# run_ablation "CB combined-reductions baseline (W2 + low_rank=4 + BAND128)" \
#     "$BASE_PATCH_1EP" \
#     '{"per_scale_mixer_widths": [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], "low_rank": 4, "lifting_offdiag_structure": "banded", "lifting_band_width": 128}' \
#     "CB: combined-reductions baseline (W2 + low_rank=4 + BAND128) (1ep, L=7)"


# ---- 8. Encoder-decoder embedding sweep -------------------------------------
# Tests the encoder-decoder embedding architecture (V → C_emb via embedding
# lookup, C_emb → C via learnable decoder, ... model ..., C → C_emb via
# learnable encoder, C_emb → V via tied embedding.weight^T). The decoder and
# encoder are SEPARATE learnable matrices; the model's nonlinearities (GELU,
# gating, lifting cascade) break the linear-inverse symmetry that would
# otherwise allow sharing the same matrix in both directions.
#
# All runs layer on top of the combined-reductions baseline (W2 + low_rank=4
# + BAND 128), so the comparison vs CB isolates the embedding-compression
# effect.
#
# Param counts at C=2048, V=50257 (full embedding = 102.93M dense):
#   C_emb=128  -> 6.43M + 0.26M + 0.26M  ≈   6.95M  (93% reduction)
#   C_emb=256  -> 12.87M + 0.53M + 0.53M ≈  13.92M  (87% reduction)
#   C_emb=512  -> 25.73M + 1.05M + 1.05M ≈  27.83M  (73% reduction)
#   C_emb=1024 -> 51.46M + 2.10M + 2.10M ≈  55.66M  (46% reduction)
#   C_emb=2048 -> 102.93M + 4.20M + 4.20M ≈ 111.33M (no compression; full-rank
#                  refinement around the standard embedding for capacity ablation)
#
# Compare 1-epoch BPB delta against CB (combined-reductions baseline). The
# elbow on the recovery-vs-density curve identifies the production sweet spot.
ED_BASE_PATCH='{"per_scale_mixer_widths": [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], "low_rank": 4, "lifting_offdiag_structure": "banded", "lifting_band_width": 128, "sparse_encoder_decoder_embedding": true}'

# run_ablation "ED128 encoder-decoder embedding C_emb=128" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH'''); b['sparse_encoder_decoder_embedding_C']=128; print(json.dumps(b))")" \
#     "ED128: encoder-decoder embedding C_emb=128, 93% reduction (1ep, L=7)"

# run_ablation "ED256 encoder-decoder embedding C_emb=256" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH'''); b['sparse_encoder_decoder_embedding_C']=256; print(json.dumps(b))")" \
#     "ED256: encoder-decoder embedding C_emb=256, 87% reduction (1ep, L=7)"

# run_ablation "ED512 encoder-decoder embedding C_emb=512" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH'''); b['sparse_encoder_decoder_embedding_C']=512; print(json.dumps(b))")" \
#     "ED512: encoder-decoder embedding C_emb=512, 73% reduction (1ep, L=7)"

# run_ablation "ED1024 encoder-decoder embedding C_emb=1024" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH'''); b['sparse_encoder_decoder_embedding_C']=1024; print(json.dumps(b))")" \
#     "ED1024: encoder-decoder embedding C_emb=1024, 46% reduction (1ep, L=7)"

# run_ablation "ED2048 encoder-decoder embedding C_emb=C=2048 (full-rank refinement)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH'''); b['sparse_encoder_decoder_embedding_C']=2048; print(json.dumps(b))")" \
#     "ED2048: encoder-decoder embedding C_emb=2048 (full-rank refinement, no compression) (1ep, L=7)"


# C_emb > C expansion probes (tied). ED2048 (C_emb = C, full-rank refinement)
# landed at +0.0022 BPB vs CB — essentially free, but the +8.39M decoder/encoder
# params learned close to identity. Going past C_emb = C makes the decoder a
# many-to-few bottleneck (C_emb → C), which is structurally different from the
# square C_emb = C case. Tests whether the bottleneck arrangement unlocks any
# additional BPB. Pessimistic ceiling: ~−0.03 vs CB at ED8192.
#   C_emb=4096 -> 205.85M + 8.39M + 8.39M ≈ 222.63M (216% of dense V·C)
#   C_emb=8192 -> 411.70M + 16.78M + 16.78M ≈ 445.26M (433% of dense V·C)
# Train VRAM (extrapolated from ED tied scaling, ~0.58 MiB/unit C_emb):
#   ED4096 ≈ 24.3 GB; ED8192 ≈ 26.7 GB — both fit comfortably on a 5090.
# run_ablation "ED4096 encoder-decoder embedding C_emb=4096 (expansion, 2x C)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH'''); b['sparse_encoder_decoder_embedding_C']=4096; print(json.dumps(b))")" \
#     "ED4096: encoder-decoder embedding C_emb=4096 (capacity probe, 2x C) (1ep, L=7)"

# run_ablation "ED8192 encoder-decoder embedding C_emb=8192 (expansion, 4x C)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH'''); b['sparse_encoder_decoder_embedding_C']=8192; print(json.dumps(b))")" \
#     "ED8192: encoder-decoder embedding C_emb=8192 (capacity probe, 4x C) (1ep, L=7)"


# Untied LM head series: same C_emb sweep but with tie_embedding_to_lm_head=false.
# In the untied case:
#   - Encoder is still allocated (it bridges C → C_emb on the output path,
#     required regardless of tying)
#   - V projection uses a SEPARATE output_embedding matrix (V × C_emb) instead
#     of reusing the input embedding's matrix
# Net effect vs the tied series: adds V·C_emb params for the separate output
# embedding (e.g., +25.73M at C_emb=512). Total stack params:
#   C_emb=128  -> 6.43 + 0.26 + 0.26 + 6.43  ≈  13.38M  (87% reduction vs V·C)
#   C_emb=256  -> 12.87 + 0.53 + 0.53 + 12.87 ≈  26.79M  (74% reduction)
#   C_emb=512  -> 25.73 + 1.05 + 1.05 + 25.73 ≈  53.56M  (48% reduction)
#   C_emb=1024 -> 51.46 + 2.10 + 2.10 + 51.46 ≈ 107.13M  (+4% over V·C)
#   C_emb=2048 -> 102.93 + 4.20 + 4.20 + 102.93 ≈ 214.26M  (+108% over V·C)
# This isolates the cost of weight tying: how much does sharing the V projection
# matrix (input ↔ output) cost us at each embedding-dim setting?
ED_BASE_PATCH_UNTIED='{"per_scale_mixer_widths": [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], "low_rank": 4, "lifting_offdiag_structure": "banded", "lifting_band_width": 128, "sparse_encoder_decoder_embedding": true, "tie_embedding_to_lm_head": false}'

# run_ablation "ED128_untied encoder-decoder embedding C_emb=128 (untied LM head)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH_UNTIED'''); b['sparse_encoder_decoder_embedding_C']=128; print(json.dumps(b))")" \
#     "ED128_untied: encoder-decoder embedding C_emb=128 with untied LM head (1ep, L=7)"

# run_ablation "ED256_untied encoder-decoder embedding C_emb=256 (untied LM head)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH_UNTIED'''); b['sparse_encoder_decoder_embedding_C']=256; print(json.dumps(b))")" \
#     "ED256_untied: encoder-decoder embedding C_emb=256 with untied LM head (1ep, L=7)"

# run_ablation "ED512_untied encoder-decoder embedding C_emb=512 (untied LM head)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH_UNTIED'''); b['sparse_encoder_decoder_embedding_C']=512; print(json.dumps(b))")" \
#     "ED512_untied: encoder-decoder embedding C_emb=512 with untied LM head (1ep, L=7)"

# run_ablation "ED1024_untied encoder-decoder embedding C_emb=1024 (untied LM head)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH_UNTIED'''); b['sparse_encoder_decoder_embedding_C']=1024; print(json.dumps(b))")" \
#     "ED1024_untied: encoder-decoder embedding C_emb=1024 with untied LM head (1ep, L=7)"

# run_ablation "ED2048_untied encoder-decoder embedding C_emb=C=2048 (untied LM head)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH_UNTIED'''); b['sparse_encoder_decoder_embedding_C']=2048; print(json.dumps(b))")" \
#     "ED2048_untied: encoder-decoder embedding C_emb=2048 with untied LM head (1ep, L=7)"

# C_emb > C expansion probes (untied). Doubles the embedding table cost vs tied
# (V × C_emb input + V × C_emb output_embedding, each with its own Adagrad
# state), so VRAM is much tighter. Both fit on a 5090, but ED8192_untied is
# close to the practical ceiling — if framework overhead pushes it over, fall
# back to gradient_checkpointing or accept the tied-only datapoint at C_emb=8192.
#   C_emb=4096 -> 205.85 + 8.39 + 8.39 + 205.85 ≈ 428.48M
#   C_emb=8192 -> 411.70 + 16.78 + 16.78 + 411.70 ≈ 856.96M
# Train VRAM (extrapolated, ~0.9 MiB/unit C_emb untied):
#   ED4096_untied ≈ 25.7 GB; ED8192_untied ≈ 29.4 GB (tight).
# run_ablation "ED4096_untied encoder-decoder embedding C_emb=4096 (untied LM head, expansion)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH_UNTIED'''); b['sparse_encoder_decoder_embedding_C']=4096; print(json.dumps(b))")" \
#     "ED4096_untied: encoder-decoder embedding C_emb=4096 with untied LM head (1ep, L=7)"

# run_ablation "ED8192_untied encoder-decoder embedding C_emb=8192 (untied LM head, expansion)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$ED_BASE_PATCH_UNTIED'''); b['sparse_encoder_decoder_embedding_C']=8192; print(json.dumps(b))")" \
#     "ED8192_untied: encoder-decoder embedding C_emb=8192 with untied LM head (1ep, L=7)"




# Uncomment the below once the embedding tests are complete.
# Uncomment the below once the embedding tests are complete.
# Uncomment the below once the embedding tests are complete.
# Uncomment the below once the embedding tests are complete.
# Uncomment the below once the embedding tests are complete.
# Uncomment the below once the embedding tests are complete.
# Uncomment the below once the embedding tests are complete.



PQEMB_BASE_1EP_PATCH='{"sparse_pq_embedding_enabled": true, "sparse_pq_embedding_density": 0.1}'

# run_ablation "PQ_EMB10_smallest_q (p=18,q=2,d=0.10,smallest_q)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_mode']='smallest_q'; print(json.dumps(b))")" \
#     "PQ_EMB10_smallest_q: sparse embedding d=0.10 mode=smallest_q (1ep, levels=7)"

# run_ablation "PQ_EMB10_structural (p=12,q=8,d=0.10,structural)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_mode']='structural'; print(json.dumps(b))")" \
#     "PQ_EMB10_structural: sparse embedding d=0.10 mode=structural (1ep, levels=7)"



# ---- Test 1 at 1 epoch ------------------------------------------------------
# Test 1 config from the (Complete) Combined Parameter Reduction section, run
# at 1 epoch instead of 5. Gives the apples-to-apples 1-epoch pairing against
# R0 (1ep) for the VRAM-reallocation comparison, isolating the architectural
# delta from the epoch-budget delta. Note: R0 at 5 epochs is NOT queued here —
# it already exists at logs/wikitext-103_2026-05-03_02-13-07/log.txt
# (1.0974 sliding BPB, 392.91M params, 23.4 GiB train VRAM).
#
# Test 1 config (from logs/wikitext-103_2026-05-01_06-33-48/log.txt):
#   epochs=5 -> 1, micro_batch_size=8, block_size=256, levels=5,
#   per_scale_mixer_widths=[1.0×3, 0.5×3] (6 entries for levels=5),
#   wavelet_crawl=True, lifting_offdiag_structure="none" (no BAND128).
#   The four reductions (mlp_expansion=10, pkm_enabled=False,
#   fwpkm_num_keys=8281, tie_embedding_to_lm_head=True) come from
#   BASE_PATCH_1EP / config.json defaults.
# run_ablation "Test1_1ep parameter-reduction config at 1 epoch" \
#     "$BASE_PATCH_1EP" \
#     '{"micro_batch_size": 8, "block_size": 256, "levels": 5, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], "wavelet_crawl": true, "lifting_offdiag_structure": "none"}' \
#     "Test1_1ep: parameter-reduction config (Test 1) at 1 epoch (bs=256, levels=5, wavelet_crawl=True, no BAND128)"


PQEMB_BASE_1EP_PATCH='{"sparse_pq_embedding_enabled": true, "sparse_pq_embedding_density": 0.1}'


# ---- PQ_EMB25 with structural mode -----------------------------------------
# At d=25%, only one valid (p, q) = (6, 2) — smallest_q and structural converge.
# Mode label is cosmetic at this density. Inherits CB stack from BASE_PATCH_1EP.
# run_ablation "PQ_EMB25_structural (p=6,q=2,d=0.25,structural==smallest_q)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_density']=0.25; b['sparse_pq_embedding_mode']='structural'; print(json.dumps(b))")" \
#     "PQ_EMB25_structural: sparse embedding d=0.25 (modes converge at this density) (1ep, levels=7)"


# ---- PQ_EMB40 with structural mode -----------------------------------------
# At d=40%, only one valid (p, q) = (3, 2) — smallest_q and structural converge.
# d=40% is the maximum density the (p, q) scheme can achieve for C=2048
# (constraints: q | C, p does NOT divide C, p,q > 1; densities 50%, 67%, 75%
# have no valid candidates). PQ_EMB40 substitutes for the 50% / 67% / 75%
# slots in the original sweep plan.
# run_ablation "PQ_EMB40_structural (p=3,q=2,d=0.40,structural==smallest_q)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "import json; b=json.loads('''$PQEMB_BASE_1EP_PATCH'''); b['sparse_pq_embedding_density']=0.40; b['sparse_pq_embedding_mode']='structural'; print(json.dumps(b))")" \
#     "PQ_EMB40_structural: sparse embedding d=0.40 (modes converge; max density for C=2048) (1ep, levels=7)"



# MLP_BASE_1EP_PATCH='{}'  # MLP-compression flags layered per-run below
#
# ---- DEPRECATED: MLP Structural Compression sweep --------------------------
# The 12 MLP_BAND/BD/PQ runs below are commented out. Mask-based MLP compression
# was deprecated in favor of dense compression directions (smaller mlp_expansion,
# smaller C) — see the (Deprecated) MLP Structural Compression section in
# README → runs.md for the rationale. NB lifting (T-lower) keeps mask-based
# only on the wavelet cascade where research-insight value is high.

# # 25% density: BAND W=256 (per-block density 25.05%), BD b=512 (25.0%), (p,q) d=0.25 (only valid is p=6, q=2)
# run_ablation "MLP_BAND25 (W=256)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":256}')")" \
#     "MLP_BAND25: MLP banded W=256 per (C,C) block, ~25% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_BD25 (b=512)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":512}')")" \
#     "MLP_BD25: MLP block_diagonal b=512 per (C,C) block, ~25% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_PQ25 (d=0.25 -> p=6,q=2)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.25,\"mlp_pq_mode\":\"structural\"}')")" \
#     "MLP_PQ25: MLP (p,q) striding at d=0.25 structural (p=6,q=2) (1ep, levels=7)"
#
# # 12.5% density: BAND W=128, BD b=256, (p,q) d=0.125 -> p=12, q=4
# run_ablation "MLP_BAND125 (W=128)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":128}')")" \
#     "MLP_BAND125: MLP banded W=128 per (C,C) block, ~12.5% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_BD125 (b=256)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":256}')")" \
#     "MLP_BD125: MLP block_diagonal b=256 per (C,C) block, 12.5% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_PQ125 (d=0.125 -> p=12,q=4)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.125,\"mlp_pq_mode\":\"structural\"}')")" \
#     "MLP_PQ125: MLP (p,q) striding at d=0.125 structural (p=12,q=4) (1ep, levels=7)"
#
# # 6.25% density: BAND W=64, BD b=128, (p,q) d=0.0625 -> p=24, q=8
# run_ablation "MLP_BAND0625 (W=64)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":64}')")" \
#     "MLP_BAND0625: MLP banded W=64 per (C,C) block, ~6.25% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_BD0625 (b=128)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":128}')")" \
#     "MLP_BD0625: MLP block_diagonal b=128 per (C,C) block, 6.25% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_PQ0625 (d=0.0625 -> p=24,q=8)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.0625,\"mlp_pq_mode\":\"structural\"}')")" \
#     "MLP_PQ0625: MLP (p,q) striding at d=0.0625 structural (p=24,q=8) (1ep, levels=7)"
#
# # 3.125% density: BAND W=32, BD b=64, (p,q) d=0.03125 -> p=48, q=16
# run_ablation "MLP_BAND03125 (W=32)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"banded\",\"mlp_band_width\":32}')")" \
#     "MLP_BAND03125: MLP banded W=32 per (C,C) block, ~3.17% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_BD03125 (b=64)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"block_diagonal\",\"mlp_block_size\":64}')")" \
#     "MLP_BD03125: MLP block_diagonal b=64 per (C,C) block, 3.125% W1+W2 density (1ep, levels=7)"
#
# run_ablation "MLP_PQ03125 (d=0.03125 -> p=48,q=16)" \
#     "$BASE_PATCH_1EP" \
#     "$(python -c "print('{\"mlp_offdiag_structure\":\"pq_strided\",\"mlp_pq_density\":0.03125,\"mlp_pq_mode\":\"structural\"}')")" \
#     "MLP_PQ03125: MLP (p,q) striding at d=0.03125 structural (p=48,q=16) (1ep, levels=7)"


# ---- NB verification run (1ep) ---------------------------------------------
# Establishes the New Baseline BPB at 1 epoch. NB = R0 + W2 mixer + T-lower
# wavelet off-diagonal masking. Replaces CB as the reference everything else
# compares against. Empty patch — uses BASE_PATCH_1EP defaults exactly.
# run_ablation "NB_1ep New Baseline verification (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{}' \
#     "NB_1ep: New Baseline = R0 + W2 + T-lower (1ep, levels=7) — reference BPB for downstream ablations"


# ---- Levels retry on NB stack -----------------------------------------------
# Previous attempts at levels=9 and levels=11 NaN'd at the L=1 / no-compression
# baseline (the levels=11 cascade-explosion cliff), and no stability fix cleared
# them. levels=13 also OOM'd without gradient_checkpointing on the uncompressed
# stack. The NB stack (R0 + W2 + T-lower) changes the math: T-lower lifting
# (50% mask, exact zeros at masked positions) plus low_rank=4 cap the spectral
# norm of each cascade matrix, which is exactly the property that determines
# whether the cascade stays bounded across deeper levels. Worth retrying as a
# quick test before the optimizer sweep — if T-lower's structural masking
# alone clears the cliff, that confirms the cascade-bandwidth hypothesis at
# low cost. If they still NaN, the cliff is downstream of structural masking
# (precision / Adagrad dynamics) and the optimizer sweep is the right next
# move.
#
# Compare 1-epoch BPB against NB's reference. levels=11 with bs=16384 leaves
# only 8 tokens at the coarsest scale; levels=13 leaves only 2 tokens —
# boundary effects may dominate even if they train; treat the BPB-vs-NB delta
# as a "did the cascade survive" signal, not a "is this a better config"
# signal.
# run_ablation "levels=9 retry on NB stack" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 9, "per_scale_mixer_widths": [0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25]}' \
#     "levels=9: retry on NB stack (1ep, per_scale_mixer_widths extended to 10 entries)"

# run_ablation "levels=11 retry on NB stack" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 11, "per_scale_mixer_widths": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]}' \
#     "levels=11: retry on NB stack (1ep, per_scale_mixer_widths extended to 12 entries)"

# run_ablation "levels=13 retry on NB stack" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 13, "per_scale_mixer_widths": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]}' \
#     "levels=13: retry on NB stack (1ep, per_scale_mixer_widths extended to 14 entries; coarsest scale has only 2 tokens at bs=16384)"


# ---- Levels retry on R0 + T-lower (no W2) ----------------------------------
# Disentangles which of NB's two ingredients (W2 mixer contraction vs T-lower
# lifting masking) actually carries the stability load at deep levels. These
# runs are the same as the NB retries above but with R0's original mixer
# widths [1.0×4, 0.5×4] (extended to match levels) instead of W2's [0.5×4,
# 0.25×4]. T-lower is preserved (inherited from BASE_PATCH_1EP).
#
# Reading guide:
#   - If NB and R0+T-lower both train: T-lower is doing the work; W2 is
#     decorative for stability. Could revisit dropping W2 for the modest BPB
#     savings of going back to R0 mixer widths.
#   - If NB trains but R0+T-lower NaNs: W2 is load-bearing; both are needed.
#   - If neither trains: T-lower alone isn't enough; need optimizer sweep.
# run_ablation "levels=9 retry on R0+T-lower" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 9, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]}' \
#     "levels=9: retry on R0+T-lower stack (1ep, per_scale_mixer_widths=[1.0x5, 0.5x5], T-lower lifting from NB defaults)"


# ---- B3_1ep: Baseline 3 verification (1ep) ---------------------------------
# Tests whether NB's BPB cost (vs Test 1) shrinks at Test 1's gradient-update
# advantage (~58,500 steps/epoch vs NB's ~7,300 at matched epoch budget). Drops
# to levels=5 since NB's levels=7 with bs=256 would leave only 256/2^7 = 2
# B3 = Test 1 + T-lower lifting − wavelet_crawl. The 1ep verification run
# uses BASE_PATCH_1EP unchanged (B3 is the default). Decision criteria:
# if best val lands within ~0.05 of Test 1 (1ep: 3.6393, 5ep: 3.3341), B3
# becomes the production stack — major reframing toward "Test 1 throughput
# + T-lower stability." If best val lands closer to the prior bs=16384
# stack's number (3.8561 at NB 1ep, etc.), the bs=16384 commitment is
# paying for something other than BPB.
# B3 redefined: B3 = Test 1 + wavelet_crawl=False (no T-lower). Architecturally
# minimal change from Test 1 — the prior T-lower-flavored B3 runs at 2026-05-09
# 19:28 (B3_1ep, BPB 1.2024 / 3.7198 / 6,947 MiB) and 21:06 (B3_L7_1ep,
# 393.03M dense) are preserved in the historical record but the runs below
# re-establish B3 on its new architecture.
run_ablation "B3_1ep Baseline 3 verification (1ep, redefined: Test 1 + wavelet_crawl=False)" \
    "$BASE_PATCH_1EP" \
    '{}' \
    "B3_1ep: Baseline 3 redefined = Test 1 + wavelet_crawl=False (1ep, bs=256, MBS=8, levels=5, R0 mixer widths; no T-lower)"

# B3_L7_1ep — same as B3_1ep but at levels=7 with R0 mixer pattern at depth 7
# ([1.0×4, 0.5×4]). Tests the boundary regime explicitly: bs=256 / 2^7 = 2
# tokens at the coarsest wavelet scale. If it trains and lands competitively
# vs B3_1ep (levels=5), the boundary doesn't bind at this regime. If it
# underperforms, levels=5 is the right choice for the bs=256 throughput
# regime. Per-width contractions (W2) deprecated — uses R0 mixer widths.
run_ablation "B3_L7_1ep Baseline 3 at levels=7 (boundary test: 2 tokens at coarsest scale)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]}' \
    "B3_L7_1ep: Baseline 3 at levels=7 (1ep, bs=256, MBS=8, R0 mixer widths [1.0x4, 0.5x4], no T-lower; boundary case — coarsest scale has only 2 tokens at bs=256)"

# 5-epoch confirmation of B3 — uses BASE_PATCH_5EP unchanged (B3 is the
# default). The matched-budget production-decision datapoint vs Test 1 5ep
# (3.3341 best val).
run_ablation "B3_5ep Baseline 3 at 5 epochs" \
    "$BASE_PATCH_5EP" \
    '{}' \
    "B3_5ep: Baseline 3 at 5 epochs (bs=256, MBS=8, levels=5, R0 mixer widths, no T-lower; production-decision datapoint vs Test 1 5ep)"

# run_ablation "levels=11 retry on R0+T-lower (no W2)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 11, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}' \
#     "levels=11: retry on R0+T-lower stack (1ep, per_scale_mixer_widths=[1.0x6, 0.5x6], T-lower lifting from NB defaults)"

# run_ablation "levels=13 retry on R0+T-lower (no W2)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 13, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}' \
#     "levels=13: retry on R0+T-lower stack (1ep, per_scale_mixer_widths=[1.0x7, 0.5x7], T-lower lifting from NB defaults; coarsest scale 2 tokens at bs=16384)"


# ---- W2 mixer contraction at 5 epochs on R0 baseline ------------------------
# 5-epoch confirmation of W2 [0.5×4, 0.25×4] on the R0 baseline (the missing
# 5ep test referenced in the (Complete) Per-Scale Mixer Width Contraction and
# Expansion section's decision text). Goal: verify the +0.0076 BPB cost from
# the 1-epoch test (logs/wikitext-103_2026-05-05_05-48-40, BPB 1.2437) holds
# at 5 epochs against R0's 5ep BPB of 1.0974.
#
# Note: BASE_PATCH_5EP has W2 mixer + low_rank=4 + T-lower lifting (NB stack).
# The override turns OFF T-lower (lifting_offdiag_structure="none") to recover
# the R0-without-lifting-mask stack, leaving the W2 mixer setting in place.
# run_ablation "Mix_R0_0.5_0.25 mixer contraction [0.5x4, 0.25x4] on R0 baseline (5 epochs)" \
#     "$BASE_PATCH_5EP" \
#     '{"lifting_offdiag_structure": "none"}' \
#     "Mix_R0_0.5_0.25: per_scale_mixer_widths=[0.5x4, 0.25x4] on R0 baseline (5ep, no lifting mask)"


# ---- Mixer contraction sweep on NB stack (1ep each) -------------------------
# Tighter per_scale_mixer_widths than W2's [0.5×4, 0.25×4], applied on top of
# the NB stack. The 0.1/0.05 contraction previously NaN'd at step 1250 on the
# uncompressed R0 stack (logs/wikitext-103_2026-05-05_04-37-47); NB's
# T-lower+low_rank=4 spectral-norm constraint should make tighter mixers
# tractable now. Compare 1-epoch BPB against NB's reference. Production
# candidate: tightest setting that lands within ~0.005 BPB of NB.
# run_ablation "Mix_NB_0.4_0.2 mixer contraction [0.4x4, 0.2x4] on NB" \
#     "$BASE_PATCH_1EP" \
#     '{"per_scale_mixer_widths": [0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]}' \
#     "Mix_NB_0.4_0.2: per_scale_mixer_widths=[0.4x4, 0.2x4] on NB (1ep)"

# run_ablation "Mix_NB_0.3_0.15 mixer contraction [0.3x4, 0.15x4] on NB" \
#     "$BASE_PATCH_1EP" \
#     '{"per_scale_mixer_widths": [0.3, 0.3, 0.3, 0.3, 0.15, 0.15, 0.15, 0.15]}' \
#     "Mix_NB_0.3_0.15: per_scale_mixer_widths=[0.3x4, 0.15x4] on NB (1ep)"

# run_ablation "Mix_NB_0.25_0.125 mixer contraction [0.25x4, 0.125x4] on NB" \
#     "$BASE_PATCH_1EP" \
#     '{"per_scale_mixer_widths": [0.25, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.125]}' \
#     "Mix_NB_0.25_0.125: per_scale_mixer_widths=[0.25x4, 0.125x4] on NB (1ep)"

# run_ablation "Mix_NB_0.1_0.05 mixer contraction [0.1x4, 0.05x4] on NB (retry of previously NaN-failed config)" \
#     "$BASE_PATCH_1EP" \
#     '{"per_scale_mixer_widths": [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]}' \
#     "Mix_NB_0.1_0.05: per_scale_mixer_widths=[0.1x4, 0.05x4] on NB (1ep, retry of previously NaN'd config)"


# ---- Cross-level group sharing on B3 stack (DEFERRED) -----------------------
# Original premise: retry lifting_level_sharing=true on the T-lower-flavored
# B3 — T-lower's per-matrix spectral-norm cap was the proposed safety net
# that might tame the cross-level coupling that NaN'd this config previously.
#
# B3 has been redefined to drop T-lower (no production parameter savings; mask
# overhead actually slightly increased train VRAM). Without T-lower in the
# default stack, the original premise no longer holds — running
# lifting_level_sharing=true on the unguarded B3 would just reproduce the
# prior NaN failure. Defer until either (a) T-lower is reintroduced as a
# stability tool for a specific run, or (b) a different stability mechanism
# (e.g. Muon-orthogonalized updates from the optimizer sweep) is in place.
# run_ablation "LLS_B3 lifting_level_sharing on B3 stack (retry of previously NaN-failed config)" \
#     "$BASE_PATCH_1EP" \
#     '{"lifting_level_sharing": true, "lifting_offdiag_structure": "lower_triangular"}' \
#     "LLS_B3: lifting_level_sharing=true on B3+T-lower stack (1ep, retry of previously NaN'd config; T-lower re-added as stability scaffold for this run only)"


echo ""
echo "============================================================"
echo "=== Queue complete."
echo "===   1) DBD     (already complete)"
echo "===   2) E5_5ep  (already complete)"
echo "===   3) LR16_5ep  (already complete)"
echo "===   4) M1..M4 magnitude-pruned off-diagonal — compare BPB delta vs 1.2361"
echo "===   5) M1r..M4r random controls at matched densities — compare BPB delta vs 1.2361"
echo "===   6) Structural priors: T_upper/lower, BD64/256, BAND64/256, MON32/64"
echo "===      — compare BPB delta vs 1.2361 alongside per-param efficiency"
echo "===   7) Combined-reductions baseline (W2 mixer + low_rank=4 + BAND128) at 1ep"
echo "===      — establishes the new reference BPB for sections 8 onwards"
echo "===   8) Encoder-decoder embedding sweep: C_emb in {128, 256, 512, 1024, 2048}"
echo "===      5 tied + 5 untied (10 runs) — compare BPB delta vs section-7 baseline;"
echo "===      tied series locates the compression elbow; untied series measures the cost of weight tying"
echo "===   9) Sparse (p, q) phantom-token embedding sweep at d=0.10"
echo "===      smallest_q (p=18,q=2) and structural (p=12,q=8) — compare BPB delta vs section-7 baseline"
echo "===  10) MLP structural compression: BAND/BD/PQ at densities 25%, 12.5%, 6.25%, 3.125%"
echo "===      12 runs (4 densities × 3 structures) — compare BPB delta vs section-7 baseline"
echo "===  11) Levels retry on CB stack: levels=9, levels=11 — does BAND128 + low_rank=4"
echo "===      clear the cascade-explosion cliff that no stability fix did at the uncompressed baseline?"
echo "===  12) 5-epoch confirmation of chosen winner — compare BPB delta vs 1.0974"
echo "===      (placeholder; uncomment after sections 4-11 complete and a winner is chosen)"
echo "==="
echo "=== Next: combine surviving winners into a new 5-epoch baseline."
echo "=== Implementation gating: sections 4, 5, and 6 all require model.py to"
echo "===   support the unified lifting_offdiag_structure flag (Option B) and"
echo "===   its per-structure parameters. Verify each variant's 'Shared lifting'"
echo "===   line moves in the expected direction before reading BPB."
echo "============================================================"
