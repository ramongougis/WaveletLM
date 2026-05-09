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


# # ---- B3 Baseline (deprecated) ----
# run_ablation "T2_1ep Baseline 3 verification (1ep, redefined: Test 1 + wavelet_crawl=False)" \
#     "$BASE_PATCH_1EP" \
#     '{}' \
#     "B3_1ep: Baseline 3 redefined = Test 1 + wavelet_crawl=False (1ep, bs=256, MBS=8, levels=5, R0 mixer widths; no T-lower)"

# # B3_L7_1ep — same as B3_1ep but at levels=7 with R0 mixer pattern at depth 7
# # ([1.0×4, 0.5×4]). Tests the boundary regime explicitly: bs=256 / 2^7 = 2
# # tokens at the coarsest wavelet scale. If it trains and lands competitively
# # vs B3_1ep (levels=5), the boundary doesn't bind at this regime. If it
# # underperforms, levels=5 is the right choice for the bs=256 throughput
# # regime. Per-width contractions (W2) deprecated — uses R0 mixer widths.
# run_ablation "B3_L7_1ep Baseline 3 at levels=7 (boundary test: 2 tokens at coarsest scale)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]}' \
#     "B3_L7_1ep: Baseline 3 at levels=7 (1ep, bs=256, MBS=8, R0 mixer widths [1.0x4, 0.5x4], no T-lower; boundary case — coarsest scale has only 2 tokens at bs=256)"

# # 5-epoch confirmation of B3 — uses BASE_PATCH_5EP unchanged (B3 is the
# # default). The matched-budget production-decision datapoint vs Test 1 5ep
# # (3.3341 best val).
# run_ablation "B3_5ep Baseline 3 at 5 epochs" \
#     "$BASE_PATCH_5EP" \
#     '{}' \
#     "B3_5ep: Baseline 3 at 5 epochs (bs=256, MBS=8, levels=5, R0 mixer widths, no T-lower; production-decision datapoint vs Test 1 5ep)"



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
