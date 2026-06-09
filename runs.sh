#!/bin/bash

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
    git pull --no-rebase --no-edit -X theirs || true
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

pause_for_decision() {
    # Coordinate-descent gate: after a dropout pair has run AND been pushed,
    # halt so the operator can read the two results and set the winning value
    # for that dropout type. The value is read from the terminal and assigned
    # to the named carried-forward variable IN MEMORY (editing runs.sh on disk
    # would NOT work — bash already executed the DO_* assignment lines and won't
    # re-read them). Empty input keeps the current value.
    #   $1 = variable name to set (e.g. DO_PROJ); empty = final pause, no var.
    #   $2 = message.
    local VARNAME="$1"
    local MSG="$2"
    echo ""
    echo "############################################################"
    echo "### COORDINATE-DESCENT PAUSE"
    echo "### ${MSG}"
    if [ -n "$VARNAME" ]; then
        echo "### Current ${VARNAME}=${!VARNAME}"
    fi
    echo "### Both results above are pushed (read them, then decide)."
    echo "############################################################"
    if [ "${DROPOUT_AUTO:-0}" = "1" ]; then
        echo "[runs.sh] DROPOUT_AUTO=1 — skipping pause, keeping ${VARNAME:-N/A}=${!VARNAME:-}."
        return
    fi
    if [ -z "$VARNAME" ]; then
        echo "### Press Enter to finish (or Ctrl-C)."
        read -r _ </dev/tty || true
        return
    fi
    # Accept any of: "0.2" | "DO_EMB=0.2" | "dropout_emb=0.2" |
    # "dropout_embedding=0.2". Strip everything up to and including the last
    # '=', trim whitespace, and require the result to be a bare number — so a
    # mistyped key never gets assigned as a string and corrupts the JSON.
    while true; do
        local RAW=""
        printf "### Winning value for %s (e.g. 0.2, or DO_EMB=0.2; blank = keep %s): " \
            "$VARNAME" "${!VARNAME}" > /dev/tty
        read -r RAW </dev/tty || { echo "[runs.sh] no input; keeping ${VARNAME}=${!VARNAME}."; break; }
        # blank -> keep current
        if [ -z "${RAW//[[:space:]]/}" ]; then
            echo "[runs.sh] ${VARNAME} kept at ${!VARNAME}."
            break
        fi
        local VAL="${RAW##*=}"          # drop any "key=" prefix (last '=' wins)
        VAL="${VAL//[[:space:]]/}"      # strip all whitespace
        # Validate: integer or decimal (optionally signed), e.g. 0.2 / .09 / 1
        if [[ "$VAL" =~ ^-?([0-9]+(\.[0-9]*)?|\.[0-9]+)$ ]]; then
            printf -v "$VARNAME" '%s' "$VAL"
            echo "[runs.sh] ${VARNAME} set to ${!VARNAME}."
            break
        fi
        echo "[runs.sh] '${RAW}' is not a valid number — try again (or blank to keep)." > /dev/tty
    done
}

benchmark_only_run() {
    # Replay the post-training steps (test benchmark + two generation passes)
    # against an existing run directory's best_model.pt. No training, no model
    # state mutation, no new run dir.
    #
    # train.py is invoked in benchmark_only mode: it reads $TARGET_DIR/config.json
    # as the source of truth for every architectural key, only re-pinning
    # benchmark_only=true and benchmark_run_dir from our temp config. The root
    # config.json is never touched (build_run_config writes only to $TMP_CFG;
    # git_commit_push reverts any unexpected changes), and the run dir's saved
    # config.json is only READ. Downstream consumers of the checkpoint
    # (HF release, future benchmark replays) continue to see the original
    # training config as authoritative.
    local LABEL="$1"
    local TARGET_DIR="$2"
    local COMMIT_MSG="$3"

    echo ""
    echo "============================================================"
    echo "=== Benchmark-only: ${LABEL}"
    echo "===   Target run dir: ${TARGET_DIR}"
    echo "============================================================"

    if [ ! -d "$TARGET_DIR" ]; then
        echo "[runs.sh] ERROR: ${TARGET_DIR} does not exist; skipping benchmark-only run"
        return
    fi
    if [ ! -f "$TARGET_DIR/best_model.pt" ]; then
        echo "[runs.sh] ERROR: ${TARGET_DIR}/best_model.pt missing; skipping"
        return
    fi

    build_run_config "{\"benchmark_only\": true, \"benchmark_run_dir\": \"${TARGET_DIR}\"}"
    python train.py --config "$TMP_CFG"
    local TRAIN_EXIT=$?
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "[runs.sh] train.py (benchmark_only) exited with code $TRAIN_EXIT; continuing"
    fi

    echo ""
    echo "=== Measuring inference VRAM (fresh process, standard) for ${TARGET_DIR}"
    python generate.py --checkpoint "$TARGET_DIR/best_model.pt" || \
        echo "[runs.sh] generate.py (standard) exited non-zero; continuing"
    echo ""
    echo "=== Measuring inference VRAM (fresh process, --strategies) for ${TARGET_DIR}"
    python generate.py --checkpoint "$TARGET_DIR/best_model.pt" --strategies || \
        echo "[runs.sh] generate.py --strategies exited non-zero; continuing"

    git_commit_push "${COMMIT_MSG}"
}

# CARRIED-FORWARD CHOSEN VALUES — edit each at its pause before continuing:
DO_EMB=0.18    # emb pair done: 0.18 BPB 1.1307 (best of 3 on BPB) vs 0.20 1.1311, 0.22 1.1309 — all within noise; 0.18 chosen (lowest BPB)
DO_PROJ=0.09   # proj pair done: 0.09 BPB 1.1304 < 0.10 1.1307 < 0.11 1.1316 (monotonic; 0.09-vs-0.10 within noise but consistent). 0.09 chosen; edge of range — probe lower at higher L
DO_MIX=0.09    # mix pair done: 0.09 BPB 1.1295 < 0.10 1.1304 < 0.11 1.1311 (monotonic; 0.09-vs-0.10 = 0.0009, ~noise floor — strongest pair yet). 0.09 chosen; edge of range — probe lower at higher L
DO_MLP=0.10    # mlp pair done: incumbent 0.10 BPB 1.1295 < 0.09 1.1305 & 0.11 1.1309 (interior min on BPB — kept). NOTE val disagrees (0.09 best on val, monotonic) — metric split; revisit at higher L. NOT an edge-winner
DO_LM=0.216    # lm_head pair done: 0.216 BPB 1.1285 < 0.240 1.1295 < 0.264 1.1305 (monotonic, both metrics agree; 0.216-vs-0.240 = 0.0010 ~floor). 0.216 chosen; edge-winner ↓. FINAL STACK best point: 1.1285 (-0.0026 vs T4)
DO_COMMON='"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": true, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450'

# ---- Complex wavelets (invertible) -------------------------------------------
# Shift-invariance-motivated complex lifting (tools/complex_wavelets.py,
# plans/complex_wavelets.md). Invertible & tied: decompose -> full complex
# coefficients -> spectral mixer (sees re+im) -> tied reconstruct (Recon∘Decomp=I
# preserved). Phase mixed via complex_mixer_activation; mixer_depth=1 required.
# IMPORTANT: all arms run wavelet_crawl=FALSE (model.py hard-errors complex+crawl),
# so the matched real CONTROL is also crawl-off — the only cross-arm variable is
# the wavelet basis. NOT directly comparable to the crawl-on T4 baseline (1.1311);
# the in-section reference is the hm=4 control, not T4. Each complex variant is
# validated only if it beats that matched-param control.

# CW-split: invertible, split-complex mixer activation (real stack on re+im).
# run_ablation "T4_cwav_inv_split_1ep T4 complex wavelet invertible/split (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "complex", "complex_construction": "invertible", "complex_mixer_activation": "split"}' \
#     "T4_cwav_inv_split_1ep: invertible complex (tied, Recon∘Decomp=I), split-complex phase mixing; vs hm=4 control"

# # CW-modphase: invertible, modulus-phase activation (softplus-gated |z|, phase kept).
# run_ablation "T4_cwav_inv_modphase_1ep T4 complex wavelet invertible/modulus_phase (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "complex", "complex_construction": "invertible", "complex_mixer_activation": "modulus_phase"}' \
#     "T4_cwav_inv_modphase_1ep: invertible complex, softplus-gated magnitude phase-preserving mixing; vs hm=4 control"

# CW-modphase-tanh: modulus_phase with a tanh gate — BOUNDED |g|<=1 (stabilizer)
# and BIPOLAR (learns discrete π phase flips), init-shifted to start at ln2 (=
# softplus init) so it begins as standard positive scaling and learns into flips.
# run_ablation "T4_cwav_inv_modphase_tanh_1ep T4 complex wavelet invertible/modulus_phase tanh-gate (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "complex", "complex_construction": "invertible", "complex_mixer_activation": "modulus_phase", "complex_gate_activation": "tanh"}' \
#     "T4_cwav_inv_modphase_tanh_1ep: tanh-gated modulus_phase (bounded, bipolar π-flips); vs softplus variant + hm=4 control"

# # CW-control: matched-param real control. Invertible wavelet is 469.99M (complex
# # predict AND update, tied reconstruct); hidden_mult=4 = 469.91M, near-exact
# # match (ratio 1.000). The in-section reference both complex arms compare against.
# run_ablation "T4_cwav_control_hm4_1ep T4 real-wavelet control hidden_mult=4 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "lifting_hidden_mult": 4}' \
#     "T4_cwav_control_hm4_1ep: real wavelet hidden_mult=4 (~470M wavelet) — near-exact matched-param control for the invertible complex runs"

# # ---- Option C: complex MIXER, REAL wavelet -----------------------------------
# Separate experiment from the complex-wavelet basis: the wavelet stays real and
# exactly invertible; the complex machinery is a per-scale learned real↔complex
# projection around the FWHT/mixer (tools/complex_wavelets.RealToComplexProjection)
# + the complex spectral pass. Tests whether complex helps in the MIXER (spectral
# space, where phase is native) even though it regressed in the wavelet. Total
# 527.13M at T4 (+134.22M projections). crawl off for clean comparison; in-section
# reference is the hm=2 control below (510.38M, ~3% under — note slight under-match).

# # CM-split: complex mixer, split activation.
# run_ablation "T4_cmix_split_1ep T4 complex mixer (Option C) / split (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "complex_mixer": true, "complex_mixer_activation": "split"}' \
#     "T4_cmix_split_1ep: Option C — real wavelet + complex mixer (learned R↔C proj), split; vs hm=2 control"

# # CM-modphase: complex mixer, modulus_phase (softplus gate).
# run_ablation "T4_cmix_modphase_1ep T4 complex mixer (Option C) / modulus_phase (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "complex_mixer": true, "complex_mixer_activation": "modulus_phase", "complex_gate_activation": "softplus"}' \
#     "T4_cmix_modphase_1ep: Option C complex mixer, softplus-gated modulus_phase; vs hm=2 control"

# # CM-control: matched-param real control for Option C. hidden_mult=2 = 510.38M
# # (~3% under Option C's 527.13M; no integer hm matches exactly — hm=3 is 19% over).
# run_ablation "T4_cmix_control_hm2_1ep T4 real-wavelet control hidden_mult=2 (1ep)" \
#     "$BASE_PATCH_1EP" \
#     '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "lifting_hidden_mult": 2}' \
#     "T4_cmix_control_hm2_1ep: real wavelet hidden_mult=2 (510.38M) — matched-param control for Option C (~3% under)"

# ---- Mixer Transform Ablation -------------------------------------------------
# Replace the FWHT slot in the per-scale mixer with alternative orthonormal
# transforms (identity / DHT / DCT) at the T4 reference config. All are
# orthonormal, so they are amplitude-matched — the only difference is the BASIS
# the per-scale mixer operates in. The mixer's linear part is basis-absorbable;
# the element-wise gate is NOT, so this tests whether gating Walsh-frequencies
# (FWHT) beats gating raw channels (identity) or other bases (DHT/DCT). Run at
# the present LR (0.0225); per-transform LR re-probes follow if a transform is
# competitive but mis-tuned (see README 'Mixer Transform Ablation' + the
# Multi-Transform §'s learning-rate note). fwht arm is a fresh same-config
# control (should reproduce ~T4). Identity first — the cheap "does the slot
# matter at all?" probe. Butterfly (learned orthogonal) deferred to a follow-up.

# Identity: no transform — mixer operates in raw coefficient space.
run_ablation "T4_mt_identity_1ep T4 mixer-transform=identity / no transform (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "identity"}' \
    "T4_mt_identity_1ep: mixer-transform ablation — identity (no transform, mixer in raw coeff space); does the FWHT slot matter at all?"

# FWHT control: fresh same-config reference (crawl off, norms on, lr 0.0225).
run_ablation "T4_mt_fwht_1ep T4 mixer-transform=fwht control (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "fwht"}' \
    "T4_mt_fwht_1ep: mixer-transform ablation — FWHT control (same config as identity/dht/dct); in-section reference"

# DHT: Discrete Hartley (orthonormal, self-inverse).
run_ablation "T4_mt_dht_1ep T4 mixer-transform=dht / Hartley (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "dht"}' \
    "T4_mt_dht_1ep: mixer-transform ablation — DHT (Hartley) basis"

# DCT: DCT-II/III (orthonormal; inverse = transpose).
run_ablation "T4_mt_dct_1ep T4 mixer-transform=dct (1ep)" \
    "$BASE_PATCH_1EP" \
    '{"levels": 7, "per_scale_mixer_widths": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], "wavelet_crawl": false, "wavelet_decomp_norm": true, "wavelet_recon_norm": true, "lr": 0.02250, "min_lr": 0.000450, "wavelet_basis": "real", "mixer_transform": "dct"}' \
    "T4_mt_dct_1ep: mixer-transform ablation — DCT basis"
