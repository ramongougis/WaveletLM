#!/bin/bash
# PTQ sweep against the 5-epoch best-run checkpoint.
# Per variant: BPB (via train.py --benchmark_only) + a short generation
# (via generate.py) for tok/s and a coherence spot-check.
#
# Output: logs/<run>/ptq/NN_name.log (one per variant) + ptq/summary.txt
# Root config.json is restored at exit (trap) so 3-seed runs can resume safely.

set -e

CKPT_DIR="logs/wikitext-103_2026-04-19_13-16-24"
CKPT="$CKPT_DIR/best_model.pt"
OUT_DIR="$CKPT_DIR/ptq"
mkdir -p "$OUT_DIR"

# Restore root config to a safe state on exit (benchmark off, quantize off)
cleanup() {
    python - <<'EOF'
import json
p = 'config.json'
c = json.load(open(p))
c['benchmark_only'] = False
c['benchmark_run_dir'] = ''
c['quantize_enabled'] = False
# Reset bit-widths to sensible defaults
c['quantize_mixer_coarse_bits'] = 8
c['quantize_mixer_mid_bits'] = 4
c['quantize_mixer_fine_bits'] = 2
c['quantize_mlp_bits'] = 4
c['quantize_lifting_bits'] = 16
c['quantize_embedding_bits'] = 8
json.dump(c, open(p, 'w'), indent=4)
EOF
}
trap cleanup EXIT

# Patch root config for a single variant's benchmark run
set_config() {
    local ENABLED="$1" COARSE="$2" MID="$3" FINE="$4" MLP="$5" LIFT="$6" EMB="$7"
    python - "$ENABLED" "$COARSE" "$MID" "$FINE" "$MLP" "$LIFT" "$EMB" "$CKPT_DIR" <<'EOF'
import json, sys
enabled, coarse, mid, fine, mlp, lift, emb, ckpt_dir = sys.argv[1:9]
p = 'config.json'
c = json.load(open(p))
c['benchmark_only'] = True
c['benchmark_run_dir'] = ckpt_dir
c['quantize_enabled'] = (enabled.lower() == 'true')
c['quantize_mixer_coarse_bits'] = int(coarse)
c['quantize_mixer_mid_bits']    = int(mid)
c['quantize_mixer_fine_bits']   = int(fine)
c['quantize_mlp_bits']          = int(mlp)
c['quantize_lifting_bits']      = int(lift)
c['quantize_embedding_bits']    = int(emb)
json.dump(c, open(p, 'w'), indent=4)
EOF
}

run_variant() {
    local NAME="$1" ENABLED="$2"
    local COARSE="$3" MID="$4" FINE="$5" MLP="$6" LIFT="$7" EMB="$8"
    local LOG="$OUT_DIR/${NAME}.log"

    echo ""
    echo "============================================================"
    echo "=== PTQ variant: $NAME"
    echo "===   mixer=${COARSE}/${MID}/${FINE}  mlp=${MLP}  lift=${LIFT}  emb=${EMB}"
    echo "============================================================"
    set_config "$ENABLED" "$COARSE" "$MID" "$FINE" "$MLP" "$LIFT" "$EMB"

    # 1) Benchmark — BPB + model size + compression ratio
    python train.py 2>&1 | tee "$LOG"

    # 2) Short generation — tok/s, qualitative coherence (not --strategies)
    if [ "$ENABLED" = "True" ]; then
        python generate.py --checkpoint "$CKPT" \
            --quantize \
            --quantize_mixer_coarse_bits "$COARSE" \
            --quantize_mixer_mid_bits "$MID" \
            --quantize_mixer_fine_bits "$FINE" \
            --quantize_mlp_bits "$MLP" \
            --quantize_lifting_bits "$LIFT" \
            --quantize_embedding_bits "$EMB" \
            --num_tokens 128 --n 1 --metrics 2>&1 | tee -a "$LOG"
    else
        python generate.py --checkpoint "$CKPT" \
            --num_tokens 128 --n 1 --metrics 2>&1 | tee -a "$LOG"
    fi
}

# ===================== Uniform quantization =====================
run_variant "01_baseline_fp16"         "False" 16 16 16 16 16 16
run_variant "02_uniform_8bit"          "True"   8  8  8  8 16  8
run_variant "03_uniform_4bit"          "True"   4  4  4  4 16  4

# ===================== Per-scale mixed precision =====================
run_variant "04_mixed_default"         "True"   8  4  2  4 16  8
run_variant "05_mixed_conservative"    "True"   8  8  4  4 16  8
run_variant "06_mixed_higher_mlp"      "True"   8  4  2  8 16  8
run_variant "07_mixed_quant_lifting"   "True"   8  4  2  4  8  8
run_variant "08_mixed_aggressive"      "True"   4  4  2  4  8  4
run_variant "09_mixed_aggressive_emb"  "True"   8  4  2  4 16  4

# ===================== Component isolation (rest at 16) =====================
run_variant "10_mixer_only_8"          "True"   8  8  8 16 16 16
run_variant "11_mixer_only_4"          "True"   4  4  4 16 16 16
run_variant "12_mlp_only_8"            "True"  16 16 16  8 16 16
run_variant "13_mlp_only_4"            "True"  16 16 16  4 16 16
run_variant "14_embedding_only_8"      "True"  16 16 16 16 16  8
run_variant "15_embedding_only_4"      "True"  16 16 16 16 16  4
run_variant "16_lifting_only_8"        "True"  16 16 16 16  8 16

# ===================== Summary =====================
SUMMARY="$OUT_DIR/summary.txt"
{
    printf "%-28s | %-8s | %-8s | %-14s | %-10s\n" "variant" "BPB(sl)" "BPB(nov)" "size(MiB)" "compress"
    printf "%-28s-+-%-8s-+-%-8s-+-%-14s-+-%-10s\n" "----------------------------" "--------" "--------" "--------------" "----------"
    for log in "$OUT_DIR"/*.log; do
        [ "$(basename "$log")" = "summary.txt" ] && continue
        NAME=$(basename "$log" .log)
        BPB_SL=$(grep -oP 'BPB:\s+\K[0-9.]+' "$log" | tail -1)
        BPB_NOV=$(grep -oP 'BPB:\s+\K[0-9.]+' "$log" | head -1)
        SIZE=$(grep -oP 'Quantized:\s+\K[0-9.]+' "$log" | head -1)
        RATIO=$(grep -oP '\K[0-9.]+x compression' "$log" | head -1)
        printf "%-28s | %-8s | %-8s | %-14s | %-10s\n" "$NAME" "${BPB_SL:-—}" "${BPB_NOV:-—}" "${SIZE:-—}" "${RATIO:-fp16}"
    done
} > "$SUMMARY"

echo ""
echo "=== PTQ sweep complete ==="
echo "Per-variant logs: $OUT_DIR/NN_name.log"
echo "Summary table:    $SUMMARY"
cat "$SUMMARY"
