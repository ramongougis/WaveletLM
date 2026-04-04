#!/bin/bash
# Generate samples for all completed training runs.
# Runs generate.py (standard + strategies) on each log folder
# that has both best_model.pt and "Training Peak VRAM:" in log.txt.

set -e

PROMPT="The history of"
NUM_TOKENS=512
SEED=1337

for dir in logs/*/; do
    log_file="${dir}log.txt"
    checkpoint="${dir}best_model.pt"
    config_file="${dir}config.json"

    # Skip if missing log, checkpoint, or config
    [ -f "$log_file" ] || continue
    [ -f "$checkpoint" ] || continue
    [ -f "$config_file" ] || continue

    # Skip if training didn't complete (check log.txt and benchmark.txt)
    if ! grep -q "Training Peak VRAM:" "$log_file" 2>/dev/null && \
       ! grep -q "Training VRAM:" "${dir}benchmark.txt" 2>/dev/null; then
        continue
    fi

    # Skip if already generated
    gen_file="${dir}generations.txt"
    if [ -f "$gen_file" ]; then
        echo "SKIP $dir (generations.txt exists)"
        continue
    fi

    echo "========================================"
    echo "Generating for: $dir"
    echo "========================================"

    {
        echo "=== GENERATION — Standard ==="
        echo "Prompt: $PROMPT"
        echo ""
        python generate.py \
            --checkpoint "$checkpoint" \
            --config "$config_file" \
            --prompt "$PROMPT" \
            --num_tokens $NUM_TOKENS \
            --seed $SEED
        echo ""
        echo ""
        echo "=== GENERATION — Strategies ==="
        echo "Prompt: $PROMPT"
        echo ""
        python generate.py \
            --checkpoint "$checkpoint" \
            --config "$config_file" \
            --prompt "$PROMPT" \
            --num_tokens $NUM_TOKENS \
            --seed $SEED \
            --strategies \
            --metrics
    } > "$gen_file" 2>&1

    echo "Saved to $gen_file"
    echo ""
done

echo "Done."
