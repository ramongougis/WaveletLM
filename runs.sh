#!/bin/bash
# PG-19 pre-release run. Uses the best-of-3-seeds recipe (seed=1337, WD=1e-6,
# everything else unchanged from the WT-103 best config) on PG-19 for 1 epoch.
# The tokenizer auto-resolves to a 32K SentencePiece BPE trained on PG-19 train
# split; on first run this is trained and cached at .cache/pg19_sp32k.model
# (adds ~5–30 min one-time setup). The trained SP model bytes are embedded in
# the saved checkpoint so users who download best_model.pt from HF can run
# generate.py out of the box.
#
# After the PG-19 run finishes, config.json is reset to the WT-103 best-run
# defaults (dataset=wikitext-103, epochs=5) so the repo's default always
# reflects the released headline recipe.

set_keys() {
    python -c "
import json
cfg = json.load(open('config.json'))
patch = json.loads('''$1''')
cfg.update(patch)
json.dump(cfg, open('config.json', 'w'), indent=4)
"
}

echo ""
echo "============================================================"
echo "=== PG-19 pre-release run (1 epoch, SentencePiece auto)"
echo "============================================================"
set_keys '{"dataset": "pg19", "epochs": 1, "eval_interval": 2500}'
python train.py
LATEST_CKPT=$(ls -dt logs/pg19_*/best_model.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    python generate.py --checkpoint "$LATEST_CKPT"
    python generate.py --checkpoint "$LATEST_CKPT" --strategies
fi

# Reset config.json to the WT-103 best-run defaults before committing, so the
# repo's default always reflects the released headline recipe.
set_keys '{"dataset": "wikitext-103", "epochs": 5, "eval_interval": 250}'

git add .
git commit --no-edit -m "PG-19 pre-release benchmark run"
git pull --no-edit
git push

echo ""
echo "============================================================"
echo "=== PG-19 run complete"
echo "===   Update runs.md with BPB/PPL for the PG-19 table,"
echo "===   then proceed to the HuggingFace upload."
echo "============================================================"
