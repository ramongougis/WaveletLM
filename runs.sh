for EPOCHS in 2 3 4 5 6 7 8 9 10; do
    python -c "
import json
cfg = json.load(open('config.json'))
cfg['C'] = 512
cfg['lr'] = 0.01
cfg['mlp_expansion'] = 1
cfg['epochs'] = $EPOCHS
json.dump(cfg, open('config.json', 'w'), indent=4)
"
    python train.py
    git add .
    git commit --no-edit -m "results for C=512, epochs=$EPOCHS run"
    git pull --no-edit
    git push
done
