for C in 2048 1024 512 256 128; do
    python -c "
import json
cfg = json.load(open('config.json'))
cfg['C'] = $C
json.dump(cfg, open('config.json', 'w'), indent=4)
"
    python train.py
    git add .
    git commit -m "results for C = $C with epochs = 1 run"
    git push
done
