for C in 1024 512 256 128; do
    python -c "
import json
cfg = json.load(open('config.json'))
cfg['C'] = $C
json.dump(cfg, open('config.json', 'w'), indent=4)
"
    python train.py
done
