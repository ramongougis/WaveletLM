for C in 1024 512 256 128; do
    if [ $C -ge 1024 ]; then LR=0.005
    else LR=0.01
    fi
    python -c "
import json
cfg = json.load(open('config.json'))
cfg['C'] = $C
cfg['lr'] = $LR
json.dump(cfg, open('config.json', 'w'), indent=4)
"
    python train.py
    git add .
    git commit -m "results for C = $C with epochs = 1 run"
    git push
done
