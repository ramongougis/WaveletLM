#!/bin/bash
# WaveletLM training runner.
#
# `config.json` is the single source of truth for run configuration.
# Edit it directly, or stash a baseline and diff against it. This script
# kicks off training and (optionally) generation; sweep / ablation logic
# previously lived here but has been removed for clarity.

python train.py
