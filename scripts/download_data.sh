#!/usr/bin/env bash
# Downloads the two v1-scope datasets into data/raw/. Re-run any time -- it just overwrites.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw/finqa data/raw/tatqa

echo "Downloading FinQA (train/dev/test)..."
curl -sL -o data/raw/finqa/train.json https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json
curl -sL -o data/raw/finqa/dev.json   https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json
curl -sL -o data/raw/finqa/test.json  https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json

echo "Downloading TAT-QA (train/dev)..."
curl -sL -o data/raw/tatqa/train.json https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json
curl -sL -o data/raw/tatqa/dev.json   https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json
# TAT-QA's public test set answers are withheld (leaderboard-style) -- dev.json is your
# eval split for this project, don't wait on test.json.

echo "Done. Sizes:"
du -h data/raw/finqa/*.json data/raw/tatqa/*.json
