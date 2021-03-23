#!/bin/bash


if [[ -z "${Insurance}" ]]; then
  export DATASET_PATH="../data/training.csv"
else
  export DATASET_PATH="$Insurance/data/training.csv"
fi

python predict.py

WEEKLY_EVALUATION=true python predict.py
