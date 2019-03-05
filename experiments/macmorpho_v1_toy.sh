#!/usr/bin/env bash
cd ..
python3 -m deeptagger train \
                      --model "rcnn" \
                      --rnn-type "lstm" \
                      --bidirectional \
                      --train-path "data/corpus/pt/macmorpho_v1_toy/train.txt" \
                      --dev-path "data/corpus/pt/macmorpho_v1_toy/dev.txt" \
                      --test-path "data/corpus/pt/macmorpho_v1_toy/test.txt" \
                      --del-word " " \
                      --del-tag "_" \
                      --output-dir "runs/testing-macmorpho_v1_toy/" \
                      --train-batch-size 128 \
                      --dev-batch-size 128 \
                      --optimizer "adam" \
                      --save-best-only \
                      --early-stopping-patience 3 \
                      --restore-best-model \
                      --final-report \
                      --epochs 2 \
                      --use-prefixes \
                      --use-suffixes \
                      --use-caps \
                      --save "saved-models/testing-toy-save/" \
                      --tensorboard
                      # --amsgrad \
                      # --nesterov \
                      # --momentum 0.9 \
                      # --lr-step-decay noam \
                      # --warmup-steps 2000 \
                      # --scheduler "exponential" \
                      # --gamma 0.1 \
                      # --embeddings-format "polyglot" \
                      # --embeddings-path "data/embeddings/polyglot/pt/embeddings_pkl.tar.bz2" \
                      # --add-embeddings-vocab \
