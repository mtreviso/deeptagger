#!/usr/bin/env bash
cd ..
python3 -m deeptagger --train-path "data/corpus/pt/macmorpho_v1_toy/train.txt" \
					  --dev-path "data/corpus/pt/macmorpho_v1_toy/dev.txt" \
					  --test-path "data/corpus/pt/macmorpho_v1_toy/test.txt" \
					  --del-word " " \
					  --del-tag "_" \
					  --embeddings-format "polyglot" \
					  --embeddings-path "data/embeddings/polyglot/pt/embeddings_pkl.tar.bz2" \
					  --output-dir "runs/testing-macmorpho_v1_toy/" \
					  --train-batch-size 128 \
					  --dev-batch-size 128
