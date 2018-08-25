#!/usr/bin/env bash
cd ..
python3 -m deeptagger --train-path "data/corpus/pt/macmorpho_v1/train.txt" \
					  --dev-path "data/corpus/pt/macmorpho_v1/dev.txt" \
					  --test-path "data/corpus/pt/macmorpho_v1/test.txt" \
					  --del-word " " \
					  --del-tag "_" \
					  --embeddings-format "polyglot" \
					  --embeddings-path "data/embeddings/polyglot/pt/embeddings_pkl.tar.bz2" \
					  --output-dir "runs/testing-macmorpho_v1/" \
					  --train-batch-size 128 \
					  --dev-batch-size 128
