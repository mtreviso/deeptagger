#!/usr/bin/env bash
cd ..
python3 -m deeptagger predict \
                      --load "saved-models/testing-toy-save/" \
					  --prediction-type classes \
					  --output-dir "predictions/macmorpho_v1_toy/" \
					  --text "Há livros escritos para evitar espaços vazios na estante ."
