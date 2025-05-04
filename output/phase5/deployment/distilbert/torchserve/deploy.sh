#!/bin/bash

# Archive the model
torch-model-archiver --model-name distilbert \
                     --version 1.0 \
                     --serialized-file model/pytorch_model.bin \
                     --handler text_classification_handler.py \
                     --extra-files model/config.json,model/vocab.txt,model/labels.json \
                     --export-path model-store

# Start TorchServe
torchserve --start --model-store model-store --models distilbert=distilbert.mar

echo "TorchServe started. Test with: curl -X POST http://localhost:8080/predictions/distilbert -T input.json"
