# TorchServe Deployment for distilbert

This directory contains files needed to deploy the model using TorchServe.

## Prerequisites

- Install TorchServe: `pip install torchserve torch-model-archiver torch-workflow-archiver`

## Steps to Deploy

1. Archive the model:

```bash
torch-model-archiver --model-name distilbert \
                     --version 1.0 \
                     --serialized-file model/pytorch_model.bin \
                     --handler text_classification_handler.py \
                     --extra-files model/config.json,model/vocab.txt,model/labels.json \
                     --export-path model-store
```

2. Start TorchServe:

```bash
torchserve --start --model-store model-store --models distilbert=distilbert.mar
```

3. Send a request to the server:

```bash
curl -X POST http://localhost:8080/predictions/distilbert -T input.json
```

where input.json contains:

```json
{
  "data": "Your text to classify"
}
```

4. Stop TorchServe when done:

```bash
torchserve --stop
```

## Configuration Files

- `model/`: Contains the model weights, tokenizer, and configuration
- `text_classification_handler.py`: Custom handler for text classification
- `model-store/`: Directory where the model archive (.mar) file will be stored

