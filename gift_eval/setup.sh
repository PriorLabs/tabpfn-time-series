#!/bin/bash

# Install dependencies
pip install -r requirements-gift-eval.txt

# Download datasets
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir ./data
