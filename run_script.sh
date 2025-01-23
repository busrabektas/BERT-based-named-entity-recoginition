#!/bin/bash

# Install required Python packages
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Run the training script
echo "Starting training..."

python3 train.py \
    --model_save_path "./models/output" \
    --dataset_path "./dataset/dataset_final.csv" \
    --num_train_epoch "5"

# Run the pipeline script
echo "Running pipeline..."

python3 pipeline.py \
    --model_load_path "./models/output" \
    --input_file "./data/test_input.txt" \
    --output_file "./data/ner_results.json"

echo "Scripts executed successfully!"
