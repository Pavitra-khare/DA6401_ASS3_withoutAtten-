# Transliteration Model - Vanilla Seq2Seq (DA6401 Assignment 3)

This repository contains an implementation of a sequence-to-sequence (Seq2Seq) model for character-level transliteration using PyTorch Lightning. It supports both LSTM/GRU/RNN-based encoder-decoder architectures with optional attention (disabled in this version).

All experiments were run on **Kaggle Notebooks** using a GPU-enabled environment.

[GitHub Repository](https://github.com/Pavitra-khare/DA6401_ASS3_withoutAtten-)

[Weights & Biases Report](https://api.wandb.ai/links/3628-pavitrakhare-indian-institute-of-technology-madras/lods8pjm)

[Sweeps on kaggle](https://www.kaggle.com/code/pavitrakhare/notebook582f84c4ed/log?scriptVersionId=240492997)

---

## Objective

- Build a transliteration model from English to an Indian script using a vanilla encoder-decoder architecture.
- Support for configurable hyperparameters (embedding size, hidden size, RNN type, bidirectionality, dropout, etc.).
- Full data preprocessing pipeline for handling Aksharantar CSV datasets.
- Visualize attention maps (when attention is used) and save predictions to CSV.

---

## Model Architecture

### Vanilla Seq2Seq (Without Attention)

| Component       | Configuration                                                                 |
|-----------------|--------------------------------------------------------------------------------|
| **Encoder**     | Embedding → Stacked RNN/GRU/LSTM (uni/bidirectional)                          |
| **Decoder**     | Embedding → Stacked RNN/GRU/LSTM → Linear Projection to Vocabulary            |
| **Loss**        | CrossEntropyLoss (token-level, ignores BOS)                                   |
| **Evaluation**  | Word-level exact-match accuracy                                                |

---

##  Key Functions and Classes

| Name                        | Type     | Purpose                                                                 |
|-----------------------------|----------|-------------------------------------------------------------------------|
| `get_config()`              | Function | Parses arguments, initializes wandb, sets device                        |
| `getDataLoaders()`          | Function | Converts data CSV to tokenized and padded tensors, returns 3 DataLoaders|
| `generate_indices()`        | Function | Maps word pairs to index tensors with BOS/EOS                           |
| `convert_word_to_indices()` | Function | Main word→index→tensor conversion utility                               |
| `getMaxLenEng/Dev()`        | Function | Extracts max sequence lengths from training data                        |
| `Encoder`, `Decoder`        | Class    | Character-level encoder/decoder without attention                      |
| `Seq2Seq`                   | Class    | High-level PyTorch Lightning module for training + inference           |
| `save_outputs_to_csv()`     | Function | Saves prediction results to a local CSV file                            |

---

## Command Line Arguments

These are defined via `argparse` in `get_config()`.

| Argument             | Default                                                  | Description                              |
|----------------------|----------------------------------------------------------|------------------------------------------|
| `--wandb_project`    | `"DA6401_ASS3_VANILLA"`                                  | Project name on wandb                    |
| `--wandb_entity`     | `"3628-pavitrakhare-indian-institute-of-technology-madras"` | Your wandb user or team               |
| `--key`              | `"<wandb-api-key>"`                                      | wandb authentication key                 |
| `--train`, `--val`, `--test` | CSV file paths                                     | Aksharantar CSV dataset locations       |
| `--hidden_layer_size`| `256`                                                    | Hidden size of RNN                       |
| `--embedding_size`   | `256`                                                    | Size of token embeddings                 |
| `--cell_type`        | `"LSTM"`                                                 | RNN variant: LSTM, GRU, or RNN           |
| `--bidirectional`    | `True`                                                   | Whether encoder is bidirectional         |
| `--dropout`          | `0.5`                                                    | Dropout inside RNN layers                |
| `--epochs`           | `15`                                                     | Max training epochs                      |

---

## Evaluation Metrics

- **Loss:** CrossEntropyLoss averaged over predicted tokens (skipping BOS).
- **Accuracy:** Word-level exact match between predicted and ground truth.

---

## How to Run

### 1. Install Dependencies

Make sure Python is installed, then run:

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install pandas numpy matplotlib seaborn
pip install wandb
```

> Note: This code has been tested in Kaggle Notebooks with GPU acceleration.
---

### 2. Run the Model

Once dependencies are installed, run the training and testing loop using:

```bash
python modular_train_vanilla.py --train <path_to_train.csv> --val <path_to_val.csv> --test <path_to_test.csv>
```

You may customize any hyperparameter (e.g., encoder layers, hidden size, dropout, etc.) by adding flags:

```bash
python modular_train_vanilla.py --train data/train.csv --val data/val.csv --test data/test.csv --cell_type GRU --dropout 0.3 --hidden_layer_size 512
```

---

##  Output

- Logs in Weights & Biases
- Predictions stored in `Output.csv`
- Optional attention visualization if using attention variant

---
