import wandb
import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import torch
import torch.nn.functional as F
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.utils.data as data
import pandas as pd
import os
import csv
from itertools import chain
warnings.filterwarnings("ignore")


import argparse
import torch
import wandb




# ──────────────────────────────────────────────────────────────────────────────
# configuration function
# ──────────────────────────────────────────────────────────────────────────────
def get_config():
    """
    Parse command-line flags, log in to Weights-and-Biases, and return both the
    populated `args` namespace and the chosen torch `device`.

    Down-stream code can still rely on:
        args.hidden_layer_size, args.encoder_layers, args.bidirectional …
    """
    # DA6401_ASS3_VANILLA
    

    #take inputs from command line
    flag_spec = [
        (("-p", "--wandb_project"),     dict(default="final Noattention")),
        (("-e", "--wandb_entity"),      dict(default="3628-pavitrakhare-indian-institute-of-technology-madras")),
        (("--key",),                   dict(dest="wandb_key",
                                            default="71aebd5eed3e9b3e37a5a3c4658f5433375d97dc")),
        (("-tr", "--train"),           dict(dest="trainFilepath",
                                            default="/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")),
        (("-va", "--val"),             dict(dest="valFilePath",
                                            default="/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")),
        (("-te", "--test"),            dict(dest="testFilePath",
                                            default="/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv")),
        (("-hs", "--hidden_layer_size"), dict(type=int, default=512)),
        (("-es", "--embedding_size"),    dict(type=int, default=64)),
        (("-enc", "--encoder_layers"),   dict(type=int, default=3)),
        (("-dec", "--decoder_layers"),   dict(type=int, default=3)),
        (("-bi", "--bidirectional"),     dict(action="store_true", dest="bidirectional",default=True)),
        (("-ct", "--cell_type"),         dict(default="LSTM")),
        (("--attention",),               dict(action="store_true",default=False)),
        (("--epochs",),                  dict(type=int, default=15)),
        (("--dropout",),                 dict(dest="drop_out", type=float, default=0.3)),
        (("-lr", "--learning_rate"),    dict(dest="learning_rate",type=float, default=0.001)),
    ]

    parser = argparse.ArgumentParser()
    for flags, kwargs in flag_spec:
        parser.add_argument(*flags, **kwargs)

    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #login to wand
    wandb.login(key=args.wandb_key, relogin=True)
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               config=vars(args))

    return args, device







# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPER  ▸  CSV / DATA INGEST
# ─────────────────────────────────────────────────────────────────────────────

def read_file_0(filepath):
    #read the 0th col of input file
    def process_row(text_row):
        return list(text_row)

    def collect_characters(reader_obj):
        all_chars = []
        for entry in reader_obj:
            all_chars += process_row(entry[0])
        return all_chars

    with open(filepath, mode='r') as file_handle:
        csv_reader = csv.reader(file_handle,delimiter='\t')
        return collect_characters(csv_reader)
    



def read_file_1(trainFilepath):
    #read the 1st col of input file
    def extract_chars(text):
        return list(text)

    def accumulate(reader_obj):
        buffer = []
        for entry in reader_obj:
            buffer += extract_chars(entry[1])
        return buffer

    with open(trainFilepath, 'r') as file_stream:
        csv_rows = csv.reader(file_stream,delimiter='\t')
        return accumulate(csv_rows)
    
    

def _char_index_from_csv(csv_path: str, column: int) -> dict[str, int]:
    """
    
    CSV, add the special delimiter '|', and return a 1-based index dictionary.
    """
    with open(csv_path, newline='') as fh:
        reader = csv.reader(fh,delimiter='\t')
        # flatten characters from all rows in that column
        charset = set(chain.from_iterable(row[column] for row in reader))

    charset.add('|')                           # guarantee delimiter presence
    return {ch: idx + 1 for idx, ch in enumerate(charset)}   # 1-based IDs




def getCharToIndex(args):
    """Character → index map for the *Latin* (source) side – column 0."""
    return _char_index_from_csv(args.trainFilepath, 1)


def getCharToIndLang(args):
    """Character → index map for the *target language* side – column 1."""
    return _char_index_from_csv(args.trainFilepath, 0)







# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER  ▸  TEXT → TENSOR CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
def convert_characters_to_indices(word, dictionary):
    # Convert each character in the word to its index using the dictionary.
    # Invalid characters are filtered out.
    def safe_lookup(char):
        return dictionary[char] if char in dictionary else -1

    mapped = [safe_lookup(ch) for ch in word]
    filtered = list(filter(lambda idx: idx >= 0, mapped))
    return filtered


def adjust_sequence_length(indices, maximumLength):
    # Trim the sequence if it's too long, or pad with zeros if too short.
    # Ensures output length matches the given max length.
    current_length = len(indices)

    def trim(seq, length):
        return seq[:length]

    def pad(seq, total_length):
        return seq + [0] * (total_length - len(seq))

    if current_length > maximumLength:
        return trim(indices, maximumLength)
    elif current_length < maximumLength:
        return pad(indices, maximumLength)
    return indices


def convert_indices_to_tensor(device, indices, dictionary):
    # Add BOS/EOS tokens (from '|') to the sequence.
    # Then convert the final list to a PyTorch tensor.
    token = dictionary.get('|', 0)

    def add_delimiters(seq, token_id):
        return [token_id] + seq + [token_id]

    sequence = add_delimiters(indices, token)
    return torch.tensor(sequence, device=device)


def convert_word_to_indices(device, word, maximumLength, dict):
    # Convert a word to a fixed-length tensor of indices.
    # Uses char map, trims/pads, and adds delimiters.
    char_ids = convert_characters_to_indices(word, dict)

    def standardize_length(seq, target_len):
        return adjust_sequence_length(seq, target_len)

    resized = standardize_length(char_ids, maximumLength)
    return convert_indices_to_tensor(device, resized, dict)







# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER  ▸  DATASET-SPECIFIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def getMaxLenEng(args):
    # Returns the maximum length of English words in the training dataset.
    # Reads from column 0 of the CSV file.
    with open(args.trainFilepath, 'r') as csv_file:
        reader = csv.reader(csv_file,delimiter='\t')
        max_length = 0

        def update_max(current_max, candidate):
            return candidate if candidate > current_max else current_max

        for record in reader:
            length = len(record[1])
            max_length = update_max(max_length, length)

        maxLenEng = max_length
    return maxLenEng


def getMaxLenDev(args):
    # Returns the maximum length of Devanagari words in the training dataset.
    # Reads from column 1 of the CSV file.
    with open(args.trainFilepath, 'r') as csv_stream:
        reader_obj = csv.reader(csv_stream,delimiter='\t')
        longest = 0

        def get_larger(a, b):
            return b if b > a else a

        for line in reader_obj:
            current_length = len(line[0])
            longest = get_larger(longest, current_length)

        maxLenDev = longest
    return maxLenDev


def keyForInput(val, char_to_idx_latin):
    # Get character for given index from Latin (English) mapping.
    # Returns empty string if not found.
    for k, v in char_to_idx_latin.items():
        if val == v:
            return k
    return ""


def keyForVal(val, charToIndLang):
    # Get character for given index from Indic (target language) mapping.
    # Returns empty string if not found.
    for k, v in charToIndLang.items():
        if val == v:
            return k
    return ""


def generate_indices(device, row, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev):
    # Converts source and target words to padded tensor representations.
    # Applies appropriate length limits and mappings.
    src_text = row[1]
    tgt_text = row[0]

    def build_indexed_tensor(text, max_len, char_map):
        return convert_word_to_indices(device, text, max_len, char_map)

    source_tensor = build_indexed_tensor(src_text, maxLenEng, char_to_idx_latin)
    target_tensor = build_indexed_tensor(tgt_text, maxLenDev, charToIndLang)

    return source_tensor, target_tensor








# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPER  ▸  DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def getDataLoaders(device, args, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev):
    # Reads CSV data and converts it into tensor pairs for train, val, and test.
    # Returns corresponding PyTorch DataLoaders for each split.

    batch_size = 32
    train_shuffle = True
    eval_shuffle = False

    # ─────── Prepare validation set ─────────────────────────────────────
    pairs_v = []
    with open(args.valFilePath, 'r') as val_file:
        csv_reader = csv.reader(val_file,delimiter='\t')

        def process_and_append(pairs, record):
            eng_tensor, hin_tensor = generate_indices(device, record, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev)
            pairs.append([eng_tensor, hin_tensor])

        for entry in csv_reader:
            process_and_append(pairs_v, entry)

    # ─────── Prepare test set ────────────────────────────────────────────
    pairs_t = []
    with open(args.testFilePath, 'r') as test_file:
        test_reader = csv.reader(test_file,delimiter='\t')

        def append_indexed_pair(container, data_row):
            src_tensor, tgt_tensor = generate_indices(device, data_row, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev)
            container.append([src_tensor, tgt_tensor])

        for record in test_reader:
            append_indexed_pair(pairs_t, record)

    # ─────── Prepare train set ───────────────────────────────────────────
    pairs = []
    with open(args.trainFilepath, 'r') as train_file:
        train_reader = csv.reader(train_file,delimiter='\t')

        def process_row_and_store(storage, row_data):
            input_tensor, target_tensor = generate_indices(device, row_data, char_to_idx_latin, charToIndLang, maxLenEng, maxLenDev)
            storage.append([input_tensor, target_tensor])

        for entry in train_reader:
            process_row_and_store(pairs, entry)

    # ─────── Create PyTorch DataLoaders ─────────────────────────────────
    def create_loader(dataset, size, shuffle_flag):
        return torch.utils.data.DataLoader(dataset, batch_size=size, shuffle=shuffle_flag)

    dataloaderTrain = create_loader(pairs, batch_size, train_shuffle)
    dataloaderVal = create_loader(pairs_v, batch_size, eval_shuffle)
    dataloaderTest = create_loader(pairs_t, batch_size, eval_shuffle)

    return dataloaderTrain, dataloaderVal, dataloaderTest









# ─────────────────────────────────────────────────────────────────────────────
# 5.  HELPER  ▸  LOGGING / I-O
# ─────────────────────────────────────────────────────────────────────────────


def save_outputs_to_csv(inputs, targets, predictions):
    # Saves the inputs, targets, and predicted outputs to a CSV file.
    # Appends to the file if it already exists, else creates a new one.
    output_file = 'Output.csv'
    already_present = os.path.isfile(output_file)

    records = {
        'input': inputs,
        'target': targets,
        'predicted': predictions
    }

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(output_file, mode='a', index=False, header=not already_present)


def log_predictions_to_wandb(inputs, targets, preds):
    # Logs predictions to Weights & Biases with visual indicators for correctness.
    # Uses a wandb Table for structured display in the dashboard.
    prediction_table = wandb.Table(columns=["Input", "Target", "Prediction Result"])

    for src, gold, guess in zip(inputs, targets, preds):
        result = "✅" if gold == guess else "❌"
        labeled_pred = f"{guess} {result}"
        prediction_table.add_data(src, gold, labeled_pred)

    wandb.log({"Predictions Overview": prediction_table})









# ─────────────────────────────────────────────────────────────────────────────
# 6.  HELPER  ▸  Tensors related functions
# ─────────────────────────────────────────────────────────────────────────────

def assign_tensor_to_generated_sequences(sequence):
    # Converts a space-separated string into a single concatenated string.
    # Also returns the total number of characters across all tokens.
    tokens = sequence.split()

    def concatenate(tokens):
        return ''.join(tokens)

    def compute_total_length(tokens):
        return sum(len(token) for token in tokens)

    combined = concatenate(tokens)
    total_len = compute_total_length(tokens)

    return combined, total_len


def assemble_assigned_generated_seq(path):
    # Breaks the combined string from a path into chunks based on its length.
    # Returns the sequence divided into fixed-size segments.
    combined_seq, total_chars = assign_tensor_to_generated_sequences(path)

    def determine_chunk_size(length, divisor=4):
        return max(1, length // divisor)

    segment_length = determine_chunk_size(total_chars)
    partitioned = assemble_tensor(combined_seq, segment_length)

    return partitioned


def assemble_tensor(final_tensor, partition_size=1):
    # Splits a string (or tensor) into chunks of specified size.
    # Returns a list of segmented chunks.
    def safe_partition_size(size):
        return max(1, size)

    chunk_size = safe_partition_size(partition_size)
    segments = [final_tensor[idx:idx + chunk_size] for idx in range(0, len(final_tensor), chunk_size)]

    return segments







# ─────────────────────────────────────────────────────────────────────────────
# 7.  Encoder, Decoder class
# ─────────────────────────────────────────────────────────────────────────────

# --------------------------------------------# --------------------------------------------

class Encoder(nn.Module):
    """
    Generic character-level encoder that can instantiate
    an RNN / GRU / LSTM with or without bidirectionality.

    Parameters
    ----------
    input_dim : int
        Size of the source-side vocabulary (incl. PAD + BOS/EOS).
    hidden_dim : int
        Dimensionality of the RNN hidden state.
    embed_size : int
        Dimensionality of the character embeddings.
    cell_type : str
        One of {"RNN", "GRU", "LSTM"} – case-sensitive.
    dropout_rate : float
        Dropout applied *inside* recurrent layers (pytorch’s `dropout` arg).
    num_layers : int
        Stacked-RNN depth.
    is_bidirectional : bool
        If True, use a bidirectional recurrent layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_size: int,
        cell_type: str,
        dropout_rate: float,
        num_layers: int,
        is_bidirectional: bool,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.is_bidirectional = is_bidirectional

        # ── embeddings ────────────────────────────────────────────────────
        self.embed_layer = nn.Embedding(num_embeddings=input_dim,
                                        embedding_dim=embed_size)

        # ── recurrent core factory ───────────────────────────────────────
        _rnn_cls = dict(RNN=nn.RNN, GRU=nn.GRU, LSTM=nn.LSTM).get(cell_type)
        if _rnn_cls is None:
            raise ValueError(f"Unsupported cell type: {cell_type!r}")

        self.rnn = _rnn_cls(
            input_size=embed_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=is_bidirectional,
        )

   
    def forward(self, input_seq: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        input_seq : LongTensor  (seq_len, batch)
            Padded character indices with BOS/EOS already attached.

        Returns
        -------
        hidden_state : Tensor or tuple(Tensor, Tensor)
            Final hidden (and cell) state(s) – identical format to
            torch’s RNN / GRU / LSTM `.forward`.
        """
        embedded = self.embed_layer(input_seq)          # (seq_len, batch, emb)
        _, hidden_state = self.rnn(embedded)
        return hidden_state





# --------------------------------------------# --------------------------------------------

# Decoder class (no attention mechanism)
class Decoder(nn.Module):
    """
    Character-level decoder (no attention).

    Parameters
    ----------
    output_dim : int
        Size of the target-side vocabulary.
    hidden_dim : int
        RNN hidden-state size (per direction).
    embed_size : int
        Embedding dimensionality.
    cell_type : {"RNN", "GRU", "LSTM"}
        Recurrent cell to instantiate.
    dropout_rate : float
        Dropout inside the stacked RNN layers.
    num_layers : int
        Number of stacked recurrent layers.
    is_bidirectional : bool
        Must match the encoder setting.
    """
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        embed_size: int,
        cell_type: str,
        dropout_rate: float,
        num_layers: int,
        is_bidirectional: bool,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.is_bidirectional = is_bidirectional

        self.embed = nn.Embedding(num_embeddings=output_dim,
                                  embedding_dim=embed_size)

        # pick the proper RNN implementation
        rnn_cls = dict(RNN=nn.RNN, GRU=nn.GRU, LSTM=nn.LSTM).get(cell_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported cell type: {cell_type!r}")

        self.rnn = rnn_cls(
            input_size=embed_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=is_bidirectional,
        )

        self.readout = nn.Linear(
            hidden_dim * (2 if is_bidirectional else 1),
            output_dim,
        )
        self.output_layer = self.readout
    
    def forward(
        self,
        input_token: torch.Tensor,             # (batch,)
        hidden_state,                          # encoder/prev state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One decoding step.

        Returns
        -------
        logits : Tensor  (batch, vocab_size)
        next_hidden : Tensor or tuple
            Same structure as `hidden_state`, advanced by one RNN step.
        """
        emb = self.embed(input_token.unsqueeze(0))       # (1, batch, emb)
        rnn_out, next_hidden = self.rnn(emb, hidden_state)
        logits = self.readout(rnn_out.squeeze(0))        # (batch, vocab)
        return logits, next_hidden
# --------------------------------------------# --------------------------------------------








# ─────────────────────────────────────────────────────────────────────────────
# 8.  High-level model (Seq2Seq / Seq2SeqAttn)
# ─────────────────────────────────────────────────────────────────────────────


#Sequence class for without attention
class Seq2Seq(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_size, embed_size, rnn_type, dropout, enc_layers, dec_layers, is_bidirectional, lr,char_to_idx_latin, charToIndLang):
        super().__init__()
        self.lr = lr
        self.rnn_type = rnn_type
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.is_bidirectional = is_bidirectional
        self.dir_mult = 2 if is_bidirectional else 1
        self.char2lat = char_to_idx_latin   # source vocab
        self.char2hin = charToIndLang       # target vocab

        # Initialize encoder and decoder
        self.encoder = Encoder(input_dim, hidden_size, embed_size, rnn_type, dropout, enc_layers, is_bidirectional)
        self.decoder = Decoder(output_dim, hidden_size, embed_size, rnn_type, dropout, dec_layers, is_bidirectional)

        # Logging containers for metrics
        self.train_loss_log, self.train_acc_log = [], []
        self.val_loss_log, self.val_acc_log = [], []
        self.test_loss_log, self.test_acc_log = [], []

    def forward(self, src_seq, tgt_seq, teacher_prob=0.5):
        # Sequence-to-sequence forward pass with teacher forcing
        batch_size, seq_len = tgt_seq.shape
        vocab_size = self.decoder.output_layer.out_features
        prediction_store = torch.zeros(seq_len, batch_size, vocab_size).to(self.device)

        src_seq = src_seq.transpose(0, 1)
        hidden_state = self._fix_hidden_dims(self.encoder(src_seq))

        current_token = tgt_seq[:, 0]
        for step in range(1, seq_len):
            pred_output, hidden_state = self.decoder(current_token, hidden_state)
            prediction_store[step] = pred_output
            current_token = pred_output.argmax(1) if teacher_prob < torch.rand(1).item() else tgt_seq[:, step]

        return prediction_store

    def _fix_hidden_dims(self, hidden):
        # Adjust hidden layers if encoder and decoder layer counts mismatch
        lstm_mode = self.rnn_type == "LSTM"
        enc_more = self.enc_layers > self.dec_layers
        dec_more = self.enc_layers < self.dec_layers

        if enc_more:
            cut = (self.enc_layers - self.dec_layers) * self.dir_mult
            if lstm_mode:
                h, c = hidden
                return (h[cut:], c[cut:])
            return hidden[cut:]

        if dec_more:
            add = self.dec_layers - self.enc_layers
            if lstm_mode:
                h, c = hidden
                h_last, c_last = h[-self.dir_mult:], c[-self.dir_mult:]
                h = torch.cat([h] + [h_last] * add, dim=0)
                c = torch.cat([c] + [c_last] * add, dim=0)
                return (h, c)
            else:
                h_last = hidden[-self.dir_mult:]
                return torch.cat([hidden] + [h_last] * add, dim=0)

        return hidden

    def forward_pass(self, src, tgt):
        # Wraps model forward for clarity
        return self(src, tgt)

    def compute_expected_values(self, preds, target):
        # Create one-hot encoded expected output tensor
        result = torch.zeros_like(preds)
        result[torch.arange(preds.shape[0]), torch.arange(preds.shape[1]).unsqueeze(1), target.cpu()] = 1
        return result

    def calculate_loss(self, preds, expected, target):
        # Cross-entropy loss computation (skip BOS)
        d = preds.shape[-1]
        return self.loss_fn(preds[1:].reshape(-1, d), expected[1:].reshape(-1, d))

    def calculate_accuracy(self, preds, target):
        # Accuracy at word level (after permute)
        preds = preds.permute(1, 0, 2)
        return self._wordwise_accuracy(preds, target)

    def update_metrics(self, loss_val, acc_val):
        # Append training metrics to logs
        self.train_loss_log.append(loss_val.detach())
        self.train_acc_log.append(torch.tensor(acc_val))

    def training_step(self, batch, batch_idx):
        # Called during each training batch
        src, tgt = batch
        logits = self.forward_pass(src, tgt)
        expected = self.compute_expected_values(logits, tgt)
        loss = self.calculate_loss(logits, expected, tgt)
        acc = self.calculate_accuracy(logits, tgt)
        self.update_metrics(loss, acc)
        return {'loss': loss}

    def forward_pass_validation(self, src, tgt):
        # Run forward without teacher forcing
        return self(src, tgt, teacher_prob=0.0)

    def validation_step(self, batch, batch_idx):
        # Called for each validation batch
        src, tgt = batch
        logits = self.forward_pass_validation(src, tgt)
        expected = self.compute_expected_values(logits, tgt)
        loss = self.calculate_loss(logits, expected, tgt)
        acc = self.calculate_accuracy(logits, tgt)
        self.val_loss_log.append(loss.detach())
        self.val_acc_log.append(torch.tensor(acc))
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        """
        One full forward pass on the test set (no teacher forcing).
        Computes and logs loss and word-level accuracy, and saves predictions.
        """
        src_seq, tgt_seq = batch
        logits = self(src_seq, tgt_seq, teacher_prob=0.0)
        gold_one_hot = self.compute_expected_values(logits, tgt_seq)
        loss_val = self.calculate_loss(logits, gold_one_hot, tgt_seq)
        acc_val = self._wordwise_accuracy(logits.permute(1, 0, 2), tgt_seq)

        # Save predictions to CSV
        src_tokens, tgt_tokens, pred_tokens = self.grid(src_seq, logits.permute(1, 0, 2), tgt_seq)

        def tokens_to_string(token_list, vocab):
            return "".join([k for tok in token_list for k, v in vocab.items() if v == tok])

        inputs = [tokens_to_string(seq.tolist(), self.char2lat) for seq in src_tokens]
        targets = [tokens_to_string(seq.tolist(), self.char2hin) for seq in tgt_tokens]
        predictions = [tokens_to_string(seq.tolist(), self.char2hin) for seq in pred_tokens]

        save_outputs_to_csv(inputs, targets, predictions)

        #uncomment below code to log predictions on test data
        log_predictions_to_wandb(inputs, targets, predictions)

        self.test_loss_log.append(loss_val.detach())
        self.test_acc_log.append(torch.tensor(acc_val))
        return {"loss": loss_val}

    def on_test_epoch_end(self):
        # Print and log test epoch results
        loss_avg = torch.stack(self.test_loss_log).mean()
        acc_avg = torch.stack(self.test_acc_log).mean()
        self.test_loss_log.clear()
        self.test_acc_log.clear()
        print({"test_loss": loss_avg.item(), "testAccuracy": acc_avg.item()})
        wandb.log({"test_loss_last": loss_avg, "testAccuracy_last": acc_avg})

    def on_train_epoch_end(self):
        # Print and log training & validation metrics
        tr_loss = torch.stack(self.train_loss_log).mean()
        val_loss = torch.stack(self.val_loss_log).mean()
        tr_acc = torch.stack(self.train_acc_log).mean()
        val_acc = torch.stack(self.val_acc_log).mean()

        self.train_loss_log.clear()
        self.val_loss_log.clear()
        self.train_acc_log.clear()
        self.val_acc_log.clear()

        print({
            "Train Loss": round(tr_loss.item(), 3),
            "Train Accuracy": round(tr_acc.item(), 3),
            "Validation Loss": round(val_loss.item(), 3),
            "Validation Accuracy": round(val_acc.item(), 3)
        })
        wandb.log({
            "Train Loss": tr_loss,
            "Train Accuracy": tr_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc
        })

    def configure_optimizers(self):
        # Adam optimizer setup
        return optim.Adam(self.parameters(), lr=self.lr)

    def loss_fn(self, out, target):
        # Cross-entropy loss wrapper
        return nn.CrossEntropyLoss()(out, target).mean()

    def _wordwise_accuracy(self, logits, gold):
        # Computes word-level match accuracy
        preds = logits.argmax(dim=-1)
        match_count = sum(torch.equal(gold[i, 1:-1], preds[i, 1:-1]) for i in range(gold.size(0)))
        return (match_count / gold.size(0)) * 100

    def grid(self, inputs, outputs, targets):
        # Converts batch output into src-tgt-predicted token lists
        guesses = outputs.argmax(dim=-1)
        src_tokens, tgt_tokens, pred_tokens = [], [], []
        for idx in range(targets.size(0)):
            src_tokens.append(inputs[idx, 1:-1])
            tgt_tokens.append(targets[idx, 1:-1])
            pred_tokens.append(guesses[idx, 1:-1])
        return src_tokens, tgt_tokens, pred_tokens







# ─────────────────────────────────────────────────────────────────────────────
# 9.  main() – training / validation / test orchestration
# ─────────────────────────────────────────────────────────────────────────────



def main():
    # Load command-line args and select device
    args, device = get_config()

    # Prepare vocab, character-index mappings, and max word lengths
    char_to_idx_latin = getCharToIndex(args)
    charToIndLang = getCharToIndLang(args)
    max_len_eng = getMaxLenEng(args)
    max_len_dev = getMaxLenDev(args)

    # Get train, val, test dataloaders
    dataloaderTrain, dataloaderVal, dataloaderTest = getDataLoaders(
        device, args, char_to_idx_latin, charToIndLang, max_len_eng, max_len_dev
    )

    # Instantiate Seq2Seq model
    model = Seq2Seq(
        len(char_to_idx_latin) + 2,
        len(charToIndLang) + 2,
        args.hidden_layer_size,
        args.embedding_size,
        args.cell_type,
        args.drop_out,
        args.encoder_layers,
        args.decoder_layers,
        args.bidirectional,
        args.learning_rate,
        char_to_idx_latin,
        charToIndLang
    )
    model.to(device)

    # Train and test the model using PyTorch Lightning
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1)
    trainer.fit(model=model, train_dataloaders=dataloaderTrain, val_dataloaders=dataloaderVal)
    trainer.test(model, dataloaderTest)
    wandb.finish()


if __name__ == "__main__":
    main()
