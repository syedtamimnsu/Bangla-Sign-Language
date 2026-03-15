import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

class SignDataset(Dataset):
    """Dataset for sign language features and labels."""
    def __init__(self, tsv_path, feature_dir, tokenizer, max_seq_len=30):
        self.annotations = pd.read_csv(tsv_path)
        self.feature_dir = feature_dir
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.sos_token_id = tokenizer.token_to_id("[SOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        feat_path = os.path.join(self.feature_dir, f"{row['Index']}.npy")
        if not os.path.exists(feat_path):
            return None, None
        features = np.load(feat_path)
        tokens = [self.sos_token_id] + self.tokenizer.encode(str(row['Names'])).ids[:self.max_seq_len-2] + [self.eos_token_id]
        padded_tokens = tokens + [self.pad_token_id] * (self.max_seq_len - len(tokens))
        return torch.FloatTensor(features), torch.LongTensor(padded_tokens)

def pad_collate_fn(batch):
    """Collate function for padding sequences."""
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    feats, tokens = zip(*batch)
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    return padded_feats, torch.stack(tokens)
