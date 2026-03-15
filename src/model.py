import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class Seq2Seq(nn.Module):
    """Sequence-to-sequence Transformer model."""
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        hid, n_heads, pf_dim, dropout = cfg['hid_dim'], cfg['n_heads'], cfg['pf_dim'], cfg['dropout']
        self.fc_in = nn.Linear(cfg['input_dim'], hid)
        self.embedding = nn.Embedding(cfg['output_dim'], hid)
        self.pos_encoder = PositionalEncoding(hid, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg['n_layers'])
        decoder_layer = nn.TransformerDecoderLayer(d_model=hid, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg['n_layers'])
        self.fc_out = nn.Linear(hid, cfg['output_dim'])

    def forward(self, src, trg):
        enc_in = self.pos_encoder(self.fc_in(src))
        enc_out = self.encoder(enc_in)
        trg_emb = self.pos_encoder(self.embedding(trg))
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg.size(1)).to(self.device)
        output = self.decoder(trg_emb, enc_out, tgt_mask=trg_mask)
        return self.fc_out(output)
