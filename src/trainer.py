import torch
import torch.nn as nn
from tqdm import tqdm

class TrainManager:
    """Manager for training the model."""
    def __init__(self, model, cfg, logger):
        self.model = model
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device("cuda" if cfg['use_cuda'] and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    def train(self, train_loader, val_loader=None):
        self.logger.info(f"Starting training on {self.device}...")
        for epoch in range(self.cfg['num_epochs']):
            self.model.train()
            epoch_loss = 0
            for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg['num_epochs']}"):
                if src.nelement() == 0:
                    continue
                src, trg = src.to(self.device), trg.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg_out = trg[:, 1:].contiguous().view(-1)
                loss = self.criterion(output, trg_out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['grad_clip'])
                self.optimizer.step()
                epoch_loss += loss.item()
            self.logger.info(f"Epoch {epoch+1}: Train Loss {epoch_loss / len(train_loader):.4f}")
