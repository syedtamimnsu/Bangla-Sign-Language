import os
import torch
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from src.utils import load_config, setup_logging, set_seed
from src.extractors import extract_and_save_features
from src.dataset import SignDataset, pad_collate_fn
from src.model import Seq2Seq
from src.trainer import TrainManager
from torch.utils.data import DataLoader

def main():
    config = load_config()
    logger = setup_logging(config)
    set_seed(config['training']['seed']) 

    logger.info("✅ Step 1: Subset creation skipped.")

    logger.info("\n▶️ Step 2: Starting feature extraction...")
    mapping_df = pd.read_csv(config['subset']['csv_path'])
    video_dir = config['subset']['video_dir']
    video_list = [os.path.join(video_dir, f"{idx}.mp4") for idx in mapping_df['Index']]
    video_list = [v for v in video_list if os.path.exists(v)]
    if not video_list:
        logger.error("No video files found. Stopping pipeline.")
        return
    extract_and_save_features(video_list, os.path.join(config['features']['output_dir'], config['training']['feature_dirs']['merged']), logger=logger)
    logger.info("✅ Step 2: Feature extraction finished.")

    logger.info("\n▶️ Step 3: Training tokenizer...")
    sentences = mapping_df['Names'].astype(str).tolist()
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=config['training']['vocab_size'], special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    os.makedirs(os.path.dirname(config['training']['tokenizer_path']), exist_ok=True)
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(config['training']['tokenizer_path'])
    config['training']['output_dim'] = tokenizer.get_vocab_size()
    logger.info(f"✅ Step 3: Tokenizer saved with vocab size: {tokenizer.get_vocab_size()}")

    logger.info("\n▶️ Step 4: Creating dataset and dataloader...")
    train_ds = SignDataset(tsv_path=config['subset']['csv_path'],
                           feature_dir=os.path.join(config['features']['output_dir'], config['training']['feature_dirs']['merged']),
                           tokenizer=tokenizer, max_seq_len=config['training']['max_seq_len'])
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
    logger.info("✅ Step 4: Dataset and dataloader ready.")

    logger.info("\n▶️ Step 5: Initializing model...")
    device = torch.device("cuda" if config['training']['use_cuda'] and torch.cuda.is_available() else "cpu")
    model = Seq2Seq(config['training'], device)
    logger.info("✅ Step 5: Model initialized.")

    logger.info("\n▶️ Step 6: Starting training...")
    trainer = TrainManager(model, config['training'], logger)
    trainer.set_tokenizer(tokenizer)
    trainer.train(train_loader)
    torch.save(model.state_dict(), config['training']['model_save_path'])  
    logger.info("\n✅ Pipeline finished successfully!")

if __name__ == '__main__':
    main()
