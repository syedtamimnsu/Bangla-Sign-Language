import logging
import os
import yaml
import torch
import random
import numpy as np

def setup_logging(config):
    """Set up logging based on config."""
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load YAML config with environment variable substitution."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    base_dir = config['base_dir']
    for section in config:
        if isinstance(config[section], dict):
            for key in config[section]:
                if isinstance(config[section][key], str):
                    config[section][key] = config[section][key].replace('${base_dir}', base_dir)
    return config

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
