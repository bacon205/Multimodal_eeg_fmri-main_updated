import os
import glob
import h5py
import copy
import yaml
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset


#  CONFIGURATION MANAGEMENT
class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.base_path = Path(os.getenv('EEG_DATA_PATH', r'E:\Intermediate\BACON_ERIC\Head_neck'))
        self.eeg_path_pw = self.base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'TF_dir' / 'pwspctrm' / 'PWS' / 'feat'
        self.eeg_path_erp = self.base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'TF_dir' / 'ERP' / 'New'
        self.eeg_path_conn = self.base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'conn_dir' / 'CONN' 
        self.label_path = self.base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'TF_dir'

        # Subject and experimental setup
        self.subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                            61, 62, 63]
        self.bands = {'alpha': 'Alpha', 'beta': 'Beta', 'theta': 'Theta'}
        self.eeg_segments = ['1_Hz', '2_Hz', '4_Hz', '6_Hz', '8_Hz', '10_Hz', '12_Hz', 
                            '14_Hz', '16_Hz', '18_Hz', '20_Hz', '25_Hz', '30_Hz', '40_Hz']
        self.func_segments = ['open', 'close']
        
        # Training hyperparameters
        self.batch_size = 8
        self.num_epochs = 50
        self.learning_rate = 5e-5
        self.weight_decay = 1e-5
        self.patience = 10
        self.n_splits = 5
        self.grad_clip = 1.0
        
        # Model architecture
        self.fusion_dim = 128
        self.hidden_dim = 64
        self.dropout = 0.65
        
        # Output paths
        self.output_dir = Path('./results')
        self.log_dir = Path('./logs')
        self.checkpoint_dir = Path('./checkpoints')
        for dir_path in [self.output_dir, self.log_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def save_config(self, path: str):
        """Save current configuration to YAML file"""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# LOGGING SETUP
def setup_logging(log_dir: Path, name: str = 'eeg_analysis'):
    """Configure logging with file and console handlers"""
    log_file = log_dir / f'{name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

# REPRODUCIBILITY
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
