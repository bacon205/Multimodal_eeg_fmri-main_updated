# === Cell 2 ===
# Imports & Configuration
import os
import sys
import glob
import copy
import h5py
import torch
import random
import warnings
import logging
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from scipy.io import loadmat
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

# Add pipeline directories to path
sys.path.insert(0, os.path.join(os.getcwd(), 'EEG_CODE'))
sys.path.insert(0, os.path.join(os.getcwd(), 'fMRI_CODE'))

# Import shared EEG components
from EEG_CODE.crossmodal_v4_enhancements import (
    EnhancedTriModalFusionNetV4,
    LearnedFusionModule,
    EnhancedERPEncoder,
    EnhancedPowerEncoder,
    get_fusion_weights_from_model
)

# Import fMRI components (avoid redefinition)
from fMRI_CODE.fmri_utils import fMRIFusionNet, ActivationEncoder, ConnectivityEncoder


# === Cell 4 ===
class BridgeConfig:
    def __init__(self):
        self.eeg_checkpoint_dir = Path('./EEG_CODE/checkpoints')
        self.fmri_checkpoint_dir = Path('./fMRI_CODE/checkpoints_fmri')
        self.eeg_hidden_dim = 128
        self.fmri_hidden_dim = 64
        self.bridge_hidden_dim = 128
        self.num_classes = 2
        self.overlap_subjects = list(range(1, 33))
        self.batch_size = 8
        self.num_epochs = 50
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.patience = 10
        self.grad_clip = 1.0
        self.dropout = 0.3
        self.eeg_base_path = Path(os.getenv('EEG_DATA_PATH', r'E:\Head_neck'))
        self.eeg_path_pw = self.eeg_base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'TF_dir' / 'pwspctrm' / 'PWS' / 'feat'
        self.eeg_path_erp = self.eeg_base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'TF_dir' / 'ERP' / 'New'
        self.eeg_path_conn = self.eeg_base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'conn_dir' / 'CONN' / 'New'
        self.eeg_label_path = self.eeg_base_path / 'EEG' / 'DATA' / 'PROC' / 'data_proc' / 'cleaned_data' / 'TF_dir'
        self.fmri_base_path = Path(r'E:\Head_neck\fMRI')
        self.fmri_data_dir = self.fmri_base_path
        self.fmri_activation_types = ['sensory', 'AN', 'LN', 'cognitive', 'DMN']
        self.fmri_connectivity_types = ['DMN']
        self.fmri_agg_method = 'both'
        self.bands = {'alpha': 'Alpha', 'beta': 'Beta', 'theta': 'Theta'}
        self.eeg_segments = ['1_Hz', '2_Hz', '4_Hz', '6_Hz', '8_Hz', '10_Hz', '12_Hz',
                             '14_Hz', '16_Hz', '18_Hz', '20_Hz', '25_Hz', '30_Hz', '40_Hz']
        self.func_segments = ['open', 'close']
        self.output_dir = Path('./results_bridge')
        self.checkpoint_dir = Path('./checkpoints_bridge')
        self.log_dir = Path('./logs_bridge')
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


# === Cell 6 ===
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config = BridgeConfig()
log_file = config.log_dir / 'bridge_fusion.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger('bridge_fusion')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger.info(f'Device: {device}')
logger.info(f'Bridge Config: overlap_subjects={len(config.overlap_subjects)}, '
            f'bridge_hidden_dim={config.bridge_hidden_dim}')
print('Cell complete: Imports & configuration loaded')


# === Cell 8 ===
# EEG Model Wrapper
class ImprovedTriModalFusionNet(nn.Module):
    def __init__(self, in_pw_dim, in_erp_dim, in_conn_dim,
                 fusion_dim=128, num_classes=2, dropout=0.3,
                 num_transformer_layers=2, num_heads=4):
        super().__init__()
        self.model = EnhancedTriModalFusionNetV4(
            erp_channels=in_erp_dim,
            pw_channels=in_pw_dim,
            conn_features=in_conn_dim,
            hidden_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads
        )
        self.fusion_weight_history = []

    def forward(self, erp, pw, conn, return_feats=False):
        if return_feats:
            logits, fusion_weights, fused_feats = self.model(
                erp, pw, conn,
                return_fusion_weights=True,
                return_fused_feats=True
            )
            return {
                'logits': logits,
                'gates': fusion_weights,
                'fused_feats': fused_feats
            }
        else:
            return self.model(erp, pw, conn)

    def get_fusion_weights(self):
        return get_fusion_weights_from_model(self.model)


# === Cell 10 ===
# Data Loading - EEG Side (robust wildcard globs)

def load_eeg_conn_features(conn_dir, subject_list, band_list, cond_list):
    """Load EEG connectivity features from .mat files using wildcard patterns."""
    conn_features = {}
    for subj in subject_list:
        for subj_str in [f'{subj:03d}', f'{subj:02d}']:
            for band_key, band_name in band_list.items():
                for cond in cond_list:
                    # Wildcard pattern: match any file containing sub+id, band, cond
                    pattern = str(conn_dir / f'*sub{subj_str}*{band_key}*{cond}*.mat')
                    files = sorted(glob.glob(pattern))
                    if not files:
                        # Try with capitalized band name
                        pattern = str(conn_dir / f'*sub{subj_str}*{band_name}*{cond}*.mat')
                        files = sorted(glob.glob(pattern))
                    if not files:
                        # Try exact legacy pattern
                        pattern = str(conn_dir / f'conn_{band_name}_{cond}_sub{subj_str}.mat')
                        files = sorted(glob.glob(pattern))
                    for f in files:
                        try:
                            mat = loadmat(f)
                            for k in mat:
                                if not k.startswith('_'):
                                    data = np.array(mat[k], dtype=np.float32).flatten()
                                    data = np.nan_to_num(data, nan=0.0)
                                    conn_key = (subj, band_key, cond, 0)
                                    conn_features[conn_key] = data
                                    break
                        except Exception as e:
                            logger.warning(f'Error loading {f}: {e}')
    logger.info(f'Loaded {len(conn_features)} EEG connectivity samples')
    return conn_features


def load_eeg_pw_features(pw_dir, subject_list, band_list, freq_list):
    """Load EEG power spectrum features from .mat files using wildcard patterns."""
    pw_features = {}
    for subj in subject_list:
        for subj_str in [f'{subj:03d}', f'{subj:02d}']:
            for band in band_list:
                for freq in freq_list:
                    pattern = str(pw_dir / f'*sub{subj_str}*{band}*{freq}*.mat')
                    files = sorted(glob.glob(pattern))
                    if not files:
                        # Legacy exact pattern
                        pattern = str(pw_dir / f'powspctrm_{band}_{freq}_sub{subj_str}.mat')
                        files = sorted(glob.glob(pattern))
                    for f in files:
                        try:
                            mat = loadmat(f)
                            for k in mat:
                                if not k.startswith('_'):
                                    data = np.array(mat[k], dtype=np.float32).flatten()
                                    data = np.nan_to_num(data, nan=0.0)
                                    pw_key = (subj, band, freq, 0)
                                    pw_features[pw_key] = data
                                    break
                        except Exception as e:
                            logger.warning(f'Error loading {f}: {e}')
    logger.info(f'Loaded {len(pw_features)} EEG power spectrum samples')
    return pw_features


def load_eeg_erp_features(erp_dir, subject_list, band_list, freq_list):
    """Load EEG ERP features from .mat/.h5 files using wildcard patterns."""
    erp_features = {}
    for subj in subject_list:
        for subj_str in [f'{subj:03d}', f'{subj:02d}']:
            for band in band_list:
                for freq in freq_list:
                    pattern = str(erp_dir / f'*sub{subj_str}*{band}*{freq}*.mat')
                    erp_files = sorted(glob.glob(pattern))
                    if not erp_files:
                        pattern = str(erp_dir / f'ERP_sub{subj_str}_{band}_{freq}*.mat')
                        erp_files = sorted(glob.glob(pattern))
                    for f in erp_files:
                        try:
                            with h5py.File(f, 'r') as hf:
                                if 'erp_struct' in hf:
                                    erp_group = hf['erp_struct']
                                elif 'erp' in hf:
                                    erp_group = hf['erp']
                                else:
                                    erp_group = hf[list(hf.keys())[0]]

                                if 'avg' in erp_group:
                                    data = np.array(erp_group['avg'], dtype=np.float32)
                                elif 'trial' in erp_group:
                                    data = np.array(erp_group['trial'], dtype=np.float32)
                                    if data.ndim == 3:
                                        data = np.mean(data, axis=0)
                                else:
                                    for dk in erp_group.keys():
                                        candidate = erp_group[dk]
                                        if hasattr(candidate, 'shape') and len(candidate.shape) >= 2:
                                            data = np.array(candidate, dtype=np.float32)
                                            break
                                    else:
                                        continue

                                data = np.nan_to_num(data, nan=0.0)
                                erp_key = (subj, band, freq, 0)
                                erp_features[erp_key] = data
                        except Exception:
                            try:
                                mat = loadmat(f)
                                for k in mat:
                                    if not k.startswith('_'):
                                        data = np.array(mat[k], dtype=np.float32)
                                        data = np.nan_to_num(data, nan=0.0)
                                        erp_key = (subj, band, freq, 0)
                                        erp_features[erp_key] = data
                                        break
                            except Exception as e2:
                                logger.warning(f'Error loading ERP {f}: {e2}')
    logger.info(f'Loaded {len(erp_features)} EEG ERP samples')
    return erp_features


def load_eeg_labels(label_dir, binary=True):
    """Load EEG clinical labels from medical_score.csv (single source of truth)."""
    csv_path = os.path.join(label_dir, 'medical_score.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Label file not found: {csv_path}')
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Postoperative evaluation'])
    if df['Subject'].dtype == object:
        df['subject_id'] = df['Subject'].str.replace('sub', '', regex=False).astype(int)
    else:
        df['subject_id'] = df['Subject'].astype(int)
    label_dict = {}
    for _, row in df.iterrows():
        subj = int(row['subject_id'])
        score = row['Postoperative evaluation']
        label_dict[subj] = 0 if score <= 2 else 1 if binary else score
    return label_dict


# Load EEG data (filtered to subjects 1-32)
logger.info('Loading EEG data for subjects 1-32...')
eeg_label_dict = load_eeg_labels(str(config.eeg_label_path))
logger.info(f'EEG labels: {len(eeg_label_dict)} subjects')
band_keys = list(config.bands.keys())
eeg_erp_features = load_eeg_erp_features(
    config.eeg_path_erp, config.overlap_subjects, band_keys, config.eeg_segments)
eeg_pw_features = load_eeg_pw_features(
    config.eeg_path_pw, config.overlap_subjects, band_keys, config.eeg_segments)
eeg_conn_features = load_eeg_conn_features(
    config.eeg_path_conn, config.overlap_subjects, config.bands, config.func_segments)

logger.info(f'EEG data loaded: ERP={len(eeg_erp_features)}, PW={len(eeg_pw_features)}, CONN={len(eeg_conn_features)}')


# === Cell 12 ===
# Data Loading - fMRI Side
def load_fmri_activation_features(data_dir, subject_list, activation_types, agg_method='both'):
    features = {}
    for subj in tqdm(subject_list, desc='Loading fMRI activations'):
        subj_features = []
        subj_dir = data_dir / f'sub-{subj}'
        for act_type in activation_types:
            filepath = subj_dir / f'subject_{subj}_activation_{act_type}.csv'
            if not filepath.exists():
                continue
            try:
                df = pd.read_csv(filepath)
                if 'Subject' in df.columns:
                    df = df.drop('Subject', axis=1)
                data = df.values.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0)
                if agg_method == 'mean':
                    agg_data = np.mean(data, axis=0)
                elif agg_method == 'std':
                    agg_data = np.std(data, axis=0)
                elif agg_method == 'both':
                    agg_data = np.concatenate([np.mean(data, axis=0), np.std(data, axis=0)])
                else:
                    raise ValueError(f'Unknown agg method: {agg_method}')
                subj_features.append(agg_data)
            except Exception as e:
                logger.warning(f'Error loading {filepath}: {e}')
        if subj_features:
            features[subj] = torch.tensor(np.concatenate(subj_features), dtype=torch.float32)
    logger.info(f'fMRI activation features: {len(features)} subjects')
    if features:
        sample = list(features.values())[0]
        logger.info(f'  Activation feature dim: {sample.shape[0]}')
    return features

def load_fmri_connectivity_features(data_dir, subject_list, connectivity_types):
    """Load fMRI connectivity features."""
    features = {}
    for subj in tqdm(subject_list, desc='Loading fMRI connectivity'):
        subj_features = []
        subj_dir = data_dir / f'sub-{subj}'
        for conn_type in connectivity_types:
            filepath = subj_dir / f'subject_{subj}_fdr_PPI_Connectivity_{conn_type}.csv'
            if not filepath.exists():
                continue
            try:
                df = pd.read_csv(filepath)
                if 'Subject' in df.columns:
                    df = df.drop('Subject', axis=1)
                data = df.values.astype(np.float32).flatten()
                data = np.nan_to_num(data, nan=0.0)
                subj_features.append(data)
            except Exception as e:
                logger.warning(f'Error loading {filepath}: {e}')
        if subj_features:
            features[subj] = torch.tensor(np.concatenate(subj_features), dtype=torch.float32)
    logger.info(f'fMRI connectivity features: {len(features)} subjects')
    if features:
        sample = list(features.values())[0]
        logger.info(f'  Connectivity feature dim: {sample.shape[0]}')
    return features

# Load fMRI data
logger.info('Loading fMRI data...')
fmri_act_features = load_fmri_activation_features(
    config.fmri_data_dir, config.overlap_subjects,
    config.fmri_activation_types, config.fmri_agg_method)
fmri_conn_features = load_fmri_connectivity_features(
    config.fmri_data_dir, config.overlap_subjects, config.fmri_connectivity_types)
logger.info(f'fMRI data loaded: Act={len(fmri_act_features)}, Conn={len(fmri_conn_features)}')

# Single source of truth for labels: eeg_label_dict (medical_score.csv)
bridge_labels = {subj: eeg_label_dict[subj] for subj in config.overlap_subjects if subj in eeg_label_dict}
logger.info(f'Bridge labels: {len(bridge_labels)} subjects')
logger.info(f'Class distribution: {dict(zip(*np.unique(list(bridge_labels.values()), return_counts=True)))}')


# === Cell 14 ===
# Subject Alignment & Bridge Dataset (with graceful degradation)

class BridgeRawDataset(Dataset):
    def __init__(self, eeg_erp, eeg_pw, eeg_conn, fmri_act, fmri_conn,
                 labels, subject_list, bands, func_segments):
        self.samples = []

        # Determine reference shapes from available data for zero-padding
        pw_shapes = [v.shape for v in eeg_pw.values()]
        conn_shapes = [v.shape for v in eeg_conn.values()]
        ref_pw_shape = pw_shapes[0] if pw_shapes else None
        ref_conn_shape = conn_shapes[0] if conn_shapes else None

        # Build per-subject EEG sample lists
        eeg_by_subj = defaultdict(list)
        for key, erp_val in eeg_erp.items():
            subj = int(key[0]) if not isinstance(key[0], int) else key[0]
            pw_val = eeg_pw.get(key)
            lookup_band = str(key[1]).lower()
            conn_val = None
            for cond in func_segments:
                conn_key = (key[0], lookup_band, cond, key[3])
                if conn_key in eeg_conn:
                    conn_val = eeg_conn[conn_key]
                    break

            # Graceful degradation: zero-pad missing PW/CONN
            if pw_val is None and ref_pw_shape is not None:
                pw_val = np.zeros(ref_pw_shape, dtype=np.float32)
                logger.debug(f'Subject {subj}: zero-padded missing PW for key {key}')
            if conn_val is None and ref_conn_shape is not None:
                conn_val = np.zeros(ref_conn_shape, dtype=np.float32)
                logger.debug(f'Subject {subj}: zero-padded missing CONN for key {key}')

            if pw_val is not None and conn_val is not None:
                eeg_by_subj[subj].append((erp_val, pw_val, conn_val))

        # Align subjects
        for subj in sorted(subject_list):
            s_id = int(subj)
            has_eeg = s_id in eeg_by_subj
            has_fmri_act = s_id in fmri_act
            has_fmri_conn = s_id in fmri_conn
            has_label = s_id in labels
            if has_eeg and has_fmri_act and has_fmri_conn and has_label:
                self.samples.append({
                    'subject': s_id,
                    'label': labels[s_id],
                    'eeg_samples': eeg_by_subj[s_id],
                    'fmri_act': fmri_act[s_id],
                    'fmri_conn': fmri_conn[s_id],
                })
            else:
                missing = []
                if not has_eeg: missing.append("EEG")
                if not has_fmri_act: missing.append("fMRI-Act")
                if not has_fmri_conn: missing.append("fMRI-Conn")
                if not has_label: missing.append("Label")
                logger.debug(f"Subject {s_id} excluded. Missing: {', '.join(missing)}")

        if len(self.samples) == 0:
            logger.error("!!! NO ALIGNED SUBJECTS FOUND !!! Check Subject IDs and file paths.")
            return
        logger.info(f'BridgeRawDataset: {len(self.samples)} aligned subjects')
        eeg_counts = [len(s["eeg_samples"]) for s in self.samples]
        if eeg_counts:
            logger.info(f'  EEG samples per subject: min={min(eeg_counts)}, max={max(eeg_counts)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['eeg_samples'], s['fmri_act'], s['fmri_conn'], s['label'], s['subject']


bridge_raw_dataset = BridgeRawDataset(
    eeg_erp_features, eeg_pw_features, eeg_conn_features,
    fmri_act_features, fmri_conn_features,
    bridge_labels, config.overlap_subjects,
    config.bands, config.func_segments
)
print(f'\nAligned subjects: {len(bridge_raw_dataset)}')


# === Cell 16 ===
# Load Pre-trained Models
def find_best_checkpoint(checkpoint_dir, pattern):
    """Find the best checkpoint file matching a glob pattern."""
    files = sorted(glob.glob(str(Path(checkpoint_dir) / pattern)))
    if not files:
        logger.warning(f'No checkpoint found matching {checkpoint_dir}/{pattern}')
        return None
    return files[-1]

def load_eeg_model(checkpoint_path, dataset_sample, fusion_dim=128):
    """Instantiate and load an EEG tri-modal model from checkpoint."""
    eeg_samples = dataset_sample[0]
    sample_erp, sample_pw, sample_conn = eeg_samples[0]
    in_erp_dim = sample_erp.shape[0]
    in_pw_dim = sample_pw.shape[0]
    in_conn_dim = sample_conn.shape[0] if sample_conn.ndim == 1 else np.prod(sample_conn.shape)
    logger.info(f'EEG model dims: ERP={in_erp_dim}, PW={in_pw_dim}, CONN={in_conn_dim}')
    model = ImprovedTriModalFusionNet(
        in_pw_dim=in_pw_dim,
        in_erp_dim=in_erp_dim,
        in_conn_dim=in_conn_dim,
        fusion_dim=fusion_dim,
        num_classes=config.num_classes
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f'EEG model loaded from {checkpoint_path}')
    else:
        logger.warning('No EEG checkpoint found, using random weights')

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

def load_fmri_model(checkpoint_path, fmri_act_dim, fmri_conn_dim, hidden_dim=64):
    model = fMRIFusionNet(
        activation_dim=fmri_act_dim,
        connectivity_dim=fmri_conn_dim,
        hidden_dim=hidden_dim,
        num_classes=config.num_classes
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f'fMRI model loaded from {checkpoint_path}')
    else:
        logger.warning('No fMRI checkpoint found, using random weights')

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

eeg_ckpt = find_best_checkpoint(config.eeg_checkpoint_dir, 'best_trimodal_fold*.pt')
fmri_ckpt = find_best_checkpoint(config.fmri_checkpoint_dir, 'best_fusion_fold*.pt')
sample_fmri_act = list(fmri_act_features.values())[0]
sample_fmri_conn = list(fmri_conn_features.values())[0]
fmri_act_dim = sample_fmri_act.shape[0]
fmri_conn_dim = sample_fmri_conn.shape[0]
logger.info(f'fMRI dims: activation={fmri_act_dim}, connectivity={fmri_conn_dim}')

# Load models
sample_data = bridge_raw_dataset[0]
eeg_model = load_eeg_model(eeg_ckpt, sample_data, fusion_dim=config.eeg_hidden_dim)
fmri_model = load_fmri_model(fmri_ckpt, fmri_act_dim, fmri_conn_dim, hidden_dim=config.fmri_hidden_dim)

n_eeg_params = sum(p.numel() for p in eeg_model.parameters())
n_fmri_params = sum(p.numel() for p in fmri_model.parameters())
logger.info(f'EEG model params: {n_eeg_params:,} (all frozen)')
logger.info(f'fMRI model params: {n_fmri_params:,} (all frozen)')


# === Cell 18 ===
# Feature Extraction Functions

@torch.no_grad()
def extract_eeg_features(model, raw_dataset, device):
    model.eval()
    features = {}
    for idx in range(len(raw_dataset)):
        eeg_samples, _, _, label, subj = raw_dataset[idx]
        feat_list = []
        for erp_np, pw_np, conn_np in eeg_samples:
            erp_t = torch.tensor(erp_np, dtype=torch.float32).unsqueeze(0).to(device)
            pw_t = torch.tensor(pw_np, dtype=torch.float32).unsqueeze(0).to(device)
            conn_t = torch.tensor(conn_np, dtype=torch.float32).unsqueeze(0).to(device)
            if conn_t.dim() > 2:
                conn_t = conn_t.view(conn_t.size(0), -1)
            try:
                out = model(erp=erp_t, pw=pw_t, conn=conn_t, return_feats=True)
                fused = out['fused_feats']
                feat_list.append(fused.cpu())
            except Exception:
                continue
        if feat_list:
            stacked = torch.cat(feat_list, dim=0)
            mean_feat = stacked.mean(dim=0)
            features[subj] = mean_feat
    logger.info(f'Extracted EEG features for {len(features)} subjects, dim={list(features.values())[0].shape[0] if features else "N/A"}')
    return features


@torch.no_grad()
def extract_fmri_features(model, fmri_act, fmri_conn, subject_list, device):
    """Extract fused features from frozen fMRI model."""
    model.eval()
    features = {}
    for subj in subject_list:
        if subj not in fmri_act or subj not in fmri_conn:
            continue
        act_t = fmri_act[subj].unsqueeze(0).to(device)
        conn_t = fmri_conn[subj].unsqueeze(0).to(device)
        try:
            _, fused = model(act_t, conn_t, return_features=True)
            features[subj] = fused.squeeze(0).cpu()
        except Exception as e:
            logger.warning(f'fMRI feature extraction failed for subject {subj}: {e}')
    logger.info(f'Extracted fMRI features for {len(features)} subjects, dim={list(features.values())[0].shape[0] if features else "N/A"}')
    return features


# Extract features
logger.info('Extracting EEG fused features...')
eeg_fused_features = extract_eeg_features(eeg_model, bridge_raw_dataset, device)

logger.info('Extracting fMRI fused features...')
fmri_fused_features = extract_fmri_features(
    fmri_model, fmri_act_features, fmri_conn_features, config.overlap_subjects, device
)

# Verify alignment
common_subjects = sorted(set(eeg_fused_features.keys()) & set(fmri_fused_features.keys()) & set(bridge_labels.keys()))
logger.info(f'Common subjects with both EEG and fMRI features: {len(common_subjects)}')

if common_subjects:
    s = common_subjects[0]
    logger.info(f'  Sample subject {s}: EEG={eeg_fused_features[s].shape}, fMRI={fmri_fused_features[s].shape}')


# === Cell 20 ===
# Bridge Fusion Model (LayerNorm in classifier)

class EEGfMRIBridgeFusionNet(nn.Module):
    def __init__(self, eeg_dim=128, fmri_dim=64, bridge_dim=128,
                 num_classes=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.bridge_dim = bridge_dim

        # Project to shared space
        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fmri_proj = nn.Sequential(
            nn.Linear(fmri_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            bridge_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Learned fusion with temperature scaling
        self.fusion = LearnedFusionModule(
            num_modalities=2,
            hidden_dim=bridge_dim,
            use_temperature=True
        )

        # Classifier â€” LayerNorm for LOOCV compatibility
        self.classifier = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim // 2),
            nn.LayerNorm(bridge_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim // 2, num_classes)
        )

    def forward(self, eeg_feats, fmri_feats, return_features=False, return_weights=False):
        eeg_proj = self.eeg_proj(eeg_feats)
        fmri_proj = self.fmri_proj(fmri_feats)
        modality_seq = torch.stack([eeg_proj, fmri_proj], dim=1)

        eeg_q = eeg_proj.unsqueeze(1)
        attn_out, attn_weights_raw = self.cross_attn(
            eeg_q, modality_seq, modality_seq
        )
        eeg_enhanced = attn_out.squeeze(1)

        if return_weights:
            fused, fusion_weights = self.fusion(
                [eeg_enhanced, fmri_proj], return_weights=True
            )
        else:
            fused = self.fusion([eeg_enhanced, fmri_proj])
            fusion_weights = None

        logits = self.classifier(fused)

        results = [logits]
        if return_features:
            results.append(fused)
        if return_weights:
            results.append(fusion_weights)
            results.append(attn_weights_raw)

        return results[0] if len(results) == 1 else tuple(results)

    def get_fusion_weights(self):
        with torch.no_grad():
            logits = self.fusion.fusion_logits
            temp = self.fusion.temperature
            weights = F.softmax(logits / temp, dim=0)
            return {
                'eeg_weight': weights[0].item(),
                'fmri_weight': weights[1].item(),
                'temperature': temp.item()
            }


# Quick architecture test
_test_bridge = EEGfMRIBridgeFusionNet(
    eeg_dim=config.eeg_hidden_dim,
    fmri_dim=config.fmri_hidden_dim,
    bridge_dim=config.bridge_hidden_dim,
    num_classes=config.num_classes,
    dropout=config.dropout
)
n_bridge_params = sum(p.numel() for p in _test_bridge.parameters())
n_trainable = sum(p.numel() for p in _test_bridge.parameters() if p.requires_grad)
print(f'Bridge model: {n_bridge_params:,} total params, {n_trainable:,} trainable')

# Smoke test
_eeg_dummy = torch.randn(4, config.eeg_hidden_dim)
_fmri_dummy = torch.randn(4, config.fmri_hidden_dim)
_logits, _fused, _fw, _aw = _test_bridge(_eeg_dummy, _fmri_dummy, return_features=True, return_weights=True)
print(f'Smoke test: logits={_logits.shape}, fused={_fused.shape}, fusion_weights={_fw.shape}, attn_weights={_aw.shape}')
del _test_bridge, _eeg_dummy, _fmri_dummy


# === Cell 22 ===
# Bridge Dataset from Pre-extracted Features

class BridgeFeatureDataset(Dataset):
    """Dataset of pre-extracted EEG and fMRI features, aligned by subject."""
    def __init__(self, eeg_features, fmri_features, labels, subject_list):
        self.samples = []
        for subj in sorted(subject_list):
            if subj in eeg_features and subj in fmri_features and subj in labels:
                self.samples.append({
                    'eeg': eeg_features[subj],
                    'fmri': fmri_features[subj],
                    'label': labels[subj],
                    'subject': subj
                })
        logger.info(f'BridgeFeatureDataset: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['eeg'], s['fmri'], s['label'], s['subject']


def collate_bridge(batch):
    eeg = torch.stack([b[0] for b in batch])
    fmri = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    subjects = [b[3] for b in batch]
    return eeg, fmri, labels, subjects


bridge_dataset = BridgeFeatureDataset(
    eeg_fused_features, fmri_fused_features, bridge_labels, common_subjects
)

all_labels = np.array([s['label'] for s in bridge_dataset.samples])
print(f'Bridge dataset: {len(bridge_dataset)} samples')
print(f'Class distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}')


# === Cell 24 ===
# Training helpers

def train_bridge_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for eeg, fmri, labels, _ in loader:
        eeg, fmri, labels = eeg.to(device), fmri.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(eeg, fmri)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate_bridge(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    all_subjects = []
    with torch.no_grad():
        for eeg, fmri, labels, subjects in loader:
            eeg, fmri = eeg.to(device), fmri.to(device)
            logits = model(eeg, fmri)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subjects.extend(subjects)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    metrics = {
        'Accuracy': accuracy_score(all_targets, all_preds),
        'F1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
        'Precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        'Recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
    }
    try:
        metrics['AUC'] = roc_auc_score(all_targets, all_probs[:, 1])
    except Exception:
        metrics['AUC'] = 0.5
    return metrics, all_targets, all_probs, all_subjects


# === Cell 25 ===
# LOOCV Training Loop with per-fold XAI collection

loo = LeaveOneOut()
n_subjects = len(bridge_dataset)

# Aggregated results across all LOO folds
loo_predictions = []     # list of (subject, true_label, pred_label, prob_class1)
loo_fusion_weights = []  # per-fold fusion weight dicts
all_fold_fused_features = {}  # subject -> fused features

# Per-subject XAI results (collected from held-out fold only)
per_subject_saliency = {}      # subj -> {'eeg': ..., 'fmri': ...}
per_subject_ig = {}            # subj -> {'eeg': ..., 'fmri': ...}
per_subject_attn_fusion = {}   # subj -> dict

logger.info(f'Starting Leave-One-Out CV with {n_subjects} subjects')

for fold_idx, (train_idx, test_idx) in enumerate(loo.split(np.zeros(n_subjects)), 1):
    test_subj = bridge_dataset.samples[test_idx[0]]['subject']
    if fold_idx % 5 == 1:
        logger.info(f'LOO fold {fold_idx}/{n_subjects}: held-out subject {test_subj}')

    train_subset = Subset(bridge_dataset, train_idx)
    test_subset = Subset(bridge_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_bridge)
    test_loader = DataLoader(test_subset, batch_size=1,
                             shuffle=False, collate_fn=collate_bridge)

    # Class weights from training set
    train_labels = all_labels[train_idx]
    cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    # Create bridge model
    bridge_model = EEGfMRIBridgeFusionNet(
        eeg_dim=config.eeg_hidden_dim,
        fmri_dim=config.fmri_hidden_dim,
        bridge_dim=config.bridge_hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)

    optimizer = optim.AdamW(bridge_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_bridge_epoch(bridge_model, train_loader, optimizer, criterion, device, config.grad_clip)
        scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = copy.deepcopy(bridge_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    # Load best model for this fold
    if best_state:
        bridge_model.load_state_dict(best_state)

    # Evaluate on held-out subject
    bridge_model.eval()
    with torch.no_grad():
        eeg_t, fmri_t, label_t, subj_list = next(iter(test_loader))
        eeg_t, fmri_t = eeg_t.to(device), fmri_t.to(device)
        logits, fused, fw, aw = bridge_model(eeg_t, fmri_t, return_features=True, return_weights=True)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        true_label = label_t[0].item()
        prob_1 = probs[0, 1].item()

        loo_predictions.append((test_subj, true_label, pred, prob_1))
        all_fold_fused_features[test_subj] = fused.squeeze(0).cpu()

    # Collect fusion weights
    fw_dict = bridge_model.get_fusion_weights()
    loo_fusion_weights.append(fw_dict)

    # --- Per-subject XAI (held-out subject only â€” no leakage) ---
    # Gradient saliency
    eeg_in = eeg_t.clone().detach().requires_grad_(True)
    fmri_in = fmri_t.clone().detach().requires_grad_(True)
    bridge_model.eval()
    logits_xai = bridge_model(eeg_in, fmri_in)
    target_cls = logits_xai.argmax(dim=1)
    bridge_model.zero_grad()
    one_hot = torch.zeros_like(logits_xai)
    one_hot.scatter_(1, target_cls.view(-1, 1), 1)
    logits_xai.backward(gradient=one_hot)
    per_subject_saliency[test_subj] = {
        'eeg': eeg_in.grad.abs().cpu().numpy().squeeze(),
        'fmri': fmri_in.grad.abs().cpu().numpy().squeeze()
    }

    # Integrated Gradients
    n_steps = 50
    eeg_base = torch.zeros_like(eeg_t)
    fmri_base = torch.zeros_like(fmri_t)
    eeg_diff = eeg_t - eeg_base
    fmri_diff = fmri_t - fmri_base
    eeg_grads_ig, fmri_grads_ig = [], []
    tc = None
    for alpha in np.linspace(0, 1, n_steps):
        ei = (eeg_base + alpha * eeg_diff).requires_grad_(True)
        fi = (fmri_base + alpha * fmri_diff).requires_grad_(True)
        lo = bridge_model(ei, fi)
        if tc is None:
            tc = lo.argmax(dim=1)
        bridge_model.zero_grad()
        oh = torch.zeros_like(lo)
        oh.scatter_(1, tc.view(-1, 1), 1)
        lo.backward(gradient=oh)
        eeg_grads_ig.append(ei.grad.detach().cpu().numpy())
        fmri_grads_ig.append(fi.grad.detach().cpu().numpy())
    eeg_ig = eeg_diff.cpu().numpy() * np.mean(eeg_grads_ig, axis=0)
    fmri_ig = fmri_diff.cpu().numpy() * np.mean(fmri_grads_ig, axis=0)
    per_subject_ig[test_subj] = {
        'eeg': np.abs(eeg_ig).squeeze(),
        'fmri': np.abs(fmri_ig).squeeze()
    }

    # Attention & fusion weights
    per_subject_attn_fusion[test_subj] = {
        'label': true_label,
        'prediction': pred,
        'fusion_weights': fw.cpu().numpy().squeeze(),
        'attn_weights': aw.cpu().numpy().squeeze(),
    }

# Aggregate LOO results
loo_targets = np.array([p[1] for p in loo_predictions])
loo_preds = np.array([p[2] for p in loo_predictions])
loo_probs = np.array([p[3] for p in loo_predictions])
loo_subjects = [p[0] for p in loo_predictions]

loo_metrics = {
    'Accuracy': accuracy_score(loo_targets, loo_preds),
    'F1': f1_score(loo_targets, loo_preds, average='weighted', zero_division=0),
    'Precision': precision_score(loo_targets, loo_preds, average='weighted', zero_division=0),
    'Recall': recall_score(loo_targets, loo_preds, average='weighted', zero_division=0),
}
try:
    loo_metrics['AUC'] = roc_auc_score(loo_targets, loo_probs)
except Exception:
    loo_metrics['AUC'] = 0.5

print(f'\n{"="*60}')
print('BRIDGE FUSION LOOCV RESULTS')
print(f'{"="*60}')
for metric, val in loo_metrics.items():
    print(f'  {metric:12s}: {val:.4f}')

eeg_w = [fw['eeg_weight'] for fw in loo_fusion_weights]
fmri_w = [fw['fmri_weight'] for fw in loo_fusion_weights]
print(f'\n  EEG weight:  {np.mean(eeg_w):.4f} +/- {np.std(eeg_w):.4f}')
print(f'  fMRI weight: {np.mean(fmri_w):.4f} +/- {np.std(fmri_w):.4f}')


# === Cell 27 ===
# Results Visualization

fig_dir = config.output_dir / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

# --- Performance Summary Table ---
summary_rows = []
for metric, val in loo_metrics.items():
    summary_rows.append({'Metric': metric, 'Value': val})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(config.output_dir / f'bridge_summary_{timestamp}.csv', index=False)
print('Performance Summary:')
print(summary_df.to_string(index=False))

# --- ROC Curve (single aggregated) ---
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(loo_targets, loo_probs)
auc_val = loo_metrics['AUC']
ax.plot(fpr, tpr, label=f'LOOCV (AUC={auc_val:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Bridge Fusion ROC Curve (LOOCV)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / f'roc_curve_loocv_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Confusion Matrix (single aggregated) ---
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(loo_targets, loo_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
ax.set_title('Confusion Matrix (LOOCV)')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(fig_dir / f'confusion_matrix_loocv_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Fusion Weight Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(eeg_w, bins=15, alpha=0.7, color='#2ecc71', label='EEG')
axes[0].hist(fmri_w, bins=15, alpha=0.7, color='#e74c3c', label='fMRI')
axes[0].set_xlabel('Weight')
axes[0].set_ylabel('Count')
axes[0].set_title('Fusion Weight Distribution Across LOO Folds')
axes[0].legend()

bars = axes[1].bar(['EEG', 'fMRI'],
                    [np.mean(eeg_w), np.mean(fmri_w)],
                    yerr=[np.std(eeg_w), np.std(fmri_w)],
                    capsize=10, color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8)
axes[1].set_ylabel('Average Weight')
axes[1].set_title('Average Fusion Weights')
axes[1].set_ylim(0, 1)
for bar, mean in zip(bars, [np.mean(eeg_w), np.mean(fmri_w)]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{mean:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / f'fusion_weights_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# --- t-SNE of Bridge Fused Features ---
if all_fold_fused_features:
    from sklearn.manifold import TSNE
    feat_subjects = sorted(all_fold_fused_features.keys())
    feat_matrix = np.stack([all_fold_fused_features[s].numpy() for s in feat_subjects])
    feat_labels = np.array([bridge_labels[s] for s in feat_subjects])

    if len(feat_subjects) > 5:
        perplexity = min(30, len(feat_subjects) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED)
        embedded = tsne.fit_transform(feat_matrix)

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in np.unique(feat_labels):
            mask = feat_labels == cls
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                       label=f'Class {cls}', alpha=0.7, s=60)
        ax.set_title('t-SNE of Bridge Fused Features')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f'tsne_bridge_features_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()


# === Cell 29 ===
# XAI - Gradient Saliency (aggregated from LOOCV)

all_eeg_saliency = np.stack([per_subject_saliency[s]['eeg'] for s in loo_subjects])
all_fmri_saliency = np.stack([per_subject_saliency[s]['fmri'] for s in loo_subjects])

mean_eeg_sal = np.mean(all_eeg_saliency, axis=0)
mean_fmri_sal = np.mean(all_fmri_saliency, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(len(mean_eeg_sal)), mean_eeg_sal, color='#2ecc71', alpha=0.7)
axes[0].set_title('Gradient Saliency: EEG Features')
axes[0].set_xlabel('Feature Dimension')
axes[0].set_ylabel('Saliency')

axes[1].bar(range(len(mean_fmri_sal)), mean_fmri_sal, color='#e74c3c', alpha=0.7)
axes[1].set_title('Gradient Saliency: fMRI Features')
axes[1].set_xlabel('Feature Dimension')
axes[1].set_ylabel('Saliency')

plt.suptitle('Gradient Saliency Analysis (LOOCV held-out)', fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / f'gradient_saliency_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

total_eeg = np.sum(mean_eeg_sal)
total_fmri = np.sum(mean_fmri_sal)
total = total_eeg + total_fmri
print(f'Gradient saliency modality importance:')
print(f'  EEG:  {total_eeg/total:.4f} ({total_eeg:.4f})')
print(f'  fMRI: {total_fmri/total:.4f} ({total_fmri:.4f})')


# === Cell 31 ===
# XAI - Integrated Gradients (aggregated from LOOCV)

all_eeg_ig = np.stack([per_subject_ig[s]['eeg'] for s in loo_subjects])
all_fmri_ig = np.stack([per_subject_ig[s]['fmri'] for s in loo_subjects])

mean_eeg_ig = np.mean(all_eeg_ig, axis=0)
mean_fmri_ig = np.mean(all_fmri_ig, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_k = 20
eeg_top_idx = np.argsort(mean_eeg_ig)[-top_k:][::-1]
fmri_top_idx = np.argsort(mean_fmri_ig)[-top_k:][::-1]

axes[0].barh(range(top_k), mean_eeg_ig[eeg_top_idx], color='#2ecc71', alpha=0.7)
axes[0].set_yticks(range(top_k))
axes[0].set_yticklabels([f'EEG-{i}' for i in eeg_top_idx])
axes[0].set_title(f'Top {top_k} EEG Feature Attributions (IG)')
axes[0].set_xlabel('Attribution')
axes[0].invert_yaxis()

axes[1].barh(range(top_k), mean_fmri_ig[fmri_top_idx], color='#e74c3c', alpha=0.7)
axes[1].set_yticks(range(top_k))
axes[1].set_yticklabels([f'fMRI-{i}' for i in fmri_top_idx])
axes[1].set_title(f'Top {top_k} fMRI Feature Attributions (IG)')
axes[1].set_xlabel('Attribution')
axes[1].invert_yaxis()

plt.suptitle('Integrated Gradients Attribution (LOOCV held-out)', fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / f'integrated_gradients_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

total_eeg_ig = np.sum(mean_eeg_ig)
total_fmri_ig = np.sum(mean_fmri_ig)
total_ig = total_eeg_ig + total_fmri_ig
print(f'Integrated Gradients modality importance:')
print(f'  EEG:  {total_eeg_ig/total_ig:.4f}')
print(f'  fMRI: {total_fmri_ig/total_ig:.4f}')


# === Cell 33 ===
# XAI - SHAP Analysis

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print('SHAP not available. Install with: pip install shap')

if SHAP_AVAILABLE:
    # Train a final model on all data for SHAP global analysis
    full_loader = DataLoader(bridge_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=collate_bridge)
    shap_model = EEGfMRIBridgeFusionNet(
        eeg_dim=config.eeg_hidden_dim,
        fmri_dim=config.fmri_hidden_dim,
        bridge_dim=config.bridge_hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)

    cw_all = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    cw_all_t = torch.tensor(cw_all, dtype=torch.float32).to(device)
    criterion_shap = nn.CrossEntropyLoss(weight=cw_all_t)
    opt_shap = optim.AdamW(shap_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        train_bridge_epoch(shap_model, full_loader, opt_shap, criterion_shap, device, config.grad_clip)

    def bridge_predict(inputs):
        inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
        eeg_part = inputs_t[:, :config.eeg_hidden_dim]
        fmri_part = inputs_t[:, config.eeg_hidden_dim:]
        shap_model.eval()
        with torch.no_grad():
            logits = shap_model(eeg_part, fmri_part)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    all_features = []
    for idx in range(len(bridge_dataset)):
        eeg, fmri, _, _ = bridge_dataset[idx]
        combined = torch.cat([eeg, fmri]).numpy()
        all_features.append(combined)
    all_features = np.array(all_features)

    n_background = min(20, len(all_features))
    background = all_features[:n_background]

    explainer = shap.KernelExplainer(bridge_predict, background)
    shap_values = explainer.shap_values(all_features, nsamples=100)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    eeg_shap = sv[:, :config.eeg_hidden_dim]
    fmri_shap = sv[:, config.eeg_hidden_dim:]

    feature_names = ([f'EEG-{i}' for i in range(config.eeg_hidden_dim)] +
                     [f'fMRI-{i}' for i in range(config.fmri_hidden_dim)])

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(sv, all_features, feature_names=feature_names,
                      max_display=20, show=False)
    plt.title('SHAP Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig(fig_dir / f'shap_summary_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

    eeg_importance = np.mean(np.abs(eeg_shap))
    fmri_importance = np.mean(np.abs(fmri_shap))
    total_shap = eeg_importance + fmri_importance

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['EEG', 'fMRI'],
                  [eeg_importance/total_shap, fmri_importance/total_shap],
                  color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8)
    ax.set_ylabel('Relative SHAP Importance')
    ax.set_title('SHAP Modality Importance')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / f'shap_modality_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f'SHAP modality importance: EEG={eeg_importance/total_shap:.4f}, fMRI={fmri_importance/total_shap:.4f}')


# === Cell 35 ===
# XAI - Attention & Fusion Weight Analysis (from LOOCV)

subject_xai = []
for subj in loo_subjects:
    d = per_subject_attn_fusion[subj]
    d['subject'] = subj
    subject_xai.append(d)

# --- Attention Heatmap ---
attn_matrix = np.stack([s['attn_weights'] for s in subject_xai])
if attn_matrix.ndim == 4:
    mean_attn = np.mean(attn_matrix, axis=(0, 1))
elif attn_matrix.ndim == 3:
    mean_attn = np.mean(attn_matrix, axis=0)
else:
    mean_attn = attn_matrix.reshape(-1, 2).mean(axis=0, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(mean_attn.reshape(1, -1), annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=['EEG', 'fMRI'], yticklabels=['Query'], ax=ax)
ax.set_title('Cross-Modal Attention Weights (Mean, LOOCV)')
plt.tight_layout()
plt.savefig(fig_dir / f'attention_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Per-Subject Fusion Weights ---
fusion_weights_arr = np.stack([s['fusion_weights'] for s in subject_xai])
if fusion_weights_arr.ndim == 1:
    fusion_weights_arr = fusion_weights_arr.reshape(-1, 2)

fig, ax = plt.subplots(figsize=(12, 5))
subjects_list = [s['subject'] for s in subject_xai]
x = np.arange(len(subjects_list))
width = 0.35

ax.bar(x - width/2, fusion_weights_arr[:, 0], width, label='EEG', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, fusion_weights_arr[:, 1], width, label='fMRI', color='#e74c3c', alpha=0.8)
ax.set_xlabel('Subject')
ax.set_ylabel('Fusion Weight')
ax.set_title('Per-Subject Dynamic Fusion Weights (LOOCV)')
ax.set_xticks(x)
ax.set_xticklabels(subjects_list, rotation=45)
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(fig_dir / f'per_subject_fusion_weights_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Class-wise Fusion Weight Comparison ---
class0_mask = np.array([s['label'] == 0 for s in subject_xai])
class1_mask = np.array([s['label'] == 1 for s in subject_xai])

print('Fusion weights by class:')
if class0_mask.any():
    c0_eeg = fusion_weights_arr[class0_mask, 0].mean()
    c0_fmri = fusion_weights_arr[class0_mask, 1].mean()
    print(f'  Class 0: EEG={c0_eeg:.4f}, fMRI={c0_fmri:.4f}')
if class1_mask.any():
    c1_eeg = fusion_weights_arr[class1_mask, 0].mean()
    c1_fmri = fusion_weights_arr[class1_mask, 1].mean()
    print(f'  Class 1: EEG={c1_eeg:.4f}, fMRI={c1_fmri:.4f}')


# === Cell 37 ===
# Summary & Export

print('=' * 70)
print('EEG-fMRI BRIDGE FUSION - FINAL SUMMARY')
print('=' * 70)

print(f'\nDataset: {len(bridge_dataset)} subjects (overlap of EEG & fMRI)')
print(f'Cross-validation: Leave-One-Out ({n_subjects} folds)')
print(f'\nBridge Fusion Performance:')
for metric, val in loo_metrics.items():
    print(f'  {metric:12s}: {val:.4f}')

print(f'\nLearned Fusion Weights:')
print(f'  EEG:  {np.mean(eeg_w):.4f} +/- {np.std(eeg_w):.4f}')
print(f'  fMRI: {np.mean(fmri_w):.4f} +/- {np.std(fmri_w):.4f}')

# Per-subject predictions
pred_df = pd.DataFrame(loo_predictions, columns=['Subject', 'True', 'Predicted', 'Prob_Class1'])
pred_df.to_csv(config.output_dir / f'bridge_loocv_predictions_{timestamp}.csv', index=False)
print(f'\nPer-subject predictions saved.')

# Save summary
summary_df.to_csv(config.output_dir / f'bridge_summary_{timestamp}.csv', index=False)

# Save fusion weights
fw_df = pd.DataFrame(loo_fusion_weights)
fw_df.to_csv(config.output_dir / f'bridge_fusion_weights_{timestamp}.csv', index=False)

subj_fw_records = [
    {'subject': s['subject'], 'label': s['label'],
     'eeg_weight': float(s['fusion_weights'][0]),
     'fmri_weight': float(s['fusion_weights'][1])}
    for s in subject_xai
]
subj_fw_df = pd.DataFrame(subj_fw_records)
subj_fw_df.to_csv(config.output_dir / f'bridge_subject_fusion_weights_{timestamp}.csv', index=False)

# Save XAI arrays
np.savez(
    config.output_dir / f'bridge_xai_arrays_{timestamp}.npz',
    gradient_saliency_eeg=mean_eeg_sal,
    gradient_saliency_fmri=mean_fmri_sal,
    integrated_gradients_eeg=mean_eeg_ig,
    integrated_gradients_fmri=mean_fmri_ig,
    per_subject_fusion_weights=fusion_weights_arr,
)

logger.info(f'All results saved to {config.output_dir}')
logger.info(f'Figures saved to {fig_dir}')
print(f'\nOutput directory: {config.output_dir}')
print(f'Figures directory: {fig_dir}')
print(f'Timestamp: {timestamp}')
print('\nBridge fusion pipeline complete.')


