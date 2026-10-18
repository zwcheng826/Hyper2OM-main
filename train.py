import os
import csv
import hashlib
import math
import random
import copy
import json
import argparse
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
VARIANT = {'model_no': 931, 'ablation_name': 'Baseline_NoDNC_hid260', 'run_id': 'Baseline_NoDNC_hid260', 'uses_rinalmo': True, 'uses_onehot_crnn': True, 'uses_transformer': True, 'msd_samples': 5, 'note': 'Static shared hypergraph with 5-D noDNC local features and Transformer.'}
PHYS_DIM = 5
LOCAL_CONTEXT_DIM = PHYS_DIM
LIVE_CONTEXT_DIM = PHYS_DIM
CENTER_GAUSSIAN_SIGMA = 0.35
LOCAL_ENCODING_ABLATION = 'noDNC'
LOCAL_CONTEXT_DIM = 5
LIVE_CONTEXT_DIM = 5

def local_feature_channels():
    groups = {'noNCP': 18, 'noDNC': 5, 'noRelativePosition': 20, 'noCenterGaussian': 20}
    return groups.get(LOCAL_ENCODING_ABLATION, 21)

def local_feature_description():
    """Describe the physical channels actually emitted by the active arm."""
    if LOCAL_ENCODING_ABLATION == 'noDNC':
        return 'NCP (3) + center-relative position + center Gaussian (5 total; DNC removed)'
    if LOCAL_ENCODING_ABLATION == 'noNCP':
        return 'standard 16-D DNC + center-relative position + center Gaussian (18 total; NCP removed)'
    if LOCAL_ENCODING_ABLATION == 'noRelativePosition':
        return 'NCP + standard 16-D DNC + center Gaussian (20 total; relative position removed)'
    if LOCAL_ENCODING_ABLATION == 'noCenterGaussian':
        return 'NCP + standard 16-D DNC + center-relative position (20 total; center Gaussian removed)'
    return 'NCP + standard 16-D DNC + center-relative position + center Gaussian (21 total)'

def configure_local_encoding_ablation(name):
    """Select a local encoding arm and synchronize all runtime dimensions."""
    global LOCAL_ENCODING_ABLATION, LOCAL_CONTEXT_DIM, LIVE_CONTEXT_DIM
    LOCAL_ENCODING_ABLATION = name
    LOCAL_CONTEXT_DIM = local_feature_channels()
    LIVE_CONTEXT_DIM = LOCAL_CONTEXT_DIM

@dataclass
class Config:
    data_root: str = './data'
    base_output_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    exp_name: str = 'Baseline_NoDNC_hid260'
    model_no: int = VARIANT['model_no']
    ablation_name: str = VARIANT['ablation_name']
    run_id: str = VARIANT['run_id']
    subsets: tuple = ('A', 'U', 'C', 'G')
    k_folds: int = 5
    seq_len: int = 41
    rinalmo_dim: int = 1280
    hidden_dim: int = 260
    strict_feature_check: bool = True
    cnn_kernel_sizes: tuple = (3, 3, 3)
    transformer_layers: int = 2
    num_heads: int = 5
    uses_rinalmo: bool = VARIANT['uses_rinalmo']
    uses_onehot_crnn: bool = VARIANT['uses_onehot_crnn']
    uses_transformer: bool = VARIANT['uses_transformer']
    use_relative_position: bool = True
    use_adaptive_incidence: bool = True
    use_hypergraph_node_update: bool = True
    hyperedges_per_type: int = 3
    relative_position_dim: int = 64
    hypergraph_dropout: float = 0.28
    hypergraph_residual_scale: float = 0.25
    reuse_existing_fold_models: bool = True
    seed: int = 2002
    num_workers: int = 0
    msd_samples: int = VARIANT['msd_samples']
    use_ensemble: bool = True
    skip_finished_folds: bool = True
    skip_finished_runs: bool = True
    delete_model_after_eval: bool = False
    evaluate_test: bool = False
    batch_size: int = 64
    eval_batch_size: int = 256
    preload_eval_features: bool = True
    accumulate_grad_batches: int = 1
    dropout: float = 0.24
    epochs: int = 35
    lr: float = 0.00075
    weight_decay: float = 0.008
    early_stop_patience: int = 10
    ema_decay: float = 0.997
    train_csv_map: dict = field(init=False)
    test_csv_map: dict = field(init=False)
    rinalmo_dir_map: dict = field(init=False)

    def __post_init__(self):
        self.train_csv_map = {subset: os.path.join(self.data_root, f'{subset}_H2Opred_Tr.csv') for subset in self.subsets}
        self.test_csv_map = {subset: os.path.join(self.data_root, f'{subset}_H2Opred_Te.csv') for subset in self.subsets}
        self.rinalmo_dir_map = {subset: os.path.join('./features', f'rinalmo_features_H2Opred_{subset}') for subset in self.subsets}

    def get_experiment_dir(self):
        return os.path.join(self.base_output_dir, self.exp_name)

    def get_run_dir(self):
        return self.get_experiment_dir()

    def get_output_dir(self):
        return os.path.join(self.get_experiment_dir(), 'model')

    def get_done_flag(self):
        return os.path.join(self.get_output_dir(), 'DONE.flag')

    def get_results_csv(self):
        return os.path.join(self.get_experiment_dir(), 'results', 'final_test_metrics.csv')

    def get_sorted_results_csv(self):
        return os.path.join(self.get_experiment_dir(), 'results', 'final_test_metrics_sorted_by_AUPRC.csv')

    def get_sorted_results_by_auc_csv(self):
        return os.path.join(self.get_experiment_dir(), 'results', 'final_test_metrics_sorted_by_AUC.csv')

    def get_detailed_results_csv(self):
        return os.path.join(self.get_experiment_dir(), 'results', 'detailed_metrics.csv')

    def validate(self):
        main_dim = self.hidden_dim * int(self.uses_rinalmo) + self.hidden_dim * int(self.uses_onehot_crnn)
        if main_dim <= 0:
            raise ValueError('RiNALMo and one-hot CRNN branches cannot both be disabled.')
        if self.uses_transformer and main_dim % self.num_heads != 0:
            raise ValueError(f'main_dim={main_dim} must be divisible by num_heads={self.num_heads}.')
        if self.hyperedges_per_type < 2:
            raise ValueError('hyperedges_per_type must be at least 2.')
        if self.hypergraph_residual_scale < 0:
            raise ValueError('hypergraph_residual_scale must be non-negative.')
        if self.uses_transformer and 2 * self.hidden_dim % self.num_heads != 0:
            raise ValueError(f'2 * hidden_dim={2 * self.hidden_dim} must be divisible by num_heads={self.num_heads}.')
        if self.uses_transformer and self.transformer_layers < 1:
            raise ValueError('transformer_layers must be at least 1 when Transformer is enabled.')

def append_result_to_csv(cfg: Config, subset_name: str, metrics: dict, opt_th: float):
    csv_path = cfg.get_results_csv()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    header = ['Model_No', 'Ablation_Name', 'Run_ID', 'Experiment_Name', 'Mode', 'Target_Subset', 'Opt_Threshold', 'LOSS', 'AUC', 'AUPRC', 'ACC', 'MCC', 'F1', 'Recall', 'Specificity', 'Precision', 'TP', 'TN', 'FP', 'FN', 'hidden_dim', 'dropout', 'cnn_kernel_sizes', 'num_heads', 'uses_rinalmo', 'uses_onehot_crnn', 'uses_transformer', 'msd_samples', 'lr', 'weight_decay', 'batch_size', 'epochs', 'seed', 'hyperedges_per_type', 'relative_position_dim', 'hypergraph_residual_scale', 'note']
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([cfg.model_no, cfg.ablation_name, cfg.run_id, cfg.run_id, 'generic', subset_name, f'{opt_th:.4f}', f"{metrics.get('loss', 0.0):.6f}", f"{metrics['auc']:.6f}", f"{metrics['auprc']:.6f}", f"{metrics['acc']:.6f}", f"{metrics['mcc']:.6f}", f"{metrics['f1']:.6f}", f"{metrics['recall']:.6f}", f"{metrics.get('specificity', 0.0):.6f}", f"{metrics['precision']:.6f}", int(metrics.get('tp', 0)), int(metrics.get('tn', 0)), int(metrics.get('fp', 0)), int(metrics.get('fn', 0)), cfg.hidden_dim, cfg.dropout, '-'.join(map(str, cfg.cnn_kernel_sizes)), cfg.num_heads, cfg.uses_rinalmo, cfg.uses_onehot_crnn, cfg.uses_transformer, cfg.msd_samples, cfg.lr, cfg.weight_decay, cfg.batch_size, cfg.epochs, cfg.seed, cfg.hyperedges_per_type, cfg.relative_position_dim, cfg.hypergraph_residual_scale, VARIANT['note']])

def append_detailed_record(cfg: Config, subset_name: str, fold: str, split: str, metrics: dict):
    csv_path = cfg.get_detailed_results_csv()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    header = ['Model_No', 'Ablation_Name', 'Run_ID', 'Experiment_Name', 'Mode', 'Target_Subset', 'Fold', 'Split', 'LOSS', 'AUC', 'AUPRC', 'ACC', 'MCC', 'F1', 'Recall', 'Specificity', 'Precision', 'TP', 'TN', 'FP', 'FN']
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([cfg.model_no, cfg.ablation_name, cfg.run_id, cfg.run_id, 'generic', subset_name, fold, split, f"{metrics.get('loss', 0.0):.6f}", f"{metrics['auc']:.6f}", f"{metrics['auprc']:.6f}", f"{metrics['acc']:.6f}", f"{metrics['mcc']:.6f}", f"{metrics['f1']:.6f}", f"{metrics['recall']:.6f}", f"{metrics.get('specificity', 0.0):.6f}", f"{metrics['precision']:.6f}", int(metrics.get('tp', 0)), int(metrics.get('tn', 0)), int(metrics.get('fp', 0)), int(metrics.get('fn', 0))])

def _write_sorted_csv(src: str, dst: str, primary_metric: str):
    if not os.path.isfile(src):
        return
    with open(src, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def sort_key(row):
        is_mean = 1 if row.get('Target_Subset') == 'AUCG_Mean' else 0
        try:
            primary = float(row.get(primary_metric, 0.0))
            auprc = float(row.get('AUPRC', 0.0))
            auc = float(row.get('AUC', 0.0))
            mcc = float(row.get('MCC', 0.0))
            f1 = float(row.get('F1', 0.0))
        except Exception:
            primary = auprc = auc = mcc = f1 = 0.0
        return (is_mean, primary, auprc, auc, mcc, f1)
    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows_sorted[0].keys())
        writer.writeheader()
        writer.writerows(rows_sorted)

def write_sorted_results(cfg: Config):
    src = cfg.get_results_csv()
    _write_sorted_csv(src, cfg.get_sorted_results_csv(), 'AUPRC')
    _write_sorted_csv(src, cfg.get_sorted_results_by_auc_csv(), 'AUC')

def mean_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            out[k] = float(np.mean(vals))
    return out

def cleanup_model_files(cfg: Config, model_paths):
    if not cfg.delete_model_after_eval:
        return
    for path in model_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            print(f'[Warn] Failed to delete model file: {path} | {e}')

def config_signature_payload(cfg: Config):
    payload = asdict(cfg)
    for key in ('base_output_dir', 'exp_name', 'run_id', 'ablation_name', 'reuse_existing_fold_models', 'skip_finished_runs', 'skip_finished_folds', 'delete_model_after_eval', 'num_workers', 'preload_eval_features', 'evaluate_test'):
        payload.pop(key, None)
    payload.pop('train_csv_map', None)
    payload.pop('test_csv_map', None)
    payload.pop('rinalmo_dir_map', None)
    payload['center_gaussian_sigma'] = float(CENTER_GAUSSIAN_SIGMA)
    return payload

def get_config_signature(cfg: Config):
    payload = config_signature_payload(cfg)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return {'algorithm': 'sha256', 'signature': hashlib.sha256(encoded.encode('utf-8')).hexdigest(), 'config': payload}

def config_signature_matches(cfg: Config, save_dir: str):
    path = os.path.join(save_dir, 'config_signature.json')
    if not os.path.isfile(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            old = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return old.get('signature') == get_config_signature(cfg)['signature']

def save_config_signature(cfg: Config, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config_signature.json'), 'w', encoding='utf-8') as handle:
        json.dump(get_config_signature(cfg), handle, ensure_ascii=False, indent=2)

def live_config_snapshot(cfg: Config):
    return {'exp_name': cfg.exp_name, 'run_id': cfg.run_id, 'hidden_dim': cfg.hidden_dim, 'hyperedges_per_type': cfg.hyperedges_per_type, 'relative_position_dim': cfg.relative_position_dim, 'hypergraph_dropout': cfg.hypergraph_dropout, 'hypergraph_residual_scale': cfg.hypergraph_residual_scale, 'dropout': cfg.dropout, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay, 'transformer_layers': cfg.transformer_layers, 'num_heads': cfg.num_heads, 'epochs': cfg.epochs, 'k_folds': cfg.k_folds, 'seed': cfg.seed, 'batch_size': cfg.batch_size, 'num_workers': cfg.num_workers, 'center_gaussian_sigma': float(CENTER_GAUSSIAN_SIGMA), 'local_context_dim': local_feature_channels(), 'uses_rinalmo': cfg.uses_rinalmo, 'uses_onehot_crnn': cfg.uses_onehot_crnn, 'uses_transformer': cfg.uses_transformer, 'use_adaptive_incidence': cfg.use_adaptive_incidence}

def save_experiment_config(cfg: Config, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_dict.pop('train_csv_map', None)
    cfg_dict.pop('test_csv_map', None)
    cfg_dict.pop('rinalmo_dir_map', None)
    cfg_dict['mode'] = 'generic'
    cfg_dict['architecture'] = f'RiNALMo ({cfg.rinalmo_dim} -> {cfg.hidden_dim}) and the {local_feature_channels()}-channel physchem CNN produce position-wise nodes. A static shared hypergraph performs node-to-hyperedge-to-node propagation. A {cfg.transformer_layers}-layer Transformer with {cfg.num_heads} heads encodes the updated sequence before center-token classification.'
    cfg_dict['optimization'] = 'BinaryFocalLoss(alpha=0.25, gamma=2.0), AdamW, OneCycleLR, EMA'
    cfg_dict['checkpoint_selection'] = 'Each fold retains only its current best EMA checkpoint. The per-fold selection score is validation AUPRC, then MCC, F1, and lower focal loss as sequential tie-breakers.'
    cfg_dict['pooling'] = 'Only the Transformer center token is classified; no Stage-2 hypergraph readout or gate is used.'
    cfg_dict['note'] = VARIANT['note']
    with open(os.path.join(save_dir, 'config_NoGAT_Main.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_sequence_label_csv(path: str, subset_name: str):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 2:
                try:
                    seq = line[0].strip().upper().replace('T', 'U')
                    label = int(float(line[1]))
                    rows.append((seq, label, subset_name))
                except Exception:
                    continue
    return rows

def load_subset_samples(cfg: Config, subset_name: str, split: str):
    subset_name = subset_name.upper()
    if split == 'train':
        return read_sequence_label_csv(cfg.train_csv_map[subset_name], subset_name)
    if split == 'test':
        return read_sequence_label_csv(cfg.test_csv_map[subset_name], subset_name)
    raise ValueError(f'Unsupported split: {split}')

def load_generic_datasets(cfg: Config):
    train_samples = []
    test_samples_by_subset = {}
    for subset in cfg.subsets:
        train_samples.extend(load_subset_samples(cfg, subset, 'train'))
        test_samples_by_subset[subset] = load_subset_samples(cfg, subset, 'test')
    return (train_samples, test_samples_by_subset)

def pad_or_truncate_seq(seq: str, seq_len: int):
    seq = (seq or '').strip().upper().replace('T', 'U')
    valid_len = min(len(seq), seq_len)
    seq_use = seq[:seq_len] if len(seq) >= seq_len else seq + 'N' * (seq_len - len(seq))
    mask = np.zeros(seq_len, dtype=np.float32)
    mask[:valid_len] = 1.0
    return (seq_use, mask, valid_len)
NCP = {'A': (1.0, 1.0, 0.0), 'G': (1.0, 0.0, 1.0), 'C': (0.0, 1.0, 1.0), 'U': (0.0, 0.0, 0.0)}
BASE_PROP = {'A': (0.27, 0.5, 0.33, 0.5, 1.0, 0.0), 'G': (0.35, 0.0, 0.67, 1.0, 1.0, 0.0), 'C': (0.23, 0.5, 0.67, 1.0, 0.0, 1.0), 'U': (0.2, 0.0, 0.33, 0.5, 0.0, 1.0)}
DINUC_ORDER = tuple((a + b for a in 'ACGU' for b in 'ACGU'))
DINUC_INDEX = {pair: index for index, pair in enumerate(DINUC_ORDER)}
DINUC = {pair: index for index, pair in enumerate(DINUC_ORDER)}

def _clean_feature_seq(seq, length):
    seq = str(seq).upper().replace('T', 'U')
    seq = ''.join((ch if ch in 'AUCG' else 'N' for ch in seq))
    return seq[:length].ljust(length, 'N')

def dinucleotide_composition(seq):
    """Return the standard 16-D overlapping RNA DNC vector.

    Each valid adjacent pair is counted once. The count is normalized by the
    number of valid adjacent pair opportunities, which equals K - 1 for a
    complete length-K RNA sequence and excludes padding/unknown boundaries.
    """
    seq = str(seq).upper().replace('T', 'U')
    counts = np.zeros(len(DINUC_ORDER), dtype=np.float32)
    valid_opportunities = 0
    for left, right in zip(seq, seq[1:]):
        pair = left + right
        if pair not in DINUC_INDEX:
            continue
        counts[DINUC_INDEX[pair]] += 1.0
        valid_opportunities += 1
    if valid_opportunities:
        counts /= float(valid_opportunities)
    return counts

def physicochemical_encoding(seq, seq_len=41):
    """Build [21, L]: NCP + broadcast standard DNC + position."""
    seq = _clean_feature_seq(seq, seq_len)
    features = np.zeros((PHYS_DIM, seq_len), dtype=np.float32)
    center = (seq_len - 1) / 2.0
    dnc = dinucleotide_composition(seq)
    for i, base_char in enumerate(seq):
        if base_char == 'N':
            continue
        features[0:3, i] = NCP[base_char]
        features[3:19, i] = dnc
        rel = (i - center) / max(center, 1.0)
        features[19, i] = rel
        features[20, i] = np.exp(-0.5 * (rel / CENTER_GAUSSIAN_SIGMA) ** 2)
    if LOCAL_ENCODING_ABLATION == 'noNCP':
        return features[3:21]
    if LOCAL_ENCODING_ABLATION == 'noDNC':
        return np.concatenate((features[0:3], features[19:21]), axis=0)
    if LOCAL_ENCODING_ABLATION == 'noRelativePosition':
        return np.concatenate((features[0:19], features[20:21]), axis=0)
    if LOCAL_ENCODING_ABLATION == 'noCenterGaussian':
        return features[0:20]
    return features

def audit_local_feature_tensor():
    """Validate shape, finiteness, and DNC normalization of the live tensor.

    A per-sequence DNC vector is allowed to contain zero-valued categories:
    a probe sequence need not contain all 16 possible adjacent pairs.
    """
    probe = ('AUGC' * 11)[:41]
    feature = physicochemical_encoding(probe, 41)
    expected_dim = local_feature_channels()
    if feature.shape != (expected_dim, 41):
        raise RuntimeError(f'Local feature shape is {feature.shape}, expected ({expected_dim}, 41).')
    if not np.isfinite(feature).all():
        raise RuntimeError('Local feature tensor contains NaN or infinity.')
    live_counts = np.count_nonzero(feature, axis=1)
    if LOCAL_ENCODING_ABLATION == 'noNCP':
        dnc_block = feature[0:16]
    elif LOCAL_ENCODING_ABLATION == 'noDNC':
        dnc_block = np.empty((0, feature.shape[1]))
    else:
        dnc_block = feature[3:19]
    dnc_sum_by_position = dnc_block.sum(axis=0)
    if LOCAL_ENCODING_ABLATION != 'noDNC' and (not np.allclose(dnc_sum_by_position, 1.0, atol=1e-06)):
        raise RuntimeError(f'Standard DNC frequencies must sum to 1 at every valid position: sums={dnc_sum_by_position.tolist()}')
    if LOCAL_ENCODING_ABLATION != 'noDNC' and np.count_nonzero(dnc_block) == 0:
        raise RuntimeError('Standard DNC audit found no valid dinucleotide category.')
    rel_index = {None: 19, 'noNCP': 16, 'noDNC': 3, 'noCenterGaussian': 19}.get(LOCAL_ENCODING_ABLATION)
    gaussian_index = {None: 20, 'noNCP': 17, 'noDNC': 4, 'noRelativePosition': 19}.get(LOCAL_ENCODING_ABLATION)
    if rel_index is not None and (not (np.any(feature[rel_index] < 0.0) and np.any(feature[rel_index] > 0.0))):
        raise RuntimeError('Center-relative position channel is not two-sided.')
    if gaussian_index is not None and (not np.all(feature[gaussian_index] > 0.0)):
        raise RuntimeError('Gaussian center-indicator channel must be positive.')
    return {'live_channel_nonzero': live_counts.tolist(), 'allowed_zero_dnc_channels': 0, 'shape': list(feature.shape), 'local_encoding_ablation': LOCAL_ENCODING_ABLATION or 'none'}

def seq_to_onehot(seq: str):
    """Compatibility wrapper; the returned tensor is the live 21-D physchem map."""
    return physicochemical_encoding(seq, 41)

def load_rinalmo_feature(seq_use: str, subset_name: str, cfg: Config):
    npy_path = os.path.join(cfg.rinalmo_dir_map[subset_name], f'{seq_use}.npy')
    if not os.path.exists(npy_path):
        msg = f'RiNALMo feature file not found: {npy_path}\nExtract features first, or check data_root, rinalmo_dir_map, and sequence preprocessing consistency.'
        if cfg.strict_feature_check:
            raise FileNotFoundError(msg)
        return np.zeros((cfg.seq_len, cfg.rinalmo_dim), dtype=np.float32)
    rinalmo_feat = np.load(npy_path, allow_pickle=False).astype(np.float32, copy=False)
    if rinalmo_feat.ndim != 2:
        squeezed = np.squeeze(rinalmo_feat)
        if squeezed.ndim == 2:
            rinalmo_feat = squeezed
    if rinalmo_feat.shape != (cfg.seq_len, cfg.rinalmo_dim):
        msg = f'Invalid RiNALMo feature shape: {npy_path} -> {rinalmo_feat.shape}; expected ({cfg.seq_len}, {cfg.rinalmo_dim}).'
        if cfg.strict_feature_check:
            raise ValueError(msg)
        fixed = np.zeros((cfg.seq_len, cfg.rinalmo_dim), dtype=np.float32)
        h = min(cfg.seq_len, rinalmo_feat.shape[0])
        w = min(cfg.rinalmo_dim, rinalmo_feat.shape[1])
        fixed[:h, :w] = rinalmo_feat[:h, :w]
        rinalmo_feat = fixed
    return rinalmo_feat

def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    pos, neg = (y_true == 1, y_true == 0)
    auc = 0.0
    if int(pos.sum()) > 0 and int(neg.sum()) > 0:
        order = np.argsort(-y_prob, kind='mergesort')
        y_true_sorted = y_true[order]
        tps = np.cumsum(y_true_sorted)
        fps = np.arange(1, len(y_true_sorted) + 1) - tps
        tpr = np.r_[0, tps / max(tps[-1], 1)]
        fpr = np.r_[0, fps / max(fps[-1], 1)]
        auc = float(np.trapz(tpr, fpr))
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = 0.0
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    numerator = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-10
    return {'auc': auc, 'auprc': auprc, 'acc': acc, 'recall': recall, 'specificity': specificity, 'precision': precision, 'f1': f1, 'mcc': numerator / denominator, 'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}

class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class RNA_Dataset(Dataset):

    def __init__(self, data_list, cfg: Config):
        self.samples = data_list
        self.cfg = cfg
        self.seq_len = cfg.seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label, subset_name = self.samples[idx]
        seq_use, mask, _ = pad_or_truncate_seq(seq, self.seq_len)
        item = {'onehot': torch.from_numpy(seq_to_onehot(seq_use)), 'mask': torch.from_numpy(mask), 'label': torch.tensor(label, dtype=torch.float32)}
        if self.cfg.uses_rinalmo:
            item['rinalmo_feat'] = torch.from_numpy(load_rinalmo_feature(seq_use, subset_name, self.cfg))
        return item

class PreloadedRNA_Dataset(Dataset):

    def __init__(self, data_list, cfg: Config, desc: str='Preload eval features'):
        self.samples = data_list
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        onehot_feats, masks, labels = ([], [], [])
        rinalmo_feats = [] if cfg.uses_rinalmo else None
        iterator = tqdm(self.samples, desc=desc, ncols=120, dynamic_ncols=True, leave=False)
        for seq, label, subset_name in iterator:
            seq_use, mask, _ = pad_or_truncate_seq(seq, self.seq_len)
            onehot_feats.append(seq_to_onehot(seq_use))
            masks.append(mask)
            labels.append(label)
            if cfg.uses_rinalmo:
                rinalmo_feats.append(load_rinalmo_feature(seq_use, subset_name, cfg))
        self.onehot_feats = torch.from_numpy(np.stack(onehot_feats).astype(np.float32, copy=False))
        self.masks = torch.from_numpy(np.stack(masks).astype(np.float32, copy=False))
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rinalmo_feats = torch.from_numpy(np.stack(rinalmo_feats).astype(np.float32, copy=False)) if cfg.uses_rinalmo else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {'onehot': self.onehot_feats[idx], 'mask': self.masks[idx], 'label': self.labels[idx]}
        if self.rinalmo_feats is not None:
            item['rinalmo_feat'] = self.rinalmo_feats[idx]
        return item

def write_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

class SharedHypergraphNodeUpdate(nn.Module):
    """One shared node -> hyperedge -> node propagation step.

    The baseline joins both modality node sets. Branch ablations retain the
    same hypergraph operation over the one modality that remains.
    """

    def __init__(self, dim, seq_len, total_edges, position_dim, dropout, use_relative_position=True, use_adaptive_incidence=True):
        super().__init__()
        self.total_edges = total_edges
        self.use_relative_position = use_relative_position
        self.use_adaptive_incidence = use_adaptive_incidence
        self.position_embedding = nn.Embedding(seq_len, position_dim)
        self.node = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.membership = nn.Linear(dim + position_dim, total_edges)
        self.edge = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.node_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.node_ln = nn.LayerNorm(dim)

    def forward(self, rinalmo_nodes, sequence_nodes, mask):
        active_nodes = [nodes for nodes in (rinalmo_nodes, sequence_nodes) if nodes is not None]
        if not active_nodes:
            raise RuntimeError('SharedHypergraphNodeUpdate requires at least one active modality branch.')
        batch, seq_len, _ = active_nodes[0].shape
        nodes = self.node(torch.cat(active_nodes, dim=1))
        pos = self.position_embedding(torch.arange(seq_len, device=nodes.device)).unsqueeze(0).expand(batch, -1, -1)
        if not self.use_relative_position:
            pos = torch.zeros_like(pos)
        pos = torch.cat([pos] * len(active_nodes), dim=1)
        joint_mask = torch.cat([mask] * len(active_nodes), dim=1)
        if self.use_adaptive_incidence:
            logits = self.membership(torch.cat([nodes, pos], dim=-1)).transpose(1, 2)
            logits = logits.masked_fill(joint_mask.unsqueeze(1) == 0, -10000.0)
            incidence = torch.softmax(logits, dim=-1)
        else:
            incidence = joint_mask.unsqueeze(1).float()
            incidence = incidence / incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)
            incidence = incidence.expand(-1, self.total_edges, -1)
        edges = self.edge(torch.bmm(incidence, nodes))
        degree = incidence.transpose(1, 2).sum(dim=-1, keepdim=True).clamp_min(1e-06)
        updated = self.node_ln(nodes + self.node_out(torch.bmm(incidence.transpose(1, 2), edges) / degree))
        updated = updated * joint_mask.unsqueeze(-1)
        chunks = list(updated.split(seq_len, dim=1))
        updated_rinalmo = chunks.pop(0) if rinalmo_nodes is not None else None
        updated_sequence = chunks.pop(0) if sequence_nodes is not None else None
        return (updated_rinalmo, updated_sequence)

class Hyper2OMModel(nn.Module):
    """Fusion model combining RiNALMo, local features, a shared hypergraph, and a Transformer."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.hidden_dim = cfg.hidden_dim
        if cfg.uses_rinalmo:
            self.rinalmo_proj = nn.Sequential(nn.LayerNorm(cfg.rinalmo_dim), nn.Linear(cfg.rinalmo_dim, self.hidden_dim * 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout * 0.5))
            self.rinalmo_ln = nn.LayerNorm(self.hidden_dim)
        if cfg.uses_onehot_crnn:
            k1, k2, k3 = cfg.cnn_kernel_sizes
            self.cnn_extractor = nn.Sequential(nn.Conv1d(local_feature_channels(), 64, kernel_size=k1, padding='same'), nn.BatchNorm1d(64), nn.GELU(), nn.Conv1d(64, 64, kernel_size=k2, padding='same'), nn.BatchNorm1d(64), nn.GELU(), nn.Conv1d(64, self.hidden_dim, kernel_size=k3, padding='same'), nn.BatchNorm1d(self.hidden_dim), nn.GELU())
            self.crnn_ln = nn.LayerNorm(self.hidden_dim)
        self.main_dim = self.hidden_dim * int(cfg.uses_rinalmo) + self.hidden_dim * int(cfg.uses_onehot_crnn)
        active_branches = int(cfg.uses_rinalmo) + int(cfg.uses_onehot_crnn)
        self.hypergraph_node_update = SharedHypergraphNodeUpdate(self.hidden_dim, cfg.seq_len, cfg.hyperedges_per_type * active_branches, cfg.relative_position_dim, cfg.hypergraph_dropout, cfg.use_relative_position, cfg.use_adaptive_incidence)
        if cfg.uses_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.main_dim, nhead=cfg.num_heads, dim_feedforward=self.main_dim * 2, dropout=cfg.dropout, batch_first=True)
            self.attention_block = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(cfg.transformer_layers)])
        self.classifier_base = nn.Sequential(nn.Linear(self.main_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU())
        self.msd_dropouts = nn.ModuleList([nn.Dropout(cfg.dropout * 0.8) for _ in range(cfg.msd_samples)])
        self.classifier_head = nn.Linear(64, 1)

    def forward(self, rinalmo_feat, onehot, mask):
        main_features = []
        if self.cfg.uses_rinalmo:
            if rinalmo_feat is None:
                raise RuntimeError('This version requires rinalmo_feat, but the batch does not provide it.')
            x_rinalmo = self.rinalmo_ln(self.rinalmo_proj(rinalmo_feat))
        else:
            x_rinalmo = None
        if self.cfg.uses_onehot_crnn:
            x_cnn_raw = self.cnn_extractor(onehot)
            x_cnn = x_cnn_raw.transpose(1, 2)
            x_seq = self.crnn_ln(x_cnn)
        else:
            x_seq = None
        if self.cfg.uses_rinalmo and self.cfg.uses_onehot_crnn:
            main_features = [x_rinalmo, x_seq]
        elif self.cfg.uses_rinalmo:
            main_features = [x_rinalmo]
        elif self.cfg.uses_onehot_crnn:
            main_features = [x_seq]
        else:
            raise RuntimeError('No main branch is enabled: RiNALMo and one-hot CRNN cannot both be disabled.')
        x_rinalmo, x_seq = self.hypergraph_node_update(x_rinalmo, x_seq, mask)
        main_features = [nodes for nodes in (x_rinalmo, x_seq) if nodes is not None]
        fused_feat = torch.cat(main_features, dim=-1)
        if self.cfg.uses_transformer:
            key_padding_mask = mask == 0
            attended_feat = fused_feat
            for layer in self.attention_block:
                attended_feat = layer(attended_feat, src_key_padding_mask=key_padding_mask)
            attended_feat = attended_feat * mask.unsqueeze(-1)
        else:
            attended_feat = fused_feat * mask.unsqueeze(-1)
        pooled_features = attended_feat[:, self.seq_len // 2, :]
        base_out = self.classifier_base(pooled_features)
        if not self.training:
            return self.classifier_head(base_out)
        logits = torch.mean(torch.stack([self.classifier_head(dropout(base_out)) for dropout in self.msd_dropouts], dim=0), dim=0)
        return logits

class ModelEMA:

    def __init__(self, model, decay=0.995):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                model_v = msd[k].detach()
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))

def run_one_epoch(model, loader, optimizer, criterion, device, is_train: bool, scheduler=None, cfg=None, ema=None, desc: str=''):
    model.train() if is_train else model.eval()
    total_loss, all_probs, all_labels = (0.0, [], [])
    accumulation_steps = cfg.accumulate_grad_batches if is_train and cfg else 1
    iterator = tqdm(loader, total=len(loader), desc=desc if desc else 'Train' if is_train else 'Eval', leave=False, ncols=120, dynamic_ncols=True)
    for i, batch in enumerate(iterator):
        rinalmo_feat = batch.get('rinalmo_feat')
        if rinalmo_feat is not None:
            rinalmo_feat = rinalmo_feat.to(device, non_blocking=True)
        onehot = batch['onehot'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        if is_train:
            logits = model(rinalmo_feat, onehot, mask).view(-1)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                if ema:
                    ema.update(model)
        else:
            with torch.inference_mode():
                logits = model(rinalmo_feat, onehot, mask).view(-1)
                loss = criterion(logits, labels)
        total_loss += loss.item() * (accumulation_steps if is_train else 1)
        all_probs.append(torch.sigmoid(logits if not is_train else logits.detach()).cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        postfix = {'loss': f'{loss.item():.4f}'}
        if is_train and scheduler is not None:
            postfix['lr'] = f'{scheduler.get_last_lr()[0]:.2e}'
        iterator.set_postfix(postfix)
    all_probs, all_labels = (np.concatenate(all_probs), np.concatenate(all_labels))
    metrics = calculate_metrics(all_labels, all_probs, threshold=0.5)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return (metrics, all_probs, all_labels)

def build_loader(samples, cfg: Config, shuffle: bool, is_eval: bool=False, desc: str=''):
    if is_eval and cfg.preload_eval_features:
        dataset = PreloadedRNA_Dataset(samples, cfg, desc=desc or 'Preload eval features')
        return DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    dataset = RNA_Dataset(samples, cfg)
    batch_size = cfg.eval_batch_size if is_eval else cfg.batch_size
    loader_kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=cfg.num_workers > 0)
    if cfg.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    return DataLoader(**loader_kwargs)

def evaluate_saved_fold_models(cfg: Config, model_paths, all_train_samples, test_samples_by_subset, device):
    """Evaluate a complete saved fold ensemble without entering the training loop."""
    print(f'[Resume] Found {len(model_paths)} complete fold checkpoints. Skipping training and evaluating them.')
    eval_models = []
    for fold_id, path in enumerate(model_paths, start=1):
        model = Hyper2OMModel(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        eval_models.append(model)
        print(f'   Loaded fold {fold_id}/{len(model_paths)}: {path}')
    validation_prediction_rows = []
    all_labels = np.array([f'{sample[2]}_{sample[1]}' for sample in all_train_samples])
    kf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    for fold_id, (_, val_idx) in enumerate(kf.split(all_train_samples, all_labels), start=1):
        val_samples = [all_train_samples[i] for i in val_idx]
        val_loader = build_loader(val_samples, cfg, shuffle=False, is_eval=True, desc=f'Preload Val {fold_id}')
        model = eval_models[fold_id - 1]
        _, probabilities, labels = run_one_epoch(model, val_loader, None, nn.BCEWithLogitsLoss().to(device), device, False, cfg=cfg, desc=f'Val {fold_id}')
        for (sequence, label, subset), probability in zip(val_samples, probabilities):
            validation_prediction_rows.append({'subset': subset, 'sequence': sequence, 'label': int(label), 'probability': float(probability)})
    write_rows(os.path.join(cfg.get_experiment_dir(), 'results', 'validation_predictions.csv'), validation_prediction_rows)
    eval_criterion = nn.BCEWithLogitsLoss().to(device)
    subset_metric_list, test_prediction_rows = ([], [])
    for subset_name in cfg.subsets:
        if subset_name not in test_samples_by_subset:
            continue
        samples = test_samples_by_subset[subset_name]
        test_loader = build_loader(samples, cfg, shuffle=False, is_eval=True, desc=f'Preload Test {subset_name}')
        fold_probs, test_true = ([], None)
        for model in eval_models:
            _, probs, labels = run_one_epoch(model, test_loader, None, eval_criterion, device, False, cfg=cfg, desc=f'Test {subset_name}')
            fold_probs.append(probs)
            test_true = labels if test_true is None else test_true
        final_probs = np.mean(fold_probs, axis=0) if cfg.use_ensemble else fold_probs[0]
        metrics = calculate_metrics(test_true, final_probs, threshold=0.5)
        append_result_to_csv(cfg, subset_name, metrics, 0.5)
        append_detailed_record(cfg, subset_name, 'Ensemble', 'Test', metrics)
        subset_metric_list.append(metrics)
        for (sequence, label, sample_subset), probability in zip(samples, final_probs):
            test_prediction_rows.append({'subset': sample_subset, 'sequence': sequence, 'label': int(label), 'probability': float(probability)})
        print(f"[Test {subset_name}] AUPRC={metrics['auprc']:.6f} | MCC={metrics['mcc']:.6f}")
    if subset_metric_list:
        mean_metrics_row = mean_metrics(subset_metric_list)
        append_result_to_csv(cfg, 'AUCG_Mean', mean_metrics_row, 0.5)
        append_detailed_record(cfg, 'AUCG_Mean', 'Ensemble', 'Test_Mean', mean_metrics_row)
    write_rows(os.path.join(cfg.get_experiment_dir(), 'results', 'test_predictions.csv'), test_prediction_rows)
    write_sorted_results(cfg)
    with open(cfg.get_done_flag(), 'w', encoding='utf-8') as handle:
        handle.write('DONE\nresume_mode=existing_fold_models\n')
    return {'mode': 'existing_fold_models', 'model_paths': model_paths}

def run_experiment(cfg: Config, all_train_samples, test_samples_by_subset, device):
    cfg.validate()
    save_dir = cfg.get_output_dir()
    done_flag = cfg.get_done_flag()
    signature_ok = config_signature_matches(cfg, save_dir)
    validation_predictions_path = os.path.join(cfg.get_experiment_dir(), 'results', 'validation_predictions.csv')
    if cfg.skip_finished_runs and os.path.exists(done_flag) and signature_ok and os.path.isfile(validation_predictions_path):
        print(f'--> Run {cfg.run_id} already has DONE.flag with a matching config signature; skipping. Use --force to rerun.')
        return {}
    if cfg.skip_finished_runs and os.path.exists(done_flag) and signature_ok:
        print(f'--> Run {cfg.run_id} is missing validation_predictions.csv; existing fold models will be reused to export it.')
    if os.path.exists(done_flag) and (not signature_ok):
        print(f'--> Run {cfg.run_id} has DONE.flag, but the config has changed; retraining.')
    os.makedirs(save_dir, exist_ok=True)
    save_experiment_config(cfg, save_dir)
    save_config_signature(cfg, save_dir)
    existing_model_paths = [os.path.join(save_dir, f'model_fold_{fold_id}_best.pt') for fold_id in range(1, cfg.k_folds + 1)]
    if cfg.reuse_existing_fold_models and signature_ok and all((os.path.isfile(path) for path in existing_model_paths)):
        return evaluate_saved_fold_models(cfg, existing_model_paths, all_train_samples, test_samples_by_subset, device)
    if cfg.reuse_existing_fold_models and (not signature_ok) and all((os.path.isfile(path) for path in existing_model_paths)):
        print(f'--> Run {cfg.run_id} has fold models, but the config signature does not match; ignoring old models and retraining.')
    print(f"\n{'=' * 90}")
    print(f'Running Formal NoGAT Main Model: {cfg.run_id} | mode=generic')
    print(f'Output dir: {save_dir}')
    print(f"Note: {VARIANT['note']}")
    print(f"{'=' * 90}")
    all_labels = np.array([f'{s[2]}_{s[1]}' for s in all_train_samples])
    kf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    saved_models_paths = []
    train_metrics_list, val_metrics_list = ([], [])
    train_val_rows = []
    selection_rows = []
    validation_prediction_rows = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_samples, all_labels)):
        fold_id = fold + 1
        print('\n' + '=' * 88)
        print(f'? Fold {fold_id}/{cfg.k_folds} ?? | Train={len(train_idx)} | Val={len(val_idx)} | batch_size={cfg.batch_size}')
        print('=' * 88)
        train_loader = build_loader([all_train_samples[i] for i in train_idx], cfg, shuffle=True)
        val_loader = build_loader([all_train_samples[i] for i in val_idx], cfg, shuffle=False, is_eval=True, desc=f'Preload Fold {fold_id} Val')
        print(f'   Train batches={len(train_loader)} | Val batches={len(val_loader)}')
        print("   Checkpoint selection: this fold's best validation AUPRC, then MCC/F1/loss tie-breakers.")
        model = Hyper2OMModel(cfg).to(device)
        ema = ModelEMA(model, decay=cfg.ema_decay)
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = max(1, cfg.epochs * math.ceil(len(train_loader) / cfg.accumulate_grad_batches))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0)
        criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0).to(device)
        best_model_path = os.path.join(save_dir, f'model_fold_{fold_id}_best.pt')
        best_auprc, best_epoch = (-float('inf'), 0)
        best_train_res, best_val_res = (None, None)
        best_val_probs, best_val_labels = (None, None)
        for epoch in range(1, cfg.epochs + 1):
            print(f'\n[Fold {fold_id}/{cfg.k_folds}] Epoch {epoch:02d}/{cfg.epochs} ????')
            train_res, _, _ = run_one_epoch(model, train_loader, optimizer, criterion, device, True, scheduler, cfg=cfg, ema=ema, desc=f'F{fold_id}/{cfg.k_folds} E{epoch:02d}/{cfg.epochs} Train')
            val_res, val_probs, val_labels = run_one_epoch(ema.ema, val_loader, optimizer, criterion, device, False, cfg=cfg, desc=f'F{fold_id}/{cfg.k_folds} E{epoch:02d}/{cfg.epochs} Val')
            print(f"Fold {fold_id}/{cfg.k_folds} | Epoch {epoch:02d}/{cfg.epochs} | Train Loss: {train_res['loss']:.4f} | Val AUC: {val_res['auc']:.4f} | Val AUPRC: {val_res['auprc']:.4f} | Val MCC: {val_res['mcc']:.4f}")
            train_val_rows.append({'fold': fold_id, 'epoch': epoch, 'split': 'train', **train_res})
            train_val_rows.append({'fold': fold_id, 'epoch': epoch, 'split': 'val', **val_res})
            if float(val_res['auprc']) > best_auprc:
                best_auprc, best_epoch = (float(val_res['auprc']), epoch)
                best_train_res, best_val_res = (train_res, val_res)
                best_val_probs, best_val_labels = (val_probs, val_labels)
                torch.save(ema.ema.state_dict(), best_model_path)
        if best_val_res is None:
            raise RuntimeError(f'Fold {fold_id} did not produce a validation checkpoint.')
        saved_models_paths.append(best_model_path)
        for sample, label, probability in zip([all_train_samples[i] for i in val_idx], best_val_labels, best_val_probs):
            validation_prediction_rows.append({'subset': sample[2], 'sequence': sample[0], 'label': int(label), 'probability': float(probability)})
        train_metrics_list.append(best_train_res)
        val_metrics_list.append(best_val_res)
        selection_rows.append({'fold': fold_id, 'selected_epoch': best_epoch, 'selection_metric': 'validation_auprc', 'best_validation_auprc': best_auprc, 'model_path': best_model_path})
        print(f'[Fold {fold_id}] selected epoch={best_epoch:02d} | val_AUPRC={best_auprc:.6f} -> {best_model_path}')
    if train_metrics_list and val_metrics_list:
        avg_train_res = {k: float(np.mean([m[k] for m in train_metrics_list])) for k in train_metrics_list[0].keys()}
        avg_val_res = {k: float(np.mean([m[k] for m in val_metrics_list])) for k in val_metrics_list[0].keys()}
        append_detailed_record(cfg, 'AUCG', 'Final_Avg', 'Train', avg_train_res)
        append_detailed_record(cfg, 'AUCG', 'Final_Avg', 'Val', avg_val_res)
    write_rows(os.path.join(cfg.get_experiment_dir(), 'results', 'train_val_metrics.csv'), train_val_rows)
    write_rows(os.path.join(cfg.get_experiment_dir(), 'results', 'model_selection.csv'), selection_rows)
    write_rows(os.path.join(cfg.get_experiment_dir(), 'results', 'validation_predictions.csv'), validation_prediction_rows)
    if not cfg.evaluate_test:
        cleanup_model_files(cfg, saved_models_paths)
        with open(done_flag, 'w', encoding='utf-8') as handle:
            handle.write('DONE\nmode=validation_only_sensitivity\n')
        return {'mode': 'validation_only', 'selected_models': selection_rows}
    print(f"\n{'-' * 50}\nStarting independent test-set evaluation (without GAT or structural matrix input)...\n{'-' * 50}")
    print(f'\n[Test] Loading {len(saved_models_paths)} fold models into GPU/CPU ...')
    eval_models = []
    for idx, path in enumerate(saved_models_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Fold model checkpoint not found: {path}')
        model = Hyper2OMModel(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        eval_models.append(model)
        print(f'   Loaded fold model {idx + 1}/{len(saved_models_paths)}: {os.path.basename(path)}')
    eval_criterion = nn.BCEWithLogitsLoss().to(device)
    subset_metric_list = []
    subset_metric_dict = {}
    test_prediction_rows = []
    for subset_name in cfg.subsets:
        if subset_name not in test_samples_by_subset:
            continue
        test_loader = build_loader(test_samples_by_subset[subset_name], cfg, shuffle=False, is_eval=True, desc=f'Preload Test {subset_name}')
        print(f'\n[Test] Subset {subset_name} | samples={len(test_samples_by_subset[subset_name])} | batches={len(test_loader)} | eval_batch_size={cfg.eval_batch_size}')
        all_fold_probs, test_true = ([], None)
        for idx, model in enumerate(eval_models):
            print(f'   -> Evaluating fold model {idx + 1}/{len(eval_models)} on subset {subset_name}')
            _, test_probs, test_labels = run_one_epoch(model, test_loader, None, eval_criterion, device, False, cfg=cfg, desc=f'Test {subset_name} Fold {idx + 1}/{len(eval_models)}')
            all_fold_probs.append(test_probs)
            if test_true is None:
                test_true = test_labels
        final_probs = np.mean(all_fold_probs, axis=0) if cfg.use_ensemble else all_fold_probs[0]
        final_metrics = calculate_metrics(test_true, final_probs, threshold=0.5)
        append_result_to_csv(cfg, subset_name, final_metrics, 0.5)
        append_detailed_record(cfg, subset_name, 'Ensemble', 'Test', final_metrics)
        for (sequence, label, sample_subset), probability in zip(test_samples_by_subset[subset_name], final_probs):
            test_prediction_rows.append({'subset': sample_subset, 'sequence': sequence, 'label': int(label), 'probability': float(probability)})
        subset_metric_list.append(final_metrics)
        subset_metric_dict[subset_name] = final_metrics
        print(f"[{cfg.run_id} | Subset {subset_name}] Final AUC: {final_metrics['auc']:.4f} | Final AUPRC: {final_metrics['auprc']:.4f} | MCC: {final_metrics['mcc']:.4f} | F1: {final_metrics['f1']:.4f}")
    if subset_metric_list:
        avg_test_metrics = mean_metrics(subset_metric_list)
        append_result_to_csv(cfg, 'AUCG_Mean', avg_test_metrics, 0.5)
        append_detailed_record(cfg, 'AUCG_Mean', 'Ensemble', 'Test_Mean', avg_test_metrics)
        subset_metric_dict['AUCG_Mean'] = avg_test_metrics
        print(f"[{cfg.run_id}] AUCG_Mean | AUC={avg_test_metrics['auc']:.4f} | AUPRC={avg_test_metrics['auprc']:.4f} | MCC={avg_test_metrics['mcc']:.4f} | F1={avg_test_metrics['f1']:.4f}")
    write_rows(os.path.join(cfg.get_experiment_dir(), 'results', 'test_predictions.csv'), test_prediction_rows)
    write_sorted_results(cfg)
    cleanup_model_files(cfg, saved_models_paths)
    with open(done_flag, 'w', encoding='utf-8') as f:
        f.write(f"DONE\nmodel_no={cfg.model_no}\nablation_name={cfg.ablation_name}\nrun_id={cfg.run_id}\nnote={VARIANT['note']}\n")
    return subset_metric_dict

def run_architecture_preflight(cfg: Config):
    """Check architecture invariants only. Hyperparameters come from Config.

    The 1/2/3 Transformer constructions are part of the training RNG sequence.
    Keep that count unchanged so repeated runs preserve the same initialization sequence.
    """
    last_model = None
    layer_checks = (1, 2, 3) if cfg.uses_transformer else (max(int(cfg.transformer_layers), 1),)
    for layers in layer_checks:
        check_cfg = copy.deepcopy(cfg)
        check_cfg.transformer_layers = layers
        last_model = Hyper2OMModel(check_cfg)
        if cfg.uses_transformer and len(last_model.attention_block) != layers:
            raise RuntimeError('Standalone Transformer layer control is inactive.')
    if getattr(last_model, 'hypergraph_node_update', None) is None:
        raise RuntimeError('Expected static SharedHypergraphNodeUpdate when hypergraph node update is on.')
    if type(last_model.hypergraph_node_update).__name__ != 'SharedHypergraphNodeUpdate':
        raise RuntimeError(f'Expected SharedHypergraphNodeUpdate, got {type(last_model.hypergraph_node_update).__name__}')
    if cfg.uses_onehot_crnn:
        first_conv = last_model.cnn_extractor[0]
        expected_channels = local_feature_channels()
        if tuple(first_conv.weight.shape[1:2]) != (expected_channels,):
            raise RuntimeError(f'Expected first Conv1d in_channels={expected_channels}, got {first_conv.weight.shape[1]}.')
        if type(first_conv).__name__ != 'Conv1d':
            raise RuntimeError(f'Expected first Conv1d, got {type(first_conv).__name__}.')
    return last_model

def configure_process_runtime():
    """Use one PyTorch thread and the first visible CUDA device."""
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

def run_seeded_experiment(cfg: Config, train_data, test_data, device):
    """Canonical training entry point."""
    set_seed(cfg.seed)
    run_architecture_preflight(cfg)
    return run_experiment(cfg, train_data, test_data, device)

def build_config(args):
    cfg = Config()
    cfg.base_output_dir = args.output_root if args.output_root is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    cfg.evaluate_test = False
    cfg.use_ensemble = True
    cfg.delete_model_after_eval = False
    cfg.reuse_existing_fold_models = not args.force
    cfg.skip_finished_runs = not args.force
    cfg.skip_finished_folds = False
    cfg.epochs = args.epochs if args.epochs is not None else cfg.epochs
    cfg.k_folds = args.folds if args.folds is not None else cfg.k_folds
    cfg.seed = args.seed if args.seed is not None else cfg.seed
    cfg.num_workers = args.num_workers if args.num_workers is not None else cfg.num_workers
    return cfg

def write_summary(cfg):

    def rows(path):
        if not os.path.isfile(path):
            return []
        with open(path, newline='', encoding='utf-8-sig') as handle:
            return list(csv.DictReader(handle))
    detailed = rows(cfg.get_detailed_results_csv())
    val = next((r for r in reversed(detailed) if r.get('Target_Subset') == 'AUCG' and r.get('Fold') == 'Final_Avg' and (r.get('Split') == 'Val')), {})
    test_rows = rows(cfg.get_results_csv())
    test = next((r for r in reversed(test_rows) if r.get('Target_Subset') == 'AUCG_Mean'), {})
    summary = {'run_id': cfg.run_id, 'experiment_name': cfg.exp_name, 'ablation_name': cfg.ablation_name, 'local_encoding_ablation': LOCAL_ENCODING_ABLATION or 'none', 'local_feature_dim': local_feature_channels(), 'validation_mean_auc': val.get('AUC', ''), 'validation_mean_auprc': val.get('AUPRC', ''), 'validation_mean_mcc': val.get('MCC', ''), 'validation_mean_f1': val.get('F1', ''), 'test_mean_auc': test.get('AUC', ''), 'test_mean_auprc': test.get('AUPRC', ''), 'test_mean_mcc': test.get('MCC', ''), 'test_mean_f1': test.get('F1', ''), 'hidden_dim': cfg.hidden_dim, 'dropout': cfg.dropout, 'lr': cfg.lr, 'batch_size': cfg.batch_size, 'transformer_layers': cfg.transformer_layers, 'num_heads': cfg.num_heads, 'hyperedges_per_type': cfg.hyperedges_per_type, 'relative_position_dim': cfg.relative_position_dim, 'hypergraph_dropout': cfg.hypergraph_dropout, 'hypergraph_residual_scale': cfg.hypergraph_residual_scale, 'weight_decay': cfg.weight_decay}
    result_dir = cfg.get_experiment_dir()
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'summary.csv'), 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    with open(os.path.join(result_dir, 'local_feature_config.json'), 'w', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'local_encoding_ablation': LOCAL_ENCODING_ABLATION or 'none', 'ncp': NCP, 'dinucleotide': DINUC, 'local_feature_dim': local_feature_channels(), 'center_gaussian_sigma': CENTER_GAUSSIAN_SIGMA, 'cnn_init': 'ordinary PyTorch Conv1d initialization'}, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(result_dir, 'architecture.json'), 'w', encoding='utf-8') as handle:
        json.dump({'input': '41-nt RNA window with center-position label', 'branch_1': f'RiNALMo token features, projected {cfg.rinalmo_dim} -> {cfg.hidden_dim}', 'branch_2': f'{local_feature_channels()}-channel physchem ({local_feature_description()})', 'local_feature_channels': {'0-2': 'NCP', '3': 'center-relative coordinate', '4': 'smooth center indicator'}, 'local_encoder': f"Conv1D kernels {','.join(map(str, cfg.cnn_kernel_sizes))}", 'stage_1': f'static shared hypergraph node update; relative_position_dim={cfg.relative_position_dim}; hypergraph_dropout={cfg.hypergraph_dropout}; {cfg.hyperedges_per_type} hyperedges per branch; adaptive incidence', 'fusion': f'updated RiNALMo and local-feature nodes -> concatenate -> {cfg.transformer_layers}-layer Transformer ({cfg.num_heads} heads) -> center token', 'selection': 'per-fold validation AUPRC', 'test': 'mean probability ensemble over selected fold checkpoints'}, handle, ensure_ascii=False, indent=2)
    return summary

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Hyper2OM information-fusion model.')
    parser.add_argument('--force', action='store_true', help='Ignore existing results and models, then retrain.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--folds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--center_gaussian_sigma', type=float, default=CENTER_GAUSSIAN_SIGMA)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    return parser.parse_args()

def main():
    global CENTER_GAUSSIAN_SIGMA
    args = parse_args()
    if LOCAL_ENCODING_ABLATION != 'noDNC' or local_feature_channels() != 5:
        raise RuntimeError('The model requires the configured 5-channel local encoding.')
    if args.center_gaussian_sigma <= 0:
        raise ValueError('--center_gaussian_sigma must be positive.')
    CENTER_GAUSSIAN_SIGMA = args.center_gaussian_sigma
    cfg = build_config(args)
    cfg.validate()
    configure_process_runtime()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_audit = audit_local_feature_tensor()
    train_data, test_data_by_subset = load_generic_datasets(cfg)
    print('=' * 90)
    print(f'Local features: hid{cfg.hidden_dim} + static shared HG (rp={cfg.relative_position_dim}, hgdrop={cfg.hypergraph_dropout}) + Transformer')
    print(f'device={device}')
    print(f'model_dir={cfg.get_output_dir()}')
    print(f'result_dir={cfg.get_experiment_dir()}')
    print(f"Local input channels={local_feature_channels()} (encoding ablation={LOCAL_ENCODING_ABLATION or 'none'})")
    print('Full channel map: 0:3 NCP | 3 relative position | 4 center Gaussian')
    print(f'Feature audit={feature_audit}')
    print('Live Config (editing class Config changes these values):')
    print(json.dumps(live_config_snapshot(cfg), ensure_ascii=False, indent=2))
    print('=' * 90)
    run_seeded_experiment(cfg, train_data, test_data_by_subset, device)
    summary = write_summary(cfg)
    print('\nFinal summary:')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Test predictions:', os.path.join(cfg.get_experiment_dir(), 'results', 'test_predictions.csv'))
    print('Test metrics:', cfg.get_results_csv())
if __name__ == '__main__':
    main()
