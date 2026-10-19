"""Recover exact MELD audio/visual PCA + scaler from erc_pipeline.py raw arrays.

Re-fits on precisely the training rows used by cmd_reduce and checks that the
reconstructed reduced vectors match the previously saved reduced arrays.
Refuses to save a potentially incompatible transform.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TARGET_DIMS = {'audio': 300, 'visual': 342}


def recover_kind(features_dir: Path, kind: str, target_dim: int) -> Path:
    raw_path = features_dir / f'X_{kind}.npy'
    reduced_path = features_dir / f'X_{kind}_{target_dim}.npy'
    meta_path = features_dir / 'metadata_filtered.csv'
    for p in (raw_path, reduced_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}; do not regenerate embeddings without checking checkpoint compatibility.')

    df = pd.read_csv(meta_path)
    X = np.load(raw_path, mmap_mode='r')
    ref = np.load(reduced_path, mmap_mode='r')
    if X.ndim != 2 or ref.shape != (len(X), target_dim) or len(df) != len(X):
        raise ValueError(f'{kind} arrays / metadata have incompatible shapes: raw {X.shape}, reduced {ref.shape}, meta {len(df)}')
    if not np.isfinite(X).all() or not np.isfinite(ref).all():
        raise ValueError(f'{kind} contains NaN or Inf')

    keep = df['split'].to_numpy() == 'train'
    fallback_path = features_dir / f'{kind}_is_fallback.npy'
    if fallback_path.exists():
        fallback = np.load(fallback_path)
        if fallback.shape != (len(df),):
            raise ValueError(f'{fallback_path} length mismatch')
        keep &= ~fallback.astype(bool)
    else:
        print(f'[WARNING] Missing {fallback_path}; cannot exclude failed-media rows as in original training.')

    n_train = int(keep.sum())
    if n_train < target_dim:
        raise ValueError(f'Not enough train rows to fit {target_dim} dimensions ({n_train})')
    print(f'[{kind}] raw={X.shape}, reduced={ref.shape}, true-train={n_train}', flush=True)

    if X.shape[1] > target_dim:
        scaler = StandardScaler().fit(X[keep])
        pca = PCA(n_components=target_dim, random_state=0).fit(scaler.transform(X[keep]))
        projected = pca.transform(scaler.transform(X)).astype(np.float32)
        bundle = {'mode': 'pca', 'scaler': scaler, 'pca': pca,
                  'input_dim': int(X.shape[1]), 'output_dim': target_dim}
    elif X.shape[1] == target_dim:
        projected = np.asarray(X, dtype=np.float32)
        bundle = {'mode': 'identity', 'input_dim': target_dim, 'output_dim': target_dim}
    else:
        # erc_pipeline: --pad -> z-score with 1e-6 + trailing zeros.
        mu = np.asarray(X[keep].mean(axis=0))
        sd = np.asarray(X[keep].std(axis=0) + 1e-6)
        projected = np.zeros(ref.shape, dtype=np.float32)
        projected[:, :X.shape[1]] = (X - mu) / sd
        bundle = {'mode': 'pad', 'mean': mu, 'std': sd,
                  'input_dim': int(X.shape[1]), 'output_dim': target_dim}

    error = np.abs(projected - ref)
    mae = float(error.mean())
    p99 = float(np.percentile(error, 99))
    maxerr = float(error.max())
    rel_rmse = float(np.sqrt(np.mean((projected - ref) ** 2)) /
                     max(float(np.sqrt(np.mean(np.asarray(ref) ** 2))), 1e-8))
    print(f'[{kind}] verification MAE={mae:.7g}, p99={p99:.7g}, max={maxerr:.7g}, rel_RMSE={rel_rmse:.7g}', flush=True)
    if rel_rmse >= 1e-3 or p99 >= 1e-2:
        raise RuntimeError(
            f'{kind}: PCA reconstruction DOES NOT match training features. '
            'Do not use this transform for DiGemo inference. '
            'Check numpy/sklearn versions, whether .npy files were regenerated, '
            'and whether the checkpoint was trained from these features.'
        )

    bundle.update({'kind': kind, 'verification_rel_rmse': rel_rmse,
                   'verification_p99_abs': p99})
    out = features_dir / f'online_{kind}_transform.joblib'
    joblib.dump(bundle, out)
    print(f'[{kind}] saved {out}', flush=True)
    return out


def transform_one(bundle: dict, value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32).reshape(1, -1)
    if x.shape[1] != bundle['input_dim']:
        raise ValueError(f"Expected raw feature dim {bundle['input_dim']}, got {x.shape[1]}")
    mode = bundle['mode']
    if mode == 'pca':
        y = bundle['pca'].transform(bundle['scaler'].transform(x))
    elif mode == 'identity':
        y = x
    elif mode == 'pad':
        y = np.zeros((1, bundle['output_dim']), dtype=np.float32)
        y[:, :x.shape[1]] = (x - bundle['mean']) / bundle['std']
    else:
        raise ValueError(f'Invalid transform mode: {mode}')
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (1, bundle['output_dim']) or not np.isfinite(y).all():
        raise ValueError('Invalid reduced feature')
    return y[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', default='features')
    ap.add_argument('--kind', choices=['audio', 'visual', 'both'], default='both')
    a = ap.parse_args()
    kinds = list(TARGET_DIMS) if a.kind == 'both' else [a.kind]
    for kind in kinds:
        recover_kind(Path(a.features_dir), kind, TARGET_DIMS[kind])
    print('DONE: transforms passed feature-level reconstruction check.')


if __name__ == '__main__':
    main()
