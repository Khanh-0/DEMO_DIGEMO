#!/usr/bin/env python3
"""
Bước 1: Chuẩn hóa iemocap_6class.csv và chia Train/Val/Test theo dialogue_id
"""
import pandas as pd
import json
import os

# Paths
CSV_PATH = r"E:\Do_an\DEMO\demo_meld\checkpoints\iemocap_6class.csv"
OUT_DIR = r"E:\Do_an\DEMO\demo_meld\checkpoints\iemocap"
SPLIT_CSV = os.path.join(OUT_DIR, "splits", "iemocap_6class_split.csv")
SPEAKER_MAP = os.path.join(OUT_DIR, "speaker_map.json")

# Label map
IEMOCAP_LABEL_MAP = {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5}

print("="*80)
print("STEP 1: Prepare IEMOCAP 6-class splits")
print("="*80)

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"\nLoaded CSV: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Verify 6-class only
assert df['is_6class'].all(), "❌ Not all rows are 6-class!"
assert set(df['emotion'].unique()) == set(IEMOCAP_LABEL_MAP.keys()), "❌ Unexpected emotions!"

# Add label_id
df['label_id'] = df['emotion'].map(IEMOCAP_LABEL_MAP)
print(f"\nLabel distribution:")
print(df['label_id'].value_counts().sort_index())

# Sort by conversation order
df = df.sort_values(by=['session', 'dialogue_id', 'start_time']).reset_index(drop=True)
print(f"\nSorted by [session, dialogue_id, start_time]")

# Split by dialogue_id
print(f"\nSplitting by dialogue_id...")

# Test = all Session 5
test_dialogues = df[df['session'] == 5]['dialogue_id'].unique()
df.loc[df['dialogue_id'].isin(test_dialogues), 'split'] = 'test'

# Train/Val = Session 1-4
train_val_df = df[df['session'].isin([1, 2, 3, 4])].copy()
train_val_dialogues = train_val_df['dialogue_id'].unique()

# Val = random 10% dialogues from each session (seed=42)
import numpy as np
np.random.seed(42)

val_dialogues = []
for sess in [1, 2, 3, 4]:
    sess_dialogues = train_val_df[train_val_df['session'] == sess]['dialogue_id'].unique()
    n_val = max(1, int(0.1 * len(sess_dialogues)))
    val_sess = np.random.choice(sess_dialogues, size=n_val, replace=False)
    val_dialogues.extend(val_sess)

df.loc[df['dialogue_id'].isin(val_dialogues), 'split'] = 'val'
df.loc[(df['session'].isin([1,2,3,4])) & (~df['dialogue_id'].isin(val_dialogues)), 'split'] = 'train'

# Check splits
print(f"\nSplit counts:")
print(df['split'].value_counts())
print(f"\nDialogue counts per split:")
for split in ['train', 'val', 'test']:
    n_dial = df[df['split'] == split]['dialogue_id'].nunique()
    n_utt = (df['split'] == split).sum()
    print(f"  {split}: {n_dial} dialogues, {n_utt} utterances")

# Check no overlap
train_dials = set(df[df['split'] == 'train']['dialogue_id'])
val_dials = set(df[df['split'] == 'val']['dialogue_id'])
test_dials = set(df[df['split'] == 'test']['dialogue_id'])

assert len(train_dials & val_dials) == 0, "❌ Train/Val overlap!"
assert len(train_dials & test_dials) == 0, "❌ Train/Test overlap!"
assert len(val_dials & test_dials) == 0, "❌ Val/Test overlap!"
print(f"\n✅ No data leakage: splits are disjoint")

# Save
os.makedirs(os.path.dirname(SPLIT_CSV), exist_ok=True)
df.to_csv(SPLIT_CSV, index=False)
print(f"\n✅ Saved: {SPLIT_CSV}")

# Save speaker_map
speaker_map = {"M": 0, "F": 1, "__other__": 0}
with open(SPEAKER_MAP, 'w') as f:
    json.dump(speaker_map, f, indent=2)
print(f"✅ Saved: {SPEAKER_MAP}")

print(f"\n{'='*80}")
print(f"STEP 1 COMPLETE")
print(f"{'='*80}")
