import os
import sys
import numpy as np
import pandas as pd

# 1. Find the parquet files
try:
    import robocasa.utils.dataset_registry_utils as dru
    ds_path = dru.get_ds_path("OpenCabinet", source="human")
    print(f"Dataset path: {ds_path}")
except Exception as e:
    print(f"Error getting dataset path: {e}")
    sys.exit(1)

chunk_dir = os.path.join(ds_path, "data", "chunk-000")
print(f"Chunk dir: {chunk_dir}")
print(f"Chunk dir exists: {os.path.exists(chunk_dir)}")

parquet_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".parquet")])
print(f"Number of parquet files: {len(parquet_files)}")
print(f"First few files: {parquet_files[:5]}")

# 2. Read the first parquet file
first_file = os.path.join(chunk_dir, parquet_files[0])
print(f"\n{'='*80}")
print(f"Reading: {first_file}")
df = pd.read_parquet(first_file)
print(f"Shape: {df.shape}")

# 3. Print ALL column names and the Python type of the first value
print(f"\n{'='*80}")
print("ALL COLUMNS, first value type, and first value shape/len:")
print(f"{'='*80}")
for col in df.columns:
    val = df[col].iloc[0]
    type_name = type(val).__module__ + "." + type(val).__name__
    extra = ""
    if isinstance(val, np.ndarray):
        extra = f"  shape={val.shape} dtype={val.dtype}"
    elif isinstance(val, (list, tuple)):
        extra = f"  len={len(val)}"
    elif isinstance(val, (int, float, np.integer, np.floating)):
        extra = f"  value={val}"
    print(f"  {col:50s} -> {type_name}{extra}")

# 4. Action column: first 3 values, type, length
print(f"\n{'='*80}")
print("ACTION column - first 3 values:")
print(f"{'='*80}")
if "action" in df.columns:
    for i in range(min(3, len(df))):
        val = df["action"].iloc[i]
        type_name = type(val).__module__ + "." + type(val).__name__
        if isinstance(val, np.ndarray):
            print(f"  [{i}] type={type_name}, shape={val.shape}, dtype={val.dtype}, values={val}")
        elif isinstance(val, list):
            print(f"  [{i}] type={type_name}, len={len(val)}, values={val}")
        else:
            print(f"  [{i}] type={type_name}, value={val}")
else:
    print("  'action' column NOT found!")

# 5. observation.state columns: first 3 values, type, length
print(f"\n{'='*80}")
print("OBSERVATION.STATE columns - first 3 values:")
print(f"{'='*80}")
state_cols = [c for c in df.columns if "observation" in c.lower() and "state" in c.lower()]
if not state_cols:
    # Also check for just 'state' columns
    state_cols = [c for c in df.columns if "state" in c.lower()]
if state_cols:
    for col in state_cols:
        print(f"\n  Column: {col}")
        for i in range(min(3, len(df))):
            val = df[col].iloc[i]
            type_name = type(val).__module__ + "." + type(val).__name__
            if isinstance(val, np.ndarray):
                print(f"    [{i}] type={type_name}, shape={val.shape}, dtype={val.dtype}, values={val}")
            elif isinstance(val, list):
                print(f"    [{i}] type={type_name}, len={len(val)}, values={val}")
            else:
                print(f"    [{i}] type={type_name}, value={val}")
else:
    print("  No observation.state columns found!")

# 6. action_mean and action_std across ALL data in the file
print(f"\n{'='*80}")
print("ACTION STATISTICS across ALL data in this file:")
print(f"{'='*80}")
if "action" in df.columns:
    all_actions = np.stack(df["action"].values)
    print(f"  all_actions shape: {all_actions.shape}")
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    print(f"  action_mean ({len(action_mean)} values):")
    for i, v in enumerate(action_mean):
        print(f"    [{i:2d}] {v:.8f}")
    print(f"  action_std ({len(action_std)} values):")
    for i, v in enumerate(action_std):
        print(f"    [{i:2d}] {v:.8f}")
else:
    print("  'action' column NOT found!")

print(f"\n{'='*80}")
print("DONE")
