import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# CONFIGURATION
# Make sure this points to the folder containing metadata.csv
DATASET_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich")
META_PATH = DATASET_DIR / "metadata.csv"

def split_dataset():
    if not META_PATH.exists():
        print(f"Error: Could not find {META_PATH}")
        sys.exit(1)

    print(f"Loading {META_PATH}...")
    df = pd.read_csv(META_PATH)
    
    # Check if empty
    if len(df) == 0:
        print("Error: Metadata file is empty!")
        sys.exit(1)

    print(f"Total samples found: {len(df)}")
    
    # Split 90% Train, 10% Validation
    # random_state=42 ensures the split is the same every time you run it
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    print(f"Training Samples:   {len(train_df)}")
    print(f"Validation Samples: {len(val_df)}")
    
    # Save split files
    train_path = DATASET_DIR / "train.csv"
    val_path = DATASET_DIR / "val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print("\nSuccess!")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")

if __name__ == "__main__":
    split_dataset()