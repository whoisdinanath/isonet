import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import argparse

# CONFIGURATION
DATASET_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich")
META_PATH = DATASET_DIR / "metadata.csv"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_dataset(train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, random_seed=42):
    """Split dataset into train/val/test sets."""
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        print(f"Warning: Ratios don't sum to 1.0 (sum={total:.3f}). Normalizing...")
        train_ratio = train_ratio / total
        val_ratio = val_ratio / total
        test_ratio = test_ratio / total
    
    if not META_PATH.exists():
        print(f"Error: Could not find {META_PATH}")
        sys.exit(1)

    print(f"Loading {META_PATH}...")
    df = pd.read_csv(META_PATH)
    
    if len(df) == 0:
        print("Error: Metadata file is empty!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DATASET SPLIT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Total samples:     {len(df)}")
    print(f"Train ratio:       {train_ratio:.1%}")
    print(f"Validation ratio:  {val_ratio:.1%}")
    print(f"Test ratio:        {test_ratio:.1%}")
    print(f"Random seed:       {random_seed}")
    print(f"{'='*60}\n")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=random_seed, shuffle=True
    )
    
    # Second split: train and validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=random_seed, shuffle=True
    )
    
    print(f"Split Results:")
    print(f"  Training:   {len(train_df):6d} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):6d} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):6d} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total:      {len(train_df) + len(val_df) + len(test_df):6d} samples")
    
    # Save splits
    train_path = DATASET_DIR / "train.csv"
    val_path = DATASET_DIR / "val.csv"
    test_path = DATASET_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Files saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split.py                              # Default: 70/15/15
  python split.py --train 0.8 --val 0.1 --test 0.1
  python split.py --seed 123
        """
    )
    
    parser.add_argument("--train", type=float, default=TRAIN_RATIO,
                        help=f"Training ratio (default: {TRAIN_RATIO})")
    parser.add_argument("--val", type=float, default=VAL_RATIO,
                        help=f"Validation ratio (default: {VAL_RATIO})")
    parser.add_argument("--test", type=float, default=TEST_RATIO,
                        help=f"Test ratio (default: {TEST_RATIO})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_dataset(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed
    )
