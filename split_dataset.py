
# split_dataset.py
# Splits raw Caltech-101 into train/, val/, test/ folder
# Usage:
#   python split_dataset.py --src "path/to/101_ObjectCategories"
#                           --dst "path/to/caltech_split"

import os
import shutil
import random
import argparse
from collections import defaultdict


def split_dataset(src_root, dst_root, train_ratio=0.70,
                  val_ratio=0.15, seed=42):
    """
    Splits each class folder into train/val/test subsets.

    Args:
        src_root    : path to 101_ObjectCategories
        dst_root    : where to create train/, val/, test/
        train_ratio : fraction for training   (default 0.70)
        val_ratio   : fraction for validation (default 0.15)
        seed        : random seed for reproducibility
    """
    random.seed(seed)

    # test_ratio is whatever is left
    test_ratio = 1.0 - train_ratio - val_ratio

    splits = ['train', 'val', 'test']

    # Create output split folders
    for split in splits:
        os.makedirs(os.path.join(dst_root, split), exist_ok=True)

    # Stats tracking
    total_train = total_val = total_test = 0
    class_count = 0

    # Loop over each class folder in source
    class_folders = sorted(os.listdir(src_root))

    for class_name in class_folders:
        class_src = os.path.join(src_root, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_src):
            continue

        # Get all image files in this class
        images = [
            f for f in os.listdir(class_src)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(images) == 0:
            continue

        # Shuffle images randomly
        random.shuffle(images)

        # Calculate split sizes
        n       = len(images)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # test gets the remainder
        n_test  = n - n_train - n_val

        # Handle edge case: very small classes
        if n_test < 1:
            n_test  = 1
            n_val   = max(1, n - n_train - n_test)
            n_train = n - n_val - n_test

        # Assign images to splits
        train_imgs = images[:n_train]
        val_imgs   = images[n_train : n_train + n_val]
        test_imgs  = images[n_train + n_val:]

        # Copy images into dst/split/class_name/
        for split_name, split_imgs in [
            ('train', train_imgs),
            ('val',   val_imgs),
            ('test',  test_imgs)
        ]:
            dst_class_dir = os.path.join(
                dst_root, split_name, class_name
            )
            os.makedirs(dst_class_dir, exist_ok=True)

            for img_file in split_imgs:
                src_path = os.path.join(class_src, img_file)
                dst_path = os.path.join(dst_class_dir, img_file)
                shutil.copy2(src_path, dst_path)

        total_train += len(train_imgs)
        total_val   += len(val_imgs)
        total_test  += len(test_imgs)
        class_count += 1

        print(f"  {class_name:<30} "
              f"total={n:>4} | "
              f"train={len(train_imgs):>3} | "
              f"val={len(val_imgs):>3} | "
              f"test={len(test_imgs):>3}")

    # Final summary
    print(f"\n{'='*55}")
    print(f"Split complete!")
    print(f"  Classes : {class_count}")
    print(f"  Train   : {total_train} images")
    print(f"  Val     : {total_val}   images")
    print(f"  Test    : {total_test}  images")
    print(f"  Total   : {total_train + total_val + total_test} images")
    print(f"\nSaved to: {dst_root}")
    print(f"  {dst_root}/train/")
    print(f"  {dst_root}/val/")
    print(f"  {dst_root}/test/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split Caltech-101 into train/val/test'
    )
    parser.add_argument(
        '--src',
        type=str,
        required=True,
        help='Path to 101_ObjectCategories folder'
    )
    parser.add_argument(
        '--dst',
        type=str,
        required=True,
        help='Where to save the split dataset'
    )
    args = parser.parse_args()

    split_dataset(src_root=args.src, dst_root=args.dst)