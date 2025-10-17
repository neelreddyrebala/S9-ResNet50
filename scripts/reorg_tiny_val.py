# scripts/reorg_tiny_val.py
import shutil, os
from pathlib import Path

def main(root="/data/tiny-imagenet-200"):
    root = Path(root)
    val_dir = root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    if not ann_path.exists():
        raise SystemExit(f"Not found: {ann_path}")

    lines = ann_path.read_text().strip().splitlines()
    for line in lines:
        img, cls, *_ = line.split('\t')
        cls_dir = val_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        src = images_dir / img
        dst = cls_dir / img
        if src.exists():
            shutil.move(str(src), str(dst))

    if images_dir.exists():
        shutil.rmtree(images_dir)
    print("Val reorganized to class subfolders.")

if __name__ == "__main__":
    main()
