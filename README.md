# ResNet-50 Training on ImageNet-1k Using AWS EC2
Goal is to train a ResNet-50 model from scratch on the ImageNet-1k (ILSVRC 2012) dataset using PyTorch and AWS EC2 GPU instance, and the target is to achieve 75% top-1 accuracy

Before training ResNet-50 on the full ImageNet-1K dataset (which contains over 1 million training images), we first trained **ResNet-50 from scratch** on **Tiny-ImageNet (200 classes)** using PyTorch and timm. This initial step helped us validate our training pipeline. This README adds a **clean S3 workflow** (backup/restore) and a **scalable plan for ImageNet‑1k**, plus **fast I/O tips** so big dataset transfers don’t crawl.

---

## What you’ll get
- A working **EC2 GPU** environment.
- A clean **Python venv** with CUDA‑enabled PyTorch.
- **Tiny‑ImageNet** downloaded and reorganized for `ImageFolder`.
- **ResNet‑50 (from scratch)** training with:
  - AutoAugment, ColorJitter, RandomErasing (Cutout)
  - Mixup, CutMix, label smoothing
  - Cosine LR schedule
- **Per‑epoch JSONL logs** (`runs/<exp>/log.jsonl`) + **checkpoints**.
- **S3-first workflow** for runs & datasets (cheap, reliable).
- **ImageNet‑1k pipeline outline** (dataset on S3, restore locally each run).

> **Your bucket (example used below):** `resnet50` (change if different)
>
> **Your recent run path on S3:** `s3://resnet50/tinyimagenet/runs/tiny-r50/`

---

## Prereqs
- AWS account with EC2 quota for GPU (Spot recommended).
- IAM Role on the instance with S3 access (recommended) **or** AWS CLI configured with access keys.
- SSH key pair (`.pem`) downloaded.

---

## 1) Launch EC2 GPU
**Region:** ap-south-1 (Mumbai)  
**Instance type:** `g5.xlarge` (or larger) — Spot preferred  
**AMI:** *Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 24.04)*  
**Storage:** 150–200 GB **gp3** (adjust as needed)  
**Network:** Security Group allows **SSH (22)** from your IP only  
**Spot tip:** Use **persistent** request and set **Interruption behavior = Stop** (so you can Stop/Start).

---

## 2) Connect & prepare Python
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_DNS_OR_IP>

nvidia-smi

# Ubuntu 24.04 DL AMI: install venv + pip once
sudo apt update
sudo apt install -y python3.12-venv python3-pip

python3 -m venv ~/venvs/tiny
source ~/venvs/tiny/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio timm==0.9.16 tqdm
```

Quick CUDA check:
```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## 3) Clone this repo
```bash
cd ~
git clone https://github.com/neelreddyrebala/S9-ResNet50.git
cd S9-ResNet50
# (optional) git checkout the branch that contains tiny imagenet scripts
ls -l src/train_tinyimagenet.py scripts/reorg_tiny_val.py
```

---

## 4) Download Tiny‑ImageNet & fix `val/`
```bash
mkdir -p ~/data && cd ~/data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip

cd ~/S9-ResNet50
source ~/venvs/tiny/bin/activate
python scripts/reorg_tiny_val.py --root ~/data/tiny-imagenet-200

# quick sanity
python - <<'PY'
from pathlib import Path
root=Path("/home/ubuntu/data/tiny-imagenet-200")
print("train classes:", len([d for d in (root/'train').iterdir() if d.is_dir()]))
print("val classes:", len([d for d in (root/'val').iterdir()   if d.is_dir()]))
PY
# both should be 200
```

---

## 5) Train ResNet‑50 (from scratch)
```bash
cd ~/S9-ResNet50
python src/train_tinyimagenet.py \
  --data-root ~/data/tiny-imagenet-200 \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.1 \
  --workers 8 \
  --fp16 \
  --save runs/tiny-r50
```
> If OOM: try `--batch-size 64`. If dataloader slow: reduce `--workers` to 4.

**Output:** per‑epoch JSON lines to console **and** `runs/tiny-r50/log.jsonl`. Checkpoints: `runs/tiny-r50/epochXXX.pt`.

---

## 6) Make a readable metrics CSV (from JSONL)
```bash
cd ~/S9-ResNet50
source ~/venvs/tiny/bin/activate
python - <<'PY'
import json, csv, pathlib
log = pathlib.Path("runs/tiny-r50/log.jsonl")
out = pathlib.Path("runs/tiny-r50/metrics.csv")
by_epoch={}
with log.open() as f:
    for line in f:
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        by_epoch[r["epoch"]]=r  # keep last occurrence per epoch
with out.open("w", newline="") as f:
    w=csv.writer(f); w.writerow(["epoch","train_loss","val_loss","top1","top5","time_sec"])
    for e in sorted(by_epoch):
        r=by_epoch[e]
        w.writerow([e, r["train"]["loss"], r["val"]["loss"], r["val"]["top1"], r["val"]["top5"], r["time_sec"]])
print(f"Wrote {out}")
PY
```

> **Note on train accuracy:** with Mixup/CutMix + label smoothing, “train accuracy” is not meaningful; we track **train loss** and rely on **validation accuracy** as the reliable signal.

---

## 7) Add S3 (cheap, reliable storage)

### 7.1 Attach IAM Role (recommended)
- IAM → Roles → **Create role** → Trusted entity: **EC2** → Attach `AmazonS3FullAccess` (or tighter custom policy) → Name: `EC2_S3Access_Role`.
- EC2 → Instances → select → **Actions → Security → Modify IAM role** → choose `EC2_S3Access_Role`.

Test:
```bash
aws sts get-caller-identity
```

### 7.2 Use your bucket
```bash
export BUCKET=resnet50   # <- change if yours differs
```

### 7.3 Upload runs + (optional) dataset
```bash
# runs: logs + checkpoints
aws s3 sync ~/S9-ResNet50/runs/tiny-r50 s3://$BUCKET/tinyimagenet/runs/tiny-r50

# dataset (optional, ~1GB)
aws s3 sync ~/data/tiny-imagenet-200 s3://$BUCKET/tinyimagenet/dataset

# verify
aws s3 ls s3://$BUCKET/tinyimagenet/runs/tiny-r50/ --recursive | head
```

### 7.4 Enable versioning & lifecycle (cost hygiene)
```bash
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled
# Add a lifecycle rule in S3 console to move old checkpoints to Standard-IA or Glacier after N days.
```

---

## 8) Safe shutdown (Spot one-time vs persistent)
- **One-time Spot:** you **cannot Stop**; backup to S3, optionally create an **EBS snapshot**, then **Terminate**.
- **Persistent Spot (recommended next time):** set **Interruption behavior = Stop** so you can Stop/Start like On-Demand.

EBS snapshot (console): EC2 → Volumes → your volume → **Create snapshot**.

Terminate (console): EC2 → Instances → select → **Instance state → Terminate**.

---

## 9) Restore on a new instance
```bash
# repo + env
git clone https://github.com/neelreddyrebala/S9-ResNet50.git
cd S9-ResNet50
python3 -m venv ~/venvs/tiny && source ~/venvs/tiny/bin/activate
pip install --upgrade pip && pip install torch torchvision torchaudio timm tqdm

# data + runs from S3
export BUCKET=resnet50
mkdir -p ~/data
aws s3 sync s3://$BUCKET/tinyimagenet/dataset ~/data/tiny-imagenet-200
aws s3 sync s3://$BUCKET/tinyimagenet/runs/tiny-r50 ./runs/tiny-r50
```

---

## 10) ImageNet‑1k (scalable S3-first pipeline)

**Layout in S3**
```
s3://$BUCKET/imagenet-1k/
  ├── train/  (1000 classes)
  └── val/    (1000 classes)
```

**Per run on EC2 (fast path)**
```bash
export BUCKET=resnet50
export DATA_ROOT=/mnt/imagenet      # local fast disk
sudo mkdir -p $DATA_ROOT && sudo chown -R $USER:$USER $DATA_ROOT

# pull only once per machine; subsequent syncs are incremental
aws s3 sync s3://$BUCKET/imagenet-1k/train $DATA_ROOT/train --no-progress
aws s3 sync s3://$BUCKET/imagenet-1k/val   $DATA_ROOT/val   --no-progress

# train (example)
python src/train_imagenet.py \
  --data-root $DATA_ROOT \
  --epochs 90 --batch-size 256 --lr 0.1 --workers 8 --fp16 \
  --mixup 0.2 --cutmix 1.0 --label-smoothing 0.1 \
  --autoaugment --colorjitter --random-erasing \
  --sched cosine --warmup-epochs 5 \
  --save runs/imagenet-r50

# backup
aws s3 sync runs/imagenet-r50 s3://$BUCKET/imagenet-1k/runs/imagenet-r50
```

> **Note:** Obtaining ImageNet‑1k requires authorization. Once you have `train/` and `val/` locally arranged as class‑folders, push them to S3 once, then re‑use S3 for all future runs.

---

## 11) Fast I/O tips (Tiny‑ImageNet felt slow → do these for ImageNet‑1k)

**A. Use a fast local path for training**  
Always **sync from S3 to local disk** (e.g., `/mnt/imagenet`) and train **from local**. S3 → RAM during training is slow.

**B. Speed up S3 transfers**
- Increase AWS CLI concurrency: create/append `~/.aws/config`
  ```
  [default]
  s3 =
      max_concurrent_requests = 50
      multipart_threshold = 64MB
      multipart_chunksize = 64MB
  ```
  (Tune up to 100/128MB if network is strong.)
- Use `--no-progress` to reduce overhead; use `--size-only` for faster re‑syncs when you trust sizes.
- Consider **s5cmd** (parallel S3 copy tool) for very large sync jobs:
  ```bash
  # install
  wget https://github.com/peak/s5cmd/releases/latest/download/s5cmd_*.tar.gz -O s5cmd.tgz
  tar -xzf s5cmd.tgz && sudo mv s5cmd /usr/local/bin/
  # example: pull train quickly with massive parallelism
  s5cmd --numworkers 128 cp "s3://$BUCKET/imagenet-1k/train/*" "$DATA_ROOT/train/"
  ```

**C. Tune EBS throughput if using EBS-heavy workflows**  
- For **gp3**, you can provision higher **throughput (MB/s)** and **IOPS** temporarily during big copies (costs extra; scale back after).

**D. Speed up HTTP downloads (first-ever fetch from the internet)**  
- If you’re pulling from plain HTTP (like Tiny‑ImageNet): use **aria2c** (multi‑connection):
  ```bash
  sudo apt-get install -y aria2
  aria2c -x 16 -s 16 -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
  ```

**E. Avoid tiny files bottlenecks**  
- If you control the dataset packaging, compress many small files into **shards** (e.g., `.tar` files per class) and use **WebDataset / tfrecords**. This massively speeds up I/O and startup time.

**F. Keep logs & checkpoints small**  
- Save **best** and **last** only; prune old epochs or move old ones to Infrequent Access via lifecycle.


---

## Appendix: Example terminal session (abridged)
```bash
# backup
export BUCKET=resnet50
aws s3 sync ~/S9-ResNet50/runs/tiny-r50 s3://$BUCKET/tinyimagenet/runs/tiny-r50
aws s3 sync ~/data/tiny-imagenet-200 s3://$BUCKET/tinyimagenet/dataset

# restore
aws s3 sync s3://$BUCKET/tinyimagenet/dataset ~/data/tiny-imagenet-200
aws s3 sync s3://$BUCKET/tinyimagenet/runs/tiny-r50 ./runs/tiny-r50
```

## Next Steps: Scaling to ImageNet-1K
- Once the pipeline is validated on Tiny-ImageNet, we follow a similar approach for ImageNet-1K.
- EBS Data synced with Amazon S3, and read from there.
- Training will be scaled for longer epochs with checkpointing and monitoring.


## Collaborators

- Neelreddy Rebala <neelreddy.rebala@gmail.com>
- Jayant Guru Shrivastava <jayantgurushrivastava@gmail.com>
- Vikas <vikasjhanitk@gmail.com>
- Divya Kamat <Divya.r.kamat@gmail.com>
