# ResNet-50 on Tiny-ImageNet (From Scratch) — EC2 Quickstart

Train **ResNet-50 from scratch** on **Tiny-ImageNet (200 classes)** using PyTorch + timm. This README walks you through launching an AWS EC2 GPU instance, setting up the environment, running a **1–2 epoch demo**, and collecting logs you can share.

## Contents
- What you’ll get
- Prereqs
- 1) Launch EC2 GPU
- 2) Connect & prepare Python
- 3) Clone this repo + checkout branch
- 4) Download Tiny-ImageNet & fix val/
- 5) Train ResNet-50 (from scratch)
- 6) Verify results & what to show
- 7) Troubleshooting
- 8) Cost hygiene
- Appendix: Example terminal session

---

## What you’ll get
- A working **EC2 GPU** box (A10G).
- A clean **Python venv** with CUDA-enabled PyTorch.
- **Tiny-ImageNet** downloaded and reorganized for `ImageFolder`.
- **ResNet-50 from scratch** (no pretrained) training for **1–2 epochs** with:
  - AutoAugment (ImageNet policy), ColorJitter, RandomErasing (Cutout)
  - Mixup + CutMix + label smoothing
  - Cosine LR schedule
- Per-epoch JSON logs and checkpoints for screenshots.

---

## Prereqs
- AWS account with **G/VT Spot quota**.
- SSH key pair (`.pem`) downloaded.
- You are a **collaborator** on this repo (so you can push branches directly).

---

## 1) Launch EC2 GPU

**Region:** ap-south-1 (Mumbai)  
**Instance type:** `g5.xlarge` (or `g5.2xlarge` if needed) on **Spot**  
**AMI:** **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)**  
**Storage:** 150–200 GB gp3  
**Network:** Public IP enabled; Security Group allows **SSH (22)** from your IP only  
**Tip:** In Spot settings, set **Interruption behavior = Stop** (optional)

Click **Launch** once the form is filled.

---

## 2) Connect & prepare Python

SSH in (replace path/IP):
```bash
chmod 400 tsai-llm-train.pem
ssh -i tsai-llm-train.pem ubuntu@<EC2_PUBLIC_DNS_OR_IP>
```

Sanity:
```bash
nvidia-smi
```

Install venv + pip (Ubuntu 24.04 DL AMI needs this once):
```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip
```

Create venv and install deps:
```bash
python3 -m venv ~/venvs/tiny
source ~/venvs/tiny/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio timm==0.9.16 tqdm
```

Verify CUDA:
```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## 3) Clone this repo + checkout branch

```bash
cd ~
git clone https://github.com/neelreddyrebala/S9-ResNet50.git
cd S9-ResNet50
git fetch origin
git checkout feat/tinyimagenet-training   # replace with your branch if different
git pull
```

Confirm the new scripts exist:
```bash
ls -l src/train_tinyimagenet.py scripts/reorg_tiny_val.py
```

---

## 4) Download Tiny-ImageNet & fix `val/`

Download into your home (no sudo):
```bash
mkdir -p ~/data && cd ~/data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip
```

Reorganize the `val/` split into class subfolders:
```bash
cd ~/S9-ResNet50
source ~/venvs/tiny/bin/activate
python scripts/reorg_tiny_val.py --root ~/data/tiny-imagenet-200
```

Quick check:
```bash
python - <<'PY'
from pathlib import Path
root=Path("/home/ubuntu/data/tiny-imagenet-200")
print("train classes:", len([d for d in (root/'train').iterdir() if d.is_dir()]))
print("val classes:", len([d for d in (root/'val').iterdir()   if d.is_dir()]))
PY
# Both should print 200
```

---

## 5) Train ResNet-50 (from scratch)

Run **1–2 epochs** with strong augmentations:
```bash
cd ~/S9-ResNet50
python src/train_tinyimagenet.py   --data-root ~/data/tiny-imagenet-200   --epochs 2   --batch-size 128   --lr 0.1   --workers 8   --fp16   --save runs/tiny-r50
```

If you hit OOM, try `--batch-size 64`.  
If dataloader errors, try `--workers 4`.

---

## 6) Verify results & what to show

Per-epoch JSON is printed to terminal and saved to:
```
runs/tiny-r50/log.jsonl
```

Examples of what you’ll see:
```json
{"epoch": 0, "train": {"loss": ...}, "val": {"loss": 4.94, "top1": 3.32, "top5": 13.51}, ...}
{"epoch": 1, "train": {"loss": ...}, "val": {"loss": 4.75, "top1": 5.24, "top5": 18.15}, ...}
```

Grab proof:
```bash
tail -n 5 runs/tiny-r50/log.jsonl
ls -lh runs/tiny-r50
```

What to report:
- **From scratch** (`pretrained=False`)
- Augs: **AutoAugment, ColorJitter, RandomErasing, Mixup, CutMix, label smoothing**
- **Cosine** LR schedule
- **Top-1/Top-5** after **1–2 epochs** (low is expected for 200-class Tiny-ImageNet with very few epochs)

---

## 7) Troubleshooting

**Permission denied on `/data`**  
Use `~/data` instead:
```bash
mkdir -p ~/data
```

**reorg script says `Not found: /data/.../val_annotations.txt`**  
Your script was using a hardcoded path. Use:
```bash
python scripts/reorg_tiny_val.py --root ~/data/tiny-imagenet-200
```

**Missing venv / pip**  
Install once on Ubuntu 24.04:
```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip
```

**`timm.loss.SoftTargetCrossEntropy` not found**  
Import explicitly and use torch.amp APIs:
```python
from torch.amp import GradScaler, autocast
from timm.loss import SoftTargetCrossEntropy
```

**CUDA OOM**  
Lower batch size:
```bash
--batch-size 64
```

**Slow dataloader**  
Lower workers:
```bash
--workers 4
```

---

## 8) Cost hygiene
- **Stop** the instance when not training (don’t Terminate if you want to keep data on EBS).
- Keep Spot; if capacity is low, try a different AZ or temporarily On-Demand.

---

## Appendix: Example terminal session

```bash
# Local
git clone https://github.com/neelreddyrebala/S9-ResNet50.git
cd S9-ResNet50

chmod 400 tsai-llm-train.pem
ssh -i "tsai-llm-train.pem" ubuntu@ec2-XX-XXX-XX-XXX.ap-south-1.compute.amazonaws.com

# On EC2
nvidia-smi

sudo apt update
sudo apt install -y python3.12-venv python3-pip

python3 -m venv ~/venvs/tiny
source ~/venvs/tiny/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio timm==0.9.16 tqdm

cd ~
git clone https://github.com/neelreddyrebala/S9-ResNet50.git
cd S9-ResNet50
git fetch origin
git checkout feat/tinyimagenet-training
git pull

mkdir -p ~/data && cd ~/data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip

cd ~/S9-ResNet50
python scripts/reorg_tiny_val.py --root ~/data/tiny-imagenet-200

python src/train_tinyimagenet.py   --data-root ~/data/tiny-imagenet-200   --epochs 2   --batch-size 128   --lr 0.1   --workers 8   --fp16   --save runs/tiny-r50

tail -n 5 runs/tiny-r50/log.jsonl
ls -lh runs/tiny-r50
```

