# %cd /content

from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm
import json
import requests

# 1️⃣ Load dataset
print("🔽 Loading Tiny ImageNet from Hugging Face...")
dataset = load_dataset("zh-plus/tiny-imagenet")

# 2️⃣ Define output directories
output_root = "/content/tiny-imagenet-200"
os.makedirs(output_root, exist_ok=True)

# 3️⃣ Extract class names FIRST
print("\n📝 Extracting class names...")
train_data = dataset["train"]
label_feature = train_data.features["label"]

# Get WordNet IDs
wnids = [label_feature.int2str(i) for i in range(label_feature.num_classes)]
print(f"✅ Found {len(wnids)} classes")
print(f"First 5 WordNet IDs: {wnids[:5]}")

# Save WordNet IDs
wnids_path = os.path.join(output_root, "wnids.txt")
with open(wnids_path, "w") as f:
    for wnid in wnids:
        f.write(f"{wnid}\n")
print(f"✅ Saved WordNet IDs to {wnids_path}")

# 🔍 Try to get human-readable names from ImageNet metadata
print("\n🔍 Fetching human-readable class names...")
try:
    # Try to fetch from TinyImageNet words file (if available online)
    try:
        url = "https://raw.githubusercontent.com/seshuad/IMagenet/master/tiny-imagenet-200/words.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            for line in response.text.split('\n'):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wordnet_mapping[parts[0]] = parts[1]
            print(f"✅ Downloaded {len(wordnet_mapping)} class descriptions")
    except:
        print("⚠️ Could not download words.txt, using partial mapping")
    
    # Create human-readable class names
    class_names = []
    for wnid in wnids:
        readable_name = wordnet_mapping.get(wnid, wnid)
        class_names.append(readable_name)
    
    print(f"First 5 readable names: {class_names[:5]}")
    
except Exception as e:
    print(f"⚠️ Could not fetch readable names: {e}")
    print("Using WordNet IDs instead")
    class_names = wnids

# Save human-readable class names to file (for Gradio app)
class_names_path = os.path.join(output_root, "class_names.txt")
with open(class_names_path, "w") as f:
    for name in class_names:
        f.write(f"{name}\n")
print(f"✅ Saved class names to {class_names_path}")

# Also save as JSON with indices
class_mapping_path = os.path.join(output_root, "class_mapping.json")
class_mapping = {i: name for i, name in enumerate(class_names)}
with open(class_mapping_path, "w") as f:
    json.dump(class_mapping, f, indent=2)
print(f"✅ Saved class mapping to {class_mapping_path}")

# Save both wnid and readable name
full_mapping_path = os.path.join(output_root, "wnid_to_name.json")
full_mapping = {wnid: name for wnid, name in zip(wnids, class_names)}
with open(full_mapping_path, "w") as f:
    json.dump(full_mapping, f, indent=2)
print(f"✅ Saved full mapping to {full_mapping_path}")

# 4️⃣ Function to save split
def save_split(split_name):
    print(f"\n📦 Processing {split_name} split...")
    split_dir = os.path.join(output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    split_data = dataset[split_name]

    # Loop through dataset
    for idx, example in enumerate(tqdm(split_data, total=len(split_data))):
        img = example["image"]
        label = example["label"]
        class_name = split_data.features["label"].int2str(label)

        # Create class directory
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Convert to RGB (fix grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Generate unique filename
        img_path = os.path.join(class_dir, f"{split_name}_{idx}.jpg")

        # Save image
        img.save(img_path, "JPEG")

    print(f"✅ Saved {split_name} split to {split_dir}")

# 5️⃣ Process both splits
save_split("train")
save_split("valid")

# 6️⃣ Quick summary
train_count = sum(len(files) for _, _, files in os.walk(os.path.join(output_root, "train")))
val_count = sum(len(files) for _, _, files in os.walk(os.path.join(output_root, "valid")))
print(f"\n✅ Done! Train images: {train_count}, Val images: {val_count}")
print(f"📁 Class names saved to: {class_names_path}")
print(f"📁 Class mapping saved to: {class_mapping_path}")
