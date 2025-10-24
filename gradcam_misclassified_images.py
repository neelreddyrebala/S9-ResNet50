import argparse
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import GradCAM from gradcam

num_classes=200
checkpoint_path="/content/runs/tinyimagenet_resnet50/model_best.pth.tar"
validation_img_path="/content/tiny-imagenet-200/valid"
# Allow argparse.Namespace to be safely unpickled
torch.serialization.add_safe_globals([argparse.Namespace])

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Create model with original output size (1000 classes)
model = models.resnet50(weights=None)

# Load weights
model.load_state_dict(state_dict)

# Replace final layer to match TinyImageNet (200 classes)
model.fc = nn.Linear(model.fc.in_features, 200)

# Set to eval mode
model.eval()
# Load dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(validation_img_path, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
class_names = val_dataset.classes


misclassified = []

with torch.no_grad():
    for img, label in val_loader:
        output = model(img)
        pred = torch.argmax(output, 1).item()
        if pred != label.item():
            misclassified.append((img, label.item(), pred))
        if len(misclassified) >= 10:  # limit to 10 samples
            break

print(f"Found {len(misclassified)} misclassified samples.")


import matplotlib.pyplot as plt
import numpy as np
import cv2

def apply_colormap_on_image(org_img, activation_map):
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_img = cv2.addWeighted(org_img, 0.6, heatmap, 0.4, 0)
    return overlayed_img


gradcam = GradCAM(model, model.layer4[-1])

# For example, show first 5 misclassified
num_show = min(5, len(misclassified))
fig, axes = plt.subplots(2, num_show, figsize=(3*num_show, 6))

for i, (img_tensor, true_label, pred_label) in enumerate(misclassified[:num_show]):
    cam, _ = gradcam.generate(img_tensor)
    gradcam.remove_hooks()

    # Convert tensor to numpy image
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = np.clip(
        (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]),
        0, 1
    )
    img_disp = np.uint8(img_np * 255)
    overlay = apply_colormap_on_image(img_disp, cam)

    # Top row: Ground Truth
    axes[0, i].imshow(img_disp)
    axes[0, i].set_title(f"GT: {class_names[true_label]}", fontsize=10)
    axes[0, i].axis("off")

    # Bottom row: Grad-CAM Prediction
    axes[1, i].imshow(overlay)
    axes[1, i].set_title(f"PRED: {class_names[pred_label]}", fontsize=10)
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
