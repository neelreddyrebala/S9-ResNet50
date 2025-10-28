import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

from grad_cam import GradCAM


def apply_colormap_on_image(org_img, activation_map, alpha=0.5):
    """Enhanced colormap application with adjustable transparency"""
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_img = cv2.addWeighted(org_img, 1-alpha, heatmap, alpha, 0)
    return overlayed_img


def denormalize_image(img_tensor):
    """Convert normalized tensor to displayable numpy array"""
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = np.clip(
        (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]),
        0, 1
    )
    return np.uint8(img_np * 255)


def create_comparison_grid(misclassified, class_names, gradcam, output_path='gradcam_comparison.png'):
    """Create a cleaner side-by-side comparison"""
    num_samples = len(misclassified)
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols*2, figsize=(4*cols, 3*rows))
    fig.patch.set_facecolor('white')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img_tensor, true_label, pred_label) in enumerate(misclassified):
        cam, _ = gradcam.generate(img_tensor)
        gradcam.remove_hooks()
        
        # Re-register hooks for next iteration
        if i < len(misclassified) - 1:
            gradcam.fwd_hook = gradcam.target_layer.register_forward_hook(gradcam._save_activation)
            gradcam.bwd_hook = gradcam.target_layer.register_full_backward_hook(gradcam._save_gradient)
        
        img_disp = denormalize_image(img_tensor)
        overlay = apply_colormap_on_image(img_disp, cam, alpha=0.45)
        
        r = i // cols
        c = i % cols
        
        # Original image
        axes[r, c*2].imshow(img_disp)
        axes[r, c*2].set_title(f'Ground Truth:\n{class_names[true_label]}', 
                               fontsize=9, weight='bold', color='green')
        axes[r, c*2].axis('off')
        axes[r, c*2].add_patch(Rectangle((0, 0), 223, 223, linewidth=3, 
                                         edgecolor='green', facecolor='none'))
        
        # Grad-CAM overlay
        axes[r, c*2+1].imshow(overlay)
        axes[r, c*2+1].set_title(f'Prediction:\n{class_names[pred_label]}', 
                                 fontsize=9, weight='bold', color='red')
        axes[r, c*2+1].axis('off')
        axes[r, c*2+1].add_patch(Rectangle((0, 0), 223, 223, linewidth=3, 
                                           edgecolor='red', facecolor='none'))
    
    plt.suptitle('Misclassification Analysis with Grad-CAM', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.show()


def main(args):
    # Allow argparse.Namespace to be safely unpickled
    torch.serialization.add_safe_globals([argparse.Namespace])

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Create model
    print("Creating ResNet50 model...")
    model = models.resnet50(weights=None)
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model.eval()

    # Initialize GradCAM on layer4 (last conv block)
    print("Initializing GradCAM...")
    gradcam = GradCAM(model, model.layer4[-1])

    # Load dataset
    print(f"Loading validation dataset from {args.validation_img_path}")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(args.validation_img_path, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    class_names = val_dataset.classes

    # Find misclassified samples
    print(f"Finding up to {args.num_show} misclassified samples...")
    misclassified = []
    with torch.no_grad():
        for img, label in val_loader:
            output = model(img)
            pred = torch.argmax(output, 1).item()
            if pred != label.item():
                misclassified.append((img, label.item(), pred))
            if len(misclassified) >= args.num_show:
                break

    print(f"Found {len(misclassified)} misclassified samples.")

    if len(misclassified) == 0:
        print("No misclassified samples found!")
        return

    # Generate visualization
    print("Generating Grad-CAM visualizations...")
    create_comparison_grid(misclassified, class_names, gradcam, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM Visualization for Misclassified Images")
    parser.add_argument("--num_classes", type=int, required=True, 
                        help="Number of output classes")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--validation_img_path", type=str, required=True, 
                        help="Path to validation image folder")
    parser.add_argument("--num_show", type=int, default=20, 
                        help="Number of misclassified samples to visualize")
    parser.add_argument("--output_path", type=str, default="gradcam_comparison.png",
                        help="Output path for the visualization")
    args = parser.parse_args()

    main(args)
