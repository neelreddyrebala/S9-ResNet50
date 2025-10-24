import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import json
import requests
from io import BytesIO

# ---------- Configuration ----------
MODEL_NAME = "resnet50"
NUM_CLASSES = 200
CHECKPOINT_PATH = "./runs/tinyimagenet_resnet50/model_best.pth.tar"

# ---------- Load ImageNet Class Labels ----------
def load_class_labels():
    """Load TinyImageNet class labels"""
    label_files = [
        "./tiny-imagenet-200/class_names.txt",
        "./tiny-imagenet-200/class_mapping.json",
        "./tiny-imagenet-200/wnids.txt",
    ]
    
    for label_file in label_files:
        try:
            if label_file.endswith(".json"):
                with open(label_file, "r") as f:
                    mapping = json.load(f)
                    return [mapping[str(i)] for i in range(len(mapping))]
            else:
                with open(label_file, "r") as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]
                    if labels:
                        print(f"✅ Loaded {len(labels)} class labels from {label_file}")
                        return labels
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ Error loading {label_file}: {e}")
            continue
    
    print("⚠️ No label files found. Using generic class names.")
    return [f"Class_{i:03d}" for i in range(NUM_CLASSES)]

class_names = load_class_labels()
print(f"📊 Loaded {len(class_names)} classes. Examples: {class_names[:3]}")

# ---------- Load Model ----------
@torch.no_grad()
def load_model():
    """Load the trained model"""
    try:
        model = models.resnet50(weights=None)
        
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            print(f"⚠️ Checkpoint not found. Using pretrained ImageNet weights.")
            model = models.resnet50(weights="IMAGENET1K_V2")
        
        if NUM_CLASSES != 1000:
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.eval()
        return model

model = load_model()

# ---------- Image Preprocessing ----------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- Grad-CAM Implementation ----------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class
    
    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

# ---------- Visualization ----------
def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on original image"""
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_img = cv2.addWeighted(org_img, 0.6, heatmap, 0.4, 0)
    return overlayed_img

# ---------- Main Prediction Function ----------
def predict(image, use_gradcam=True):
    """Main prediction function"""
    if image is None:
        return None, None, "Please upload an image", None
    
    try:
        # Store original image for display
        original_img = np.array(image)
        
        # Preprocess
        img_tensor = preprocess(image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Prepare results
        top_prediction = f"{class_names[top5_idx[0]]}"
        confidence = f"{top5_prob[0]*100:.2f}%"
        
        top5_results = {
            class_names[top5_idx[i]]: float(top5_prob[i]*100)
            for i in range(5)
        }
        
        # Generate Grad-CAM
        gradcam_img = None
        if use_gradcam:
            gradcam = GradCAM(model, model.layer4[-1])
            cam, pred_class = gradcam.generate(img_tensor)
            gradcam.remove_hooks()
            
            img_array = np.array(image.resize((224, 224)))
            gradcam_img = apply_colormap_on_image(img_array, cam)
        
        return original_img, gradcam_img, top_prediction, confidence, top5_results
        
    except Exception as e:
        return None, None, "Error", str(e), None

# ---------- Gradio Interface ----------
custom_css = """
.main-header {
    text-align: center;
    margin-bottom: 2rem;
}
.results-panel {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.prediction-box {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1976d2;
    text-align: center;
    padding: 1rem;
    background: #e3f2fd;
    border-radius: 8px;
    margin: 1rem 0;
}
.confidence-box {
    font-size: 1.2rem;
    color: #2e7d32;
    text-align: center;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="ImageNet Classifier") as demo:
    
    # Header
    gr.Markdown(
        """
        # Deep learning image classifier
        ### ResNet50 model with gradient-weighted class activation mapping
        """
    )
    
    with gr.Tabs():
        # Main Classification Tab
        with gr.Tab("Classifier"):
            with gr.Row():
                # Left Column - Input Section
                with gr.Column(scale=1):
                    gr.Markdown("### Upload image")
                    
                    image_input = gr.Image(
                        type="pil", 
                        label="Select an image",
                        height=400
                    )
                    
                    gr.Markdown("### Example images")
                    example_images = gr.Examples(
                        examples=[
                            ["https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=400"],
                            ["https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"],
                            ["https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=400"],
                            ["https://images.unsplash.com/photo-1546527868-ccb7ee7dfa6a?w=400"],
                            ["https://images.unsplash.com/photo-1552053831-71594a27632d?w=400"],
                            ["https://images.unsplash.com/photo-1470093851219-69951fcbb533?w=400"],
                        ],
                        inputs=image_input,
                        label="Click to use",
                        examples_per_page=6
                    )
                    
                    with gr.Row():
                        predict_btn = gr.Button(
                            "Analyze image", 
                            variant="primary", 
                            size="lg",
                            scale=2
                        )
                        clear_btn = gr.ClearButton(
                            components=[image_input], 
                            value="Clear",
                            size="lg",
                            scale=1
                        )
                
                # Right Column - Results Section
                with gr.Column(scale=1):
                    gr.Markdown("### Analysis results")
                    
                    # Original and Grad-CAM side by side
                    with gr.Row():
                        original_output = gr.Image(label="Original image", height=250)
                        gradcam_output = gr.Image(label="Grad-CAM visualization", height=250)
                    
                    # Prediction results
                    with gr.Group(elem_classes="results-panel"):
                        gr.Markdown("**Primary classification**")
                        prediction_output = gr.Textbox(
                            label="Predicted class",
                            lines=1,
                            show_label=False,
                            elem_classes="prediction-box"
                        )
                        confidence_output = gr.Textbox(
                            label="Confidence",
                            lines=1,
                            show_label=False,
                            elem_classes="confidence-box"
                        )
                        
                        gr.Markdown("**Top 5 predictions**")
                        top5_output = gr.Label(
                            label="Confidence distribution",
                            num_top_classes=5
                        )
        
        # About Tab
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## Model information
                
                This application uses a deep convolutional neural network for image classification. The model has been trained to recognize 200 different object categories from the TinyImageNet dataset.
                
                **Architecture:** ResNet50 (deep residual network)  
                **Training dataset:** TinyImageNet  
                **Number of classes:** 200  
                **Input resolution:** 224×224 pixels  
                **Framework:** PyTorch
                
                ---
                
                ## About Grad-CAM
                
                Gradient-weighted class activation mapping (Grad-CAM) is a visualization technique that provides visual explanations for decisions from convolutional neural network models. The heatmap highlights regions of the input image that were most influential in the model's classification decision.
                
                **Interpretation guide:**
                - Warmer colors (red, yellow) indicate regions with higher importance
                - Cooler colors (blue, purple) indicate regions with lower importance
                - The visualization helps understand what features the model focuses on when making predictions
                
                ---
                
                ## Technical details
                
                **Preprocessing:**
                - Images are resized to 256×256 pixels
                - Center cropped to 224×224 pixels
                - Normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                **Inference:**
                - Real-time classification with CPU/GPU support
                - Softmax activation for probability distribution
                - Top-5 predictions displayed with confidence scores
                
                **Grad-CAM implementation:**
                - Applied to the final convolutional layer (layer4)
                - Uses gradient information to weight feature maps
                - Produces class-discriminative localization maps
                - Overlaid on original image with 60/40 weighting
                
                **Model checkpoint:**
                - Location: `/content/runs/tinyimagenet_resnet50/model_best.pth.tar`
                - Fallback: Pretrained ImageNet weights if checkpoint unavailable
                
                ---
                
                ## Usage instructions
                
                1. Upload an image using the file picker or select from example images
                2. Click the "Analyze image" button to run classification
                3. View the original image alongside the Grad-CAM heatmap
                4. Check the predicted class with confidence score
                5. Review the top 5 predictions to see alternative classifications
                
                For best results, use clear images with the main subject centered and well-lit.
                """
            )
    
    # Event handlers
    predict_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[original_output, gradcam_output, prediction_output, confidence_output, top5_output]
    )

# ---------- Launch ----------
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0"
    )
