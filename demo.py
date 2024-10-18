import torch
from models import vgg19
import gdown
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np
import os

# Check if the model file exists, if not, download it
model_path = "pretrained_models/model_qnrf.pth"
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = "https://drive.google.com/uc?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru"
    gdown.download(url, model_path, quiet=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
def load_model(model_path, device):
    model = vgg19()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

model = load_model(model_path, device)

# Prediction function
def predict(inp):
    # Ensure input is in RGB format
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inp = transform(inp).unsqueeze(0)
    inp = inp.to(device)
    
    with torch.no_grad():
        outputs, _ = model(inp)
    
    count = torch.sum(outputs).item()
    
    # Generate density map visualization
    vis_img = outputs[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return vis_img, f"Predicted Count: {int(count)}"

# Gradio interface
title = "Distribution Matching for Crowd Counting"
desc = "A demo of DM-Count, a NeurIPS 2020 paper by Wang et al. Outperforms the state-of-the-art methods by a " \
       "large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50. " \
       "This demo uses the QNRF trained model. Try it by uploading an image or clicking on an example " \
       "(could take up to 20s if running on CPU)."

examples = [
    ["example_images/3.png"],
    ["example_images/2.png"],
    ["example_images/1.png"],
]

iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy", label="Predicted Density Map"),
        gr.Text(label="Predicted Count")
    ],
    title=title,
    description=desc,
    examples=examples,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()