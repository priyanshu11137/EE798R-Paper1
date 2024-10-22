import torch
from models import vgg19
import gdown
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt  # Add this import for displaying images

# Function to download model if not exists
def download_model_if_not_exists(model_path, url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(url, model_path, quiet=False)

# Function to load the model
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

# Prediction function
def predict(image_path, model, device):
    # Load image
    inp = Image.open(image_path).convert('RGB')
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inp_tensor = transform(inp).unsqueeze(0)
    inp_tensor = inp_tensor.to(device)
    
    with torch.no_grad():
        outputs, _ = model(inp_tensor)
    
    count = torch.sum(outputs).item()
    
    # Generate density map visualization
    vis_img = outputs[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return np.array(inp), vis_img, f"Predicted Count: {int(count)}"

def show_images(original_image, density_map, predicted_count):
    # Display the original image and the density map using matplotlib
    plt.figure(figsize=(10, 5))

    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')

    # Show density map
    plt.subplot(1, 2, 2)
    plt.imshow(density_map)
    plt.title(f'Density Map\n{predicted_count}')

    # Show the plots
    plt.show()

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Crowd Counting using DM-Count Model')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='pretrained_models/model_qnrf.pth', help='Path to the pre-trained model')
    parser.add_argument('--download_model', action='store_true', help='Download the model if it does not exist')
    args = parser.parse_args()

    # URL of the pre-trained model
    model_url = "https://drive.google.com/uc?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru"

    # Check if the model file exists, download if requested
    if args.download_model:
        download_model_if_not_exists(args.model_path, model_url)

    # Check for available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = load_model(args.model_path, device)

    # Perform prediction
    original_image, density_map, predicted_count = predict(args.img_path, model, device)

    # Display the images directly
    show_images(original_image, density_map, predicted_count)
