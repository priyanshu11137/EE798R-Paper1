import torch
from models import vgg19
import gdown
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
import argparse

def download_model_if_not_exists(model_path, url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(url, model_path, quiet=False)

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
    
    return np.array(inp), vis_img, int(count)

def save_results(original_image, density_map, count):
    # Convert original image from RGB to BGR for OpenCV
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Add text with count to density map
    height, width = density_map.shape[:2]
    text = f"Predicted Count: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 20
    cv2.putText(density_map, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Create a white border between images
    border_size = 10
    border_color = (255, 255, 255)
    
    # Ensure both images have the same height
    max_height = max(original_bgr.shape[0], density_map.shape[0])
    original_resized = cv2.resize(original_bgr, (int(original_bgr.shape[1] * max_height / original_bgr.shape[0]), max_height))
    density_resized = cv2.resize(density_map, (int(density_map.shape[1] * max_height / density_map.shape[0]), max_height))
    
    # Create combined image with border
    combined_width = original_resized.shape[1] + density_resized.shape[1] + border_size
    combined_height = max_height
    combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
    
    # Place images in combined image
    combined_image[:, :original_resized.shape[1]] = original_resized
    combined_image[:, -density_resized.shape[1]:] = density_resized
    
    # Save the combined image
    output_path = 'crowd_counting_result.jpg'
    cv2.imwrite(output_path, combined_image)
    print(f"Results saved as '{output_path}'")
    return output_path

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

    # Save the results
    save_results(original_image, density_map, predicted_count)
