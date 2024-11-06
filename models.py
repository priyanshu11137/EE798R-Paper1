import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['DMCount', 'vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.reg_layer(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

class DMCount(nn.Module):
    def __init__(self, backbone=None):
        super(DMCount, self).__init__()
        self.backbone = backbone if backbone is not None else vgg19()
        
        # Density layers for each scale
        self.density_layer_4x4 = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())
        self.density_layer_8x8 = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())
        self.density_layer_16x16 = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def _get_density_and_normed(self, x, output_size, density_layer):
        """Helper function to compute density and normalized density maps"""
        # Resize to target size
        x_resized = F.adaptive_avg_pool2d(x, output_size)
        
        # Get density map
        mu = density_layer(x_resized)
        
        # Normalize
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        
        return mu, mu_normed

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Generate density maps for each scale
        mu_4x4, mu_4x4_normed = self._get_density_and_normed(
            features, (4, 4), self.density_layer_4x4
        )
        
        mu_8x8, mu_8x8_normed = self._get_density_and_normed(
            features, (8, 8), self.density_layer_8x8
        )
        
        mu_16x16, mu_16x16_normed = self._get_density_and_normed(
            features, (16, 16), self.density_layer_16x16
        )
        
        # Return all scales and their normalized versions
        return (
            (mu_4x4, mu_4x4_normed),       # Small scale (4x4)
            (mu_8x8, mu_8x8_normed),       # Medium scale (8x8)
            (mu_16x16, mu_16x16_normed)    # Large scale (16x16)
        )
