# Copyright 2025 Anshuman Sahoo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GAN Architecture for Panoramic Image Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ========================================
# Weight Initialization
# ========================================
def init_weights(m):
    """Initialize network weights"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0, 0.02)

# ========================================
# Residual Block
# ========================================
class ResidualBlock(nn.Module):
    """Lines 95-108 from notebook"""
    def __init__(self, channels, use_spectral_norm=False):
        super().__init__()
        norm_fn = spectral_norm if use_spectral_norm else lambda x: x
        self.conv1 = norm_fn(nn.Conv2d(channels, channels, 3, 1, 1))
        self.conv2 = norm_fn(nn.Conv2d(channels, channels, 3, 1, 1))
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)

# ========================================
# Generator Network
# ========================================
class ImprovedGenerator(nn.Module):
    """Lines 110-195 from notebook"""
    def __init__(self, noise_dim, class_dim, img_channels):
        super().__init__()
        self.label_emb = nn.Embedding(class_dim, 128)
        
        # Initial projection
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 128, 512 * 8 * 16),
            nn.ReLU(True)
        )
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        self.res1 = ResidualBlock(256)
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.res2 = ResidualBlock(128)
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.res3 = ResidualBlock(64)
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )
        self.res4 = ResidualBlock(32)
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(32, img_channels, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        lbl = self.label_emb(labels)
        x = torch.cat([noise, lbl], dim=1)
        
        out = self.fc(x).view(x.size(0), 512, 8, 16)
        
        out = self.layer1(out)
        out = self.res1(out)
        
        out = self.layer2(out)
        out = self.res2(out)
        
        out = self.layer3(out)
        out = self.res3(out)
        
        out = self.layer4(out)
        out = self.res4(out)
        
        return self.final(out)

# ========================================
# Discriminator Network
# ========================================
class ImprovedDiscriminator(nn.Module):
    """Lines 197-245 from notebook"""
    def __init__(self, class_dim, img_channels):
        super().__init__()
        
        # Progressive downsampling
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection discriminator
        self.embed = spectral_norm(nn.Embedding(class_dim, 512))
        self.linear = spectral_norm(nn.Linear(512, 1))

    def forward(self, x, labels):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        feats = self.global_pool(out).view(x.size(0), -1)
        
        # Projection
        proj = torch.sum(self.embed(labels) * feats, dim=1, keepdim=True)
        out = self.linear(feats)
        
        return out + proj

# ========================================
# EMA Helper
# ========================================
class EMA:
    def __init__(self, model, decay, noise_dim=128, class_dim=3, img_channels=3):
        self.decay = decay
        self.ema_model = ImprovedGenerator(noise_dim, class_dim, img_channels)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.copy_(ema_p * self.decay + (1. - self.decay) * p)

                
