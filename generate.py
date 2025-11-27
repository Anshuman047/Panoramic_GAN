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
Image Generation Script
Generate panoramic images from trained models
"""

import torch
import argparse
import os
from torchvision.utils import save_image, make_grid
from models import ImprovedGenerator

def load_generator(checkpoint_path, device='cuda'):
    """Load trained generator"""
    
    # Validation
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if not checkpoint_path.endswith(('.pt', '.pth')):
        raise ValueError(f"Invalid model file format. Expected .pt or .pth, got: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    
    try:
        generator = ImprovedGenerator(
            noise_dim=128,
            class_dim=3,
            img_channels=3
        ).to(device)
        
        if checkpoint_path.endswith('.pt'):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'ema_generator' in ckpt:
                generator.load_state_dict(ckpt['ema_generator'])
                print("Loaded EMA generator")
            elif 'generator' in ckpt:
                generator.load_state_dict(ckpt['generator'])
                print("Loaded generator")
            else:
                raise KeyError("Checkpoint does not contain 'generator' or 'ema_generator' key")
        else:  # .pth file
            generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print("Loaded from state dict")
        
        generator.eval()
        return generator
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def generate_images(generator, num_images, class_idx, device='cuda'):
    """Generate images"""
    with torch.no_grad():
        noise = torch.randn(num_images, 128, device=device)
        labels = torch.full((num_images,), class_idx, dtype=torch.long, device=device)
        images = generator(noise, labels)
    return images

def main():
    parser = argparse.ArgumentParser(description='Generate panoramic images')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_images', type=int, default=25, help='Number of images')
    parser.add_argument('--class_type', type=str, default='city', choices=['city', 'forest', 'rural'])
    parser.add_argument('--output_dir', type=str, default='./generated')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Class mapping
    class_map = {'city': 0, 'forest': 1, 'rural': 2}
    class_idx = class_map[args.class_type]
    
    # Load model
    print(f"Loading model from {args.model}")
    generator = load_generator(args.model, args.device)
    
    # Generate
    print(f"Generating {args.num_images} {args.class_type} images...")
    images = generate_images(generator, args.num_images, class_idx, args.device)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    grid = make_grid(images, nrow=5, normalize=True, value_range=(-1, 1))
    output_path = os.path.join(args.output_dir, f'{args.class_type}_generated.png')
    save_image(grid, output_path)
    print(f"Saved to {output_path}")

if __name__ == '__main__':

    main()
