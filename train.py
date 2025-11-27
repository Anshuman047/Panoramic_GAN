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
Panoramic GAN Training Script 
==============================
Training script for conditional GAN generating 256x128 panoramic images
"""

import os
import math
import argparse
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import shutil
import json
from pathlib import Path

from models import ImprovedGenerator, ImprovedDiscriminator, EMA, init_weights

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description="Panoramic GAN Training")
parser.add_argument("--root", type=str, default="/path/to/dataset", 
                    help="Path to dataset root directory")
parser.add_argument("--out", type=str, default="./outputs",
                    help="Output directory for samples and checkpoints")
parser.add_argument("--split_dataset", action="store_true",
                    help="Split dataset into train/val (first run only)")
parser.add_argument("--train_ratio", type=float, default=0.8,
                    help="Training set ratio (default: 0.8 for 80/20 split)")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for training")
parser.add_argument("--epochs", type=int, default=200,
                    help="Total number of training epochs")
parser.add_argument("--noise_dim", type=int, default=128,
                    help="Dimension of noise vector")
parser.add_argument("--n_classes", type=int, default=3,
                    help="Number of scene classes")
parser.add_argument("--img_h", type=int, default=128,
                    help="Image height")
parser.add_argument("--img_w", type=int, default=256,
                    help="Image width")
parser.add_argument("--lr_g", type=float, default=1e-5,
                    help="Generator learning rate")
parser.add_argument("--lr_d", type=float, default=2e-5,
                    help="Discriminator learning rate")
parser.add_argument("--beta1", type=float, default=0.0,
                    help="Adam beta1")
parser.add_argument("--beta2", type=float, default=0.9,
                    help="Adam beta2")
parser.add_argument("--ema_decay_warmup", type=float, default=0.995,
                    help="EMA decay during warmup")
parser.add_argument("--ema_decay_later", type=float, default=0.9999,
                    help="EMA decay after warmup")
parser.add_argument("--ema_switch_epoch", type=int, default=10,
                    help="Epoch to switch EMA decay")
parser.add_argument("--r1_gamma", type=float, default=15.0,
                    help="R1 gradient penalty weight")
parser.add_argument("--resume", type=str, default="",
                    help="Path to checkpoint to resume from")
parser.add_argument("--sample_interval", type=int, default=500,
                    help="Interval for saving samples during training")
parser.add_argument("--log_interval", type=int, default=100,
                    help="Interval for logging")
parser.add_argument("--checkpoint_interval", type=int, default=5,
                    help="Interval for saving checkpoints")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print("="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Dataset: {args.root}")
print(f"Output: {args.out}")
print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Image size: {args.img_h}×{args.img_w}")
print(f"LR (G/D): {args.lr_g}/{args.lr_d}")
print(f"R1 gamma: {args.r1_gamma}")
print(f"EMA decay: {args.ema_decay_warmup} → {args.ema_decay_later}")
print("="*60)

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ============================================================================
# DATASET SPLITTING FUNCTION
# ============================================================================

def split_dataset(root_dir, train_ratio=0.8, seed=42):
    """
    Split dataset into train and validation sets.
    Creates split_info.json with file lists for reproducibility.
    """
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    random.seed(seed)
    np.random.seed(seed)
    
    split_info = {
        'train': {},
        'val': {},
        'stats': {}
    }
    
    class_names = ['city', 'forest', 'rural']
    
    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        
        if not os.path.isdir(class_path):
            print(f"Warning: {class_path} not found, skipping")
            continue
        
        # Get all image files
        all_images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle
        random.shuffle(all_images)
        
        # Split
        n_train = int(len(all_images) * train_ratio)
        train_images = all_images[:n_train]
        val_images = all_images[n_train:]
        
        # Store in split_info
        split_info['train'][class_name] = train_images
        split_info['val'][class_name] = val_images
        split_info['stats'][class_name] = {
            'total': len(all_images),
            'train': len(train_images),
            'val': len(val_images)
        }
        
        print(f"\n{class_name.upper()}:")
        print(f"  Total: {len(all_images)}")
        print(f"  Train: {len(train_images)} ({len(train_images)/len(all_images)*100:.1f}%)")
        print(f"  Val:   {len(val_images)} ({len(val_images)/len(all_images)*100:.1f}%)")
    
    # Calculate totals
    total_train = sum(split_info['stats'][c]['train'] for c in class_names if c in split_info['stats'])
    total_val = sum(split_info['stats'][c]['val'] for c in class_names if c in split_info['stats'])
    total_all = total_train + total_val
    
    split_info['stats']['overall'] = {
        'total': total_all,
        'train': total_train,
        'val': total_val,
        'train_ratio': train_ratio,
        'seed': seed
    }
    
    print("\n" + "="*60)
    print(f"OVERALL SPLIT:")
    print(f"  Total: {total_all}")
    print(f"  Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"  Val:   {total_val} ({total_val/total_all*100:.1f}%)")
    print("="*60 + "\n")
    
    # Save split info
    split_file = os.path.join(root_dir, 'split_info.json')
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f" Split info saved to: {split_file}")
    print(" This ensures reproducible train/val splits across runs\n")
    
    return split_info

# ============================================================================
# DATASET CLASS WITH SPLIT 
# ============================================================================

class PanoramicDataset(Dataset):
    """Dataset loader with train/val split support"""
    
    def __init__(self, root_dir, transform, mode='train', split_info=None):
        """
        Args:
            root_dir: Root directory with class folders
            transform: Image transformations
            mode: 'train' or 'val'
            split_info: Dictionary with split information (from split_info.json)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.paths = []
        self.labels = []
        self.class_to_idx = {'city': 0, 'forest': 1, 'rural': 2}
        
        if split_info is None:
            # Try to load split_info.json
            split_file = os.path.join(root_dir, 'split_info.json')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    split_info = json.load(f)
                print(f"[Dataset] Loaded split info from {split_file}")
            else:
                raise FileNotFoundError(
                    f"split_info.json not found in {root_dir}. "
                    f"Run with --split_dataset flag first."
                )
        
        # Load images based on split
        for cname, idx in self.class_to_idx.items():
            if cname not in split_info[mode]:
                print(f"Warning: {cname} not in split_info[{mode}], skipping")
                continue
            
            image_files = split_info[mode][cname]
            folder = os.path.join(root_dir, cname)
            
            for f in image_files:
                self.paths.append(os.path.join(folder, f))
                self.labels.append(idx)
        
        print(f"[Dataset - {mode.upper()}] Loaded {len(self.paths)} images")
        
        # Print class distribution
        for cname, idx in self.class_to_idx.items():
            count = sum(1 for l in self.labels if l == idx)
            print(f"  - {cname}: {count} images")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = torch.tensor(self.labels[i], dtype=torch.long)
            return img, label
        except Exception as e:
            print(f"Error loading {self.paths[i]}: {e}")
            return self.__getitem__(random.randint(0, len(self.paths) - 1))

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

transform_train = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w), Image.LANCZOS),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ============================================================================
# HANDLE DATASET SPLITTING
# ============================================================================

split_info = None

if args.split_dataset:
    # First time: split the dataset
    split_info = split_dataset(args.root, train_ratio=args.train_ratio, seed=42)
else:
    # Check if split_info.json exists
    split_file = os.path.join(args.root, 'split_info.json')
    if not os.path.exists(split_file):
        print("\n" + "="*60)
        print("ERROR: split_info.json not found!")
        print("="*60)
        print("Please run with --split_dataset flag first:")
        print(f"  python train.py --root {args.root} --split_dataset")
        print("="*60 + "\n")
        exit(1)
    
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    print(f"[INFO] Loaded existing split from {split_file}")
    print(f"       Train: {split_info['stats']['overall']['train']} images")
    print(f"       Val:   {split_info['stats']['overall']['val']} images\n")

# ============================================================================
# CREATE DATASETS AND DATALOADERS
# ============================================================================

dataset_train = PanoramicDataset(args.root, transform_train, mode='train', split_info=split_info)
dataloader = DataLoader(
    dataset_train, 
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=2, 
    pin_memory=True, 
    drop_last=True
)

# ============================================================================
# MODEL INSTANTIATION
# ============================================================================

print("\nInitializing models...")
generator = ImprovedGenerator(args.noise_dim, args.n_classes, 3).to(device)
discriminator = ImprovedDiscriminator(args.n_classes, 3).to(device)

# Apply weight initialization
generator.apply(init_weights)
discriminator.apply(init_weights)

# EMA
ema = EMA(generator, decay=args.ema_decay_warmup)

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# ============================================================================
# OPTIMIZERS & SCHEDULERS
# ============================================================================

optimizer_G = optim.Adam(
    generator.parameters(), 
    lr=args.lr_g, 
    betas=(args.beta1, args.beta2), 
    eps=1e-8
)

optimizer_D = optim.Adam(
    discriminator.parameters(), 
    lr=args.lr_d, 
    betas=(args.beta1, args.beta2), 
    eps=1e-8
)

def get_lr_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs):
    """Cosine annealing with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler_G = get_lr_scheduler(optimizer_G)
scheduler_D = get_lr_scheduler(optimizer_D)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def hinge_d_loss(real_logits, fake_logits):
    """Hinge loss for discriminator"""
    return torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))

def hinge_g_loss(fake_logits):
    """Hinge loss for generator"""
    return -torch.mean(fake_logits)

def compute_r1_penalty(real_logits, real_imgs):
    """R1 gradient penalty"""
    grad_real = torch.autograd.grad(
        outputs=real_logits.sum(), 
        inputs=real_imgs,
        create_graph=True, 
        retain_graph=True
    )[0]
    grad_penalty = grad_real.view(grad_real.size(0), -1).pow(2).sum(1).mean()
    return grad_penalty

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

@torch.no_grad()
def save_samples(gen_model, epoch, per_class=4):
    """Save sample images during training"""
    gen_model.eval()
    total = args.n_classes * per_class
    z = torch.randn(total, args.noise_dim, device=device)
    labels = torch.tensor(
        [i for i in range(args.n_classes) for _ in range(per_class)], 
        device=device
    )
    
    fake = gen_model(z, labels)
    grid = make_grid(fake, nrow=per_class, normalize=True, value_range=(-1, 1), pad_value=1)
    path = os.path.join(args.out, f"panorama_epoch_{epoch:03d}.png")
    save_image(grid, path)
    gen_model.train()

# Fixed noise for consistent visualization
fixed_noise = torch.randn(25, args.noise_dim, device=device)
fixed_labels = torch.randint(0, args.n_classes, (25,), device=device)

# ============================================================================
# RESUME FROM CHECKPOINT
# ============================================================================

start_epoch = 0
if args.resume and os.path.exists(args.resume):
    print(f"\n[INFO] Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    
    generator.load_state_dict(ckpt.get("generator", {}))
    discriminator.load_state_dict(ckpt.get("discriminator", {}))
    
    if "ema_generator" in ckpt:
        ema.ema_model.load_state_dict(ckpt["ema_generator"])
    
    optimizer_G.load_state_dict(ckpt.get("optimizer_G", {}))
    optimizer_D.load_state_dict(ckpt.get("optimizer_D", {}))
    
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"[INFO] Resumed from epoch {start_epoch}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

generator.train()
discriminator.train()

for epoch in range(start_epoch, args.epochs):
    # Update EMA decay
    if epoch >= args.ema_switch_epoch:
        ema.decay = args.ema_decay_later
    
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    
    for i, (real_imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        batch_size = real_imgs.size(0)
        
        # ================================================================
        # Train Discriminator
        # ================================================================
        noise = torch.randn(batch_size, args.noise_dim, device=device)
        fake_imgs = generator(noise, labels).detach()
        
        # Discriminator outputs
        real_logits = discriminator(real_imgs, labels)
        fake_logits = discriminator(fake_imgs, labels)
        
        # Hinge loss
        d_loss = hinge_d_loss(real_logits, fake_logits)
        
        # R1 gradient penalty
        if args.r1_gamma > 0:
            real_imgs.requires_grad_(True)
            real_logits_r1 = discriminator(real_imgs, labels)
            r1_penalty = compute_r1_penalty(real_logits_r1, real_imgs)
            d_loss = d_loss + args.r1_gamma * r1_penalty
            real_imgs.requires_grad_(False)
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # ================================================================
        # Train Generator
        # ================================================================
        noise = torch.randn(batch_size, args.noise_dim, device=device)
        fake_imgs = generator(noise, labels)
        fake_logits = discriminator(fake_imgs, labels)
        
        # Generator loss (hinge)
        g_loss = hinge_g_loss(fake_logits)
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        # Update EMA
        ema.update(generator)
        
        # Accumulate losses
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        
        # ================================================================
        # Logging & Sampling
        # ================================================================
        if i % args.log_interval == 0:
            print(f"[Epoch {epoch+1}/{args.epochs}] [Batch {i+1}/{len(dataloader)}] "
                  f"[D Loss: {d_loss.item():.4f}] [G Loss: {g_loss.item():.4f}]")
        
        if i % args.sample_interval == 0:
            with torch.no_grad():
                generator.eval()
                sample_imgs = generator(fixed_noise, fixed_labels)
                grid = make_grid(sample_imgs, nrow=5, normalize=True, value_range=(-1, 1))
                save_image(grid, os.path.join(args.out, f"sample_epoch{epoch+1}_batch{i}.png"))
                generator.train()
    
    # Update learning rate schedulers
    scheduler_G.step()
    scheduler_D.step()
    
    # End of epoch logging
    avg_g_loss = epoch_g_loss / len(dataloader)
    avg_d_loss = epoch_d_loss / len(dataloader)
    print(f"\n[Epoch {epoch+1} Complete] Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")
    print(f"Current LR - G: {scheduler_G.get_last_lr()[0]:.2e}, D: {scheduler_D.get_last_lr()[0]:.2e}\n")
    
    # Save samples from EMA model
    save_samples(ema.ema_model, epoch + 1)
    
    # Save checkpoint
    if (epoch + 1) % args.checkpoint_interval == 0:
        checkpoint = {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'ema_generator': ema.ema_model.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'args': args
        }
        checkpoint_path = os.path.join(args.out, "checkpoints", f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Save final models
torch.save(generator.state_dict(), os.path.join(args.out, "final_generator.pth"))
torch.save(ema.ema_model.state_dict(), os.path.join(args.out, "final_generator_ema.pth"))
torch.save(discriminator.state_dict(), os.path.join(args.out, "final_discriminator.pth"))

print(f"\nFinal models saved to: {args.out}")
print("  - final_generator.pth")
print("  - final_generator_ema.pth (recommended)")

print("  - final_discriminator.pth")
