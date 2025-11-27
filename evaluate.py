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
Academic Evaluation for Panoramic GANs with Train/Val Split
==============================================

Implements validated metrics with intellectual rigor:
- FID (Heusel et al. NIPS 2017) 
- IS (Salimans et al. NIPS 2016) 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import json

from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.models import inception_v3
from PIL import Image
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
from scipy import linalg
from skimage.metrics import structural_similarity as ssim

from models import ImprovedGenerator

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    model_paths: Dict[str, str]
    dataset_path: str
    output_dir: str = "./academic_evaluation"
    use_validation: bool = True  
    
    # Evaluation parameters
    num_samples: int = 1000  
    batch_size: int = 32
    is_splits: int = 10
    
    # Model parameters
    noise_dim: int = 128
    n_classes: int = 3
    img_height: int = 128
    img_width: int = 256
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

# =============================================================================
# VALIDATED GAN METRICS 
# =============================================================================

class ValidatedGANMetrics:
    """FID and IS calculation with proper validation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.inception_model = self._load_inception()
        self.inception_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _load_inception(self):
        """Load Inception V3 for FID/IS calculation"""
        model = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        model.eval()
        model.to(self.device)
        
        self.fid_features = []
        def save_fid_features(module, input, output):
            pooled = F.adaptive_avg_pool2d(output, (1, 1))
            self.fid_features.append(pooled.view(output.size(0), -1).cpu().detach())
        
        model.avgpool.register_forward_hook(save_fid_features)
        return model
    
    def calculate_fid(self, real_images: List[torch.Tensor], 
                     fake_images: List[torch.Tensor]) -> Tuple[float, Dict]:
        """
        Calculate FID (Fréchet Inception Distance)
        Reference: Heusel et al. "GANs Trained by a Two Time-Scale Update Rule 
                   Converge to a Local Nash Equilibrium" NIPS 2017
        """
        print("Calculating FID (Heusel et al. NIPS 2017)...")
        
        real_features = self._extract_features(real_images)
        fake_features = self._extract_features(fake_images)
        
        fid_score = self._calculate_frechet_distance(real_features, fake_features)
        
        stats = {
            'fid_score': float(fid_score),
            'n_real_samples': len(real_images),
            'n_fake_samples': len(fake_images),
            'note': f'Sample size {len(real_images)} '
        }
        
        print(f"   FID: {fid_score:.2f}")
        return fid_score, stats
    
    def calculate_inception_score(self, images: List[torch.Tensor]) -> Tuple[float, float, Dict]:
        """
        Calculate Inception Score
        Reference: Salimans et al. "Improved Techniques for Training GANs" NIPS 2016
        """
        print("Calculating Inception Score (Salimans et al. NIPS 2016)...")
        
        predictions = self._get_predictions(images)
        
        scores = []
        n_splits = self.config.is_splits
        split_size = len(predictions) // n_splits
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else len(predictions)
            part = predictions[start_idx:end_idx]
            
            # Calculate KL divergence
            p_y = np.mean(part, axis=0) + 1e-16
            p_y = p_y / np.sum(p_y)
            
            kl_divs = []
            for pred in part:
                pred = pred + 1e-16
                pred = pred / np.sum(pred)
                kl_div = np.sum(pred * (np.log(pred) - np.log(p_y)))
                kl_divs.append(kl_div)
            
            scores.append(np.exp(np.mean(kl_divs)))
        
        mean_is = np.mean(scores)
        std_is = np.std(scores)
        
        stats = {
            'split_scores': scores,
            'n_splits': n_splits,
            'n_samples': len(images)
        }
        
        print(f"   IS: {mean_is:.3f} ± {std_is:.3f}")
        return mean_is, std_is, stats
    
    def _extract_features(self, images: List[torch.Tensor]) -> np.ndarray:
        """Extract Inception features for FID"""
        self.fid_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.config.batch_size),
                         desc="Extracting features"):
                batch = images[i:i + self.config.batch_size]
                batch = torch.stack(batch).to(self.device)
                
                # Resize to 299x299 for Inception
                batch = F.interpolate(batch, size=(299, 299), 
                                     mode='bilinear', align_corners=False)
                
                # Normalize from [-1, 1] to ImageNet stats
                batch = (batch + 1.0) / 2.0
                batch = self.inception_transform(batch)
                
                _ = self.inception_model(batch)
        
        return torch.cat(self.fid_features, dim=0).numpy()
    
    def _get_predictions(self, images: List[torch.Tensor]) -> np.ndarray:
        """Get Inception predictions for IS"""
        classifier = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        classifier.eval().to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.config.batch_size),
                         desc="Getting predictions"):
                batch = images[i:i + self.config.batch_size]
                batch = torch.stack(batch).to(self.device)
                
                batch = F.interpolate(batch, size=(299, 299), 
                                     mode='bilinear', align_corners=False)
                batch = (batch + 1.0) / 2.0
                batch = self.inception_transform(batch)
                
                logits = classifier(batch)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def _calculate_frechet_distance(self, real_features: np.ndarray, 
                                   fake_features: np.ndarray) -> float:
        """Calculate Fréchet distance between feature distributions"""
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Add epsilon for numerical stability
        eps = 1e-6
        sigma_real += eps * np.eye(sigma_real.shape[0])
        sigma_fake += eps * np.eye(sigma_fake.shape[0])
        
        diff = mu_real - mu_fake
        
        # Calculate sqrt of product of covariances
        try:
            covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = (diff @ diff + np.trace(sigma_real) + 
                   np.trace(sigma_fake) - 2 * np.trace(covmean))
        except:
            # Fallback calculation
            eigvals_real = np.maximum(linalg.eigvals(sigma_real), eps)
            eigvals_fake = np.maximum(linalg.eigvals(sigma_fake), eps)
            
            fid = (diff @ diff + np.sum(eigvals_real) + 
                   np.sum(eigvals_fake) - 
                   2 * np.sum(np.sqrt(eigvals_real * eigvals_fake)))
        
        return float(max(0.0, fid))

# =============================================================================
# DATASET LOADING WITH TRAIN/VAL SPLIT 
# =============================================================================

class DatasetLoader:
    """Load real images for evaluation with train/val split support"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def load_real_images(self, max_images: int = None, use_validation: bool = True) -> List[torch.Tensor]:
        """
        Load real images with standard preprocessing.
        
        Args:
            max_images: Maximum number of images to load
            use_validation: If True, load only validation split; if False, load all
        """
        dataset_path = Path(self.config.dataset_path)
        
        if use_validation:
            print(f"Loading VALIDATION images from {self.config.dataset_path}")
            split_file = dataset_path / 'split_info.json'
            
            if not split_file.exists():
                print(f"\n{'='*60}")
                print("WARNING: split_info.json not found!")
                print(f"{'='*60}")
                print("Cannot load validation set. Options:")
                print("1. Run train.py with --split_dataset first")
                print("2. Set use_validation=False to load all images")
                print(f"{'='*60}\n")
                raise FileNotFoundError(f"split_info.json not found at {split_file}")
            
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            
            image_files_dict = split_info['val']
            print(f"Loaded validation split info:")
            print(f"  Total validation images: {split_info['stats']['overall']['val']}")
        else:
            print(f"Loading ALL images from {self.config.dataset_path}")
            image_files_dict = None
        
        transform = transforms.Compose([
            transforms.Resize((self.config.img_height, self.config.img_width), 
                            Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        images = []
        class_names = ['city', 'forest', 'rural']
        
        for class_name in class_names:
            class_path = dataset_path / class_name
            if not class_path.exists():
                print(f"Warning: {class_path} not found")
                continue
            
            if use_validation and image_files_dict:
                # Load only validation images
                if class_name not in image_files_dict:
                    print(f"Warning: {class_name} not in validation split")
                    continue
                
                image_list = image_files_dict[class_name]
                image_paths = [class_path / img_name for img_name in image_list]
            else:
                # Load all images
                image_paths = (list(class_path.glob("*.jpg")) + 
                              list(class_path.glob("*.jpeg")) + 
                              list(class_path.glob("*.png")))
            
            # Limit if max_images specified
            if max_images:
                max_per_class = max_images // 3
                image_paths = image_paths[:max_per_class]
            
            for img_path in tqdm(image_paths, desc=f"Loading {class_name}"):
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = transform(image)
                    images.append(image_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        # Shuffle and limit to max_images if specified
        if max_images and len(images) > max_images:
            random.shuffle(images)
            images = images[:max_images]
        
        split_type = "validation" if use_validation else "all"
        print(f"Loaded {len(images)} {split_type} images")
        return images

# =============================================================================
# COMPREHENSIVE EVALUATOR
# =============================================================================

class AcademicPanoramicEvaluator:
    """Main evaluator class"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.gan_metrics = ValidatedGANMetrics(config)
        self.dataset_loader = DatasetLoader(config)
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self._set_seeds()
    
    def _set_seeds(self):
        """Set seeds for reproducibility"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
    
    def evaluate_model(self, model_path: str, model_name: str, 
                      real_images: List[torch.Tensor]) -> Dict:
        """Complete evaluation of a single model"""
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*80}")
        
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(real_images),
            'evaluation_set': 'validation' if self.config.use_validation else 'all',
            'sample_size_note': f'{len(real_images)} samples used'
        }
        
        try:
            # Load generator
            generator = self._load_generator(model_path, model_name)
            if generator is None:
                results['status'] = 'Failed - Model Loading Error'
                return results
            
            # Generate images
            fake_images = self._generate_images(generator, self.config.num_samples)
            
            print("\nCalculating Metrics...")
            
            # 1. Validated FID
            fid_score, fid_stats = self.gan_metrics.calculate_fid(real_images, fake_images)
            results['fid_analysis'] = {
                'score': fid_score,
                'reference': 'Heusel et al. NIPS 2017',
                'stats': fid_stats
            }
            
            # 2. Validated IS
            is_mean, is_std, is_stats = \
                self.gan_metrics.calculate_inception_score(fake_images)
            
            results['inception_score_analysis'] = {
                'mean': float(is_mean),
                'std': float(is_std),
                'reference': 'Salimans et al. NIPS 2016',
                'stats': is_stats
            }
            
            # 3. Quality Assessment
            results['quality_assessment'] = self._assess_quality(results)
            print(f"\n   Overall Quality: {results['quality_assessment']['overall_quality']}")
            
            results['status'] = 'Success'
            
            # Save samples
            self._save_samples(fake_images, model_name)
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            results['status'] = f'Failed - {str(e)}'
        
        finally:
            if 'generator' in locals():
                del generator
            if 'fake_images' in locals():
                del fake_images
            torch.cuda.empty_cache()
        
        return results
    
    def _load_generator(self, model_path: str, model_name: str) -> Optional[nn.Module]:
        """Load generator"""
        print(f"Loading {model_name}...")
        
        try:
            generator = ImprovedGenerator(
                noise_dim=self.config.noise_dim,
                class_dim=self.config.n_classes,
                img_channels=3
            ).to(torch.device(self.config.device))
            
            if model_path.endswith('.pt'):
                checkpoint = torch.load(model_path, map_location=self.config.device, 
                                       weights_only=False)
                if 'ema_generator' in checkpoint:
                    generator.load_state_dict(checkpoint['ema_generator'])
                    print(f"  ✓ Loaded EMA generator")
                elif 'generator' in checkpoint:
                    generator.load_state_dict(checkpoint['generator'])
                    print(f"  ✓ Loaded regular generator")
                else:
                    print(f"  ✗ No generator in checkpoint")
                    return None
            else:
                state_dict = torch.load(model_path, map_location=self.config.device, 
                                       weights_only=True)
                generator.load_state_dict(state_dict)
                print(f"  ✓ Loaded from state dict")
            
            generator.eval()
            return generator
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None
    
    def _generate_images(self, generator: nn.Module, num_images: int) -> List[torch.Tensor]:
        """Generate images"""
        print(f"Generating {num_images} images...")
        
        images = []
        device = torch.device(self.config.device)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_images, self.config.batch_size), 
                         desc="Generating"):
                batch_size = min(self.config.batch_size, num_images - i)
                
                noise = torch.randn(batch_size, self.config.noise_dim, device=device)
                labels = torch.randint(0, self.config.n_classes, (batch_size,), 
                                      device=device)
                
                batch_images = generator(noise, labels)
                images.extend([img.cpu() for img in batch_images])
        
        return images
    
    def _assess_quality(self, results: Dict) -> Dict:
        """Academic quality assessment"""
        fid_score = results['fid_analysis']['score']
        is_score = results['inception_score_analysis']['mean']
        
        assessment = {
            'overall_quality': 'Unknown'
        }
        
        # Determine overall quality based on FID and IS only
        if fid_score <= 50 and is_score >= 3.0:
            assessment['overall_quality'] = 'Excellent'
        elif fid_score <= 80 and is_score >= 2.5:
            assessment['overall_quality'] = 'Good'
        elif fid_score <= 120 and is_score >= 2.0:
            assessment['overall_quality'] = 'Fair'
        else:
            assessment['overall_quality'] = 'Poor'
        
        return assessment
    
    def _save_samples(self, images: List[torch.Tensor], model_name: str):
        """Save sample images"""
        sample_images = images[:25]
        grid = make_grid(sample_images, nrow=5, normalize=True, value_range=(-1, 1))
        
        output_path = Path(self.config.output_dir) / f"{model_name}_samples.png"
        save_image(grid, output_path)
        print(f"Saved samples to {output_path}")

# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def run_academic_evaluation(model_paths: Dict[str, str],
                           dataset_path: str,
                           output_dir: str = "./academic_evaluation",
                           num_samples: int = 1000,
                           use_validation: bool = True) -> pd.DataFrame:
    """
    Run academic evaluation on panoramic GAN models
    
    Args:
        model_paths: Dictionary of {model_name: model_path}
        dataset_path: Path to dataset root directory
        output_dir: Output directory for results
        num_samples: Number of samples (1000 default)
        use_validation: If True, use validation split; if False, use all images
    
    Returns:
        DataFrame with results
    """
    
    print("\n" + "="*80)
    print("ACADEMIC PANORAMIC GAN EVALUATION")
    print("="*80)
    
    split_mode = "VALIDATION SET" if use_validation else "ALL IMAGES"
    print(f"Evaluation mode: {split_mode}")
    
    # Create configuration
    config = EvaluationConfig(
        model_paths=model_paths,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_samples=num_samples,
        use_validation=use_validation
    )
    
    # Initialize evaluator
    evaluator = AcademicPanoramicEvaluator(config)
    
    # Load real images
    real_images = evaluator.dataset_loader.load_real_images(
        max_images=num_samples,
        use_validation=use_validation
    )
    
    if len(real_images) == 0:
        print("No real images loaded. Exiting.")
        return pd.DataFrame()
    
    # Evaluate all models
    all_results = []
    successful_evaluations = 0
    
    for model_name, model_path in model_paths.items():
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            continue
        
        result = evaluator.evaluate_model(model_path, model_name, real_images)
        all_results.append(result)
        
        if result['status'] == 'Success':
            successful_evaluations += 1
    
    if successful_evaluations == 0:
        print("\n No models were successfully evaluated!")
        return pd.DataFrame()
    
    # Create results DataFrame
    df_rows = []
    for result in all_results:
        if result['status'] == 'Success':
            row = {
                'Model': result['model_name'],
                'FID_Score': result['fid_analysis']['score'],
                'IS_Mean': result['inception_score_analysis']['mean'],
                'Quality': result['quality_assessment']['overall_quality'],
                'Eval_Set': result['evaluation_set']
            }
            df_rows.append(row)
    
    df = pd.DataFrame(df_rows)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = Path(output_dir) / f"detailed_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save CSV
    csv_path = Path(output_dir) / f"summary_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Display results
    print(f"\n{'='*100}")
    print("EVALUATION RESULTS")
    print(f"{'='*100}")
    
    # Simple table
    print("\n" + df.to_string(index=False, float_format='%.3f'))
    
    # Best models 
    print(f"\n{'='*100}")
    print("BEST MODEL RANKINGS")
    print(f"{'='*100}")
    
    if not df.empty:
        best_fid = df.loc[df['FID_Score'].idxmin()]
        best_is = df.loc[df['IS_Mean'].idxmax()]
        
        print(f"\n Best FID: {best_fid['Model']} ({best_fid['FID_Score']:.2f})")
        print(f" Best IS: {best_is['Model']} ({best_is['IS_Mean']:.3f})")
    
    print(f"\n Results saved to:")
    print(f"    CSV: {csv_path}")
    print(f"    JSON: {json_path}")
    
    print(f"\n ACADEMIC EVALUATION COMPLETE!")
    print(f"   Successfully evaluated {successful_evaluations}/{len(model_paths)} models")
    print(f"   Evaluation set: {split_mode}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Panoramic GAN Models')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--use_validation', action='store_true',
                       help='Use validation set')
    
    args = parser.parse_args()
    
    # Find all model files in the directory
    model_dir = Path(args.model_dir)
    model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
    
    if not model_files:
        print(f"No model files found in {model_dir}")
        exit(1)
    
    # Build model_paths dictionary
    model_paths = {}
    for model_file in model_files:
        model_name = model_file.stem  # filename without extension
        model_paths[model_name] = str(model_file)
    
    print(f"Found {len(model_paths)} models:")
    for name in model_paths.keys():
        print(f"  - {name}")
    
    # Run evaluation
    results_df = run_academic_evaluation(
        model_paths=model_paths,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_validation=args.use_validation

    )
