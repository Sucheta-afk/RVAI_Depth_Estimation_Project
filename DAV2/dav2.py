import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DAV2MetricTester:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", device="cuda", use_scale_alignment=True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.use_scale_alignment = use_scale_alignment

        print(f"🚀 Using device: {self.device}")
        print(f"📏 Scale alignment: {'Enabled' if use_scale_alignment else 'Disabled (metric model)'}")

    def load_model(self):
        print(f"📥 Loading Depth Anything V2 Metric ({self.model_id}) from Hugging Face...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            print("✅ Model loaded!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Check your internet connection and Hugging Face credentials.")
            return False

    def load_kitti_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        return image, image_np, image.size

    def load_kitti_depth(self, depth_path):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 256.0  # KITTI units: mm -> meters
        valid_mask = depth > 0
        return depth, valid_mask

    def predict_depth(self, image_pil, original_size):
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            start = time.time()
            outputs = self.model(**inputs)
            inference_time = time.time() - start

        pred = outputs.predicted_depth.squeeze(0)
        # Interpolate to original size
        pred_resized = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0),
            size=(original_size[1], original_size[0]),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
        return pred_resized, inference_time

    def align_scale(self, pred, gt, valid_mask):
        """
        Align prediction scale to ground truth using least squares
        This is standard practice for metric depth evaluation
        """
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        
        # Compute optimal scale and shift using least squares
        # pred_aligned = scale * pred + shift
        A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
        scale, shift = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
        
        pred_aligned = scale * pred + shift
        
        return pred_aligned, scale, shift

    def compute_metrics(self, pred, gt, valid_mask):
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        
        # Clamp to avoid numerical issues
        pred = np.clip(pred, 1e-3, None)
        gt = np.clip(gt, 1e-3, None)
        
        thresh = np.maximum((gt / pred), (pred / gt))
        delta_1 = (thresh < 1.25).mean()
        delta_2 = (thresh < 1.25 ** 2).mean()
        delta_3 = (thresh < 1.25 ** 3).mean()
        rmse = np.sqrt(((pred - gt) ** 2).mean())
        rmse_log = np.sqrt(((np.log(pred) - np.log(gt)) ** 2).mean())
        mae = np.abs(pred - gt).mean()
        abs_rel = (np.abs(pred - gt) / gt).mean()
        sq_rel = (((pred - gt) ** 2) / gt).mean()
        return {
            'delta_1': delta_1,
            'delta_2': delta_2,
            'delta_3': delta_3,
            'rmse': rmse,
            'rmse_log': rmse_log,
            'mae': mae,
            'abs_rel': abs_rel,
            'sq_rel': sq_rel
        }

    def visualize_single(self, image_np, pred_depth, gt_depth, valid_mask, metrics, output_path, scale_info):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        im1 = axes[0, 1].imshow(pred_depth, cmap='plasma', vmin=0, vmax=80)
        axes[0, 1].set_title('Predicted Depth (Scale-Aligned)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='meters')
        gt_display = gt_depth.copy()
        gt_display[~valid_mask] = np.nan
        im2 = axes[1, 0].imshow(gt_display, cmap='plasma', vmin=0, vmax=80)
        axes[1, 0].set_title('Ground Truth Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label='meters')
        error = np.abs(pred_depth - gt_depth)
        error[~valid_mask] = np.nan
        im3 = axes[1, 1].imshow(error, cmap='hot', vmin=0, vmax=10)
        axes[1, 1].set_title('Absolute Error')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, label='meters')
        
        metrics_text = f"Scale: {scale_info['scale']:.4f}, Shift: {scale_info['shift']:.4f}\n\n"
        metrics_text += '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat'))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {output_path}")

    def test_on_kitti(self, image_dir, gt_dir, output_dir, num_samples=10):
        """
        Test on KITTI dataset with separate image and ground truth directories
        
        Args:
            image_dir: Path to directory containing RGB images
            gt_dir: Path to directory containing ground truth depth images
            output_dir: Path to save results
            num_samples: Number of images to process
        """
        image_path = Path(image_dir)
        gt_path = Path(gt_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = sorted(list(image_path.glob("*.png")))
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
            
        print(f"📁 Found {len(image_files)} images in: {image_dir}")
        
        # Check if ground truth directory exists
        if not gt_path.exists():
            print(f"⚠️  Ground truth directory not found: {gt_dir}")
            print(f"    Will generate depth predictions only (no metrics)")
            has_ground_truth = False
        else:
            print(f"✅ Ground truth directory found: {gt_dir}")
            has_ground_truth = True
        
        # Process only num_samples images
        image_files = image_files[:num_samples]
        all_metrics = []
        
        for idx, img_path in enumerate(image_files):
            print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_path.name}")
            
            # Load image
            image_pil, image_np, original_size = self.load_kitti_image(img_path)
            
            # Predict depth
            try:
                pred_depth_raw, inf_time = self.predict_depth(image_pil, original_size)
                print(f"   ⏱️  Inference time: {inf_time:.3f}s ({1/inf_time:.1f} FPS)")
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
                continue
            
            # Check if corresponding ground truth exists
            # Since you've renamed files to match (1.png, 2.png, etc.), 
            # GT file has the same name as the image file
            gt_file = gt_path / img_path.name
            
            if has_ground_truth and gt_file.exists():
                print(f"   📁 GT file: {gt_file}")
                
                # Load ground truth
                gt_depth, valid_mask = self.load_kitti_depth(gt_file)
                
                # Apply scale alignment only if enabled (for relative depth models)
                if self.use_scale_alignment:
                    pred_depth, scale, shift = self.align_scale(pred_depth_raw, gt_depth, valid_mask)
                    scale_info = {'scale': scale, 'shift': shift}
                    print(f"   🔧 Scale alignment: scale={scale:.4f}, shift={shift:.4f}")
                else:
                    pred_depth = pred_depth_raw
                    scale_info = {'scale': 1.0, 'shift': 0.0}
                    print(f"   📏 Using raw metric depth (no alignment)")
                
                # Compute metrics
                metrics = self.compute_metrics(pred_depth, gt_depth, valid_mask)
                metrics['inference_time'] = inf_time
                all_metrics.append(metrics)
                
                print(f"   📊 δ1: {metrics['delta_1']:.3f} | RMSE: {metrics['rmse']:.3f} | AbsRel: {metrics['abs_rel']:.3f}")
                
                # Visualize with metrics
                vis_path = output_path / f"result_{idx:03d}_{img_path.stem}.png"
                self.visualize_single(image_np, pred_depth, gt_depth, valid_mask, metrics, vis_path, scale_info)
            else:
                # Just save prediction without metrics
                if has_ground_truth:
                    print(f"   ⚠️  Ground truth not found: {gt_file}")
                
                pred_depth = pred_depth_raw
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].imshow(image_np)
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                im1 = axes[1].imshow(pred_depth, cmap='plasma')
                axes[1].set_title('Predicted Depth (Raw)')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, label='depth units')
                plt.tight_layout()
                vis_path = output_path / f"result_{idx:03d}_{img_path.stem}_nogt.png"
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   💾 Saved prediction only: {vis_path}")
        
        # Print average metrics if available
        if all_metrics:
            # Convert numpy types to Python native types for JSON serialization
            avg_metrics = {k: float(np.mean([m[k] for m in all_metrics]))
                            for k in all_metrics[0].keys()}
            print("\n" + "="*50)
            print("📊 AVERAGE METRICS")
            print("="*50)
            for k, v in avg_metrics.items():
                print(f"{k:15s}: {v:.4f}")
            print("="*50)
            
            # Convert all metrics to Python native types
            per_image_metrics = []
            for m in all_metrics:
                per_image_metrics.append({k: float(v) for k, v in m.items()})
            
            results_file = output_path / "metrics.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'average_metrics': avg_metrics,
                    'per_image_metrics': per_image_metrics
                }, f, indent=2)
            print(f"\n💾 Results saved to: {results_file}")

# ========== MAIN ==========

if __name__ == "__main__":
    # Updated paths for val_selection_cropped dataset
    IMAGE_DIR = r"D:\RVAI - Last\Common\KITTI Benchmark\data_depth_selection\depth_selection\val_selection_cropped\image"
    GT_DIR = r"D:\RVAI - Last\Common\KITTI Benchmark\data_depth_selection\depth_selection\val_selection_cropped\groundtruth_depth"
    OUTPUT_DIR = r"D:\RVAI - Last\DAV2\results_dav2_metric_LAST"
    NUM_SAMPLES = 20
    
    # IMPORTANT: Set to False for metric models to test true absolute depth accuracy
    USE_SCALE_ALIGNMENT = False  # Set to True if using a relative/monocular model
    
    print("="*60)
    print("KITTI Depth Anything V2 Evaluation")
    print("="*60)
    print(f"Image directory:  {IMAGE_DIR}")
    print(f"GT directory:     {GT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print("="*60 + "\n")
    
    tester = DAV2MetricTester(device='cpu', use_scale_alignment=USE_SCALE_ALIGNMENT)
    
    if not tester.load_model():
        print("\n❌ Model loading failed!")
        exit(1)
    
    tester.test_on_kitti(IMAGE_DIR, GT_DIR, OUTPUT_DIR, NUM_SAMPLES)
    print("\n✅ Testing complete!")