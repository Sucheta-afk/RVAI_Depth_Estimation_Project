import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from depth_anything_3.api import DepthAnything3


class DAV3KITTITester:
    def __init__(self, model_name='da3mono-large', device='cuda', use_scale_alignment=True):
        """
        model_name: 'da3mono-small', 'da3mono-base', 'da3mono-large', 
                   'da3metric-small', 'da3metric-base', 'da3metric-large'
        use_scale_alignment: True for mono models, False for metric models
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.use_scale_alignment = use_scale_alignment

        print(f"🚀 Using device: {self.device}")
        print(f"📏 Scale alignment: {'Enabled' if use_scale_alignment else 'Disabled (metric model)'}")

    def load_model(self):
        """Load DA3 monocular model"""
        print(f"📥 Loading Depth Anything V3: {self.model_name}...")

        try:
            self.model = DepthAnything3.from_pretrained(f"depth-anything/{self.model_name}")
            self.model = self.model.to(device=self.device)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def load_kitti_depth(self, depth_path):
        """Load KITTI depth GT"""
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 256.0  # convert to meters
        valid_mask = depth > 0
        return depth, valid_mask

    def predict_depth(self, image_path):
        """Run DA3 inference on single image"""
        start_time = time.time()

        prediction = self.model.inference(
            image=[str(image_path)],
            intrinsics=None,
            extrinsics=None,
            infer_gs=False,
            export_feat_layers=None
        )

        inference_time = time.time() - start_time

        depth_map = prediction.depth[0]
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        return depth_map, inference_time

    def align_prediction_to_gt(self, pred, gt, valid_mask):
        """Scale & shift alignment for monocular depth"""
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        A = np.vstack([pred_valid, np.ones(len(pred_valid))]).T
        s, t = np.linalg.lstsq(A, gt_valid, rcond=None)[0]

        pred_aligned = s * pred + t
        return pred_aligned, s, t

    def compute_metrics(self, pred, gt, valid_mask):
        """Compute standard KITTI metrics"""
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        
        # Clamp to avoid numerical issues
        pred = np.clip(pred, 1e-3, None)
        gt = np.clip(gt, 1e-3, None)

        thresh = np.maximum(gt / pred, pred / gt)
        delta_1 = (thresh < 1.25).mean()
        delta_2 = (thresh < 1.25**2).mean()
        delta_3 = (thresh < 1.25**3).mean()

        rmse = np.sqrt(((pred - gt) ** 2).mean())
        rmse_log = np.sqrt(((np.log(pred) - np.log(gt)) ** 2).mean())
        mae = np.abs(pred - gt).mean()
        abs_rel = (np.abs(pred - gt) / gt).mean()
        sq_rel = (((pred - gt) ** 2) / gt).mean()

        return {
            'delta_1': float(delta_1),
            'delta_2': float(delta_2),
            'delta_3': float(delta_3),
            'rmse': float(rmse),
            'rmse_log': float(rmse_log),
            'mae': float(mae),
            'abs_rel': float(abs_rel),
            'sq_rel': float(sq_rel)
        }

    def visualize_result(self, image_path, pred_depth, gt_depth,
                         valid_mask, metrics, output_path, scale_info):
        """2x2 visualization matching DAV2 style"""
        image = np.array(Image.open(image_path).convert('RGB'))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Input Image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # 2. Predicted Depth (Scale-Aligned)
        im1 = axes[0, 1].imshow(pred_depth, cmap='plasma', vmin=0, vmax=80)
        axes[0, 1].set_title('Predicted Depth (Scale-Aligned)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='meters')
        
        # 3. Ground Truth Depth
        gt_display = gt_depth.copy()
        gt_display[~valid_mask] = np.nan
        im2 = axes[1, 0].imshow(gt_display, cmap='plasma', vmin=0, vmax=80)
        axes[1, 0].set_title('Ground Truth Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label='meters')
        
        # 4. Absolute Error
        error = np.abs(pred_depth - gt_depth)
        error[~valid_mask] = np.nan
        im3 = axes[1, 1].imshow(error, cmap='hot', vmin=0, vmax=10)
        axes[1, 1].set_title('Absolute Error')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, label='meters')
        
        # Metrics text
        metrics_text = f"Scale: {scale_info['scale']:.4f}, Shift: {scale_info['shift']:.4f}\n\n"
        metrics_text += '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items() 
                                   if k not in ['inference_time', 'fps']])
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
            
            # Predict depth
            try:
                pred_depth_raw, inf_time = self.predict_depth(img_path)
                print(f"   ⏱️  Inference time: {inf_time:.3f}s ({1/inf_time:.1f} FPS)")
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
                continue
            
            # Check if corresponding ground truth exists
            # Since files are renamed to match (1.png, 2.png, etc.)
            gt_file = gt_path / img_path.name
            
            if has_ground_truth and gt_file.exists():
                print(f"   📁 GT file: {gt_file}")
                
                # Load ground truth
                gt_depth, valid_mask = self.load_kitti_depth(gt_file)
                
                # Resize prediction to match GT shape
                pred_depth_raw = cv2.resize(
                    pred_depth_raw,
                    (gt_depth.shape[1], gt_depth.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Apply scale alignment only for mono models, not for metric models
                if self.use_scale_alignment:
                    pred_depth, scale, shift = self.align_prediction_to_gt(
                        pred_depth_raw, gt_depth, valid_mask)
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
                self.visualize_result(img_path, pred_depth, gt_depth, 
                                    valid_mask, metrics, vis_path, scale_info)
            else:
                # Just save prediction without metrics
                if has_ground_truth:
                    print(f"   ⚠️  Ground truth not found: {gt_file}")
                
                pred_depth = pred_depth_raw
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].imshow(np.array(Image.open(img_path)))
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


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Updated paths to match DAV2 script
    IMAGE_DIR = r"D:\RVAI - Last\Common\KITTI Benchmark\data_depth_selection\depth_selection\val_selection_cropped\image"
    GT_DIR = r"D:\RVAI - Last\Common\KITTI Benchmark\data_depth_selection\depth_selection\val_selection_cropped\groundtruth_depth"
    OUTPUT_DIR = r"D:\RVAI - Last\DAV3\results_dav3_metric_LAST"
    MODEL_NAME = 'da3metric-large'  # or 'da3mono-large' for monocular
    NUM_SAMPLES = 20
    
    # IMPORTANT: Set to False for metric models, True for mono models
    USE_SCALE_ALIGNMENT = False  # False for da3metric-*, True for da3mono-*

    print("="*60)
    print("KITTI Depth Anything V3 Evaluation")
    print("="*60)
    print(f"Image directory:  {IMAGE_DIR}")
    print(f"GT directory:     {GT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model:            {MODEL_NAME}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print("="*60 + "\n")

    tester = DAV3KITTITester(
        model_name=MODEL_NAME, 
        device='cuda',
        use_scale_alignment=USE_SCALE_ALIGNMENT
    )

    if not tester.load_model():
        print("\n❌ Model loading failed!")
        exit(1)

    tester.test_on_kitti(IMAGE_DIR, GT_DIR, OUTPUT_DIR, NUM_SAMPLES)
    print("\n✅ Testing complete!")