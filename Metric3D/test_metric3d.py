import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json


class Metric3DKITTITester:
    def __init__(self, hub_model='metric3d_vit_small', device='cuda'):
        """
        hub_model options:
            - metric3d_convnext_tiny
            - metric3d_convnext_large
            - metric3d_vit_small
            - metric3d_vit_large
            - metric3d_vit_giant2
        """
        self.hub_model = hub_model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None

        print(f"🚀 Using device: {self.device}")
        print(f"📦 Model: {self.hub_model}")

    # ---------------------- MODEL LOADING ----------------------
    def load_model(self):
        print("📥 Loading Metric3D via PyTorch Hub...")

        self.model = torch.hub.load(
            'yvanyin/metric3d',
            self.hub_model,
            pretrain=True
        ).to(self.device)

        self.model.eval()
        print("✅ Metric3D model ready!")
        return True

    # ---------------------- KITTI LOADING ----------------------
    def load_kitti_depth(self, depth_path):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 256.0
        valid_mask = depth > 0
        return depth, valid_mask

    # ---------------------- PREDICTION ----------------------
    def predict_depth(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)

        # prepare input
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            t0 = time.time()

            # NEW API: model.inference({"input": tensor})
            pred_depth, confidence, output_dict = self.model.inference(
                {"input": img}
            )

            pred_depth = pred_depth[0, 0].cpu().numpy()
            t1 = time.time()

        return pred_depth, (t1 - t0)

    # ---------------------- METRICS ----------------------
    def compute_metrics(self, pred, gt, valid):
        pred = pred[valid]
        gt = gt[valid]

        pred = np.clip(pred, 1e-3, None)
        gt = np.clip(gt, 1e-3, None)

        thresh = np.maximum(gt / pred, pred / gt)
        delta_1 = (thresh < 1.25).mean()
        delta_2 = (thresh < (1.25 ** 2)).mean()
        delta_3 = (thresh < (1.25 ** 3)).mean()

        rmse = np.sqrt(((pred - gt) ** 2).mean())
        rmse_log = np.sqrt(((np.log(pred) - np.log(gt)) ** 2).mean())
        mae = np.abs(pred - gt).mean()
        abs_rel = (np.abs(pred - gt) / gt).mean()
        sq_rel = (((pred - gt) ** 2) / gt).mean()

        return {
            "delta_1": float(delta_1),
            "delta_2": float(delta_2),
            "delta_3": float(delta_3),
            "rmse": float(rmse),
            "rmse_log": float(rmse_log),
            "mae": float(mae),
            "abs_rel": float(abs_rel),
            "sq_rel": float(sq_rel),
        }

    # ---------------------- VISUALIZATION ----------------------
    def visualize(self, image_path, pred, gt, valid_mask, metrics, save_path):
        image = np.array(Image.open(image_path).convert("RGB"))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        im1 = axes[0, 1].imshow(pred, cmap="plasma", vmin=0, vmax=80)
        axes[0, 1].set_title("Metric3D Depth")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1])

        gt_show = gt.copy()
        gt_show[~valid_mask] = np.nan
        im2 = axes[1, 0].imshow(gt_show, cmap="plasma", vmin=0, vmax=80)
        axes[1, 0].set_title("Ground Truth")
        axes[1, 0].axis("off")
        plt.colorbar(im2, ax=axes[1, 0])

        error = np.abs(pred - gt)
        error[~valid_mask] = np.nan
        im3 = axes[1, 1].imshow(error, cmap="hot", vmin=0, vmax=10)
        axes[1, 1].set_title("Absolute Error")
        axes[1, 1].axis("off")
        plt.colorbar(im3, ax=axes[1, 1])

        text = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        fig.text(
            0.03, 0.02, text, fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat")
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"💾 Saved: {save_path}")

    # ---------------------- MAIN KITTI TEST ----------------------
    def test_on_kitti(self, image_dir, gt_dir, out_dir, num_images=20):
        image_dir = Path(image_dir)
        gt_dir = Path(gt_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(image_dir.glob("*.png"))[:num_images]

        print(f"📁 Testing on {len(images)} images...\n")
        metrics_list = []

        for i, img_path in enumerate(images):
            print(f"[{i+1}/{len(images)}] {img_path.name}")

            pred, inf_t = self.predict_depth(img_path)
            print(f"  ⏱ {inf_t:.3f}s ({1/inf_t:.1f} FPS)")

            gt_path = gt_dir / img_path.name
            if gt_path.exists():
                gt, valid = self.load_kitti_depth(gt_path)

                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                metrics = self.compute_metrics(pred, gt, valid)
                metrics["inference_time"] = float(inf_t)
                metrics_list.append(metrics)

                vis_path = out_dir / f"result_{i:03d}.png"
                self.visualize(img_path, pred, gt, valid, metrics, vis_path)

                print(f"  📊 δ1={metrics['delta_1']:.3f}, RMSE={metrics['rmse']:.3f}")
            else:
                print("  ⚠️ GT missing — skipping metrics.")

        if metrics_list:
            avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            print("\n📊 AVERAGE METRICS:")
            for k, v in avg.items():
                print(f"{k:15s}: {v:.4f}")

            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"average": avg, "per_image": metrics_list}, f, indent=2)

            print("\n💾 Saved metrics.json")


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    IMAGE_DIR = r"D:\RVAI - Last\Common\KITTI Benchmark\data_depth_selection\depth_selection\val_selection_cropped\image"
    GT_DIR = r"D:\RVAI - Last\Common\KITTI Benchmark\data_depth_selection\depth_selection\val_selection_cropped\groundtruth_depth"
    OUTPUT_DIR = r"D:\RVAI - Last\Metric3D\results_metric3d"

    NUM_SAMPLES = 20

    tester = Metric3DKITTITester(
        hub_model="metric3d_vit_small",     # CHANGE MODEL HERE
        device="cuda"
    )

    tester.load_model()
    tester.test_on_kitti(IMAGE_DIR, GT_DIR, OUTPUT_DIR, NUM_SAMPLES)

    print("\n✅ DONE!")
