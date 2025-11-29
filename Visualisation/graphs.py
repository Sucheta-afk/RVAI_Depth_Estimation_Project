import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

class DepthMetricsVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Add Metric3D color
        self.colors = {
            "DAV2":   "#3498db",
            "DAV3":   "#e74c3c",
            "APPLE":  "#9b59b6",
            "M3D":    "#2ecc71"   # green
        }

    def load_metrics(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    # ===============================================
    # Main unified function
    # ===============================================
    def create_comparison_plots(self, model_paths: dict, output_dir):
        """
        model_paths = {
            "DAV2":  "path/to/metrics.json",
            "DAV3":  "...",
            "APPLE": "...",
            "M3D":   "..."
        }
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all model jsons
        models = {name: self.load_metrics(path) for name, path in model_paths.items()}

        print("📊 Creating visualizations...")

        self.plot_average_metrics(models, output_path)
        self.plot_per_image_comparison(models, output_path)
        self.plot_metric_distributions(models, output_path)
        self.plot_error_analysis(models, output_path)
        self.create_dashboard(models, output_path)

        print(f"✅ All visualizations saved to: {output_path}")

    # ===============================================
    # 1. Average Metrics
    # ===============================================
    def plot_average_metrics(self, models, output_path):
        metrics = ['delta_1', 'delta_2', 'delta_3', 'abs_rel', 'rmse', 'mae']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            names = list(models.keys())
            vals = [models[n]['average_metrics'][metric] for n in names]
            colors = [self.colors[n] for n in names]

            bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="black")

            # Label above bars
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h, f"{h:.4f}",
                        ha="center", va="bottom", fontweight="bold")

            # Determine best performer
            if metric.startswith("delta"):
                best = np.argmax(vals)
            else:
                best = np.argmin(vals)

            bars[best].set_edgecolor("gold")
            bars[best].set_linewidth(3)

            ax.set_title(metric.upper(), fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Average Metrics Comparison Across Models", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "comparison_average_metrics.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   ✓ Average metrics comparison saved")

    # ===============================================
    # 2. Per-image Comparison
    # ===============================================
    def plot_per_image_comparison(self, models, output_path):
        metrics = ['delta_1', 'abs_rel', 'rmse']
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Determine shortest dataset (20 images each)
        num = min(len(models[n]['per_image_metrics']) for n in models)

        img_idx = np.arange(1, num + 1)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for name, data in models.items():
                vals = [x[metric] for x in data['per_image_metrics'][:num]]
                ax.plot(img_idx, vals, marker='o', linewidth=2,
                        label=name, color=self.colors[name])

            ax.set_title(f"Per-Image {metric.upper()}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Image Index")
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / "comparison_per_image.png", dpi=150)
        plt.close()
        print("   ✓ Per-image comparison saved")

    # ===============================================
    # 3. Distribution Plots
    # ===============================================
    def plot_metric_distributions(self, models, output_path):
        metrics = ['delta_1', 'delta_2', 'delta_3', 'abs_rel', 'rmse', 'mae']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            values = [[img[metric] for img in models[name]['per_image_metrics']]
                      for name in models]

            bp = ax.boxplot(values, labels=models.keys(), showmeans=True, patch_artist=True)

            for patch, color in zip(bp['boxes'], self.colors.values()):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_title(f"{metric.upper()} Distribution", fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "comparison_distributions.png", dpi=150)
        plt.close()
        print("   ✓ Distribution plots saved")

    # ===============================================
    # 4. Error Analysis Scatter Plots
    # ===============================================
    def plot_error_analysis(self, models, output_path):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        num = min(len(models[n]['per_image_metrics']) for n in models)

        def get(vals, key): return [x[key] for x in vals[:num]]

        # RMSE vs AbsRel
        ax = axes[0]
        for name, data in models.items():
            ax.scatter(get(data['per_image_metrics'], 'rmse'),
                       get(data['per_image_metrics'], 'abs_rel'),
                       s=100, alpha=0.7, color=self.colors[name],
                       label=name, edgecolors='black')
        ax.set_title("RMSE vs AbsRel")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("AbsRel")
        ax.grid(True)
        ax.legend()

        # Delta1 vs RMSE
        ax = axes[1]
        for name, data in models.items():
            ax.scatter(get(data['per_image_metrics'], 'delta_1'),
                       get(data['per_image_metrics'], 'rmse'),
                       s=100, alpha=0.7, color=self.colors[name],
                       label=name, edgecolors='black')
        ax.set_title("Δ1 vs RMSE")
        ax.set_xlabel("Δ1")
        ax.set_ylabel("RMSE")
        ax.grid(True)
        ax.legend()

        # Inference time
        ax = axes[2]
        for name, data in models.items():
            ax.scatter(range(num), get(data['per_image_metrics'], 'inference_time'),
                       s=100, alpha=0.7, color=self.colors[name],
                       label=name, edgecolors='black')
        ax.set_title("Inference Time per Image")
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / "comparison_error_analysis.png", dpi=150)
        plt.close()
        print("   ✓ Error analysis saved")

    # ===============================================
    # 5. Dashboard Table + Bar Charts
    # ===============================================
    def create_dashboard(self, models, output_path):

        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        fig.suptitle("Comprehensive Metrics Dashboard Across All Models",
                     fontsize=18, fontweight="bold")

        names = list(models.keys())
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis("off")

        metrics = ["delta_1", "delta_2", "delta_3", "rmse", "abs_rel", "mae", "inference_time"]

        # Build table
        table_data = [["Metric"] + names + ["Winner"]]

        for m in metrics:
            vals = [models[n]['average_metrics'][m] for n in names]

            # Determine winner
            if m.startswith("delta"):
                best = names[np.argmax(vals)]
            else:
                best = names[np.argmin(vals)]

            row = [m.upper()] + [f"{v:.4f}" for v in vals] + [best]
            table_data.append(row)

        table = ax1.table(cellText=table_data, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)

        # Charts on bottom
        metrics_to_plot = ["delta_1", "delta_2", "rmse", "abs_rel", "mae", "rmse_log"]
        positions = [(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

        for metric, pos in zip(metrics_to_plot, positions):
            ax = fig.add_subplot(gs[pos[0], pos[1]])

            vals = [models[n]['average_metrics'][metric] for n in names]
            bar_colors = [self.colors[n] for n in names]

            bars = ax.bar(names, vals, color=bar_colors, edgecolor="black", linewidth=2)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h,
                        f"{h:.4f}", ha="center", va="bottom",
                        fontweight="bold", fontsize=10)

            ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
            ax.grid(axis='y', alpha=0.3)

        plt.savefig(output_path / "dashboard_comprehensive.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("   ✓ Comprehensive dashboard saved")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    model_paths = {
        "DAV2":  r"D:\RVAI - Last\DAV2\results_dav2_metric_LAST\metrics.json",
        "DAV3":  r"D:\RVAI - Last\DAV3\results_dav3_metric_LAST\metrics.json",
        "APPLE": r"D:\RVAI - Last\Apple-Depth-pro\depthpro_results\metrics.json",
        "M3D":   r"D:\RVAI - Last\Metric3D\metric3d_results\metrics.json"
    }

    OUTPUT_DIR = r"D:\RVAI - Last\Visualisation\DepthMetricsComparison_metric2d"

    visualizer = DepthMetricsVisualizer()
    visualizer.create_comparison_plots(model_paths, OUTPUT_DIR)
