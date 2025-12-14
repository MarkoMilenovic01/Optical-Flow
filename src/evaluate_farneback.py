import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset_preparation import load_flo, flow_to_hsv


# ==================================================================================
#                       SINTEL DATASET FOR OPENCV ONLY
# ==================================================================================
class SintelDataset:
    def __init__(self, root, subset="clean", resize=None):
        self.root = root
        self.subset = subset
        self.resize = resize

        self.img1_paths = []
        self.img2_paths = []
        self.flow_paths = []

        subset_dir = os.path.join(root, subset)
        flow_dir = os.path.join(root, "flow")

        scenes = sorted(os.listdir(subset_dir))

        for scene in scenes:
            scene_dir = os.path.join(subset_dir, scene)
            frames = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])

            for i in range(len(frames) - 1):
                f1 = frames[i]
                f2 = frames[i + 1]
                flo = f"frame_{i+1:04d}.flo"

                self.img1_paths.append(os.path.join(scene_dir, f1))
                self.img2_paths.append(os.path.join(scene_dir, f2))
                self.flow_paths.append(os.path.join(flow_dir, scene, flo))

    def __len__(self):
        return len(self.img1_paths)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.img1_paths[idx])[:, :, ::-1] / 255.0
        img2 = cv2.imread(self.img2_paths[idx])[:, :, ::-1] / 255.0
        flow = load_flo(self.flow_paths[idx])

        if self.resize is not None:
            W, H = self.resize
            h0, w0, _ = img1.shape

            img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_LINEAR)

            flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= (W / w0)
            flow[..., 1] *= (H / h0)

        return img1, img2, flow


# ==================================================================================
#                     ENDPOINT ERROR (EPE)
# ==================================================================================
def epe(pred, gt):
    return np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=2)))


# ==================================================================================
#              RUN FARNEBACK FOR ONE PARAMETER CONFIGURATION
# ==================================================================================
def run_config(dataset, num_samples, resize, out_dir, subset, farneback_params):
    
    os.makedirs(out_dir, exist_ok=True)
    all_epe = []

    for i in range(num_samples):

        img1, img2, gt_flow = dataset[i]

        g1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Compute Farneback
        flow_pred = cv2.calcOpticalFlowFarneback(
            g1, g2, None, **farneback_params
        )

        err = epe(flow_pred, gt_flow)
        all_epe.append(err)

        pred_hsv = flow_to_hsv(flow_pred)
        gt_hsv = flow_to_hsv(gt_flow)

        # ---- Visualization ----
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img1)
        plt.title("Image 1")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_hsv)
        plt.title("GT Flow")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_hsv)
        plt.title(f"Pred Flow (EPE={err:.3f})")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{subset}_{i:04d}.png"), dpi=120)
        plt.close()

        print(f"[{i:04d}]  EPE = {err:.3f}")

    arr = np.array(all_epe)
    return float(arr.mean()), float(arr.std())


# ==================================================================================
#                      MASTER EVALUATION CONTROLLER
# ==================================================================================
def evaluate_multiple_farneback(
    root="data/MPI-Sintel-complete/training",
    subset="clean",
    num_samples=200,
    resize=(1024, 432),
    out_dir="farneback_multi_eval",
):

    os.makedirs(out_dir, exist_ok=True)
    dataset = SintelDataset(root, subset=subset, resize=resize)
    num_samples = min(num_samples, len(dataset))

    # =============================================================
    # PARAMETER CONFIGURATIONS
    # =============================================================
    configs = [
        ("default",
         dict(pyr_scale=0.5, levels=5, winsize=15, iterations=5,
              poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)),

        ("larger_window",
         dict(pyr_scale=0.5, levels=5, winsize=40, iterations=5,
              poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)),

        ("more_pyramid_levels",
         dict(pyr_scale=0.5, levels=8, winsize=25, iterations=5,
              poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)),

        ("smooth_motion",
         dict(pyr_scale=0.3, levels=6, winsize=25, iterations=7,
              poly_n=9, poly_sigma=1.7, flags=0)),
    ]

    # =============================================================
    # RUN ALL CONFIGURATIONS
    # =============================================================
    summary_path = os.path.join(out_dir, "summary_all_configs.txt")
    fsum = open(summary_path, "w")
    fsum.write("FARNEBACK MULTI-CONFIG RESULTS\n")
    fsum.write("---------------------------------\n\n")

    for cfg_name, params in configs:
        print(f"\n==============================")
        print(f"Testing Farneback config: {cfg_name}")
        print("==============================")

        cfg_dir = os.path.join(out_dir, cfg_name)
        mean_epe, std_epe = run_config(
            dataset, num_samples, resize, cfg_dir, subset, params
        )

        fsum.write(f"Config: {cfg_name}\n")
        fsum.write(f"Mean EPE: {mean_epe:.4f}\n")
        fsum.write(f"Std  EPE: {std_epe:.4f}\n")
        fsum.write(f"Params: {params}\n\n")

    fsum.close()
    print("\n✓ All configs evaluated! Check:", out_dir)


# ==================================================================================
# MAIN
# ==================================================================================
if __name__ == "__main__":
    evaluate_multiple_farneback(
        root=r"data/MPI-Sintel-complete/training",
        subset="clean",
        num_samples=1041,
        resize=(1024, 432),
        out_dir="farneback_multi_test"
    )
