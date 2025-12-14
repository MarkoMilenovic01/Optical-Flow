import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

from model import FlowNetSingleOutput
from dataset_preparation import load_flo, flow_to_hsv


# =====================================================================
#                 SINTEL DATASET (single-output evaluation)
# =====================================================================
class SintelDataset(torch.utils.data.Dataset):
    """
    Dataset for MPI-Sintel optical flow evaluation.
    Supports resizing and correct flow scaling.
    """
    def __init__(self, root, subset="clean", resize=None):
        """
        root   : path to MPI-Sintel-complete/training
        subset : "clean" or "final"
        resize : (W, H) or None — must be divisible by 8 for U-Net
        """
        self.resize = resize

        subset_dir = os.path.join(root, subset)
        flow_dir   = os.path.join(root, "flow")

        self.img1_paths = []
        self.img2_paths = []
        self.flow_paths = []

        scenes = sorted(os.listdir(subset_dir))

        for scene in scenes:
            scene_img_dir  = os.path.join(subset_dir, scene)
            scene_flow_dir = os.path.join(flow_dir, scene)

            frames = sorted([f for f in os.listdir(scene_img_dir) if f.endswith(".png")])

            for i in range(len(frames) - 1):
                f1 = frames[i]
                f2 = frames[i + 1]
                flo = f"frame_{i+1:04d}.flo"

                self.img1_paths.append(os.path.join(scene_img_dir, f1))
                self.img2_paths.append(os.path.join(scene_img_dir, f2))
                self.flow_paths.append(os.path.join(scene_flow_dir, flo))

    def __len__(self):
        return len(self.img1_paths)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.img1_paths[idx])[:, :, ::-1] / 255.0
        img2 = cv2.imread(self.img2_paths[idx])[:, :, ::-1] / 255.0
        flow = load_flo(self.flow_paths[idx])  # (H, W, 2)

        # ---------------------- Resize + Scale Flow ----------------------
        if self.resize is not None:
            W, H = self.resize
            H0, W0, _ = img1.shape

            img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_LINEAR)

            flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= (W / W0)
            flow[..., 1] *= (H / H0)

        # ---------------------- Convert to tensors -----------------------
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float()
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).float()
        flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()

        imgs = torch.cat([img1, img2], dim=0)  # [6, H, W]
        return imgs, flow


# =====================================================================
#                     Endpoint Error (EPE)
# =====================================================================
def epe(pred, gt):
    """
    pred, gt  : [1, 2, H, W]
    Returns mean endpoint error (float).
    """
    return torch.norm(pred - gt, p=2, dim=1).mean().item()


# =====================================================================
#                   Evaluate FlowNetSingleOutput
# =====================================================================

def evaluate_single(
    model_path,
    root="data/MPI-Sintel-complete/training",
    subset="clean",
    num_samples=None,
    resize=None,
    out_dir="eval_single",
    device=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n===============================")
    print(" Single-Output Model Evaluation")
    print("===============================\n")

    # ---------------------- Dataset ----------------------
    dataset = SintelDataset(root, subset=subset, resize=resize)
    total = len(dataset)

    if num_samples is None or num_samples > total:
        num_samples = total

    print(f"Subset       : {subset}")
    print(f"Resize       : {resize}")
    print(f"Evaluate N   : {num_samples}/{total}")
    print(f"Model        : {model_path}")
    print(f"Device       : {device}\n")

    os.makedirs(out_dir, exist_ok=True)

    # ---------------------- Load Model ----------------------
    model = FlowNetSingleOutput().to(device)
    state = torch.load(model_path)
    model.load_state_dict(state["model_state"])
    model.eval()

    epe_list = []      # <-- collect all errors here

    # ---------------------- Evaluation Loop ----------------------
    with torch.no_grad():
        for i in range(num_samples):

            imgs, gt_flow = dataset[i]
            imgs = imgs.unsqueeze(0).to(device)
            gt_flow = gt_flow.unsqueeze(0).to(device)

            pred_flow = model(imgs)
            err = epe(pred_flow, gt_flow)

            epe_list.append(err)     # store error

            # ----------------- Visualization -----------------
            pred_np = pred_flow[0].cpu().permute(1, 2, 0).numpy()
            gt_np   = gt_flow[0].cpu().permute(1, 2, 0).numpy()
            img1    = imgs[0, :3].cpu().permute(1, 2, 0).numpy()

            pred_hsv = flow_to_hsv(pred_np)
            gt_hsv   = flow_to_hsv(gt_np)

            plt.figure(figsize=(14, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(img1)
            plt.title("Image 1")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_hsv)
            plt.title("Ground Truth Flow")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_hsv)
            plt.title(f"Predicted Flow (EPE={err:.3f})")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{subset}_{i:04d}.png"), dpi=120)
            plt.close()

            print(f"[{i:04d}] EPE = {err:.3f}")

    # ---------------------- Statistics ----------------------
    epe_array = np.array(epe_list)
    mean_epe = float(epe_array.mean())
    std_epe  = float(epe_array.std())

    print("\n=====================================")
    print(f"  EPE Mean ({subset}) = {mean_epe:.4f}")
    print(f"  EPE Std  ({subset}) = {std_epe:.4f}")
    print("=====================================\n")

    # Save summary
    with open(os.path.join(out_dir, f"{subset}_summary.txt"), "w") as f:
        f.write(f"Subset     : {subset}\n")
        f.write(f"Model      : {model_path}\n")
        f.write(f"Samples    : {num_samples}\n")
        f.write(f"Mean EPE   : {mean_epe:.4f}\n")
        f.write(f"Std  EPE   : {std_epe:.4f}\n")

    return mean_epe, std_epe



# =====================================================================
#                                MAIN
# =====================================================================
if __name__ == "__main__":

    ROOT = r"data\MPI-Sintel-complete\training"

    evaluate_single(
        model_path="checkpoints/flownet_single_best.pt",
        root=ROOT,
        subset="clean",
        num_samples=1041,
        resize=(1024, 432),      # must be divisible by 8
        out_dir="eval_single_clean",
    )
