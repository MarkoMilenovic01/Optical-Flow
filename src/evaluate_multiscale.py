import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

from model import FlowNetMultiOutput
from dataset_preparation import load_flo, flow_to_hsv


# =====================================================================
#                 SINTEL DATASET FOR MULTI-SCALE EVALUATION
# =====================================================================
class SintelDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset="clean", resize=None):
        self.resize = resize

        img_dir = os.path.join(root, subset)
        flow_dir = os.path.join(root, "flow")

        self.img1_paths = []
        self.img2_paths = []
        self.flow_paths = []

        scenes = sorted(os.listdir(img_dir))

        for scene in scenes:
            scene_img_dir = os.path.join(img_dir, scene)
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
        flow = load_flo(self.flow_paths[idx])

        if self.resize is not None:
            W, H = self.resize
            H0, W0, _ = img1.shape

            img1 = cv2.resize(img1, (W, H))
            img2 = cv2.resize(img2, (W, H))

            flow = cv2.resize(flow, (W, H))
            flow[..., 0] *= (W / W0)
            flow[..., 1] *= (H / H0)

        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float()
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).float()
        flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()

        imgs = torch.cat([img1, img2], dim=0)
        return imgs, flow


# =====================================================================
#                  MULTI-SCALE EPE (evaluation only)
# =====================================================================
def multi_scale_epe(preds, gt):
    total = 0.0
    for p in preds:
        _, _, h, w = p.shape
        gt_small = torch.nn.functional.interpolate(
            gt, size=(h, w), mode="bilinear", align_corners=False
        )
        e = torch.norm(p - gt_small, p=2, dim=1).mean().item()
        total += e
    return total / len(preds)


# =====================================================================
#                  MULTI-SCALE FlowNet EVALUATION
# =====================================================================
def evaluate_multi(
    model_path,
    root="data/MPI-Sintel-complete/training",
    subset="clean",
    num_samples=None,
    resize=None,
    out_dir="eval_multi",
    device=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n====================================")
    print(" Multi-Output FlowNet Evaluation")
    print("====================================\n")

    dataset = SintelDataset(root, subset=subset, resize=resize)
    total = len(dataset)
    if num_samples is None or num_samples > total:
        num_samples = total

    print(f"Subset     : {subset}")
    print(f"Resize     : {resize}")
    print(f"Samples    : {num_samples}/{total}")
    print(f"Model      : {model_path}")
    print(f"Device     : {device}\n")

    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = FlowNetMultiOutput().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_epe = []

    # ---------------- Evaluation Loop ----------------
    with torch.no_grad():
        for i in range(num_samples):

            imgs, gt_flow = dataset[i]
            imgs = imgs.unsqueeze(0).to(device)
            gt_flow = gt_flow.unsqueeze(0).to(device)

            preds = model(imgs)
            err = multi_scale_epe(preds, gt_flow)
            all_epe.append(err)

            # Build GT pyramid for all resolutions
            gt_pyr = []
            for p in preds:
                _, _, h, w = p.shape
                gt_resized = torch.nn.functional.interpolate(
                    gt_flow, size=(h, w), mode="bilinear", align_corners=False
                )
                gt_pyr.append(gt_resized)

            # Convert all to HSV
            img1_np = imgs[0, :3].cpu().permute(1, 2, 0).numpy()
            gt_hsv = [flow_to_hsv(gt_pyr[k][0].cpu().permute(1, 2, 0).numpy()) for k in range(4)]
            pr_hsv = [flow_to_hsv(preds[k][0].cpu().permute(1, 2, 0).numpy()) for k in range(4)]

            # ----------------------------------------------------------
            # FIXED LAYOUT (3 rows × 4 columns)
            # ----------------------------------------------------------
            plt.figure(figsize=(14, 18))  # Tall figure for vertical stacking

            # ---------------- Row 1: Full resolution ----------------
            plt.subplot(4, 3, 1)
            plt.imshow(img1_np)
            plt.title("Image 1")
            plt.axis("off")

            plt.subplot(4, 3, 2)
            plt.imshow(gt_hsv[0])
            plt.title("GT (H)")
            plt.axis("off")

            plt.subplot(4, 3, 3)
            plt.imshow(pr_hsv[0])
            plt.title(f"Pred (H)\nEPE={err:.3f}")
            plt.axis("off")

            # ---------------- Row 2: H/2 ----------------
            plt.subplot(4, 3, 4)
            plt.imshow(gt_hsv[1])
            plt.title("GT (H/2)")
            plt.axis("off")

            plt.subplot(4, 3, 5)
            plt.imshow(pr_hsv[1])
            plt.title("Pred (H/2)")
            plt.axis("off")

            # leave subplot(4,3,6) empty
            plt.subplot(4, 3, 6)
            plt.axis("off")

            # ---------------- Row 3: H/4 ----------------
            plt.subplot(4, 3, 7)
            plt.imshow(gt_hsv[2])
            plt.title("GT (H/4)")
            plt.axis("off")

            plt.subplot(4, 3, 8)
            plt.imshow(pr_hsv[2])
            plt.title("Pred (H/4)")
            plt.axis("off")

            plt.subplot(4, 3, 9)
            plt.axis("off")

            # ---------------- Row 4: H/8 ----------------
            plt.subplot(4, 3, 10)
            plt.imshow(gt_hsv[3])
            plt.title("GT (H/8)")
            plt.axis("off")

            plt.subplot(4, 3, 11)
            plt.imshow(pr_hsv[3])
            plt.title("Pred (H/8)")
            plt.axis("off")

            plt.subplot(4, 3, 12)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{subset}_{i:04d}.png"), dpi=120)
            plt.close()

            print(f"[{i:04d}]  EPE = {err:.3f}")

    # =====================================================================
    # FINAL STATS
    # =====================================================================
    all_epe = np.array(all_epe)
    mean_epe = float(all_epe.mean())
    std_epe = float(all_epe.std())

    print("\n=====================================")
    print(f" Mean EPE = {mean_epe:.3f}")
    print(f" Std  EPE = {std_epe:.3f}")
    print("=====================================\n")

    # Save summary file
    with open(os.path.join(out_dir, f"{subset}_summary.txt"), "w") as f:
        f.write(f"Subset       : {subset}\n")
        f.write(f"Model        : {model_path}\n")
        f.write(f"Samples      : {num_samples}\n")
        f.write(f"Mean EPE     : {mean_epe:.4f}\n")
        f.write(f"Std EPE      : {std_epe:.4f}\n")

    return mean_epe, std_epe



# =====================================================================
#                                MAIN
# =====================================================================
if __name__ == "__main__":
    ROOT = r"data\MPI-Sintel-complete\training"

    evaluate_multi(
        model_path="checkpoints/flownet_multi_best.pt",
        root=ROOT,
        subset="clean",
        num_samples=1041,
        resize=(1024, 432),
        out_dir="eval_multi_clean",
    )
