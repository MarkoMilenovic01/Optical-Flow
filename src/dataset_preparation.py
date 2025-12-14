import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# --------------------------
# 1.1 — LOAD .FLO FILE
# --------------------------
def load_flo(path):
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {path}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = np.reshape(data, (h, w, 2))
        return flow.astype(np.float32)


# --------------------------
# 1.1 — DATASET CLASS
# --------------------------
class FlyingChairs(Dataset):
    def __init__(self, root, resize=None):
        """
        root   : path to FlyingChairs_release/data
        resize : tuple (new_w, new_h) or None

        If resize is not None, images and flows are resized and
        flow vectors are scaled accordingly.
        """
        self.root = root
        self.resize = resize

        self.img1_list = sorted([f for f in os.listdir(root) if "img1" in f])
        self.img2_list = [f.replace("img1", "img2") for f in self.img1_list]
        self.flow_list = [f.replace("img1.ppm", "flow.flo") for f in self.img1_list]

    def __len__(self):
        return len(self.img1_list)

    def _load_pair(self, idx):
        img1_path = os.path.join(self.root, self.img1_list[idx])
        img2_path = os.path.join(self.root, self.img2_list[idx])
        flow_path = os.path.join(self.root, self.flow_list[idx])

        img1 = np.array(Image.open(img1_path)).astype(np.float32) / 255.0  # (H, W, 3)
        img2 = np.array(Image.open(img2_path)).astype(np.float32) / 255.0  # (H, W, 3)
        flow = load_flo(flow_path)  # (H, W, 2)

        if self.resize is not None:
            new_w, new_h = self.resize
            H, W, _ = img1.shape

            # resize images
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # resize flow and scale vectors
            # OpenCV uses (width, height)
            flow = cv2.resize(flow, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_x = new_w / W
            scale_y = new_h / H
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y

        return img1, img2, flow

    def __getitem__(self, idx):
        img1, img2, flow = self._load_pair(idx)

        # stack images: (H, W, 6) -> (6, H, W)
        stacked = np.concatenate([img1, img2], axis=2).transpose(2, 0, 1)
        flow = flow.transpose(2, 0, 1)  # (2, H, W)

        # use from_numpy to avoid extra copy, ensure float32
        stacked_t = torch.from_numpy(stacked).float()
        flow_t = torch.from_numpy(flow).float()

        return stacked_t, flow_t


# --------------------------
# 1.1 — DATALOADERS
# --------------------------
def get_loader(root, batch_size=4, resize=None, num_workers=2):
    """
    Simple loader (no validation split), for debugging / quick tests.
    """
    dataset = FlyingChairs(root, resize=resize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers)


def get_train_val_loaders(
    root,
    batch_size=8,
    val_split=0.1,
    resize=None,
    num_workers=2,
    seed=42,
):
    """
    Returns train_loader, val_loader with random split.

    val_split : fraction of data used for validation (e.g. 0.1 = 10%).
    """
    dataset = FlyingChairs(root, resize=resize)
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # reproducible split
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# --------------------------
# 1.2 — FLOW → HSV VISUALIZATION
# --------------------------
def flow_to_hsv(flow):
    """
    flow : (H, W, 2) numpy array
    returns: RGB image visualizing flow
    """
    flow = flow.astype(np.float32)
    fx = flow[..., 0]
    fy = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(fx, fy)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    hsv[..., 0] = (angle * 180 / np.pi / 2)                     # Hue = angle
    hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255,
                                cv2.NORM_MINMAX)                # Saturation = magnitude
    hsv[..., 2] = 255                                           # Value = constant

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


# --------------------------
# 1.2 — VISUALIZATION DEMO
# --------------------------
def visualize_sample(root, resize=None, index=0):
    dataset = FlyingChairs(root, resize=resize)
    img, flow = dataset[index]

    img1 = img[:3].permute(1, 2, 0).numpy()
    img2 = img[3:].permute(1, 2, 0).numpy()

    flow_np = flow.permute(1, 2, 0).numpy()
    flow_hsv = flow_to_hsv(flow_np)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image 1")
    plt.imshow(img1)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Image 2")
    plt.imshow(img2)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Flow (HSV)")
    plt.imshow(flow_hsv)
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# --------------------------
# SAVE MULTIPLE VISUALIZATIONS
# --------------------------
def save_visualizations(root,
                        resize=None,
                        indices=None,
                        out_dir="visualizations"):
    """
    root      : dataset folder
    resize    : (w, h) or None
    indices   : list of sample indices to visualize, e.g. [0, 10, 42]
    out_dir   : output directory where images will be saved
    """

    os.makedirs(out_dir, exist_ok=True)

    dataset = FlyingChairs(root, resize=resize)

    if indices is None:
        # if None → visualize first 10 samples
        indices = list(range(10))

    print(f"Saving {len(indices)} samples to: {out_dir}")

    for idx in indices:
        img, flow = dataset[idx]

        img1 = img[:3].permute(1, 2, 0).numpy()
        img2 = img[3:].permute(1, 2, 0).numpy()

        flow_np = flow.permute(1, 2, 0).numpy()
        flow_hsv = flow_to_hsv(flow_np)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(img1)
        ax[0].set_title(f"Image 1 — idx={idx}")
        ax[0].axis("off")

        ax[1].imshow(img2)
        ax[1].set_title("Image 2")
        ax[1].axis("off")

        ax[2].imshow(flow_hsv)
        ax[2].set_title("Flow HSV")
        ax[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{idx}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()

    print("Done.")


# --------------------------
# RUN SIMPLE TEST
# --------------------------
if __name__ == "__main__":
    root = "data\FlyingChairs\FlyingChairs_release\data"

    # quick loader test
    loader = get_loader(root, batch_size=2, resize=None)
    for images, flows in loader:
        print("Batch images:", images.shape)  # (B, 6, H, W)
        print("Batch flows:", flows.shape)    # (B, 2, H, W)
        break

    save_visualizations(root, out_dir="vis_output")

