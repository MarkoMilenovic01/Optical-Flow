import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio


from model import FlowNetSingleOutput
from dataset_preparation import flow_to_hsv


# ================================================================
# 1. Extract frames from video
# ================================================================
def extract_frames(video_path, out_dir="video_frames"):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = frame[:, :, ::-1]  # BGR → RGB
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:05d}.png"),
                    frame_rgb[:, :, ::-1])  # save back as BGR PNG
        print("Input frame:", frame.shape)

        idx += 1

    cap.release()
    print(f"✓ Extracted {idx} frames → {out_dir}")
    return out_dir, idx


# ================================================================
# 2. Preprocess frames for FlowNet
# ================================================================
def preprocess_pair(img1, img2, resize=None):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    if resize is not None:
        W, H = resize
        img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_LINEAR)

    t1 = torch.from_numpy(img1.transpose(2, 0, 1)).float()
    t2 = torch.from_numpy(img2.transpose(2, 0, 1)).float()
    stacked = torch.cat([t1, t2], dim=0)   # (6, H, W)

    return stacked.unsqueeze(0)             # (1, 6, H, W)


# ================================================================
# 3. Run FlowNetSingleOutput on frame pairs
# ================================================================
def run_flownet(model, img1, img2, device, resize=None):
    inp = preprocess_pair(img1, img2, resize=resize).to(device)

    with torch.no_grad():
        flow = model(inp)[0]    # (2, H, W)

    return flow.cpu().permute(1, 2, 0).numpy()


# ================================================================
# 4. Run Farnebäck
# ================================================================
def run_farneback(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5,
        levels=5,
        winsize=25,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    return flow


# ================================================================
# 5. Process video using both models
# ================================================================
def process_video(
    video_path,
    flownet_model_path="checkpoints/flownet_single_best.pt",
    resize=(512, 256),
    out_dir="video_results",
    device=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Load video frames ----------------
    frames_dir, n_frames = extract_frames(video_path)

    # ---------------- Load FlowNet ---------------------
    print("Loading FlowNet model...")
    model = FlowNetSingleOutput().to(device)
    ckpt = torch.load(flownet_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Storage for GIF frames
    gif_flownet = []
    gif_farneback = []
    gif_side_by_side = []

    # ---------------- Process pairs -------------------
    for i in range(n_frames - 1):
        img1 = cv2.imread(os.path.join(frames_dir, f"frame_{i:05d}.png"))[:, :, ::-1]
        img2 = cv2.imread(os.path.join(frames_dir, f"frame_{i+1:05d}.png"))[:, :, ::-1]

        # === FlowNet output ===
        flow_dl = run_flownet(model, img1, img2, device, resize=resize)
        hsv_dl = flow_to_hsv(flow_dl)

        # === Farneback output ===
        flow_fb = run_farneback(img1, img2)
        hsv_fb = flow_to_hsv(flow_fb)

        # ---------------------------------------------------------------------
        # Store frames for GIFs
        # ---------------------------------------------------------------------
        gif_flownet.append((hsv_dl * 255).astype(np.uint8))
        gif_farneback.append((hsv_fb * 255).astype(np.uint8))

        # Side-by-side stacked horizontally
        side = np.hstack([hsv_dl, hsv_fb])
        gif_side_by_side.append((side * 255).astype(np.uint8))

        # === Save visualization ===
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(img1)
        plt.title("Frame")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(hsv_dl)
        plt.title("FlowNet Optical Flow")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(hsv_fb)
        plt.title("Farneback Optical Flow")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"flow_{i:05d}.png"), dpi=120)
        plt.close()

        print(f"Processed frame pair {i}/{n_frames-2}")

    # -------------------------------------------------------------------------
    # Create GIFs
    # -------------------------------------------------------------------------
    print("\nCreating GIF animations...")

    imageio.mimsave(os.path.join(out_dir, "flownet.gif"), gif_flownet, fps=10)
    imageio.mimsave(os.path.join(out_dir, "farneback.gif"), gif_farneback, fps=10)
    imageio.mimsave(os.path.join(out_dir, "comparison.gif"), gif_side_by_side, fps=10)

    print("✓ GIFs saved!")
    print(f"    {out_dir}/flownet.gif")
    print(f"    {out_dir}/farneback.gif")
    print(f"    {out_dir}/comparison.gif")

    print("\n✓ Finished! All results saved to:", out_dir)



# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    process_video(
        video_path="video.mp4",
        flownet_model_path="checkpoints/flownet_single_best.pt",
        resize=(576,1024),
        out_dir="video_flow_output"
    )
