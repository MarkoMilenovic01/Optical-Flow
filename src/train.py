import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os

from model import FlowNetSingleOutput, FlowNetMultiOutput
from dataset_preparation import get_train_val_loaders


# --------------------------------------------------
# EPE LOSS
# --------------------------------------------------
def epe_loss(pred, gt):
    return torch.norm(pred - gt, p=2, dim=1).mean()


# --------------------------------------------------
# MULTI-SCALE LOSS
# --------------------------------------------------
def multi_scale_loss(preds, gt):
    total = 0.0
    for flow in preds:
        _, _, h, w = flow.shape
        gt_resized = F.interpolate(gt, size=(h, w), mode="bilinear", align_corners=False)
        total += epe_loss(flow, gt_resized)
    return total / len(preds)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
def validate(model, val_loader, device, multi_output=False):
    model.eval()
    total = 0
    n = 0

    with torch.no_grad():
        for imgs, flows in val_loader:
            imgs = imgs.to(device)
            flows = flows.to(device)

            preds = model(imgs)
            loss = multi_scale_loss(preds, flows) if multi_output else epe_loss(preds, flows)

            total += loss.item()
            n += 1

    model.train()
    return total / n


# --------------------------------------------------
# SAVE CHECKPOINT
# --------------------------------------------------
def save_checkpoint(model, optimizer, step, val_loss, path):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint → {path}")


# --------------------------------------------------
# TRAIN FUNCTION (AMP ENABLED)
# --------------------------------------------------
def train_optical_flow(
    model_type="single",
    root="data/FlyingChairs/FlyingChairs_release/data",
    resize=None,
    batch_size=8,
    total_steps=50_000,
    lr=1e-4,
    val_split=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    print("\n=========================================")
    print(f" TRAINING MODEL: {model_type.upper()}-OUTPUT (AMP ENABLED) ")
    print("===========================================\n")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    train_loader, val_loader = get_train_val_loaders(
        root, batch_size=batch_size, val_split=val_split, resize=resize
    )
    train_iter = iter(train_loader)

    # -----------------------------
    # MODEL SELECTION
    # -----------------------------
    if model_type == "single":
        model = FlowNetSingleOutput().to(device)
        multi_output = False
    else:
        model = FlowNetMultiOutput().to(device)
        multi_output = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_steps = []

    # -----------------------------
    # AMP SCALER
    # -----------------------------
    scaler = torch.amp.GradScaler("cuda")

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    step = 0
    start_time = time.time()
    last_print_time = start_time  # for per-10-iteration timing

    while step < total_steps:

        try:
            imgs, flows = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            imgs, flows = next(train_iter)

        imgs = imgs.to(device)
        flows = flows.to(device)

        # -----------------------------
        # AMP FORWARD + BACKWARD
        # -----------------------------
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            preds = model(imgs)
            loss = multi_scale_loss(preds, flows) if multi_output else epe_loss(preds, flows)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

        # -----------------------------
        # PRINT PROGRESS EVERY 10 STEPS
        # -----------------------------
        if step % 10 == 0:
            now = time.time()
            iter_time = now - last_print_time
            last_print_time = now

            print(
                f"STEP {step:6d}/{total_steps} "
                f"| loss={loss.item():.5f} "
                f"| 10-iter-time={iter_time:.3f}s"
            )

        # -----------------------------
        # VALIDATION + CHECKPOINT EVERY 1000 STEPS
        # -----------------------------
        if step % 1000 == 0:
            val_loss = validate(model, val_loader, device, multi_output)
            val_losses.append(val_loss)
            val_steps.append(step)

            print(f"[VAL] step={step:6d} | train={loss.item():.5f} | val={val_loss:.5f}")

            save_checkpoint(
                model, optimizer, step, val_loss,
                f"checkpoints/flownet_{model_type}_last.pt"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, step, val_loss,
                    f"checkpoints/flownet_{model_type}_best.pt"
                )
                print("✓ New BEST model saved!")

        # -----------------------------
        # LR DECAY SCHEDULE (40%, 60%, 80%)
        # -----------------------------
        decay_steps = [
            int(total_steps * 0.4),
            int(total_steps * 0.6),
            int(total_steps * 0.8),
        ]
        if step in decay_steps:
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            print("LR decayed →", optimizer.param_groups[0]["lr"])

        step += 1

    print("\nTraining complete.\n")

    # -------------------------------------------------
    # SAVE TRAINING CURVES
    # -------------------------------------------------
    plt.figure(figsize=(11, 5))

    N = 50
    train_steps = list(range(0, len(train_losses), N))
    train_downsampled = [train_losses[i] for i in train_steps]

    plt.plot(train_steps, train_downsampled, label="Train Loss", linewidth=1.5)
    plt.plot(val_steps, val_losses, label="Validation Loss", linewidth=3, marker="o")

    plt.xlabel("Training Steps")
    plt.ylabel("EPE Loss")
    plt.title(f"FlowNet Training Curve ({model_type}-output, AMP)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = f"curve_{model_type}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved training curve → {out_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    ROOT = r"data\FlyingChairs\FlyingChairs_release\data"

    # train_optical_flow(
    #     model_type="single",
    #     root=ROOT,
    #     batch_size=8,
    #     total_steps=20_000,
    #     resize=(256, 256),
    # )

    train_optical_flow(
        model_type="multi",
        root=ROOT,
        batch_size=8,
        total_steps=20_000,
        resize=(256, 256),
    )
