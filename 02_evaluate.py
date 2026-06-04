import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import kikuchipy as kp
from pathlib import Path

from config import *
from utils import (load_checkpoint, read_ang_xy,
                   make_gaussian_bandpass_window, bandpass_filter_float01,
                   savefig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def _save(name, kikuchi=False):
    savefig(FIG_DIR, name, kikuchi=kikuchi, dpi=DPI)

print("do you like rigatoni pasta?")

ck = load_checkpoint(SAVE_PTH, device)
model = ck["model"]
scaler = ck["scaler"]
mask_uv = ck["mask_uv"]
Hp, Wp = ck["Hp"], ck["Wp"]
Ny_roi = ck["Ny_roi"]; Nx_roi = ck["Nx_roi"]
print(f"R={model.R}  Ny={Ny_roi}  Nx={Nx_roi}  best_val={ck['best_val']:.6f}")

ys_scan = torch.linspace(-1, 1, steps=Ny_roi, device=device)
xs_scan = torch.linspace(-1, 1, steps=Nx_roi, device=device)
print("loading up2:uh yeah. I I I love it. Why? do you know where we can get some?")

s = kp.load(UP2_PATH, lazy=True)
N_up2 = int(s.data.shape[0])

x_coords, y_coords = read_ang_xy(ANG_PATH)
if len(x_coords) > N_up2:
    x_coords = x_coords[:N_up2]; y_coords = y_coords[:N_up2]
g_w = int(np.max(x_coords)) + 1

x0, x1 = ROI_SCAN["x0"], ROI_SCAN["x1"]
y0, y1 = ROI_SCAN["y0"], ROI_SCAN["y1"]
xs_np = np.arange(x0, x1, STRIDE_SCAN, dtype=np.int64)
ys_np = np.arange(y0, y1, STRIDE_SCAN, dtype=np.int64)
idx_roi = ((ys_np * g_w)[:, None] + xs_np[None, :]).reshape(-1)
idx_roi = idx_roi[idx_roi < N_up2]

roi_flat = np.array(s.data[idx_roi].compute()).astype(np.float32)
roi_clean = roi_flat.reshape(Ny_roi, Nx_roi, Hp, Wp)
del roi_flat

if scaler:
    dmin = scaler.data_min; dmax = scaler.data_max
else:
    dmin = float(roi_clean.min()); dmax = float(roi_clean.max())

print("\nGT vs Recon kikuchi plotting: I want your honest opinion.")
cmap_v = plt.cm.viridis.copy(); cmap_v.set_bad(alpha=0.0)

test_pts = [(0,0), (Ny_roi//2, Nx_roi//2), (Ny_roi-1, Nx_roi-1)]
for (iy, ix) in test_pts:
    with torch.no_grad():
        xy = torch.stack([xs_scan[ix], ys_scan[iy]]).unsqueeze(0).float()
        a = model.coeffs_xy(xy)[0]
        pn = model.reconstruct_pattern_from_coeffs(a).cpu().numpy()
    pred = scaler.denormalize(pn) if scaler else pn
    gt = roi_clean[iy, ix].copy()
    err = np.abs(pred - gt)
    if USE_MASK:
        gt[~mask_uv]=np.nan; pred[~mask_uv]=np.nan; err[~mask_uv]=np.nan
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), dpi=600)
    for a_, img_, ttl in zip(ax, [gt, pred, err],
                              [f"GT (iy={iy},ix={ix})", "Pred (INR)", "Abs Error"]):
        im = a_.imshow(img_, cmap=cmap_v); a_.set_title(ttl); a_.axis("off")
        fig.colorbar(im, ax=a_, fraction=0.046)
    plt.tight_layout()
    _save(f"recon_iy{iy}_ix{ix}.png", kikuchi=True)

print("\nComputing MAE map: No, I.. I do. I really do. I like it.")
mask_uv_t = torch.from_numpy(mask_uv.astype(np.bool_)).to(device)
bias_t_ = torch.tensor(float(scaler.bias), device=device, dtype=torch.float32) \
             if scaler else None
scale_t_ = torch.tensor(float(scaler.scale), device=device, dtype=torch.float32) \
             if scaler else None

grid_y, grid_x = torch.meshgrid(ys_scan, xs_scan, indexing="ij")
xy_all = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
Npts = xy_all.shape[0]
mae_flat = np.empty((Npts,), dtype=np.float32)
BATCH_MAP = 128

with torch.no_grad():
    for start in range(0, Npts, BATCH_MAP):
        end = min(start+BATCH_MAP, Npts)
        xy_b = xy_all[start:end].to(device).float()
        a = model.coeffs_xy(xy_b)
        pn_b = torch.einsum("br,ruv->buv", a, model.B)
        pred_b = (pn_b - bias_t_) / scale_t_ if scale_t_ is not None else pn_b
        flat_idx = np.arange(start, end, dtype=np.int64)
        iy_b2 = flat_idx // Nx_roi
        ix_b2 = flat_idx % Nx_roi
        gt_b = torch.from_numpy(roi_clean[iy_b2, ix_b2].astype(np.float32)).to(device)
        err_b = torch.abs(pred_b - gt_b)
        mae_flat[start:end] = err_b[:, mask_uv_t].mean(dim=1).detach().cpu().numpy()

mae_map = mae_flat.reshape(Ny_roi, Nx_roi)
print(f"MAE: min={mae_map.min():.2f} | mean={mae_map.mean():.2f} | max={mae_map.max():.2f}")

fig, ax = plt.subplots(figsize=(8, 7), dpi=600)
im = ax.imshow(mae_map, origin="upper", cmap="YlOrRd", aspect="equal")
ax.set_title(f"MAE map (stride={STRIDE_SCAN})", fontsize=13)
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAE")
plt.tight_layout()
_save("MAE_map.png", kikuchi=False)

intensity_range = float(dmax - dmin)
mae_norm_map = mae_map / (intensity_range + 1e-12)
fig, ax = plt.subplots(figsize=(8, 7), dpi=600)
im = ax.imshow(mae_norm_map * 100, origin="upper", cmap="YlOrRd", aspect="equal")
ax.set_title("Reconstruction MAE map (% of intensity range)", fontsize=13)
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAE (%)")
plt.tight_layout()
_save("MAE_map_normalized.png", kikuchi=False)

print("\nSaving basis images: I've been meaning to make some rigatoni pasta.")
cmap_basis = plt.cm.Spectral_r.copy(); cmap_basis.set_bad(alpha=0.0)
with torch.no_grad():
    B_np = model.B.detach().cpu().numpy()
R_show = min(12, B_np.shape[0])
plt.figure(figsize=(12, 8), dpi=200)
for i in range(R_show):
    img = B_np[i].copy()
    if USE_MASK: img[~mask_uv] = np.nan
    plt.subplot(3, 4, i+1); plt.imshow(img, cmap=cmap_basis)
    plt.title(f"Basis #{i}"); plt.axis("off")
plt.tight_layout()
_save("Basis_images.png", kikuchi=True)

Wbp = make_gaussian_bandpass_window(Hp, Wp, SIGMA_LOW, SIGMA_HIGH)

show_pts = [(0,0),(0,Nx_roi-1),(Ny_roi-1,0),(Ny_roi-1,Nx_roi-1),(Ny_roi//2,Nx_roi//2)]
recon_show = []
with torch.no_grad():
    for (iy, ix) in show_pts:
        xy = torch.stack([xs_scan[ix], ys_scan[iy]]).unsqueeze(0).float()
        a = model.coeffs_xy(xy)[0]
        pn = model.reconstruct_pattern_from_coeffs(a).cpu().numpy()
        recon_show.append(scaler.denormalize(pn) if scaler else pn)
recon_show = np.stack(recon_show, axis=0).astype(np.float32)
recon_bp_show = bandpass_filter_float01(recon_show, Wbp)
gt_show = np.stack([roi_clean[iy, ix].astype(np.float32) for iy,ix in show_pts])
gt_bp_show = bandpass_filter_float01(gt_show, Wbp)

cmap_gray2 = plt.cm.gray.copy(); cmap_gray2.set_bad(alpha=0.0)
fig, axes = plt.subplots(len(show_pts), 4, figsize=(16, 16), dpi=200)
for r, (iy, ix) in enumerate(show_pts):
    axes[r,0].imshow(gt_show[r], cmap=cmap_gray2); axes[r,0].set_title(f"GT raw ({iy},{ix})"); axes[r,0].axis("off")
    axes[r,1].imshow(gt_bp_show[r], cmap=cmap_gray2, vmin=0, vmax=1); axes[r,1].set_title("GT bandpass"); axes[r,1].axis("off")
    axes[r,2].imshow(recon_show[r], cmap=cmap_gray2); axes[r,2].set_title("Recon raw"); axes[r,2].axis("off")
    axes[r,3].imshow(recon_bp_show[r],cmap=cmap_gray2, vmin=0, vmax=1); axes[r,3].set_title("Recon bandpass"); axes[r,3].axis("off")
plt.tight_layout()
_save("bandpass_comparison.png", kikuchi=True)

print("\n" + "="*100)
print(f"figures are here: {Path(FIG_DIR).resolve()}")
print("That.. that that's great. Why don't we go right now? Why don't we go right now and you could go make some rigatoni pasta?")
print("="*100)