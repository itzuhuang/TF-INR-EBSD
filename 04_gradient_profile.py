import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.func import jacfwd, vmap
from pathlib import Path
from config import *
from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs(PROFILE_OUT_DIR, exist_ok=True)

AUTOGRAD_BATCH = 256

def build_grid(Ny, Nx):
    ys = torch.linspace(-1, 1, steps=Ny, device=device)
    xs = torch.linspace(-1, 1, steps=Nx, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

def compute_profile(model, Ny, Nx):
    xy_all = build_grid(Ny, Nx)
    Npts = Ny * Nx
    profile_flat = np.empty(Npts, dtype=np.float32)
    t0 = time.time()
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def coefficient_at_one_point(xy):
        return model.coeffs_xy(xy.unsqueeze(0)).squeeze(0)

    jacobian_at_one_point = jacfwd(coefficient_at_one_point)
    print("computing gradient: You think it's this easy? You think it's a simple answer?")

    for start in range(0, Npts, AUTOGRAD_BATCH):
        end = min(Npts, start + AUTOGRAD_BATCH)
        xy_b = xy_all[start:end]
        J = vmap(jacobian_at_one_point)(xy_b)
        S_b = torch.sqrt(torch.sum(J.square(), dim=(1, 2)))
        profile_flat[start:end] = S_b.detach().cpu().numpy().astype(np.float32)
        
        if (start // AUTOGRAD_BATCH) % 500 == 0:
            print(f"{end}/{Npts} ({time.time()-t0:.1f}s)")
        del J, S_b

    profile = profile_flat.reshape(Ny, Nx)
    print(f"gradient: min={profile.min():.4f}  mean={profile.mean():.4f}  max={profile.max():.4f} "
          "| It's not. It's never what you think it is.")

    return profile

def compute_s_at_point(model, x_norm, y_norm):
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    xy = torch.tensor(
        [x_norm, y_norm],
        device=device,
        dtype=torch.float32
    )

    def coefficient_at_one_point(z):
        return model.coeffs_xy(z.unsqueeze(0)).squeeze(0)
        
    J = jacfwd(coefficient_at_one_point)(xy)
    S = torch.sqrt(torch.sum(J.square()))
    return float(S.detach().cpu())

def save_pclip_png_svg(arr, label, out_dir, plow=PLOW, phigh=PHIGH, dpi=DPI, cmap=CMAP):
    Ny, Nx = arr.shape
    vmin = float(np.percentile(arr, plow))
    vmax = float(np.percentile(arr, phigh))
    fig, ax = plt.subplots(figsize=(Nx/100, Ny/100), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.imshow(
        arr,
        origin="upper",
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest"
    )
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out_dir = Path(out_dir)
    stem = f"{label}_gradient_profile"

    path_png = out_dir / f"{stem}.png"
    fig.savefig(
        path_png,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        dpi=dpi
    )
    print(f"PNG saved: {path_png}  (vmin={vmin:.4f} @ p{plow}, vmax={vmax:.4f} @ p{phigh})")

    path_svg = out_dir / f"{stem}.svg"
    fig.savefig(
        path_svg,
        format="svg",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True
    )
    print(f"SVG saved: {path_svg}")
    plt.close(fig)

for cfg in SMAP_DATASETS:
    label = cfg["label"]
    pth = cfg["pth"]
    ck = load_checkpoint(pth, device)
    model = ck["model"]
    Ny = ck["Ny_roi"]
    Nx = ck["Nx_roi"]
    Hp = ck["Hp"]
    Wp = ck["Wp"]
    print(f"loaded: Ny={Ny} Nx={Nx} Hp={Hp} Wp={Wp} R={model.R}")
    print(f"\n[{label}] Computing automatic gradient ...")

    profile = compute_profile(model, Ny, Nx)
    npy_path = Path(PROFILE_OUT_DIR) / f"{label}_profile.npy"
    np.save(npy_path, profile)
    print(f"NPY saved: {npy_path}")
    save_pclip_png_svg(profile, label, PROFILE_OUT_DIR)
    del model, profile, ck
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print(f"done. Saved in: {Path(PROFILE_OUT_DIR).resolve()}")
print("in the end, what matters?")
print("nothing, right?")
print("so you make it for yourself...you make your rigatoni pasta.")
print("="*70)
