
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path
from config import *
from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs(PROFILE_OUT_DIR, exist_ok=True)

def build_grid(Ny, Nx):
    ys = torch.linspace(-1, 1, steps=Ny, device=device)
    xs = torch.linspace(-1, 1, steps=Nx, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

def compute_profile(model, Ny, Nx):
    xy_all = build_grid(Ny, Nx)
    Npts = Ny * Nx
    all_a = []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, Npts, COEFF_BATCH):
            end = min(Npts, start + COEFF_BATCH)
            xy_b = xy_all[start:end].float()
            a = model.coeffs_xy(xy_b)
            all_a.append(a.cpu().numpy().astype(np.float32))
            if (start // COEFF_BATCH) % 500 == 0:
                print(f"{end}/{Npts} ({time.time()-t0:.1f}s)")
    A = np.concatenate(all_a, axis=0).reshape(Ny, Nx, -1)
    print(f"computing gradient: You think it's this easy? You think it's a simple answer?")
    gy = np.gradient(A, axis=0)
    gx = np.gradient(A, axis=1)
    profile = np.sqrt((gy**2 + gx**2).sum(axis=2)).astype(np.float32)
    print(f"gradient: min={profile.min():.4f}  mean={profile.mean():.4f}  max={profile.max():.4f} "
           "| It's not. It's never what you think it is.")
    return profile

def save_pclip_png_svg(arr, label, out_dir, plow=PLOW, phigh=PHIGH, dpi=DPI, cmap=CMAP):
    Ny, Nx = arr.shape
    vmin = float(np.percentile(arr, plow))
    vmax = float(np.percentile(arr, phigh))
    fig, ax = plt.subplots(figsize=(Nx/100, Ny/100), dpi=dpi)
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ax.imshow(arr, origin="upper", cmap=cmap,
              norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
              interpolation="nearest")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_dir = Path(out_dir)
    stem = f"{label}_gradient_profile"
    path_png = out_dir / f"{stem}.png"
    fig.savefig(path_png, format="png", bbox_inches="tight",
                pad_inches=0, transparent=True, dpi=dpi)
    print(f"PNG saved: {path_png}  (vmin={vmin:.4f} @ p{plow}, vmax={vmax:.4f} @ p{phigh})")
    path_svg = out_dir / f"{stem}.svg"
    fig.savefig(path_svg, format="svg", bbox_inches="tight",
                pad_inches=0, transparent=True)
    print(f"SVG saved: {path_svg}")
    plt.close(fig)

for cfg in SMAP_DATASETS:
    label = cfg["label"]
    pth = cfg["pth"]
    ck = load_checkpoint(pth, device)
    model = ck["model"]
    Ny = ck["Ny_roi"]; Nx = ck["Nx_roi"]
    Hp = ck["Hp"]; Wp = ck["Wp"]
    print(f"loaded: Ny={Ny} Nx={Nx} Hp={Hp} Wp={Wp} R={model.R}")
    print(f"\n[{label}] Computing numerical gradient ...")
    profile = compute_profile(model, Ny, Nx)
    npy_path = Path(PROFILE_OUT_DIR) / f"{label}_profile.npy"
    np.save(npy_path, profile)
    print(f"NPY saved: {npy_path}")
    save_pclip_png_svg(profile, label, PROFILE_OUT_DIR)
    del model, profile, ck
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print(f"profile done. Outputs in: {Path(PROFILE_OUT_DIR).resolve()}")
print("in the end, what matters?")
print("nothing, right?")
print("so you make it for yourself...you make your rigatoni pasta.")
print("="*70)
