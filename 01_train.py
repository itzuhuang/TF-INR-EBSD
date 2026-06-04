import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import kikuchipy as kp
from pathlib import Path
from scipy.io import savemat

from config import *
from utils import (SineLayer, SirenXY, TensorFactorizedINR,
                   DataScaler, read_ang_xy, savefig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print("device:", device)

FIG_DIR_P = Path(FIG_DIR)
FIG_DIR_P.mkdir(parents=True, exist_ok=True)

def _save(name, kikuchi=False):
    savefig(FIG_DIR, name, kikuchi=kikuchi, dpi=DPI)

print("\n" + "="*60)
print("Loading data: is anyone out there?...")
print("="*60)

s = kp.load(UP2_PATH, lazy=True)
N_up2 = int(s.data.shape[0])
print(f"[UP2] N = {N_up2}")

x_coords, y_coords = read_ang_xy(ANG_PATH)
if len(x_coords) > N_up2:
    x_coords = x_coords[:N_up2]
    y_coords = y_coords[:N_up2]

g_w = int(np.max(x_coords)) + 1
g_h = int(np.max(y_coords)) + 1
print(f"grid: width={g_w}, height={g_h}")

x0, x1 = ROI_SCAN["x0"], ROI_SCAN["x1"]
y0, y1 = ROI_SCAN["y0"], ROI_SCAN["y1"]
xs_train = np.arange(x0, x1, STRIDE_SCAN, dtype=np.int64)
ys_train = np.arange(y0, y1, STRIDE_SCAN, dtype=np.int64)
xs_index = np.arange(x0, x1, STRIDE_INDEX, dtype=np.int64)
ys_index = np.arange(y0, y1, STRIDE_INDEX, dtype=np.int64)
Ny_roi, Nx_roi = len(ys_train), len(xs_train)
Ny_idx, Nx_idx = len(ys_index), len(xs_index)

idx_roi = ((ys_train * g_w)[:, None] + xs_train[None, :]).reshape(-1)
idx_roi = idx_roi[idx_roi < N_up2]
print(f"Train grid: ({Ny_roi}, {Nx_roi}) = {Ny_roi*Nx_roi} patterns")
print(f"Index grid: ({Ny_idx}, {Nx_idx}) = {Ny_idx*Nx_idx} patterns")

tmp0 = np.array(s.data[0].compute())
Hp, Wp = tmp0.shape
print(f"pattern size: {Hp} x {Wp}")

mem_gb = Ny_roi * Nx_roi * Hp * Wp * 4 / 1e9
print(f"est. RAM for roi array: {mem_gb:.1f} GB ")

roi_flat = np.array(s.data[idx_roi].compute()).astype(np.float32)
roi = roi_flat.reshape(Ny_roi, Nx_roi, Hp, Wp)
del roi_flat
print(f"ROI shape: {roi.shape}")

yy, xx = np.mgrid[0:Hp, 0:Wp]
cy, cx = (Hp-1)/2.0, (Wp-1)/2.0
mask_uv = ((yy-cy)**2 + (xx-cx)**2 <= (min(Hp,Wp)*0.48)**2)

roi_clean = roi.copy()

if USE_NORMALIZATION:
    valid_vals = roi_clean[..., mask_uv] if USE_MASK else roi_clean
    dmin, dmax = float(valid_vals.min()), float(valid_vals.max())
    scaler = DataScaler(dmin, dmax, target_min=-1, target_max=1)
    roi_norm = scaler.normalize(roi_clean).astype(np.float32)
    print(f"Normalized [{dmin:.1f}, {dmax:.1f}] -> [-1, 1]")
else:
    scaler = None
    roi_norm = roi_clean.astype(np.float32)

ys_scan = torch.linspace(-1, 1, steps=Ny_roi, device=device)
xs_scan = torch.linspace(-1, 1, steps=Nx_roi, device=device)
ys_scan_idx = torch.linspace(-1, 1, steps=Ny_idx, device=device)
xs_scan_idx = torch.linspace(-1, 1, steps=Nx_idx, device=device)
us_uv = torch.linspace(-1, 1, steps=Hp, device=device)
vs_uv = torch.linspace(-1, 1, steps=Wp, device=device)

if USE_MASK:
    uv = np.stack(np.where(mask_uv), axis=1).astype(np.int64)
    uv = uv[(uv[:,0]>=1) & (uv[:,0]<=Hp-2) & (uv[:,1]>=1) & (uv[:,1]<=Wp-2)]
    uv_idx_t = torch.tensor(uv, dtype=torch.long, device=device)
    print(f"n_mask_pixels = {int(uv_idx_t.shape[0])}")
else:
    uv_idx_t = None

show_pts = [(0,0),(0,Nx_roi-1),(Ny_roi-1,0),(Ny_roi-1,Nx_roi-1),(Ny_roi//2,Nx_roi//2)]
cmap_gray = plt.cm.gray.copy(); cmap_gray.set_bad(alpha=0.0)

plt.figure(figsize=(10, 14), dpi=200)
for k, (iy, ix) in enumerate(show_pts, 1):
    raw_show = roi[iy, ix].copy()
    if USE_MASK: raw_show[~mask_uv] = np.nan
    plt.subplot(len(show_pts), 1, k)
    plt.title(f"RAW (iy,ix)=({iy},{ix})")
    plt.imshow(raw_show, cmap=cmap_gray); plt.axis("off")
plt.tight_layout()
_save("raw_patterns.png", kikuchi=True)

print("\n" + "="*60)
print("PCA initialization: i'm starving, i'm so dehydrated...")
print("="*60)

Nmap = Ny_roi * Nx_roi
Nsamp = int(min(PCA_MAX_SAMPLES, Nmap))
print(f"Nmap={Nmap}, PCA sampling {Nsamp} ({100*Nsamp/Nmap:.1f}%)")

rng = np.random.default_rng(PCA_SEED)
flat = rng.choice(Nmap, size=Nsamp, replace=False)
iy_s = (flat // Nx_roi).astype(np.int64)
ix_s = (flat %  Nx_roi).astype(np.int64)
Ximg = roi_norm[iy_s, ix_s].astype(np.float32)

if USE_MASK:
    X = Ximg[:, mask_uv].reshape(Nsamp, -1)
    mask_idx = np.where(mask_uv.reshape(-1))[0].astype(np.int64)
    mask_idx_t = torch.tensor(mask_idx, dtype=torch.long)
else:
    X = Ximg.reshape(Nsamp, -1)
    mask_idx_t = None

X_t = torch.from_numpy(X).float()
mu = X_t.mean(dim=0, keepdim=True)
Xc = X_t - mu
total_var = Xc.var(dim=0, unbiased=True).sum().clamp(min=1e-12)
q = int(min(PCA_MAX_R, Xc.shape[0]-1, Xc.shape[1]))
U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
ev = (S**2) / float(max(1, Nsamp-1))
cum = torch.cumsum(ev, dim=0) / total_var
hit = (cum >= PCA_TARGET_VAR).nonzero(as_tuple=False)
k_pca = int(hit[0].item())+1 if hit.numel() > 0 else int(cum.numel())
R_PCA = 1 + k_pca

PCA_B_INIT = torch.zeros((R_PCA, Hp, Wp), dtype=torch.float32)
if USE_MASK:
    mf = torch.zeros((Hp*Wp,), dtype=torch.float32)
    mf[mask_idx_t] = mu[0]; PCA_B_INIT[0] = mf.view(Hp, Wp)
else:
    PCA_B_INIT[0] = mu[0].view(Hp, Wp)
for i in range(min(k_pca, V.shape[1])):
    vec = V[:, i].float()
    if USE_MASK:
        cf = torch.zeros((Hp*Wp,), dtype=torch.float32)
        cf[mask_idx_t] = vec; PCA_B_INIT[i+1] = cf.view(Hp, Wp)
    else:
        PCA_B_INIT[i+1] = vec.view(Hp, Wp)
PCA_B_INIT = PCA_B_INIT.to(device)

print(f"k={k_pca} -> R_PCA={R_PCA} | Training R={R}")
print(f"cum_var = {float(cum[k_pca-1].item()):.6f}")

ev_ratio = (ev / total_var).detach().cpu().numpy()
cum_ratio = cum.detach().cpu().numpy()
n_show = int(min(40, len(ev_ratio)))
x_pca = np.arange(1, n_show+1)

plt.figure(figsize=(8, 5), dpi=600)
plt.stem(x_pca, ev_ratio[:n_show], basefmt=" ", linefmt="C3-", markerfmt="C3o")
plt.plot(x_pca, cum_ratio[:n_show], "C0o-", label="cumulative")
plt.axhline(PCA_TARGET_VAR, linestyle="--", linewidth=1.5)
plt.xlabel("Components"); plt.ylabel("Explained variance ratio")
plt.title("PCA explained variance"); plt.legend(); plt.tight_layout()
_save("pca_explained_variance.png", kikuchi=False) 

print("Please tell me you know a way out of here bro")

HF_REP_SAMPLES = int(min(Ny_roi*Nx_roi, 128))
valid_core = np.zeros((Hp, Wp), dtype=np.bool_)
valid_core[1:-1, 1:-1] = True
if USE_MASK: valid_core = valid_core & mask_uv

uv_pool = np.stack(np.where(valid_core), axis=1).astype(np.int64)
uv_pool_t = torch.tensor(uv_pool, dtype=torch.long, device=device) \
            if uv_pool.size > 0 else None

side = int(np.ceil(np.sqrt(HF_REP_SAMPLES)))
iy_grid = np.linspace(0, Ny_roi-1, num=side, dtype=np.int64)
ix_grid = np.linspace(0, Nx_roi-1, num=side, dtype=np.int64)
IYg, IXg = np.meshgrid(iy_grid, ix_grid, indexing="ij")
rep_iy = IYg.reshape(-1)[:HF_REP_SAMPLES]
rep_ix = IXg.reshape(-1)[:HF_REP_SAMPLES]

hf_count_map = np.zeros((Hp, Wp), dtype=np.float32)
for iy_r, ix_r in zip(rep_iy, rep_ix):
    ref_img = torch.from_numpy(roi_norm[iy_r, ix_r]).float()
    gx = 0.5*(ref_img[2:,1:-1] - ref_img[:-2,1:-1])
    gy = 0.5*(ref_img[1:-1,2:] - ref_img[1:-1,:-2])
    gmag = torch.sqrt(gx*gx + gy*gy + 1e-12).detach().cpu().numpy()
    gf = np.zeros((Hp,Wp), dtype=np.float32); gf[1:-1,1:-1] = gmag
    vals = gf[valid_core]
    if vals.size == 0: continue
    thr_i = np.quantile(vals, 1.0 - HF_TOP_RATIO)
    hf_count_map += ((gf >= thr_i) & valid_core).astype(np.float32)

hf_freq_map = hf_count_map / float(len(rep_iy)) if len(rep_iy) > 0 \
              else hf_count_map.copy()
vals_freq = hf_freq_map[valid_core]
if vals_freq.size > 0:
    thr_freq = np.quantile(vals_freq, 1.0 - HF_TOP_RATIO)
    mask_high_uv = (hf_freq_map >= thr_freq) & valid_core
else:
    mask_high_uv = np.zeros((Hp, Wp), dtype=np.bool_)

mask_low_uv = valid_core & (~mask_high_uv)
uv_high = np.stack(np.where(mask_high_uv), axis=1).astype(np.int64)
uv_low = np.stack(np.where(mask_low_uv), axis=1).astype(np.int64)
uv_idx_high_t = torch.tensor(uv_high, dtype=torch.long, device=device) \
                 if uv_high.size > 0 else None
uv_idx_low_t = torch.tensor(uv_low, dtype=torch.long, device=device) \
                 if uv_low.size > 0 else None
nH = 0 if uv_idx_high_t is None else int(uv_idx_high_t.shape[0])
nL = 0 if uv_idx_low_t is None else int(uv_idx_low_t.shape[0])
w_hf = float(np.clip(float(nH)/float(max(1, nH+nL)), HF_FLOOR, HF_CEIL))
print(f"HF pixels={nH}, LF pixels={nL}, w_hf={w_hf:.4f}")

if uv_pool.size > 0:
    prob_vals = hf_freq_map[uv_pool[:,0], uv_pool[:,1]].astype(np.float64) + 1e-6
    prob_vals /= prob_vals.sum()
    uv_prob_t = torch.tensor(prob_vals, dtype=torch.float32, device=device)
else:
    uv_prob_t = None

mask_high_flat_t = torch.from_numpy(
    mask_high_uv.reshape(-1)).to(device=device, dtype=torch.bool)

def sample_batch_xyuv(BATCH):
    iy_np = np.random.randint(0, Ny_roi, size=BATCH, dtype=np.int64)
    ix_np = np.random.randint(0, Nx_roi, size=BATCH, dtype=np.int64)

    if (uv_pool_t is None) or (uv_prob_t is None) or (uv_pool_t.shape[0] < 8):
        if USE_MASK:
            pick = torch.randint(0, uv_idx_t.shape[0], (BATCH,), device=device)
            uu, vv = uv_idx_t[pick, 0], uv_idx_t[pick, 1]
        else:
            uu = torch.randint(0, Hp, (BATCH,), device=device)
            vv = torch.randint(0, Wp, (BATCH,), device=device)
    else:
        Gnum = int(round(w_hf * BATCH)); Unum = BATCH - Gnum
        uu_parts, vv_parts = [], []
        if Gnum > 0:
            pick_g = torch.multinomial(uv_prob_t, Gnum, replacement=True)
            uu_parts.append(uv_pool_t[pick_g, 0]); vv_parts.append(uv_pool_t[pick_g, 1])
        if Unum > 0:
            if USE_MASK:
                pick_u = torch.randint(0, uv_idx_t.shape[0], (Unum,), device=device)
                uu_parts.append(uv_idx_t[pick_u, 0]); vv_parts.append(uv_idx_t[pick_u, 1])
            else:
                uu_parts.append(torch.randint(0, Hp, (Unum,), device=device))
                vv_parts.append(torch.randint(0, Wp, (Unum,), device=device))
        uu = torch.cat(uu_parts); vv = torch.cat(vv_parts)

    perm = torch.randperm(BATCH, device=device)
    uu, vv = uu[perm], vv[perm]
    perm_np = perm.detach().cpu().numpy()
    iy_np, ix_np = iy_np[perm_np], ix_np[perm_np]

    is_high = mask_high_flat_t[uu * Wp + vv]
    w = torch.where(is_high.unsqueeze(1),
                    torch.full((BATCH,1), float(A_WEIGHT), device=device),
                    torch.full((BATCH,1), float(B_WEIGHT), device=device))

    iy_t = torch.from_numpy(iy_np).to(device=device, dtype=torch.long)
    ix_t = torch.from_numpy(ix_np).to(device=device, dtype=torch.long)
    u0, v0 = us_uv[uu], vs_uv[vv]
    du = (torch.rand(BATCH, device=device) - 0.5) * (2.0/(Hp-1))
    dv = (torch.rand(BATCH, device=device) - 0.5) * (2.0/(Wp-1))
    u = (u0+du).clamp(-1,1); v = (v0+dv).clamp(-1,1)

    coords = torch.stack([xs_scan[ix_t], ys_scan[iy_t], u, v], dim=1).float()
    target = torch.from_numpy(
        roi_norm[iy_np, ix_np,
                 uu.detach().cpu().numpy(),
                 vv.detach().cpu().numpy()].astype(np.float32)
    ).to(device=device).unsqueeze(1)
    return coords, target, w

print("\n" + "="*98)
print("training: oh follow you? you are gonna help me? you're a hero, "
      "I have been stuck here for so long...")
print("="*98)
print(f"Nmap={Nmap} | batch={BATCH} | steps={STEPS}")
print(f"val_every={VAL_EVERY} | Patience={PATIENCE} | LR_warmup={LR_WARMUP}")

Nuv = int(uv_idx_t.shape[0]) if (USE_MASK and uv_idx_t is not None) else Hp*Wp
Ntotal = Nmap * Nuv
Nval = int(min(200000, max(1, int(0.01 * Ntotal))))
print(f"Nval={Nval}")

rng_v = np.random.default_rng(0)
flat_v = rng_v.integers(0, Nmap, size=Nval, endpoint=False)
val_iy = (flat_v // Nx_roi).astype(np.int64)
val_ix = (flat_v %  Nx_roi).astype(np.int64)
if USE_MASK and uv_idx_t is not None:
    pick_uv = rng_v.integers(0, int(uv_idx_t.shape[0]), size=Nval, endpoint=False)
    uv_np = uv_idx_t.detach().cpu().numpy()
    val_uu = uv_np[pick_uv, 0].astype(np.int64)
    val_vv = uv_np[pick_uv, 1].astype(np.int64)
else:
    val_uu = rng_v.integers(0, Hp, size=Nval, endpoint=False).astype(np.int64)
    val_vv = rng_v.integers(0, Wp, size=Nval, endpoint=False).astype(np.int64)

val_iy_t = torch.from_numpy(val_iy).to(device=device, dtype=torch.long)
val_ix_t = torch.from_numpy(val_ix).to(device=device, dtype=torch.long)
val_uu_t = torch.from_numpy(val_uu).to(device=device, dtype=torch.long)
val_vv_t = torch.from_numpy(val_vv).to(device=device, dtype=torch.long)
du_s = 2.0/max(1, Hp-1); dv_s = 2.0/max(1, Wp-1)
with torch.no_grad():
    val_u = (us_uv[val_uu_t] + (torch.rand(Nval,device=device)-0.5)*du_s).clamp(-1,1)
    val_v = (vs_uv[val_vv_t] + (torch.rand(Nval,device=device)-0.5)*dv_s).clamp(-1,1)
val_coords = torch.stack([xs_scan[val_ix_t], ys_scan[val_iy_t], val_u, val_v], dim=1).float()
val_target = torch.from_numpy(
    roi_norm[val_iy, val_ix, val_uu, val_vv].astype(np.float32)
).to(device=device).unsqueeze(1)
val_w = torch.ones((Nval,1), device=device)

model = TensorFactorizedINR(
    Hp=Hp, Wp=Wp, R=R,
    hidden_features=HIDDEN_FEATURES,
    hidden_layers=HIDDEN_LAYERS,
    first_omega_0=FIRST_OMEGA_0,
    hidden_omega_0=HIDDEN_OMEGA_0,
    basis_init="pca",
    PCA_B_INIT=PCA_B_INIT,
).to(device)

opt = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=LR_GAMMA, patience=LR_PATIENCE,
    threshold=MIN_DELTA, threshold_mode="abs",
    cooldown=0, min_lr=LR_MIN)

train_steps=[]; train_loss_hist=[]
val_steps=[]; val_loss_hist=[]
best_val = float("inf"); best_state=None; bad=0
start_t = time.time()
model.train()

for step in range(STEPS):
    coords, target, w = sample_batch_xyuv(BATCH)
    pred = model(coords)
    loss = (w * (pred - target).pow(2)).mean() + L2_BASIS * (model.B**2).mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    train_steps.append(step)
    train_loss_hist.append(float(loss.detach().cpu()))

    if step % PRINT_EVERY == 0:
        with torch.no_grad():
            cur_lr = float(opt.param_groups[0]["lr"])
            b_std  = float(model.B.std())
        elapsed = time.time() - start_t
        eta     = elapsed / max(1, step+1) * (STEPS - step - 1)
        print(f"[train] step {step:6d}/{STEPS} | loss={float(loss):.6f} | "
              f"B_std={b_std:.3e} | lr={cur_lr:.3e} | "
              f"t={elapsed:.0f}s")

    if step % VAL_EVERY == 0 and step > 0:
        model.eval()
        VBATCH = 65536
        vloss_sum = 0.0
        with torch.no_grad():
            for vs in range(0, Nval, VBATCH):
                ve = min(vs+VBATCH, Nval)
                vp = model(val_coords[vs:ve])
                vl = (val_w[vs:ve]*(vp - val_target[vs:ve]).pow(2)).mean()
                vloss_sum += float(vl) * (ve-vs)
            vloss_val = vloss_sum / Nval + L2_BASIS * float((model.B**2).mean())
        model.train()
        val_steps.append(step); val_loss_hist.append(vloss_val)

        if step >= LR_WARMUP:
            scheduler.step(vloss_val)

        improved = (best_val - vloss_val) > MIN_DELTA
        if improved:
            best_val = vloss_val
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            if len(val_loss_hist) > WARMUP: bad += 1

        cur_lr = float(opt.param_groups[0]["lr"])
        print(f"[val] step {step:6d} | val={vloss_val:.6f} | "
              f"best={best_val:.6f} | bad={bad}/{PATIENCE} | lr={cur_lr:.3e}")

        if bad >= PATIENCE:
            print(f"[early stop] step={step} best_val={best_val:.6f}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

plt.figure(figsize=(8,5), dpi=600)
plt.plot(train_steps, train_loss_hist, alpha=0.3, label="train")
if val_steps: plt.plot(val_steps, val_loss_hist, "o-", ms=3, label="val")
plt.yscale("log"); plt.xlabel("step"); plt.ylabel("MSE loss")
plt.title(f"Learning curve (stride={STRIDE_SCAN}, BATCH={BATCH})")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
_save("learning_curve.png", kikuchi=False)

if SAVE_PTH:
    os.makedirs(os.path.dirname(os.path.abspath(SAVE_PTH)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "Hp": int(Hp), "Wp": int(Wp), "R": int(model.R),
        "Ny_roi": int(Ny_roi), "Nx_roi": int(Nx_roi),
        "Ny_idx": int(Ny_idx), "Nx_idx": int(Nx_idx),
        "ROI_SCAN": dict(ROI_SCAN),
        "STRIDE_SCAN": int(STRIDE_SCAN), "STRIDE_INDEX": int(STRIDE_INDEX),
        "USE_MASK": bool(USE_MASK), "mask_uv": mask_uv.astype(np.bool_),
        "USE_NORMALIZATION": bool(USE_NORMALIZATION),
        "scaler": None if scaler is None else {
            "data_min": float(scaler.data_min),
            "data_max": float(scaler.data_max),
            "target_min": float(scaler.target_min),
            "target_max": float(scaler.target_max)},
        "hidden_features": int(HIDDEN_FEATURES),
        "hidden_layers": int(HIDDEN_LAYERS),
        "first_omega_0": float(FIRST_OMEGA_0),
        "hidden_omega_0":float(HIDDEN_OMEGA_0),
        "best_val": float(best_val),
        "BATCH": int(BATCH), "STEPS": int(STEPS),
    }, SAVE_PTH)
    print(f"Checkpoint saved: {SAVE_PTH}")

print("\n" + "="*60)
print("training done. so you make it for yourself...you make your rigatoni pasta.")
print(f"Figures = {FIG_DIR_P.resolve()}")
print(f".pth = {SAVE_PTH}")
print(f"best_val = {best_val:.6f}")
print("="*60)
