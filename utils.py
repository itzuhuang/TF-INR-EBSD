import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
from pathlib import Path

_fig_counter = [0]

def savefig(fig_dir, name=None, kikuchi=False, dpi=600):
    _fig_counter[0] += 1
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fname = name if name else f"figure_{_fig_counter[0]:03d}.png"
    path_png = fig_dir / fname
    plt.savefig(path_png, bbox_inches="tight", dpi=dpi)
    if not kikuchi:
        path_svg = fig_dir / fname.replace(".png", ".svg")
        plt.savefig(path_svg, bbox_inches="tight", format="svg")
        print(f"[figure saved] {path_png}  +  {path_svg.name}")
    else:
        print(f"[figure saved] {path_png}")
    plt.close()

def read_ang_xy(ang_path):
    df = pd.read_csv(ang_path, sep=r"\s+", header=None, skiprows=133,
                     usecols=[3, 4], names=["x", "y"], engine="python")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    return df["x"].to_numpy(dtype=np.int64), df["y"].to_numpy(dtype=np.int64)


def read_ang_header_params(path):
    keys = {"x-star": None, "y-star": None, "z-star": None,
            "SampleTiltAngle": None, "CameraElevationAngle": None,
            "CameraAzimuthalAngle": None, "LatticeConstants": None}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("#"):
                break
            for k in list(keys.keys()):
                if k in line:
                    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                    if not nums:
                        continue
                    keys[k] = list(map(float, nums[:6])) if k == "LatticeConstants" \
                               else float(nums[0])
    for k in ["x-star", "y-star", "z-star", "SampleTiltAngle"]:
        if keys[k] is None:
            raise ValueError(f"Missing {k} in ANG header.")
    keys["CameraElevationAngle"] = keys["CameraElevationAngle"] or 0.0
    keys["CameraAzimuthalAngle"] = keys["CameraAzimuthalAngle"] or 0.0
    keys["LatticeConstants"]     = keys["LatticeConstants"] or \
                                   [3.560, 3.560, 3.560, 90.0, 90.0, 90.0]
    return keys

class DataScaler:
    def __init__(self, data_min, data_max, target_min=-1, target_max=1):
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        denom = self.data_max - self.data_min
        if abs(denom) < 1e-12:
            denom = 1.0
        self.scale = (self.target_max - self.target_min) / denom
        self.bias = self.target_min - self.scale * self.data_min

    def normalize(self, data):
        return data * self.scale + self.bias

    def denormalize(self, data):
        return (data - self.bias) / self.scale

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                     1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                     np.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenXY(nn.Module):
    def __init__(self, in_features=2, out_features=32,
                 hidden_features=512, hidden_layers=4,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        net = [SineLayer(in_features, hidden_features,
                         is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            net.append(SineLayer(hidden_features, hidden_features,
                                 omega_0=hidden_omega_0))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                 np.sqrt(6 / hidden_features) / hidden_omega_0)
        net.append(final)
        self.net = nn.Sequential(*net)

    def forward(self, xy):
        return self.net(xy)


class TensorFactorizedINR(nn.Module):
    def __init__(self, Hp, Wp, R=512,
                 hidden_features=512, hidden_layers=4,
                 first_omega_0=30, hidden_omega_0=30,
                 basis_init="pca", PCA_B_INIT=None):
        super().__init__()
        self.Hp = Hp
        self.Wp = Wp
        self.R = int(R)

        if self.R > 1:
            self.A = SirenXY(
                in_features=2, out_features=self.R - 1,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0)
        else:
            self.A = None

        if basis_init == "pca" and PCA_B_INIT is not None:
            B = PCA_B_INIT.detach().clone()
            if B.shape[0] != self.R:
                B = B[:self.R] if self.R < B.shape[0] else torch.cat(
                    [B, 1e-2 * torch.randn(self.R - B.shape[0], Hp, Wp,
                                           device=B.device, dtype=B.dtype)], dim=0)
        elif basis_init == "zeros":
            B = torch.zeros(self.R, Hp, Wp)
        else:
            B = 1e-2 * torch.randn(self.R, Hp, Wp)
        self.B = nn.Parameter(B)

    def coeffs_xy(self, xy):
        n = xy.shape[0]
        ones = torch.ones((n, 1), device=xy.device, dtype=xy.dtype)
        if self.R == 1:
            return ones
        return torch.cat([ones, self.A(xy)], dim=1)

    def _bilinear_sample_B(self, u, v):
        uf = (u + 1) * 0.5 * (self.Hp - 1)
        vf = (v + 1) * 0.5 * (self.Wp - 1)
        u0 = torch.floor(uf).long().clamp(0, self.Hp - 1)
        v0 = torch.floor(vf).long().clamp(0, self.Wp - 1)
        u1 = (u0 + 1).clamp(0, self.Hp - 1)
        v1 = (v0 + 1).clamp(0, self.Wp - 1)
        wu = (uf - u0.float()).clamp(0, 1)
        wv = (vf - v0.float()).clamp(0, 1)
        B00 = self.B[:, u0, v0].T; B10 = self.B[:, u1, v0].T
        B01 = self.B[:, u0, v1].T; B11 = self.B[:, u1, v1].T
        b = ((1-wu)*(1-wv))[:,None]*B00 + (wu*(1-wv))[:,None]*B10 + \
            ((1-wu)*wv)[:,None]*B01 + (wu*wv)[:,None]*B11
        return b

    def forward(self, coords):
        xy = coords[:, :2]
        u = coords[:, 2]
        v = coords[:, 3]
        a = self.coeffs_xy(xy)
        b = self._bilinear_sample_B(u, v)
        return (a * b).sum(dim=1, keepdim=True)

    def reconstruct_pattern_from_coeffs(self, a_vec):
        return torch.einsum("r,ruv->uv", a_vec, self.B)

def load_checkpoint(pth_path, device):
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    Hp = int(ckpt["Hp"]);  Wp = int(ckpt["Wp"])
    Ny_roi = int(ckpt["Ny_roi"]); Nx_roi = int(ckpt["Nx_roi"])
    Ny_idx = int(ckpt.get("Ny_idx", Ny_roi))
    Nx_idx = int(ckpt.get("Nx_idx", Nx_roi))
    sd = ckpt["model_state_dict"]
    R = int(sd["B"].shape[0])
    hf = int(ckpt.get("hidden_features", 512))
    hl = int(ckpt.get("hidden_layers", 4))
    fo = float(ckpt.get("first_omega_0", 30.0))
    ho = float(ckpt.get("hidden_omega_0", 30.0))
    mask = np.array(ckpt["mask_uv"]).astype(bool)

    model = TensorFactorizedINR(Hp=Hp, Wp=Wp, R=R,
                                hidden_features=hf, hidden_layers=hl,
                                first_omega_0=fo, hidden_omega_0=ho,
                                basis_init="zeros")
    model.load_state_dict(sd)
    model = model.to(device).eval()

    sc = ckpt.get("scaler", None)
    scaler = DataScaler(sc["data_min"], sc["data_max"],
                        sc["target_min"], sc["target_max"]) if sc else None

    return dict(
        model=model, scaler=scaler, mask_uv=mask,
        Hp=Hp, Wp=Wp,
        Ny_roi=Ny_roi, Nx_roi=Nx_roi,
        Ny_idx=Ny_idx, Nx_idx=Nx_idx,
        ROI_SCAN=ckpt.get("ROI_SCAN", {}),
        STRIDE_SCAN=int(ckpt.get("STRIDE_SCAN", 1)),
        STRIDE_INDEX=int(ckpt.get("STRIDE_INDEX", 1)),
        best_val=float(ckpt.get("best_val", float("nan"))),
        BATCH=int(ckpt.get("BATCH", 0)), 
        STEPS=int(ckpt.get("STEPS", 0))
    )

def make_gaussian_bandpass_window(H, W, sigma_low=0.25, sigma_high=0.035):
    fy = np.fft.fftfreq(H); fx = np.fft.fftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    R2 = FX**2 + FY**2
    Wbp = np.exp(-R2 / (2 * sigma_low**2)) * (1 - np.exp(-R2 / (2 * sigma_high**2)))
    return (Wbp / (Wbp.max() + 1e-12)).astype(np.float32)


def bandpass_filter_float01(pats, Wbp, eps=1e-12):
    F = np.fft.fft2(pats, axes=(-2, -1))
    out = np.fft.ifft2(F * Wbp[None], axes=(-2, -1)).real.astype(np.float32)
    flat = out.reshape(out.shape[0], -1)
    mn = flat.min(axis=1)[:, None]
    mx = flat.max(axis=1)[:, None]
    return ((flat - mn) / (mx - mn + eps)).reshape(out.shape)


def to_uint8(pats_float):
    out = np.empty_like(pats_float, dtype=np.uint8)
    for i in range(pats_float.shape[0]):
        p = pats_float[i]
        mn = float(np.nanmin(p)); mx = float(np.nanmax(p))
        if not (np.isfinite(mn) and np.isfinite(mx)) or (mx - mn) < 1e-12:
            out[i] = 0
            continue
        out[i] = np.clip(255 * (p - mn) / (mx - mn), 0, 255).astype(np.uint8)
    return out

def attach_detector(sig, det):
    if   hasattr(sig, "detector"): sig.detector = det
    elif hasattr(sig, "set_detector"): sig.set_detector(det)
    else: raise AttributeError("cannot attach detector")


def extract_fit_vector(index_data, n_expected):
    def _norm(arr):
        arr = np.squeeze(np.asarray(arr))
        if arr.ndim == 1 and arr.shape[0] == n_expected:
            return arr.astype(np.float32)
        if arr.ndim == 2:
            for sl in [arr[1, :], arr[:, 1], arr[-1, :], arr[:, -1]]:
                if sl.shape[0] == n_expected:
                    return sl.astype(np.float32)
        return None
    names = ["fit", "Fit", "fitdeg", "fit_deg"]
    src   = index_data.dtype.names if (isinstance(index_data, np.ndarray) and
                                        index_data.dtype.names) else \
            (index_data.keys() if isinstance(index_data, dict) else [])
    for nm in names:
        if nm in src:
            v = _norm(index_data[nm])
            if v is not None:
                return v
    return None
