
RUN_GT_INDEXING = True 
RUN_RECON_INDEXING = True   

import time
import numpy as np
import torch
import kikuchipy as kp
from pathlib import Path
from scipy.io import savemat

from orix.crystal_map import PhaseList, Phase
from diffpy.structure import Atom, Lattice, Structure

from config import *
from utils import (load_checkpoint, read_ang_xy, read_ang_header_params,
                   make_gaussian_bandpass_window, bandpass_filter_float01,
                   to_uint8, attach_detector, extract_fit_vector)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

print("\n" + "="*30)
print("loading checkpoint..yum yum.")
print("="*30)

ck = load_checkpoint(SAVE_PTH, device)
model = ck["model"]
scaler = ck["scaler"]
mask_uv = ck["mask_uv"]
Hp, Wp = ck["Hp"], ck["Wp"]
Ny_roi = ck["Ny_roi"]; Nx_roi = ck["Nx_roi"]
Ny_idx = ck["Ny_idx"]; Nx_idx = ck["Nx_idx"]
print(f"R={model.R} Ny_roi={Ny_roi} Nx_roi={Nx_roi}")
print(f"Ny_idx={Ny_idx} Nx_idx={Nx_idx} best_val={ck['best_val']:.6f}")

ys_scan = torch.linspace(-1, 1, steps=Ny_roi, device=device)
xs_scan = torch.linspace(-1, 1, steps=Nx_roi, device=device)
ys_scan_idx = torch.linspace(-1, 1, steps=Ny_idx, device=device)
xs_scan_idx = torch.linspace(-1, 1, steps=Nx_idx, device=device)

print("Yeah. No, that sounds delicious. Why don't we go find some right now?")

hdr = read_ang_header_params(ANG_PATH)
pc_tsl = (hdr["x-star"], hdr["y-star"], hdr["z-star"])
sample_tilt = float(hdr["SampleTiltAngle"])
det_tilt = float(hdr["CameraElevationAngle"])
det_azim = float(hdr["CameraAzimuthalAngle"])
a_lc,b_lc,c_lc,al_lc,be_lc,ga_lc = hdr["LatticeConstants"]
print(f"PC={pc_tsl}, tilt={sample_tilt}, det_tilt={det_tilt}")

det = kp.detectors.EBSDDetector(
    shape=(Hp, Wp), pc=pc_tsl, convention="tsl",
    sample_tilt=sample_tilt, tilt=det_tilt, azimuthal=det_azim)

structure = Structure(
    lattice=Lattice(a_lc,b_lc,c_lc,al_lc,be_lc,ga_lc),
    atoms=[Atom("Ni", [0,0,0])])
phase_list = PhaseList(Phase(name="Ni", space_group=225, structure=structure))
reflectors = [[1,1,1],[2,0,0],[2,2,0],[3,1,1],[2,2,2],[4,0,0]]
indexer = det.get_indexer(phase_list=phase_list, reflectors=reflectors)
print("Yummy in my tummy. Rigatoni Pasta.")

Wbp = make_gaussian_bandpass_window(Hp, Wp, SIGMA_LOW, SIGMA_HIGH)

if RUN_GT_INDEXING:
    print("\n" + "="*72)
    print("GT (UP2) Hough indexing: you're rambling. I I really need some water. You know")
    print("="*72)

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

    N = Ny_roi * Nx_roi
    q_gt = np.full((N, 4), np.nan, dtype=np.float64)
    fit_gt = np.full((N,), np.nan, dtype=np.float32)
    CHUNK_R = 128
    CHUNK_I = 128
    t0 = time.time()

    for start in range(0, N, CHUNK_R):
        end = min(N, start + CHUNK_R)
        idx_chunk = idx_roi[start:end]
        pats = np.array(s.data[idx_chunk].compute())
        if pats.ndim == 4 and pats.shape[1] == 1:
            pats = pats[:, 0, :, :]
        pats = pats.astype(np.float32, copy=False)
        pats_bp = bandpass_filter_float01(pats, Wbp)
        pats_u8 = to_uint8(pats_bp)

        sig = kp.signals.EBSD(pats_u8)
        attach_detector(sig, det)
        res = sig.hough_indexing(phase_list=phase_list, indexer=indexer,
                                 chunksize=min(CHUNK_I, end-start),
                                 verbose=0, return_index_data=True)
        xmap, idata = res if (isinstance(res, tuple) and len(res)==2) else (res, None)
        q_gt[start:end] = np.array(xmap.rotations.data, dtype=np.float64)
        if idata is not None:
            fv = extract_fit_vector(idata, end-start)
            if fv is not None: fit_gt[start:end] = fv

        elapsed = time.time() - t0
        eta = elapsed / max(1, end) * (N - end)
        print(f"[GT] {end}/{N} fit={np.nanmean(fit_gt[start:end]):.3f} ETA={eta:.0f}s")

    import os; os.makedirs(str(Path(GT_OUT_MAT).parent), exist_ok=True)
    savemat(GT_OUT_MAT, {
        "q_gt": q_gt.reshape(Ny_roi, Nx_roi, 4),
        "fit_gt": fit_gt.reshape(Ny_roi, Nx_roi),
        "Ny_roi": np.int32(Ny_roi),
        "Nx_roi": np.int32(Nx_roi),
        "pc_tsl": np.array(pc_tsl, dtype=np.float64),
        "sample_tilt":float(sample_tilt),
        "det_tilt": float(det_tilt),
        "det_azim": float(det_azim),
        "sigma_low": np.float32(SIGMA_LOW),
        "sigma_high": np.float32(SIGMA_HIGH),
        "ROI_y0": np.int32(ROI_SCAN["y0"]), "ROI_y1": np.int32(ROI_SCAN["y1"]),
        "ROI_x0": np.int32(ROI_SCAN["x0"]), "ROI_x1": np.int32(ROI_SCAN["x1"]),
        "lattice_constants": np.array(hdr["LatticeConstants"], dtype=np.float64),
    })
    print(f"GT .mat saved: {GT_OUT_MAT}")

if RUN_RECON_INDEXING:
    print("\n" + "="*60)
    print("Recon patterns start Hough indexing")
    print(f"({Ny_idx}x{Nx_idx} = {Ny_idx*Nx_idx} points), please don't leave me.")
    print("Okay. I'm really glad that we can share these moments together. It's nice, I guess ")
    print("="*60)

    N_total_idx = Ny_idx * Nx_idx
    q_rc_flat = np.full((N_total_idx, 4), np.nan, dtype=np.float64)
    fit_rc_flat = np.full((N_total_idx,), np.nan, dtype=np.float32)
    flat_ptr = 0

    def iter_chunks_idx(Ny, Nx, chunk_n):
        buf = []
        for iy in range(Ny):
            for ix in range(Nx):
                buf.append((iy, ix))
                if len(buf) >= chunk_n:
                    yield buf; buf = []
        if buf: yield buf

    @torch.no_grad()
    def recon_chunk_pts_idx(pts):
        iy_b = torch.tensor([p[0] for p in pts], device=device, dtype=torch.long)
        ix_b = torch.tensor([p[1] for p in pts], device=device, dtype=torch.long)
        xy_b = torch.stack([xs_scan_idx[ix_b], ys_scan_idx[iy_b]], dim=1).float()
        a_b = model.coeffs_xy(xy_b)
        pn = torch.einsum("nr,ruv->nuv", a_b, model.B).cpu().numpy()
        return scaler.denormalize(pn).astype(np.float32) if scaler else pn.astype(np.float32)

    t_idx_start = time.time()
    for pts in iter_chunks_idx(Ny_idx, Nx_idx, CHUNKSIZE_RECON):
        n_chunk = len(pts)
        pats_raw = recon_chunk_pts_idx(pts)
        pats_bp = bandpass_filter_float01(pats_raw, Wbp)
        pats_u8 = to_uint8(pats_bp)

        sig = kp.signals.EBSD(pats_u8)
        attach_detector(sig, det)
        res = sig.hough_indexing(phase_list=phase_list, indexer=indexer,
                                 chunksize=min(CHUNKSIZE_INDEX2, n_chunk),
                                 verbose=0, return_index_data=True)
        xmap, idata = res if (isinstance(res, tuple) and len(res)==2) else (res, None)

        q = np.array(xmap.rotations.data, dtype=np.float64)
        if q.shape != (n_chunk, 4):
            raise ValueError(f"Quaternion shape {q.shape} != {(n_chunk,4)}")
        q_rc_flat[flat_ptr:flat_ptr+n_chunk] = q

        if idata is not None:
            fv = extract_fit_vector(idata, n_expected=n_chunk)
            if fv is not None: fit_rc_flat[flat_ptr:flat_ptr+n_chunk] = fv

        elapsed_idx = time.time() - t_idx_start
        eta_idx = elapsed_idx / max(1, flat_ptr+n_chunk) * (N_total_idx - flat_ptr - n_chunk)
        print(f"[RECON] {flat_ptr+n_chunk}/{N_total_idx} | "
              f"fit={np.nanmean(fit_rc_flat[flat_ptr:flat_ptr+n_chunk]):.3f} | "
              f"ETA={eta_idx:.0f}s")
        flat_ptr += n_chunk

    import os; os.makedirs(str(Path(RECON_OUT_MAT).parent), exist_ok=True)
    savemat(RECON_OUT_MAT, {
        "q_recon": q_rc_flat.reshape(Ny_idx, Nx_idx, 4),
        "fit_recon": fit_rc_flat.reshape(Ny_idx, Nx_idx),
        "mae_map": np.zeros((Ny_roi, Nx_roi), dtype=np.float32), 
        "Ny_idx": np.int32(Ny_idx), "Nx_idx": np.int32(Nx_idx),
        "Ny_train": np.int32(Ny_roi), "Nx_train": np.int32(Nx_roi),
        "STRIDE_SCAN": np.int32(STRIDE_SCAN),
        "STRIDE_INDEX": np.int32(STRIDE_INDEX),
        "ROI_y0": np.int32(ROI_SCAN["y0"]), "ROI_y1": np.int32(ROI_SCAN["y1"]),
        "ROI_x0": np.int32(ROI_SCAN["x0"]), "ROI_x1": np.int32(ROI_SCAN["x1"]),
        "pc_tsl": np.array(pc_tsl, dtype=np.float64),
        "sample_tilt": float(sample_tilt),
        "det_tilt": float(det_tilt),
        "det_azim": float(det_azim),
        "sigma_low": np.float32(SIGMA_LOW),
        "sigma_high": np.float32(SIGMA_HIGH),
        "lattice_constants": np.array(hdr["LatticeConstants"], dtype=np.float64),
        "BATCH": np.int32(ck.get("BATCH", 0)),
        "STEPS": np.int32(ck.get("STEPS", 0)),
        "best_val": float(ck["best_val"]),
    })
    print(f"Recon .mat saved at {RECON_OUT_MAT} | You don't think we're going to get out of here?")

print("\n" + "="*66)
print("indexing done. What's the point of walking if there's nowhere to go? ")
if RUN_GT_INDEXING: print(f"GT .mat → {GT_OUT_MAT}")
if RUN_RECON_INDEXING: print(f"Recon.mat → {RECON_OUT_MAT}")
print("="*66)
