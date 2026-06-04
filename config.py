# this is the config, define all input/output paths and parameters here.

#P2_PATH = r"D:\Downloads\data_718\AM718\AM718\1um\AM718_1um_120x120.up2"
#ANG_PATH = r"D:\Downloads\data_718\AM718\AM718\1um\AM718_1um.ang"

UP2_PATH = r"D:\Downloads\718RX\1um\718RX_1um_120x120.up2"
ANG_PATH = r"D:\Downloads\718RX\1um\718RX_1um.ang"

SAVE_PTH = r"D:\Alaska_Data\github_test\RX_stride1_small_roi_test2.pth"
GT_OUT_MAT = r"D:\Alaska_Data\github_test\RX_gt_bandpass.mat"
RECON_OUT_MAT = r"D:\Alaska_Data\github_test\RX_recon_bandpass.mat"

FIG_DIR = r"D:\Alaska_Data\github_test\result_plot"
PROFILE_OUT_DIR = r"D:\Alaska_Data\github_test\result_plot\s_profile"

ROI_SCAN = dict(y0=000, y1=900, x0=000, x1=1000)
STRIDE_SCAN = 1
STRIDE_INDEX = 1

USE_NORMALIZATION = True
USE_MASK = True

R = 512
HIDDEN_FEATURES = 512
HIDDEN_LAYERS = 4
FIRST_OMEGA_0 = 30
HIDDEN_OMEGA_0 = 30
STEPS = 200000
BATCH = 32768
LR  = 1e-4
L2_BASIS = 1e-6
VAL_EVERY = 1000
PATIENCE = 30
MIN_DELTA = 1e-5
WARMUP = 3
LR_GAMMA = 0.5
LR_PATIENCE = 10
LR_MIN = 1e-6
LR_WARMUP = 4000
PRINT_EVERY = 1000

# HF weighting
HF_TOP_RATIO = 0.15
A_WEIGHT = 2.0
B_WEIGHT = 1.0
HF_FLOOR = 0.15
HF_CEIL = 0.50

# set your PCA target
PCA_TARGET_VAR = 0.95
PCA_MAX_SAMPLES = 200000
PCA_MAX_R = 512
PCA_SEED = 0

# this is for hough indexing, please ref: https://www.sciencedirect.com/science/article/pii/S0304399118303322
SIGMA_LOW = 0.25
SIGMA_HIGH = 0.035
CHUNKSIZE_RECON = 32
CHUNKSIZE_INDEX2 = 64

# down below are for plotting gradient profile
COEFF_BATCH = 2048

# put your trained pth here
SMAP_DATASETS = [
    dict(label="RX", pth=r"D:\Alaska_Data\RX_fullmap_stride1_test2.pth"),
    dict(label="AM", pth=r"D:\Alaska_Data\AM_fullmap_final_stride1_test2.pth"),
]
# your gradient profile plotting style
DPI = 600
PLOW = 2.0
PHIGH = 98.0
CMAP = "viridis"