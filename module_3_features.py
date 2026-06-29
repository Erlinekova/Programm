import os
import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import module3


SPARSE_PATH      = str(module3["sparse_path"])
IMAGES_DIR       = str(module3["images_dir"])
FEATURES_DIR     = str(module3["features_dir"])
OUTPUT_DIR       = str(module3["output_dir"])
N_BOOTSTRAP      = module3["n_bootstrap"]
CI_PERCENTILE    = module3["ci_percentile"]
CI_THRESHOLD     = module3["ci_threshold"]
K_NEIGHBORS      = module3["k_neighbors"]
PCA_WEIGHT_DECAY = module3["pca_weight_decay"]
PAI_PATCH_SIZE   = module3["pai_patch_size"]


rec = pycolmap.Reconstruction(SPARSE_PATH)
points3D_dict = rec.points3D
point_ids = sorted(points3D_dict.keys())
n_points = len(point_ids)

coords = np.zeros((n_points, 3), dtype=np.float64)
errors = np.zeros(n_points, dtype=np.float64)
tracks = np.zeros(n_points, dtype=np.int32)
colors = np.zeros((n_points, 3), dtype=np.uint8)

for i, pid in enumerate(point_ids):
    pt = points3D_dict[pid]
    coords[i] = pt.xyz
    errors[i] = pt.error
    tracks[i] = len(pt.track.elements)
    colors[i] = pt.color


surface_class = (
    np.load(os.path.join(FEATURES_DIR, 'surface_class.npy'))
    if os.path.exists(os.path.join(FEATURES_DIR, 'surface_class.npy'))
    else np.zeros(n_points, dtype=np.int32)
)
class_conf = (
    np.load(os.path.join(FEATURES_DIR, 'class_confidence.npy'))
    if os.path.exists(os.path.join(FEATURES_DIR, 'class_confidence.npy'))
    else np.zeros(n_points, dtype=np.float32)
)
n_views = (
    np.load(os.path.join(FEATURES_DIR, 'n_views.npy'))
    if os.path.exists(os.path.join(FEATURES_DIR, 'n_views.npy'))
    else np.zeros(n_points, dtype=np.int32)
)

if len(surface_class) != n_points:
    surface_class = np.zeros(n_points, dtype=np.int32)


ERROR_mean = errors
ERROR_std = np.where(tracks > 1, errors * np.sqrt(1.0 - 1.0 / np.maximum(tracks, 2)), 0.0)
ERROR_std = ERROR_std * (1.0 + 0.5 * np.log1p(tracks / 10.0))
TRACK = tracks.astype(np.float32)

tree = cKDTree(coords)
distances, _ = tree.query(coords, k=K_NEIGHBORS + 1, workers=-1)
mean_dist = distances[:, 1:].mean(axis=1)
LPD = 1.0 / (mean_dist + 1e-6)
LPD = LPD / (LPD.max() + 1e-9)

normals_path = os.path.join(FEATURES_DIR, 'normals.npy')
normals = np.load(normals_path) if os.path.exists(normals_path) else None

if normals is None:
    normals = np.zeros((n_points, 3), dtype=np.float32)
    for i in tqdm(range(0, n_points, 1000), desc="Нормали"):
        end = min(i + 1000, n_points)
        _, indices = tree.query(coords[i:end], k=K_NEIGHBORS)
        for j, idx in enumerate(indices):
            pts = coords[idx]
            centered = pts - pts.mean(axis=0)
            try:
                _, vecs = np.linalg.eigh(np.cov(centered.T))
                normals[i + j] = vecs[:, 0]
            except Exception:
                normals[i + j] = [0, 0, 1]

distances_to_k, indices_k = tree.query(coords, k=K_NEIGHBORS)
NCS_weighted = np.zeros(n_points, dtype=np.float32)
for i in tqdm(range(0, n_points, 5000), desc="NCS_weighted"):
    end = min(i + 5000, n_points)
    for j in range(end - i):
        idx = indices_k[i + j]
        dist = distances_to_k[i + j]
        weights = np.exp(-(dist[1:] ** 2) / (PCA_WEIGHT_DECAY ** 2))
        dot_products = np.abs((normals[idx[1:]] * normals[i + j]).sum(axis=1))
        NCS_weighted[i + j] = (
            np.average(dot_products, weights=weights) if weights.sum() > 0 else 0.0
        )
NCS_weighted = np.clip(NCS_weighted, 0, 1)

RGH = distances[:, 1:].std(axis=1)
RGH = RGH / (RGH.max() + 1e-9)

PAI = ERROR_mean * (1.0 - NCS_weighted)
PAI = PAI / (PAI.max() + 1e-9)


proxy_quality = (
    (TRACK / (TRACK.max() + 1e-9)) *
    (LPD / (LPD.max() + 1e-9)) *
    (1.0 - ERROR_mean / (ERROR_mean.max() + 1e-9))
)
proxy_quality = np.clip(proxy_quality, 0, 1)

features = {
    'ERROR_mean':   ERROR_mean,
    'ERROR_std':    ERROR_std,
    'TRACK':        TRACK,
    'LPD':          LPD,
    'NCS_weighted': NCS_weighted,
    'RGH':          RGH,
    'PAI':          PAI,
}

correlations = {}
feature_names = list(features.keys())
for fname in feature_names:
    r, _ = pearsonr(features[fname], proxy_quality)
    correlations[fname] = r

abs_corrs = np.array([abs(correlations[f]) for f in feature_names])
weights = np.array([np.sign(correlations[f]) * abs(correlations[f]) for f in feature_names])
weights = weights / (abs_corrs.sum() + 1e-9)

features_normalized = {}
for fname in feature_names:
    f = features[fname]
    f_min, f_max = f.min(), f.max()
    if f_max - f_min > 1e-9:
        features_normalized[fname] = (f - f_min) / (f_max - f_min)
    else:
        features_normalized[fname] = np.zeros_like(f)

for fname in ['ERROR_mean', 'ERROR_std', 'RGH', 'PAI']:
    if correlations[fname] < 0:
        features_normalized[fname] = 1.0 - features_normalized[fname]

RCS = np.zeros(n_points, dtype=np.float32)
for fname, w in zip(feature_names, weights):
    RCS += w * features_normalized[fname]
RCS = RCS - RCS.min()
if RCS.max() > 1e-9:
    RCS = RCS / RCS.max()
RCS = np.clip(RCS, 0, 1)


RCS_bootstrap = np.zeros((N_BOOTSTRAP, n_points), dtype=np.float32)
for b in tqdm(range(N_BOOTSTRAP), desc="Бутстрап"):
    weights_perturbed = weights + np.random.normal(0, 0.05, len(weights))
    rcs_b = np.zeros(n_points, dtype=np.float32)
    for fname, w in zip(feature_names, weights_perturbed):
        rcs_b += w * features_normalized[fname]
    rcs_b = rcs_b - rcs_b.min()
    if rcs_b.max() > 1e-9:
        rcs_b = rcs_b / rcs_b.max()
    RCS_bootstrap[b] = np.clip(rcs_b, 0, 1)

RCS_lower = np.percentile(RCS_bootstrap, 5, axis=0)
RCS_upper = np.percentile(RCS_bootstrap, 95, axis=0)
CI_width  = RCS_upper - RCS_lower
uncertain = (CI_width > CI_THRESHOLD).astype(np.int32)


os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, 'points.npy'), coords)
np.save(os.path.join(OUTPUT_DIR, 'normals.npy'), normals)
np.save(os.path.join(OUTPUT_DIR, 'colors.npy'), colors)

df = pd.DataFrame({
    'point_id':         point_ids,
    'x':                coords[:, 0],
    'y':                coords[:, 1],
    'z':                coords[:, 2],
    'ERROR_mean':       ERROR_mean,
    'ERROR_std':        ERROR_std,
    'TRACK':            TRACK,
    'LPD':              LPD,
    'NCS_weighted':     NCS_weighted,
    'RGH':              RGH,
    'PAI':              PAI,
    'RCS':              RCS,
    'CI_lower':         RCS_lower,
    'CI_upper':         RCS_upper,
    'CI_width':         CI_width,
    'uncertain':        uncertain,
    'surface_class':    surface_class,
    'class_confidence': class_conf,
    'n_views':          n_views,
})
df.to_csv(os.path.join(OUTPUT_DIR, 'features.csv'), index=False)

np.save(os.path.join(OUTPUT_DIR, 'rcs.npy'), RCS)
np.save(os.path.join(OUTPUT_DIR, 'uncertain.npy'), uncertain)
np.save(os.path.join(OUTPUT_DIR, 'ci_width.npy'), CI_width)
np.save(os.path.join(OUTPUT_DIR, 'correlations.npy'),
        np.array([correlations[f] for f in feature_names]))

corr_df = pd.DataFrame({
    'feature':   feature_names,
    'pearson_r': [correlations[f] for f in feature_names],
    'weight':    weights,
})
corr_df.to_csv(os.path.join(OUTPUT_DIR, 'correlations.csv'), index=False)


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Признаки и RCS", fontsize=14, fontweight='bold')

axes[0, 0].hist(RCS, bins=50, color='steelblue', edgecolor='black')
axes[0, 0].set_title("Распределение RCS")
axes[0, 0].set_xlabel("RCS")
axes[0, 0].set_ylabel("Количество точек")
axes[0, 0].axvline(0.35, color='red', linestyle='--', label='Порог 0.35')
axes[0, 0].axvline(0.50, color='green', linestyle='--', label='Порог 0.50')
axes[0, 0].legend()

axes[0, 1].scatter(ERROR_mean, RCS, s=1, alpha=0.3, c='steelblue')
axes[0, 1].set_title("RCS vs ERROR_mean")
axes[0, 1].set_xlabel("ERROR_mean")
axes[0, 1].set_ylabel("RCS")
r, _ = pearsonr(ERROR_mean, RCS)
axes[0, 1].text(0.05, 0.95, f"Pearson r = {r:+.3f}",
                transform=axes[0, 1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0, 2].scatter(TRACK, RCS, s=1, alpha=0.3, c='steelblue')
axes[0, 2].set_title("RCS vs TRACK")
axes[0, 2].set_xlabel("TRACK")
axes[0, 2].set_ylabel("RCS")
r, _ = pearsonr(TRACK, RCS)
axes[0, 2].text(0.05, 0.95, f"Pearson r = {r:+.3f}",
                transform=axes[0, 2].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[1, 0].hist(CI_width, bins=50, color='orange', edgecolor='black')
axes[1, 0].set_title("Распределение CI_width")
axes[1, 0].set_xlabel("CI_width")
axes[1, 0].set_ylabel("Количество точек")
axes[1, 0].axvline(CI_THRESHOLD, color='red', linestyle='--',
                   label=f'Порог {CI_THRESHOLD}')
axes[1, 0].legend()

axes[1, 1].barh(feature_names[::-1],
                [correlations[f] for f in feature_names[::-1]],
                color='steelblue')
axes[1, 1].set_title("Корреляция признаков")
axes[1, 1].set_xlabel("Pearson r")
axes[1, 1].axvline(0, color='black', linewidth=0.5)

unique_classes = np.unique(surface_class)
unique_classes = [c for c in unique_classes if c in (0, 1)]
class_means = [RCS[surface_class == c].mean() for c in unique_classes]
class_names = {0: 'Норма', 1: 'Стекло'}
colors_bar = {0: 'gray', 1: 'blue'}
axes[1, 2].bar(
    [class_names[c] for c in unique_classes],
    class_means,
    color=[colors_bar[c] for c in unique_classes]
)
axes[1, 2].set_title("Средний RCS по классам")
axes[1, 2].set_ylabel("RCS")
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'features_report.png'),
            dpi=150, bbox_inches='tight')
plt.close()


print(f"\nТочек обработано: {n_points:,}")
print(f"Средний RCS: {RCS.mean():.3f} (σ = {RCS.std():.3f})")
print(f"Ненадёжных точек: {uncertain.sum():,} ({100 * uncertain.mean():.1f}%)")
print(f"Стекло: {(surface_class == 1).sum():,} точек ({100 * (surface_class == 1).mean():.1f}%)")
print(f"Файлы сохранены в: {OUTPUT_DIR}")