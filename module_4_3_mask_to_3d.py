import os
import sys
import numpy as np
import cv2
import pandas as pd
import pycolmap
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from config import module4_3

SPARSE_PATH  = str(module4_3["sparse_path"])
IMAGES_DIR   = str(module4_3["images_dir"])
MASKS_DIR    = str(module4_3["masks_dir"])
FEATURES_DIR = str(module4_3["features_dir"])
OUTPUT_DIR   = str(module4_3["output_dir"])

MIN_VIEWS = module4_3["min_views"]
MIN_CONF  = module4_3["min_conf"]
MAX_IMAGES = module4_3["max_images"]
N_CLASSES  = module4_3["n_classes"]


def load_inputs():
    rec = pycolmap.Reconstruction(SPARSE_PATH)

    pts_path = os.path.join(FEATURES_DIR, "points.npy")
    nor_path = os.path.join(FEATURES_DIR, "normals.npy")
    if not os.path.exists(pts_path):
        sys.exit(f"Не найдено {pts_path}\nСначала выполните module_3_features.py")

    points  = np.load(pts_path)
    normals = np.load(nor_path) if os.path.exists(nor_path) else None

    mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith("_mask.png")])
    if not mask_files:
        sys.exit(f"Нет масок в {MASKS_DIR}\nСначала запустите module_4_1_segmentation.py")

    return rec, points, normals, mask_files


class MaskCache:
    def __init__(self, masks_dir):
        self.masks_dir = masks_dir
        self._mask_cache = {}
        self._conf_cache = {}
        self._index = {}
        for f in os.listdir(masks_dir):
            if f.endswith("_mask.png"):
                stem = f[:-len("_mask.png")]
                mp = os.path.join(masks_dir, f)
                cp = os.path.join(masks_dir, f"{stem}_confidence.png")
                self._index[stem] = (mp, cp)

    def get(self, img_name):
        stem_full = os.path.splitext(os.path.basename(img_name))[0]
        for stem in (stem_full, stem_full.lower()):
            if stem in self._index:
                if stem not in self._mask_cache:
                    mp, cp = self._index[stem]
                    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    c_img = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
                    self._mask_cache[stem] = m
                    self._conf_cache[stem] = (
                        c_img.astype(np.float32) / 255.0
                        if c_img is not None
                        else (np.ones_like(m, dtype=np.float32) * 0.5 if m is not None else None)
                    )
                return self._mask_cache[stem], self._conf_cache[stem]
        return None, None


def get_pose(image_data):
    cfw = image_data.cam_from_world
    if callable(cfw):
        cfw = cfw()
    return cfw.rotation.matrix(), cfw.translation


def get_intrinsics(camera):
    p = camera.params
    model = str(camera.model)
    if 'SIMPLE' in model or len(p) == 3:
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


def transfer_masks(points, normals, rec, mask_cache, max_images=MAX_IMAGES):
    n = len(points)
    votes   = np.zeros((n, N_CLASSES), dtype=np.float64)
    n_views = np.zeros(n, dtype=np.int32)

    images_list = list(rec.images.items())
    if max_images:
        images_list = images_list[:max_images]

    skipped = 0
    for img_id, img_data in images_list:
        img_name = img_data.name
        mask, conf_map = mask_cache.get(img_name)
        if mask is None:
            skipped += 1
            continue

        mask_h, mask_w = mask.shape
        cam = rec.cameras[img_data.camera_id]
        R, t = get_pose(img_data)
        fx, fy, cx, cy = get_intrinsics(cam)

        pts_cam = (R @ points.T).T + t
        in_front = pts_cam[:, 2] > 0.1
        z_safe = np.where(in_front, pts_cam[:, 2], 1.0)
        u = fx * pts_cam[:, 0] / z_safe + cx
        v = fy * pts_cam[:, 1] / z_safe + cy

        in_frame = (
            in_front
            & (u >= 0) & (u < mask_w - 1)
            & (v >= 0) & (v < mask_h - 1)
        )
        if in_frame.sum() == 0:
            continue

        idx = np.where(in_frame)[0]
        u_i = u[idx].astype(int).clip(0, mask_w - 1)
        v_i = v[idx].astype(int).clip(0, mask_h - 1)

        cls_vals  = mask[v_i, u_i].astype(np.int32)
        conf_vals = conf_map[v_i, u_i].astype(np.float32)

        if normals is not None:
            cam_pos_world = -R.T @ t
            dirs = cam_pos_world[np.newaxis, :] - points[idx]
            norms_len = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
            dirs_unit = dirs / norms_len
            cos_theta = np.abs((dirs_unit * normals[idx]).sum(axis=1)).clip(0, 1)
        else:
            cos_theta = np.ones(len(idx), dtype=np.float32)

        weight = (conf_vals * cos_theta).astype(np.float64)

        for c in range(N_CLASSES):
            class_mask = cls_vals == c
            if class_mask.any():
                np.add.at(votes[:, c], idx[class_mask], weight[class_mask])
        np.add.at(n_views, idx, 1)

    total_votes   = votes.sum(axis=1)
    surface_class = votes.argmax(axis=1).astype(np.uint8)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_conf = np.where(
            total_votes > 1e-9,
            votes.max(axis=1) / (total_votes + 1e-9),
            0.0
        ).astype(np.float32)

    uncertain = (n_views < MIN_VIEWS) | (total_votes < MIN_CONF)
    surface_class[uncertain] = 0
    class_conf[uncertain]    = 0.0

    return surface_class, class_conf, n_views, skipped


def plot_report(points, surface_class, class_conf, n_views, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    colors = np.array([
        [0.75, 0.75, 0.75],
        [0.23, 0.47, 1.0],
    ])
    pt_colors = colors[surface_class]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Классы поверхностей в 3D", fontsize=12, fontweight='bold')
    kw = dict(s=1.5, linewidths=0)

    axes[0].scatter(points[:, 0], points[:, 1], c=pt_colors, **kw)
    axes[0].set_title("Классы (XY)")
    axes[0].set_aspect('equal')

    axes[1].scatter(points[:, 0], points[:, 2], c=pt_colors, **kw)
    axes[1].set_title("Классы (XZ)")
    axes[1].set_aspect('equal')

    sc = axes[2].scatter(points[:, 0], points[:, 1], c=class_conf,
                         cmap='RdYlGn', vmin=0, vmax=1, **kw)
    plt.colorbar(sc, ax=axes[2], label='Уверенность')
    axes[2].set_title("Уверенность")
    axes[2].set_aspect('equal')

    sc2 = axes[3].scatter(points[:, 0], points[:, 1], c=n_views,
                          cmap='YlOrRd', **kw)
    plt.colorbar(sc2, ax=axes[3], label='Снимков')
    axes[3].set_title("Число снимков")
    axes[3].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "surface_classes_report.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def visualize_3d_classes(points, surface_class, save_dir):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((len(points), 3))
    colors[surface_class == 0] = [0.75, 0.75, 0.75]
    colors[surface_class == 1] = [0.23, 0.47, 1.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path = os.path.join(save_dir, "surface_classes_3d.ply")
    o3d.io.write_point_cloud(out_path, pcd)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Surface Classes",
        width=1400, height=900,
    )


def visualize_3d_confidence(points, class_conf, save_dir):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((len(points), 3))
    for i in range(len(points)):
        c = class_conf[i]
        if c < 0.5:
            colors[i] = [1.0, c * 2, 0.0]
        else:
            colors[i] = [2.0 - c * 2, 1.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path = os.path.join(save_dir, "confidence_3d.ply")
    o3d.io.write_point_cloud(out_path, pcd)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Classification Confidence",
        width=1400, height=900,
    )


def main():
    rec, points, normals, mask_files = load_inputs()
    cache = MaskCache(MASKS_DIR)

    surface_class, class_conf, n_views, skipped = transfer_masks(
        points, normals, rec, cache, MAX_IMAGES
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "surface_class.npy"),    surface_class)
    np.save(os.path.join(OUTPUT_DIR, "class_confidence.npy"), class_conf)
    np.save(os.path.join(OUTPUT_DIR, "n_views.npy"),          n_views)

    csv_path = os.path.join(FEATURES_DIR, "features.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['surface_class']    = surface_class
        df['class_confidence'] = class_conf
        df['n_views']          = n_views
        df.to_csv(csv_path, index=False)

    plot_report(points, surface_class, class_conf, n_views, OUTPUT_DIR)
    visualize_3d_classes(points, surface_class, OUTPUT_DIR)
    visualize_3d_confidence(points, class_conf, OUTPUT_DIR)

    n_total = len(points)
    n_norm  = (surface_class == 0).sum()
    n_glass = (surface_class == 1).sum()
    n_unc   = (class_conf == 0).sum()

    print(f"\nОбработано точек: {n_total:,}")
    print(f"Обычная поверхность: {n_norm:,} ({100*n_norm/n_total:.1f}%)")
    print(f"Стекло:              {n_glass:,} ({100*n_glass/n_total:.1f}%)")
    print(f"Не удалось классифицировать: {n_unc:,}")
    if skipped:
        print(f"Снимков без маски: {skipped}")
    print(f"Средняя уверенность: {class_conf[class_conf > 0].mean():.3f}")
    print(f"Результаты сохранены в: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()