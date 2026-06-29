import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from config import module2


PLY_PATH    = str(module2["ply_path"])
VOXEL_SIZE  = module2["voxel_size"]
K_NEIGHBORS = module2["k_neighbors"]
MAX_POINTS  = module2["max_points"]

if not os.path.exists(PLY_PATH):
    raise FileNotFoundError(f"Файл не найден: {PLY_PATH}")


def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    print(f"Исходных точек: {len(pcd.points):,}")
    return pcd


def optimize_point_cloud(pcd):
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    points = np.asarray(pcd.points)
    if len(points) > MAX_POINTS:
        idx = np.random.choice(len(points), MAX_POINTS, replace=False)
        points = points[idx]
        pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def compute_normals(pcd):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=K_NEIGHBORS)
    )
    return np.asarray(pcd.normals)


def compute_density(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=K_NEIGHBORS)
    density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-12)
    return density


def compute_curvature(points):
    tree = cKDTree(points)
    _, indices = tree.query(points, k=K_NEIGHBORS)
    curvature = np.zeros(len(points))
    for i in range(len(points)):
        neighbors = points[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)
        curvature[i] = eigvals[0] / (eigvals.sum() + 1e-12)
    return curvature


def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + 1e-12)


def colorize_point_cloud(pcd, feature):
    feature = normalize_feature(feature)
    feature = np.power(feature, 0.5)
    colors = np.zeros((len(feature), 3))
    colors[:, 0] = feature
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize(pcd, title):
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
        width=1400,
        height=900,
    )


def compute_nn_distance(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    return distances[:, 1]


def compute_density_variation(points, density):
    tree = cKDTree(points)
    _, indices = tree.query(points, k=K_NEIGHBORS)
    variation = np.zeros(len(points))
    for i in range(len(points)):
        local_density = density[indices[i]]
        variation[i] = np.std(local_density)
    return variation


def compute_statistics(density, curvature, nn_distance):
    print("\nСтатистика облака:")
    print(f"Density mean:    {np.mean(density):.4f}")
    print(f"Density std:     {np.std(density):.4f}")
    print(f"Curvature mean:  {np.mean(curvature):.4f}")
    print(f"Curvature std:   {np.std(curvature):.4f}")
    print(f"NN dist mean:    {np.mean(nn_distance):.4f}")
    print(f"NN dist std:     {np.std(nn_distance):.4f}")



if __name__ == "__main__":
    pcd = load_point_cloud(PLY_PATH)
    pcd = optimize_point_cloud(pcd)
    points = np.asarray(pcd.points)

    normals = compute_normals(pcd)

    density = compute_density(points)
    density_pcd = o3d.geometry.PointCloud()
    density_pcd.points = pcd.points
    density_pcd = colorize_point_cloud(density_pcd, density)
    visualize(density_pcd, "Density Map")

    curvature = compute_curvature(points)
    curvature_pcd = o3d.geometry.PointCloud()
    curvature_pcd.points = pcd.points
    curvature_pcd = colorize_point_cloud(curvature_pcd, curvature)
    visualize(curvature_pcd, "Curvature Map")