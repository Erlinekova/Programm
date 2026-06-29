import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from config import module1


if module1.get("use_input_prompt", False):
    PLY_PATH = input(f"Путь до .ply: ").strip()
else:
    PLY_PATH = str(module1["ply_path"])


if not os.path.exists(PLY_PATH):
    raise FileNotFoundError(f"Файл не найден: {PLY_PATH}")

VOXEL_SIZE    = module1["voxel_size"]
POINT_SIZE_3D = module1["point_size"]

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError("Облако пустое")
    print(f"Загружено точек: {len(pcd.points):,}")
    return pcd

def optimize_point_cloud(pcd: o3d.geometry.PointCloud, 
                         voxel_size: float) -> o3d.geometry.PointCloud:
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled

def visualize_3d(pcd: o3d.geometry.PointCloud, point_size: float = 2.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud", width=1400, height=900)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pcd = load_point_cloud(PLY_PATH)
    pcd = optimize_point_cloud(pcd, VOXEL_SIZE)

    points = np.asarray(pcd.points)
    visualize_3d(pcd, POINT_SIZE_3D)