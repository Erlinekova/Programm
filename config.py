from pathlib import Path

BASE_DIR = Path(r"F:\UNIVER\original\models\Truck\corrected")
DATA_ROOT = Path(r"F:\UNIVER\original\models\Truck")

DENSE_PLY = r"E:\UNIVER\data\Truck\Truck.ply"
VOXEL_SIZE = 0.003
K_NEIGHBORS = 10
POINT_SIZE_3D = 2.0

module1 = {
    "ply_path": DENSE_PLY,
    "voxel_size": VOXEL_SIZE,
    "point_size": POINT_SIZE_3D,
    "use_input_prompt": True,
}

module2 = {
    "ply_path": DENSE_PLY,
    "voxel_size": 0.01,
    "k_neighbors": K_NEIGHBORS,
    "max_points": 50000,
}

module3 = {
    "sparse_path": DATA_ROOT / "Dense" / "sparse",
    "images_dir": DATA_ROOT / "Dense" / "images",
    "features_dir": DATA_ROOT / "features",
    "output_dir": DATA_ROOT / "features",
    "n_bootstrap": 200,
    "ci_percentile": 90,
    "ci_threshold": 0.20,
    "k_neighbors": K_NEIGHBORS,
    "pca_weight_decay": 0.5,
    "pai_patch_size": 11,
}

module4_1 = {
    "model_path": r"F:\UNIVER\original\runs\segment\glass_detector\weights\best.pt",
    "images_dir": DATA_ROOT / "Source_images",
    "masks_dir": DATA_ROOT / "neiro" / "masks",
    "conf_thresh": 0.55,
}

module4_2 = {
    "masks_dir": DATA_ROOT / "neiro" / "masks",
}

module4_3 = {
    "sparse_path": DATA_ROOT / "Dense" / "sparse",
    "images_dir": DATA_ROOT / "Dense" / "images",
    "masks_dir": DATA_ROOT / "neiro" / "masks",
    "features_dir": DATA_ROOT / "features",
    "output_dir": DATA_ROOT / "features",
    "min_views": 10,
    "min_conf": 0.3,
    "max_images": None,
    "n_classes": 3,
}

module5 = {
    "features_dir": DATA_ROOT / "features",
    "output_dir": DATA_ROOT / "features",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 8,
    "num_leaves": 50,
    "shap_sample_size": 10000,
    "glass_threshold_search": True,
}

module6 = {
    "truck_ply": DATA_ROOT / "Truck.ply",
    "sparse_path": DATA_ROOT / "Dense" / "sparse",
    "masks_dir": DATA_ROOT / "neiro" / "masks",
    "output_dir": BASE_DIR / "output",
    "output_ply": "truck_glass_red.ply",
    "transform_path": BASE_DIR / "T_truck2colmap.npy",
    "camera": {
        "fx": 1158.774071214973,
        "fy": 1158.774071214973,
        "cx": 979.0,
        "cy": 542.5,
        "width": 1958,
        "height": 1085,
    },
    "windows": {
        "windshield": "000064.jpg",
        "left_side": "000043.jpg",
        "right_side": "000101.jpg",
    },
    "front_band": 0.6,
    "face_threshold": -0.3,
    "grid_step": 0.004,
    "dilate_meters": 0.12,
    "frame_band": 0.10,
    "hole_growth_meters": 0.06,
    "voxel_dedup": 0.005,
    "red_color": (255, 0, 0),
}