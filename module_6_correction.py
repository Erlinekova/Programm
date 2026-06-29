import os
from collections import deque

import numpy as np
import open3d as o3d
from scipy import ndimage
from PIL import Image

from config import module6, BASE_DIR


TRUCK_PLY   = str(module6["truck_ply"])
SPARSE_PATH = str(module6["sparse_path"])
MASKS_DIR   = str(module6["masks_dir"])
T_PATH      = str(module6["transform_path"])
OUTPUT_DIR  = str(module6["output_dir"])
OUTPUT_PLY  = os.path.join(OUTPUT_DIR, module6["output_ply"])

_cam = module6["camera"]
FX = _cam["fx"]
FY = _cam["fy"]
CX = _cam["cx"]
CY = _cam["cy"]
W  = _cam["width"]
H  = _cam["height"]

WINDOWS     = module6["windows"]
FRONT_BAND  = module6["front_band"]
FACE_THRESH = module6["face_threshold"]
GRID_STEP   = module6["grid_step"]
DILATE_M    = module6["dilate_meters"]
FRAME_BAND  = module6["frame_band"]
HOLE_GROW   = module6["hole_growth_meters"]
VOXEL_DEDUP = module6["voxel_dedup"]
RED         = module6["red_color"]


def quat_to_R(qw, qx, qy, qz):
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def load_cam(name):
    for line in open(os.path.join(SPARSE_PATH, "images.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if len(t) >= 10 and t[-1] == name:
            R = quat_to_R(*map(float, t[1:5]))
            tvec = np.array(list(map(float, t[5:8])))
            return R, tvec
    raise ValueError("нет камеры для " + name)


def project(Pcol, R, t):
    Xc = Pcol @ R.T + t
    z = Xc[:, 2]
    fr = z > 1e-9
    u = np.where(fr, FX * Xc[:, 0] / np.where(fr, z, 1) + CX, -1)
    v = np.where(fr, FY * Xc[:, 1] / np.where(fr, z, 1) + CY, -1)
    return u, v, z, fr


def near_cut(zz, nb=80, gf=0.05):
    zmin, zmax = np.percentile(zz, [0.5, 99.5])
    if zmax - zmin < 1e-6:
        return zmax + 1e-3

    h, e = np.histogram(zz, bins=nb, range=(zmin, zmax))
    fl = max(3, 0.01 * h.sum())

    i = 0
    while i < nb and h[i] < fl:
        i += 1
    if i >= nb:
        return zmax + 1e-3

    pk = h[i]
    j = i
    while j < nb:
        pk = max(pk, h[j])
        if h[j] < max(3, gf * pk):
            break
        j += 1

    if j >= nb:
        return zmax + 1e-3
    return e[j]


def mask_roi(mask_png, area_min=2000):
    a = np.asarray(Image.open(mask_png).convert("RGB"))
    h, w = a.shape[:2]
    blue = (a[..., 2].astype(int) > 150) & (a[..., 0] < 100) & (a[..., 1] < 100)

    vis = np.zeros_like(blue)
    best = None
    ys, xs = np.where(blue)

    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if vis[sy, sx]:
            continue
        q = deque([(sy, sx)])
        vis[sy, sx] = 1
        x1 = x2 = sx
        y1 = y2 = sy
        ar = 0
        while q:
            cy, cx = q.popleft()
            ar += 1
            x1 = min(x1, cx)
            x2 = max(x2, cx)
            y1 = min(y1, cy)
            y2 = max(y2, cy)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and blue[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = 1
                        q.append((ny, nx))
        if ar >= area_min and (best is None or ar > best[4]):
            best = (x1, y1, x2, y2, ar)

    return best, (w, h)


def fill_window(Pcol, Ncol, R, t, roi):
    """Заливка проёма стекла на RANSAC-плоскости: прямоугольник маски как база
    плюс геодезическое наращивание базы до реконструированной рамки окна."""
    x1, y1, x2, y2 = roi
    pad = 25
    ex1, ey1, ex2, ey2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad

    u, v, z, fr = project(Pcol, R, t)
    inroi = fr & (u >= ex1) & (u <= ex2) & (v >= ey1) & (v <= ey2)
    if inroi.sum() < 500:
        return np.zeros((0, 3))

    zmin = np.percentile(z[inroi], 1)
    viewdir = R.T @ np.array([0, 0, 1.0])
    facing = (Ncol @ viewdir) < FACE_THRESH
    front = inroi & (z <= zmin + FRONT_BAND) & facing

    f3 = Pcol[front]
    if len(f3) < 100:
        return np.zeros((0, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(f3)
    pl, inl = pcd.segment_plane(0.02, 3, 3000)
    nrm = np.array(pl[:3])
    nrm /= np.linalg.norm(nrm)
    if nrm @ viewdir > 0:
        nrm = -nrm

    fin = f3[inl]
    p0 = fin.mean(0)
    a0 = np.array([0, 1.0, 0]) if abs(nrm @ np.array([0, 1.0, 0])) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(nrm, a0)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)

    A = (fin - p0) @ e1
    B = (fin - p0) @ e2
    mg = 0.20
    a_lo, b_lo = A.min() - mg, B.min() - mg
    ga, gb = np.meshgrid(np.arange(a_lo, A.max() + mg, GRID_STEP),
                         np.arange(b_lo, B.max() + mg, GRID_STEP))
    Hg, Wd = ga.shape
    G = p0 + ga.ravel()[:, None] * e1 + gb.ravel()[:, None] * e2

    Xc = G @ R.T + t
    zc = Xc[:, 2]
    frg = zc > 1e-9
    gu = FX * Xc[:, 0] / np.where(frg, zc, 1) + CX
    gv = FY * Xc[:, 1] / np.where(frg, zc, 1) + CY

    base = (frg & (gu >= x1) & (gu <= x2) & (gv >= y1) & (gv <= y2)).reshape(Hg, Wd)
    keep = base.copy()

    rp = Pcol[inroi]
    dpl = np.abs((rp - p0) @ nrm)
    near = rp[dpl < FRAME_BAND]
    ia = ((((near - p0) @ e1) - a_lo) / GRID_STEP).astype(int)
    ib = ((((near - p0) @ e2) - b_lo) / GRID_STEP).astype(int)
    ok = (ia >= 0) & (ia < Wd) & (ib >= 0) & (ib < Hg)
    solid = np.zeros((Hg, Wd), bool)
    solid[ib[ok], ia[ok]] = True
    solid = ndimage.binary_closing(solid, iterations=1)

    if base.any():
        bys, bxs = np.where(base)
        D = int(DILATE_M / GRID_STEP)
        allowed = np.zeros((Hg, Wd), bool)
        allowed[max(0, bys.min() - D):bys.max() + D + 1,
                max(0, bxs.min() - D):bxs.max() + D + 1] = True
        st = ndimage.generate_binary_structure(2, 1)
        grow = base.copy()
        for _ in range(int(HOLE_GROW / GRID_STEP)):
            grow = (ndimage.binary_dilation(grow, st) & ~solid & allowed) | base
        keep = grow & frg.reshape(Hg, Wd)

    Xg = G[keep.ravel()]
    if len(Xg) > 50:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(Xg)
        p, _ = p.remove_statistical_outlier(20, 2.0)
        Xg = np.asarray(p.points)
    return Xg


def save_ply(red_truck):
    DT = np.dtype([
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    with open(TRUCK_PLY, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            hdr += f.readline()
        n = int([l for l in hdr.decode().split("\n") if l.startswith("element vertex")][0].split()[-1])
        data = np.fromfile(f, dtype=DT, count=n)

    g = np.zeros(len(red_truck), dtype=DT)
    g["x"], g["y"], g["z"] = red_truck[:, 0], red_truck[:, 1], red_truck[:, 2]
    g["r"], g["g"], g["b"] = RED

    allv = np.concatenate([data, g])
    nh = hdr.replace(("element vertex %d" % n).encode(),
                     ("element vertex %d" % len(allv)).encode())

    os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)
    with open(OUTPUT_PLY, "wb") as f:
        f.write(nh)
        allv.tofile(f)
    print("сохранено:", OUTPUT_PLY, "| +", len(red_truck), "красных, всего", len(allv))


def main():
    T = np.load(T_PATH)
    Minv = np.linalg.inv(T[:3, :3])

    truck = o3d.io.read_point_cloud(TRUCK_PLY)
    if not truck.has_normals():
        print("нормалей нет — оцениваю...")
        truck.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.05, 30))

    Pcol = np.asarray(truck.points) @ T[:3, :3].T + T[:3, 3]
    Ncol = np.asarray(truck.normals) @ T[:3, :3].T
    Ncol /= np.linalg.norm(Ncol, axis=1, keepdims=True) + 1e-9

    allred = []
    for win, frame in WINDOWS.items():
        best, (mw, mh) = mask_roi(os.path.join(MASKS_DIR, frame[:-4] + "_overlay.png"))
        if best is None:
            print(f"  {win}/{frame}: нет маски")
            continue

        R, t = load_cam(frame)
        sx, sy = W / mw, H / mh
        x1, y1, x2, y2, ar = best
        roi = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

        pts = fill_window(Pcol, Ncol, R, t, roi)
        print(f"  {win}/{frame}: {len(pts)} точек")
        if len(pts):
            allred.append(pts)

    red_col = np.vstack(allred) if allred else np.zeros((0, 3))
    if len(red_col):
        vox = np.round(red_col / VOXEL_DEDUP).astype(np.int64)
        _, uq = np.unique(vox, axis=0, return_index=True)
        red_col = red_col[np.sort(uq)]

    red_truck = (red_col - T[:3, 3]) @ Minv.T
    np.save(os.path.join(str(BASE_DIR), "output_red_points.npy"), red_truck)
    save_ply(red_truck)


if __name__ == "__main__":
    main()
