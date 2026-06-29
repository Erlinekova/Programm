import cv2
import numpy as np
from pathlib import Path

from config import module4_2


masks_dir = Path(module4_2["masks_dir"])

if not masks_dir.exists():
    print(f"Папка не найдена: {masks_dir}")
    exit()

mask_files = sorted(masks_dir.glob("*_mask.png"))
if not mask_files:
    print(f"Нет масок в {masks_dir}")
    exit()

found_masks = []
for f in mask_files:
    mask = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if mask is None:
        continue
    uniques = np.unique(mask)
    if 1 in uniques or 2 in uniques:
        overlay_path = f.parent / f"{f.stem.replace('_mask', '_overlay')}.png"
        found_masks.append({
            'mask': f,
            'overlay': overlay_path if overlay_path.exists() else None,
            'has_glass': 1 in uniques
        })

if not found_masks:
    print("Масок со стеклом не найдено")
else:
    print(f"Найдено масок с дефектами: {len(found_masks)}")
    print("Цвета на overlay:")
    print("Синий   = Стекло")

    print("\nФайлы:")
    for m in found_masks:
        status = []
        if m['has_glass']:
            status.append("стекло")
        if m['overlay']:
            print(f"{m['overlay'].name}")