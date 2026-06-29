import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from config import module4_1


def run_segmentation(model_path, images_dir, masks_dir, conf_thresh=0.25):
    model = YOLO(model_path)
    img_dir = Path(images_dir)
    out_dir = Path(masks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    if not img_files:
        print(f"Нет изображений в {img_dir}")
        return

    processed = 0
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        conf_map = np.zeros((h, w), dtype=np.float32)
        results = model(img_path, conf=conf_thresh, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
                if int(cls) != 0:
                    continue
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_id = 1
                roi_mask = mask[y1:y2, x1:x2]
                roi_conf = conf_map[y1:y2, x1:x2]
                new_conf = np.full(roi_mask.shape, conf, dtype=np.float32)
                roi_mask[:] = np.where(new_conf > roi_conf, class_id, roi_mask)
                roi_conf[:] = np.maximum(roi_conf, new_conf)
        stem = img_path.stem
        cv2.imwrite(str(out_dir / f'{stem}_mask.png'), mask)
        cv2.imwrite(str(out_dir / f'{stem}_confidence.png'),
                    (conf_map * 255).astype(np.uint8))
        overlay = img.copy()
        overlay[mask == 1] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f'{stem}_overlay.png'), overlay)
        processed += 1

    print(f"Обработано изображений: {processed}")
    print(f"Маски сохранены в: {out_dir}")


if __name__ == '__main__':
    run_segmentation(
        model_path  = str(module4_1["model_path"]),
        images_dir  = str(module4_1["images_dir"]),
        masks_dir   = str(module4_1["masks_dir"]),
        conf_thresh = module4_1["conf_thresh"],
    )