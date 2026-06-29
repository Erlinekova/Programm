import os
import sys
import glob
import yaml

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO


DATA_YAML = r"F:\UNIVER\original\dataset\Mirrow\data.yaml"

if not os.path.exists(DATA_YAML):
    print(f"data.yaml не найден: {DATA_YAML}")
    sys.exit(1)

print(f"data.yaml: {DATA_YAML}")

model = YOLO("yolov8s-seg.pt")

print("Запуск обучения...")
results = model.train(
    data=DATA_YAML,
    epochs=25,
    imgsz=640,
    batch=4,
    device="cpu",
    name="glass_detector",
    patience=10,
    augment=True,
    verbose=True,
    save=True,
    save_period=5,
)

best_pt = os.path.join(results.save_dir, "weights", "best.pt")
last_pt = os.path.join(results.save_dir, "weights", "last.pt")

print(f"\nРезультаты: {results.save_dir}")
print(f"best.pt: {best_pt} (существует: {os.path.exists(best_pt)})")
print(f"last.pt: {last_pt} (существует: {os.path.exists(last_pt)})")

if not os.path.exists(best_pt):
    print("best.pt не создан, используйте last.pt")

try:
    with open(DATA_YAML, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train_imgs = cfg.get('train', '')
    if train_imgs and os.path.exists(train_imgs):
        test_img = glob.glob(os.path.join(train_imgs, "*.jpg"))[:1]
        if test_img:
            print(f"\nТест на {test_img[0]}")
            model.predict(test_img[0], conf=0.25, save=True)
except Exception as e:
    print(f"Тест пропущен: {e}")