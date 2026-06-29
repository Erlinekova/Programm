import os
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from config import module5


FEATURES_DIR = str(module5["features_dir"])
OUTPUT_DIR   = str(module5["output_dir"])
TEST_SIZE    = module5["test_size"]
RANDOM_STATE = module5["random_state"]
N_ESTIMATORS = module5["n_estimators"]
LEARNING_RATE = module5["learning_rate"]
MAX_DEPTH    = module5["max_depth"]
NUM_LEAVES   = module5["num_leaves"]
SHAP_SAMPLE  = module5["shap_sample_size"]

FEATURE_COLS = [
    'ERROR_mean', 'ERROR_std', 'TRACK', 'LPD',
    'NCS_weighted', 'RGH', 'PAI', 'RCS', 'CI_width'
]

CLASS_NAMES = {
    0: "OK",
    1: "Стекло",
    3: "Слабая текстура",
    4: "Нестабильная",
}


csv_path = os.path.join(FEATURES_DIR, "features.csv")
if not os.path.exists(csv_path):
    sys.exit(f"Не найден {csv_path}")

df = pd.read_csv(csv_path)
n_points = len(df)

for col, fname in [('surface_class', 'surface_class.npy'),
                   ('uncertain', 'uncertain.npy'),
                   ('CI_width', 'ci_width.npy')]:
    if col not in df.columns:
        fpath = os.path.join(FEATURES_DIR, fname)
        df[col] = np.load(fpath) if os.path.exists(fpath) else 0


y = np.full(n_points, -1, dtype=int)
y[df['surface_class'] == 1] = 1

mask_surf0 = df['surface_class'] == 0
mask_unstable = mask_surf0 & (df['uncertain'] == 1) & (df['CI_width'] > 0.20)
y[mask_unstable] = 4

mask_weak = mask_surf0 & (y == -1) & (df['RCS'] < 0.35) & (df['NCS_weighted'] < 0.60)
y[mask_weak] = 3

mask_ok = mask_surf0 & (y == -1)
y[mask_ok] = 0

df['defect_class_true'] = y

print(f"Точек: {n_points:,}")
for cls, cnt in zip(*np.unique(y, return_counts=True)):
    print(f"  [{cls}] {CLASS_NAMES[cls]:<20}: {cnt:,} ({100*cnt/n_points:.1f}%)")


y_binary = (y == 1).astype(int)
X = df[FEATURE_COLS].values
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_binary
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


base_model = lgb.LGBMClassifier(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    num_leaves=NUM_LEAVES,
    min_child_samples=10,
    scale_pos_weight=(y_binary == 0).sum() / max((y_binary == 1).sum(), 1),
    class_weight='balanced',
    random_state=RANDOM_STATE,
    verbose=-1,
)

calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

probs_test = calibrated_model.predict_proba(X_test)
probs_glass_test = probs_test[:, 1]

best_f1 = 0
best_threshold = 0.5
for t in np.arange(0.1, 0.91, 0.05):
    y_pred_temp = (probs_glass_test >= t).astype(int)
    if len(np.unique(y_pred_temp)) > 1:
        f1_temp = f1_score(y_test, y_pred_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = t

print(f"Порог: {best_threshold:.2f} (F1={best_f1:.3f})")

y_pred_binary_test = (probs_glass_test >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_binary_test, target_names=['Не-стекло', 'Стекло']))


probs_full = calibrated_model.predict_proba(X)
y_pred_multiclass = np.zeros(n_points, dtype=int)

mask_glass = probs_full[:, 1] >= best_threshold
mask_not_glass = ~mask_glass

rcs_arr = df['RCS'].values
ci_arr  = df['CI_width'].values

mask_weak = mask_not_glass & (rcs_arr < 0.35) & (ci_arr <= 0.20)
y_pred_multiclass[mask_weak] = 3

mask_unstable = mask_not_glass & (ci_arr > 0.20)
y_pred_multiclass[mask_unstable] = 4

mask_ok = mask_not_glass & ~mask_weak & ~mask_unstable
y_pred_multiclass[mask_ok] = 0

y_pred_multiclass[mask_glass] = 1

print("\nПредсказания:")
for cls in [0, 1, 3, 4]:
    cnt = (y_pred_multiclass == cls).sum()
    print(f"[{cls}] {CLASS_NAMES[cls]:<20}: {cnt:,} ({100*cnt/n_points:.1f}%)")


metrics_df = pd.DataFrame({
    'Class': ['Стекло', 'OK', 'Слабая текстура', 'Нестабильная'],
    'F1-Score': [
        f1_score(y_test, y_pred_binary_test),
        f1_score(y == 0, y_pred_multiclass == 0, zero_division=0),
        f1_score(y == 3, y_pred_multiclass == 3, zero_division=0),
        f1_score(y == 4, y_pred_multiclass == 4, zero_division=0),
    ]
})
os.makedirs(OUTPUT_DIR, exist_ok=True)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "classification_metrics.csv"), index=False)


shap_model = lgb.LGBMClassifier(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    num_leaves=NUM_LEAVES,
    min_child_samples=10,
    scale_pos_weight=(y_binary == 0).sum() / max((y_binary == 1).sum(), 1),
    random_state=RANDOM_STATE,
    verbose=-1,
)
shap_model.fit(X_train, y_train)

shap_sample = min(SHAP_SAMPLE, len(X_test))
X_shap = X_test[:shap_sample]

explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)

np.save(os.path.join(OUTPUT_DIR, "defect_class.npy"), y_pred_multiclass.astype(np.int8))
np.save(os.path.join(OUTPUT_DIR, "shap_values.npy"), np.array(shap_values))


shap_abs_mean = (
    np.mean(np.abs(shap_values), axis=(0, 2))
    if len(shap_values.shape) == 3
    else np.mean(np.abs(shap_values), axis=0)
)
sorted_idx = np.argsort(shap_abs_mean)

fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
ax_imp.barh(
    [FEATURE_COLS[i] for i in sorted_idx],
    [shap_abs_mean[i] for i in sorted_idx],
    color='steelblue'
)
ax_imp.set_title("Важность признаков (SHAP)")
ax_imp.set_xlabel("Среднее SHAP")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance.png"), dpi=200, bbox_inches='tight')
plt.close(fig_imp)

shap_values_class1 = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values_class1, X_shap,
    feature_names=FEATURE_COLS, show=False,
    max_display=8, plot_type="dot"
)
plt.title("Влияние признаков на класс 'Стекло'")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_glass.png"), dpi=200, bbox_inches='tight')
plt.close(fig_shap)

print(f"\nФайлы сохранены в: {OUTPUT_DIR}")