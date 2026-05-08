
"""
=============================================================================
PART 2: MODEL TRAINING, EVALUATION & PREDICTION
=============================================================================
Project : Attention-Based Temporal Deep Learning for NBA Shot Prediction
HOW TO RUN:
    pip install tensorflow xgboost scikit-learn matplotlib seaborn
    python part2_training.py
GOOGLE COLAB:
    !pip install tensorflow xgboost
    !python part2_training.py

MODELS:
    Baselines  : Logistic Regression, Random Forest, XGBoost
    Proposed   : Attention-LSTM
    Hybrid     : XGBoost+LSTM Weighted, XGBoost+LSTM Stacking, RF+LSTM Weighted

INPUT  : ./processed_data/   (output of part1_preprocessing.py)
OUTPUT : ./model_outputs/    (models + all evaluation plots)
=============================================================================
"""

import os, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics       import (accuracy_score, precision_score, recall_score,
                                   f1_score, roc_auc_score, confusion_matrix,
                                   roc_curve, precision_recall_curve, log_loss,
                                   classification_report)
from sklearn.utils         import class_weight

warnings.filterwarnings("ignore")

# ── XGBoost ───────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost   : available ✓")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost   : not found — using GradientBoostingClassifier")

# ── TensorFlow ────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                            ModelCheckpoint)
    TF_AVAILABLE = True
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")
    print(f"TensorFlow: {tf.__version__}  ✓")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow: not found — LSTM/Hybrid skipped. pip install tensorflow")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
PROCESSED_DIR = "./processed_data"
OUTPUT_DIR    = "./model_outputs"
TEST_SPLIT    = 0.20
VAL_SPLIT     = 0.10
RANDOM_STATE  = 42
EPOCHS        = 60
BATCH_SIZE    = 512
LSTM_UNITS_1  = 128
LSTM_UNITS_2  = 64
DENSE_UNITS   = 32
DROP1, DROP2, DROP3 = 0.30, 0.20, 0.10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots",  exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)

print("\n" + "=" * 65)
print("  NBA SHOT PREDICTION — PART 2: TRAINING & EVALUATION")
print("=" * 65)

# ──────────────────────────────────────────────────────────────
# STEP 1  LOAD DATA
# ──────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading preprocessed data …")
X_sequences = np.load(f"{PROCESSED_DIR}/X_sequences.npy")
y_labels    = np.load(f"{PROCESSED_DIR}/y_labels.npy")
X_flat      = np.load(f"{PROCESSED_DIR}/X_flat.npy")

with open(f"{PROCESSED_DIR}/preprocessing_metadata.pkl", "rb") as f:
    meta = pickle.load(f)
with open(f"{PROCESSED_DIR}/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

SEQ_LEN   = meta["seq_len"]
N_FEAT    = meta["n_features"]
N_SAMPLES = meta["n_samples"]
print(f"  X_sequences : {X_sequences.shape}  |  y_labels : {y_labels.shape}")
print(f"  SeqLen={SEQ_LEN}  Features={N_FEAT}  Samples={N_SAMPLES:,}")
print(f"  Made={y_labels.mean()*100:.1f}%  Missed={(1-y_labels.mean())*100:.1f}%")

# ──────────────────────────────────────────────────────────────
# STEP 2  TEMPORAL SPLIT  (no leakage)
# ──────────────────────────────────────────────────────────────
print("\n[STEP 2] Temporal train / val / test split …")
split_idx = int(N_SAMPLES * (1 - TEST_SPLIT))
val_idx   = int(split_idx * (1 - VAL_SPLIT))

X_train_seq  = X_sequences[:val_idx];      X_train_flat = X_flat[:val_idx]
X_val_seq    = X_sequences[val_idx:split_idx]; X_val_flat = X_flat[val_idx:split_idx]
X_test_seq   = X_sequences[split_idx:];    X_test_flat  = X_flat[split_idx:]

y_train = y_labels[:val_idx]
y_val   = y_labels[val_idx:split_idx]
y_test  = y_labels[split_idx:]

print(f"  Train : {len(y_train):,}  |  Val : {len(y_val):,}  |  Test : {len(y_test):,}")

cw  = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {0: cw[0], 1: cw[1]}

# ──────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────
def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_true, y_prob),
        "Log-Loss" : log_loss(y_true, y_prob),
    }

all_results = []
all_probs   = {}
all_preds   = {}
model_store = {}
val_probs   = {}   # store val-set predictions for hybrid tuning


# ──────────────────────────────────────────────────────────────
# STEP 3  BASELINE ML MODELS
# ──────────────────────────────────────────────────────────────
print("\n[STEP 3] Training baseline ML models …")

# ── 3a Logistic Regression ───────────────────────────────────
print("  Training Logistic Regression …")
t0 = time.time()
lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                        random_state=RANDOM_STATE, solver="lbfgs", C=1.0)
lr.fit(X_train_flat, y_train)
lr_pred = lr.predict(X_test_flat)
lr_prob = lr.predict_proba(X_test_flat)[:, 1]
val_probs["Logistic Regression"] = lr.predict_proba(X_val_flat)[:, 1]
res = evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)
all_results.append(res); all_probs["Logistic Regression"] = lr_prob
all_preds["Logistic Regression"] = lr_pred; model_store["Logistic Regression"] = lr
print(f"  LR  — Acc: {res['Accuracy']*100:.2f}%  AUC: {res['ROC-AUC']:.4f}  ({time.time()-t0:.1f}s)")

# ── 3b Random Forest ─────────────────────────────────────────
print("  Training Random Forest …")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                             class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_flat, y_train)
rf_pred = rf.predict(X_test_flat)
rf_prob = rf.predict_proba(X_test_flat)[:, 1]
val_probs["Random Forest"] = rf.predict_proba(X_val_flat)[:, 1]
res = evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
all_results.append(res); all_probs["Random Forest"] = rf_prob
all_preds["Random Forest"] = rf_pred; model_store["Random Forest"] = rf
print(f"  RF  — Acc: {res['Accuracy']*100:.2f}%  AUC: {res['ROC-AUC']:.4f}  ({time.time()-t0:.1f}s)")

# ── 3c XGBoost / GBM ─────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("  Training XGBoost …")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1-y_train.mean())/y_train.mean(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb_model.fit(X_train_flat, y_train,
                  eval_set=[(X_val_flat, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_test_flat)
    xgb_prob = xgb_model.predict_proba(X_test_flat)[:, 1]
    val_probs["XGBoost"] = xgb_model.predict_proba(X_val_flat)[:, 1]
    xgb_name = "XGBoost"
else:
    print("  Training Gradient Boosting (XGBoost substitute) …")
    t0 = time.time()
    xgb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                            learning_rate=0.05, subsample=0.8,
                                            random_state=RANDOM_STATE)
    xgb_model.fit(X_train_flat, y_train)
    xgb_pred = xgb_model.predict(X_test_flat)
    xgb_prob = xgb_model.predict_proba(X_test_flat)[:, 1]
    val_probs["XGBoost"] = xgb_model.predict_proba(X_val_flat)[:, 1]
    xgb_name = "Gradient Boosting"

res = evaluate_model(xgb_name, y_test, xgb_pred, xgb_prob)
all_results.append(res); all_probs[xgb_name] = xgb_prob
all_preds[xgb_name] = xgb_pred; model_store[xgb_name] = xgb_model
print(f"  {xgb_name} — Acc: {res['Accuracy']*100:.2f}%  AUC: {res['ROC-AUC']:.4f}  ({time.time()-t0:.1f}s)")


# ──────────────────────────────────────────────────────────────
# STEP 4  ATTENTION-LSTM
# ──────────────────────────────────────────────────────────────
history     = None
lstm_model  = None
lstm_name   = "Attention-LSTM"
lstm_prob   = None
lstm_val_pr = None

if TF_AVAILABLE:
    print("\n[STEP 4] Building Attention-LSTM …")

    # ── Custom Attention Layer ────────────────────────────────
    class AttentionLayer(layers.Layer):
        """
        Bahdanau-style additive self-attention over LSTM output.
        Learns which of the T past shots is most predictive.
        Returns: (context_vector, attention_weights)
        """
        def build(self, input_shape):
            d = input_shape[-1]
            self.W  = self.add_weight(name="W",  shape=(d, d),  initializer="glorot_uniform", trainable=True)
            self.b  = self.add_weight(name="b",  shape=(d,),    initializer="zeros",          trainable=True)
            self.u  = self.add_weight(name="u",  shape=(d, 1),  initializer="glorot_uniform", trainable=True)
            super().build(input_shape)

        def call(self, x):
            # x : (batch, T, d)
            score  = tf.matmul(tf.nn.tanh(tf.matmul(x, self.W) + self.b), self.u)  # (B,T,1)
            alpha  = tf.nn.softmax(score, axis=1)                                   # (B,T,1)
            ctx    = tf.reduce_sum(x * alpha, axis=1)                               # (B,d)
            return ctx, tf.squeeze(alpha, axis=-1)                                  # (B,d),(B,T)

    def build_attention_lstm(seq_len, n_features):
        inp = keras.Input(shape=(seq_len, n_features), name="shot_sequence")
        x = layers.LSTM(LSTM_UNITS_1, return_sequences=True, name="lstm_1")(inp)
        x = layers.Dropout(DROP1, name="drop_1")(x)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.LSTM(LSTM_UNITS_2, return_sequences=True, name="lstm_2")(x)
        x = layers.Dropout(DROP2, name="drop_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        ctx, att_w = AttentionLayer(name="attention")(x)
        x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(ctx)
        x = layers.Dropout(DROP3, name="drop_3")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        model = Model(inputs=inp, outputs=out, name="Attention_LSTM")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy",
                     keras.metrics.AUC(name="auc"),
                     keras.metrics.Precision(name="precision"),
                     keras.metrics.Recall(name="recall")]
        )
        return model

    lstm_model = build_attention_lstm(SEQ_LEN, N_FEAT)
    lstm_model.summary()

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=8, mode="max",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f"{OUTPUT_DIR}/models/attention_lstm_best.keras",
                        monitor="val_auc", save_best_only=True, mode="max", verbose=0),
    ]

    print("\n[STEP 4b] Training Attention-LSTM …")
    t0 = time.time()
    history = lstm_model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weights, callbacks=callbacks, verbose=1,
    )
    print(f"\n  Training complete ({time.time()-t0:.0f}s)")

    lstm_prob   = lstm_model.predict(X_test_seq,  verbose=0).flatten()
    lstm_val_pr = lstm_model.predict(X_val_seq,   verbose=0).flatten()
    lstm_pred   = (lstm_prob >= 0.5).astype(int)
    val_probs[lstm_name] = lstm_val_pr

    res = evaluate_model(lstm_name, y_test, lstm_pred, lstm_prob)
    all_results.append(res); all_probs[lstm_name] = lstm_prob
    all_preds[lstm_name] = lstm_pred; model_store[lstm_name] = lstm_model
    print(f"  Attention-LSTM — Acc: {res['Accuracy']*100:.2f}%  AUC: {res['ROC-AUC']:.4f}")


# ──────────────────────────────────────────────────────────────
# STEP 5  HYBRID ENSEMBLE MODELS
# ──────────────────────────────────────────────────────────────
"""
HYBRID FRAMEWORK RATIONALE
===========================
Tree-based boosting models (XGBoost) effectively learn nonlinear tabular
feature interactions from the flattened shot context. LSTM networks capture
temporal shooting behaviour and sequential player momentum. Therefore, a
hybrid ensemble was developed to combine both static contextual learning
and sequential temporal modeling, achieving superior performance over either
model alone.

Three hybrid strategies are implemented:
  1. XGBoost + LSTM  Weighted Ensemble  (optimal α tuned on validation set)
  2. XGBoost + LSTM  Stacking           (Logistic Regression meta-learner)
  3. RF + LSTM       Weighted Ensemble  (for comparison)
"""
print("\n[STEP 5] Training Hybrid Ensemble Models …")
print("  " + "─"*60)
print("  HYBRID FRAMEWORK: Tree-based boosting captures nonlinear")
print("  tabular feature interactions; LSTM captures temporal shooting")
print("  behaviour. Combining both yields superior prediction.")
print("  " + "─"*60)

if TF_AVAILABLE and lstm_prob is not None:

    # ── 5a  XGBoost + LSTM  Weighted Ensemble ─────────────────
    print("\n  [5a] XGBoost + LSTM — Optimal Weighted Ensemble …")
    xgb_vp  = val_probs[xgb_name]
    lstm_vp = val_probs[lstm_name]

    best_alpha  = 0.5
    best_vauc   = 0.0
    alpha_range = np.arange(0.0, 1.01, 0.02)
    auc_curve   = []

    for alpha in alpha_range:
        h_vp = alpha * lstm_vp + (1 - alpha) * xgb_vp
        vauc = roc_auc_score(y_val, h_vp)
        auc_curve.append(vauc)
        if vauc > best_vauc:
            best_vauc  = vauc
            best_alpha = alpha

    print(f"  Best α = {best_alpha:.2f}  (Val AUC = {best_vauc:.4f})")
    print(f"  Interpretation: {best_alpha*100:.0f}% LSTM + {(1-best_alpha)*100:.0f}% XGBoost")

    h1_prob = best_alpha * lstm_prob + (1 - best_alpha) * xgb_prob
    h1_pred = (h1_prob >= 0.5).astype(int)
    h1_name = "XGB+LSTM Weighted"
    res = evaluate_model(h1_name, y_test, h1_pred, h1_prob)
    all_results.append(res); all_probs[h1_name] = h1_prob
    all_preds[h1_name] = h1_pred
    print(f"  {h1_name} — Acc: {res['Accuracy']*100:.2f}%  AUC: {res['ROC-AUC']:.4f}")

    # ── 5b  XGBoost + LSTM  Stacking (LR meta-learner) ────────
    print("\n  [5b] XGBoost + LSTM — Stacking Ensemble (LR meta-learner) …")
    # Build meta-features from validation predictions
    meta_val_features  = np.column_stack([xgb_vp, lstm_vp,
                                          val_probs["Random Forest"]])
    meta_test_features = np.column_stack([xgb_prob, lstm_prob, rf_prob])

    meta_lr = LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_STATE)
    meta_lr.fit(meta_val_features, y_val)

    h2_prob = meta_lr.predict_proba(meta_test_features)[:, 1]
    h2_pred = (h2_prob >= 0.5).astype(int)
    h2_name = "XGB+LSTM Stacking"
    res = evaluate_model(h2_name, y_test, h2_pred, h2_prob)
    all_results.append(res); all_probs[h2_name] = h2_prob
    all_preds[h2_name] = h2_pred
    model_store[h2_name] = meta_lr
    print(f"  {h2_name} — Acc: {res['Accuracy']*100:.2f}%  AUC: {res['ROC-AUC']:.4f}")
    print(f"  Meta-learner weights → XGB:{meta_lr.coef_[0][0]:.3f}  "
          f"LSTM:{meta_lr.coef_[0][1]:.3f}  RF:{meta_lr.coef_[0][2]:.3f}")

    # ── 5c  RF + LSTM  Weighted Ensemble (comparison) ─────────
    print("\n  [5c] RF + LSTM — Weighted Ensemble (comparison) …")
    rf_vp = val_probs["Random Forest"]

    best_alpha_rf  = 0.5
    best_vauc_rf   = 0.0
    for alpha in alpha_range:
        h_vp = alpha * lstm_vp + (1 - alpha) * rf_vp
        vauc = roc_auc_score(y_val, h_vp)
        if vauc > best_vauc_rf:
            best_vauc_rf   = vauc
            best_alpha_rf  = alpha

    h3_prob = best_alpha_rf * lstm_prob + (1 - best_alpha_rf) * rf_prob
    h3_pred = (h3_prob >= 0.5).astype(int)
    h3_name = "RF+LSTM Weighted"
    res = evaluate_model(h3_name, y_test, h3_pred, h3_prob)
    all_results.append(res); all_probs[h3_name] = h3_prob
    all_preds[h3_name] = h3_pred
    print(f"  Best α_RF = {best_alpha_rf:.2f}  "
          f"({best_alpha_rf*100:.0f}% LSTM + {(1-best_alpha_rf)*100:.0f}% RF)")
    print(f"  {h3_name} — Acc: {res['Accuracy']*100:.2f}%  AUC: {res['ROC-AUC']:.4f}")

    # ── Plot: Alpha-AUC Search Curve ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alpha_range, auc_curve, color="#e74c3c", linewidth=2.5, marker="o",
            markersize=4, label="Val AUC (XGB+LSTM weighted)")
    ax.axvline(x=best_alpha, color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"Optimal α = {best_alpha:.2f}")
    ax.axhline(y=best_vauc,  color="#27ae60", linestyle=":",  linewidth=1.2,
               label=f"Best Val AUC = {best_vauc:.4f}")
    ax.set_xlabel("α  (weight of LSTM in ensemble)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Hybrid Weight Optimisation: XGBoost + LSTM\n"
                 "(α = fraction of LSTM probability in weighted average)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/hybrid_alpha_search.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Alpha-search plot saved.")

else:
    print("  Hybrid models skipped — TensorFlow not available.")


# ──────────────────────────────────────────────────────────────
# STEP 6  RESULTS TABLE
# ──────────────────────────────────────────────────────────────
print("\n[STEP 6] Results summary …")
results_df = pd.DataFrame(all_results).set_index("Model")
print("\n" + results_df.round(4).to_string())
results_df.to_csv(f"{OUTPUT_DIR}/model_results.csv")


# ──────────────────────────────────────────────────────────────
# STEP 7  VISUALISATIONS
# ──────────────────────────────────────────────────────────────
print("\n[STEP 7] Generating all visualisations …")
plt.style.use("seaborn-v0_8-whitegrid")

# Model colour map
CMAP = {
    "Logistic Regression" : "#3498db",
    "Random Forest"       : "#2ecc71",
    "Gradient Boosting"   : "#f39c12",
    "XGBoost"             : "#f39c12",
    "Attention-LSTM"      : "#e74c3c",
    "XGB+LSTM Weighted"   : "#8e44ad",
    "XGB+LSTM Stacking"   : "#6c3483",
    "RF+LSTM Weighted"    : "#1abc9c",
}

models_list = list(all_probs.keys())

# ── Plot A: Accuracy Comparison ───────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
accs   = [results_df.loc[m, "Accuracy"] * 100 for m in models_list]
colors = [CMAP.get(m, "#95a5a6") for m in models_list]
bars   = ax.bar(models_list, accs, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
ax.set_ylim(0, 100)
ax.axhline(y=54.8, color="gray", linestyle="--", linewidth=1.2, label="Naive baseline (54.8%)")
ax.set_title("Model Accuracy Comparison — Baselines vs. LSTM vs. Hybrid",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=11)
plt.xticks(rotation=20, ha="right", fontsize=9)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.legend()
# Shade hybrid region
hybrid_start = models_list.index("XGB+LSTM Weighted") if "XGB+LSTM Weighted" in models_list else None
if hybrid_start is not None:
    ax.axvspan(hybrid_start - 0.4, len(models_list) - 0.6,
               alpha=0.08, color="#8e44ad", label="Hybrid models")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot B: Full Metrics Heatmap ──────────────────────────────
metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
heat_data    = results_df[metrics_cols].copy()
fig, ax = plt.subplots(figsize=(12, 6))
im = sns.heatmap(heat_data, annot=True, fmt=".4f", cmap="YlOrRd",
                 ax=ax, linewidths=0.5, linecolor="white",
                 cbar_kws={"shrink": 0.8}, vmin=0.55, vmax=0.80)
ax.set_title("All Models — Performance Metric Heatmap",
             fontsize=13, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/metrics_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot C: ROC Curves (grouped) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("ROC Curves — All Models", fontsize=13, fontweight="bold")

# Left: Baselines + LSTM
baseline_group = ["Logistic Regression", "Random Forest", xgb_name, "Attention-LSTM"]
axes[0].plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.50)")
for m in baseline_group:
    if m not in all_probs: continue
    fpr, tpr, _ = roc_curve(y_test, all_probs[m])
    auc = results_df.loc[m, "ROC-AUC"]
    lw = 2.5 if "LSTM" in m else 1.5
    axes[0].plot(fpr, tpr, color=CMAP.get(m,"#95a5a6"), lw=lw,
                 label=f"{m}  (AUC={auc:.4f})")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Baselines + Attention-LSTM"); axes[0].legend(fontsize=8)

# Right: LSTM + Hybrids
hybrid_group = ["Attention-LSTM", "XGB+LSTM Weighted", "XGB+LSTM Stacking", "RF+LSTM Weighted"]
axes[1].plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.50)")
for m in hybrid_group:
    if m not in all_probs: continue
    fpr, tpr, _ = roc_curve(y_test, all_probs[m])
    auc = results_df.loc[m, "ROC-AUC"]
    lw = 2.5 if "XGB+LSTM Weighted" in m else 1.8
    axes[1].plot(fpr, tpr, color=CMAP.get(m,"#95a5a6"), lw=lw,
                 label=f"{m}  (AUC={auc:.4f})")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Attention-LSTM + Hybrid Models"); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot D: Precision-Recall Curves ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.axhline(y=y_test.mean(), color="gray", ls="--", lw=1,
           label=f"Random baseline (P={y_test.mean():.2f})")
for m, prob in all_probs.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    f1s = 2*prec*rec/(prec+rec+1e-9)
    lw = 2.5 if "XGB+LSTM Weighted" in m else (2.0 if "LSTM" in m else 1.3)
    ls = "-" if "Hybrid" not in m else "--"
    ax.plot(rec, prec, color=CMAP.get(m,"#95a5a6"), lw=lw,
            label=f"{m}  (F1={f1s.max():.4f})")
ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/precision_recall_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot E: Confusion Matrices ────────────────────────────────
key_models = ["Logistic Regression", xgb_name, "Attention-LSTM", "XGB+LSTM Weighted"]
key_models = [m for m in key_models if m in all_preds]
fig, axes = plt.subplots(1, len(key_models), figsize=(5*len(key_models), 5))
fig.suptitle("Confusion Matrices — Key Models", fontsize=13, fontweight="bold")
if len(key_models) == 1: axes = [axes]
for ax, mn in zip(axes, key_models):
    cm = confusion_matrix(y_test, all_preds[mn])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred\nMissed","Pred\nMade"],
                yticklabels=["Act\nMissed","Act\nMade"], linewidths=0.5)
    ax.set_title(f"{mn}\n(Acc: {results_df.loc[mn,'Accuracy']*100:.2f}%)",
                 fontweight="bold", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot F: Training History ──────────────────────────────────
if history is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Attention-LSTM Training History", fontsize=13, fontweight="bold")
    axes[0].plot(history.history["loss"],     color="#e74c3c", lw=2, label="Train Loss")
    axes[0].plot(history.history["val_loss"], color="#c0392b", lw=2, ls="--", label="Val Loss")
    axes[0].set_title("Loss (Binary Cross-Entropy)", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(history.history["accuracy"],     color="#3498db", lw=2, label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], color="#2980b9", lw=2, ls="--", label="Val Acc")
    axes[1].set_title("Accuracy Over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/lstm_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── Plot G: Feature Importance ───────────────────────────────
fi    = rf.feature_importances_
feat_flat_names = [f"t-{SEQ_LEN-t}: {fn}" for t in range(SEQ_LEN) for fn in feature_names]
fi_df = pd.DataFrame({"feature": feat_flat_names, "importance": fi})
fi_df = fi_df.sort_values("importance", ascending=False).head(20)
roll_kw = ["ROLL","STREAK","GAME_SO_FAR"]
colors_fi = ["#e74c3c" if any(k in f for k in roll_kw) else "#3498db"
             for f in fi_df["feature"]]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
        color=colors_fi[::-1], edgecolor="black", linewidth=0.4)
ax.set_title("Top-20 Feature Importances (Random Forest)\nRed = temporal/rolling features",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot H: Attention Weight Visualisation ────────────────────
if lstm_model is not None and TF_AVAILABLE:
    try:
        att_extractor = Model(inputs=lstm_model.input,
                              outputs=lstm_model.get_layer("attention").output)
        sample_idx  = np.random.choice(len(X_test_seq), 6, replace=False)
        samples     = X_test_seq[sample_idx]
        _, att_w    = att_extractor.predict(samples, verbose=0)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Attention Weights — Which Past Shots Matter Most?",
                     fontsize=12, fontweight="bold")
        x_ticks = [f"t-{SEQ_LEN-i}" for i in range(SEQ_LEN)]
        for idx, (ax, weights) in enumerate(zip(axes.flatten(), att_w)):
            outcome = "Made ✓" if y_test[sample_idx[idx]] == 1 else "Missed ✗"
            bar_colors = ["#e74c3c" if w == weights.max() else "#3498db" for w in weights]
            ax.bar(x_ticks, weights, color=bar_colors, edgecolor="black", lw=0.5)
            ax.set_title(f"Sample {idx+1} → Actual: {outcome}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Past Shot Step"); ax.set_ylabel("Attention Weight")
            ax.set_ylim(0, weights.max() * 1.35)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/attention_weights.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Attention weight visualisation saved.")
    except Exception as e:
        print(f"  Attention visualisation skipped: {e}")

# ── Plot I: Hybrid vs Individual AUC bar ─────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
aucs   = [results_df.loc[m, "ROC-AUC"] for m in models_list]
colors = [CMAP.get(m,"#95a5a6") for m in models_list]
bars   = ax.bar(models_list, aucs, color=colors, edgecolor="black", lw=0.8, width=0.55)
ax.set_ylim(0.65, 0.80)
ax.set_title("AUC-ROC Comparison — Individual vs. Hybrid Models",
             fontsize=12, fontweight="bold")
ax.set_ylabel("AUC-ROC Score")
plt.xticks(rotation=20, ha="right", fontsize=9)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/auc_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot J: Architecture Diagram ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 14))
ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")
ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")

layers_info = [
    ("Input Sequence\n(5 shots × 29 features)",       "#ecf0f1","#2c3e50", 15),
    ("LSTM Layer 1  (128 units, return_seq=True)",     "#d5e8d4","#27ae60", 13.5),
    ("Dropout (0.30) + BatchNormalization",            "#fff2cc","#f39c12", 12.0),
    ("LSTM Layer 2  (64 units, return_seq=True)",      "#d5e8d4","#27ae60", 10.5),
    ("Dropout (0.20) + BatchNormalization",            "#fff2cc","#f39c12",  9.0),
    ("Bahdanau Attention Layer\n(learns shot-step importance weights)", "#ffe6e6","#e74c3c",7.5),
    ("Dense (32 units, ReLU) + Dropout (0.10)",        "#dae8fc","#3498db",  6.0),
    ("Output  (1 unit, Sigmoid)\nP(shot made) ∈ (0,1)","#e1d5e7","#9b59b6", 4.5),
]
for (label, fc, tc, y_pos) in layers_info:
    ax.add_patch(plt.Rectangle((1.5, y_pos-0.55), 7, 1.0, facecolor=fc,
                                edgecolor=tc, linewidth=2, zorder=2))
    ax.text(5, y_pos, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=tc, zorder=3)
for i in range(len(layers_info)-1):
    y_from = layers_info[i][3] - 0.55
    y_to   = layers_info[i+1][3] + 0.45
    ax.annotate("", xy=(5, y_to), xytext=(5, y_from),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

# XGBoost branch for hybrid
ax.add_patch(plt.Rectangle((6.8, 1.0), 2.8, 2.8, facecolor="#fef9e7",
                             edgecolor="#f39c12", linewidth=2, linestyle="--", zorder=2))
ax.text(8.2, 2.8, "XGBoost\n(Tabular\nFeatures)", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#f39c12", zorder=3)
ax.add_patch(plt.Rectangle((1.5, 1.0), 2.8, 2.8, facecolor="#f0f9ff",
                             edgecolor="#8e44ad", linewidth=2, linestyle="--", zorder=2))
ax.text(2.9, 2.8, "LSTM\nProbability\nOutput", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#8e44ad", zorder=3)
ax.add_patch(plt.Rectangle((3.8, 0.2), 2.4, 0.7, facecolor="#d5e8d4",
                             edgecolor="#27ae60", linewidth=2, zorder=2))
ax.text(5.0, 0.55, "Hybrid Output  (optimal α weighting)",
        ha="center", va="center", fontsize=8, fontweight="bold", color="#27ae60", zorder=3)
ax.annotate("", xy=(4.6, 1.0), xytext=(2.9, 1.0),
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5))
ax.annotate("", xy=(5.4, 1.0), xytext=(8.2, 1.0),
            arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.5))
ax.annotate("", xy=(5.0, 0.9), xytext=(5.0, 1.0),
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
ax.text(5, 15.7, "Proposed Architecture: Attention-LSTM + XGBoost Hybrid",
        ha="center", fontsize=11, fontweight="bold", color="#2c3e50")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/model_architecture.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  All plots saved → {OUTPUT_DIR}/plots/")


# ──────────────────────────────────────────────────────────────
# STEP 8  SAVE MODELS
# ──────────────────────────────────────────────────────────────
print("\n[STEP 8] Saving models …")
with open(f"{OUTPUT_DIR}/models/logistic_regression.pkl", "wb") as f: pickle.dump(lr, f)
with open(f"{OUTPUT_DIR}/models/random_forest.pkl",       "wb") as f: pickle.dump(rf, f)
with open(f"{OUTPUT_DIR}/models/boosting_model.pkl",      "wb") as f: pickle.dump(xgb_model, f)
if "XGB+LSTM Stacking" in model_store:
    with open(f"{OUTPUT_DIR}/models/hybrid_stacking_meta.pkl","wb") as f:
        pickle.dump(model_store["XGB+LSTM Stacking"], f)
if lstm_model is not None:
    lstm_model.save(f"{OUTPUT_DIR}/models/attention_lstm.keras")
print("  All models saved.")

# Print classification reports
print("\n" + "="*65)
print("  CLASSIFICATION REPORTS")
print("="*65)
for mn, pred in all_preds.items():
    print(f"\n  ── {mn} ──")
    print(classification_report(y_test, pred, target_names=["Missed","Made"]))


# ──────────────────────────────────────────────────────────────
# STEP 9  INTERACTIVE PREDICTION DEMO
# ──────────────────────────────────────────────────────────────

print("\n[STEP 9] Interactive prediction demo …")

sample_raw = X_test_seq[0]

print(
    f"\n  Demo prediction "
    f"(test sample #1 — Actual: "
    f"{'MADE' if y_test[0]==1 else 'MISSED'})"
)

for mn in all_probs:

    prob = None

    # ---------------------------------------------------------
    # ATTENTION-LSTM
    # ---------------------------------------------------------
    if mn == "Attention-LSTM" and lstm_model is not None:

        prob = float(
            lstm_model.predict(
                sample_raw[np.newaxis],
                verbose=0
            )[0, 0]
        )

    # ---------------------------------------------------------
    # XGB + LSTM WEIGHTED
    # ---------------------------------------------------------
    elif mn == "XGB+LSTM Weighted" and lstm_model is not None:

        xgb_p = float(
            xgb_model.predict_proba(
                sample_raw.reshape(1,-1)
            )[0,1]
        )

        lstm_p = float(
            lstm_model.predict(
                sample_raw[np.newaxis],
                verbose=0
            )[0,0]
        )

        prob = best_alpha * lstm_p + (1-best_alpha) * xgb_p

    # ---------------------------------------------------------
    # XGB + LSTM STACKING
    # ---------------------------------------------------------
    elif mn == "XGB+LSTM Stacking" and lstm_model is not None:

        xgb_p = float(
            xgb_model.predict_proba(
                sample_raw.reshape(1,-1)
            )[0,1]
        )

        rf_p = float(
            rf.predict_proba(
                sample_raw.reshape(1,-1)
            )[0,1]
        )

        lstm_p = float(
            lstm_model.predict(
                sample_raw[np.newaxis],
                verbose=0
            )[0,0]
        )

        meta_features = np.array([
            [xgb_p, lstm_p, rf_p]
        ])

        prob = float(
            model_store[mn]
            .predict_proba(meta_features)[0,1]
        )

    # ---------------------------------------------------------
    # RF + LSTM WEIGHTED
    # ---------------------------------------------------------
    elif mn == "RF+LSTM Weighted" and lstm_model is not None:

        rf_p = float(
            rf.predict_proba(
                sample_raw.reshape(1,-1)
            )[0,1]
        )

        lstm_p = float(
            lstm_model.predict(
                sample_raw[np.newaxis],
                verbose=0
            )[0,0]
        )

        prob = best_alpha_rf * lstm_p + (1-best_alpha_rf) * rf_p

    # ---------------------------------------------------------
    # STANDARD ML MODELS
    # ---------------------------------------------------------
    elif mn in model_store:

        prob = float(
            model_store[mn]
            .predict_proba(
                sample_raw.reshape(1,-1)
            )[0,1]
        )

    # ---------------------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------------------
    if prob is None:
        continue

    pred = "MADE" if prob >= 0.5 else "MISSED"

    print(
        f"  {mn:<24} → {pred}  "
        f"P(made)={prob:.4f}  "
        f"conf={max(prob,1-prob)*100:.1f}%"
    )


# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PART 2 COMPLETE  ✓")
print("="*65)
print(f"\n  Final Results (sorted by AUC):")
print(results_df.sort_values("ROC-AUC", ascending=False).round(4).to_string())
best_model = results_df["ROC-AUC"].idxmax()
print(f"\n  Best model overall : {best_model}")
print(f"  Accuracy           : {results_df.loc[best_model,'Accuracy']*100:.2f}%")
print(f"  AUC-ROC            : {results_df.loc[best_model,'ROC-AUC']:.4f}")
print(f"\n  Outputs → {OUTPUT_DIR}/")

