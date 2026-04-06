# %% [markdown]
# # SONA Result Graphs (FER2013 + RAVDESS)
# This notebook-style script runs evaluation, exports y_true/y_pred, and plots result graphs.
# Open this file in Jupyter or VS Code and run cells top to bottom.

# %%
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.database import SessionLocal, init_db
from app.services.model_evaluation_service import evaluation_service
from app.models.prediction_log import PredictionLog
from sqlalchemy.exc import OperationalError

sns.set_theme(style="whitegrid")

RESULTS_DIR = ROOT_DIR / "notebooks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1: Run evaluation to generate fresh prediction logs
# This uses FER2013 and RAVDESS test splits. It writes prediction logs into SQLite.

# %%
try:
    init_db()
except OperationalError as exc:
    if "already exists" not in str(exc):
        raise

db = SessionLocal()
try:
    fer_csv = ROOT_DIR / "datasets" / "fer2013" / "fer2013" / "fer2013.csv"
    if not fer_csv.exists():
        fer_csv = ROOT_DIR / "datasets" / "fer2013" / "fer2013.csv"

    ravdess_root = ROOT_DIR / "datasets" / "ravdess"

    face_result = evaluation_service.run_face_evaluation(db, csv_path=str(fer_csv))
    voice_result = evaluation_service.run_voice_evaluation(db, data_dir=str(ravdess_root))
    payload = {"face": face_result, "voice": voice_result}
    print("Evaluation complete:", payload.keys())
finally:
    db.close() 

# %% [markdown]
# ## Step 2: Load latest prediction logs and export y_true/y_pred


# %%
from sqlalchemy import select

def _latest_tag(db, input_type: str):
    row = db.execute(
        select(PredictionLog.run_tag)
        .where(PredictionLog.input_type == input_type)
        .order_by(PredictionLog.timestamp.desc(), PredictionLog.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row

def _load_rows(db, input_type: str, run_tag: str):
    rows = db.execute(
        select(PredictionLog)
        .where(PredictionLog.input_type == input_type, PredictionLog.run_tag == run_tag)
    ).scalars().all()
    return rows

with SessionLocal() as db:
    face_tag = _latest_tag(db, "face")
    voice_tag = _latest_tag(db, "voice")

    face_rows = _load_rows(db, "face", face_tag) if face_tag else []
    voice_rows = _load_rows(db, "voice", voice_tag) if voice_tag else []

face_true = [r.actual_label for r in face_rows]
face_pred = [r.predicted_label for r in face_rows]
voice_true = [r.actual_label for r in voice_rows]
voice_pred = [r.predicted_label for r in voice_rows]

# Combine for an overall result
all_true = face_true + voice_true
all_pred = face_pred + voice_pred

np.save(RESULTS_DIR / "y_true.npy", np.array(all_true))
np.save(RESULTS_DIR / "y_pred.npy", np.array(all_pred))
np.save(RESULTS_DIR / "y_true_face.npy", np.array(face_true))
np.save(RESULTS_DIR / "y_pred_face.npy", np.array(face_pred))
np.save(RESULTS_DIR / "y_true_voice.npy", np.array(voice_true))
np.save(RESULTS_DIR / "y_pred_voice.npy", np.array(voice_pred))

print("Saved:", RESULTS_DIR / "y_true.npy", RESULTS_DIR / "y_pred.npy")

labels = ["happy","sad","angry","fear","disgust","surprise","neutral"]

# %% [markdown]
# ## Step 3: Overall Metrics (Macro Average)

# %%
acc = accuracy_score(all_true, all_pred)
prec = precision_score(all_true, all_pred, average="macro", zero_division=0)
rec = recall_score(all_true, all_pred, average="macro", zero_division=0)
f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

plt.figure(figsize=(6,4))
plt.bar(
    ["Accuracy","Precision","Recall","F1"],
    [acc, prec, rec, f1],
    color=["#7b5cff","#4ade80","#fbbf24","#60a5fa"],
)
plt.ylim(0,1)
plt.title("Overall Metrics (Macro Avg)")
plt.show()

# %% [markdown]
# ## Step 4: Face vs Voice Metrics (Macro Average)

# %%
face_metrics = {
    "Accuracy": accuracy_score(face_true, face_pred) if face_true else 0.0,
    "Precision": precision_score(face_true, face_pred, average="macro", zero_division=0) if face_true else 0.0,
    "Recall": recall_score(face_true, face_pred, average="macro", zero_division=0) if face_true else 0.0,
    "F1": f1_score(face_true, face_pred, average="macro", zero_division=0) if face_true else 0.0,
}
voice_metrics = {
    "Accuracy": accuracy_score(voice_true, voice_pred) if voice_true else 0.0,
    "Precision": precision_score(voice_true, voice_pred, average="macro", zero_division=0) if voice_true else 0.0,
    "Recall": recall_score(voice_true, voice_pred, average="macro", zero_division=0) if voice_true else 0.0,
    "F1": f1_score(voice_true, voice_pred, average="macro", zero_division=0) if voice_true else 0.0,
}

x = np.arange(len(face_metrics))
width = 0.35

plt.figure(figsize=(7,4))
plt.bar(x - width/2, list(face_metrics.values()), width, label="Face")
plt.bar(x + width/2, list(voice_metrics.values()), width, label="Voice")
plt.xticks(x, list(face_metrics.keys()))
plt.ylim(0,1)
plt.title("Face vs Voice Metrics (Macro Avg)")
plt.legend()
plt.show()

# %% [markdown]
# ## Step 5: Confusion Matrix (Overall)

# %%
cm = confusion_matrix(all_true, all_pred, labels=labels)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Overall)")
plt.show()

# %% [markdown]
# ## Step 6: Emotion-wise Accuracy (Overall)

# %%
emotion_acc = {}
for emo in labels:
    idx = [i for i, y in enumerate(all_true) if y == emo]
    if not idx:
        emotion_acc[emo] = 0
        continue
    y_t = [all_true[i] for i in idx]
    y_p = [all_pred[i] for i in idx]
    emotion_acc[emo] = accuracy_score(y_t, y_p)

plt.figure(figsize=(7,4))
sns.barplot(x=list(emotion_acc.keys()), y=list(emotion_acc.values()), palette="mako")
plt.ylim(0,1)
plt.title("Emotion-wise Accuracy")
plt.xticks(rotation=30)
plt.show()

# %% [markdown]
# ## Step 7: Overall Accuracy (Print)

# %%
print("Overall Accuracy:", round(acc, 4))

# %% [markdown]
# ## Notes on training curves
# If you have train/val history from training, save it as history.npy and load it here.
# Otherwise, you can skip curves or add them later once training history is exported.

