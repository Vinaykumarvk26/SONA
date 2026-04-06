# %% [markdown]
# # SONA Evaluation Graphs Renderer
# Reads evaluation_* files and writes PNG graphs into backend/notebooks/results.

# %%
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Overall metrics bar chart
# %%
metrics_path = RESULTS_DIR / "evaluation_metrics.json"
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

labels = ["accuracy", "precision", "recall", "f1_score"]
values = [metrics.get(k, 0.0) for k in labels]

plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Precision", "Recall", "F1"], values, color=["#7b5cff", "#4ade80", "#fbbf24", "#60a5fa"])
plt.ylim(0, 1)
plt.title("Overall Metrics")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "evaluation_overall_metrics.png", dpi=200)
plt.close()

# %% [markdown]
# ## Face vs Voice comparison
# %%
face_voice = pd.read_csv(RESULTS_DIR / "evaluation_face_vs_voice.csv")

plt.figure(figsize=(7, 4))
x = range(len(face_voice["metric"]))
plt.bar([i - 0.2 for i in x], face_voice["face"], width=0.4, label="Face")
plt.bar([i + 0.2 for i in x], face_voice["voice"], width=0.4, label="Voice")
plt.xticks(list(x), face_voice["metric"])
plt.ylim(0, 1)
plt.title("Face vs Voice Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "evaluation_face_vs_voice.png", dpi=200)
plt.close()

# %% [markdown]
# ## Emotion-wise accuracy
# %%
emotion_acc = pd.read_csv(RESULTS_DIR / "evaluation_emotion_accuracy.csv")

plt.figure(figsize=(7, 4))
sns.barplot(x="emotion", y="accuracy", data=emotion_acc, palette="mako")
plt.ylim(0, 1)
plt.title("Emotion-wise Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "evaluation_emotion_accuracy.png", dpi=200)
plt.close()

# %% [markdown]
# ## Confusion matrix
# %%
cm = pd.read_csv(RESULTS_DIR / "evaluation_confusion_matrix.csv", index_col=0)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm.columns, yticklabels=cm.index)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Evaluation)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "evaluation_confusion_matrix.png", dpi=200)
plt.close()

print("Saved PNGs in:", RESULTS_DIR)
