
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from knn import run_model as run_knn
from Logistic import run_model as run_logistic
from tree import run_model as run_tree
from random_forest import run_model as run_random_forest
from gradient_boost import run_model as run_gradient_boost


# ==========================================
# 1. RUN ALL MODELS
# ==========================================

models = {
    "KNN": run_knn,
    "Logistic Regression": run_logistic,
    "Decision Tree": run_tree,
    "Random Forest": run_random_forest,
    "Gradient Boosting": run_gradient_boost
}

results = []

for name, run in models.items():

    y_test, y_pred = run()

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ) * 100,
        "Recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ) * 100,
        "F1-Score": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ) * 100
    })


# ==========================================
# 2. COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.2f}%".format,
        "Precision": "{:.2f}%".format,
        "Recall": "{:.2f}%".format,
        "F1-Score": "{:.2f}%".format
    }
))


# ==========================================
# 3. BEST MODEL
# ==========================================

best_model = results_df.iloc[0]

print("\n========== BEST MODEL ==========")
print("Model:", best_model["Model"])
print(f"Accuracy: {best_model['Accuracy']:.2f}%")


# ==========================================
# 4. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
