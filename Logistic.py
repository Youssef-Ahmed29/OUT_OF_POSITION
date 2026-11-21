
import pandas as pd
from pathlib import Path
from io import StringIO

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = Path(__file__).resolve().parent / "dataset"

with open(file_path, "r") as file:
    lines = file.readlines()

data_start = next(
    i for i, line in enumerate(lines)
    if line.strip().upper() == "@DATA"
)

# only 7 features are important
columns = [
    "Age",
    "Height",
    "Matches",
    "Minutes/Match",
    "Goals/Match",
    "Assists/Match",
    "Yellow Cards/Match",
    "Red Cards/Match",
    "Shots/Match",
    "Pass Success Rate",
    "Aerials Won",
    "class"
]

df = pd.read_csv(
    StringIO("".join(lines[data_start + 1:])),
    header=None,
    names=columns
)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=[
    "Age",
    "Matches",
    "Red Cards/Match",
    "class"
])

y = df["class"]


# ==========================================
# 3. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 5. LOGISTIC REGRESSION
# ==========================================

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# ==========================================
# 6. EVALUATION
# ==========================================

print("\n========== LOGISTIC REGRESSION ==========")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Classification Report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=logistic_model.classes_
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression Confusion Matrix")
plt.show()
def run_model():
    # Your existing dataset loading and preprocessing code

    # Your existing KNN training code

    return y_test, y_pred