
import pandas as pd
from pathlib import Path
from io import StringIO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
# 5. KNN WITH K = 3
# ==========================================

knn_3 = KNeighborsClassifier(n_neighbors=3)

knn_3.fit(X_train, y_train)

y_pred_3 = knn_3.predict(X_test)

accuracy_3 = accuracy_score(y_test, y_pred_3)

print("========== KNN (K=3) ==========")
print("Test Accuracy:", accuracy_3)

# ==========================================
# 6. KNN WITH K = 5
# ==========================================

knn_5 = KNeighborsClassifier(n_neighbors=5)

knn_5.fit(X_train, y_train)

y_pred_5 = knn_5.predict(X_test)

accuracy_5 = accuracy_score(y_test, y_pred_5)

print("\n========== KNN (K=5) ==========")
print("Test Accuracy:", accuracy_5)

# ==========================================
# 7. COMPARE BOTH MODELS
# ==========================================

print("\n========== COMPARISON ==========")

print(f"K=3 Accuracy: {accuracy_3 * 100:.2f}%")
print(f"K=5 Accuracy: {accuracy_5 * 100:.2f}%")

if accuracy_3 > accuracy_5:
    print("K=3 performs better.")
elif accuracy_5 > accuracy_3:
    print("K=5 performs better.")
else:
    print("Both models have the same accuracy.")
# ==========================================
# 8. CONFUSION MATRIX (K = 5)
# ==========================================

cm = confusion_matrix(y_test, y_pred_5)

print("\n========== CONFUSION MATRIX (K=5) ==========")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=knn_5.classes_
)

disp.plot(cmap="Blues", values_format="d")

plt.title("KNN Confusion Matrix (K=5)")
plt.show()
def run_model():
    # Your existing dataset loading and preprocessing code

    # Your existing KNN training code

    return y_test, y_pred_5