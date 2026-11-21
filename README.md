# Out of Position – Player Classification System

A desktop app that predicts a football player's position (Defender, Attacker,
Goalkeeper or Midfielder) from his match statistics. It is only the interface:
the models themselves come from your existing scripts.

## Folder layout

Everything must be in the **same folder**:

```
your-project/
├── out_of_position.py     <- the GUI (run this one)
├── knn.py
├── Logistic.py
├── tree.py
├── random_forest.py
├── gradient_boost.py
├── positions.py           (optional, not used by the GUI)
└── dataset                (the ARFF-style file with the @DATA line)
```

## Requirements

- Python 3.9 or newer
- Packages: `pip install pandas numpy scikit-learn matplotlib`
- tkinter (the window toolkit). It is included with Python on Windows and
  macOS. On Linux install it once:
  - Fedora: `sudo dnf install python3-tkinter`
  - Ubuntu / Debian: `sudo apt install python3-tk`

## Run

```
python out_of_position.py
```

The first start takes a few seconds ("Loading dataset and training models…")
because your five scripts are loaded and their models trained.

## How to use it

**Position Prediction**
1. Type the eight player stats: Height, Minutes / Match, Goals / Match,
   Assists / Match, Yellow Cards / Match, Shots / Match, Pass Success Rate,
   Aerials Won.
2. Pick a classification model (Random Forest is selected by default).
3. Press **PREDICT POSITION**. You get the predicted position, the confidence
   and the probability of every position.

Don't want to type? Under "Fill with a typical", press **Attacker**,
**Midfielder**, **Defender** or **Goalkeeper**. The form is filled with the
median stats of that position in your dataset. **Clear** empties the form.

**Dashboard** – number of players, features and positions, the best model,
a model-accuracy chart and the number of players per position.

**Model Evaluation** – accuracy, precision, recall and F1-score of all five
models (same table as `positions.py`). Click a model, or choose it in the
dropdown, to see its confusion matrix and per-position results.

## How it works

`out_of_position.py` imports `knn`, `Logistic`, `tree`, `random_forest` and
`gradient_boost` and reuses what they already built:

| Model               | Taken from your script                         |
|---------------------|------------------------------------------------|
| KNN (K = 5)         | `knn.knn_5` + `knn.scaler`                     |
| Logistic Regression | `Logistic.logistic_model` + `Logistic.scaler`  |
| Decision Tree       | `tree.decision_tree`                           |
| Random Forest       | `random_forest.random_forest`                  |
| Gradient Boosting   | `gradient_boost.gradient_boosting`             |

The evaluation numbers come from each script's `run_model()` function.
The GUI trains nothing on its own, so if you change a model in one of your
scripts, the GUI follows automatically.

Your scripts call `plt.show()` when they are imported. The GUI switches
matplotlib to the non-interactive `Agg` backend so no extra chart windows
open. Their `print` output (classification reports, etc.) still appears in the
terminal at startup; that is normal.

## Position codes

The class column of the dataset is shown with names. The mapping comes from
the header of the dataset file:

| Code | Position   |
|------|------------|
| 0    | Defender   |
| 1    | Attacker (called "Forward" in the dataset header) |
| 2    | Goalkeeper |
| 3    | Midfielder |

It is defined in the `POSITION_NAMES` block near the top of
`out_of_position.py` if you ever need to change it.

## Results on the full dataset (3,611 players, 80/20 split)

| Model               | Test accuracy |
|---------------------|---------------|
| Gradient Boosting   | 84.23 %       |
| Random Forest       | 83.13 %       |
| Logistic Regression | 82.30 %       |
| KNN (K = 5)         | 80.08 %       |
| Decision Tree       | 77.04 %       |

## Troubleshooting

- **"Could not load your models"** – one of the scripts or the `dataset` file
  is missing from the folder, or a package is not installed. The message says
  which one.
- **`ModuleNotFoundError: No module named 'tkinter'`** – install tkinter (see
  Requirements).
- **A prediction asks for a missing/invalid value** – every one of the eight
  boxes must contain a number; use a dot or a comma for decimals.
- **Wrong or odd position names** – check `POSITION_NAMES` against the
  `@DATA` section header of your dataset file.
