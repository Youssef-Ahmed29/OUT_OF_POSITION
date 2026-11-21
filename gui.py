"""
OUT OF POSITION - Player Classification System  (GUI only)

Run:
    python out_of_position.py

Put this file in the SAME folder as your existing files:
    knn.py  Logistic.py  tree.py  random_forest.py  gradient_boost.py  dataset

This file only adds the interface. It imports your scripts, reuses the models
they already trained (knn_5, logistic_model, decision_tree, random_forest,
gradient_boosting + their scalers) and calls your run_model() functions for the
evaluation numbers, exactly like positions.py does.

Your scripts call plt.show() when imported; matplotlib is switched to the
non-interactive "Agg" backend here so no extra chart windows open.

Needs:  pip install pandas numpy scikit-learn matplotlib
        (tkinter ships with Python; on Fedora: sudo dnf install python3-tkinter)
"""

import importlib
import queue
import sys
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("Agg")  # must happen before your scripts import pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", message=".*non-interactive.*")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ======================================================================
# 1. YOUR MODELS  (imported from your own scripts)
# ======================================================================

def pct(value) -> float:
    return round(float(value) * 100, 2)


# class codes in the dataset -> position names shown in the interface
# (taken from the header of your dataset file: 0 Defender, 1 Forward,
#  2 Goalkeeper, 3 Midfielder - "Forward" is shown as "Attacker")
POSITION_NAMES = {
    "0": "Defender",
    "1": "Attacker",
    "2": "Goalkeeper",
    "3": "Midfielder",
}


def position_name(label) -> str:
    text = str(label).strip()
    try:
        text = str(int(float(text)))  # "2", 2 and 2.0 all become "2"
    except ValueError:
        pass
    return POSITION_NAMES.get(text, text)


class ModelHub:
    """Collects the trained models and results from your existing scripts."""

    # display name -> (module, fitted model variable, scaler variable or None)
    SOURCES = {
        "KNN": ("knn", "knn_5", "scaler"),
        "Logistic Regression": ("Logistic", "logistic_model", "scaler"),
        "Decision Tree": ("tree", "decision_tree", None),
        "Random Forest": ("random_forest", "random_forest", None),
        "Gradient Boosting": ("gradient_boost", "gradient_boosting", None),
    }

    def __init__(self):
        self.models = {}   # name -> (fitted model, scaler or None)
        self.metrics = {}  # name -> dict
        modules = {}

        for name, (module_name, model_attr, scaler_attr) in self.SOURCES.items():
            module = importlib.import_module(module_name)
            modules[name] = module
            model = getattr(module, model_attr)
            scaler = getattr(module, scaler_attr) if scaler_attr else None
            self.models[name] = (model, scaler)

            y_test, y_pred = module.run_model()
            y_true = [str(v) for v in y_test]
            y_hat = [str(v) for v in y_pred]
            classes = [str(c) for c in model.classes_]
            report = classification_report(
                y_true, y_hat, labels=classes, output_dict=True, zero_division=0
            )
            self.metrics[name] = {
                "accuracy": pct(accuracy_score(y_true, y_hat)),
                "precision": pct(precision_score(y_true, y_hat, average="weighted", zero_division=0)),
                "recall": pct(recall_score(y_true, y_hat, average="weighted", zero_division=0)),
                "f1": pct(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
                "classes": [position_name(c) for c in classes],
                "cm": confusion_matrix(y_true, y_hat, labels=classes).tolist(),
                "per_class": [
                    (
                        position_name(c),
                        pct(report[c]["precision"]),
                        pct(report[c]["recall"]),
                        pct(report[c]["f1-score"]),
                        int(report[c]["support"]),
                    )
                    for c in classes
                ],
            }

        plt.close("all")  # drop the figures your scripts built on import

        ref = modules["KNN"]  # dataset info is identical in every script
        self.features = list(ref.X.columns)
        self.medians = [float(v) for v in ref.X.median()]
        # "typical player" of each position: median of every stat in that class
        by_class = ref.df.groupby("class")[self.features].median()
        self.profiles = {
            position_name(code): [float(v) for v in row]
            for code, row in by_class.iterrows()
        }
        self.n_players = len(ref.df)
        self.n_train = len(ref.y_train)
        self.n_test = len(ref.y_test)
        counts = ref.df["class"].astype(str).value_counts().sort_index()
        self.class_counts = [(position_name(k), int(v)) for k, v in counts.items()]

        self.model_names = list(self.models.keys())
        self.best = max(self.metrics, key=lambda n: self.metrics[n]["accuracy"])

    def predict(self, model_name, values):
        model, scaler = self.models[model_name]
        sample = pd.DataFrame([values], columns=self.features)
        if scaler is not None:
            sample = scaler.transform(sample)
        label = position_name(model.predict(sample)[0])
        probs = model.predict_proba(sample)[0]
        ranking = sorted(
            ((position_name(c), pct(p)) for c, p in zip(model.classes_, probs)),
            key=lambda r: -r[1],
        )
        return label, pct(probs.max()), ranking


# ======================================================================
# 2. THEME + SMALL WIDGETS
# ======================================================================

BG = "#0a111c"          # app / card background
CARD = "#0f1a2b"        # raised panels
INPUT = "#17233a"       # inputs
BORDER = "#1c2940"
TEXT = "#f1f5f9"
MUTED = "#8b9ab0"
ACCENT = "#21c897"
ACCENT_HOVER = "#2ee0ad"
ACCENT_DARK = "#0f3b33"
ON_ACCENT = "#06251d"
ERROR = "#f87171"

FONT_FAMILY = "Helvetica"


def pick_font(root):
    available = set(tkfont.families(root))
    for name in (
        "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue",
        "Noto Sans", "Cantarell", "DejaVu Sans",
    ):
        if name in available:
            return name
    return "Helvetica"


def F(size, weight="normal"):
    return (FONT_FAMILY, size, weight)


def mix(c1, c2, t):
    """Blend hex colour c1 towards c2 by t (0..1)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def round_rect(c, x1, y1, x2, y2, r, **kw):
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return c.create_polygon(pts, smooth=True, **kw)


def label(parent, text, size=10, color=TEXT, weight="normal", **kw):
    bg = kw.pop("bg", parent.cget("bg"))
    return tk.Label(
        parent, text=text, font=F(size, weight), fg=color, bg=bg,
        bd=0, highlightthickness=0, **kw
    )


def panel(parent, **kw):
    return tk.Frame(
        parent, bg=CARD, highlightbackground=BORDER, highlightcolor=BORDER,
        highlightthickness=1, bd=0, **kw
    )


class Logo(tk.Canvas):
    def __init__(self, parent, size=38):
        super().__init__(parent, width=size, height=size, bg=parent.cget("bg"),
                         highlightthickness=0)
        round_rect(self, 0, 0, size, size, 10, fill=ACCENT, outline=ACCENT)
        s = size / 38
        self.create_line(
            9 * s, 20 * s, 15 * s, 20 * s, 18 * s, 12 * s, 21 * s, 27 * s,
            24 * s, 20 * s, 29 * s, 20 * s,
            fill=ON_ACCENT, width=2.4, joinstyle="round", capstyle="round",
        )


class NavIcon(tk.Canvas):
    def __init__(self, parent, kind):
        super().__init__(parent, width=32, height=32, bg=parent.cget("bg"),
                         highlightthickness=0)
        self.kind = kind
        self.draw(False)

    def draw(self, active):
        self.delete("all")
        bg = self.master.cget("bg")
        color = ACCENT if active else MUTED
        if active:
            round_rect(self, 0, 0, 32, 32, 9, fill=ACCENT_DARK, outline=ACCENT_DARK)
        if self.kind == "dashboard":
            for x, y in ((8, 8), (17, 8), (8, 17), (17, 17)):
                round_rect(self, x, y, x + 7, y + 7, 2, fill=bg if not active else ACCENT_DARK,
                           outline=color, width=1.6)
        elif self.kind == "predict":
            self.create_oval(8, 8, 24, 24, outline=color, width=1.6)
            for x1, y1, x2, y2 in ((16, 4, 16, 10), (16, 22, 16, 28),
                                   (4, 16, 10, 16), (22, 16, 28, 16)):
                self.create_line(x1, y1, x2, y2, fill=color, width=1.6)
        else:
            for x, top in ((9, 17), (16, 9), (23, 13)):
                self.create_line(x, 24, x, top, fill=color, width=3, capstyle="round")


class NavItem(tk.Frame):
    def __init__(self, parent, kind, text, command):
        super().__init__(parent, bg=parent.cget("bg"), cursor="hand2")
        self.icon = NavIcon(self, kind)
        self.icon.pack(side="left", padx=(0, 8))
        self.text = label(self, text, 11)
        self.text.pack(side="left")
        for w in (self, self.icon, self.text):
            w.bind("<Button-1>", lambda _e: command())
            w.configure(cursor="hand2")

    def set_active(self, active):
        self.icon.draw(active)
        self.text.configure(
            fg=TEXT if active else MUTED, font=F(11, "bold" if active else "normal")
        )


class RoundedEntry(tk.Canvas):
    """Dark rounded text input with a placeholder."""

    def __init__(self, parent, placeholder="Enter value", height=38, on_enter=None):
        super().__init__(parent, height=height, bg=parent.cget("bg"),
                         highlightthickness=0, cursor="xterm")
        self.placeholder = placeholder
        self.placeholder_on = True
        self.focused = False
        self.entry = tk.Entry(
            self, bd=0, relief="flat", bg=INPUT, fg=MUTED, insertbackground=TEXT,
            highlightthickness=0, font=F(10), disabledbackground=INPUT,
        )
        self.entry.insert(0, placeholder)
        self.win = self.create_window(12, height // 2, window=self.entry, anchor="w")
        self.bind("<Configure>", self._redraw)
        self.bind("<Button-1>", lambda _e: self.entry.focus_set())
        self.entry.bind("<FocusIn>", self._focus_in)
        self.entry.bind("<FocusOut>", self._focus_out)
        if on_enter:
            self.entry.bind("<Return>", lambda _e: on_enter())

    def _redraw(self, _event=None):
        w, h = self.winfo_width(), self.winfo_height()
        self.delete("shape")
        outline = ACCENT if self.focused else INPUT
        round_rect(self, 1, 1, w - 1, h - 1, 9, fill=INPUT, outline=outline,
                   width=1.5, tags="shape")
        self.tag_lower("shape")
        self.itemconfigure(self.win, width=max(w - 24, 10))

    def _focus_in(self, _e):
        self.focused = True
        if self.placeholder_on:
            self.entry.delete(0, "end")
            self.entry.configure(fg=TEXT)
            self.placeholder_on = False
        self._redraw()

    def _focus_out(self, _e):
        self.focused = False
        if not self.entry.get().strip():
            self.entry.delete(0, "end")
            self.entry.insert(0, self.placeholder)
            self.entry.configure(fg=MUTED)
            self.placeholder_on = True
        self._redraw()

    def get(self):
        return "" if self.placeholder_on else self.entry.get().strip()

    def set(self, value):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)
        self.entry.configure(fg=TEXT)
        self.placeholder_on = False

    def focus(self):
        self.entry.focus_set()


class AccentButton(tk.Canvas):
    def __init__(self, parent, text, command, height=48):
        super().__init__(parent, height=height, bg=parent.cget("bg"),
                         highlightthickness=0, cursor="hand2")
        self.text, self.command, self.fill = text, command, ACCENT
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Enter>", lambda _e: self._hover(ACCENT_HOVER))
        self.bind("<Leave>", lambda _e: self._hover(ACCENT))
        self.bind("<ButtonRelease-1>", lambda _e: self.command())

    def _hover(self, color):
        self.fill = color
        self._redraw()

    def _redraw(self):
        w, h = self.winfo_width(), self.winfo_height()
        self.delete("all")
        round_rect(self, 0, 0, w, h, 11, fill=self.fill, outline=self.fill)
        self.create_text(w / 2, h / 2, text=self.text, fill=ON_ACCENT,
                         font=F(11, "bold"))


class BarChart(tk.Canvas):
    """Horizontal bars. rows = [(label, value, text)], value in 0..max_value."""

    ROW_H = 30

    def __init__(self, parent, rows, max_value=100.0, bg=None):
        super().__init__(parent, height=len(rows) * self.ROW_H + 4,
                         bg=bg or parent.cget("bg"), highlightthickness=0)
        self.rows, self.max_value = rows, max_value
        self.font = tkfont.Font(family=FONT_FAMILY, size=10)
        self.bind("<Configure>", lambda _e: self._redraw())

    def _redraw(self):
        self.delete("all")
        w = self.winfo_width()
        left = max((self.font.measure(r[0]) for r in self.rows), default=0) + 16
        right = max((self.font.measure(r[2]) for r in self.rows), default=0) + 14
        track_w = max(w - left - right, 20)
        for i, (name, value, text) in enumerate(self.rows):
            y = i * self.ROW_H + self.ROW_H / 2 + 2
            self.create_text(0, y, text=name, anchor="w", fill=TEXT, font=self.font)
            round_rect(self, left, y - 6, left + track_w, y + 6, 6,
                       fill=BORDER, outline=BORDER)
            frac = max(min(value / self.max_value, 1.0), 0.0) if self.max_value else 0
            if frac > 0:
                color = ACCENT if i == 0 else mix(ACCENT, BG, 0.5)
                round_rect(self, left, y - 6, left + max(track_w * frac, 12), y + 6, 6,
                           fill=color, outline=color)
            self.create_text(w, y, text=text, anchor="e", fill=MUTED, font=self.font)


class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                            style="Dark.Vertical.TScrollbar")
        self.inner = tk.Frame(self.canvas, bg=BG)
        win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width)
        )
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self.inner.bind("<Enter>", self._bind_wheel)

    def _bind_wheel(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", lambda _e: self._scroll(-1))
        self.canvas.bind_all("<Button-5>", lambda _e: self._scroll(1))

    def _unbind_wheel(self, _e=None):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)

    def _wheel(self, event):
        self._scroll(-1 if event.delta > 0 else 1)

    def _scroll(self, step):
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(step * 3, "units")


def page_header(parent, title, subtitle):
    label(parent, title, 17, TEXT, "bold").pack(anchor="w")
    label(parent, subtitle, 10, MUTED).pack(anchor="w", pady=(4, 16))


def section_title(parent, text):
    label(parent, text, 12, TEXT, "bold").pack(anchor="w", pady=(18, 8))


def pretty(name):
    return name.replace("/", " / ")


# ======================================================================
# 3. VIEWS
# ======================================================================

class DashboardView(ScrollFrame):
    def __init__(self, parent, hub):
        super().__init__(parent)
        body = tk.Frame(self.inner, bg=BG)
        body.pack(fill="both", expand=True, padx=32, pady=28)
        page_header(body, "Dashboard",
                    "Overview of the dataset and the trained classification models.")

        grid = tk.Frame(body, bg=BG)
        grid.pack(fill="x")
        grid.columnconfigure((0, 1, 2, 3), weight=1, uniform="stat")
        best = hub.metrics[hub.best]["accuracy"]
        stats = [
            ("PLAYERS", f"{hub.n_players:,}", f"{hub.n_train} train / {hub.n_test} test"),
            ("FEATURES", str(len(hub.features)), "used by every model"),
            ("POSITIONS", str(len(hub.class_counts)), "classes to predict"),
            ("BEST MODEL", hub.best, f"{best:.2f}% test accuracy"),
        ]
        for i, (title, value, sub) in enumerate(stats):
            card = panel(grid)
            card.grid(row=0, column=i, sticky="nsew",
                      padx=(0 if i == 0 else 6, 0 if i == 3 else 6))
            inner = tk.Frame(card, bg=CARD)
            inner.pack(fill="both", expand=True, padx=14, pady=12)
            label(inner, title, 8, MUTED, "bold", bg=CARD).pack(anchor="w")
            label(inner, value, 15 if i == 3 else 19, ACCENT if i == 3 else TEXT,
                  "bold", bg=CARD, wraplength=170, justify="left").pack(anchor="w", pady=(4, 2))
            label(inner, sub, 9, MUTED, bg=CARD).pack(anchor="w")

        section_title(body, "Model accuracy")
        acc_panel = panel(body)
        acc_panel.pack(fill="x")
        ranked = sorted(hub.metrics.items(), key=lambda kv: -kv[1]["accuracy"])
        BarChart(
            acc_panel,
            [(n, m["accuracy"], f"{m['accuracy']:.2f}%") for n, m in ranked],
            bg=CARD,
        ).pack(fill="x", padx=16, pady=14)

        section_title(body, "Players per position")
        dist_panel = panel(body)
        dist_panel.pack(fill="x")
        top = max(c for _, c in hub.class_counts)
        BarChart(
            dist_panel,
            [(n, c, f"{c:,}") for n, c in hub.class_counts],
            max_value=top, bg=CARD,
        ).pack(fill="x", padx=16, pady=14)


class PredictView(ScrollFrame):
    def __init__(self, parent, hub):
        super().__init__(parent)
        self.hub = hub
        body = tk.Frame(self.inner, bg=BG)
        body.pack(fill="both", expand=True, padx=32, pady=28)
        page_header(body, "Predict a Player's Position",
                    "Enter the player's performance statistics and select a classification model.")

        grid = tk.Frame(body, bg=BG)
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1, uniform="col")
        self.entries = {}
        for i, feature in enumerate(hub.features):
            cell = tk.Frame(grid, bg=BG)
            cell.grid(row=i // 2, column=i % 2, sticky="ew",
                      padx=(0, 8) if i % 2 == 0 else (8, 0), pady=(0, 12))
            label(cell, pretty(feature), 9, MUTED).pack(anchor="w")
            entry = RoundedEntry(cell, on_enter=self.predict)
            entry.pack(fill="x", pady=(5, 0))
            self.entries[feature] = entry

        label(body, "CLASSIFICATION MODEL", 8, MUTED, "bold").pack(anchor="w", pady=(4, 5))
        self.model_var = tk.StringVar(value="Random Forest")
        combo = ttk.Combobox(body, textvariable=self.model_var, values=hub.model_names,
                             state="readonly", style="Dark.TCombobox", font=F(11, "bold"))
        combo.pack(fill="x", ipady=6)
        combo.bind("<<ComboboxSelected>>", lambda _e: combo.selection_clear())

        AccentButton(body, "PREDICT POSITION", self.predict).pack(fill="x", pady=(14, 8))

        label(body, "FILL WITH A TYPICAL", 8, MUTED, "bold").pack(anchor="w", pady=(6, 6))
        row = tk.Frame(body, bg=BG)
        row.pack(fill="x")
        for position in ("Attacker", "Midfielder", "Defender", "Goalkeeper"):
            chip = tk.Label(row, text=position, font=F(10, "bold"), fg=TEXT, bg=INPUT,
                            padx=16, pady=7, bd=0, cursor="hand2")
            chip.pack(side="left", padx=(0, 8))
            chip.bind("<Enter>", lambda _e, c=chip: c.configure(bg=ACCENT_DARK, fg=ACCENT))
            chip.bind("<Leave>", lambda _e, c=chip: c.configure(bg=INPUT, fg=TEXT))
            chip.bind("<Button-1>", lambda _e, p=position: self.fill_position(p))
        clear = label(row, "Clear", 9, MUTED)
        clear.pack(side="right")
        clear.configure(cursor="hand2")
        clear.bind("<Button-1>", lambda _e: self.clear())
        self.status = label(body, "", 9, ERROR)
        self.status.pack(anchor="w", pady=(8, 0))

        self.result = tk.Frame(body, bg=BG)
        self.result.pack(fill="x", pady=(8, 0))

    def fill_position(self, position):
        """Fill the form with the median stats of that position in the dataset."""
        values = self.hub.profiles.get(position)
        if not values:
            self.status.configure(text=f"No {position} players found in the dataset.")
            return
        for feature, value in zip(self.hub.features, values):
            self.entries[feature].set(f"{round(value, 2):g}")
        self.status.configure(text="")

    def clear(self):
        for entry in self.entries.values():
            entry.set("")
            entry.entry.delete(0, "end")
            entry.entry.insert(0, entry.placeholder)
            entry.entry.configure(fg=MUTED)
            entry.placeholder_on = True
        self.status.configure(text="")
        for w in self.result.winfo_children():
            w.destroy()

    def predict(self):
        values = []
        for feature in self.hub.features:
            raw = self.entries[feature].get()
            if not raw:
                self.status.configure(text=f"Enter a value for {pretty(feature)}.")
                self.entries[feature].focus()
                return
            try:
                number = float(raw.replace(",", "."))
            except ValueError:
                self.status.configure(text=f"{pretty(feature)} must be a number.")
                self.entries[feature].focus()
                return
            if not np.isfinite(number):
                self.status.configure(text=f"{pretty(feature)} must be a finite number.")
                self.entries[feature].focus()
                return
            values.append(number)

        self.status.configure(text="")
        model = self.model_var.get()
        prediction, confidence, ranking = self.hub.predict(model, values)

        for w in self.result.winfo_children():
            w.destroy()
        card = panel(self.result)
        card.pack(fill="x")
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="x", padx=18, pady=16)
        label(inner, "PREDICTED POSITION", 8, MUTED, "bold", bg=CARD).pack(anchor="w")
        label(inner, prediction, 22, ACCENT, "bold", bg=CARD).pack(anchor="w", pady=(4, 2))
        label(inner, f"{confidence:.2f}% confidence  ·  {model}", 9, MUTED,
              bg=CARD).pack(anchor="w", pady=(0, 12))
        BarChart(inner, [(n, p, f"{p:.1f}%") for n, p in ranking[:6]], bg=CARD).pack(fill="x")


class EvaluationView(ScrollFrame):
    def __init__(self, parent, hub):
        super().__init__(parent)
        self.hub = hub
        self.body = tk.Frame(self.inner, bg=BG)
        self.body.pack(fill="both", expand=True, padx=32, pady=28)
        page_header(self.body, "Model Evaluation",
                    "Test-set results of every model (20% hold-out, stratified).")

        section_title(self.body, "Comparison")
        self.table_holder = tk.Frame(self.body, bg=BG)
        self.table_holder.pack(fill="x")

        section_title(self.body, "Model details")
        label(self.body, "MODEL", 8, MUTED, "bold").pack(anchor="w", pady=(0, 5))
        self.model_var = tk.StringVar(value=hub.best)
        combo = ttk.Combobox(self.body, textvariable=self.model_var, values=hub.model_names,
                             state="readonly", style="Dark.TCombobox", font=F(11, "bold"))
        combo.pack(fill="x", ipady=6)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._select(self.model_var.get()))

        self.detail = tk.Frame(self.body, bg=BG)
        self.detail.pack(fill="x", pady=(14, 0))
        self._select(hub.best)

    def _select(self, name):
        self.model_var.set(name)
        self._render_table(name)
        self._render_detail(name)

    def _render_table(self, selected):
        for w in self.table_holder.winfo_children():
            w.destroy()
        card = panel(self.table_holder)
        card.pack(fill="x")
        cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        for c in range(len(cols)):
            card.columnconfigure(c, weight=3 if c == 0 else 2)
        for c, title in enumerate(cols):
            label(card, title, 9, MUTED, "bold", bg=CARD,
                  anchor="w" if c == 0 else "e").grid(
                row=0, column=c, sticky="ew", padx=14, pady=(10, 6))

        ranked = sorted(self.hub.metrics.items(), key=lambda kv: -kv[1]["accuracy"])
        for r, (name, m) in enumerate(ranked, start=1):
            is_sel = name == selected
            bg = mix(CARD, ACCENT, 0.12) if is_sel else CARD
            cells = [
                name + ("   ★" if name == self.hub.best else ""),
                f"{m['accuracy']:.2f}%", f"{m['precision']:.2f}%",
                f"{m['recall']:.2f}%", f"{m['f1']:.2f}%",
            ]
            for c, text in enumerate(cells):
                lbl = label(card, text, 10, ACCENT if (c == 1 and is_sel) else TEXT,
                            "bold" if (c == 0 or is_sel) else "normal", bg=bg,
                            anchor="w" if c == 0 else "e", cursor="hand2")
                lbl.grid(row=r, column=c, sticky="nsew", ipady=7, padx=0)
                lbl.configure(padx=14)
                lbl.bind("<Button-1>", lambda _e, n=name: self._select(n))
        tk.Frame(card, bg=CARD, height=4).grid(row=len(ranked) + 1, column=0, columnspan=5)

    def _render_detail(self, name):
        for w in self.detail.winfo_children():
            w.destroy()
        m = self.hub.metrics[name]

        cards = tk.Frame(self.detail, bg=BG)
        cards.pack(fill="x")
        cards.columnconfigure((0, 1, 2, 3), weight=1, uniform="metric")
        for i, (title, key) in enumerate(
            [("ACCURACY", "accuracy"), ("PRECISION", "precision"),
             ("RECALL", "recall"), ("F1-SCORE", "f1")]
        ):
            card = panel(cards)
            card.grid(row=0, column=i, sticky="nsew",
                      padx=(0 if i == 0 else 6, 0 if i == 3 else 6))
            inner = tk.Frame(card, bg=CARD)
            inner.pack(fill="both", expand=True, padx=14, pady=12)
            label(inner, title, 8, MUTED, "bold", bg=CARD).pack(anchor="w")
            label(inner, f"{m[key]:.2f}%", 18, ACCENT if key == "accuracy" else TEXT,
                  "bold", bg=CARD).pack(anchor="w", pady=(4, 0))

        section_title(self.detail, f"Confusion matrix · {name}")
        cm_panel = panel(self.detail)
        cm_panel.pack(fill="x")
        cm_inner = tk.Frame(cm_panel, bg=CARD)
        cm_inner.pack(padx=16, pady=14, anchor="w")
        classes, cm = m["classes"], m["cm"]
        peak = max(max(row) for row in cm) or 1
        label(cm_inner, "Actual ↓   Predicted →", 8, MUTED, bg=CARD).grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 6))
        for j, cls in enumerate(classes):
            label(cm_inner, cls, 9, MUTED, "bold", bg=CARD, wraplength=84).grid(
                row=0, column=j + 1, padx=3, pady=(0, 6))
        for i, cls in enumerate(classes):
            label(cm_inner, cls, 9, MUTED, "bold", bg=CARD, anchor="e").grid(
                row=i + 1, column=0, sticky="e", padx=(0, 10), pady=3)
            for j in range(len(classes)):
                value = cm[i][j]
                t = value / peak
                fill = mix(INPUT, ACCENT, 0.15 + 0.85 * t) if value else INPUT
                tk.Label(cm_inner, text=str(value), width=7, height=2, bg=fill,
                         fg=ON_ACCENT if t > 0.55 else TEXT, font=F(11, "bold"),
                         bd=0).grid(row=i + 1, column=j + 1, padx=3, pady=3)

        section_title(self.detail, "Per-position results")
        pc = panel(self.detail)
        pc.pack(fill="x")
        for c in range(5):
            pc.columnconfigure(c, weight=3 if c == 0 else 2)
        for c, title in enumerate(["Position", "Precision", "Recall", "F1-Score", "Support"]):
            label(pc, title, 9, MUTED, "bold", bg=CARD,
                  anchor="w" if c == 0 else "e").grid(
                row=0, column=c, sticky="ew", padx=14, pady=(10, 6))
        for r, (cls, p, rc, f1, sup) in enumerate(m["per_class"], start=1):
            for c, text in enumerate([cls, f"{p:.2f}%", f"{rc:.2f}%", f"{f1:.2f}%", str(sup)]):
                label(pc, text, 10, TEXT, "bold" if c == 0 else "normal", bg=CARD,
                      anchor="w" if c == 0 else "e").grid(
                    row=r, column=c, sticky="ew", padx=14, pady=4)
        tk.Frame(pc, bg=CARD, height=6).grid(row=len(m["per_class"]) + 1, column=0, columnspan=5)


# ======================================================================
# 4. APP SHELL
# ======================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        global FONT_FAMILY
        FONT_FAMILY = pick_font(self)

        self.title("Out of Position - Player Classification System")
        self.geometry("1080x780")
        self.minsize(900, 640)
        self.configure(bg=BG)
        self._setup_styles()

        # ---- sidebar -------------------------------------------------
        self.sidebar = tk.Frame(self, bg=BG, width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Frame(self, bg=BORDER, width=1).pack(side="left", fill="y")

        brand = tk.Frame(self.sidebar, bg=BG)
        brand.pack(fill="x", padx=22, pady=(26, 18))
        Logo(brand).pack(side="left", padx=(0, 10))
        texts = tk.Frame(brand, bg=BG)
        texts.pack(side="left")
        label(texts, "OUT OF POSITION", 12, TEXT, "bold").pack(anchor="w")
        label(texts, "PLAYER CLASSIFICATION SYSTEM", 7, MUTED).pack(anchor="w")

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=22)

        nav = tk.Frame(self.sidebar, bg=BG)
        nav.pack(fill="x", padx=22, pady=18)
        self.nav_items = {}
        for key, kind, text in (
            ("dashboard", "dashboard", "Dashboard"),
            ("predict", "predict", "Position Prediction"),
            ("evaluation", "evaluation", "Model Evaluation"),
        ):
            item = NavItem(nav, kind, text, lambda k=key: self.show(k))
            item.pack(anchor="w", pady=7)
            self.nav_items[key] = item

        # ---- content -------------------------------------------------
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)
        self.loading = label(self.content, "Loading dataset and training models…", 12, MUTED)
        self.loading.place(relx=0.5, rely=0.5, anchor="center")

        self.pages = {}
        self.current = None
        self._queue = queue.Queue()
        threading.Thread(target=self._train, daemon=True).start()
        self.after(100, self._poll)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Dark.TCombobox", fieldbackground=INPUT, background=INPUT,
            foreground=TEXT, arrowcolor=MUTED, bordercolor=INPUT,
            lightcolor=INPUT, darkcolor=INPUT, padding=(10, 4),
            selectbackground=INPUT, selectforeground=TEXT,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", INPUT)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", INPUT)],
            selectforeground=[("readonly", TEXT)],
            bordercolor=[("focus", ACCENT)],
            arrowcolor=[("hover", TEXT)],
        )
        style.configure(
            "Dark.Vertical.TScrollbar", troughcolor=BG, background=BORDER,
            bordercolor=BG, lightcolor=BORDER, darkcolor=BORDER, arrowcolor=MUTED,
        )
        self.option_add("*TCombobox*Listbox.background", INPUT)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", ON_ACCENT)
        self.option_add("*TCombobox*Listbox.font", F(10))

    # ---- background training ----------------------------------------
    def _train(self):
        try:
            self._queue.put(("ok", ModelHub()))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self._queue.put(("error", exc))

    def _poll(self):
        try:
            status, payload = self._queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return

        if status == "error":
            messagebox.showerror(
                "Could not load your models",
                f"{payload}\n\nMake sure knn.py, Logistic.py, tree.py, random_forest.py, "
                f"gradient_boost.py and the 'dataset' file are in the same folder as "
                f"out_of_position.py:\n{BASE_DIR}",
            )
            self.destroy()
            return

        self.hub = payload
        self.loading.destroy()
        self.pages = {
            "dashboard": DashboardView(self.content, self.hub),
            "predict": PredictView(self.content, self.hub),
            "evaluation": EvaluationView(self.content, self.hub),
        }
        self.show("predict")

    def show(self, key):
        if key not in self.pages:
            return
        if self.current:
            self.pages[self.current].pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        self.current = key
        for k, item in self.nav_items.items():
            item.set_active(k == key)


if __name__ == "__main__":
    App().mainloop()
