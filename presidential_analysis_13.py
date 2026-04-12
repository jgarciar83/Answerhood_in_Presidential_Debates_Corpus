## Presidential Dataset Analysis — Refactored ##

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import krippendorff

# ─────────────────────────────────────────────
# Load Datasets
# ─────────────────────────────────────────────

folder_path = r"C:\Users\jgarc\OneDrive\Desktop\Debate"

all_dfs = []
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".xlsx"):
        file_path = os.path.join(folder_path, file_name)
        all_dfs.append(pd.read_excel(file_path))

combined_df = pd.concat(all_dfs, ignore_index=True)
print("Combined shape:", combined_df.shape)

# ─────────────────────────────────────────────
# Filtering (before encoding — raw columns still exist)
# ─────────────────────────────────────────────

# Remove rows where any layer column contains "Instructions"
mask = (
    combined_df[["layer_1", "layer_2", "layer_3"]]
    .astype(str)
    .apply(lambda col: col.str.contains("Instructions", case=False, na=False))
    .any(axis=1)
)
combined_df = combined_df[~mask]

# Drop rows where layer_3 contains bare "ATTACK" (ambiguous — faulty spreadsheet)
# These cannot be reliably mapped to ATTACK_QUESTIONER or ATTACK_OPPONENT
attack_mask = combined_df["layer_3"].astype(str).str.contains(r'\bATTACK\b', case=False, na=False)
n_attack = attack_mask.sum()
if n_attack > 0:
    print(f"Dropping {n_attack} row(s) with ambiguous 'ATTACK' label in layer_3.")
    combined_df = combined_df[~attack_mask]

# Remove rows with missing questions
combined_df = combined_df[combined_df["question"].notna()].reset_index(drop=True)

# ─────────────────────────────────────────────
# Inter-Annotator Agreement (IAA)
# ─────────────────────────────────────────────
# Computed on raw label columns BEFORE encoding so we work with
# the original string annotations exactly as the coders entered them.

def compute_iaa(df, layer_col):
    """
    Pivot annotator labels for one layer, then compute:
      - Cohen's Kappa   (exactly 2 annotators)
      - Fleiss' Kappa   (3+ annotators)
      - Krippendorff's Alpha (any number, handles missing data)

    Requires columns: 'utterance_id', 'annotator_id', and the layer column.
    Multi-label cells (comma-separated) use only the FIRST label so that
    agreement is computed on the primary annotation decision.
    """
    # Keep only rows that have a real annotation for this layer
    sub = df[["utterance", "annotator_id", layer_col]].copy()
    sub = sub[sub[layer_col].notna()]
    sub = sub[~sub[layer_col].astype(str).str.strip().isin(["", "nan"])]

    # For multi-label cells, take only the first label
    sub[layer_col] = (
        sub[layer_col]
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )

    # Pivot: rows = items, columns = annotators
    pivot = sub.pivot_table(
        index="utterance",
        columns="annotator_id",
        values=layer_col,
        aggfunc="first"
    )

    annotators = pivot.columns.tolist()
    n_annotators = len(annotators)

    print(f"\n  Layer column : {layer_col}")
    print(f"  Annotators   : {n_annotators}  →  {annotators}")
    print(f"  Items with ≥1 annotation : {len(pivot)}")

    # Items annotated by ALL raters (may be 0 in partial-overlap designs)
    pivot_complete = pivot.dropna()
    print(f"  Items with all raters    : {len(pivot_complete)}")

    # Items annotated by AT LEAST 2 raters (needed for any agreement metric)
    pivot_overlap = pivot[pivot.notna().sum(axis=1) >= 2]
    print(f"  Items with ≥2 raters     : {len(pivot_overlap)}")

    if len(pivot_overlap) < 2:
        print("  ⚠  Too few overlapping items to compute agreement — skipping.")
        return

    # ── Convert string labels to integers ──
    # Use all values in the full pivot (not just overlap) so the mapping is stable
    all_labels = [v for v in pd.unique(pivot.values.ravel()) if pd.notna(v)]
    label_to_int = {lbl: i for i, lbl in enumerate(sorted(all_labels))}

    # For Krippendorff: use full pivot with NaN preserved (handles missing data)
    pivot_alpha = pivot.apply(lambda col: col.map(label_to_int))  # NaN stays NaN
    # krippendorff expects shape (n_annotators, n_items)
    alpha = krippendorff.alpha(
        reliability_data=pivot_alpha.T.values,
        level_of_measurement="nominal"
    )
    print(f"  Krippendorff's Alpha     : {alpha:.4f}  (handles missing data)")

    # For Cohen's / Fleiss': use only fully overlapping rows
    # If zero complete rows, fall back to pairwise Cohen's Kappa instead
    if len(pivot_complete) >= 2:
        pivot_int = pivot_complete.replace(label_to_int).astype(int)
        ratings_array = pivot_int.values  # shape (n_items, n_annotators)

        if n_annotators == 2:
            kappa = cohen_kappa_score(ratings_array[:, 0], ratings_array[:, 1])
            print(f"  Cohen's Kappa            : {kappa:.4f}")

        if n_annotators >= 3:
            agg, _ = aggregate_raters(ratings_array)
            fk = fleiss_kappa(agg)
            print(f"  Fleiss' Kappa            : {fk:.4f}")

    else:
        # No item was rated by all annotators — compute pairwise Cohen's Kappa
        # for every pair that shares at least 2 overlapping items
        print(f"  No fully overlapping items — computing pairwise Cohen's Kappa:")
        pairs = [(a, b) for i, a in enumerate(annotators)
                         for b in annotators[i+1:]]
        any_pair = False
        for a, b in pairs:
            shared = pivot[[a, b]].dropna()
            if len(shared) < 2:
                continue
            shared_int = shared.apply(lambda col: col.map(label_to_int)).astype(int)
            unique_labels = pd.unique(shared_int.values.ravel())
            if len(unique_labels) < 2:
                print(f"    {a} vs {b}: skipped — only one label present (n={len(shared)})")
                continue
            k = cohen_kappa_score(shared_int[a], shared_int[b])
            print(f"    {a} vs {b}: κ = {k:.4f}  (n={len(shared)})")
            any_pair = True
        if not any_pair:
            print("    ⚠  No annotator pair shares ≥2 items.")


def run_iaa(df):
    """Detect annotator count and run IAA for all three layers."""
    if "annotator_id" not in df.columns:
        print("⚠  'annotator_id' column not found — skipping IAA.")
        return
    if "utterance" not in df.columns:
        print("⚠  'utterance' column not found — skipping IAA.")
        return

    # ── Normalise known duplicate annotator IDs before any computation ──
    # 'Torney' and 'TM' are the same annotator entered under two different IDs.
    # Collapse to a single canonical ID so agreement is not inflated by
    # treating one person as two raters on the same items.
    id_aliases = {"Torney": "TM"}
    df = df.copy()
    df["annotator_id"] = df["annotator_id"].replace(id_aliases)
    print(f"\n  ⚠  Annotator ID normalisation applied:")
    for old_id, new_id in id_aliases.items():
        print(f"     '{old_id}' → '{new_id}' (same annotator, duplicate entry)")

    unique_annotators = df["annotator_id"].dropna().unique()
    n = len(unique_annotators)

    print("\n" + "="*50)
    print("  Inter-Annotator Agreement")
    print("="*50)
    print(f"  Unique annotator IDs found: {n}  →  {unique_annotators.tolist()}")

    if n < 2:
        print("  ⚠  Need at least 2 annotators to compute agreement.")
        return

    for layer_col in ["layer_1", "layer_2", "layer_3"]:
        if layer_col in df.columns:
            compute_iaa(df, layer_col)
        else:
            print(f"\n  ⚠  Column '{layer_col}' not found — skipping.")

    print()


# Run IAA on raw data (before encoding drops the original columns)
run_iaa(combined_df)

# ─────────────────────────────────────────────
# Label Maps
# ─────────────────────────────────────────────
# NOTE: All real labels start at 1. Zero (0) is reserved as the
# "no label" sentinel so it is never confused with a real class.

label_map_1 = {"DIRECT": 1, "NOT_DIRECT": 2}
label_map_2 = {"TOPIC_SHIFT": 1, "PASS_BLAME": 2, "STORY_TELLING": 3}
label_map_3 = {"ATTACK_QUESTIONER": 1, "ATTACK_OPPONENT": 2, "DEFERRED_RESPONSE": 3}

# ─────────────────────────────────────────────
# Label Encoding Helpers
# ─────────────────────────────────────────────

def encode_labels(cell, label_map):
    """Parse a cell (string, list, or NaN) and map tokens to integer codes.
    Warns if any token is not found in the label map (catches annotation typos).
    """
    if pd.isna(cell):
        return [0]

    if isinstance(cell, str):
        tokens = [x.strip() for x in cell.split(",") if x.strip()]
    elif isinstance(cell, list):
        tokens = [str(x).strip() for x in cell]
    else:
        return [0]

    # Warn on unknown tokens so annotation typos don't silently disappear
    unknown = [tok for tok in tokens if tok not in label_map]
    if unknown:
        print(f"  Warning: unknown label(s) {unknown} in cell: '{cell}'")

    codes = [label_map.get(tok, 0) for tok in tokens]
    return codes if codes else [0]


def normalize_labels(codes):
    """Ensure list, remove zeros, convert numpy ints, fall back to [0]."""
    if not isinstance(codes, list):
        return [0]
    cleaned = [int(x) for x in codes if isinstance(x, (int, np.integer)) and x != 0]
    return cleaned if cleaned else [0]


def enforce_layer_rules(row):
    """
    Enforce:
      - Layer 1 must have exactly one label (keep first if multiple).
      - Layers 2 and 3 only allowed when layer_1 == NOT_DIRECT (code 2).
    """
    layer1 = row["layer_1_code"]

    if len(layer1) > 1:
        layer1 = [layer1[0]]
    row["layer_1_code"] = layer1

    if layer1[0] != label_map_1["NOT_DIRECT"]:
        row["layer_2_code"] = [0]
        row["layer_3_code"] = [0]

    return row

# ─────────────────────────────────────────────
# Encode → Normalize → Enforce
# ─────────────────────────────────────────────

combined_df["layer_1_code"] = combined_df["layer_1"].apply(lambda c: encode_labels(c, label_map_1))
combined_df["layer_2_code"] = combined_df["layer_2"].apply(lambda c: encode_labels(c, label_map_2))
combined_df["layer_3_code"] = combined_df["layer_3"].apply(lambda c: encode_labels(c, label_map_3))

for col in ["layer_1_code", "layer_2_code", "layer_3_code"]:
    combined_df[col] = combined_df[col].apply(normalize_labels)

combined_df = combined_df.apply(enforce_layer_rules, axis=1)

# Drop raw annotation columns now that encoding is complete
combined_df = combined_df.drop(columns=["layer_1", "layer_2", "layer_3", "notes (optional)"])

# ─────────────────────────────────────────────
# Drop rows with no valid Layer 1 label
# (layer_1_code == [0] means encoding failed — bad/missing annotation)
# ─────────────────────────────────────────────

invalid_mask = combined_df["layer_1_code"].apply(lambda x: x == [0])
n_dropped = invalid_mask.sum()
if n_dropped > 0:
    print(f"\nDropping {n_dropped} row(s) with no valid Layer 1 label.")
    combined_df = combined_df[~invalid_mask].reset_index(drop=True)


# ─────────────────────────────────────────────
# Build Features and Labels
# ─────────────────────────────────────────────

combined_df["text"] = combined_df["question"] + " [SEP] " + combined_df["utterance"]
X = combined_df["text"].tolist()

y1 = combined_df["layer_1_code"].apply(lambda x: x[0]).tolist()   # single-label
y2 = np.array(combined_df["layer_2_code"].tolist(), dtype=object)
y3 = np.array(combined_df["layer_3_code"].tolist(), dtype=object)

# ─────────────────────────────────────────────
# Train / Test Split
# ─────────────────────────────────────────────

indices = np.arange(len(X))

# Stratified on Layer 1; same indices reused for layers 2 & 3
X_train, X_test, y1_train, y1_test, idx_train, idx_test = train_test_split(
    X, y1, indices,
    test_size=0.2,
    random_state=42,
    stratify=y1
)

y2_train, y2_test = y2[idx_train].tolist(), y2[idx_test].tolist()
y3_train, y3_test = y3[idx_train].tolist(), y3[idx_test].tolist()

# ─────────────────────────────────────────────
# Filter to NOT_DIRECT rows for Layers 2 & 3
# (DIRECT rows carry only the dummy [0] label and add noise)
# ─────────────────────────────────────────────

def filter_not_direct(X_split, y1_split, y2_split, y3_split):
    """Return X, y2, y3 keeping only rows where Layer 1 == NOT_DIRECT (2)."""
    nd_idx = [i for i, label in enumerate(y1_split) if label == 2]
    return (
        [X_split[i] for i in nd_idx],
        [y2_split[i] for i in nd_idx],
        [y3_split[i] for i in nd_idx],
    )

X_train_nd, y2_train_nd, y3_train_nd = filter_not_direct(X_train, y1_train, y2_train, y3_train)
X_test_nd,  y2_test_nd,  y3_test_nd  = filter_not_direct(X_test,  y1_test,  y2_test,  y3_test)

print(f"\nNOT_DIRECT rows — train: {len(X_train_nd)}, test: {len(X_test_nd)}")

# ─────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────

def sanity_check(y_train, y_test, name):
    print(f"\n---- Sanity check: {name} ----")
    print(f"  Empty lists   — train: {sum(len(x)==0 for x in y_train)}, "
          f"test: {sum(len(x)==0 for x in y_test)}")
    print(f"  Non-list rows — train: {sum(not isinstance(x, list) for x in y_train)}, "
          f"test: {sum(not isinstance(x, list) for x in y_test)}")
    print(f"  numpy ints    — train: {sum(any(isinstance(v, np.integer) for v in x) for x in y_train)}, "
          f"test: {sum(any(isinstance(v, np.integer) for v in x) for x in y_test)}")

sanity_check(y2_train_nd, y2_test_nd, "y2 (NOT_DIRECT only)")
sanity_check(y3_train_nd, y3_test_nd, "y3 (NOT_DIRECT only)")

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def build_layer1_pipeline():
    """Single-label logistic regression for Layer 1."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
        ("clf",   LogisticRegression(max_iter=10000, solver="saga", class_weight="balanced"))
    ])

def build_multilabel_pipeline():
    """One-vs-rest logistic regression for Layers 2 & 3."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
        ("clf",   OneVsRestClassifier(LogisticRegression(max_iter=10000, solver="saga", class_weight="balanced")))
    ])

# ─────────────────────────────────────────────
# Evaluation Functions
# ─────────────────────────────────────────────

def train_and_evaluate_layer1(X_tr, X_te, y_tr, y_te):
    """Train and evaluate the Layer 1 single-label classifier."""
    model = build_layer1_pipeline()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    print("\n" + "="*50)
    print("  Layer 1 Evaluation (DIRECT vs NOT_DIRECT)")
    print("="*50)
    print(classification_report(
        y_te, preds,
        target_names=["DIRECT", "NOT_DIRECT"],
        labels=[1, 2]
    ))
    return model


def train_and_evaluate(X_tr, X_te, y_tr, y_te, layer_name):
    """Binarize labels (zeros stripped), train, predict, and report."""

    # Strip dummy 0s before binarizing so the model never trains on them
    y_tr_clean = [[v for v in row if v != 0] for row in y_tr]
    y_te_clean = [[v for v in row if v != 0] for row in y_te]

    mlb = MultiLabelBinarizer()
    y_tr_bin = mlb.fit_transform(y_tr_clean)
    y_te_bin = mlb.transform(y_te_clean)

    model = build_multilabel_pipeline()
    model.fit(X_tr, y_tr_bin)
    preds_bin = model.predict(X_te)

    # Guard: OneVsRestClassifier squeezes to 1D when only one class exists.
    # Reshape to (n_samples, 1) so column slicing always works correctly.
    if y_te_bin.ndim == 1:
        y_te_bin = y_te_bin.reshape(-1, 1)
    if preds_bin.ndim == 1:
        preds_bin = preds_bin.reshape(-1, 1)

    classes = mlb.classes_

    print(f"\n{'='*50}")
    print(f"  {layer_name} Evaluation")
    print(f"{'='*50}")

    # All classes in the binarizer are now real (no 0 sentinel to filter)
    present_classes = np.where(y_te_bin.sum(axis=0) > 0)[0]

    if len(present_classes) == 0:
        print("No real classes present in test set.")
        return model, mlb

    target_names = [str(classes[i]) for i in present_classes]
    print(f"Classes present in test set: {target_names}")

    y_true_eval = y_te_bin[:, present_classes]
    y_pred_eval = preds_bin[:, present_classes]

    print(classification_report(
        y_true_eval,
        y_pred_eval,
        target_names=target_names,
        zero_division=0
    ))

    return model, mlb

# ─────────────────────────────────────────────
# Train & Evaluate All Layers
# ─────────────────────────────────────────────

# Layer 1 — full dataset
layer1_model = train_and_evaluate_layer1(X_train, X_test, y1_train, y1_test)

# Layers 2 & 3 — NOT_DIRECT rows only
multi_label_tasks = {
    "Layer 2": (X_train_nd, X_test_nd, y2_train_nd, y2_test_nd),
    "Layer 3": (X_train_nd, X_test_nd, y3_train_nd, y3_test_nd),
}

models = {}
for layer_name, (X_tr, X_te, y_tr, y_te) in multi_label_tasks.items():
    model, mlb = train_and_evaluate(X_tr, X_te, y_tr, y_te, layer_name)
    models[layer_name] = (model, mlb)