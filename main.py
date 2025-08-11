#!/usr/bin/env python3
import os
import time
import math
import tempfile
import warnings

import numpy as np
import pandas as pd
from Bio import SeqIO

from flask import Flask, request, render_template_string, redirect, url_for, flash

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # for flashing messages

# Config
BINDINGDB_TSV = "BindingDB_All.tsv"
CHUNKSIZE = 100_000
FP_RADIUS = 2
FP_SIZE = 2048
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
LOCAL_FASTA_FILE = "BindingDBTargetSequences.fasta"

# Globals for RDKit Morgan fingerprint
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

# Target name for this session — set here; can be extended later
TARGET_NAME = "Beta-lactamase TEM"

# Helper functions
def clean_affinity_value(val):
    if pd.isna(val):
        return None
    try:
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
    except Exception:
        pass
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return None
        if s[0] in (">", "<", "~"):
            s = s[1:].strip()
        s = s.replace(",", " ")
        for tok in s.split():
            try:
                return float(tok)
            except Exception:
                continue
    return None

def smiles_to_fp(smiles):
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        bitv = MORGAN_GEN.GetFingerprint(mol)
        arr = np.zeros((FP_SIZE,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bitv, arr)
        return arr
    except Exception:
        return None

def load_and_filter_bindingdb(target_name, force_rebuild=False):
    cache_file = f"cache_{target_name.replace(' ', '_').lower()}.pkl"
    if os.path.exists(cache_file) and not force_rebuild:
        return pd.read_pickle(cache_file)

    if not os.path.exists(BINDINGDB_TSV):
        raise FileNotFoundError(f"BindingDB TSV file not found: {BINDINGDB_TSV}")

    usecols = ['Target Name', 'Ligand SMILES', 'Kd (nM)', 'IC50 (nM)', 'Ki (nM)']
    chunks = []
    for chunk in pd.read_csv(BINDINGDB_TSV, sep='\t', usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
        filtered = chunk[chunk['Target Name'] == target_name]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df.to_pickle(cache_file)
    return df

def compute_paffinity(df):
    if df.empty:
        return df
    affinities = []
    for _, row in df.iterrows():
        v = None
        for col in ['Kd (nM)', 'IC50 (nM)', 'Ki (nM)']:
            val = row.get(col)
            parsed = clean_affinity_value(val)
            if parsed is not None:
                v = parsed
                break
        affinities.append(v)
    df = df.copy()
    df['affinity_nM'] = affinities
    df = df.dropna(subset=['affinity_nM'])
    df['pAffinity'] = -np.log10(df['affinity_nM'] * 1e-9)
    return df

def build_features(df, force_rebuild=False):
    cache_file = f"features_{df['Target Name'].iloc[0].replace(' ', '_').lower()}.npz"
    if os.path.exists(cache_file) and not force_rebuild:
        data = np.load(cache_file)
        return data['X'], data['y']

    fps = []
    ys = []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row['Ligand SMILES'])
        if fp is not None:
            fps.append(fp)
            ys.append(row['pAffinity'])
    if not fps:
        raise RuntimeError("No valid fingerprints found.")
    X = np.vstack(fps)
    y = np.array(ys)
    np.savez_compressed(cache_file, X=X, y=y)
    return X, y

def train_model(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [200, 400, 800],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rnd = RandomizedSearchCV(rf, param_dist, n_iter=8, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, random_state=42)
    rnd.fit(X_train, y_train)
    best_rf = rnd.best_estimator_

    # Validation predictions
    y_val_pred = best_rf.predict(X_val)
    val_rmse = math.sqrt(mean_squared_error(y_val, y_val_pred))

    # Train final model on train+val
    best_rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    # Test predictions
    y_test_pred = best_rf.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # --- PLOTS ---

    # Predicted vs Actual (Validation)
    plt.figure(figsize=(6,6))
    plt.scatter(y_val, y_val_pred, alpha=0.6, edgecolors='k')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
    plt.xlabel("Actual pAffinity (Validation)")
    plt.ylabel("Predicted pAffinity (Validation)")
    plt.title("Validation: Predicted vs Actual")
    plt.savefig(os.path.join(PLOTS_DIR, "pred_vs_actual_val.png"))
    plt.close()

    # Predicted vs Actual (Test)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual pAffinity (Test)")
    plt.ylabel("Predicted pAffinity (Test)")
    plt.title("Test: Predicted vs Actual")
    plt.savefig(os.path.join(PLOTS_DIR, "pred_vs_actual_test.png"))
    plt.close()

    # Residuals plot (Test)
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k')
    plt.hlines(0, min(y_test_pred), max(y_test_pred), colors='r', linestyles='dashed')
    plt.xlabel("Predicted pAffinity (Test)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Test Residuals Plot")
    plt.savefig(os.path.join(PLOTS_DIR, "residuals_test.png"))
    plt.close()

    # Feature importance (top 20)
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(indices)), importances[indices], color='b', align='center')
    plt.xticks(range(len(indices)), indices)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
    plt.close()

    return best_rf, val_rmse, test_rmse, test_r2


def find_target_like_sequences(fasta_path, keywords, length_range=None, strict=True):
    candidates = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        desc = rec.description.lower()
        seq_len = len(rec.seq)
        keyword_ok = all(kw in desc for kw in keywords)
        length_ok = True
        if strict and length_range:
            length_ok = (length_range[0] <= seq_len <= length_range[1])
        if strict:
            if keyword_ok and length_ok:
                candidates.append(rec)
        else:
            if keyword_ok:
                candidates.append(rec)
    return candidates

def score_candidates(model, smiles_list, top_k=10):
    fps = []
    smis = []
    for smi in smiles_list:
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            smis.append(smi)
    if not fps:
        return []
    Xcand = np.vstack(fps)
    preds = model.predict(Xcand)
    order = np.argsort(-preds)
    results = []
    for idx in order[:top_k]:
        p = preds[idx]
        nM = 10 ** (-p) * 1e9
        results.append({'smiles': smis[idx], 'pAffinity': p, 'predicted_nM': nM})
    return results

# Globals to store trained models in memory keyed by target
trained_models = {}

# -- HTML Templates --
INDEX_HTML = """
<!doctype html>
<title>AIREP Inhibitor Recommendation</title>
<h1>AIREP Inhibitor Recommendation for Target: {{ target_name }}</h1>

<form method=post enctype=multipart/form-data action="{{ url_for('recommend') }}">
  {% if target_name != "Beta-lactamase TEM" %}
    <label>Upload FASTA file:<br><input type=file name=fasta required></label><br><br>
  {% else %}
    <p><em>Using local FASTA file: {{ local_fasta_file }}</em></p>
  {% endif %}
  <label>Upload Candidates SMILES file (optional):<br><input type=file name=candidates></label><br><br>
  <label>Top K recommendations:<br><input type=number name=topk value=10 min=1 max=100></label><br><br>
  <button type=submit>Recommend Inhibitors</button>
</form>

{% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}
    <hr>
    {% for category, message in messages %}
      <p style="color: {% if category == 'error' %}red{% else %}green{% endif %};"><strong>{{ message }}</strong></p>
    {% endfor %}
  {% endif %}
{% endwith %}

{% if recommendations %}
  <h2>Top {{ recommendations|length }} Inhibitor Recommendations</h2>
  <table border=1 cellpadding=5>
    <tr><th>SMILES</th><th>pAffinity</th><th>Predicted nM</th></tr>
    {% for rec in recommendations %}
      <tr>
        <td style="font-family: monospace;">{{ rec.smiles }}</td>
        <td>{{ "%.3f"|format(rec.pAffinity) }}</td>
        <td>{{ "%.1f"|format(rec.predicted_nM) }}</td>
      </tr>
    {% endfor %}
  </table>
{% endif %}
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, target_name=TARGET_NAME, local_fasta_file=LOCAL_FASTA_FILE)

@app.route('/recommend', methods=['POST'])
def recommend():
    topk = int(request.form.get('topk', 10))

    model = trained_models.get(TARGET_NAME, None)
    if model is None:
        flash(f"Model for target '{TARGET_NAME}' not loaded.", "error")
        return redirect(url_for('index'))

    if TARGET_NAME == "Beta-lactamase TEM":
        # Use local fasta file
        fasta_path = LOCAL_FASTA_FILE
        if not os.path.exists(fasta_path):
            flash(f"Local FASTA file '{LOCAL_FASTA_FILE}' not found.", "error")
            return redirect(url_for('index'))
    else:
        if 'fasta' not in request.files:
            flash("FASTA file is required for recommendation.", "error")
            return redirect(url_for('index'))
        fasta_file = request.files['fasta']
        with tempfile.NamedTemporaryFile(delete=False) as tmpf:
            fasta_path = tmpf.name
            fasta_file.save(fasta_path)

    candidates_file = request.files.get('candidates', None)

    candidate_smiles = []
    if candidates_file:
        ext = os.path.splitext(candidates_file.filename)[1].lower()
        try:
            if ext in ['.smi', '.txt']:
                candidates_file.stream.seek(0)
                candidate_smiles = [line.strip().split()[0] for line in candidates_file.stream if line.strip()]
            else:
                candidates_file.stream.seek(0)
                df_cand = pd.read_csv(candidates_file)
                if 'smiles' in df_cand.columns:
                    candidate_smiles = list(df_cand['smiles'].dropna().astype(str))
        except Exception:
            candidate_smiles = []

    keywords = [w.lower() for w in TARGET_NAME.split() if len(w) > 2]
    length_range = (200, 400)

    try:
        candidates = find_target_like_sequences(fasta_path, keywords, length_range=length_range, strict=True)
        if not candidates:
            candidates = find_target_like_sequences(fasta_path, keywords, length_range=None, strict=False)
        if not candidates:
            flash("No sequences matching target found in FASTA.", "error")
            if TARGET_NAME != "Beta-lactamase TEM":
                os.remove(fasta_path)
            return redirect(url_for('index'))

        cache_file = f"cache_{TARGET_NAME.replace(' ', '_').lower()}.pkl"
        if os.path.exists(cache_file):
            df_cached = pd.read_pickle(cache_file)
            default_smiles = list(pd.unique(df_cached['Ligand SMILES'].dropna()))
        else:
            default_smiles = []

        all_smiles = candidate_smiles + default_smiles
        recommendations = score_candidates(model, all_smiles, top_k=topk)

        return render_template_string(INDEX_HTML, recommendations=recommendations, target_name=TARGET_NAME, local_fasta_file=LOCAL_FASTA_FILE)
    finally:
        # Remove temp fasta only if not local fasta
        if TARGET_NAME != "Beta-lactamase TEM":
            os.remove(fasta_path)

def train_and_load_model():
    print(f"[+] Loading and filtering data for target: {TARGET_NAME} ...")
    df = load_and_filter_bindingdb(TARGET_NAME, force_rebuild=False)
    if df.empty:
        raise RuntimeError(f"No BindingDB data found for target '{TARGET_NAME}'.")
    df = compute_paffinity(df)
    if df.empty:
        raise RuntimeError(f"No usable affinities after cleaning for target '{TARGET_NAME}'.")
    X, y = build_features(df, force_rebuild=False)
    print(f"[+] Training model for target: {TARGET_NAME} ...")
    model, val_rmse, test_rmse, test_r2 = train_model(X, y)
    print(f"[+] Training complete. Validation RMSE={val_rmse:.3f}, Test RMSE={test_rmse:.3f}, Test R^2={test_r2:.3f}")
    model_file = f"model_{TARGET_NAME.replace(' ', '_').lower()}.joblib"
    joblib.dump(model, model_file)
    trained_models[TARGET_NAME] = model

if __name__ == "__main__":
    try:
        train_and_load_model()
    except Exception as e:
        print(f"[!] Error during model training: {e}")
        exit(1)
    print("[+] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
