#!/usr/bin/env python3
import os
from dotenv import load_dotenv
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
from sklearn.metrics import mean_squared_error, r2_score, auc, precision_recall_curve

import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY")  # for flashing messages

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
    # 1. Precision-Recall Curve (Good for showing screening reliability)
    plot_precision_recall(y_test, y_test_pred, threshold=7.0)

    # 2. Parity Plot with Error Tunnels (Shows prediction confidence)
    plot_parity_with_error(y_test, y_test_pred)
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

    # Feature Importance Table (top 20)
    top_importances = importances[indices]

    # Create a 2D list for cellText: [[Index, Importance Value], ...]
    table_data = []
    for index, importance in zip(indices, top_importances):
        # Format importance to a string with 4 decimal places
        table_data.append([str(index), f"{importance:.4f}"])

    # Define column headers
    col_labels = ["Feature Index", "Importance Value"]

    # Create the figure and axes, hiding standard plot axes for a clean table look
    fig, ax = plt.subplots(figsize=(4, 7))  # Adjusted figure size for a vertical table
    ax.axis('off')
    ax.axis('tight')

    # Draw the table
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='left')  # Align text to the left for readability

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale width and height slightly

    # Add a title and save the figure
    plt.title("Top 20 Feature Importances Table", pad=20)
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance_table.png"))
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
 <label>Upload Candidates SMILES file (optional):<br><input type=file name=candidates></label>
{% if session.candidate_smiles %}
  <p><em>Using {{ session.candidate_smiles|length }} molecules from previous upload (persisted in session).</em></p>
{% endif %}
<br><br>

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

{% if user_recommendations %}
  <h2>Your Uploaded SMILES Recommendations (Top {{ user_recommendations|length }})</h2>
  <table border=1 cellpadding=5>
    <tr><th>SMILES</th><th>pAffinity</th><th>Predicted nM</th></tr>
    {% for rec in user_recommendations %}
      <tr>
        <td style="font-family: monospace;">{{ rec.smiles }}</td>
        <td>{{ "%.3f"|format(rec.pAffinity) }}</td>
        <td>{{ "%.1f"|format(rec.predicted_nM) }}</td>
      </tr>
    {% endfor %}
  </table>
{% endif %}

{% if db_recommendations %}
  <h2>BindingDB Default Inhibitors (Top {{ db_recommendations|length }})</h2>
  <table border=1 cellpadding=5>
    <tr><th>SMILES</th><th>pAffinity</th><th>Predicted nM</th></tr>
    {% for rec in db_recommendations %}
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

from flask import session

@app.route('/recommend', methods=['POST'])
def recommend():
    topk = int(request.form.get('topk', 10))

    # --- Load trained model ---
    model = trained_models.get(TARGET_NAME, None)
    if model is None:
        flash(f"Model for target '{TARGET_NAME}' not loaded.", "error")
        return redirect(url_for('index'))

    # --- Handle FASTA source ---
    if TARGET_NAME == "Beta-lactamase TEM":
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

    # --- Parse candidate inhibitors (upload or session) ---
    candidate_smiles = []
    candidates_file = request.files.get('candidates', None)

    if candidates_file and candidates_file.filename:
        ext = os.path.splitext(candidates_file.filename)[1].lower()
        try:
            if ext in ['.smi', '.txt']:
                candidates_file.stream.seek(0)
                candidate_smiles = [
                    line.decode('utf-8').strip().split()[0]  # decode bytes to str
                    for line in candidates_file.stream
                    if line.strip()
                ]
            else:
                candidates_file.stream.seek(0)
                df_cand = pd.read_csv(candidates_file)
                if 'smiles' in df_cand.columns:
                    candidate_smiles = list(df_cand['smiles'].dropna().astype(str))
        except Exception:
            candidate_smiles = []

        # ✅ Save uploaded SMILES in cookie-backed session
        print(candidate_smiles)
        session['candidate_smiles'] = candidate_smiles
    else:
        # ✅ Reuse previous uploaded SMILES if no new file is provided
        candidate_smiles = session.get('candidate_smiles', [])

    # --- Sequence search for target validation ---
    keywords = [w.lower() for w in TARGET_NAME.split() if len(w) > 2]
    length_range = (200, 400)

    try:
        candidates = find_target_like_sequences(
            fasta_path, keywords, length_range=length_range, strict=True
        )
        if not candidates:
            candidates = find_target_like_sequences(
                fasta_path, keywords, length_range=None, strict=False
            )
        if not candidates:
            flash("No sequences matching target found in FASTA.", "error")
            if TARGET_NAME != "Beta-lactamase TEM":
                os.remove(fasta_path)
            return redirect(url_for('index'))

        # --- BindingDB inhibitors cache ---
        cache_file = f"cache_{TARGET_NAME.replace(' ', '_').lower()}.pkl"
        default_smiles = []
        if os.path.exists(cache_file):
            df_cached = pd.read_pickle(cache_file)
            default_smiles = list(pd.unique(df_cached['Ligand SMILES'].dropna()))

        # --- Generate recommendations ---
        user_recommendations = []
        db_recommendations = []

        if candidate_smiles:
            user_recommendations = score_candidates(model, candidate_smiles, top_k=topk)
        if default_smiles:
            db_recommendations = score_candidates(model, default_smiles, top_k=topk)
        print(
            f"[DEBUG] {len(candidate_smiles)} molecules uploaded, {len(user_recommendations)} valid after fingerprinting.")

        return render_template_string(
            INDEX_HTML,
            user_recommendations=user_recommendations,
            db_recommendations=db_recommendations,
            target_name=TARGET_NAME,
            local_fasta_file=LOCAL_FASTA_FILE,
            session=session  # <--- add this
        )


    finally:
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


def plot_precision_recall(y_true, y_pred, threshold=7.0):
    # Convert pAffinity to Binary (1 = Active, 0 = Inactive)
    y_true_bin = (y_true >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_true_bin, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='darkblue', lw=2, label=f'PR AUC = {pr_auc:.2f}')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlabel('Recall (Fraction of real inhibitors found)')
    plt.ylabel('Precision (Fraction of predicted inhibitors that are real)')
    plt.title('Conclusion: Model Screening Performance')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "conclusion_pr_curve.png"))
    plt.close()


def plot_parity_with_error(y_true, y_pred):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.5, c='teal', edgecolors='k')

    # Draw the ideal 1:1 line
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, 'r-', lw=2, label="Ideal")

    # Draw +/- 1.0 log unit tunnels
    plt.fill_between(lims, [x - 1.0 for x in lims], [x + 1.0 for x in lims],
                     color='gray', alpha=0.2, label="+/- 1.0 Log Unit")

    plt.xlabel("Actual pAffinity")
    plt.ylabel("Predicted pAffinity")
    plt.title("Prediction Confidence Analysis")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "conclusion_parity_tunnel.png"))
    plt.close()
if __name__ == "__main__":
    try:
        train_and_load_model()
    except Exception as e:
        print(f"[!] Error during model training: {e}")
        exit(1)
    print("[+] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
