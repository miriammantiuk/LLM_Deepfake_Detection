import pandas as pd
import json
import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, roc_auc_score, roc_curve, auc)
import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer, util
import itertools
import torch
import plotly.express as px




# ---------------------------------------------------------
# GLOBALE KONFIGURATION
# ---------------------------------------------------------
JSON_FOLDER = '.'
INFO_DATEI = 'dataset_info.xlsx'
VIDEO_SOURCE_PATH = r'data\processed'
RESULTS_FOLDER = 'results'
SUMMARIZED_FILE = os.path.join(RESULTS_FOLDER, 'results_summarized.xlsx')

# ========== BASE MODEL NAMES ==========
BASE_MODEL_DISPLAY_NAMES = {
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'gpt5_2_instant': 'GPT 5.2',
    'gpt5_2': 'GPT 5.2',
    'qwen3.5-397b-a17b': 'Qwen 3.5',
    'kimi-k2.5': 'Kimi k2.5',
    'seed-2.0-lite': 'Seed 2.0 Lite'
}

# ========== VARIANT ORDER ==========
# Definiert die logische Reihenfolge der Prompt-Varianten für ALLE Tabellen und Plots
VARIANT_ORDER = ['', '+I', '+T', '+I+T']

# Farben zentral definieren
def get_model_color(model_name):
    """
    Identifiziert die Variante und weist den passenden Farb-Shade zu.
    Sucht nach technischen Suffixen (_indicators) und Labels (+I).
    """
    name_str = str(model_name).lower() # Wir arbeiten hier mit kleingeschrieben
    
    # 1. Familie identifizieren
    if 'gemini' in name_str: family = 'Gemini'
    elif 'gpt' in name_str: family = 'GPT'
    elif 'qwen' in name_str: family = 'Qwen'
    elif 'kimi' in name_str: family = 'Kimi'
    elif 'seed' in name_str: family = 'Seed'
    else: family = 'Other'
        
    # 2. Variante (Index 0-3) identifizieren 
    if '_thinking_indicators' in name_str or '+i+t' in name_str:
        variant = 3 # Hellste Farbe
    elif '_thinking' in name_str or '+t' in name_str:
        variant = 2
    elif '_indicators' in name_str or '+i' in name_str:
        variant = 1
    else:
        variant = 0 # Baseline (Dunkelste Farbe)
        
    # Farbpaletten (Dunkel -> Hell)
    palettes = {
        'Gemini': ['#08519c', '#3182bd', '#6baed6', '#bdd7e7'],
        'GPT':    ['#006d2c', '#31a354', '#74c476', '#bae4b3'],
        'Qwen':   ['#a63603', '#e6550d', '#fd8d3c', '#fdbe85'],
        'Kimi':   ['#54278f', '#756bb1', '#9e9ac8', '#cbc9e2'],
        'Seed':   ['#ce1256', '#e7298a', '#df65b0', '#d4b9da'],
        'Other':  ['#252525', '#737373', '#bdbdbd', '#f0f0f0']
    }
    
    return palettes[family][variant]

# Listen zentral definieren
RUN_SUFFIXES = ["_1", "_2", "_3"]
META_COLS = ['dataset', 'gender', 'video_length','deepfake_category','deepfake_type', 'audio']

# Plotting Style
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0}) 

# Hauptordner für Plots
base_plot_folder = 'plots'
os.makedirs(base_plot_folder, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# Stopwörter für die Keyword-Analyse 
DOMAIN_STOPS = list(ENGLISH_STOP_WORDS) + [
    'video', 'clip', 'footage', 'appears', 'seems', 'looks', 'shows', 
    'detected', 'generated', 'ai', 'content', 'likely', 'probability', 
    'confidence', 'based', 'analysis', 'observe', 'observed', 'visual', 
    'anomalies', 'artifacts', 'signs', 'potential', 'deepfake'
]

# Globale Anker-Definitionen — genutzt von cluster_keywords UND analyze_region_coverage
KEYWORD_ANCHORS = {
    # MARE 
    'Skin':          'skin, cheek, forehead, complexion, dermal, face',
    'Nose':          'nose, nostril, nasal',
    'Mouth':         'mouth, lip, lips',
    'Teeth':         'tooth, teeth',
    'Eye':           'right-eye, left-eye, eye, ocular',
    'Eyebrow':       'right-eyebrow, left-eyebrow, eyebrow, brow',
    'Chin':          'chin, jaw, jawline, lower face',
    'Beard':         'beard, mustache, moustache, goatee',
    'Hairline':      'hairline, hair line, hair',
    'Ear':           'ear, ears',
    # Ergänzte Keywords Body
    'Head_Neck':     'neck, head, throat',
    'Torso':         'shoulder, torso, chest, arm, posture',
    'Hands':         'hand, hands, finger, fingers',
    # Ergänzte Keywords Background
    'Lighting':      'lighting, illumination, shadow, brightness, light source',
    'Scene':         'scene, background, environment, setting',
    'Temporal':      'flickering, temporal, inconsistency, frame rate',
    'General_Artifacts':     'edge, noise, blur, blending, compression, artifact, consistent, natural',
  
    # Ergänzte Keywords Audio
    'Voice':         'voice, speech, pronunciation, accent, audio, sound, tone',
    'Lip_Sync':      'lip sync, synchronization, mouth movement, lipsync',
}

KEYWORD_HIERARCHY = {
    'Skin':            ('Face',       'Frame'),
    'Nose':            ('Face',       'Frame'),
    'Mouth':           ('Face',       'Frame'),
    'Teeth':           ('Face',       'Frame'),
    'Eye':             ('Face',       'Frame'),
    'Eyebrow':         ('Face',       'Frame'),
    'Chin':            ('Face',       'Frame'),
    'Beard':           ('Face',       'Frame'),
    'Hairline':        ('Face',       'Frame'),
    'Ear':             ('Face',       'Frame'),
    'Head_Neck':       ('Body',       'Frame'),
    'Torso':           ('Body',       'Frame'),
    'Hands':           ('Body',       'Frame'),
    'Lighting':        ('Background', 'Frame'),
    'Scene':           ('Background', 'Frame'),
    'Temporal':        ('Background', 'Frame'),
    'Voice':           ('Audio',      'Audio'),
    'Lip_Sync':        ('Audio',      'Audio'),
    'General_Artifacts': ('Background', 'Frame'),
}

# SBERT zentral laden (für semantische Analysen)
_model_sbert = None

def get_sbert():
    global _model_sbert
    if _model_sbert is None:
        print(" -> Lade SBERT-Modell (all-MiniLM-L6-v2)...")
        _model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    return _model_sbert

# ---------------------------------------------------------
# HILFSFUNKTIONEN (METRIKEN & LOGIK)
# ---------------------------------------------------------

def run_baseline_distribution_analysis(df_master, save_folder="Baseline_Distributions"):
    """
    Erstellt Basisverteilungs-Plots für ALLE Features im Datensatz, um 
    Sampling Bias vor der eigentlichen Auswertung sichtbar zu machen.
    - Numerische Features -> Histogramm
    - Kategoriale Features -> Countplot (Balkendiagramm)
    """
    print("\n=== Baseline Distribution Analysis (Datensatzübersicht) ===")
    
    target_folder = os.path.join(base_plot_folder, save_folder)
    os.makedirs(target_folder, exist_ok=True)
    
    # CRITICAL: Duplikate entfernen, um auf exakt N=50 Videos zu kommen
    if 'video_id' in df_master.columns:
        df_unique = df_master.drop_duplicates(subset=['video_id']).copy()
    else:
        print(" Warnung: Spalte 'video_id' nicht gefunden. Verteilung könnte verzerrt sein!")
        df_unique = df_master.copy()
        
    n_videos = len(df_unique)
    print(f" -> Analysiere einzigartige Videos: N = {n_videos}")

    for feat in META_COLS:
        if feat not in df_unique.columns:
            continue
            
        plt.figure(figsize=(7, 4))
        
        # 1. FALL: Numerische Features (z.B. video_length)
        if pd.api.types.is_numeric_dtype(df_unique[feat]):
            sns.histplot(data=df_unique, x=feat, discrete=True, kde=True, 
             color='#4e79a7', edgecolor='white', linewidth=1.2)
            plt.title(f'Basisverteilung (Numerisch): {feat} (N={n_videos})', fontsize=12, fontweight='bold')
            plt.xlabel(feat, fontsize=10)
        
        # 2. FALL: Kategoriale Features (z.B. dataset, deepfake_category)
        else:
            # Sortiere Balken nach Häufigkeit für eine sauberere Optik
            order = df_unique[feat].value_counts().index
            ax = sns.countplot(data=df_unique, x=feat, order=order, palette='Set2')
            
            plt.title(f'Basisverteilung (Kategorisch): {feat} (N={n_videos})', fontsize=12, fontweight='bold')
            plt.xlabel('')
            plt.xticks(rotation=30, ha='right')
            
            # Schreibe die genaue Anzahl (n=...) über jeden Balken
            for p in ax.patches:
                n_count = int(p.get_height())
                ax.annotate(f'n={n_count}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, 
                            color='black', xytext=(0, 3), textcoords='offset points')

        # Gemeinsame Formatierung für beide Diagrammtypen
        plt.ylabel('Anzahl der Videos', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Y-Achse etwas höher setzen, damit die 'n=' Labels nicht abgeschnitten werden
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax * 1.15)
        
        plt.tight_layout()
        file_path = os.path.join(target_folder, f'Dist_{feat}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" -> Verteilungs-Plot für '{feat}' gespeichert.")

    print("=== Baseline Analysis abgeschlossen. ===\n")

def calculate_metrics(y_true, y_pred, y_prob=None):
    if len(y_true) == 0:
        return {'Accuracy (%)': 0, 'Precision (%)': 0, 'Recall (%)': 0, 'F1-Score (%)': 0, 'ROC AUC': 'N/A', 'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    
    roc_auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = None 

    return {
        'Accuracy (%)': round(acc * 100, 1),
        'Precision (%)': round(precision * 100, 1),
        'Recall (%)': round(recall * 100, 1),
        'F1-Score (%)': round(f1 * 100, 1),
        'ROC AUC': round(roc_auc, 1) if roc_auc is not None else 'N/A',
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

def get_word_count(val):
    if isinstance(val, list): return len(" ".join(str(v) for v in val).split())
    elif isinstance(val, str): return len(val.split())
    return 0

def get_plot_label(model_name):
    """
    Konvertiert den technischen Modellnamen automatisch in einen Plot-Label.
    - Nutzt BASE_MODEL_DISPLAY_NAMES für bekannte Modelle.
    - Erkennt suffixbasierte Varianten wie _indicators und _thinking.
    - Für unbekannte Modelle: nimmt den "rohen" Modellnamen (ohne apply map) und ergänzt ggf. das Suffix.
    """
    # Priorität: Bekannte Basen + Mapping
    sorted_bases = sorted(BASE_MODEL_DISPLAY_NAMES.keys(), key=len, reverse=True)
    for base_key in sorted_bases:
        if model_name.startswith(base_key):
            rest = model_name[len(base_key):]
            display_name = BASE_MODEL_DISPLAY_NAMES[base_key]
            suffix = ''
            if '_indicators' in rest:
                suffix += '+I'
            if '_thinking' in rest:
                suffix += '+T'
            return display_name + suffix

    # Unbekanntes Modell: Basisname + optionaler Suffix-Parsing
    suffix_map = [('_indicators', '+I'), ('_thinking', '+T')]
    display_name = model_name
    suffix = ''
    for key, token in suffix_map:
        if display_name.endswith(key):
            display_name = display_name[: -len(key)]
            suffix += token

    # Modellname sinnvoll abkürzen (z.B. Falls technisch mit Pfad/Version): falls leer, Original verwenden
    if display_name.strip() == '':
        display_name = model_name

    return display_name + suffix

# ---------------------------------------------------------
# ETL & AGGREGATION 
# ---------------------------------------------------------

def run_analysis(suffix=""):
    output_datei = os.path.join(RESULTS_FOLDER, f'results{suffix}.xlsx')
    print(f"--- JSON Extraktion für {suffix} ---")
    json_files = glob.glob(os.path.join(JSON_FOLDER, f'*{suffix}.json'))
    
    if not json_files: return

    all_data = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            df_t = pd.DataFrame(data)
            df_t['model'] = os.path.splitext(os.path.basename(file_path))[0]
            df_t['video_id'] = df_t['video_id'].astype(str).str.replace(r'(?i)\.mp4$', '', regex=True)
            if 'justification' in df_t.columns:
                df_t['justification_length'] = df_t['justification'].apply(get_word_count)
            all_data.append(df_t)
        except Exception as e:
            print(f" [WARN] Fehler beim Laden von {file_path}: {e}")

    if not all_data: return

    df_gesamt = pd.concat(all_data, ignore_index=True)
    df_gesamt['y_pred'] = df_gesamt['assessment'].map({"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1})
    df_gesamt.to_excel(output_datei, index=False)
    print(f" gespeichert: {output_datei}")

def run_aggregation_and_benchmark():
    """
    1. Aggregation: Liest Einzel-Runs und merged sie (Manuell, um die Master-Datei zu bauen).
    2. Benchmarking: Nutzt iterate_runs() für die Berechnung der Metriken.
    """
    print("\n=== Aggregation & Benchmarking ===")
    
    # --- TEIL 1: AGGREGATION (Datei erstellen) ---
    dfs = {}
    
    # 1. Daten laden
    for suffix in RUN_SUFFIXES:
        try:
            df = pd.read_excel(os.path.join(RESULTS_FOLDER, f'results{suffix}.xlsx'))
            
            # Base Model erstellen (für Merge Key)
            df['base_model'] = df['model'].str.replace(rf'{suffix}$', '', regex=True)

            # Definieren der gewünschten Reihenfolge
            cols = list(df.columns)
            if 'video_id' in cols and 'base_model' in cols:
                # Alles außer video_id und base_model
                other_cols = [c for c in cols if c not in ['video_id', 'base_model']]
                # Neue Reihenfolge erzwingen
                df = df[['video_id', 'base_model'] + other_cols]

            # Umbenennen für Merge (außer Keys)
            merge_keys = ['video_id', 'base_model']
            rename_map = {c: f'{c}{suffix}' for c in df.columns if c not in merge_keys}
            dfs[suffix] = df.rename(columns=rename_map)
            
        except FileNotFoundError: 
            print(f"Warnung: results{suffix}.xlsx fehlt.")
            return

    # 2. Merge 
    df_final = dfs[RUN_SUFFIXES[0]]
    for suffix in RUN_SUFFIXES[1:]:
        df_final = pd.merge(df_final, dfs[suffix], on=['video_id', 'base_model'], how='outer')

    # 3. Ground Truth anhängen
    try:
        df_info = pd.read_excel(INFO_DATEI)
        df_info['video_id'] = df_info['video_id'].astype(str)
        df_final = pd.merge(df_final, df_info, on='video_id', how='left')

        if 'deepfake' in df_final.columns:
            df_final['y_true'] = df_final['deepfake'].map({"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1})
    except Exception as e:
        print(f" [WARN] Ground Truth konnte nicht geladen werden: {e}")
    
    # Speichern
    df_final.to_excel(SUMMARIZED_FILE, index=False)
    print(f" {SUMMARIZED_FILE} erstellt.")

    # --- TEIL 2: BENCHMARKING ---
    print(" -> Berechne Metriken (Mean ± Std) für verschiedene Thresholds...")
    results = []
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for run_label, df_run in iterate_runs(df_final):
        for model_name, group in df_run.groupby('base_model'):
            
            # Probability holen (falls vorhanden)
            y_score = None
            if 'probability_fake' in group.columns:
                y_score = pd.to_numeric(group['probability_fake'], errors='coerce').fillna(0)
                if y_score.max() > 1.0: y_score = y_score / 100.0
            
            # Für jeden Threshold Metriken berechnen
            for threshold in thresholds:
                if y_score is not None:
                    y_pred_thresh = (y_score > threshold).astype(int)
                else:
                    # Fallback: Verwende original y_pred nur für Threshold 0.5
                    if threshold == 0.5:
                        y_pred_thresh = group['y_pred']
                    else:
                        continue  # Überspringe andere Thresholds, wenn keine Probs vorhanden
                
                # Metriken berechnen
                metrics = calculate_metrics(group['y_true'], y_pred_thresh, y_score)
                
                # Metadaten
                metrics.update({
                    'Model': f"{model_name} ({run_label.replace('_', ' ')})", 
                    'Base_Model': model_name,
                    'Type': run_label.replace('_', ' '),
                    'Threshold': threshold,
                    'Videos': len(group)
                })
                results.append(metrics)

    # --- TEIL 3: REPORTS ERSTELLEN ---
    if not results:
        print(" Keine Benchmark-Ergebnisse.")
        return

    df_metrics = pd.DataFrame(results)

    # Schöne Namen hinzufügen
    df_metrics['Display_Name'] = df_metrics['Base_Model'].map(BASE_MODEL_DISPLAY_NAMES).fillna(df_metrics['Base_Model'])

    # A) Detailed Report
    cols_order = ['Model', 'Display_Name', 'Type', 'Threshold', 'Accuracy (%)', 'F1-Score (%)', 'Videos']
    cols_order = [c for c in cols_order if c in df_metrics.columns] + [c for c in df_metrics.columns if c not in cols_order]
    df_metrics[cols_order].to_excel(os.path.join(RESULTS_FOLDER, 'benchmark_detailed_runs.xlsx'), index=False)
    print(" benchmark_detailed_runs.xlsx erstellt.")

    # B) Scientific Report & LaTeX Generierung
    metric_cols = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC AUC']
    metric_cols = [c for c in metric_cols if c in df_metrics.columns]
    
    with pd.ExcelWriter(os.path.join(RESULTS_FOLDER, 'benchmark_scientific_report.xlsx')) as writer:
        for threshold in thresholds:
            # 1. Daten filtern für aktuellen Threshold
            sub_df = df_metrics[df_metrics['Threshold'] == threshold]
            if sub_df.empty:
                continue
                
            # 2. Aggregation (Mean & Std) berechnen
            summary = sub_df.groupby('Base_Model')[metric_cols].agg(['mean', 'std'])
            summary.index = summary.index.map(get_plot_label) # Schöne Namen
            summary = summary.round(1)
            
            # 3. In Excel-Sheet speichern
            summary.to_excel(writer, sheet_name=f'Threshold_{threshold}')
            
            # 4. === LaTeX Auto-Generierung für DIESEN Threshold ===
            # Ermittle die Indizes der Bestwerte (basierend auf dem Mittelwert 'mean')
            idx_max_acc = summary[('Accuracy (%)', 'mean')].idxmax()
            idx_max_f1 = summary[('F1-Score (%)', 'mean')].idxmax()
            
            latex_df = pd.DataFrame(index=summary.index)
            
            for metric in metric_cols:
                if metric in summary.columns.levels[0]:
                    # Hole Mittelwerte und Standardabweichungen als formatierte Strings
                    means = summary[(metric, 'mean')].map('{:.1f}'.format).str.replace('.', '{,}')
                    stds = summary[(metric, 'std')].map('{:.1f}'.format).str.replace('.', '{,}')
                    
                    formatted_cells = []
                    for idx in summary.index:
                        # Prüfen, ob dieser Wert fett markiert werden soll
                        is_best_acc = (metric == 'Accuracy (%)' and idx == idx_max_acc)
                        is_best_f1 = (metric == 'F1-Score (%)' and idx == idx_max_f1)
                        
                        if is_best_acc or is_best_f1:
                            cell_text = f"$\\mathbf{{{means[idx]} \\pm {stds[idx]}}}$"
                        else:
                            # Standard-Format
                            cell_text = f"${means[idx]} \\pm {stds[idx]}$"
                        
                        formatted_cells.append(cell_text)
                    
                    # Spaltenname für LaTeX säubern
                    safe_metric_name = metric.replace('%', r'\%')
                    latex_df[safe_metric_name] = formatted_cells

            # Layout-Feinschliff: Index zu normaler Spalte machen
            latex_df = latex_df.reset_index()
            latex_df.rename(columns={latex_df.columns[0]: 'Base Model'}, inplace=True)

            # Speichere die .tex Datei
            filename = os.path.join(RESULTS_FOLDER, f'benchmark_results_{threshold}.tex')
            col_format = 'l' + 'c' * (len(latex_df.columns) - 1)
            latex_df.to_latex(filename, escape=False, index=False, column_format=col_format)
            print(f" LaTeX-Datei für Threshold {threshold} erstellt: {filename}")

    print("\n=== Generiere F1-Score Heatmap für alle Modelle ===")

    # 1. Sicherstellen, dass die Daten numerisch sind
    df_metrics['Threshold'] = pd.to_numeric(df_metrics['Threshold'])
    df_metrics['F1-Score (%)'] = pd.to_numeric(df_metrics['F1-Score (%)'])

    # 2. Saubere Namen für die Heatmap erstellen (ohne Suffixe, aber mit +I/+T)
    df_metrics['Clean_Name'] = df_metrics['Base_Model'].apply(get_plot_label)

    # 3. Pivot-Tabelle erstellen 
    pivot_df = df_metrics.pivot_table(
        index='Clean_Name', 
        columns='Threshold',
        values='F1-Score (%)',
        aggfunc='mean'
    )

    # 4. Globale Sortierung basierend auf BASE_MODEL_DISPLAY_NAMES und VARIANT_ORDER erzwingen
    familien = sorted(list(set(BASE_MODEL_DISPLAY_NAMES.values())))
    
    sorted_list = []
    for fam in familien:
        for var in VARIANT_ORDER:
            name = f"{fam}{var}"
            if name in pivot_df.index:
                sorted_list.append(name)
                
    # Sortierung anwenden (Fallback für Modelle, die evtl. anders heißen)
    remaining = [x for x in pivot_df.index if x not in sorted_list]
    pivot_df = pivot_df.reindex(sorted_list + remaining)

    # 5. Heatmap zeichnen
    plt.figure(figsize=(10, 8)) 
    
    ax = sns.heatmap(
        pivot_df, 
        annot=True,       
        fmt=".1f",        
        cmap="YlGnBu",    
        cbar_kws={'label': 'F1-Score (%)'},
        linewidths=.5,    
        vmin=0, vmax=100  
    )

    # 6. VISUELLE CLUSTER BILDEN (Dynamische weiße Trennlinien zwischen Familien)
    y_lines = []
    current_fam = None
    for i, name in enumerate(pivot_df.index):
        # Das erste Wort ist meist die Familie (Gemini, GPT...)
        fam = str(name).split()[0] 
        if current_fam and fam != current_fam:
            y_lines.append(i)
        current_fam = fam

    if y_lines:
        ax.hlines(y_lines, *ax.get_xlim(), colors='white', linewidth=3)

    # plt.title('Entwicklung des F1-Scores über steigende Konfidenz-Schwellenwerte', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Klassifikations-Schwellenwert', fontsize=12)
    plt.ylabel('', fontsize=12) 
    
    plt.tight_layout()

    # Speichern (mit Fallback, falls base_plot_folder nicht global definiert ist)
    heatmap_filename = 'f1_threshold_heatmap.png'
    try:
        save_path = os.path.join(base_plot_folder, heatmap_filename)
    except NameError:
        save_path = heatmap_filename
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" -> {heatmap_filename} generiert.")

# ---------------------------------------------------------
# DATEN-LADEN & ITERATION ÜBER RUNS
# ---------------------------------------------------------

def load_master_data():
    """
    Lädt Aggregated Results und Dataset Info EINMALIG, merged sie und bereitet alles vor.
    Gibt ein optimiertes DataFrame zurück, das an alle Funktionen übergeben wird.
    """
    print("\n--- MASTER DATA LOADER ---")
    if not os.path.exists(SUMMARIZED_FILE):
        print(f"Warnung: {SUMMARIZED_FILE} existiert noch nicht.")
        return None

    print(" -> Lade aggregierte Ergebnisse...")
    df = pd.read_excel(SUMMARIZED_FILE)
    df['video_id'] = df['video_id'].astype(str)

    if os.path.exists(INFO_DATEI):
        df_info = pd.read_excel(INFO_DATEI)
        df_info['video_id'] = df_info['video_id'].astype(str)

        # Nur Spalten aus df_info nehmen, die in df noch fehlen (außer dem Join-Key)
        new_cols = [c for c in df_info.columns
                    if c != 'video_id' and c not in df.columns]

        if new_cols:
            print(f" -> Merge Metadaten aus {INFO_DATEI}: {new_cols}")
            df = pd.merge(df, df_info[['video_id'] + new_cols],
                          on='video_id', how='left')
        else:
            print(f" -> Alle Metadaten bereits vorhanden, kein Merge nötig.")

        if 'y_true' not in df.columns and 'deepfake' in df.columns:
            df['y_true'] = df['deepfake'].map(
                {"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1}
            )
    else:
        print(f"Warnung: {INFO_DATEI} fehlt! Analysen werden unvollständig sein.")

    print(f"Daten geladen: {len(df)} Zeilen, {len(df.columns)} Spalten.")
    return df

def iterate_runs(df_master):
    """
    Generator-Funktion: Iteriert über Runs und liefert vorbereitete DataFrames zurück.
    Erspart das ständige "if col not in df..." und Renaming in jeder Funktion.
    Yields: (run_label, df_run)
    """
    for suffix in RUN_SUFFIXES:
        run_label = f"Run{suffix}"
        col_pred = f'y_pred{suffix}'
        
        if col_pred not in df_master.columns:
            continue
            
        # Relevante Spalten auswählen
        # Basis-Infos + Metadaten + Run-spezifische Spalten
        cols_base = ['video_id', 'base_model', 'y_true'] + [c for c in META_COLS if c in df_master.columns]
        
        # Run Spalten finden 
        cols_run = [c for c in df_master.columns if c.endswith(suffix)]
        
        # Subset erstellen
        df_run = df_master[cols_base + cols_run].copy()
        
        # Umbenennen (Suffix entfernen für einheitlichen Zugriff)
        rename_dict = {c: c.replace(suffix, '') for c in cols_run}
        # explizit sicherstellen, dass y_pred_1 -> y_pred wird
        rename_dict[col_pred] = 'y_pred' 
        
        df_run = df_run.rename(columns=rename_dict)
        
        # Minimal Cleaning
        df_run = df_run.dropna(subset=['y_true', 'y_pred'])
        
        yield run_label, df_run


# ---------------------------------------------------------
# ANALYSE FUNKTIONEN
# ---------------------------------------------------------

def generate_plots(df_master):
    print("\n=== Generiere Plots (Per Run) ===") 
    
    for run_label, df_run in iterate_runs(df_master):
        print(f"Plots für {run_label}...")
        run_folder = os.path.join(base_plot_folder, run_label)
        os.makedirs(run_folder, exist_ok=True)

        for model_name, group in df_run.groupby('base_model'):
            # Confusion Matrix
            cm = confusion_matrix(group['y_true'], group['y_pred'], labels=[0, 1])
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plot_label = get_plot_label(model_name)
            plt.title(f'CM: {plot_label}')
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, f'CM_{model_name}.png'))
            plt.close()

            # ROC Curve (nur wenn Wahrscheinlichkeiten vorhanden)
            if 'probability_fake' in group.columns:
                y_prob = pd.to_numeric(group['probability_fake'], errors='coerce').fillna(0)
                if y_prob.max() > 1.0: y_prob = y_prob / 100.0
                fpr, tpr, _ = roc_curve(group['y_true'], y_prob)
                roc_auc_val = auc(fpr, tpr)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC: {plot_label}')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(run_folder, f'ROC_{model_name}.png'))
                plt.close()

def run_global_baseline_roc_analysis(df_master):
    """
    Erstellt einen aggregierten ROC-Plot nur für die Baseline-Modelle über alle 3 Runs.
    """
    print("\n=== Global Baseline ROC Analysis (Aggregated) ===")
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guessing (AUC = 0.5)')
    
    # Ordner für aggregierte Plots sicherstellen
    agg_folder = os.path.join(base_plot_folder, 'Aggregated')
    os.makedirs(agg_folder, exist_ok=True)

    has_data = False
    
    # iterieren nur über die Basis-Modelle (Baselines)
    for tech_base_name, display_name in BASE_MODEL_DISPLAY_NAMES.items():
        # Filtere das Master-DF für dieses Modell
        model_group = df_master[df_master['base_model'] == tech_base_name]
        
        if model_group.empty:
            continue
            
        all_y_true = []
        all_y_prob = []
        
        # Sammle Wahrscheinlichkeiten und Labels aus allen 3 Durchläufen
        for suffix in RUN_SUFFIXES:
            prob_col = f'probability_fake{suffix}'
            if prob_col in model_group.columns:
                # Wir nehmen die Ground Truth (y_true) und die Konfidenz dieses Runs
                probs = pd.to_numeric(model_group[prob_col], errors='coerce').fillna(0)
                if probs.max() > 1.0: probs = probs / 100.0
                
                all_y_prob.extend(probs.tolist())
                all_y_true.extend(model_group['y_true'].tolist())
        
        if all_y_prob:
            has_data = True
            fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
            roc_auc_val = auc(fpr, tpr)
            
            # Farbe für die Baseline (Variante 0) holen
            color = get_model_color(tech_base_name)
            
            plt.plot(fpr, tpr, color=color, lw=3, 
                     label=f'{display_name} (AUC = {roc_auc_val:.2f})')

    if has_data:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
        plt.title('Baseline ROC', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(agg_folder, 'Global_Baseline_ROC_Comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" -> Aggregierter Baseline ROC-Plot gespeichert: {save_path}")
    else:
        print(" -> Keine Wahrscheinlichkeitsdaten für ROC-Plot gefunden.")


def run_family_variant_roc_comparison(df_master):
    """
    Erstellt Familien-Grafiken basierend auf den GLOBALEN Definitionen.
    """    
    print("\n=== Family Variant ROC Comparison (Aggregated) ===")
    
    family_folder = os.path.join(base_plot_folder, 'Family_Comparison')
    os.makedirs(family_folder, exist_ok=True)

    # 1. Familien dynamisch aus den Display Names extrahieren
    all_families = sorted(list(set(BASE_MODEL_DISPLAY_NAMES.values())))
    
    for fam in all_families:
        plt.figure(figsize=(8, 7))
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        
        has_plot_data = False
        
        # 2. Alle Modelle finden, die zu dieser Familie gehören
        model_variants_in_fam = [
            m for m in df_master['base_model'].unique() 
            if fam in get_plot_label(m)
        ]
        
        # Sortierung nach deiner globalen VARIANT_ORDER
        model_variants_in_fam.sort(key=lambda x: next((i for i, v in enumerate(VARIANT_ORDER) if v in get_plot_label(x)), 99))

        for tech_model_name in model_variants_in_fam:
            all_y_true = []
            all_y_prob = []
            
            model_rows = df_master[df_master['base_model'] == tech_model_name]
            
            for r_suffix in RUN_SUFFIXES:
                prob_col = f'probability_fake{r_suffix}'
                if prob_col in model_rows.columns:
                    # --- SCHRITT 1: FILTERN ---
                    # Wir nehmen nur Zeilen, die sowohl ein Label als auch eine Probability haben
                    temp_df = model_rows.dropna(subset=['y_true', prob_col])
                    
                    if not temp_df.empty:
                        probs = pd.to_numeric(temp_df[prob_col], errors='coerce').fillna(0)
                        if probs.max() > 1.0: probs = probs / 100.0
                        
                        all_y_prob.extend(probs.tolist())
                        all_y_true.extend(temp_df['y_true'].astype(int).tolist())
            
            # --- SCHRITT 2: VALIDIERUNG VOR DEM PLOTTEN ---
            if all_y_prob and len(np.unique(all_y_true)) > 1:
                has_plot_data = True
                color = get_model_color(tech_model_name)
                
                label = get_plot_label(tech_model_name)
                fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
                roc_auc_val = auc(fpr, tpr)
                
                # Wir plotten mit der individuellen Farbe
                plt.plot(fpr, tpr, color=color, lw=2.5, 
                         label=f'{label} (AUC = {roc_auc_val:.2f})')

        if has_plot_data:
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(f'ROC: {fam}', fontsize=13, fontweight='bold')
            plt.legend(loc="lower right", fontsize=9)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            
            plt.savefig(os.path.join(family_folder, f'ROC_Comparison_{fam}.png'), dpi=250)
            plt.close()
            print(f" -> Grafik erstellt für Familie: {fam}")
        else:
            plt.close()
            print(f" -> Keine validen Daten für Familie {fam} gefunden.")

def run_feature_importance_analysis(df_master):
    """
    Ersetzt Random Forest Feature Importance (nicht valide bei N=50) durch:
    - Normalized Stacked Bar Chart für kategorische Features
    - Swarmplot für numerische Features (z.B. video_length)
    Jeweils ein Facet-Grid über alle Baseline-Modelle pro Feature.
    """
    print("\n=== Feature Importance (Stacked Bar + Swarm, Per Run & Feature) ===")

    # Nur Baseline-Modelle (keine +I / +T Varianten)
    baseline_keys = [k for k in BASE_MODEL_DISPLAY_NAMES
                     if '_indicators' not in k and '_thinking' not in k]

    for run_label, df_run in iterate_runs(df_master):
        print(f" -> {run_label}...")

        save_dir = os.path.join(base_plot_folder, run_label, 'Feature_Importance')
        os.makedirs(save_dir, exist_ok=True)

        df_run = df_run[df_run['base_model'].isin(baseline_keys)].copy()
        df_run['correct'] = (df_run['y_pred'] == df_run['y_true']).astype(int)
        df_run['Outcome'] = df_run['correct'].map({1: 'Korrekt', 0: 'Fehler'})
        df_run['Model'] = df_run['base_model'].apply(get_plot_label)

        if df_run.empty:
            print(f"    Keine Baseline-Daten für {run_label}.")
            continue

        models_ordered = sorted(df_run['Model'].unique())
        n_models = len(models_ordered)

        for feat in META_COLS:
            if feat not in df_run.columns:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df_run[feat])

            if is_numeric:
                # ── SWARMPLOT (Strip + Box) ──────────────────────────────────
                fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)
                if n_models == 1:
                    axes = [axes]

                for ax, model in zip(axes, models_ordered):
                    sub = df_run[df_run['Model'] == model].dropna(subset=[feat])
                    palette = {'Korrekt': '#2ca02c', 'Fehler': '#d62728'}

                    # ---> NEU: order=['Korrekt', 'Fehler'] erzwingt einheitliche X-Achse <---
                    sns.boxplot(data=sub, x='Outcome', y=feat, ax=ax,
                                palette=palette, width=0.4, order=['Korrekt', 'Fehler'],
                                showfliers=False, linewidth=1.2)
                    sns.stripplot(data=sub, x='Outcome', y=feat, ax=ax,
                                  palette=palette, size=4, alpha=0.7, jitter=True,
                                  order=['Korrekt', 'Fehler'])

                    ax.set_title(model, fontsize=10, fontweight='bold')
                    ax.set_xlabel('')
                    if ax == axes[0]:
                        ax.set_ylabel(feat, fontsize=10)
                    else:
                        ax.set_ylabel('')

                fig.suptitle(f'{run_label} – Verteilung „{feat}" nach Klassifikationsergebnis',
                             fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'Swarm_{feat}.png'), dpi=300, bbox_inches='tight')
                plt.close()

            else:
                # ── NORMALIZED STACKED BAR CHART ────────────────────────────
                fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 5), sharey=True)
                if n_models == 1:
                    axes = [axes]

                for ax, model in zip(axes, models_ordered):
                    sub = df_run[df_run['Model'] == model].dropna(subset=[feat])

                    # Normalisierte Anteile pro Gruppe berechnen
                    counts = (sub.groupby([feat, 'Outcome'], observed=False)
                                .size()
                                .unstack(fill_value=0))

                    # Sicherstellen, dass beide Spalten vorhanden sind
                    for col in ['Korrekt', 'Fehler']:
                        if col not in counts.columns:
                            counts[col] = 0

                    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100
                    counts_pct = counts_pct[['Korrekt', 'Fehler']]

                    counts_pct.plot(
                        kind='bar', stacked=True, ax=ax,
                        color=['#2ca02c', '#d62728'],
                        edgecolor='white', linewidth=0.5,
                        legend=(ax == axes[-1])
                    )

                    # N pro Gruppe als Annotation
                    for i, (_, row) in enumerate(counts.iterrows()):
                        n_total = int(row.sum())
                        ax.text(i, 101, f'n={n_total}', ha='center', va='bottom',
                                fontsize=7, color='gray')

                    ax.set_title(model, fontsize=10, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
                    ax.set_ylim(0, 115)
                    if ax == axes[0]:
                        ax.set_ylabel('Anteil (%)', fontsize=10)
                    else:
                        ax.set_ylabel('')
                    ax.axhline(50, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

                    if ax == axes[-1]:
                        ax.legend(loc='upper right', fontsize=8, title='Ergebnis')

                fig.suptitle(f'{run_label} – Klassifikationsergebnis nach „{feat}"',
                             fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'StackedBar_{feat}.png'), dpi=300, bbox_inches='tight')
                plt.close()


def run_feature_analysis(df_master):
    print("\n=== Feature Analysis (Per Run) ===")
    all_results = []
    
    features_to_analyze = META_COLS 
    
    for run_label, df_run in iterate_runs(df_master):
        for model_name, model_df in df_run.groupby('base_model'):
            for feat in features_to_analyze:
                if feat not in model_df.columns: continue
                
                for group_name, grp_data in model_df.groupby(feat, observed=False):
                    if len(grp_data) < 5: continue
                    m = calculate_metrics(grp_data['y_true'], grp_data['y_pred'])
                    m.update({'Run': run_label, 'Model': model_name, 'Display_Name': BASE_MODEL_DISPLAY_NAMES.get(model_name, model_name), 'Feature': feat, 'Group': str(group_name), 'Videos': len(grp_data)})
                    if 'ROC AUC' in m: del m['ROC AUC']
                    all_results.append(m)

    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res = df_res.sort_values(['Run', 'Display_Name', 'Feature'])
        df_res.to_excel(os.path.join(RESULTS_FOLDER, 'feature_analysis.xlsx'), index=False)
        
        # Heatmaps
        for run_label in df_res['Run'].unique():
            run_folder = os.path.join(base_plot_folder, run_label)
            os.makedirs(run_folder, exist_ok=True)
            run_data = df_res[df_res['Run'] == run_label]
            
            for feat in run_data['Feature'].unique():
                subset = run_data[run_data['Feature'] == feat]
                try:
                    # Pivot für Datenstruktur
                    pivot = subset.pivot(index='Display_Name', columns='Group', values='F1-Score (%)')
                    # Labels schön machen
                    pivot.index = [get_plot_label(model) for model in pivot.index]
                    
                    # ENTSCHEIDUNG: Heatmap oder Barplot?
                    n_groups = len(pivot.columns)
                    
                    if n_groups <= 3: 
                        # === STANDARD: GROUPED BAR CHART ===
                        df_plot = subset.copy()

                        # SCHRITT 0: Die fehlende Spalte erstellen!
                        df_plot['Model_Short'] = df_plot['Model'].apply(get_plot_label)

                        # 1. Jetzt finden wir die einzigartigen Modelle
                        unique_models = df_plot['Model_Short'].unique()

                        # 2. Dictionary erstellen (nutzt jetzt die korrekten Keys aus Model_Short)
                        color_dict = {m: get_model_color(m) for m in unique_models}

                        # 3. Den Plot erstellen
                        plt.figure(figsize=(12, 6))
                        sns.barplot(
                            data=df_plot, 
                            x='Group', 
                            y='F1-Score (%)', 
                            hue='Model_Short', 
                            palette=color_dict,
                            hue_order=sorted(unique_models), # Sortiert die Legende alphabetisch (Gemini, GPT...)
                            edgecolor='black'
                        )

                        
                        plt.title(f'{run_label}: {feat} Performance by Model')
                        plt.ylabel('F1-Score (%)')
                        plt.xlabel('')
                        plt.ylim(0, 100)
                        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.savefig(os.path.join(run_folder, f'Feature_Analysis_{feat}_Bar.png'))
                        plt.close()
                        
                    else:
                        # === STANDARD: HEATMAP ===
                        plt.figure(figsize=(10, len(pivot)*0.5 + 2))
                        sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt='.1f', vmin=0, vmax=100)
                        plt.title(f'{run_label}: {feat} Performance')
                        plt.tight_layout()
                        plt.savefig(os.path.join(run_folder, f'Feature_Analysis_{feat}_Heatmap.png'))
                        plt.close()
                except Exception as e:
                    print(f" [WARN] Feature-Plot für '{feat}' fehlgeschlagen: {e}")

def run_fairness_analysis(df_master):
    print("\n=== Fairness Analysis (Per Run) ===")
    all_results = []
    
    for run_label, df_run in iterate_runs(df_master):
        print(f" -> {run_label}...")
        for model_name, model_df in df_run.groupby('base_model'):
            global_acc = accuracy_score(model_df['y_true'], model_df['y_pred'])
            
            for feat in META_COLS:
                if feat not in model_df.columns: continue
                
                for name in model_df[feat].dropna().unique():
                    g = model_df[model_df[feat] == name]
                    rest = model_df[model_df[feat] != name]
                    if len(g) < 5 or len(rest) < 5: continue
                    
                    acc_group = accuracy_score(g['y_true'], g['y_pred'])
                    tbl = [[(g['y_true']==g['y_pred']).sum(), (g['y_true']!=g['y_pred']).sum()],
                           [(rest['y_true']==rest['y_pred']).sum(), (rest['y_true']!=rest['y_pred']).sum()]]
                    
                    if (len(g) < 30) or any(v < 5 for r in tbl for v in r):
                        _, p = fisher_exact(tbl); test = "Fisher"
                    else:
                        _, p, _, _ = chi2_contingency(tbl); test = "Chi2"
                    
                    all_results.append({
                        'Run': run_label, 'Model': model_name, 'Display_Name': BASE_MODEL_DISPLAY_NAMES.get(model_name, model_name), 'Feature': feat, 'Group': str(name),
                        'Accuracy (%)': round(acc_group*100,1), 'Diff to Global': round((acc_group-global_acc)*100,2),
                        'P-Value': round(p,4), 'Test': test
                    })

    if all_results:
        df_res = pd.DataFrame(all_results).sort_values(['Run', 'Display_Name', 'P-Value'])
        df_res.to_excel(os.path.join(RESULTS_FOLDER, 'fairness_analysis.xlsx'), index=False)
        
        # Heatmaps (Bias)
        for run_label in df_res['Run'].unique():
            run_folder = os.path.join(base_plot_folder, run_label)
            run_data = df_res[df_res['Run'] == run_label]
            
            for feat in run_data['Feature'].unique():
                try:
                    subset = run_data[run_data['Feature'] == feat]
                    pivot = subset.pivot(index='Display_Name', columns='Group', values='Diff to Global')
                    # Konvertiere Index-Labels zu kurzen Plot-Namen
                    pivot.index = [get_plot_label(model) for model in pivot.index]
                    plt.figure(figsize=(10, len(pivot)*0.5 + 2))
                    sns.heatmap(pivot, annot=True, cmap='RdBu', center=0, fmt='.1f')
                    plt.title(f'{run_label}: Bias by {feat}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_folder, f'Fairness_Bias_{feat}.png'))
                    plt.close()
                except Exception as e:
                    print(f" [WARN] Fairness-Plot für '{feat}' fehlgeschlagen: {e}")





def run_intra_model_consistency_check(df_master):
    """
    Prüft die Konsistenz der Begründungen über alle definierten Runs hinweg.
    Bildet automatisch alle paarweisen Vergleiche (z.B. Run 1 vs 2, 2 vs 3, 1 vs 3).
    """
    print("\n=== Intra-Model Consistency Check (Stability) ===")
    
    # 1. Dynamische Spalten basierend auf globalen Suffixen generieren
    justification_cols = [f'justification{suffix}' for suffix in RUN_SUFFIXES]
    
    if not all(c in df_master.columns for c in justification_cols):
        fehlend = [c for c in justification_cols if c not in df_master.columns]
        print(f"Warnung: Folgende Begründungs-Spalten fehlen: {fehlend}")
        return

    df_cons = df_master.dropna(subset=justification_cols + ['base_model', 'video_id']).copy()
    if df_cons.empty:
        print("Keine vollständigen Daten für Konsistenz-Check vorhanden.")
        return

    results = []
    
    # 2. Berechnung pro Modell
    for model_name, group in df_cons.groupby('base_model'):
        
        # A) Alle Texte extrahieren und encoden
        embeddings = []
        for col in justification_cols:
            texts = group[col].astype(str).tolist()
            # show_progress_bar=False hält den Konsolen-Output sauberer
            emb = get_sbert().encode(texts, convert_to_tensor=True, show_progress_bar=False)
            embeddings.append(emb)
        
        # B) Paarweise Ähnlichkeiten berechnen
        n_runs = len(embeddings)
        if n_runs < 2:
            print("Warnung: Es werden mindestens 2 Runs für einen Vergleich benötigt.")
            return
            
        pairwise_sims = []
        # Bildet alle möglichen Paare (bei 3 Runs: 0-1, 0-2, 1-2)
        for i, j in itertools.combinations(range(n_runs), 2):
            sim = util.cos_sim(embeddings[i], embeddings[j]).diag().cpu().numpy()
            pairwise_sims.append(sim)
        
        # C) Durchschnittliche Konsistenz über alle Paare berechnen
        # pairwise_sims ist eine Liste von Arrays. mean(axis=0) mittelt pro Video.
        avg_consistency = np.mean(pairwise_sims, axis=0)
        
        for vid, score in zip(group['video_id'], avg_consistency):
            results.append({
                'base_model': model_name,
                'video_id': vid,
                'Consistency_Score': score
            })

    if not results: return

    # Excel Speichern
    df_res = pd.DataFrame(results)
    df_res.to_excel(os.path.join(RESULTS_FOLDER, 'consistency_analysis_intra_model_raw.xlsx'), index=False)
    
    # Aggregation berechnen
    summary = df_res.groupby('base_model')['Consistency_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    summary.reset_index().to_excel(os.path.join(RESULTS_FOLDER, 'consistency_summary_per_model.xlsx'), index=False)
    print("\n--- Summary ---")
    print(summary)

    # Plot-Labels hinzufügen
    df_res['plot_label'] = df_res['base_model'].map(get_plot_label)

    # --- PLOT 1: Boxplot (Verteilung) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_res, x='plot_label', y='Consistency_Score', hue='Consistency_Score', palette="Blues", legend=False)
    plt.title('Reasoning Stability Distribution (SBERT)')
    plt.ylabel('Consistency Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_folder, 'Model_Consistency_Boxplot.png'))
    plt.close()

    # --- PLOT 2: Barplot mit Standardabweichung ---
    plt.figure(figsize=(10, 6))
    x_pos = range(len(summary))
    summary.index = summary.index.map(get_plot_label)
    model_colors = [get_model_color(model) for model in summary.index]
    
    plt.bar(x_pos, summary['mean'], yerr=summary['std'], align='center', alpha=0.8, ecolor='black', capsize=10, color=model_colors)
    plt.xticks(x_pos, summary.index, rotation=45)
    plt.ylabel('Mean Semantic Consistency (0-1)')
    plt.title('Model Reasoning Stability (Mean ± Std Dev)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_folder, 'Model_Consistency_BarChart_with_Std.png'))
    plt.close()
    
    print("Consistency Plots (Boxplot & BarChart) gespeichert.")




def analyze_region_coverage(df):
    """
    Prüft für jede Anker-Region aus KEYWORD_ANCHORS, wie oft sie in den
    Justifications vorkommt, und zeigt welche häufigen Keywords noch nicht
    abgedeckt sind (Kandidaten zum Ergänzen der Anker).
    """
    justification_cols = [f'justification{s}' for s in RUN_SUFFIXES]
    justification_cols = [c for c in justification_cols if c in df.columns]
    if not justification_cols:
        print("Keine Justification-Spalten gefunden.")
        return

    all_texts = pd.concat(
        [df[c].dropna().astype(str) for c in justification_cols]
    ).str.lower()
    total = len(all_texts)

    # Aus KEYWORD_ANCHORS ein Regex-Pattern pro Region bauen
    # Anker-Text z.B. "skin, cheek, forehead" → r'skin|cheek|forehead'
    region_patterns = {
        region: '|'.join(re.escape(t.strip()) for t in anchor_text.split(','))
        for region, anchor_text in KEYWORD_ANCHORS.items()
    }

    print("\n=== Region Coverage in Justifications ===")
    print(f"Gesamt-Justifications: {total}\n")

    results = []
    for region, pattern in region_patterns.items():
        l2, l1 = KEYWORD_HIERARCHY.get(region, ('?', '?'))
        count = all_texts.str.contains(pattern, regex=True).sum()
        pct   = round(count / total * 100, 1)
        results.append({'Region (L3)': region, 'L2': l2, 'L1': l1,
                        'Treffer': count, 'Anteil (%)': pct})
        print(f"  [{l1:6} › {l2:12}] {region:<18} {count:>5}x  ({pct}%)")

    df_res = pd.DataFrame(results).sort_values('Anteil (%)', ascending=False)

    # Nicht abgedeckte häufige Keywords finden
    combined_pattern = '|'.join(f'(?:{p})' for p in region_patterns.values())
    cv = CountVectorizer(ngram_range=(1, 2), stop_words=DOMAIN_STOPS, min_df=5, max_features=500)
    try:
        mat = cv.fit_transform(all_texts)
        kw_counts = pd.DataFrame({
            'keyword': cv.get_feature_names_out(),
            'count':   mat.sum(axis=0).A1
        }).sort_values('count', ascending=False)

        kw_counts['covered'] = kw_counts['keyword'].str.contains(combined_pattern, regex=True)
        uncovered = kw_counts[~kw_counts['covered']].head(40)
        uncovered['Anteil (%)'] = (uncovered['count'] / total * 100).round(1)

        print("\n--- Top-40 nicht abgedeckte häufige Keywords ---")
        print("(Kandidaten für neue Anker in KEYWORD_ANCHORS)\n")
        for _, row in uncovered.iterrows():
            print(f"  {row['keyword']:<35} {int(row['count']):>5}x  ({row['Anteil (%)']}%)")

        save_path = os.path.join(RESULTS_FOLDER, 'region_coverage.xlsx')
        with pd.ExcelWriter(save_path) as writer:
            df_res.to_excel(writer, sheet_name='Coverage', index=False)
            uncovered.to_excel(writer, sheet_name='Uncovered_Keywords', index=False)
        print(f"\n -> Ergebnisse gespeichert: {save_path}")
    except ValueError:
        save_path = os.path.join(RESULTS_FOLDER, 'region_coverage.xlsx')
        df_res.to_excel(save_path, index=False)

    return df_res


def export_keyword_inventory(df):
    """Extrahiert alle Keywords (1-3 Wörter) über alle Runs hinweg und speichert sie hartcodiert als CSV."""
    
    # Erzeuge die genauen Spaltennamen basierend auf der globalen RUN_SUFFIXES Liste
    justification_cols = [f'justification{suffix}' for suffix in RUN_SUFFIXES]
    
    # Zur Sicherheit: Nur die Spalten behalten, die auch wirklich im DataFrame sind
    justification_cols = [col for col in justification_cols if col in df.columns]
    if not justification_cols:
        print("Fehler: Keine 'justification_X' Spalten im DataFrame gefunden.")
        return

    # Texte aus all diesen Spalten in einer flachen Liste sammeln
    texts = []
    for col in justification_cols:
        texts.extend(df[col].dropna().astype(str).tolist())
        
    if not texts: 
        print("Keine Texte für Inventar gefunden.")
        return

    # Extraktion von Unigrammen bis Trigrammen
    cv = CountVectorizer(ngram_range=(1, 3), stop_words=DOMAIN_STOPS, min_df=2)
    counts = cv.fit_transform(texts)
    
    inventory = pd.DataFrame({
        'keyword': cv.get_feature_names_out(),
        'count': counts.sum(axis=0).A1
    }).sort_values(by='count', ascending=False)
    
    save_path = os.path.join(RESULTS_FOLDER, 'keyword_inventory.xlsx')
    inventory.to_excel(save_path, index=False)
    print(f"Inventar gespeichert: keyword_inventory.xlsx (Basierend auf {len(justification_cols)} Runs)")
    return inventory


def cluster_keywords(inventory_df, output_filename='keyword_inventory_clustered.xlsx'):
    """
    Jedes Keyword wird via SBERT einem MARE-Begriff (Level 3) zugeordnet.
    Level 2 und Level 1 werden statisch aus der Hierarchie abgeleitet:

    Frame
    ├── Face       → MARE-Regionen (Skin, Nose, Mouth, ...)
    ├── Body       → Head_Neck, Torso, Hands
    └── Background → Lighting, Scene, Temporal

    Audio
    ├── Voice      → Stimme, Sprache, Klang
    └── Lip_Sync   → Lippensynchronität
    """
    print(" -> Starte SBERT-Clustering")

    model = get_sbert()

    # Globale KEYWORD_ANCHORS und KEYWORD_HIERARCHY verwenden
    df = inventory_df.copy()

    region_names = list(KEYWORD_ANCHORS.keys())
    anchor_texts = list(KEYWORD_ANCHORS.values())

    # Embeddings & Cosine Similarity
    region_embeddings  = model.encode(anchor_texts, convert_to_tensor=True)
    keyword_list       = df['keyword'].tolist()
    keyword_embeddings = model.encode(keyword_list, convert_to_tensor=True)

    cosine_scores = util.cos_sim(keyword_embeddings, region_embeddings)
    best_indices  = torch.argmax(cosine_scores, dim=1).tolist()
    confidences   = torch.max(cosine_scores, dim=1).values.tolist()

    # Level 3 zuweisen
    df['level_3']    = [region_names[i] for i in best_indices]
    df['confidence'] = confidences

    # Zu unsichere Zuweisungen → 'General Artifacts'
    df.loc[df['confidence'] < 0.4, 'level_3'] = 'General_Artifacts'

    # Level 2 und Level 1 aus statischer Hierarchie ableiten
    # Fallback: Background/Frame für unbekannte Werte
    df['level_2'] = df['level_3'].map(
        lambda r: KEYWORD_HIERARCHY.get(r, ('Background', 'Frame'))[0]
    )
    df['level_1'] = df['level_3'].map(
        lambda r: KEYWORD_HIERARCHY.get(r, ('Background', 'Frame'))[1]
    )

    # Debug: Grenzfälle mit allen Scores speichern
    scores_df = pd.DataFrame(
        cosine_scores.cpu().numpy(),
        columns=region_names,
        index=keyword_list
    )
    scores_df.index.name = 'keyword'
    scores_df['winner']     = df['level_3'].values
    scores_df['confidence'] = confidences
    ambiguous = scores_df[scores_df['confidence'] < 0.5].reset_index()
    if not ambiguous.empty:
        debug_path = os.path.join(RESULTS_FOLDER, 'debug_ambiguous_keywords.xlsx')
        ambiguous.to_excel(debug_path, index=False)
        print(f" -> {len(ambiguous)} mehrdeutige Keywords gespeichert: {debug_path}")

    # Speichern: sortiert nach Hierarchie
    df = df.sort_values(
        by=['level_1', 'level_2', 'level_3', 'confidence'],
        ascending=[True, True, True, False]
    )
    save_path = os.path.join(RESULTS_FOLDER, output_filename)
    df.to_excel(save_path, index=False)

    print(f"Keyword-Clustering abgeschlossen. Datei gespeichert: {save_path}")
    return df

# --- MODUL 2: KEYWORD-PLOTS MIT HIERARCHIE & TP/FP-FARBEN ---

def _build_shared_vectorizer(groups: dict, max_features=150):
    """
    Fittet einen gemeinsamen CountVectorizer, Vocabulary-Selektion auf TP+FP.
    TP und FP sind analytisch am relevantesten — ihre Keywords dominieren die Auswahl.
    FN und TN werden danach mit derselben Vocabulary transformiert.
    """
    # Vocabulary auf TP+FP fitten (wichtigste Gruppen für die Analyse)
    fit_texts = groups.get('TP', []) + groups.get('FP', [])
    if not fit_texts:
        # Fallback: alle Texte verwenden
        fit_texts = [t for texts in groups.values() for t in texts]
    if not fit_texts:
        return None, {}
    cv = CountVectorizer(ngram_range=(1, 3), stop_words=DOMAIN_STOPS, max_features=max_features)
    try:
        cv.fit(fit_texts)
    except ValueError:
        return None, {}
    matrices = {}
    for name, texts in groups.items():
        if texts:
            matrices[name] = cv.transform(texts)
        else:
            matrices[name] = None
    return cv, matrices


def _counts_from_matrix(mat, cv, n_texts, normalize=False):
    """Summiert eine sparse Matrix zu einem keyword→count DataFrame."""
    if mat is None or n_texts == 0:
        return pd.DataFrame(columns=['keyword', 'count'])
    counts = mat.sum(axis=0).A1
    if normalize:
        counts = counts / n_texts
    return pd.DataFrame({'keyword': cv.get_feature_names_out(), 'count': counts})


def _build_lookup(clustered_df):
    """Baut eine keyword→level_3/level_2/level_1 Lookup-Tabelle aus clustered_df."""
    lookup = clustered_df[['keyword', 'level_3']].drop_duplicates('keyword').copy()
    lookup['level_2'] = lookup['level_3'].map(lambda r: KEYWORD_HIERARCHY.get(r, ('Background', 'Frame'))[0])
    lookup['level_1'] = lookup['level_3'].map(lambda r: KEYWORD_HIERARCHY.get(r, ('Background', 'Frame'))[1])
    return lookup


def _extract_texts(model_df):
    """Extrahiert TP/FP/FN/TN-Justification-Listen aus einem Modell-DataFrame."""
    masks = {
        'TP': (model_df['y_true'] == 1) & (model_df['y_pred'] == 1),
        'FP': (model_df['y_true'] == 0) & (model_df['y_pred'] == 1),
        'FN': (model_df['y_true'] == 1) & (model_df['y_pred'] == 0),
        'TN': (model_df['y_true'] == 0) & (model_df['y_pred'] == 0),
    }
    return {k: model_df[m]['justification'].dropna().tolist() for k, m in masks.items()}


def plot_treemap_keywords(df_kw, model_name, run_label, save_folder):
    """
    Interaktive Treemap (HTML) + statisches PNG.
    Fläche = Gesamthäufigkeit, Farbe = TP-Anteil (grün = TP-dominiert, rot = FP-dominiert).
    """

    df = df_kw[df_kw['keyword'].str.split().str.len() >= 2].copy()
    count_cols = [c for c in ['TP', 'FP', 'FN', 'TN'] if c in df.columns]
    df['total'] = df[count_cols].sum(axis=1)
    df = df[df['total'] > 0]
    if df.empty:
        return

    # tp_ratio: TP-Anteil an TP+FP (Präzisions-Proxy; TN/FN bewusst ausgeschlossen)
    tp_fp = df.get('TP', 0) + df.get('FP', 0)
    df['tp_ratio'] = df['TP'] / tp_fp.replace(0, float('nan'))
    plot_label = get_plot_label(model_name)
    safe_name  = re.sub(r'[^\w\-_]', '_', model_name)

    fig = px.treemap(
        df,
        path=[px.Constant('All'), 'level_1', 'level_2', 'level_3', 'keyword'],
        values='total',
        color='tp_ratio',
        color_continuous_scale=['#de2d26', '#f7f7f7', '#2ca25f'],
        color_continuous_midpoint=0.5,
        range_color=[0, 1],
        title=f"{plot_label} | {run_label}",
        hover_data={'TP': True, 'FP': True, 'tp_ratio': ':.2f'},
    )
    fig.update_traces(textinfo='label+value')
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_colorbar=dict(title='TP-Anteil', tickvals=[0, 0.5, 1], ticktext=['FP', '50/50', 'TP'])
    )

    html_path = os.path.join(save_folder, f"{safe_name}_treemap_{run_label}.html")
    fig.write_html(html_path)

    try:
        png_path = os.path.join(save_folder, f"{safe_name}_treemap_{run_label}.png")
        fig.write_image(png_path, width=1400, height=900)
    except Exception:
        pass  # kaleido nicht installiert — nur HTML gespeichert


def plot_grouped_rates(df_kw, group_col, model_name, run_label, save_folder, level_name, top_n=20):
    """
    Grouped Bar Chart mit unabhängigen normalisierten Raten (Erwähnungen pro Text).
    Jede Gruppe (TP/FP/FN) hat einen eigenen Balken — kein gemeinsamer Nenner,
    da TP/FP/FN-Pools unterschiedlich groß sind. Scientifically correct.

    Interpretation: Ein langer FP-Balken bei 'Face' bedeutet, dass das Modell
    Face-Keywords in FP-Texten häufiger erwähnt als in TP-Texten (pro Text).
    """
    cols = [c for c in ['TP', 'FP', 'FN', 'TN'] if c in df_kw.columns]
    df = df_kw.groupby(group_col)[cols].sum().reset_index()
    df['total'] = df[cols].sum(axis=1)
    df = df[df['total'] > 0].nlargest(top_n, 'total').sort_values('total')

    if df.empty:
        return

    plot_label = get_plot_label(model_name)
    safe_name  = re.sub(r'[^\w\-_]', '_', model_name)
    labels     = df[group_col].tolist()
    n_groups   = len(cols)
    bar_height = 0.25
    palette    = {'TP': '#2ca25f', 'FP': '#de2d26', 'FN': '#f59b00', 'TN': '#6baed6'}

    _, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.7 + 1)))

    global_max = df[cols].max().max()
    y = range(len(labels))
    for i, c in enumerate(cols):
        offsets = [pos + (i - n_groups / 2 + 0.5) * bar_height for pos in y]
        vals = df[c].values
        bars = ax.barh(offsets, vals, height=bar_height, color=palette[c], label=c)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(val + global_max * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}', va='center', fontsize=7, color=palette[c])

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Erwähnungsrate (Nennungen pro Text, normalisiert)', fontsize=10)
    ax.set_title(f"{plot_label} | {run_label} | {level_name}", fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    filename = f"{safe_name}_grouped_{level_name}_{run_label}.png"
    plt.savefig(os.path.join(save_folder, filename), dpi=200)
    plt.close()


def plot_diverging_bars(df_kw, group_col, model_name, run_label, save_folder, level_name, top_n=20):
    """
    Diverging Bar Chart: TP-Balken nach rechts (grün), FP-Balken nach links (rot).
    Sofort sichtbar welche Keywords/Gruppen FP- oder TP-charakteristisch sind.
    """
    df = df_kw.groupby(group_col)[['TP', 'FP']].sum().reset_index()
    df['total'] = df['TP'] + df['FP']
    df = df[df['total'] > 0].nlargest(top_n, 'total').sort_values('total')

    if df.empty:
        return

    plot_label = get_plot_label(model_name)
    safe_name  = re.sub(r'[^\w\-_]', '_', model_name)
    labels     = df[group_col].tolist()
    max_val    = df[['TP', 'FP']].max().max()

    _, ax = plt.subplots(figsize=(11, max(4, len(df) * 0.55 + 1)))

    ax.barh(labels, df['TP'].values,  color='#2ca25f', label='TP', zorder=2)
    ax.barh(labels, -df['FP'].values, color='#de2d26', label='FP', zorder=2)
    ax.axvline(0, color='black', linewidth=1, zorder=3)

    offset = max_val * 0.015
    for idx, row in enumerate(df.itertuples()):
        if row.TP > 0:
            ax.text(row.TP + offset, idx, f'{row.TP:.2f}',
                    va='center', fontsize=8, color='#2ca25f')
        if row.FP > 0:
            ax.text(-row.FP - offset, idx, f'{row.FP:.2f}',
                    va='center', ha='right', fontsize=8, color='#de2d26')

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.2f}'))
    ax.set_xlabel('← FP  |  TP →', fontsize=10)
    ax.set_title(f"{plot_label} | {run_label} | {level_name}", fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=1)
    plt.tight_layout()

    filename = f"{safe_name}_diverging_{level_name}_{run_label}.png"
    plt.savefig(os.path.join(save_folder, filename), dpi=200)
    plt.close()


def plot_keywords_hierarchical(groups, model_name, run_label, save_folder, clustered_df):
    """
    Erstellt Keyword-Plots auf allen Hierarchie-Ebenen mit TP/FP/FN/TN.
    groups: Dict {'TP': [...], 'FP': [...], 'FN': [...], 'TN': [...]}
    Rates sind normalisiert (Erwähnungen pro Text) für Vergleichbarkeit.
    """
    cv, matrices = _build_shared_vectorizer(groups)
    if cv is None:
        return

    df_kw = pd.DataFrame({'keyword': cv.get_feature_names_out()})
    for name, texts in groups.items():
        rates = _counts_from_matrix(matrices[name], cv, len(texts), normalize=True)
        df_kw[name] = df_kw['keyword'].map(rates.set_index('keyword')['count']).fillna(0) if not rates.empty else 0.0
    df_kw['total'] = df_kw[list(groups.keys())].sum(axis=1)

    # Hierarchie-Labels anhängen — level_2/level_1 kommen aus _build_lookup (immer aktuell)
    df_kw = pd.merge(df_kw, _build_lookup(clustered_df), on='keyword', how='inner')
    if df_kw.empty:
        return

    # --- Treemap: gesamte Hierarchie auf einen Blick (TP-Anteil als Farbe) ---
    # plot_treemap_keywords(df_kw, model_name, run_label, save_folder)

    # --- Überblick L1/L2: Stacked % (TP/FP/FN) + Diverging (TP vs FP) ---
    plot_grouped_rates(df_kw, 'level_1', model_name, run_label, save_folder, 'L1_Modality',  top_n=5)
    plot_grouped_rates(df_kw, 'level_2', model_name, run_label, save_folder, 'L2_Regions',   top_n=10)
    plot_diverging_bars(df_kw,   'level_2', model_name, run_label, save_folder, 'L2_Regions',   top_n=10)

    # --- L3 und L4 je L2-Kategorie (Face / Body / Background / Audio) ---
    for l2_name, df_l2 in df_kw.groupby('level_2'):
        if df_l2.empty:
            continue
        safe_l2 = l2_name.replace('/', '_')
        # Stacked % zeigt TP/FP/FN-Komposition je Subregion/Keyword
        plot_grouped_rates(df_l2, 'level_3', model_name, run_label, save_folder,
                              f'L3_Subregion_{safe_l2}', top_n=15)
        plot_grouped_rates(df_l2, 'keyword',  model_name, run_label, save_folder,
                              f'L4_Keywords_{safe_l2}',  top_n=20)
        # Diverging zeigt TP vs. FP Stärke (ohne FN) für schnellen Vergleich
        plot_diverging_bars(df_l2, 'level_3', model_name, run_label, save_folder,
                            f'L3_Subregion_{safe_l2}', top_n=15)
        plot_diverging_bars(df_l2, 'keyword',  model_name, run_label, save_folder,
                            f'L4_Keywords_{safe_l2}',  top_n=20)

    



def plot_model_comparison_heatmap(model_texts, clustered_df, save_folder, metric='FP', level='level_2'):
    """
    Heatmap: Zeile = Region (L2 oder L3), Spalte = Modell.
    Zellwert = normalisierte Erwähnungsrate für die gewählte Metrik (FP/TP/FN/TN).
    Erlaubt direkten Vergleich: Welches Modell hat bei welcher Region die höchste FP-Rate?

    model_texts: {model_name: {'TP': [...], 'FP': [...], 'FN': [...], 'TN': [...]}}
    """
    lookup = _build_lookup(clustered_df)
    records = []

    for model_name, texts in model_texts.items():
        groups = {k: v for k, v in texts.items()}
        cv, matrices = _build_shared_vectorizer(groups)
        if cv is None:
            continue

        n = len(texts.get(metric, []))
        if n == 0:
            continue

        df_counts = _counts_from_matrix(matrices.get(metric), cv, n, normalize=True)
        if df_counts.empty:
            continue

        df_counts = pd.merge(df_counts, lookup, on='keyword', how='inner')
        if df_counts.empty:
            continue

        for region, grp in df_counts.groupby(level):
            records.append({
                'Model':  get_plot_label(model_name),
                'Region': region,
                'Rate':   grp['count'].mean()  # Mittelwert: unabhängig von Keyword-Anzahl je Region
            })

    if not records:
        print(f"Keine Daten für Modellvergleichs-Heatmap ({metric}).")
        return

    df_heat = pd.DataFrame(records).pivot(index='Region', columns='Model', values='Rate').fillna(0)

    # Sortierung: Regionen nach Gesamt-Rate absteigend
    df_heat = df_heat.loc[df_heat.sum(axis=1).sort_values(ascending=False).index]

    fig_h = max(5, len(df_heat) * 0.5 + 1)
    fig_w = max(10, len(df_heat.columns) * 1.2 + 2)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        df_heat, annot=True, fmt='.3f', cmap='YlOrRd',
        linewidths=0.4, cbar_kws={'label': f'{metric}-Rate (Nennungen pro Text)'}
    )
    plt.title(f'Modellvergleich: {metric}-Rate je {level} (aggregiert über alle Runs)', fontsize=12)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()

    fname = os.path.join(save_folder, f'model_comparison_heatmap_{metric}_{level}.png')
    plt.savefig(fname, dpi=250)
    plt.close()
    print(f" -> Modellvergleichs-Heatmap gespeichert: {fname}")


def plot_chapter_overview_heatmap(model_texts, clustered_df, save_folder):
    """
    Übersichts-Heatmap für den Kapitelanfang:
    3 Subplots nebeneinander — TP-Rate | FP-Rate | FN-Rate
    Zeilen = Modelle (Display-Namen), Spalten = L2-Regionen
    Farbe = normalisierte Erwähnungsrate (Nennungen pro Text)

    Ermöglicht auf einen Blick: Wo erkennt jedes Modell korrekt (TP),
    wo produziert es Fehlalarme (FP), wo übersieht es Deepfakes (FN)?
    """
    lookup = _build_lookup(clustered_df)

    L2_ORDER = ['Face', 'Background', 'Body', 'Audio']
    metrics   = ['TP', 'FP', 'FN']
    titles    = ['TP-Rate\n(korrekte Deepfake-Erkennung)',
                 'FP-Rate\n(Fehlalarme bei echten Videos)',
                 'FN-Rate\n(übersehene Deepfakes)']
    cmaps     = ['Greens', 'Reds', 'Oranges']

    # --- Rates berechnen ---
    # {metric: DataFrame mit index=Modell, columns=L2}
    rate_tables = {m: {} for m in metrics}

    for model_name, texts in model_texts.items():
        groups = {k: v for k, v in texts.items()}
        cv, matrices = _build_shared_vectorizer(groups)
        if cv is None:
            continue

        display = get_plot_label(model_name)
        for metric in metrics:
            n = len(texts.get(metric, []))
            if n == 0:
                for l2 in L2_ORDER:
                    rate_tables[metric].setdefault(l2, {})[display] = 0.0
                continue

            df_m = _counts_from_matrix(matrices.get(metric), cv, n, normalize=True)
            if df_m.empty:
                continue
            df_m = pd.merge(df_m, lookup, on='keyword', how='inner')
            if df_m.empty:
                continue

            for l2, grp in df_m.groupby('level_2'):
                rate_tables[metric].setdefault(l2, {})[display] = grp['count'].mean()

    # DataFrames bauen und auf L2_ORDER einschränken
    dfs = {}
    for metric in metrics:
        df = pd.DataFrame(rate_tables[metric]).reindex(columns=L2_ORDER).fillna(0)
        # Modellreihenfolge: nach Familien sortieren
        family_order = []
        for fam in sorted(set(BASE_MODEL_DISPLAY_NAMES.values())):
            for var in ['', '+I', '+T', '+I+T']:
                name = f'{fam}{var}'
                if name in df.index:
                    family_order.append(name)
        remaining = [m for m in df.index if m not in family_order]
        dfs[metric] = df.reindex(family_order + remaining)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(dfs['TP']) * 0.55 + 2)),
                             sharey=True)

    for ax, metric, title, cmap in zip(axes, metrics, titles, cmaps):
        df_plot = dfs[metric]
        im = ax.imshow(df_plot.values, aspect='auto', cmap=cmap, vmin=0)

        # Achsenbeschriftungen
        ax.set_xticks(range(len(L2_ORDER)))
        ax.set_xticklabels(L2_ORDER, fontsize=9, rotation=30, ha='right')
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot.index, fontsize=8)
        ax.set_title(title, fontsize=9, pad=8)

        # Trennlinien zwischen Modellfamilien
        prev_fam = None
        for i, name in enumerate(df_plot.index):
            fam = next((v for _, v in BASE_MODEL_DISPLAY_NAMES.items()
                        if name.startswith(v)), None)
            if fam and fam != prev_fam and i > 0:
                ax.axhline(i - 0.5, color='white', linewidth=2)
            prev_fam = fam

        # Zellwerte eintragen
        for r in range(df_plot.shape[0]):
            for c in range(df_plot.shape[1]):
                val = df_plot.values[r, c]
                text_color = 'white' if val > df_plot.values.max() * 0.6 else 'black'
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax, shrink=0.6,
                     label='Ø Nennungen pro Text')

    fig.suptitle('Modellvergleich: Keyword-Erwähnungsraten je L2-Region (aggregiert über alle Runs)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()

    fname = os.path.join(save_folder, 'chapter_overview_heatmap_L2.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Kapitel-Übersichts-Heatmap gespeichert: {fname}")


# ---------------------------------------------------------------------------
# MODULARE HILFSFUNKTIONEN — können einzeln aufgerufen werden
# ---------------------------------------------------------------------------

def _load_clustered_df(force_recluster=False):
    """
    Lädt clustered_df aus dem gecachten Excel, falls vorhanden.
    Mit force_recluster=True wird immer neu geclustert (SBERT läuft neu).
    Gibt None zurück wenn weder Cache noch Inventar vorhanden oder Cache veraltet ist.
    """
    cache_path = os.path.join(RESULTS_FOLDER, 'keyword_inventory_clustered.xlsx')
    if not force_recluster and os.path.exists(cache_path):
        df = pd.read_excel(cache_path)
        valid_l3 = set(KEYWORD_HIERARCHY.keys())
        cached_l3 = set(df['level_3'].dropna().unique())
        invalid = cached_l3 - valid_l3
        if invalid:
            print(f" -> Cache veraltet (unbekannte level_3-Werte: {invalid}) — starte Re-Clustering ...")
            return None
        # Re-derive level_2/level_1 to ensure consistency with current KEYWORD_HIERARCHY
        df['level_2'] = df['level_3'].map(lambda r: KEYWORD_HIERARCHY.get(r, ('Background', 'Frame'))[0])
        df['level_1'] = df['level_3'].map(lambda r: KEYWORD_HIERARCHY.get(r, ('Background', 'Frame'))[1])
        print(f" -> Lade gecachtes Keyword-Clustering: {cache_path}")
        return df
    print(" -> Kein Cache gefunden oder force_recluster=True — starte SBERT-Clustering ...")
    return None


def _extract_all_model_texts(df_master):
    """Extrahiert TP/FP/FN/TN-Texte pro Modell aggregiert über alle Runs."""
    all_model_texts = {}
    for _, df_run in iterate_runs(df_master):
        for model_name, model_df in df_run.groupby('base_model'):
            t = _extract_texts(model_df)
            if model_name not in all_model_texts:
                all_model_texts[model_name] = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
            for k in ('TP', 'FP', 'FN', 'TN'):
                all_model_texts[model_name][k].extend(t[k])
    return all_model_texts


def run_kw_clustering(df_master, force_recluster=False):
    """
    Schritt 1 (langsam): Inventar erstellen + SBERT-Clustering.
    Ergebnis wird gecacht → danach nicht mehr nötig neu zu laufen.
    Aufruf: run_kw_clustering(df_master, force_recluster=True)
    """
    clustered_df = _load_clustered_df(force_recluster)
    if clustered_df is not None:
        return clustered_df
    inventory_df = export_keyword_inventory(df_master)
    if inventory_df is None:
        return None
    return cluster_keywords(inventory_df)


def run_kw_per_model_plots(df_master, clustered_df=None, force_recluster=False):
    """
    Schritt 2: Hierarchische Plots pro Run × Modell.
    Lädt clustered_df aus Cache wenn nicht übergeben.
    Aufruf: run_kw_per_model_plots(df_master)
    """
    if clustered_df is None:
        clustered_df = _load_clustered_df(force_recluster)
    if clustered_df is None:
        print("Fehler: Kein clustered_df — zuerst run_kw_clustering() aufrufen.")
        return

    for run_label, df_run in iterate_runs(df_master):
        run_folder = os.path.join(base_plot_folder, run_label, 'Keywords_Analysis')
        os.makedirs(run_folder, exist_ok=True)
        for model_name, model_df in df_run.groupby('base_model'):
            t = _extract_texts(model_df)
            print(f"  [{run_label}] {model_name}: TP={len(t['TP'])}, FP={len(t['FP'])}, FN={len(t['FN'])}, TN={len(t['TN'])}")
            plot_keywords_hierarchical(t, model_name, run_label, run_folder, clustered_df)



def run_kw_comparison_heatmaps(df_master, clustered_df=None, force_recluster=False):
    """
    Schritt 4: Modellvergleichs-Heatmaps + Kapitel-Übersichts-Heatmap.
    Schnell hinzufügbar ohne die anderen Schritte neu laufen zu lassen.
    Aufruf: run_kw_comparison_heatmaps(df_master)
    """
    if clustered_df is None:
        clustered_df = _load_clustered_df(force_recluster)
    if clustered_df is None:
        print("Fehler: Kein clustered_df — zuerst run_kw_clustering() aufrufen.")
        return

    all_model_texts = _extract_all_model_texts(df_master)
    cmp_folder = os.path.join(base_plot_folder, 'Aggregated', 'Model_Comparison')
    os.makedirs(cmp_folder, exist_ok=True)

    print(" -> Erstelle Modellvergleichs-Heatmaps ...")
    for metric in ['FP', 'FN']:
        plot_model_comparison_heatmap(all_model_texts, clustered_df, cmp_folder,
                                      metric=metric, level='level_2')
        plot_model_comparison_heatmap(all_model_texts, clustered_df, cmp_folder,
                                      metric=metric, level='level_3')

    print(" -> Erstelle Kapitel-Übersichts-Heatmap ...")
    plot_chapter_overview_heatmap(all_model_texts, clustered_df, cmp_folder)


# --- MANAGER: Alles auf einmal ---
def run_justification_deep_analysis(df_master, force_recluster=False):
    """
    Führt alle Keyword-Analyse-Schritte aus.
    Mit force_recluster=False wird SBERT-Clustering aus dem Cache geladen
    wenn keyword_inventory_clustered.xlsx bereits existiert.

    Einzelne Schritte können auch direkt aufgerufen werden:
      run_kw_clustering(df_master)           # nur SBERT (langsam, einmalig)
      run_kw_per_model_plots(df_master)      # Plots pro Run × Modell
      run_kw_comparison_heatmaps(df_master)  # Vergleichs-Heatmaps (schnell)
    """
    analyze_region_coverage(df_master)

    clustered_df = run_kw_clustering(df_master, force_recluster=force_recluster)
    if clustered_df is None:
        return

    run_kw_per_model_plots(df_master, clustered_df)
    run_kw_comparison_heatmaps(df_master, clustered_df)



def run_best_per_family_ensemble(df_master):
    """
    Speichert Ergebnisse als Excel und als formatierte LaTeX-Tabelle.
    """
    print("\n=== Best-per-Family Ensemble (Per Run) ===")
    
    # Familien dynamisch aus globalen Definitionen ableiten
    families = sorted(list(set(BASE_MODEL_DISPLAY_NAMES.values())))
    ensemble_results = []

    all_runs = {r: d for r, d in iterate_runs(df_master)}

    for run_label in RUN_SUFFIXES:
        run_name = f"Run{run_label}"
        df_run = all_runs.get(run_name)

        if df_run is None: continue

        print(f" -> Berechne Ensemble für {run_name}...")
        best_models_tech = []
        best_models_display = []
        
        for fam_name in families:
            fam_df = df_run[df_run['base_model'].apply(lambda x: get_plot_label(x).startswith(fam_name))]
            if fam_df.empty: continue
            
            best_f1 = -1
            champion_tech = None
            
            for model_tech_name, model_data in fam_df.groupby('base_model'):
                f1 = f1_score(model_data['y_true'], model_data['y_pred'], pos_label=1, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    champion_tech = model_tech_name
            
            if champion_tech:
                best_models_tech.append(champion_tech)
                best_models_display.append(get_plot_label(champion_tech))

        if not best_models_tech: continue

        # Ensemble Berechnung (Majority Voting)
        df_champions = df_run[df_run['base_model'].isin(best_models_tech)].copy()
        ensemble_preds = df_champions.groupby('video_id')['y_pred'].mean()
        ensemble_binary = (ensemble_preds > 0.5).astype(int)
        
        y_true_map = df_run.drop_duplicates('video_id').set_index('video_id')['y_true']
        common_idx = ensemble_binary.index.intersection(y_true_map.index)
        
        metrics = calculate_metrics(y_true_map.loc[common_idx], ensemble_binary.loc[common_idx])
        metrics.update({
            'Run': run_name.replace('_', ' '),
            'Models_Used_Display': ", ".join(best_models_display)
        })
        ensemble_results.append(metrics)

    # ── TOP-3 BEST-PER-FAMILY ENSEMBLE ───────────────────────────────────────
    # Wie Best-per-Family, aber nur die 3 Familien mit dem höchsten F1-Score
    top3_results = []

    for run_label in RUN_SUFFIXES:
        run_name = f"Run{run_label}"
        df_run = all_runs.get(run_name)
        if df_run is None:
            continue

        print(f" -> Berechne Top-3 Best-per-Family Ensemble für {run_name}...")

        # Bestes Modell je Familie + zugehöriger F1-Score ermitteln
        family_champions = []
        for fam_name in families:
            fam_df = df_run[df_run['base_model'].apply(
                lambda x: get_plot_label(x).startswith(fam_name))]
            if fam_df.empty:
                continue
            best_f1, champion_tech = -1, None
            for model_tech_name, model_data in fam_df.groupby('base_model'):
                f1 = f1_score(model_data['y_true'], model_data['y_pred'],
                              pos_label=1, zero_division=0)
                if f1 > best_f1:
                    best_f1, champion_tech = f1, model_tech_name
            if champion_tech:
                family_champions.append((champion_tech, best_f1))

        # Nur die 3 besten Familien verwenden
        family_champions.sort(key=lambda x: x[1], reverse=True)
        top3_tech    = [m for m, _ in family_champions[:3]]
        top3_display = [get_plot_label(m) for m in top3_tech]

        df_top3 = df_run[df_run['base_model'].isin(top3_tech)].copy()
        ensemble_preds = df_top3.groupby('video_id')['y_pred'].mean()
        ensemble_binary = (ensemble_preds > 0.5).astype(int)

        y_true_map = df_run.drop_duplicates('video_id').set_index('video_id')['y_true']
        common_idx = ensemble_binary.index.intersection(y_true_map.index)

        metrics = calculate_metrics(y_true_map.loc[common_idx],
                                    ensemble_binary.loc[common_idx])
        metrics.update({
            'Run': run_name.replace('_', ' '),
            'Models_Used_Display': ", ".join(top3_display)
        })
        top3_results.append(metrics)

    if ensemble_results:
        df_ens  = pd.DataFrame(ensemble_results)
        df_top3 = pd.DataFrame(top3_results) if top3_results else pd.DataFrame()

        # 1. EXCEL SPEICHERN (zwei Reiter)
        with pd.ExcelWriter(os.path.join(RESULTS_FOLDER, 'results_ensemble_best_per_family.xlsx'), engine='openpyxl') as writer:
            df_ens.to_excel(writer, sheet_name='Best_per_Family', index=False)
            if not df_top3.empty:
                df_top3.to_excel(writer, sheet_name='Top3_Global', index=False)
        
        # 2. LATEX TABELLEN ERSTELLEN
        # ── Hilfsfunktion: DataFrame → .tex ──────────────────────────────────
        def _to_latex_file(df_in, filename, caption, label):
            df_l = df_in[cols_latex].copy()
            for col in ['F1-Score (%)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)']:
                df_l[col] = df_l[col].map(lambda x: f"{x:.1f}".replace('.', ','))
            df_l.columns = [c.replace('%', r'\%').replace('_', r'\_')
                            for c in df_l.columns]
            df_l.to_latex(
                filename,
                index=False,
                escape=False,
                column_format='lcccc p{6cm}',
                caption=caption,
                label=label
            )
            print(f" -> LaTeX-Tabelle gespeichert: {filename}")

        cols_latex = ['Run', 'F1-Score (%)', 'Accuracy (%)',
                      'Precision (%)', 'Recall (%)', 'Models_Used_Display']

        # Best-per-Family
        _to_latex_file(
            df_ens,
            os.path.join(RESULTS_FOLDER, 'table_ensemble_best_per_family.tex'),
            caption="Ergebnisse des Best-per-Family Ensembles über alle Runs",
            label="tab:ensemble_best_per_family"
        )

        # Top-3 Global
        if not df_top3.empty:
            _to_latex_file(
                df_top3,
                os.path.join(RESULTS_FOLDER, 'table_ensemble_top3.tex'),
                caption="Ergebnisse des Top-3 Best-per-Family Ensembles über alle Runs",
                label="tab:ensemble_top3"
            )


def run_worst_case_extraction(df_master):
    """
    Qualitative Analyse: "Hardest Samples".
    Findet nicht nur totale Versager (0%), sondern die Top-N schwierigsten Videos.
    Sortiert nach Fehlerquote.
    """
    print("\n=== Worst Case / Hardest Samples Analysis (Top 10) ===")
    
    for run_label, df_run in iterate_runs(df_master):
        # 1. Metadaten (y_true) pro Video holen
        meta_cols_needed = ['y_true'] + [c for c in META_COLS if c in df_run.columns]
        meta_info = (df_run[['video_id'] + meta_cols_needed]
                     .drop_duplicates('video_id')
                     .set_index('video_id'))

        # 2. Pivot: Zeilen=Videos, Spalten=Modelle, Werte=is_correct
        df_run['is_correct'] = (df_run['y_pred'] == df_run['y_true']).astype(int)
        pivot = df_run.pivot(index='video_id', columns='base_model', values='is_correct')

        model_cols = [c for c in pivot.columns]
        pivot['correct_count'] = pivot[model_cols].sum(axis=1)
        pivot['total_models']  = len(model_cols)
        pivot['failure_rate']  = 1.0 - (pivot['correct_count'] / pivot['total_models'])

        # y_true je Video anhängen für FN/FP-Trennung
        pivot = pivot.join(meta_info[['y_true']], how='left')

        def _top10(mask, label):
            sub = pivot[mask & (pivot['failure_rate'] > 0)]
            sub = sub.sort_values(
                by=['failure_rate', 'correct_count'], ascending=[False, True]
            ).head(10)
            if sub.empty:
                print(f"    {label}: keine Fehler.")
                return pd.DataFrame()
            result = sub[['correct_count', 'total_models', 'failure_rate']].join(
                meta_info, how='inner'
            )
            result['failure_rate'] = result['failure_rate'].round(2)
            print(f"    {label}: {len(result)} Videos "
                  f"(härtestes: {result.index[0]}, "
                  f"{result.iloc[0]['failure_rate']*100:.0f}% Fehlerquote)")
            return result

        # 3. FN = echte Deepfakes, die als Real klassifiziert wurden
        fn_result = _top10(pivot['y_true'] == 1, 'FN (Deepfake als Real)')
        # 4. FP = echte Real-Videos, die als Fake klassifiziert wurden
        fp_result = _top10(pivot['y_true'] == 0, 'FP (Real als Deepfake)')

        if not fn_result.empty or not fp_result.empty:
            filename = os.path.join(RESULTS_FOLDER, f'{run_label}_Hardest_Samples.xlsx')
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                if not fn_result.empty:
                    fn_result.to_excel(writer, sheet_name='FN_Deepfake_als_Real')
                if not fp_result.empty:
                    fp_result.to_excel(writer, sheet_name='FP_Real_als_Deepfake')
            print(f" -> {run_label}: Hardest Samples gespeichert in '{filename}'.")

            # ── FRAME-EXTRAKTION ─────────────────────────────────────────────
            for error_type, result_df in [('FN', fn_result), ('FP', fp_result)]:
                if result_df.empty:
                    continue
                frame_dir = os.path.join('frames', run_label, error_type)
                os.makedirs(frame_dir, exist_ok=True)

                for video_id in result_df.index:
                    # Videodatei suchen: data/processed/{run_nr}/{video_id}.mp4
                    pattern = os.path.join(VIDEO_SOURCE_PATH, '**',
                                           f'{video_id}.mp4')
                    matches = glob.glob(pattern, recursive=True)
                    if not matches:
                        print(f"    [WARN] Video nicht gefunden: {video_id}")
                        continue

                    cap = cv2.VideoCapture(matches[0])
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # Mittleren Frame extrahieren
                    mid = max(0, total_frames // 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        out_path = os.path.join(frame_dir, f'{video_id}.jpg')
                        cv2.imwrite(out_path, frame)
                    else:
                        print(f"    [WARN] Frame konnte nicht gelesen werden: {video_id}")

                print(f"    Frames gespeichert: frames/{run_label}/{error_type}/")
        else:
            print(f" -> {run_label}: Keine Fehler gefunden.")

    print(" Hardest Samples Analyse fertig.")

def run_audio_analysis(df_master):
    """
    Separate Analyse: Klassifikationsleistung nach Audio-Präsenz.
    Nur Baseline-Modelle, normalisierter Stacked Bar + Metriktabelle pro Run.
    Setzt voraus, dass 'audio' in dataset_info.xlsx vorhanden ist.
    """
    print("\n=== Audio Analysis ===")

    AUDIO_COL = 'audio'
    baseline_keys = [k for k in BASE_MODEL_DISPLAY_NAMES
                     if '_indicators' not in k and '_thinking' not in k]

    save_dir = os.path.join(base_plot_folder, 'Audio_Analysis')
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = []

    for run_label, df_run in iterate_runs(df_master):
        if AUDIO_COL not in df_run.columns:
            print(f" -> '{AUDIO_COL}'-Spalte nicht in df_master gefunden. "
                  f"Prüfe ob dataset_info.xlsx die Spalte enthält.")
            return

        df_run = df_run[df_run['base_model'].isin(baseline_keys)].copy()
        df_run['Outcome'] = (df_run['y_pred'] == df_run['y_true']).map(
            {True: 'Korrekt', False: 'Fehler'}
        )
        df_run['Model'] = df_run['base_model'].apply(get_plot_label)
        df_run[AUDIO_COL] = df_run[AUDIO_COL].astype(str).str.strip()

        audio_groups = sorted(df_run[AUDIO_COL].dropna().unique())
        models_ordered = sorted(df_run['Model'].unique())
        n_models = len(models_ordered)

        # ── STACKED BAR CHART ────────────────────────────────────────────────
        fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models_ordered):
            sub = df_run[df_run['Model'] == model].dropna(subset=[AUDIO_COL])

            counts = (sub.groupby([AUDIO_COL, 'Outcome'], observed=False)
                        .size()
                        .unstack(fill_value=0))
            for col in ['Korrekt', 'Fehler']:
                if col not in counts.columns:
                    counts[col] = 0

            counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100
            counts_pct = counts_pct[['Korrekt', 'Fehler']]

            counts_pct.plot(
                kind='bar', stacked=True, ax=ax,
                color=['#2ca02c', '#d62728'],
                edgecolor='white', linewidth=0.5,
                legend=(ax == axes[-1])
            )

            for i, (grp, row) in enumerate(counts.iterrows()):
                ax.text(i, 101, f'n={int(row.sum())}', ha='center', va='bottom',
                        fontsize=7, color='gray')

            ax.set_title(model, fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
            ax.set_ylim(0, 115)
            ax.axhline(50, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
            if ax == axes[0]:
                ax.set_ylabel('Anteil (%)', fontsize=10)
            else:
                ax.set_ylabel('')
            if ax == axes[-1]:
                ax.legend(loc='upper right', fontsize=8, title='Ergebnis')

        fig.suptitle(f'{run_label} – Klassifikationsergebnis nach Audio-Präsenz',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'Audio_StackedBar_{run_label}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ── METRIKTABELLE PRO AUDIO-GRUPPE ───────────────────────────────────
        for model_name, model_df in df_run.groupby('base_model'):
            for grp in audio_groups:
                sub = model_df[model_df[AUDIO_COL] == grp]
                if len(sub) < 3:
                    continue
                m = calculate_metrics(sub['y_true'], sub['y_pred'])
                m.update({
                    'Run': run_label,
                    'Model': get_plot_label(model_name),
                    'Audio': grp,
                    'N': len(sub)
                })
                all_metrics.append(m)

    if all_metrics:
        df_out = pd.DataFrame(all_metrics)
        cols_front = ['Run', 'Model', 'Audio', 'N',
                      'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
        cols_front = [c for c in cols_front if c in df_out.columns]
        df_out[cols_front].to_excel(
            os.path.join(RESULTS_FOLDER, 'Audio_Metrics.xlsx'), index=False
        )
        print(f" -> Audio_Metrics.xlsx und Plots gespeichert in '{save_dir}'.")


def run_inter_model_similarity(df_master):
    """
    Cross-Model Semantic Similarity via SBERT.
    Ausgabe pro Run:
      1. PCA-Scatter-Plot (alle 20 Varianten, Farbe=Familie, Form=Prompt-Variante)
      2. 5×5 Familien-Heatmap (nur Baselines, mittlere Ähnlichkeit je Familie)
    """
    from sklearn.decomposition import PCA

    print("\n=== Inter-Model Semantic Similarity (SBERT) ===")

    # Marker-Formen je Prompt-Variante
    VARIANT_MARKERS = {'': 'o', '+I': 's', '+T': '^', '+I+T': 'D'}

    for run_label, df_run in iterate_runs(df_master):
        save_dir = os.path.join(base_plot_folder, run_label)
        os.makedirs(save_dir, exist_ok=True)

        if 'justification' not in df_run.columns:
            print(f" -> {run_label}: Keine Justification-Spalte.")
            continue

        models = sorted(df_run['base_model'].dropna().unique())
        print(f" -> {run_label}: Encodiere {len(models)} Modelle...")

        # ── Mittlere Embeddings berechnen (1 Vektor pro Modell) ──────────────
        mean_embeddings = {}
        for m in models:
            texts = (df_run[df_run['base_model'] == m]['justification']
                     .dropna().astype(str).tolist())
            if len(texts) < 3:
                continue
            emb = get_sbert().encode(texts, convert_to_tensor=False,
                                     show_progress_bar=False)
            mean_embeddings[m] = emb.mean(axis=0)

        valid_models = list(mean_embeddings.keys())
        if len(valid_models) < 2:
            print(f" -> {run_label}: Zu wenige Modelle.")
            continue

        emb_matrix = np.stack([mean_embeddings[m] for m in valid_models])
        labels     = [get_plot_label(m) for m in valid_models]
        colors     = [get_model_color(m) for m in valid_models]

        variants = [next((v for v in ['+I+T', '+T', '+I', ''] if l.endswith(v)), '') for l in labels]

        # ── PLOT 1: PCA-Scatter ───────────────────────────────────────────────
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_matrix)
        var_explained = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (label, color, variant) in enumerate(zip(labels, colors, variants)):
            marker = VARIANT_MARKERS.get(variant, 'o')
            ax.scatter(coords[i, 0], coords[i, 1],
                       color=color, marker=marker,
                       s=120, edgecolors='white', linewidths=0.8, zorder=3)
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=7.5, color=color)

        # Legende: Prompt-Varianten
        from matplotlib.lines import Line2D
        legend_variants = [
            Line2D([0], [0], marker=m, color='gray', linestyle='None',
                   markersize=8, label=f'Baseline{v}' if v == '' else v)
            for v, m in VARIANT_MARKERS.items()
        ]
        ax.legend(handles=legend_variants, title='Prompt-Variante',
                  loc='lower right', fontsize=8)

        ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}% Varianz)', fontsize=10)
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}% Varianz)', fontsize=10)
        ax.set_title(f'{run_label}: Semantische Ähnlichkeit der Justifications (PCA)',
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Similarity_PCA_Scatter.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> {run_label}: PCA-Scatter gespeichert.")

        # ── PLOT 2: 5×5 Familien-Heatmap (nur Baselines) ─────────────────────
        families = sorted(set(BASE_MODEL_DISPLAY_NAMES.values()))
        baseline_keys = {v: k for k, v in BASE_MODEL_DISPLAY_NAMES.items()
                         if '_indicators' not in k and '_thinking' not in k}

        fam_embs = {}
        for fam, tech_key in baseline_keys.items():
            if tech_key in mean_embeddings:
                fam_embs[fam] = mean_embeddings[tech_key]

        fam_names = sorted(fam_embs.keys())
        if len(fam_names) < 2:
            continue

        n_fam = len(fam_names)
        sim_fam = np.zeros((n_fam, n_fam))
        for i, f1 in enumerate(fam_names):
            for j, f2 in enumerate(fam_names):
                e1 = fam_embs[f1] / np.linalg.norm(fam_embs[f1])
                e2 = fam_embs[f2] / np.linalg.norm(fam_embs[f2])
                sim_fam[i, j] = float(np.dot(e1, e2))

        df_fam = pd.DataFrame(sim_fam, index=fam_names, columns=fam_names)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df_fam, annot=True, fmt='.2f', cmap='YlOrRd',
                    vmin=0.5, vmax=1.0, linewidths=0.5,
                    cbar_kws={'label': 'Kosinus-Ähnlichkeit'}, ax=ax)
        ax.set_title(f'{run_label}: Semantische Ähnlichkeit zwischen Modellfamilien\n'
                     f'(Baseline-Modelle)',
                     fontsize=11, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Similarity_Family_Heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> {run_label}: Familien-Heatmap gespeichert.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Raw Data Extraction & Aggregation 
    for s in RUN_SUFFIXES: run_analysis(s)
    run_aggregation_and_benchmark()
    
    df_ground_truth = pd.read_excel(INFO_DATEI)
    run_baseline_distribution_analysis(df_ground_truth)

    # Master Data einmal laden
    df_master = load_master_data()
    
    if df_master is not None:

        # --- GLOBALE / AGGREGIERTE ANALYSEN  ---
        #run_global_baseline_roc_analysis(df_master)   
        #run_family_variant_roc_comparison(df_master)
        
        # --- PER-RUN ANALYSEN ---
        #generate_plots(df_master) 
        #run_feature_importance_analysis(df_master)
        #run_feature_analysis(df_master)
        #run_fairness_analysis(df_master)
        #run_intra_model_consistency_check(df_master)
        #run_inter_model_similarity(df_master)
        #run_audio_analysis(df_master)
        #run_justification_deep_analysis(df_master)
        #run_best_per_family_ensemble(df_master)
        run_worst_case_extraction(df_master)
        
    print("\n=== Fertig ===")