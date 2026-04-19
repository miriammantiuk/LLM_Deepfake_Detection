import pandas as pd
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer, util
import itertools
import torch



# ---------------------------------------------------------
# GLOBALE KONFIGURATION
# ---------------------------------------------------------
JSON_FOLDER = '.'               
INFO_DATEI = 'dataset_info.xlsx' 
VIDEO_SOURCE_PATH = r'data\processed'  
SUMMARIZED_FILE = 'results_summarized.xlsx'

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
META_COLS = ['dataset', 'gender', 'video_length','deepfake_category','deepfake_type']

# Plotting Style
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0}) 

# Hauptordner für Plots
base_plot_folder = 'plots'
os.makedirs(base_plot_folder, exist_ok=True)


# Stopwörter für die Keyword-Analyse 
DOMAIN_STOPS = list(ENGLISH_STOP_WORDS) + [
    'video', 'clip', 'footage', 'appears', 'seems', 'looks', 'shows', 
    'detected', 'generated', 'ai', 'content', 'likely', 'probability', 
    'confidence', 'based', 'analysis', 'observe', 'observed', 'visual', 
    'anomalies', 'artifacts', 'signs', 'potential', 'deepfake'
]

# SBERT zentral laden (für semantische Analysen)
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------
# HILFSFUNKTIONEN (METRIKEN & LOGIK)
# ---------------------------------------------------------
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
    output_datei = f'results{suffix}.xlsx'
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
        except: pass

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
            df = pd.read_excel(f'results{suffix}.xlsx')
            
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
    except: pass
    
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
    df_metrics[cols_order].to_excel('benchmark_detailed_runs.xlsx', index=False)
    print(" benchmark_detailed_runs.xlsx erstellt.")

    # B) Scientific Report & LaTeX Generierung
    metric_cols = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC AUC']
    metric_cols = [c for c in metric_cols if c in df_metrics.columns]
    
    with pd.ExcelWriter('benchmark_scientific_report.xlsx') as writer:
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
            filename = f'benchmark_results_{threshold}.tex'
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

    # Check ob Metadaten schon da sind, sonst nachladen
    if not all(col in df.columns for col in META_COLS):
        print("Merged Metadaten (dataset_info)...")
        if os.path.exists(INFO_DATEI):
            df_info = pd.read_excel(INFO_DATEI)
            df_info['video_id'] = df_info['video_id'].astype(str)
            # Left join um Datenverlust zu vermeiden
            df = pd.merge(df, df_info, on='video_id', how='left', suffixes=('', '_info'))
            
            # Falls y_true fehlte, füllen
            if 'y_true' not in df.columns and 'deepfake' in df.columns:
                 df['y_true'] = df['deepfake'].map({"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1})
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
    agg_folder = os.path.join(base_plot_folder, 'Aggregated_Analysis')
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
    print("\n=== Feature Importance (Per Run) ===")
    
    for run_label, df_run in iterate_runs(df_master):
        save_dir = os.path.join(base_plot_folder, run_label, 'Feature_Importance')
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"{run_label}...")
        
        for model_name, group in df_run.groupby('base_model'):
            sub_df = group.copy()
            if len(sub_df) < 10: continue
            
            sub_df['error'] = (sub_df['y_pred'] != sub_df['y_true']).astype(int)
            if sub_df['error'].nunique() < 2: continue

            meta_cols = [c for c in META_COLS if c in sub_df.columns]
            if not meta_cols:
                continue

            try:
                X_raw = sub_df[meta_cols]
                X = pd.get_dummies(X_raw, dummy_na=True).fillna(0)

                clf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=42,
                )
                clf.fit(X, sub_df["error"])

                fi = (
                    pd.DataFrame(
                        {"Feature": X.columns, "Importance": clf.feature_importances_}
                    )
                    .sort_values("Importance", ascending=False)
                    .head(10)
                )

                plt.figure(figsize=(8, 6))
                sns.barplot(data=fi, x="Importance", y="Feature")
                plot_label = get_plot_label(model_name)
                plt.title(f"Error Drivers: {plot_label}")
                plt.tight_layout()

                safe_name = re.sub(r"[^\w\-_\. ]", "_", str(model_name))
                plt.savefig(os.path.join(save_dir, f"FI_{safe_name}.png"), dpi=200)
                plt.close()

            except Exception as e:
                print(f"[WARN] {run_label} | {model_name}: {e}")
                continue

def run_standard_correlation_analysis(df_master):
    print("\n=== Correlation Analysis (Per Run) ===")
    
    for run_label, df_run in iterate_runs(df_master):
        run_folder = os.path.join(base_plot_folder, run_label)
        os.makedirs(run_folder, exist_ok=True)
        
        # Zielvariable: Error (1 = Falsch, 0 = Richtig)
        y = (df_run['y_pred'] != df_run['y_true']).astype(int)
        

        df_analysis = pd.DataFrame({'error': y})
        
        for col in META_COLS:
            if col not in df_run.columns:
                continue
            
            # Prüfen: Ist die Spalte numerisch?
            if pd.api.types.is_numeric_dtype(df_run[col]):
                df_analysis[col] = df_run[col]
            else:
                # Kategorisch (z.B. gender, dataset): One-Hot-Encoding
                # drop_first=True verhindert Redundanz (z.B. nur 'gender_male', nicht 'gender_female')
                dummies = pd.get_dummies(df_run[col], prefix=col, drop_first=True)
                df_analysis = pd.concat([df_analysis, dummies], axis=1)

        # Korrelation berechnen & Plotten
        if df_analysis.shape[1] > 1:
            # Korrelation aller Spalten zur Spalte 'error'
            corr = df_analysis.corr()['error'].drop('error').sort_values()
            
            plt.figure(figsize=(10, 8))
        
            # Farben: Rot = Verursacht Fehler (Positive Corr), Blau = Verhindert Fehler (Negative Corr)
            colors = ['#4e79a7' if x < 0 else '#e15759' for x in corr.values]
            
            corr.plot(kind='barh', color=colors)
            
            plt.title(f'Feature Correlation with Prediction Error ({run_label.replace("_", " ")})')
            plt.xlabel("Correlation Coefficient (Pearson)")
            plt.axvline(0, color='black', linewidth=0.8) # Nulllinie zur Orientierung
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(run_folder, 'Correlation_BarChart.png'))
            plt.close()
            print(f" -> Plot für {run_label} erstellt (nur META_COLS).")

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
        df_res.to_excel('feature_analysis.xlsx', index=False)
        
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
                except: pass

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
        df_res.to_excel('fairness_analysis.xlsx', index=False)
        
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
                except: pass





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
            emb = model_sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False)
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
    df_res.to_excel('consistency_analysis_intra_model_raw.xlsx', index=False)
    
    # Aggregation berechnen
    summary = df_res.groupby('base_model')['Consistency_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    summary.reset_index().to_excel('consistency_summary_per_model.xlsx', index=False)
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
    
    save_path = os.path.join(base_plot_folder, 'keyword_inventory.xlsx')
    inventory.to_excel(save_path, index=False)
    print(f"Inventar gespeichert: keyword_inventory.xlsx (Basierend auf {len(justification_cols)} Runs)")
    return inventory


def cluster_keywords(inventory_df, output_filename='keyword_inventory_mare_clustered.xlsx'):
    """
    Nutzt die offiziellen MARE-Keywords als Anker für das SBERT-Clustering.
    """
    print(" -> Starte SBERT-Clustering mit MARE-Ankern...")

    # 1. Modell laden
    model = model_sbert

    # 2. Anker-Mapping basierend auf MARE Paper
    mare_anchors = {
        'Skin': 'skin, cheek, forehead, complexion, dermal, face',
        'Nose': 'nose, nostril, nasal',
        'Mouth': 'mouth, lip, lips',
        'Teeth': 'tooth, teeth',
        'Left_Eye': 'left eye, left-eye, l eye, lefteye, eye, ocular',
        'Right_Eye': 'right eye, right-eye, r eye, righteye, eye, ocular',
        'Left_Eyebrow': 'left eyebrow, left brow, left-eyebrow, eyebrow, brow',
        'Right_Eyebrow': 'right eyebrow, right brow, right-eyebrow, eyebrow, brow',
        'Chin': 'chin, jaw, jawline, lower face',
        'Beard': 'beard, mustache, moustache, goatee',
        'Hairline': 'hairline, hair line, hair',
        'Ear': 'ear, ears'
    }

    region_names = list(mare_anchors.keys())
    anchor_texts = list(mare_anchors.values())

    # 3. Embeddings berechnen
    # Wir berechnen die Vektoren für die MARE-Gruppen und deine extrahierten Keywords
    region_embeddings = model.encode(anchor_texts, convert_to_tensor=True)
    keyword_list = inventory_df['keyword'].tolist()
    keyword_embeddings = model.encode(keyword_list, convert_to_tensor=True)

    # 4. Cosine Similarity (Semantischer Vergleich)
    cosine_scores = util.cos_sim(keyword_embeddings, region_embeddings)

    # 5. Zuweisung
    # Wir suchen für jedes Keyword die Region mit der höchsten Ähnlichkeit
    best_region_indices = torch.argmax(cosine_scores, dim=1).tolist()
    confidences = torch.max(cosine_scores, dim=1).values.tolist()

    inventory_df['assigned_region'] = [region_names[i] for i in best_region_indices]
    inventory_df['confidence'] = confidences

    # Schwellenwert: Wenn die Ähnlichkeit zu gering ist (< 0.4), markieren wir es als 'Other/Technical'
    inventory_df.loc[inventory_df['confidence'] < 0.4, 'assigned_region'] = 'Other/Technical'

    # 6. Speichern
    save_path = os.path.join(base_plot_folder, output_filename)
    # Sortierung für den Audit (Review)
    inventory_df = inventory_df.sort_values(by=['assigned_region', 'confidence'], ascending=[True, False])
    inventory_df.to_excel(save_path, index=False)
    
    print(f"Keyword-Clustering abgeschlossen. Datei gespeichert: {save_path}")
    return inventory_df

# --- MODUL 2: EINZEL-PLOTS (PRO MODELL & RUN) ---
def plot_individual_keywords(texts, model_name, condition_label, save_folder, run_label):
    """Erstellt einen Balkenchart der Top-Phrasen für ein Modell/Szenario."""
    if not texts or len(texts) < 3: 
        return

    # Fokus auf 2-3 Wörter für mehr Kontext
    cv = CountVectorizer(ngram_range=(2, 3), stop_words=DOMAIN_STOPS, max_features=15)
    try:
        counts = cv.fit_transform(texts)
        df_plot = pd.DataFrame({
            'Phrase': cv.get_feature_names_out(),
            'Count': counts.sum(axis=0).A1
        }).sort_values(by='Count', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_plot, x='Count', y='Phrase', hue='Phrase', palette='flare', legend=False)
        plt.title(f"Run: {run_label}\nModel: {model_name} ({condition_label})", fontsize=12)
        plt.xlabel("Häufigkeit")
        plt.ylabel("")
        plt.tight_layout()
        
        filename = f"{model_name}_{condition_label}_{run_label}.png"
        plt.savefig(os.path.join(save_folder, filename), dpi=300)
        plt.close()
        
    except ValueError:
        pass # Falls keine n-gramme extrahierbar sind

    


# --- MODUL 3: HALLUZINATIONSMATRIX ---
def plot_hallucination_matrix(aggregated_fp_dict, output_path, expected_models):
    """Erstellt eine Heatmap der Fehlerbegründungen über alle Modelle hinweg."""
    matrix_data = []
    
    for model_name, fp_texts in aggregated_fp_dict.items():
        if not fp_texts: continue
        
        cv = CountVectorizer(ngram_range=(2, 3), stop_words=DOMAIN_STOPS, max_features=20)
        try:
            counts = cv.fit_transform(fp_texts)
            words = cv.get_feature_names_out()
            sums = counts.sum(axis=0).A1
            
            total_fps = len(fp_texts)
            for word, count in zip(words, sums):
                matrix_data.append({
                    'Model': model_name,
                    'Phrase': word,
                    'Frequency (%)': round((count / total_fps) * 100, 1)
                })
        except ValueError:
            continue

    if not matrix_data:
        print("Keine Daten für Halluzinationsmatrix vorhanden.")
        return

    df_matrix = pd.DataFrame(matrix_data)
    top_overall = df_matrix.groupby('Phrase')['Frequency (%)'].sum().nlargest(20).index
    
    df_pivot = df_matrix[df_matrix['Phrase'].isin(top_overall)].pivot(
        index='Phrase', columns='Model', values='Frequency (%)'
    ).fillna(0.0)

    # Alle Modelle in der X-Achse erzwingen
    df_pivot = df_pivot.reindex(columns=expected_models).fillna(0.0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(df_pivot, annot=True, cmap="YlOrRd", fmt=".1f", cbar_kws={'label': 'Vorkommen in FPs (%)'})
    plt.title("Halluzinations-Matrix: Warum irren sich die Modelle?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Halluzinationsmatrix gespeichert: {output_path}")


# --- MANAGER: DIE HAUPTFUNKTION ---
def run_justification_deep_analysis(df_master):
    """Koordiniert die gesamte Analyse."""
    
    # 1. Zentrales Inventar für das Mapping erstellen und clustern
    inventory_df = export_keyword_inventory(df_master)
    if inventory_df is not None:
        cluster_keywords(inventory_df)
    
    aggregated_fps = {} 

    # 2. Durch alle Runs und Modelle iterieren
    # (Annahme: iterate_runs ist eine existierende Hilfsfunktion)
    for run_label, df_run in iterate_runs(df_master):
        
        run_folder = os.path.join(base_plot_folder, run_label, 'Keywords_Analysis')
        os.makedirs(run_folder, exist_ok=True)
        
        for model_name, model_df in df_run.groupby('base_model'):
            # Masken für TP und FP (Halluzinationen)
            tp_mask = (model_df['y_true'] == 1) & (model_df['y_pred'] == 1)
            fp_mask = (model_df['y_true'] == 0) & (model_df['y_pred'] == 1)
            
            tp_list = model_df[tp_mask]['justification'].dropna().tolist()
            fp_list = model_df[fp_mask]['justification'].dropna().tolist()
            
            # Einzel-Plots erstellen
            plot_individual_keywords(tp_list, model_name, "TP", run_folder, run_label)
            plot_individual_keywords(fp_list, model_name, "FP", run_folder, run_label)
            
            # Für die globale Matrix sammeln
            if model_name not in aggregated_fps: 
                aggregated_fps[model_name] = []
            aggregated_fps[model_name].extend(fp_list)

    # 3. Finale Matrix über alle Runs hinweg
    matrix_path = os.path.join(base_plot_folder, 'global_hallucination_matrix.png')
    expected_models = sorted(aggregated_fps.keys())
    plot_hallucination_matrix(aggregated_fps, matrix_path, expected_models)



def run_best_per_family_ensemble(df_master):
    """
    Speichert Ergebnisse als Excel und als formatierte LaTeX-Tabelle.
    """
    print("\n=== Best-per-Family Ensemble (Per Run) ===")
    
    # Familien dynamisch aus globalen Definitionen ableiten
    families = sorted(list(set(BASE_MODEL_DISPLAY_NAMES.values())))
    ensemble_results = []

    for run_label in RUN_SUFFIXES:
        run_name = f"Run{run_label}"
        # Nutze iterate_runs Logik indirekt oder direkt filtern
        df_run_master = list(iterate_runs(df_master))
        # Suche den passenden Run in der Liste
        df_run = next((d for r, d in df_run_master if r == run_name), None)
        
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

    if ensemble_results:
        df_ens = pd.DataFrame(ensemble_results)
        
        # 1. EXCEL SPEICHERN
        df_ens.to_excel('results_ensemble_best_per_family.xlsx', index=False)
        
        # 2. LATEX TABELLE ERSTELLEN
        # Spalten selektieren wie gewünscht
        cols_latex = ['Run', 'F1-Score (%)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'Models_Used_Display']
        df_latex = df_ens[cols_latex].copy()
        
        # Formatierung: Punkt zu Komma (German Style) und Rundung
        for col in ['F1-Score (%)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)']:
            df_latex[col] = df_latex[col].map(lambda x: f"{x:.1f}".replace('.', ','))
        
        # Spaltennamen für LaTeX säubern (Prozentzeichen escapen)
        df_latex.columns = [c.replace('%', r'\%').replace('_', r'\_') for c in df_latex.columns]
        
        # Models_Used_Display: Kommas in der Liste schöner umbrechen für LaTeX
        df_latex[df_latex.columns[-1]] = df_latex[df_latex.columns[-1]].str.replace(', ', ', ')

        # Speichern als .tex Datei
        latex_file = 'table_ensemble_results.tex'
        # Spaltendefinition: l für Run, c für die 4 Metriken, p{6cm} für die langen Modellnamen
        col_format = 'lcccc p{6cm}' 
        
        df_latex.to_latex(
            latex_file, 
            index=False, 
            escape=False, 
            column_format=col_format,
            caption="Ergebnisse des Best-per-Family Ensembles über alle Inferenzdurchläufe",
            label="tab:ensemble_results"
        )
        
        print(f" -> LaTeX-Tabelle gespeichert: {latex_file}")


def run_worst_case_extraction(df_master):
    """
    Qualitative Analyse: "Hardest Samples".
    Findet nicht nur totale Versager (0%), sondern die Top-N schwierigsten Videos.
    Sortiert nach Fehlerquote.
    """
    print("\n=== Worst Case / Hardest Samples Analysis (Top 10) ===")
    
    for run_label, df_run in iterate_runs(df_master):
        # 1. Pivot Tabelle: Zeilen=Videos, Spalten=Modelle, Werte=0/1
        df_run['is_correct'] = (df_run['y_pred'] == df_run['y_true']).astype(int)
        
        # Pivot erstellen
        pivot = df_run.pivot(index='video_id', columns='base_model', values='is_correct')
        
        # 2. Metriken pro Video berechnen
        # Summe der korrekten Vorhersagen (Zeilensumme)
        pivot['correct_count'] = pivot.sum(axis=1)
        pivot['total_models'] = pivot.shape[1] - 1 # Anzahl Modelle (minus die Spalte correct_count selbst)
        
        # Fehlerquote berechnen (1.0 = Alle falsch, 0.0 = Alle richtig)
        pivot['failure_rate'] = 1.0 - (pivot['correct_count'] / pivot['total_models'])
        
        # 3. Filtern & Sortieren
        # Wir nehmen Videos, bei denen mindestens 50% der Modelle falsch lagen (Majority Fail)
        # ODER einfach die Top 10 schlechtesten, egal wie gut sie sind.
        
        # Sortieren: Höchste Failure Rate zuerst, bei Gleichstand wenigste Correct Counts
        hardest_samples = pivot.sort_values(by=['failure_rate', 'correct_count'], ascending=[False, True]).head(10)
        
        # Nur Videos nehmen, die überhaupt Fehler hatten (Failure Rate > 0)
        hardest_samples = hardest_samples[hardest_samples['failure_rate'] > 0]
        
        if not hardest_samples.empty:
            # 4. Metadaten wieder anhängen
            # Wir holen uns Dataset, Type etc. aus dem Original-DF (die erste Zeile pro Video reicht)
            meta_cols_needed = ['y_true'] + [c for c in META_COLS if c in df_run.columns]
            meta_info = df_run[['video_id'] + meta_cols_needed].drop_duplicates('video_id').set_index('video_id')
            
            # Join: Hardest Samples + Metadaten
            result = hardest_samples[['correct_count', 'total_models', 'failure_rate']].join(meta_info, how='inner')
            
            # Runden für schöne Excel
            result['failure_rate'] = result['failure_rate'].round(1)
            
            # Speichern
            filename = f'{run_label}_Hardest_Samples.xlsx'
            result.to_excel(filename)
            print(f" -> {run_label}: Top {len(result)} schwierigste Videos gespeichert in '{filename}'.")
            
            # Kurzer Print zur Info
            top_video = result.index[0]
            top_fail = result.iloc[0]['failure_rate'] * 100
            print(f"    (Härtestes Video: {top_video} mit {top_fail:.0f}% Fehlerquote)")
            
        else:
            print(f" -> {run_label}: Perfekter Run? Keine Fehler gefunden.")

    print(" Hardest Samples Analyse fertig.")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Raw Data Extraction & Aggregation 
    for s in RUN_SUFFIXES: run_analysis(s)
    run_aggregation_and_benchmark()
    
    # 2. Master Data einmal laden
    df_master = load_master_data()
    
    if df_master is not None:

        # --- GLOBALE / AGGREGIERTE ANALYSEN  ---
        #run_global_baseline_roc_analysis(df_master)   
        #run_family_variant_roc_comparison(df_master)
        
        # --- PER-RUN ANALYSEN ---
        #generate_plots(df_master) 
        #run_feature_importance_analysis(df_master)
        #run_standard_correlation_analysis(df_master)
        #run_feature_analysis(df_master)
        #run_fairness_analysis(df_master)
        #run_intra_model_consistency_check(df_master)
        run_justification_deep_analysis(df_master)
        #run_best_per_family_ensemble(df_master)
        #run_worst_case_extraction(df_master)
        
    print("\n=== Fertig ===")