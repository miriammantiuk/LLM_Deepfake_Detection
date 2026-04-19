import pandas as pd
import json
import glob
import os
import ast
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, roc_auc_score, 
                             roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURATION ---
JSON_FOLDER = '.'               # Ordner mit den JSON-Dateien
INFO_DATEI = 'dataset_info.xlsx' # Excel mit Ground Truth
VIDEO_SOURCE_PATH = r'data\processed'  # Pfad zu den Original-Videos
# ---------------------

# Plotting Style
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0}) 

# Hauptordner für Plots
base_plot_folder = 'plots'
os.makedirs(base_plot_folder, exist_ok=True)

# ---------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Berechnet verschiedene Metriken für binäre Klassifikation.
    """
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
        'Accuracy (%)': round(acc * 100, 2),
        'Precision (%)': round(precision * 100, 2),
        'Recall (%)': round(recall * 100, 2),
        'F1-Score (%)': round(f1 * 100, 2),
        'ROC AUC': round(roc_auc, 4) if roc_auc is not None else 'N/A',
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }


def get_word_count(val):
    """
    Zählt die Wörter in einer Begründung.
    """
    if isinstance(val, list): return len(" ".join(str(v) for v in val).split())
    elif isinstance(val, str): return len(val.split())
    return 0

# ---------------------------------------------------------
# ANALYSE-FUNKTIONEN
# ---------------------------------------------------------

def run_analysis(suffix=""):
    """
    Lädt alle JSON-Dateien mit dem angegebenen Suffix, extrahiert die relevanten Daten und speichert sie in einer Excel-Datei.
    """
    output_datei = f'results{suffix}.xlsx'
    print(f"--- SCHRITT 1: JSON-Dateien laden für Gruppe {suffix} ---")
    json_files = glob.glob(os.path.join(JSON_FOLDER, f'*{suffix}.json'))
    
    if not json_files:
        print(f"Keine JSON-Dateien für Gruppe {suffix} gefunden!")
        return

    all_data = []
    for file_path in json_files:
        filename = os.path.basename(file_path)
        model_name = os.path.splitext(filename)[0]
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            df_temp = pd.DataFrame(data)
            df_temp['model'] = model_name 
            df_temp['video_id'] = df_temp['video_id'].astype(str).str.replace(r'(?i)\.mp4$', '', regex=True)
            df_temp['justification_length'] = df_temp['justification'].apply(get_word_count) if 'justification' in df_temp.columns else 0
            all_data.append(df_temp)
            print(f"Geladen: {filename} ({len(df_temp)} Videos)")
        except Exception as e: print(f"Fehler bei {filename}: {e}")

    if not all_data: return

    df_gesamt = pd.concat(all_data, ignore_index=True)
    label_mapping = {"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1}
    df_gesamt['y_pred'] = df_gesamt['assessment'].map(label_mapping)
    df_gesamt.to_excel(output_datei, index=False)
    print(f"Rohe Ergebnisse für {suffix} gespeichert.")

def calculate_aligned_probability(preds_list, probs_list):
    """
    Nimmt eine Liste von Vorhersagen (0/1) und Wahrscheinlichkeiten.
    Berechnet den Durchschnitt der Wahrscheinlichkeiten der Mehrheitsklasse.
    """
    # Daten bereinigen (NaNs entfernen)
    valid_data = []
    for p, prob in zip(preds_list, probs_list):
        if pd.notna(p) and pd.notna(prob):
            valid_data.append((int(p), float(prob)))
            
    if not valid_data: return 0.0

    # 1. Gewinner ermitteln (Majority Vote)
    clean_preds = [x[0] for x in valid_data]
    winner = 1 if sum(clean_preds) > (len(clean_preds) / 2) else 0
    
    # 2. Nur Wahrscheinlichkeiten des Gewinners mitteln
    winning_probs = [x[1] for x in valid_data if x[0] == winner]
    
    return np.mean(winning_probs) if winning_probs else 0.0

def create_unified_justification(preds_list, texts_list):
    """
    Nimmt eine Liste von Vorhersagen und Texten.
    Kombiniert die Texte der Mehrheitsklasse ohne Duplikate.
    """
    # Daten bereinigen
    valid_data = []
    for p, txt in zip(preds_list, texts_list):
        if pd.notna(p) and isinstance(txt, str) and len(txt) > 5:
            valid_data.append((int(p), txt))
            
    if not valid_data: return ""

    # 1. Gewinner ermitteln
    clean_preds = [x[0] for x in valid_data]
    winner = 1 if sum(clean_preds) > (len(clean_preds) / 2) else 0
    
    # 2. Texte mergen
    sentences = []
    for pred, text in valid_data:
        if pred == winner:
            parts = re.split(r'(?<=[.!?]) +', text)
            for p in parts:
                clean_p = p.strip()
                if clean_p and clean_p not in sentences:
                    # Fuzzy-Check gegen Duplikate (erste 20 Zeichen)
                    if not any(clean_p[:20] == s[:20] for s in sentences):
                        sentences.append(clean_p)
                        
    return " ".join(sentences)

def run_aggregation_and_benchmark():
    """
    Input:  results_1.xlsx, results_2.xlsx, results_3.xlsx, dataset_info.xlsx
    Analysiert alle Runs, aggregiert die Ergebnisse aller Runs (Majority Voting, Aligned Mean und Aligned Justification) und führt eine wissenschaftliche Auswertung (Durchschnitt je Modell inkl. Standardabweichung) durch.
    Output: results_aggregated.xlsx, benchmark_detailed_runs.xlsx, benchmark_scientific_report.xlsx
    """
    print("\n=== Aggregation & Wissenschaftliche Auswertung ===")
    output_datei = 'results_aggregated.xlsx'
    benchmark_file_detailed = 'benchmark_detailed_runs.xlsx'
    benchmark_file_scientific = 'benchmark_scientific_report.xlsx'
    
    dfs = {}
    suffixes = ["_1", "_2", "_3"]
    for suffix in suffixes:
        try:
            df = pd.read_excel(f'results{suffix}.xlsx')
            df['base_model'] = df['model'].str.replace(rf'{suffix}$', '', regex=True)
            cols = ['video_id', 'base_model', 'y_pred', 'probability_fake', 'justification_length', 'justification']
            cols = [c for c in cols if c in df.columns]
            df_clean = df[cols].copy()
            rename_map = {c: f'{c}{suffix}' for c in cols if c not in ['video_id', 'base_model']}
            df_clean = df_clean.rename(columns=rename_map)
            dfs[suffix] = df_clean
        except FileNotFoundError as e: 
            print(f"Fehlt: {e.filename} – Stelle sicher, dass die Datei existiert.")
            return

    df_merged = dfs["_1"]
    for suffix in ["_2", "_3"]:
        df_merged = pd.merge(df_merged, dfs[suffix], on=['video_id', 'base_model'], how='outer')

    print("Joine Ground Truth...")
    try:
        df_info = pd.read_excel(INFO_DATEI)
        df_info['video_id'] = df_info['video_id'].astype(str)
        df_final = pd.merge(df_merged, df_info, on='video_id', how='left')
        df_final['y_true'] = df_final['deepfake'].map({"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1})
    except FileNotFoundError as e: 
            print(f"Fehlt: {e.filename} – Stelle sicher, dass die Datei existiert.")
            return

    # Majority Vote
    pred_cols = [f'y_pred{s}' for s in suffixes]
    votes = df_final[pred_cols]
    vote_count = votes.notna().sum(axis=1)
    vote_sum = votes.fillna(0).sum(axis=1)
    df_final['y_pred'] = (vote_sum > (vote_count / 2)).astype(float)  # float, damit NaN möglich
    df_final.loc[vote_count == 0, 'y_pred'] = np.nan
 
    print("Berechne Aligned Probability & Unified Justification...")
    
    # Wrapper für Probability
    def apply_prob_logic(row):
        # Wir bauen die Listen aus den Spalten
        preds = [row.get(f'y_pred{s}') for s in ["_1", "_2", "_3"]]
        probs = [row.get(f'probability_fake{s}') for s in ["_1", "_2", "_3"]]
        return calculate_aligned_probability(preds, probs)

    # Wrapper für Justification
    def apply_text_logic(row):
        preds = [row.get(f'y_pred{s}') for s in ["_1", "_2", "_3"]]
        texts = [row.get(f'justification{s}') for s in ["_1", "_2", "_3"]]
        return create_unified_justification(preds, texts)

    # Anwenden
    df_final['probability_fake'] = df_final.apply(apply_prob_logic, axis=1)
    df_final['justification'] = df_final.apply(apply_text_logic, axis=1)
    
    # Wortzahl der neuen, kombinierten Begründung
    df_final['justification_length'] = df_final['justification'].apply(get_word_count)
    
    # Speichern
    df_final.to_excel(output_datei, index=False)
    
    # --- Benchmarking ---
    results = []
    df_eval = df_final.dropna(subset=['y_true'])
    eval_targets = [('Run 1', 'y_pred_1', 'probability_fake_1'), ('Run 2', 'y_pred_2', 'probability_fake_2'), 
                    ('Run 3', 'y_pred_3', 'probability_fake_3'), ('Majority Voting', 'y_pred', 'probability_fake')]
    
    for model_name, group in df_eval.groupby('base_model'):
        for run_name, pred_col, prob_col in eval_targets:
            if pred_col not in group.columns: continue
            sub = group.dropna(subset=[pred_col])
            if sub.empty: continue
            
            y_score = sub[prob_col].fillna(0) if prob_col in sub.columns else None
            if y_score is not None and pd.to_numeric(y_score, errors='coerce').max() > 1.0:
                y_score = y_score / 100.0
            
            metrics = calculate_metrics(sub['y_true'], sub[pred_col], y_score)
            metrics.update({'Model': f"{model_name} ({run_name})", 'Base_Model': model_name, 'Type': run_name, 'Videos': len(sub)})
            results.append(metrics)

    df_metrics = pd.DataFrame(results)
    if not df_metrics.empty:
        cols = ['Model', 'Base_Model', 'Type'] + [c for c in df_metrics.columns if c not in ['Model', 'Base_Model', 'Type']]
        df_metrics[cols].to_excel(benchmark_file_detailed, index=False)
        print(f"Bericht erstellt: '{benchmark_file_detailed}'")
        
        runs_only = df_metrics[df_metrics['Type'].isin(['Run 1', 'Run 2', 'Run 3'])].copy()
        if 'ROC AUC' in runs_only.columns: runs_only['ROC AUC'] = pd.to_numeric(runs_only['ROC AUC'], errors='coerce')
        
        metric_cols = [c for c in ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC AUC', 'TN', 'FP', 'FN', 'TP'] if c in runs_only.columns]
        summary = runs_only.groupby('Base_Model')[metric_cols].agg(['mean', 'std'])
        
        report_df = pd.DataFrame(index=summary.index)
        for col in metric_cols:
            fmt = '{:.2f}' if '%' in col or 'ROC' in col else '{:.1f}'
            report_df[col] = summary[col]['mean'].map(fmt.format) + " ± " + summary[col]['std'].fillna(0).map(fmt.format)
        
        report_df.reset_index().to_excel(benchmark_file_scientific, index=False)
        print(f"Bericht erstellt: '{benchmark_file_scientific}'")

def generate_plots():
    """
    Input:  results_aggregated.xlsx, benchmark_scientific_report.xlsx
    Analysiert alle Runs, generiert Plots für jedes Modell und speichert sie im plots/-Ordner.
    Output: Confusion Matrix, ROC Curve, Bar Chart F1 Mean in plots/Run_X/ und plots/Aggregated/
    """
    
    print("\n=== Generiere Plots (Je Run & Aggregiert) ===")  
    
    try:
        df = pd.read_excel('results_aggregated.xlsx')
        df_metrics = pd.read_excel('benchmark_scientific_report.xlsx')
    except FileNotFoundError as e: 
            print(f"Fehlt: {e.filename} – Stelle sicher, dass die Datei existiert.")
            return

    run_configs = [
        ('Run_1', '_1'), 
        ('Run_2', '_2'), 
        ('Run_3', '_3'), 
        ('Aggregated', '') 
    ]

    for run_name, suffix in run_configs:
        pred_col = f'y_pred{suffix}'
        prob_col = f'probability_fake{suffix}'
        
        if pred_col not in df.columns:
            continue
            
        run_folder = os.path.join(base_plot_folder, run_name)
        os.makedirs(run_folder, exist_ok=True)
        
        df_run = df.dropna(subset=['y_true', pred_col])
        print(f" -> Erstelle Plots für {run_name}...")

        # 1. Confusion Matrix
        for model_name, group in df_run.groupby('base_model'):
            cm = confusion_matrix(group['y_true'], group[pred_col], labels=[0, 1])
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                        xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'CM: {model_name} ({run_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, f'CM_{model_name}.png'))
            plt.close()

        # 2. ROC Curve
        plt.figure(figsize=(10, 8))
        has_roc = False
        for model_name, group in df_run.groupby('base_model'):
            if prob_col not in group.columns: continue
            probs = pd.to_numeric(group[prob_col], errors='coerce').fillna(0)
            if probs.max() > 1.0: probs = probs / 100.0
            if len(np.unique(group['y_true'])) > 1:
                fpr, tpr, _ = roc_curve(group['y_true'], probs)
                roc_auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={roc_auc_val:.2f})')
                has_roc = True
        
        if has_roc:
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve Comparison - {run_name}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(run_folder, 'ROC_Curve_Comparison.png'))
        plt.close()

    # 3. Bar Chart F1 Mean
    if not df_metrics.empty and 'F1-Score (%)' in df_metrics.columns:
        df_metrics['F1_Mean'] = df_metrics['F1-Score (%)'].astype(str).apply(lambda x: float(x.split(' ± ')[0]))
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_metrics.sort_values('F1_Mean', ascending=False), x='Base_Model', y='F1_Mean')
        plt.title('Average F1-Score (Mean over Runs)')
        plt.ylabel('F1-Score (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(base_plot_folder, 'Comparison_F1_Score_Mean.png'))
        plt.close()

    plt.close('all')
    
    print(f"Alle Plots gespeichert in '{base_plot_folder}'")

# --- FEATURE IMPORTANCE JE RUN & MODELL ---
def run_feature_importance_analysis():
    """
    Input: results_aggregated.xlsx
    Führt eine Feature Importance Analyse pro Run und Modell, sowie für die aggregierten Daten durch.
    Output: Feature Importance Plots in plots/Run_X/Feature_Importance/
    """
    print("\n=== Feature Importance Analysis (Per Run & Per Model) ===")
    
    try: 
        df = pd.read_excel('results_aggregated.xlsx')
    except FileNotFoundError as e: 
            print(f"Fehlt: {e.filename} – Stelle sicher, dass die Datei existiert.")
            return

    # Welche Runs analysieren wir?
    run_configs = [
        ('Run_1', '_1'), 
        ('Run_2', '_2'), 
        ('Run_3', '_3'), 
        ('Aggregated', '') 
    ]

    for run_name, suffix in run_configs:
        pred_col = f'y_pred{suffix}'
        
        # Falls Spalte nicht da (z.B. Aggregated nicht gewünscht oder Fehler)
        if pred_col not in df.columns:
            continue
            
        # Pfad: plots/Run_1/Feature_Importance/
        save_dir = os.path.join(base_plot_folder, run_name, 'Feature_Importance')
        os.makedirs(save_dir, exist_ok=True)
        
        print(f" -> Berechne Feature Importance für {run_name}...")
        
        # Wir müssen für JEDES Modell separat berechnen
        for model_name, group in df.groupby('base_model'):
            # Daten bereinigen
            sub_df = group.dropna(subset=['y_true', pred_col]).copy()
            if len(sub_df) < 10: # Zu wenige Daten für RF
                continue
                
            # Ziel: Error vorhersagen
            sub_df['error'] = (sub_df[pred_col] != sub_df['y_true']).astype(int)
            
            # Features auswählen

            features = ['video_length']
            cat_features = ['dataset', 'gender', 'deepfake_type'] 
                
            # Features zusammenbauen
            X = sub_df[features].copy()
            
            # One-Hot für Kategorien
            valid_cats = [c for c in cat_features if c in sub_df.columns]
            if valid_cats:
                X = pd.concat([X, pd.get_dummies(sub_df[valid_cats], drop_first=False)], axis=1)
                
            X = X.fillna(0)
            y = sub_df['error']
            
            # Wenn keine Fehler gemacht wurden (oder nur Fehler), kann RF nichts lernen
            if y.nunique() < 2:
                continue

            # Random Forest Training
            try:
                clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, class_weight='balanced')
                clf.fit(X, y)
                
                fi = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
                fi = fi.sort_values('Importance', ascending=False).head(10) # Top 10 Features
                
                # Plotten
                plt.figure(figsize=(8, 6))
                sns.barplot(data=fi, x='Importance', y='Feature')
                plt.title(f'Error Drivers: {model_name} ({run_name})')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                
                # Speichern
                safe_model_name = re.sub(r'[^\w\-_\. ]', '_', model_name)
                plt.savefig(os.path.join(save_dir, f'FI_{safe_model_name}.png'))
                plt.close()
                
            except Exception as e:
                print(f"Fehler bei {model_name} in {run_name}: {e}")

    print("Feature Importance Plots gespeichert.")

def run_standard_correlation_analysis():
    """
    Input: results_aggregated.xlsx
    Führt eine Standard-Korrelationsanalyse pro Run (sowie aggregiert) und Modell durch.
    Output: Bar Chart Correlation mit Error in plots/Run_X/ und plots/Aggregated/
    """
    print("\n=== Standard Correlation Analysis (Clean Bar Chart per Run) ===")
    try: 
        df = pd.read_excel('results_aggregated.xlsx')
    except FileNotFoundError as e: 
            print(f"Fehlt: {e.filename} – Stelle sicher, dass die Datei existiert.")
            return
    
    run_configs = [
        ('Run_1', '_1'), 
        ('Run_2', '_2'), 
        ('Run_3', '_3'), 
        ('Aggregated', '') 
    ]

    for run_name, suffix in run_configs:
        pred_col = f'y_pred{suffix}'
        if pred_col not in df.columns: continue
            
        run_folder = os.path.join(base_plot_folder, run_name)
        os.makedirs(run_folder, exist_ok=True)
        
        df_run = df.dropna(subset=['y_true', pred_col]).copy()
        df_run['error'] = (df_run[pred_col] != df_run['y_true']).astype(int)
        
        exclude_keywords = ['video_id', 'model', 'base_model', 'y_true', 'deepfake', 'justification', 'error', 'grp']
        numeric_cols = df_run.select_dtypes(include=[np.number]).columns.tolist()
        
        clean_numeric_cols = []
        for c in numeric_cols:
            if any(k in c for k in exclude_keywords): continue
            if 'y_pred' in c: continue
            if 'probability' in c: continue
            if 'justification_length' in c:
                if suffix == '' and c == 'justification_length': clean_numeric_cols.append(c)
                elif suffix != '' and c.endswith(suffix): clean_numeric_cols.append(c)
                continue
            clean_numeric_cols.append(c)

        cat_cols = ['dataset', 'gender', 'deepfake_type']
        cat_cols = [c for c in cat_cols if c in df_run.columns]
        
        df_analysis = df_run[clean_numeric_cols + ['error']].copy()
        if cat_cols: df_analysis = pd.concat([df_analysis, pd.get_dummies(df_run[cat_cols], prefix_sep=': ')], axis=1)
        
        if df_analysis.shape[1] > 1:
            corr_with_error = df_analysis.corr(method='pearson')['error'].drop('error').sort_values(ascending=True)
            plt.figure(figsize=(10, 8))
            colors = ['#4e79a7' if x < 0 else '#e15759' for x in corr_with_error.values]
            corr_with_error.plot(kind='barh', color=colors)
            plt.title(f'Correlation with Prediction Error ({run_name})', fontsize=14)
            plt.xlabel('Pearson Correlation Coefficient')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, 'Correlation_BarChart.png'))
            plt.close()
            print(f" -> Korrelation gespeichert: {run_name}")

def run_feature_analysis():
    """
    Input: results_aggregated.xlsx
    Führt eine Feature Analyse DURCHGETRENNT nach Modellen durch.
    Output: feature_analysis.xlsx + Heatmap Plots
    """
    print("\n=== Feature Analyse (Per Model) ===")
    try: 
        df = pd.read_excel('results_aggregated.xlsx')
    except FileNotFoundError: 
        return

    # Sicherheitscheck: Nur Zeilen mit validen Predictions
    df = df.dropna(subset=['y_true', 'y_pred'])

    # Vorbereitung: Video Length in Bins einteilen (global)
    if 'video_length' in df.columns:
        df['video_length_grp'] = pd.cut(df['video_length'], 5, labels=['Very Short','Short','Medium','Long','Very Long'])
    
    results = []
    
    # 1. ÄUSSERE SCHLEIFE: Iteration über jedes Modell
    for model_name, model_df in df.groupby('base_model'):
        
        # 2. INNERE SCHLEIFE: Iteration über die Features
        features_to_analyze = ['dataset', 'gender', 'deepfake_type', 'video_length_grp']
        
        for feat in features_to_analyze:
            if feat not in model_df.columns: 
                continue
            
            # 3. GRUPPEN-ANALYSE: Z.B. "Male" vs "Female" für dieses Modell
            for group_name, grp_data in model_df.groupby(feat, observed=False):
                if len(grp_data) < 5: 
                    continue # Zu wenig Daten für statistische Relevanz
                
                # Metriken berechnen
                m = calculate_metrics(grp_data['y_true'], grp_data['y_pred'])
                
                # Wichtig: Modell-Name mit speichern!
                m.update({
                    'Model': model_name,
                    'Feature': feat,
                    'Group': str(group_name),
                    'Videos': len(grp_data)
                })
                
                # ROC AUC macht hier oft keinen Sinn (zu kleine Gruppen), daher entfernen
                if 'ROC AUC' in m: del m['ROC AUC']
                
                results.append(m)

    if not results:
        print("Keine Ergebnisse für Feature-Analyse gefunden.")
        return

    # Speichern als Excel
    df_results = pd.DataFrame(results)
    
    # Spalten ordnen für Lesbarkeit
    cols = ['Model', 'Feature', 'Group', 'Accuracy (%)', 'F1-Score (%)', 'Videos']
    cols = [c for c in cols if c in df_results.columns] + [c for c in df_results.columns if c not in cols]
    df_results = df_results[cols]
    
    df_results.to_excel('feature_analysis.xlsx', index=False)
    print(f" feature_analysis.xlsx gespeichert.")

    # --- ZUSATZ: Heatmap Visualisierung ---
    # Wir erstellen eine Heatmap für jedes Feature, um Modelle zu vergleichen
    print(" Erstelle Feature-Heatmaps...")
    
    for feat in features_to_analyze:
        # Daten filtern für dieses Feature
        subset = df_results[df_results['Feature'] == feat]
        if subset.empty: continue
        
        # Pivot: Zeilen=Modelle, Spalten=Gruppen (z.B. Male/Female), Wert=F1-Score
        try:
            pivot = subset.pivot(index='Model', columns='Group', values='F1-Score (%)')
            
            plt.figure(figsize=(10, len(pivot)*0.5 + 2))
            sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt='.1f', vmin=0, vmax=100)
            plt.title(f'Model Performance by {feat} (F1-Score)')
            plt.tight_layout()
            plt.savefig(os.path.join(base_plot_folder, f'Feature_Analysis_{feat}.png'))
            plt.close()
        except Exception as e:
            print(f"Konnte Heatmap für {feat} nicht erstellen: {e}")

    print(" Feature Heatmaps gespeichert.")

def run_fairness_analysis():
    """
    Input: results_aggregated.xlsx
    Führt eine Fairness Analyse (Chi2 + Fisher) PRO MODELL durch.
    Output: fairness_analysis.xlsx
    """
    print("\n=== Refined Fairness Analyse (Per Model) ===")
    try: 
        df = pd.read_excel('results_aggregated.xlsx')
    except FileNotFoundError: 
        return
    
    # Bereinigen
    df = df.dropna(subset=['y_true', 'y_pred'])
    results = []

    # 1. ÄUSSERE SCHLEIFE: Iteration über jedes Modell
    for model_name, model_df in df.groupby('base_model'):
        
        # Globale Genauigkeit DIESES Modells (als Referenz)
        model_global_acc = accuracy_score(model_df['y_true'], model_df['y_pred'])

        # 2. Feature Schleife
        for feat in ['gender', 'dataset', 'deepfake_type']:
            if feat not in model_df.columns: continue
            
            # 3. Gruppen Schleife (z.B. Male vs Rest)
            for name in model_df[feat].dropna().unique():
                g = model_df[model_df[feat] == name]
                rest = model_df[model_df[feat] != name]
                
                # Minimum Sample Size Check
                if len(g) < 5 or len(rest) < 5: continue
                
                acc_group = accuracy_score(g['y_true'], g['y_pred'])
                
                # Contingency Table für Chi2/Fisher
                # [[Gruppe Richtig, Gruppe Falsch], [Rest Richtig, Rest Falsch]]
                tbl = [
                    [(g['y_true'] == g['y_pred']).sum(), (g['y_true'] != g['y_pred']).sum()],
                    [(rest['y_true'] == rest['y_pred']).sum(), (rest['y_true'] != rest['y_pred']).sum()]
                ]
                
                # Automatische Wahl des Tests
                if (len(g) < 30) or any(v < 5 for r in tbl for v in r):
                    _, p = fisher_exact(tbl)
                    test = "Fisher"
                else:
                    _, p, _, _ = chi2_contingency(tbl)
                    test = "Chi2"
                
                # Bias-Richtung: Ist die Gruppe besser oder schlechter als der Durchschnitt?
                diff = acc_group - model_global_acc

                results.append({
                    'Model': model_name,           # WICHTIG: Welches Modell?
                    'Feature': feat,
                    'Group': str(name),
                    'Accuracy (%)': round(acc_group * 100, 2),
                    'Diff to Global': round(diff * 100, 2), # Zeigt Bias-Richtung
                    'P-Value': round(p, 4),
                    'Significance': '*' if p < 0.05 else '', # Markierung für schnelle Lesbarkeit
                    'Test': test,
                    'Sample_Size': len(g)
                })
                
    if results:
        # Sortieren: Erst nach Modell, dann nach Signifikanz (P-Value)
        res_df = pd.DataFrame(results).sort_values(by=['Model', 'P-Value'])
        res_df.to_excel('fairness_analysis.xlsx', index=False)
        print(" fairness_analysis.xlsx gespeichert.")
    else:
        print(" Keine signifikanten Fairness-Daten gefunden.")

def run_best_per_family_majority():
    """
    Input: benchmark_detailed_runs.xlsx, results_aggregated.xlsx
    Führt ein Best per Family (GPT, Gemini, Qwen) Majority Voting Ensemble durch. 
    Identifiziert die besten Modelle pro Familie basierend auf dem F1-Score
    Output: results_best_family_ensemble.xlsx, benchmark_ensemble_metrics.xlsx
    """
    print("\n=== Best per Family Ensemble ===")
    try: 
        bench = pd.read_excel('benchmark_detailed_runs.xlsx')
        res = pd.read_excel('results_aggregated.xlsx')
    except FileNotFoundError as e: 
            print(f"Fehlt: {e.filename} – Stelle sicher, dass die Datei existiert.")
            return

    # 1. Beste Modelle identifizieren 
    maj = bench[bench['Type']=='Majority Voting']
    best_models = []
    print("Ausgewählte Modelle für das Ensemble:")
    for fam in ['gpt', 'gemini', 'qwen']:
        sub = maj[maj['Base_Model'].str.lower().str.contains(fam)]
        if not sub.empty:
            best_model = sub.loc[sub['F1-Score (%)'].idxmax(), 'Base_Model']
            best_models.append(best_model)
            print(f" -> {fam.upper()}: {best_model}")

    if len(best_models) < 2: 
        print("Zu wenige Modelle für ein Ensemble gefunden.")
        return

    # 2. Ensemble Logik 
    df_filtered = res[res['base_model'].isin(best_models)].copy()

    def ensemble_logic(group):
            preds = group['y_pred'].tolist()
            probs = group['probability_fake'].tolist()
            texts = group['justification'].tolist()
            
            # Aufruf der universellen Funktionen
            final_prob = calculate_aligned_probability(preds, probs)
            final_text = create_unified_justification(preds, texts)
            
            # Majority Vote (für das Label selbst) noch kurz berechnen
            clean_preds = [p for p in preds if pd.notna(p)]
            if not clean_preds: final_pred = 0
            else: final_pred = 1 if sum(clean_preds) >= (len(clean_preds)/2) else 0

            return pd.Series({
                'y_true': group['y_true'].iloc[0],
                'y_pred_ensemble': final_pred,
                'probability_fake_ensemble': final_prob,
                'justification': final_text
            })

    df_ens = (
    df_filtered
      .groupby('video_id', group_keys=False)
      .apply(ensemble_logic, include_groups=False)
      .reset_index()
)


    # Speichern
    df_ens.to_excel('results_best_family_ensemble.xlsx', index=False)
    print(f" results_best_family_ensemble.xlsx gespeichert (mit aligned Justification).")
    
    # 3. Metriken 
    metrics = calculate_metrics(df_ens['y_true'], df_ens['y_pred_ensemble'])
    df_metrics = pd.DataFrame([metrics])
    df_metrics.insert(0, 'Ensemble_Models', ", ".join(best_models))
    df_metrics.to_excel('benchmark_ensemble_metrics.xlsx', index=False)
    print("\nEnsemble Performance:")
    print(df_metrics.to_string(index=False))

def run_worst_case_extraction(limit=20):
    print("\n=== Worst Case Extraction (Sorted by Confidence) ===")
    try: 
        df = pd.read_excel('results_best_family_ensemble.xlsx')
    except FileNotFoundError: 
        return

    # Fehler finden
    df['error'] = df['y_true'] != df['y_pred_ensemble']
    errors = df[df['error']].copy()
    
    if errors.empty: 
        print(" Keine Fehler im Ensemble gefunden! (Perfektes Modell?)")
        return

    # Video-Pfade laden
    video_map = {}
    for r, _, files in os.walk(VIDEO_SOURCE_PATH):
        for f in files: 
            video_map[os.path.splitext(f)[0]] = os.path.join(r, f)
    
    out = 'worst_cases_analysis'
    os.makedirs(out, exist_ok=True)

    # 1. False Positives (Real, aber als Fake erkannt)
    # Wir wollen die Fälle, wo die Probability SEHR HOCH war (nahe 1.0)
    fp = errors[errors['y_true'] == 0].sort_values('probability_fake_ensemble', ascending=False).head(limit)
    
    # 2. False Negatives (Fake, aber als Real erkannt)
    # Wir wollen die Fälle, wo die Probability SEHR NIEDRIG war (nahe 0.0)
    fn = errors[errors['y_true'] == 1].sort_values('probability_fake_ensemble', ascending=True).head(limit)

    # Kopieren und Berichten
    for label_name, sub_df in [('False_Positives_Real_labeled_as_Fake', fp), ('False_Negatives_Fake_labeled_as_Real', fn)]:
        if sub_df.empty: continue
        
        d = os.path.join(out, label_name)
        os.makedirs(d, exist_ok=True)
        
        print(f" Extrahiere {len(sub_df)} {label_name}...")
        
        for idx, row in sub_df.iterrows():
            vid = str(row['video_id'])
            prob = row.get('probability_fake_ensemble', 0)
            
            if vid in video_map:
                src = video_map[vid]
                # Dateiname inkl. Probability für einfache Analyse: "99perc_video123.mp4"
                new_name = f"{int(prob*100)}perc_{os.path.basename(src)}"
                dst = os.path.join(d, new_name)
                shutil.copy2(src, dst)

    # Report speichern
    pd.concat([fp, fn]).to_excel(os.path.join(out, 'worst_cases_report.xlsx'), index=False)
    print(f" worst_cases_report.xlsx gespeichert.")

def run_justification_deep_analysis():
    """
    Input: results_aggregated.xlsx
    Output: 
      1. Similarity Heatmap (Global)
      2. Keyword Plots (Per Model & Per Condition)
    """
    print("\n=== Justification Deep Analysis (Per Model Keywords) ===")
    try: 
        df = pd.read_excel('results_aggregated.xlsx')
    except: 
        return

    # 1. SEMANTISCHE ÄHNLICHKEIT (Bleibt Global - Vergleich der Modelle)
    print(" -> Erstelle Similarity Heatmap...")
    model_data = []
    for model_name, group in df.groupby('base_model'):
        # Wir nehmen max 100 Texte pro Modell für die Similarity
        txt = " ".join(group['justification'].dropna().astype(str).tolist()[:100])
        if len(txt) > 20:
            model_data.append({'name': model_name, 'text': txt})
    
    if len(model_data) > 1:
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform([m['text'] for m in model_data])
        sim = cosine_similarity(matrix)
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim, annot=True, xticklabels=[m['name'] for m in model_data], yticklabels=[m['name'] for m in model_data], cmap='Blues')
        plt.title('Semantic Similarity between Model Justifications')
        plt.tight_layout()
        plt.savefig(os.path.join(base_plot_folder, 'Justification_Similarity_Heatmap.png'))
        plt.close()

    # 2. KEYWORD EXTRAKTION (NEU: Pro Modell getrennt!)
    print(" -> Erstelle Keyword Plots pro Modell...")
    
    # Ordner für Keyword-Plots erstellen, damit es übersichtlich bleibt
    kw_folder = os.path.join(base_plot_folder, 'Keywords_Per_Model')
    os.makedirs(kw_folder, exist_ok=True)

    for model_name, model_df in df.groupby('base_model'):
        # Wir untersuchen nur Modelle, die genug Daten haben
        if len(model_df) < 10: continue

        subsets = {
            'TP_Fake': model_df[(model_df['y_true'] == 1) & (model_df['y_pred'] == 1)],
            'TN_Real': model_df[(model_df['y_true'] == 0) & (model_df['y_pred'] == 0)]
        }

        for condition, sub in subsets.items():
            all_texts = sub['justification'].dropna().astype(str).tolist()
            
            # Mindestens 5 Texte nötig für eine Analyse
            if len(all_texts) > 5:
                try:
                    # Bigramme (2 Wörter) extrahieren
                    cv = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=10)
                    counts = cv.fit_transform(all_texts)
                    words = cv.get_feature_names_out()
                    freqs = counts.sum(axis=0).A1
                    
                    df_kw = pd.DataFrame({'Feature': words, 'Frequency': freqs}).sort_values('Frequency', ascending=False)
                    
                    plt.figure(figsize=(8, 5))
                    sns.barplot(
                        data=df_kw, 
                        x='Frequency', 
                        y='Feature', 
                        hue='Feature', 
                        palette='magma' if 'Fake' in condition else 'viridis', 
                        legend=False
                    )
                    
                    # Titel enthält jetzt den Modellnamen!
                    plt.title(f'{model_name}: {condition.replace("_", " ")}')
                    plt.tight_layout()
                    
                    # Dateiname: plots/Keywords_Per_Model/GPT-4o_TP_Fake.png
                    safe_name = re.sub(r'[^\w\-_\. ]', '_', model_name)
                    plt.savefig(os.path.join(kw_folder, f'{safe_name}_{condition}.png'))
                    plt.close()
                except ValueError:
                    # Passiert, wenn Texte nur Stopwords enthalten oder leer sind
                    continue

    print(" Keyword Plots gespeichert.")

if __name__ == "__main__":
    for s in ["_1", "_2", "_3"]: run_analysis(s)
    run_aggregation_and_benchmark()
    generate_plots() 
    run_feature_importance_analysis()
    run_standard_correlation_analysis() 
    run_feature_analysis()
    run_fairness_analysis()
    run_best_per_family_majority()
    run_worst_case_extraction()
    run_justification_deep_analysis()
    print("\n=== Fertig ===")