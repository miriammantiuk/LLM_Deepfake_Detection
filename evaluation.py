import pandas as pd
import json
import glob
import os
import ast
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, roc_auc_score, 
                             roc_curve, r2_score)
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# --- KONFIGURATION ---
JSON_FOLDER = '.'               # Ordner mit den JSON-Dateien (Ergebnisse)
INFO_DATEI = 'dataset_info.xlsx' # Excel mit Ground Truth
VIDEO_SOURCE_PATH = r'data\processed'  # Pfad zu den Original-Videos
# ---------------------

# Plotting Style global setzen
sns.set_style("whitegrid")

def calculate_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
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
        'ROC AUC': round(roc_auc, 4) if roc_auc is not None else 'N/A'
    }

def safe_parse_list(val): 
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []

def get_word_count(val):
    """Berechnet die Anzahl der Wörter, egal ob Liste oder String."""
    if isinstance(val, list):
        # Liste zu einem String verbinden und splitten
        text = " ".join(str(v) for v in val)
        return len(text.split())
    elif isinstance(val, str):
        # String splitten
        return len(val.split())
    return 0

def run_analysis(suffix=""):
    """
    Liest NUR JSON ein, standardisiert IDs und speichert rohe Ergebnisse.
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
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            df_temp = pd.DataFrame(data)
            df_temp['model'] = model_name 
            df_temp['video_id'] = df_temp['video_id'].astype(str).str.replace(r'.mp4|.MP4', '', regex=True)
            
            # NEU: Justification Length basierend auf WORT-ANZAHL
            if 'justification' in df_temp.columns:
                df_temp['justification_length'] = df_temp['justification'].apply(get_word_count)
            else:
                df_temp['justification_length'] = 0
            
            all_data.append(df_temp)
            print(f"Geladen: {filename} ({len(df_temp)} Videos)")
            
        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

    if not all_data:
        return

    df_gesamt = pd.concat(all_data, ignore_index=True)

    # Mapping Pred
    label_mapping = {"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1}
    df_gesamt['y_pred'] = df_gesamt['assessment'].map(label_mapping)

    df_gesamt.to_excel(output_datei, index=False)
    print(f"Rohe Ergebnisse für {suffix} gespeichert als '{output_datei}'.")


def run_majority_analysis():
    print("\n=== Majority Voting & Globale Auswertung ===")
    output_datei = 'results_majority.xlsx'
    benchmark_datei = 'benchmark_results_majority.xlsx'
    plots_dir = 'plots_majority'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Laden der 3 Durchgänge
    dfs = {}
    suffixes = ["_1", "_2", "_3"]
    
    for suffix in suffixes:
        try:
            df = pd.read_excel(f'results{suffix}.xlsx')
            df['base_model'] = df['model'].str.replace(rf'{suffix}$', '', regex=True)
            
            # --- UPDATE: 'indicators' hinzufügen ---
            cols = ['video_id', 'base_model', 'y_pred', 'probability_fake', 'justification_length']
            
            # Prüfen ob indicators oder justification existieren
            if 'justification' in df.columns: cols.append('justification')
            if 'indicators' in df.columns: cols.append('indicators') # <--- WICHTIG

            cols = [c for c in cols if c in df.columns]
            df_clean = df[cols].copy()
            
            # Umbenennen
            rename_map = {
                'y_pred': f'y_pred{suffix}',
                'probability_fake': f'probability_fake{suffix}',
                'justification': f'justification{suffix}',
                'justification_length': f'justification_length{suffix}',
                'indicators': f'indicators{suffix}' # <--- WICHTIG
            }
            df_clean = df_clean.rename(columns=rename_map)
            dfs[suffix] = df_clean
            
        except FileNotFoundError:
            print(f"Fehler: results{suffix}.xlsx nicht gefunden.")
            return

    # 2. Mergen (unverändert)
    df_merged = dfs["_1"]
    for suffix in ["_2", "_3"]:
        df_merged = pd.merge(df_merged, dfs[suffix], on=['video_id', 'base_model'], how='outer')

    # 3. Ground Truth (unverändert)
    print("Joine Dataset Infos (Ground Truth)...")
    try:
        df_info = pd.read_excel(INFO_DATEI)
        df_info['video_id'] = df_info['video_id'].astype(str)
        df_final = pd.merge(df_merged, df_info, on='video_id', how='left')
        label_mapping = {"Real": 0, "real": 0, "Fake": 1, "fake": 1, 0: 0, 1: 1}
        df_final['y_true'] = df_final['deepfake'].map(label_mapping)
    except FileNotFoundError:
        print(f"Kritischer Fehler: {INFO_DATEI} nicht gefunden.")
        return

    # 4. Majority Vote (unverändert)
    pred_cols = ['y_pred_1', 'y_pred_2', 'y_pred_3']
    valid_pred_cols = [c for c in pred_cols if c in df_final.columns]
    if not valid_pred_cols: return

    votes = df_final[valid_pred_cols].sum(axis=1)
    df_final['y_pred'] = votes.apply(lambda x: 1 if x >= 2 else 0)
    
    # Averages
    prob_cols = [f'probability_fake{s}' for s in suffixes]
    valid_prob_cols = [c for c in prob_cols if c in df_final.columns]
    if valid_prob_cols: df_final['probability_fake'] = df_final[valid_prob_cols].mean(axis=1)
    
    len_cols = [f'justification_length{s}' for s in suffixes]
    valid_len_cols = [c for c in len_cols if c in df_final.columns]
    if valid_len_cols: df_final['justification_length'] = df_final[valid_len_cols].mean(axis=1)

    # --- UPDATE: Indicators zusammenführen ---
    # Wir nehmen die Indicators aus Lauf 1 als primäre Quelle (oder füllen mit 2/3 auf)
    if 'indicators_1' in df_final.columns:
        df_final['indicators'] = df_final['indicators_1']
    elif 'indicators_2' in df_final.columns:
        df_final['indicators'] = df_final['indicators_2']
    
    df_final.to_excel(output_datei, index=False)
    print(f"Zusammengefasste Daten (inkl. Indicators) gespeichert in '{output_datei}'.")
    
    # ... (Rest der Funktion: Benchmarking Code bleibt gleich wie vorher) ...
    # (Ich kürze den Benchmark-Teil hier ab, da er unverändert ist, füge ihn aber 
    # im finalen Code-Block unten vollständig ein, damit du keine Lücken hast.)
    results = []
    df_eval = df_final.dropna(subset=['y_true'])
    eval_targets = [
        ('Run 1', 'y_pred_1', 'probability_fake_1'),
        ('Run 2', 'y_pred_2', 'probability_fake_2'),
        ('Run 3', 'y_pred_3', 'probability_fake_3'),
        ('Majority', 'y_pred', 'probability_fake')
    ]
    for model_name, group in df_eval.groupby('base_model'):
        for run_name, pred_col, prob_col in eval_targets:
            if pred_col not in group.columns: continue
            sub_group = group.dropna(subset=[pred_col])
            if sub_group.empty: continue
            y_true = sub_group['y_true']
            y_pred = sub_group[pred_col]
            y_score = None
            if prob_col in sub_group.columns:
                probs = sub_group[prob_col].fillna(0)
                y_score = probs / 100.0 if probs.mean() > 1.0 else probs
            metrics = calculate_metrics(y_true, y_pred, y_score)
            metrics['Model'] = f"{model_name} ({run_name})"
            metrics['Base_Model'] = model_name
            metrics['Type'] = run_name
            metrics['Videos'] = len(sub_group)
            results.append(metrics)

    df_metrics = pd.DataFrame(results)
    if not df_metrics.empty:
        df_metrics = df_metrics.sort_values(by=['Base_Model', 'F1-Score (%)'], ascending=[True, False])
        cols = ['Model', 'Base_Model', 'Type'] + [c for c in df_metrics.columns if c not in ['Model', 'Base_Model', 'Type']]
        df_metrics = df_metrics[cols]
        df_metrics.to_excel(benchmark_datei, index=False)
        print("\n--- Benchmark Ergebnisse ---")
        print(df_metrics.to_string(index=False))


def run_feature_analysis():
    print("\n=== Feature Analyse ===")
    try:
        df = pd.read_excel('results_majority.xlsx')
    except FileNotFoundError:
        print("results_majority.xlsx nicht gefunden.")
        return
        
    df_eval = df.dropna(subset=['y_true', 'y_pred'])
    features = ['dataset', 'gender', 'deepfake_type', 'video_length']
    results = []

    for feature in features:
        if feature not in df_eval.columns: continue
        
        if feature == 'video_length':
            df_eval['temp_group'] = pd.cut(df_eval['video_length'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        else:
            df_eval['temp_group'] = df_eval[feature]

        for group_name, group in df_eval.groupby('temp_group', observed=False):
            if len(group) < 10: continue
            
            metrics = calculate_metrics(group['y_true'], group['y_pred'])
            metrics['Feature'] = feature
            metrics['Group'] = str(group_name)
            metrics['Videos'] = len(group)
            del metrics['ROC AUC'] 
            results.append(metrics)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        cols = ['Feature', 'Group'] + [c for c in df_results.columns if c not in ['Feature', 'Group']]
        df_results = df_results[cols]
        df_results.to_excel('feature_analysis.xlsx', index=False)
        print(df_results.to_string(index=False))

def run_fairness_analysis():
    print("\n=== Fairness Analyse (mit statistischer Signifikanz) ===")
    try:
        df = pd.read_excel('results_majority.xlsx')
    except FileNotFoundError:
        print("results_majority.xlsx nicht gefunden.")
        return
    
    df_eval = df.dropna(subset=['y_true', 'y_pred'])
    features = ['gender', 'dataset', 'deepfake_type']
    results = []
    
    global_acc = accuracy_score(df_eval['y_true'], df_eval['y_pred'])
    
    for feature in features:
        if feature not in df_eval.columns: continue
        groups = df_eval[feature].unique()
        
        for group_name in groups:
            group_df = df_eval[df_eval[feature] == group_name]
            rest_df = df_eval[df_eval[feature] != group_name]
            
            if len(group_df) < 10 or len(rest_df) < 10: continue
            
            acc_group = accuracy_score(group_df['y_true'], group_df['y_pred'])
            ppr_group = (group_df['y_pred'] == 1).mean()
            ppr_rest = (rest_df['y_pred'] == 1).mean()
            
            group_correct = (group_df['y_true'] == group_df['y_pred']).sum()
            group_error = len(group_df) - group_correct
            rest_correct = (rest_df['y_true'] == rest_df['y_pred']).sum()
            rest_error = len(rest_df) - rest_correct
            
            contingency_table = [[group_correct, group_error], [rest_correct, rest_error]]
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            
            results.append({
                'Feature': feature,
                'Group': str(group_name),
                'Videos': len(group_df),
                'Accuracy (%)': round(acc_group * 100, 2),
                'Global Accuracy (%)': round(global_acc * 100, 2),
                'Diff (%)': round((acc_group - global_acc) * 100, 2),
                'P-Value': round(p_val, 4),
                'Significant (p<0.05)': 'YES' if p_val < 0.05 else 'No',
                'Disparate Impact': round(ppr_group / ppr_rest, 2) if ppr_rest > 0 else 'N/A'
            })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by='P-Value', ascending=True)
        output_file = 'fairness_analysis_sig.xlsx'
        df_results.to_excel(output_file, index=False)
        print(f"Fairness Analyse gespeichert in '{output_file}'")
    else:
        print("Keine Ergebnisse für Fairness-Analyse.")

from sklearn.feature_extraction.text import CountVectorizer

import re

def run_correlation_regression_analysis():
    print("\n=== Correlation Analysis (Text Mining form Justification) ===")
    try:
        df = pd.read_excel('results_majority.xlsx')
    except FileNotFoundError:
        print("results_majority.xlsx nicht gefunden.")
        return
    
    df_eval = df.dropna(subset=['y_true', 'y_pred'])
    
    # 1. Error-Spalte (1 = Fehler, 0 = Korrekt)
    df_eval['error'] = (df_eval['y_pred'] != df_eval['y_true']).astype(int)
    
    # 2. Basis-Features
    numeric_cols = ['video_length', 'error', 'justification_length', 'probability_fake']
    available_numeric = [c for c in numeric_cols if c in df_eval.columns]
    
    # 3. Kategorische Features (One-Hot)
    cat_features = ['dataset', 'gender', 'deepfake_type']
    available_cats = [c for c in cat_features if c in df_eval.columns]
    
    # Kopie für die Korrelation
    df_corr = df_eval[available_numeric + available_cats].copy()
    
    if available_cats:
        df_corr = pd.get_dummies(df_corr, columns=available_cats, dummy_na=False)

    # 4. TEXT MINING: Verbesserte Extraktion
    text_cols_created = [] # Liste zum Speichern der Text-Spaltennamen
    
    if 'justification' in df_eval.columns:
        print("Extrahiere Keywords aus 'justification'...")
        
        # Verbesserte Clean-Funktion
        def clean_text(val):
            if isinstance(val, list): 
                # Falls Liste von Listen oder Strings, alles flachklopfen
                text = " ".join(str(v) for v in val)
            elif pd.isna(val): 
                return ""
            else:
                text = str(val)
            
            # Bereinigung: Nur Buchstaben behalten, alles lowercase
            text = re.sub(r'[^a-zA-Z\s]', '', text) 
            return text.lower()

        text_data = df_eval['justification'].apply(clean_text)
        
        # PARAMETER ANPASSUNG:
        # ngram_range=(1, 2): Findet "blur" UND "motion blur"
        # min_df=2: Wort muss in mindestens 2 Videos vorkommen (statt 5), um erfasst zu werden
        # max_features=30: Mehr Wörter zulassen
        try:
            vec = CountVectorizer(stop_words='english', max_features=30, min_df=2, ngram_range=(1, 2)) 
            X = vec.fit_transform(text_data)
            
            keywords = vec.get_feature_names_out()
            print(f"Gefundene Top-Keywords ({len(keywords)}): {keywords}")
            
            # Prefix 'txt_' hinzufügen
            text_cols_created = [f"txt_{k}" for k in keywords]
            
            X_df = pd.DataFrame(X.toarray(), columns=text_cols_created, index=df_eval.index)
            
            # Binär machen (Wort kommt vor: 1, sonst 0)
            X_df = (X_df > 0).astype(int)
            
            df_corr = pd.concat([df_corr, X_df], axis=1)
            
        except ValueError as e:
            print(f"Text Mining übersprungen: {e}")

    # 5. Korrelation
    # Entferne Spalten ohne Varianz (nur 0 oder nur 1)
    df_corr = df_corr.loc[:, df_corr.std() > 0]

    if df_corr.empty or df_corr.shape[1] < 2:
        print("Zu wenige Daten für Korrelation.")
        return

    corr_matrix = df_corr.corr(method='spearman')

    # 6. Plotting (Nur Top 20 Korrelationen zum Error plotten, damit es lesbar bleibt)
    # Wir filtern die Matrix auf Korrelationen mit 'error'
    error_corr = corr_matrix['error'].drop('error') # Self-correlation droppen
    top_corr_features = error_corr.abs().sort_values(ascending=False).head(20).index.tolist()
    
    # Kleine Heatmap nur für die relevantesten Faktoren
    if top_corr_features:
        cols_to_plot = ['error'] + top_corr_features
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df_corr[cols_to_plot].corr(method='spearman')[['error']].sort_values(by='error', ascending=False),
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            center=0, 
            vmin=-0.5, vmax=0.5 # Skala begrenzen für bessere Sichtbarkeit
        )
        plt.title('Top Correlations with Prediction Error')
        plt.tight_layout()
        plt.savefig('extended_plots/correlation_error_factors.png')
        plt.close()
        print("Fehler-Korrelations-Chart gespeichert in 'extended_plots/correlation_error_factors.png'.")

    # 7. GETRENNTE ANALYSE 
    print("\n--- DETAILLIERTE FEHLER-ANALYSE ---")
    
    # A) Metadaten
    meta_cols = [c for c in corr_matrix.index if not c.startswith('txt_') and c != 'error']
    if meta_cols:
        print("\n[Metadaten] Was korreliert mit Fehlern?")
        print(corr_matrix.loc[meta_cols, 'error'].sort_values(ascending=False).head(5).to_string())

    # B) Keywords (Text)
    txt_cols = [c for c in corr_matrix.index if c.startswith('txt_')]
    if txt_cols:
        print("\n[Justification Keywords] Welche Wörter deuten auf Fehler hin?")
        # Positive Korrelation: Wort da -> Fehler passiert eher (z.B. Unsicherheitswörter)
        print("--> Wort vorhanden = EHER FALSCH:")
        print(corr_matrix.loc[txt_cols, 'error'].sort_values(ascending=False).head(5).to_string())
        
        print("\n--> Wort vorhanden = EHER RICHTIG:")
        # Negative Korrelation: Wort da -> Fehler passiert seltener (Modell hat es verstanden)
        print(corr_matrix.loc[txt_cols, 'error'].sort_values(ascending=True).head(5).to_string())
    else:
        print("\n[Justification Keywords] Keine Text-Korrelationen gefunden (Wörter zu selten oder keine Varianz).")

def run_best_per_family_majority():
    print("\n=== Best per Family Ensemble ===")
    try:
        df_bench = pd.read_excel('benchmark_results_majority.xlsx')
        df_res = pd.read_excel('results_majority.xlsx')
    except FileNotFoundError:
        print("Dateien fehlen. Bitte erst Majority Analysis laufen lassen.")
        return

    df_bench_maj = df_bench[df_bench['Type'] == 'Majority']
    families = {'gpt': 'gpt', 'gemini': 'gemini', 'qwen': 'qwen'}
    
    best_models = []
    for fam_name, keyword in families.items():
        fam_df = df_bench_maj[df_bench_maj['Base_Model'].str.lower().str.contains(keyword)]
        if not fam_df.empty:
            best = fam_df.loc[fam_df['F1-Score (%)'].idxmax()]
            best_models.append(best['Base_Model'])
            print(f"Beste {fam_name}: {best['Base_Model']} (F1: {best['F1-Score (%)']}%)")

    if len(best_models) < 2:
        print("Zu wenige Familien-Modelle für ein Ensemble.")
        return

    df_filtered = df_res[df_res['base_model'].isin(best_models)]
    
    ensemble_data = []
    for vid, group in df_filtered.groupby('video_id'):
        preds = group['y_pred'].tolist()
        if len(preds) > 0:
            final_pred = 1 if sum(preds) >= (len(preds)/2) else 0
            if 'justification_1' in group.columns:
                 justs = group['justification_1'].tolist()
            else:
                 justs = []
            
            ensemble_data.append({
                'video_id': vid,
                'y_true': group['y_true'].iloc[0],
                'y_pred_ensemble': final_pred,
                'justification': justs
            })
            
    df_ensemble = pd.DataFrame(ensemble_data)
    df_ensemble.to_excel('results_best_family_ensemble.xlsx', index=False)
    
    metrics = calculate_metrics(df_ensemble['y_true'], df_ensemble['y_pred_ensemble'])
    print("\nEnsemble Metrics:")
    print(metrics)

def run_worst_case_extraction(limit=20):
    print("\n=== Worst Case Extraction & Analysis ===")
    
    input_file = 'results_best_family_ensemble.xlsx'
    output_dir = 'worst_cases_analysis'
    
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Datei '{input_file}' nicht gefunden. Bitte erst Ensemble laufen lassen.")
        return

    if 'y_true' not in df.columns or 'y_pred_ensemble' not in df.columns:
        print("Spalten fehlen im Ensemble-File.")
        return

    df_errors = df[df['y_true'] != df['y_pred_ensemble']].copy()
    if df_errors.empty:
        print("Keine Fehler im Ensemble gefunden!")
        return

    print(f"Gefunden: {len(df_errors)} Fehler insgesamt.")

    fn_df = df_errors[(df_errors['y_true'] == 1) & (df_errors['y_pred_ensemble'] == 0)]
    fp_df = df_errors[(df_errors['y_true'] == 0) & (df_errors['y_pred_ensemble'] == 1)]

    print(f"Indiziere Videos in '{VIDEO_SOURCE_PATH}' (inkl. Unterordner)...")
    video_map = {}
    
    if not os.path.exists(VIDEO_SOURCE_PATH):
        print(f"FEHLER: Quellpfad '{VIDEO_SOURCE_PATH}' existiert nicht!")
        return

    file_count = 0
    for root, dirs, files in os.walk(VIDEO_SOURCE_PATH):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                base_name = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                video_map[base_name] = full_path
                file_count += 1
    
    print(f"Index erstellt: {file_count} Videos gefunden.")

    os.makedirs(os.path.join(output_dir, 'false_negatives'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'false_positives'), exist_ok=True)

    def copy_files_recursive(subset_df, subfolder, limit_count):
        count = 0
        missing = 0
        for _, row in subset_df.head(limit_count).iterrows():
            vid_id = str(row['video_id'])
            if vid_id in video_map:
                src_file = video_map[vid_id]
                filename = os.path.basename(src_file)
                dst_file = os.path.join(output_dir, subfolder, filename)
                try:
                    shutil.copy2(src_file, dst_file)
                    count += 1
                except Exception as e:
                    print(f"Fehler: {e}")
            else:
                missing += 1
        return count, missing

    print(f"Kopiere bis zu {limit} Videos pro Kategorie...")
    n_fn, miss_fn = copy_files_recursive(fn_df, 'false_negatives', limit)
    n_fp, miss_fp = copy_files_recursive(fp_df, 'false_positives', limit)
    print(f"Kopiert: FN={n_fn}, FP={n_fp}")

    report_path = os.path.join(output_dir, 'worst_cases_report.xlsx')
    df_report = pd.concat([fn_df, fp_df])
    df_report['error_type'] = df_report.apply(
        lambda x: 'False Negative' if x['y_true'] == 1 else 'False Positive', axis=1
    )
    df_report['found_path'] = df_report['video_id'].map(video_map)
    df_report.to_excel(report_path, index=False)
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='error_type', data=df_report, palette='Set2')
    plt.title('Verteilung der Fehler-Typen')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()

def run_justification_similarity_analysis():
    print("\n=== Justification Similarity ===")
    try:
        df = pd.read_excel('results_best_family_ensemble.xlsx')
    except FileNotFoundError:
        return
        
    if 'justification' not in df.columns: return

    similarities = []
    vectorizer = TfidfVectorizer(stop_words='english')

    for _, row in df.iterrows():
        justs = safe_parse_list(row['justification'])
        texts = []
        for j in justs:
            if isinstance(j, list): texts.append(" ".join(str(x) for x in j))
            elif isinstance(j, str): texts.append(j)
            
        if len(texts) < 2: continue
        
        try:
            tfidf = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf)
            upper_indices = np.triu_indices_from(sim_matrix, k=1)
            if len(upper_indices[0]) > 0:
                avg_sim = np.mean(sim_matrix[upper_indices])
                similarities.append(avg_sim)
        except ValueError:
            continue

    if similarities:
        print(f"Avg Similarity: {np.mean(similarities):.4f}")
        plt.figure(figsize=(8,6))
        sns.histplot(similarities, kde=True, color='purple')
        plt.title('Distribution of Justification Similarity')
        plt.savefig('extended_plots/justification_similarity.png')
        plt.close()

if __name__ == "__main__":
    os.makedirs('extended_plots', exist_ok=True)

    # 1. Einzel-Analysen (Nur Datenvorbereitung)
    for suffix in ["_1", "_2", "_3"]:
        run_analysis(suffix)
    
    # 2. Majority Merge (Hier passiert die Magie und der GT-Merge)
    run_majority_analysis()
    
    # 3. Vertiefende Analysen
    run_feature_analysis()
    run_fairness_analysis()
    run_correlation_regression_analysis()
    
    # 4. Ensemble
    run_best_per_family_majority()
    
    # 5. Extraction
    run_worst_case_extraction(limit=20)
    
    run_justification_similarity_analysis()
    
    print("\n=== Pipeline erfolgreich abgeschlossen! ===")