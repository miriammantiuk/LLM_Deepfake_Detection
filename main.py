# ---------------------------------------------------------
# main.py — Entry point for the deepfake detection evaluation pipeline
# ---------------------------------------------------------
import pandas as pd

from config import RUN_SUFFIXES, INFO_DATEI
from evaluation import (
    run_analysis,
    run_aggregation_and_benchmark,
    run_baseline_distribution_analysis,
    load_master_data,
    run_global_baseline_roc_analysis,
    run_family_variant_roc_comparison,
    generate_plots,
    run_feature_importance_analysis,
    run_feature_analysis,
    run_fairness_analysis,
    run_intra_model_consistency_check,
    run_inter_model_similarity,
    run_justification_deep_analysis,
    run_best_per_family_ensemble,
    run_worst_case_extraction,
)

if __name__ == "__main__":

    # Raw Data Extraction & Aggregation
    for s in RUN_SUFFIXES:
        run_analysis(s)
    run_aggregation_and_benchmark()

    df_ground_truth = pd.read_excel(INFO_DATEI)
    run_baseline_distribution_analysis(df_ground_truth)

    # Load master dataset once and pass to all analysis functions
    df_master = load_master_data()

    if df_master is not None:

        # --- GLOBAL / AGGREGATED ANALYSES ---
        run_global_baseline_roc_analysis(df_master)
        run_family_variant_roc_comparison(df_master)

        # --- PER-RUN ANALYSES ---
        generate_plots(df_master)
        run_feature_importance_analysis(df_master)
        run_feature_analysis(df_master)
        run_fairness_analysis(df_master)
        run_intra_model_consistency_check(df_master)
        run_inter_model_similarity(df_master)
        run_justification_deep_analysis(df_master)
        run_best_per_family_ensemble(df_master)
        run_worst_case_extraction(df_master)

    print("\n=== Finished ===")
