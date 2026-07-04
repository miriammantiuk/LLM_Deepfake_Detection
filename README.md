# LLM-Based Zero-Shot Deepfake Video Detection

This repository contains the inference pipelines and evaluation code for a Zero-Shot Deepfake Video Detection study using Large Language Models.

Five multimodal LLMs (Gemini 3 Flash, GPT-5.2, Qwen 3.5, Kimi k2.5, Seed 2.0 Lite) are evaluated
on a balanced dataset of N=50 videos across four prompt variants (Baseline, +Indicators, +Thinking, +Indicators+Thinking).

---

## Repository Structure

```
.
├── Gemini.py              # Inference — Gemini 3 Flash (Google Files API)
├── Qwen.py                # Inference — Qwen 3.5 (DashScope, fps=2)
├── kimi.py                # Inference — Kimi k2.5 (Moonshot Files API)
├── seed.py                # Inference — Seed 2.0 Lite (OpenRouter, Base64)
├── main.py                # Entry point: runs the full evaluation pipeline
├── evaluation.py          # All analysis functions (benchmark, fairness, NLP, …)
├── config.py              # Global constants (model names, paths, plot style)
├── metrics.py             # Core metric functions (accuracy, F1, ROC AUC, …)
├── dataset_info.xlsx      # Ground truth labels and video metadata
├── <model>_<run>.json     # Inference outputs, one file per model variant × run
├── data/
│   ├── processed/         # Input videos (not included in repository)
│   └── raw/               # Raw source material (not included in repository)
├── frames/                # Extracted video frames, organised by run (auto-created)
│   ├── Run_1/
│   ├── Run_2/
│   └── Run_3/
├── results/               # Excel, LaTeX, and summary outputs (auto-created)
├── plots/                 # Generated figures, organised by analysis type (auto-created)
│   ├── Run_1/
│   ├── Run_2/
│   ├── Run_3/
│   ├── Aggregated/
│   ├── Alle_Runs/
│   ├── Baseline_Distributions/
│   └── Family_Comparison/
├── requirements.txt
└── README.md
```

> **Note:** GPT-5.2 was evaluated manually via the ChatGPT web interface due to the absence
> of native video support in the API at the time of data collection.
> Results were saved as JSON files following the same format as the other models.

---

## Installation

```bash
git clone https://github.com/miriammantiuk/LLM_Deepfake_Detection.git
cd LLM_Deepfake_Detection
pip install -r requirements.txt
```

Requires **Python 3.11**.

---

## Environment Variables

Set the following API keys as environment variables before running the inference scripts:

| Variable | Used by |
|---|---|
| `GEMINI_API_KEY` | Gemini.py |
| `QWEN_API_KEY` | Qwen.py |
| `MOONSHOT_API_KEY` | kimi.py |
| `OPENROUTER_API_KEY` | seed.py |

---

## Dataset Structure

Place the videos under `data/processed/`.

A `dataset_info.xlsx` file with ground truth labels and metadata
(columns: `video_id`, `label`, `dataset`, `gender`, `video_length`,
`deepfake_category`, `deepfake_type`, `audio`) must be placed in the project root.

---

## Running Inference

Each script processes all `.mp4` files under `data/processed/` and saves results
to a JSON file in the project root. Already-processed videos are skipped automatically (resume-safe).

Configure the desired variant at the top of each script:

```python
thinking_budget  = 0          # Gemini: -1 = dynamic reasoning, 0 = disabled
enable_thinking  = False      # Qwen:   True / False
thinking_type    = "disabled" # Kimi:   "enabled" / "disabled"
thinking_type    = False      # Seed:   True / False

indicators_type  = "disabled" # all scripts: "enabled" / "disabled"
```

Then run:

```bash
python Gemini.py
python Qwen.py
python kimi.py
python seed.py
```

For multiple runs, rename the output file and add `_2`, `_3`, etc. to the filename before starting the next run.

Output files are named automatically, e.g. `gemini-3-flash-preview_1.json`,
`gemini-3-flash-preview_thinking_1.json`, etc.

---

## Running the Evaluation Pipeline

```bash
python main.py
```

The pipeline expects JSON result files in the project root, one per model variant and run
(suffix `_1`, `_2`, `_3`):

```
gemini-3-flash-preview_1.json
gemini-3-flash-preview_2.json
gemini-3-flash-preview_3.json
gemini-3-flash-preview_thinking_1.json
...
```

Individual analysis steps can be toggled by commenting/uncommenting the corresponding
function calls in `main.py`.

The pipeline runs the following analyses:

| Step | Function | Description |
|---|---|---|
| 1 | `run_analysis` | JSON parsing, label extraction, per-run Excel export |
| 2 | `run_aggregation_and_benchmark` | Mean ± std across runs, LaTeX benchmark tables |
| 3 | `run_baseline_distribution_analysis` | Dataset metadata distributions |
| 4 | `run_global_baseline_roc_analysis` | Averaged ROC curves (per-run interpolation) |
| 5 | `run_family_variant_roc_comparison` | ROC per model family across prompt variants |
| 6 | `generate_plots` | F1 heatmaps, confusion matrices, similarity plots |
| 7 | `run_feature_importance_analysis` | Stacked bar charts: metadata vs. model outcome |
| 8 | `run_feature_analysis` | F1 score breakdown by metadata feature |
| 9 | `run_fairness_analysis` | Fisher/Chi² tests + BH-FDR correction, Cramér's V |
| 10 | `run_intra_model_consistency_check` | Agreement across runs per model |
| 11 | `run_inter_model_similarity` | Cross-model agreement, PCA, SBERT embeddings |
| 12 | `run_justification_deep_analysis` | N-gram keyword extraction, region coverage |
| 13 | `run_best_per_family_ensemble` | Best-per-family and Top-3 majority-vote ensembles |
| 14 | `run_worst_case_extraction` | Hardest samples across models |
| 15 | `run_significance_tests` | McNemar, Cochran's Q, ensemble significance |

All outputs (Excel files, plots, LaTeX tables) are saved to `results/` and `plots/`.

---

## Reproducibility Notes

- PCA uses `random_state=42` for deterministic results.
- All models were evaluated with provider-default temperature settings.
- N=3 runs per configuration. Results are reported as mean ± std across runs.
- ROC AUC is computed per run and then averaged (interpolated to a 200-point FPR grid).
- Multiple-comparisons correction: Benjamini-Hochberg FDR per (model, run) group.
- SBERT model: `all-MiniLM-L6-v2` (downloaded automatically on first run).
