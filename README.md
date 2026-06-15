# LLM-Based Zero-Shot Deepfake Video Detection

This repository contains the inference pipelines and evaluation code a Zero-Shot Deepfake Video Detection using Large Language Models.

Five multimodal LLMs (Gemini 3 Flash, GPT-5.2, Qwen 3.5, Kimi k2.5, Seed 2.0 Lite) are evaluated
on a balanced dataset of N=50 videos across four prompt variants (Baseline, +Indicators, +Thinking, +Indicators+Thinking).

---

## Repository Structure

```
.
├── inference/
│   ├── Gemini.py          # Gemini 3 Flash — Google Files API
│   ├── Qwen.py            # Qwen 3.5 — DashScope (fps=2)
│   ├── kimi.py            # Kimi k2.5 — Moonshot Files API
│   └── seed.py            # Seed 2.0 Lite — OpenRouter (Base64)
├── evaluation/
│   └── evaluation.py      # Full analysis pipeline (4 phases)
├── data/
│   └── processed/         # Input videos (not included)
├── results/               # JSON outputs from inference (auto-created)
├── plots/                 # Generated figures (auto-created)
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
| `QWEN_API_KEY` | Qwen.py |
| `MOONSHOT_API_KEY` | Kimi.py |
| `OPENROUTER_API_KEY` | Seed.py |
| `GEMINI_API_KEY` | Gemini.py |

---

## Dataset Structure

Place the videos under `data/processed/` 

A `dataset_info.xlsx` file with ground truth labels and metadata
(columns: `video_id`, `label`, `dataset`, `gender`, `video_length`,
`deepfake_category`, `deepfake_type`, `audio`) must be placed in the project root.

---

## Running Inference

Each script processes all `.mp4` files under `data/processed/` and saves results
to a JSON file. Already-processed videos are skipped automatically (resume-safe).

Configure the desired variant at the top of each script:

```python
thinking_budget  = 0         # Gemini: -1 = dynamic reasoning, 0 = disabled
enable_thinking  = False     # Qwen:   True / False
thinking_type    = "disabled" # Kimi:  "enabled" / "disabled"
thinking_type    = False     # Seed:   True / False

indicators_type  = "disabled" # all scripts: "enabled" / "disabled"
```

Then run:

```bash
python inference/Gemini.py
python inference/Qwen.py
python inference/kimi.py
python inference/seed.py
```
For multiple runs add _2,_3, etc. at the end of the output file.

Output files are named automatically, e.g. `gemini-3-flash-preview.json`,
`gemini-3-flash-preview_thinking.json`, etc.

---

## Running the Evaluation Pipeline

```bash
python evaluation/evaluation.py
```

The pipeline expects the following JSON result files in the project root,
one per model variant and run (suffix `_1`, `_2`, `_3`):

```
gemini-3-flash-preview_1.json
gemini-3-flash-preview_2.json
gemini-3-flash-preview_3.json
gemini-3-flash-preview_thinking_1.json
...
```

The pipeline runs four phases automatically:

| Phase | Description |
|---|---|
| 1 | Data aggregation and pre-processing |
| 2 | Statistical performance and fairness metrics |
| 3 | Semantic analysis of justifications (SBERT, N-gram) |
| 4 | Robustness tests: Best-per-Family Ensemble, Worst-Case Extraction |

All outputs (Excel files, plots, LaTeX tables) are saved to `results/` and `plots/`.

---

## Reproducibility Notes

- PCA uses `random_state=42` for deterministic results.
- All models were evaluated with provider-default temperature settings.
- N=3 runs per configuration. Results are reported as mean ± std across runs.
- SBERT model: `all-MiniLM-L6-v2` (downloaded automatically on first run).

---

