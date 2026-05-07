# IU CXR Report Generation with CLFIR-Guided Evidence Fusion

This repository contains the cleaned deliverable for an IU chest X-ray report generation project. The main submitted pipeline is `clfir_final`, implemented as a Jupytext-paired notebook and Python script.

The full local development workspace, datasets, model weights, retrieval banks, LLM caches, and per-stage artifact folders are intentionally not included. Only the submission notebook/code and compact evaluation outputs are provided.

## Repository Contents

```text
.
├── notebooks/
│   ├── iu_pipeline_e2e_clfir_final.ipynb
│   └── iu_pipeline_e2e_clfir_final.py
└── results/
    ├── clfir_final/
    │   ├── summary.json
    │   └── per_study_summary.csv
    ├── more_imp1/
    │   ├── summary.json
    │   └── per_study_summary.csv
    └── comparisons/
        ├── judge_criteria_comparison_dashboard.png
        ├── automatic_metrics_comparison_dashboard.png
        ├── corpus_automatic_metrics.png
        ├── judge_criteria_paired_scores.csv
        ├── automatic_metrics_paired_scores.csv
        └── comparison_summary.json
```

## Project Overview

The pipeline generates concise IU-style chest X-ray reports with two required sections: `Findings:` and `Impression:`.

At a high level, it uses:

- **CheXOne** as an image-first direct report draft.
- **BioViL-T / CLFIR adapter retrieval** to retrieve visually related studies.
- **CheXbert-style label evidence** to summarize retrieved reports in a coarse label space.
- **Deterministic evidence fusion** to combine direct image evidence with retrieval-text evidence.
- **LLM report composition** to write the final report from the approved evidence.
- **LLM judging** to compare generated reports against IU reference reports.

The submitted notebook is designed to resume from disk artifacts when run in the original local environment. External datasets, model snapshots, retrieval banks, and generated stage artifacts are excluded from this repository due to size and portability.

## Submitted Pipeline

The primary submitted pipeline is:

- Notebook: `notebooks/iu_pipeline_e2e_clfir_final.ipynb`
- Paired script: `notebooks/iu_pipeline_e2e_clfir_final.py`
- Output directory in the original run: `current_clfir_final`

Important run settings:

- Evaluation limit: `300`
- Composer model: `qwen3.5:9b`
- Judge model: `gemma3:12b`
- Jupytext format: paired `.ipynb` and `.py`

The paired Python file is included so the notebook can be reviewed as executable source without relying only on notebook JSON.

## Experimental Comparison

The `results/more_imp1/` folder contains compact outputs from an experimental Adapter2 variant. That variant used a RadGraph-family evidence arbitration layer and achieved a slightly higher LLM judge score, but it is not the main clean submission pipeline.

The comparison plots in `results/comparisons/` compare `clfir_final` and `more_imp1` over the same 300 IU samples.

## 300-Sample Results

### CLFIR Final

Pipeline:

- Completed reports: `300/300`
- Completed judge scores: `300/300`
- Judge overall: `7.713`
- BLEU: `0.123`
- ROUGE-1: `0.416`
- ROUGE-2: `0.174`
- ROUGE-L: `0.311`
- METEOR: `0.353`
- BERTScore F1: `0.857`
- sacreBLEU: `13.145`
- chrF: `41.809`
- chrF++: `38.663`

CheXOne direct baseline:

- Judge overall: `7.327`

### Adapter2 More-Improvement Experiment

Pipeline:

- Completed reports: `300/300`
- Completed judge scores: `300/300`
- Judge overall: `7.763`
- BLEU: `0.115`
- ROUGE-1: `0.412`
- ROUGE-2: `0.173`
- ROUGE-L: `0.313`
- METEOR: `0.331`
- BERTScore F1: `0.854`
- sacreBLEU: `11.571`
- chrF: `38.867`
- chrF++: `35.862`

CheXOne direct baseline:

- Judge overall: `7.320`

## Interpretation

The two pipelines show a useful tradeoff:

- `clfir_final` is stronger on automatic text-overlap metrics such as BLEU, METEOR, BERTScore, sacreBLEU, and chrF.
- `more_imp1` is slightly stronger on the LLM judge metrics, especially clinical accuracy and groundedness.

For submission, `clfir_final` is used as the primary clean pipeline. The `more_imp1` results are included to document the experimental evidence-arbitration direction.

## Notes on Reproducibility

This repository does not include:

- IU-Xray or MIMIC datasets.
- Model weights or local Ollama model files.
- Full retrieval banks.
- Full per-stage pipeline artifacts.
- LLM prompt caches.

The compact result files under `results/` are included so the reported metrics and comparison plots can be inspected without requiring the original full artifact bundle.

