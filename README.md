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
    ├── baseline/
    │   └── .gitkeep
    ├── clfir_final/
    │   ├── summary.json
    │   └── per_study_summary.csv
    ├── more_imp1/
    │   ├── summary.json
    │   └── per_study_summary.csv
    ├── proposed_improvement_1/
    │   └── .gitkeep
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

![IU chest X-ray report generation pipeline](assets/iu_cxr_pipeline_diagram.png)

## CLFIR Adapter Training and Use

The CLFIR adapter used by the retrieval stage is a fine-tuned image-text retrieval projection module. It is not a report generator by itself. Its role is to map BioViL-T image embeddings into a retrieval space that is better aligned with radiology report text.

Training setup:

- Image encoder input: BioViL-T projected global image embedding, `128d`.
- Text encoder: `pritamdeka/S-PubMedBert-MS-MARCO`.
- Shared retrieval projection size: `512d`.
- Max text length: `320` tokens.
- Mixup lambda range: `0.92` to `0.99`.
- Optimizer from checkpoint state: AdamW, initial learning rate `3e-5`, weight decay `0.01`, betas `(0.9, 0.999)`.
- Selected checkpoint: epoch `18`.
- Validation retrieval metrics at the selected checkpoint:
  - image-to-text recall@1: `0.0758`
  - image-to-text recall@5: `0.2302`
  - image-to-text recall@10: `0.3247`
  - text-to-image recall@1: `0.0790`
  - text-to-image recall@5: `0.2230`
  - text-to-image recall@10: `0.3243`

At inference time, the query IU chest X-ray is embedded with BioViL-T, projected through the trained CLFIR image projection layer, L2-normalized, and searched against the CLFIR FAISS visual bank. The retrieved studies provide soft evidence for later fusion; they are not copied directly into the final report.

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

## Baseline Results

The `results/baseline/` folder is reserved for the cleaned Yi et al.-style five-agent baseline reproduction. It is kept separate from the CheXOne direct baseline reported inside the CLFIR result summaries.

Baseline design:

- Reproduction target: the multimodal multi-agent radiology report generation framework described by Yi et al.
- Single-agent baseline: LLaVA-Med 1.5 7B generates an IU chest X-ray report directly from the image.
- Retrieval agent: OpenCLIP ResNet-50 retriever fine-tuned on 3,000 MIMIC-CXR image-report pairs.
- Retrieval checkpoint: `retriever_openclip_mimic3000.pt`, supplied locally as `retriever_openclip_mimic3000.tar` in the development workspace. Model weights are not committed to this repository.
- Five-agent flow: retrieve related reports, draft from retrieved context, generate a visual description with LLaVA-Med, refine the draft, then synthesize the final report.
- Local LLM adaptation: Gemini can be used when an API key is provided; otherwise the cleaned notebook uses local Ollama models for consistency with the CLFIR runs.
- Local Ollama defaults: composer `qwen3.5:9b`, judge `gemma3:12b`.
- Evaluation alignment: the judge prompt and automatic metric output schema were cleaned to match the CLFIR notebooks.

The cleaned baseline notebook removes brittle fallback behavior where possible, keeps checkpointed generation/evaluation outputs, and writes to a dedicated artifact root in the development workspace. It is included conceptually as the literature-inspired reference pipeline, while this repository intentionally excludes its large model files, retrieval checkpoint, full caches, and raw per-stage artifacts.

When the files are added, `results/baseline/` should contain the same compact result schema used by the other runs:

Expected files:

- `results/baseline/summary.json`
- `results/baseline/per_study_summary.csv`

The baseline section should report the same categories as the submitted pipeline: completed reports, completed judge scores, LLM judge overall, automatic text metrics, and any clinical judge sub-scores included in the provided result files.

## Proposed Improvement 1 Results

The `results/proposed_improvement_1/` folder is reserved for the first proposed improvement pipeline. This is not the main submitted `clfir_final` method and it is not a CLFIR adapter pipeline.

Proposed Improvement 1 design:

- Starting point: an earlier IU end-to-end pipeline with image-first report generation, visual retrieval, pathology-aware retrieval, multi-stage report composition, and evaluation.
- Direct image model: CheXOne produces the image-first draft report.
- Retrieval: uses the original visual/pathology retrieval structure from the proposed-improvement notebook rather than adding CLFIR.
- Label evidence: VisualCheXbert and keyword-derived label fallbacks were removed from the cleaned version.
- CheXbert use: CheXbert labels are used for report-text evidence extraction, including positive, uncertain, negative, and blank/not-mentioned states.
- Important label-state fix: raw CheXbert negative `0` is mapped internally to the state `negative` and score `-1.0`; blank/not-mentioned remains `0.0`.
- CheXOne label-prompt outputs are not used as evidence.
- Local LLM adaptation: Gemini can be used when an API key is provided, but the cleaned local path uses the same Ollama defaults as the CLFIR notebooks.
- Local Ollama defaults: composer `qwen3.5:9b`, judge `gemma3:12b`.
- Evaluation alignment: the final evaluation block was updated to report the same compact metrics as `clfir_final`, including BLEU, ROUGE, METEOR, BERTScore, sacreBLEU, chrF, chrF++, and LLM judge overall.

The purpose of this experiment is to test whether the earlier non-CLFIR retrieval pipeline can be made cleaner and more comparable by replacing weak fallback label logic with CheXbert-derived text evidence and by standardizing the LLM/evaluation setup. It should be compared against both the Yi-style baseline and `clfir_final` only after its own independent run outputs are added.

When the files are added, `results/proposed_improvement_1/` should contain the same compact result schema:

Expected files:

- `results/proposed_improvement_1/summary.json`
- `results/proposed_improvement_1/per_study_summary.csv`

This section should summarize what changed relative to `clfir_final`, then report the same evaluation categories: completed reports, completed judge scores, LLM judge overall, automatic text metrics, and clinical judge sub-scores when available.

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
