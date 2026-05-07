# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Evidence-Guided IU CXR Report Generation
#
# This notebook runs an end-to-end IU chest X-ray report pipeline that keeps the
# image-first CheXOne report as the primary draft and uses retrieval, coarse
# label evidence, and a report composer to produce a concise Findings/Impression
# report. It is designed for repeatable local experiments: each stage writes a
# per-study artifact, checks whether matching artifacts already exist, and
# resumes from disk instead of restarting completed work.
#
# The active configuration writes to `current_clfir_final`. In the default local
# setup, Stage 5 and Stage 7 use Ollama-backed LLM calls (`qwen3.5:9b` composer
# and `gemma3:12b` judge unless overridden by environment variables). If local
# Ollama is disabled, the same stage wrappers can use Gemini models when a
# working API key is available.
#
# Pipeline stages:
# 1. Stage 1 imports or generates a CheXOne image-first report and converts it
#    into a deterministic 14-label proxy.
# 2. Stage 2a embeds the query image with BioViL-T and searches the CLFIR
#    adapter-2 visual bank.
# 3. Stage 2c applies deterministic reranking to the retrieved visual hits.
# 4. Stage 3 aggregates CheXbert labels from the retrieved reports into
#    text-side evidence.
# 5. Stage 4 fuses the image-side and retrieval-text label vectors into
#    confirmed, image-only, text-only, conflict, or absent states.
# 6. Stage 5 composes the final report from the fused evidence, direct draft,
#    and retrieved context.
# 7. Stage 7 judges both the pipeline report and the CheXOne direct baseline,
#    then the evaluation block writes per-study and aggregate metrics.
#
# Design rules:
# - artifacts are durable and versioned by directory;
# - completed stage artifacts are reused unless a `FORCE_*` flag is enabled;
# - external verified banks and CheXOne reports are preferred when configured;
# - LLM prompts are cached with retry metadata;
# - kernel restarts should not lose completed work.

# %%
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import ast
import csv
import getpass
import json
import math
import os
import re
import sys
import time
from urllib import error, parse, request

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# %% [markdown]
# ## Optional dependency install
#
# This cell is off by default. Enable it only if the runtime is missing the
# required packages.

# %%
OPTIONAL_INSTALL = False

if OPTIONAL_INSTALL:
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "numpy",
            "pandas",
            "pillow",
            "transformers",
            "torch",
            "faiss-cpu",
            "hi-ml-multimodal",
            "jupytext",
            "nltk",
            "rouge-score",
            "bert-score",
            "sacrebleu",
        ],
        check=True,
    )

# %% [markdown]
# ## Global setup
#
# We keep imports for heavy external libraries in dedicated cells or functions so
# that opening the notebook does not immediately fail if a library is missing.

# %%
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
    return target


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def report_quality(report_text: str) -> dict[str, Any]:
    original = str(report_text or "")
    text = normalize_text(original)
    text = re.sub(r"\b(findings|impression)\s*:", " ", text)
    text = re.sub(r"\b(report|exam|comparison)\s*:", " ", text)
    text = re.sub(r"[_\\s]+", " ", text).strip()
    alpha_words = re.findall(r"[a-zA-Z]{4,}", text)
    has_content = len(text) >= REPORT_MIN_CONTENT_CHARS and len(alpha_words) >= REPORT_MIN_ALPHA_WORDS
    reason = "ok" if has_content else "empty_or_section_headers_only"
    return {
        "has_clinical_content": has_content,
        "reason": reason,
        "content_chars": len(text),
        "alpha_word_count": len(alpha_words),
    }


def hit_has_clinical_content(row: dict[str, Any]) -> bool:
    quality = row.get("report_quality")
    if isinstance(quality, dict) and "has_clinical_content" in quality:
        return bool(quality["has_clinical_content"])
    return bool(report_quality(str(row.get("report_text", "")))["has_clinical_content"])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def retry_sleep_seconds(attempt_index: int) -> int:
    schedule = [2, 5, 10, 20, 30]
    return schedule[min(attempt_index, len(schedule) - 1)]


def file_is_nonempty(path: str | Path) -> bool:
    target = Path(path)
    return target.exists() and target.stat().st_size > 0


def download_to_path(url: str, destination: str | Path) -> Path:
    target = Path(destination)
    ensure_dir(target.parent)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    request.urlretrieve(url, tmp_target)
    tmp_target.replace(target)
    return target


def artifact_has_success_status(path: str | Path) -> bool:
    target = Path(path)
    if not target.exists():
        return False
    try:
        payload = read_json(target)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("status") == "success"


def detect_device() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def find_workspace_root() -> Path:
    anchors: list[Path] = []
    if "__file__" in globals():
        anchors.append(Path(__file__).resolve().parent)
    anchors.append(Path.cwd().resolve())

    for anchor in anchors:
        candidates = [anchor, *anchor.parents]
        for candidate in candidates:
            if (candidate / "IU-Xray").exists() and (candidate / "MIMIC").exists():
                return candidate

    raise RuntimeError(
        "Could not locate a local workspace containing `IU-Xray` and `MIMIC`."
    )


def ensure_gemini_api_key_interactive() -> str | None:
    global USE_LOCAL_OLLAMA_LLM, COMPOSER_MODEL_NAME, JUDGE_MODEL_NAME, ENABLE_LLM_JUDGE, STOP_BEFORE_GEMINI_STAGES, LLM_PARALLEL_WORKERS
    existing = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if existing and existing.strip().lower() == "none":
        os.environ["USE_LOCAL_OLLAMA_LLM"] = "1"
        USE_LOCAL_OLLAMA_LLM = True
        COMPOSER_MODEL_NAME = LOCAL_OLLAMA_COMPOSER_MODEL_NAME
        JUDGE_MODEL_NAME = LOCAL_OLLAMA_JUDGE_MODEL_NAME
        ENABLE_LLM_JUDGE = True
        STOP_BEFORE_GEMINI_STAGES = False
        LLM_PARALLEL_WORKERS = 1
        print("Gemini API key set to 'none'; using local Ollama for Stage 5 and Stage 7.")
        return None
    if existing:
        USE_LOCAL_OLLAMA_LLM = False
        os.environ["USE_LOCAL_OLLAMA_LLM"] = "0"
        COMPOSER_MODEL_NAME = "gemini-2.5-flash"
        JUDGE_MODEL_NAME = "gemini-2.5-pro"
        ENABLE_LLM_JUDGE = True
        STOP_BEFORE_GEMINI_STAGES = False
        print("Using Gemini for Stage 5 and Stage 7.")
        return existing
    print("Gemini API key not found in environment.")
    try:
        entered = getpass.getpass("Enter GEMINI_API_KEY, or type 'none' to use local Ollama: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("No interactive Gemini API key input available. Continuing without Gemini stages.")
        return None
    if entered.lower() == "none":
        os.environ["USE_LOCAL_OLLAMA_LLM"] = "1"
        os.environ["GEMINI_API_KEY"] = "none"
        USE_LOCAL_OLLAMA_LLM = True
        COMPOSER_MODEL_NAME = LOCAL_OLLAMA_COMPOSER_MODEL_NAME
        JUDGE_MODEL_NAME = LOCAL_OLLAMA_JUDGE_MODEL_NAME
        ENABLE_LLM_JUDGE = True
        STOP_BEFORE_GEMINI_STAGES = False
        LLM_PARALLEL_WORKERS = 1
        print("Using local Ollama for Stage 5 and Stage 7.")
        return None
    if entered:
        os.environ["GEMINI_API_KEY"] = entered
        USE_LOCAL_OLLAMA_LLM = False
        os.environ["USE_LOCAL_OLLAMA_LLM"] = "0"
        COMPOSER_MODEL_NAME = "gemini-2.5-flash"
        JUDGE_MODEL_NAME = "gemini-2.5-pro"
        ENABLE_LLM_JUDGE = True
        STOP_BEFORE_GEMINI_STAGES = False
        print("Using Gemini for Stage 5 and Stage 7.")
        return entered
    print("No Gemini API key provided. Gemini-dependent stages will skip or fail preflight if enabled.")
    return None

# %% [markdown]
# ## Configuration
#
# This cell defines the experiment size, artifact locations, model backends, and
# resume behavior. The expected local layout is:
# - `WORKSPACE_ROOT/IU-Xray/...` for IU reports, projections, and normalized
#   images;
# - `WORKSPACE_ROOT/MIMIC/...` for the optional MIMIC retrieval-bank source;
# - `WORKSPACE_ROOT/pipeline/artifacts/iu_pipeline_bundle/...` for external
#   verified artifacts, model cache files, and this run's output directory.
#
# The current run writes under `current_clfir_final`. `IU_EVAL_LIMIT` controls
# how many IU test studies are processed. The default is 200, with external
# CheXOne reports and retrieval-bank artifacts loaded from the shared bundle.
#
# LLM behavior is controlled by `USE_LOCAL_OLLAMA_LLM`. In the checked-in
# configuration it defaults to local Ollama, sets one LLM worker, and uses
# environment-overridable composer and judge model names. If Ollama is disabled,
# Gemini model names remain configured and the API-key helper decides whether
# Stage 5/7 can run.

# %%
# Resume behavior:
# - keep `ARTIFACT_ROOT` stable
# - if you run 5 samples first and later raise `IU_EVAL_LIMIT` to 200, the
#   first 5 per-study artifacts are reused automatically
# - only the missing samples run unless a matching `FORCE_*` flag is `True`

IU_TEST_COUNT = 590
IU_EVAL_LIMIT = 300
IU_TRAIN_BANK_TARGET = 0
MIMIC_BANK_TARGET = 5000
RETRIEVAL_TOP_K = 5
REPORT_QUALITY_OVERFETCH_FACTOR = 10
REPORT_MIN_CONTENT_CHARS = 25
REPORT_MIN_ALPHA_WORDS = 4

STAGE3_MODEL_NAME = "CheXbert"
COMPOSER_MODEL_NAME = "gemini-2.5-flash"
JUDGE_MODEL_NAME = "gemini-2.5-pro"
ENABLE_LLM_JUDGE = False
LOCAL_OLLAMA_COMPOSER_MODEL_NAME = os.getenv("LOCAL_OLLAMA_COMPOSER_MODEL_NAME", "qwen3.5:9b").strip()
LOCAL_OLLAMA_JUDGE_MODEL_NAME = os.getenv("LOCAL_OLLAMA_JUDGE_MODEL_NAME", "gemma3:12b").strip()
LOCAL_OLLAMA_CHAT_URL = os.getenv("LOCAL_OLLAMA_CHAT_URL", "http://localhost:11434/api/chat").strip()
LOCAL_OLLAMA_TIMEOUT_SECONDS = int(os.getenv("LOCAL_OLLAMA_TIMEOUT_SECONDS", "1000"))
LOCAL_OLLAMA_COMPOSER_NUM_CTX = int(os.getenv("LOCAL_OLLAMA_COMPOSER_NUM_CTX", "32768"))
LOCAL_OLLAMA_COMPOSER_NUM_PREDICT = int(os.getenv("LOCAL_OLLAMA_COMPOSER_NUM_PREDICT", "16384"))
LOCAL_OLLAMA_COMPOSER_TEMPERATURE = float(os.getenv("LOCAL_OLLAMA_COMPOSER_TEMPERATURE", "0.7"))
LOCAL_OLLAMA_COMPOSER_TOP_P = float(os.getenv("LOCAL_OLLAMA_COMPOSER_TOP_P", "0.9"))
LOCAL_OLLAMA_JUDGE_NUM_CTX = int(os.getenv("LOCAL_OLLAMA_JUDGE_NUM_CTX", "8192"))
LOCAL_OLLAMA_JUDGE_NUM_PREDICT = int(os.getenv("LOCAL_OLLAMA_JUDGE_NUM_PREDICT", "1024"))
LOCAL_OLLAMA_JUDGE_TEMPERATURE = float(os.getenv("LOCAL_OLLAMA_JUDGE_TEMPERATURE", "0.1"))
LOCAL_OLLAMA_JUDGE_TOP_P = float(os.getenv("LOCAL_OLLAMA_JUDGE_TOP_P", "0.8"))
USE_LOCAL_OLLAMA_LLM = os.getenv("USE_LOCAL_OLLAMA_LLM", "1").strip().lower() in {"1", "true", "yes"}

FORCE_REBUILD_BANKS = False
FORCE_STAGE1_REFRESH = False
FORCE_CHEXONE_DIRECT_REFRESH = False
FORCE_STAGE5_REFRESH = False
FORCE_STAGE7_REFRESH = False

MAX_LLM_RETRIES = 2
BANK_CHECKPOINT_EVERY = 50
QUERY_IMAGE_MAX_DIM = 768
LLM_PARALLEL_WORKERS = int(os.getenv("LLM_PARALLEL_WORKERS", str(max(1, math.ceil(IU_EVAL_LIMIT / 50)))))

IMAGE_POSITIVE_THRESHOLD = 0.50
TEXT_POSITIVE_THRESHOLD = 0.50
CONFLICT_GAP_THRESHOLD = 0.40

CHEXONE_MODEL_ID = "StanfordAIMI/CheXOne"
BIOVILT_MODEL_ID = "microsoft/BiomedVLP-BioViL-T"

USE_EXTERNAL_VERIFIED_ARTIFACTS = True
STOP_BEFORE_GEMINI_STAGES = True
LOCAL_FILES_ONLY = False

if USE_LOCAL_OLLAMA_LLM:
    COMPOSER_MODEL_NAME = LOCAL_OLLAMA_COMPOSER_MODEL_NAME
    JUDGE_MODEL_NAME = LOCAL_OLLAMA_JUDGE_MODEL_NAME
    ENABLE_LLM_JUDGE = True
    STOP_BEFORE_GEMINI_STAGES = False
    LLM_PARALLEL_WORKERS = 1

ensure_gemini_api_key_interactive()

LOCAL_CHEXBERT_PREDICTIONS_PATH = ""
LOCAL_CHEXONE_SNAPSHOT_DIR = ""
LOCAL_BIOVILT_WEIGHTS_PATH = ""
AUTO_DOWNLOAD_BIOVILT_WEIGHTS = True
BIOVILT_WEIGHTS_URL = "https://huggingface.co/microsoft/BiomedVLP-BioViL-T/resolve/v1.0/biovil_t_image_model_proj_size_128.pt"
BIOVILT_WEIGHTS_FILENAME = "biovil_t_image_model_proj_size_128.pt"
BERTSCORE_MODEL_TYPE = "distilbert-base-uncased"

PIPELINE_SCHEMA_VERSION = "stage1_unified_v2"

CHEXPERT_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "No Finding",
]

NEGATION_TOKENS = (
    "no ",
    "without ",
    "absent ",
    "negative for ",
    "free of ",
    "no evidence of ",
)

HEDGE_TOKENS = (
    "possible",
    "possibly",
    "may represent",
    "may reflect",
    "cannot exclude",
    "could represent",
    "suggesting",
    "suggestive of",
    "likely",
    "probable",
)

DEVICE = detect_device()
WORKSPACE_ROOT = find_workspace_root()

IU_ROOT = WORKSPACE_ROOT / "IU-Xray"
IU_REPORTS_CSV = IU_ROOT / "indiana_reports.csv"
IU_PROJECTIONS_CSV = IU_ROOT / "indiana_projections.csv"
IU_IMAGES_DIR = IU_ROOT / "images" / "images_normalized"

MIMIC_ROOT = WORKSPACE_ROOT / "MIMIC"
MIMIC_TRAIN_CSV = MIMIC_ROOT / "mimic_cxr_aug_train.csv"
MIMIC_VALIDATE_CSV = MIMIC_ROOT / "mimic_cxr_aug_validate.csv"
MIMIC_IMAGE_ROOT = MIMIC_ROOT / "official_data_iccv_final"

BUNDLE_ROOT = (WORKSPACE_ROOT / "pipeline" / "artifacts" / "iu_pipeline_bundle").resolve()
EXTERNAL_ARTIFACTS_ROOT = BUNDLE_ROOT / "external"
MODEL_CACHE_DIR = BUNDLE_ROOT / "models"
MANAGED_BIOVILT_WEIGHTS_PATH = MODEL_CACHE_DIR / "biovilt" / BIOVILT_WEIGHTS_FILENAME

DEFAULT_EXTERNAL_BANK_ROOT = EXTERNAL_ARTIFACTS_ROOT / "full_retrieval_bank_kaggle_2xt4" / "edd340ce92d2"
DEFAULT_EXTERNAL_CHEXONE_REPORT_DIR = EXTERNAL_ARTIFACTS_ROOT / "chexone_iu590_fullres_2xt4" / "2d8eaefd8923" / "report_prompt"
CHEXONE_REPORT_CHEXPERT_LABELS_PATH = MODEL_CACHE_DIR / "chexbert_on_chexone_report_prompt" / "labels_by_uid.json"
EXTERNAL_BANK_ROOT = DEFAULT_EXTERNAL_BANK_ROOT
EXTERNAL_CHEXONE_REPORT_DIR = DEFAULT_EXTERNAL_CHEXONE_REPORT_DIR
EXTERNAL_BANKS_DIR = EXTERNAL_BANK_ROOT / "banks"
EXTERNAL_SPLITS_DIR = EXTERNAL_BANK_ROOT / "splits"
EXTERNAL_IU_TRAIN_MANIFEST_PATH = EXTERNAL_SPLITS_DIR / "iu_train_manifest_canonical_2069.json"
EXTERNAL_IU_VAL_MANIFEST_PATH = EXTERNAL_SPLITS_DIR / "iu_val_manifest_canonical_296.json"
EXTERNAL_IU_EVAL_MANIFEST_SOURCE_PATH = EXTERNAL_SPLITS_DIR / "iu_test_manifest_canonical_590.json"
EXTERNAL_MIMIC_BANK_MANIFEST_PATH = EXTERNAL_BANKS_DIR / "mimic_bank_manifest_fraction_0_3333333333.json"
EXTERNAL_IU_BANK_MANIFEST_PATH = EXTERNAL_BANKS_DIR / "iu_train_bank_manifest_2365.json"
EXTERNAL_RETRIEVAL_BANK_MANIFEST_PATH = EXTERNAL_BANKS_DIR / "retrieval_bank_manifest_fraction_0_3333333333_iu_2365.json"
EXTERNAL_VISUAL_BANK_DIR = EXTERNAL_BANKS_DIR / "visual"
EXTERNAL_PATHOLOGY_BANK_DIR = EXTERNAL_BANKS_DIR / "pathology"
EXTERNAL_CHEXBERT_METADATA_PATH = EXTERNAL_PATHOLOGY_BANK_DIR / "mimic_pathology_metadata.json"
EXTERNAL_CLFIR_ADAPTER2_ROOT = EXTERNAL_ARTIFACTS_ROOT / "clfir_retrieval_adapter_2"
EXTERNAL_CLFIR_ADAPTER2_VISUAL_BANK_DIR = EXTERNAL_CLFIR_ADAPTER2_ROOT / "banks" / "visual_clfir"

ARTIFACT_ROOT = BUNDLE_ROOT / "current_clfir_final"
SPLITS_DIR = ARTIFACT_ROOT / "splits"
BANKS_DIR = ARTIFACT_ROOT / "banks"
STAGE1_DIR = ARTIFACT_ROOT / "stage1"
STAGE2A_DIR = ARTIFACT_ROOT / "stage2a"
STAGE2A_QUERY_EMBED_DIR = ARTIFACT_ROOT / "stage2a_query_embeddings"
STAGE2C_DIR = ARTIFACT_ROOT / "stage2c"
STAGE3_DIR = ARTIFACT_ROOT / "stage3"
STAGE4_DIR = ARTIFACT_ROOT / "stage4"
CHEXONE_DIRECT_DIR = ARTIFACT_ROOT / "chexone_direct"
STAGE5_DIR = ARTIFACT_ROOT / "stage5"
JUDGING_PIPELINE_DIR = ARTIFACT_ROOT / "judging_pipeline"
JUDGING_CHEXONE_DIR = ARTIFACT_ROOT / "judging_chexone_direct"
LLM_FLASH_DIR = ARTIFACT_ROOT / "llm_cache" / "flash"
LLM_PRO_DIR = ARTIFACT_ROOT / "llm_cache" / "pro"
EVAL_DIR = ARTIFACT_ROOT / "evaluation"

for directory in [
    SPLITS_DIR,
    BANKS_DIR / "visual",
    MODEL_CACHE_DIR / "biovilt",
    STAGE1_DIR,
    STAGE2A_DIR,
    STAGE2A_QUERY_EMBED_DIR,
    STAGE2C_DIR,
    STAGE3_DIR,
    STAGE4_DIR,
    CHEXONE_DIRECT_DIR,
    STAGE5_DIR,
    JUDGING_PIPELINE_DIR,
    JUDGING_CHEXONE_DIR,
    LLM_FLASH_DIR,
    LLM_PRO_DIR,
    EVAL_DIR,
]:
    ensure_dir(directory)

print(f"Workspace root: {WORKSPACE_ROOT}")
print(f"IU root: {IU_ROOT}")
print(f"MIMIC root: {MIMIC_ROOT}")
print(f"Bundle root: {BUNDLE_ROOT}")
print(f"Artifact root: {ARTIFACT_ROOT}")
print(f"Device: {DEVICE}")
print(f"Use external verified artifacts: {USE_EXTERNAL_VERIFIED_ARTIFACTS}")
print(f"Stop before Gemini stages: {STOP_BEFORE_GEMINI_STAGES}")
print(f"LLM parallel workers: {LLM_PARALLEL_WORKERS}")
if USE_EXTERNAL_VERIFIED_ARTIFACTS:
    print(f"External bank root: {EXTERNAL_BANK_ROOT}")
    print(f"External CheXOne report dir: {EXTERNAL_CHEXONE_REPORT_DIR}")
    print(f"External CLFIR adapter-2 visual bank dir: {EXTERNAL_CLFIR_ADAPTER2_VISUAL_BANK_DIR}")


# %% [markdown]
# ## External-Artifact Mode
#
# When `use_external_verified_artifacts=True`, this notebook does not recompute
# the IU report artifacts or the retrieval banks. It loads:
# - the verified external retrieval bank
# - the verified external CheXbert label metadata
# - the verified external CheXOne report JSONs
#
# In that mode:
# - Stage 1 is imported from the verified CheXOne report JSONs
# - the label vector used by retrieval is derived deterministically from that
#   imported report text and stored inside the same Stage 1 artifact
# - the only remaining required model compute before Gemini is Stage 2a query
#   image embedding for visual retrieval, and those query embeddings are cached
#   locally under the run folder


# %% [markdown]
# ## Shared dataclasses

# %%
@dataclass(slots=True)
class IUStudy:
    uid: str
    image_path: str
    findings: str
    impression: str
    ground_truth_report: str


@dataclass(slots=True)
class MIMICStudy:
    subject_id: str
    study_id: str
    image_path: str
    all_image_paths: list[str]
    report_text: str
    split: str
    source_dataset: str = "mimic"


@dataclass(slots=True)
class RetrievalHit:
    rank: int
    score: float
    subject_id: str
    study_id: str
    image_path: str
    report_text: str
    split: str
    source_dataset: str = "mimic"
    labels: dict[str, float] = field(default_factory=dict)
    visual_score: float | None = None
    combined_score: float | None = None
    report_quality: dict[str, Any] = field(default_factory=dict)


# %% [markdown]
# ## Dataset helpers

# %%
STUDY_ID_PATTERN = re.compile(r"/(s\d+)/")


def parse_list_cell(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def choose_frontal_path(study_paths: list[str], ap_paths: set[str], pa_paths: set[str], lateral_paths: set[str]) -> str | None:
    pa_candidates = [path for path in study_paths if path in pa_paths]
    if pa_candidates:
        return sorted(pa_candidates)[0]
    ap_candidates = [path for path in study_paths if path in ap_paths]
    if ap_candidates:
        return sorted(ap_candidates)[0]
    return None


def build_ground_truth_text(findings: str, impression: str) -> str:
    sections: list[str] = []
    if str(findings).strip() and str(findings).strip().lower() != "nan":
        sections.append(str(findings).strip())
    if str(impression).strip() and str(impression).strip().lower() != "nan":
        sections.append(str(impression).strip())
    return " ".join(sections).strip()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?;])\s+|\n+", str(text).strip())
    return [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks if re.sub(r"\s+", " ", chunk).strip()]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for item in items:
        key = normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(item.strip())
    return kept


def choose_iu_frontal_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    frontal = [row for row in rows if re.search(r"frontal|pa|ap", str(row.get("projection", "")).lower())]
    return frontal[0] if frontal else None


def build_stage1_prompt() -> str:
    return """Write only a concise chest X-ray report in standard radiology prose.

Rules:
- Output plain report text only.
- No headings.
- No bullets.
- No brackets.
- No labels.
- No explanations.
- Mention only findings visible on the image.
"""


def format_confirmed_fusion_lines(fusion_table: list[dict[str, Any]]) -> str:
    confirmed_rows = [
        row for row in fusion_table
        if str(row.get("state")) == "CONFIRMED" and str(row.get("finding")) != "No Finding"
    ]
    if not confirmed_rows:
        return "No 14-label findings reached CONFIRMED status."
    return "\n".join(
        f"- {row['finding']}: CONFIRMED (image={row['image_score']:.3f}, text={row['text_score']:.3f})"
        for row in confirmed_rows
    )


def format_fusion_guardrail_lines(fusion_table: list[dict[str, Any]]) -> str:
    non_confirmed_rows = [row for row in fusion_table if str(row.get("finding")) != "No Finding"]
    if not non_confirmed_rows:
        return "No additional label guardrails."
    return "\n".join(
        f"- {row['finding']}: {row['state']} (image={row['image_score']:.3f}, text={row['text_score']:.3f})"
        for row in non_confirmed_rows
    )


# %% [markdown]
# ## IU evaluation manifest code

# %%
def build_iu_study_table() -> list[IUStudy]:
    with IU_REPORTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reports = list(csv.DictReader(handle))
    with IU_PROJECTIONS_CSV.open("r", encoding="utf-8", newline="") as handle:
        projections = list(csv.DictReader(handle))

    by_uid: dict[str, list[dict[str, str]]] = {}
    for row in projections:
        by_uid.setdefault(str(row["uid"]), []).append(row)

    studies: list[IUStudy] = []
    for report in tqdm(reports, total=len(reports), desc="Building IU study table", unit="report"):
        uid = str(report["uid"])
        projection_rows = by_uid.get(uid)
        if not projection_rows:
            continue
        projection_row = choose_iu_frontal_row(projection_rows)
        if projection_row is None:
            continue
        image_path = IU_IMAGES_DIR / str(projection_row["filename"])
        if not image_path.exists():
            continue
        findings = str(report.get("findings", "") or "").strip()
        impression = str(report.get("impression", "") or "").strip()
        ground_truth = build_ground_truth_text(findings, impression)
        if not ground_truth:
            continue
        studies.append(
            IUStudy(
                uid=uid,
                image_path=str(image_path),
                findings=findings,
                impression=impression,
                ground_truth_report=ground_truth,
            )
        )
    return studies


# %% [markdown]
# ## IU train/test manifest run

# %%
def split_iu_train_test(studies: list[IUStudy], test_count: int) -> tuple[list[IUStudy], list[IUStudy]]:
    ordered = sorted(studies, key=lambda item: int(item.uid))
    if test_count <= 0:
        raise ValueError("iu_test_count must be positive.")
    if len(ordered) <= test_count:
        raise ValueError(f"Usable IU study count {len(ordered)} is not larger than test count {test_count}.")
    return ordered[:-test_count], ordered[-test_count:]

def _resolve_existing_path(path_value: str | Path) -> Path:
    resolved = map_external_dataset_path(path_value)
    if not resolved.exists():
        raise FileNotFoundError(f"Required path not found: {resolved}")
    return resolved


def map_external_dataset_path(path_value: str | Path) -> Path:
    raw_path = Path(path_value).expanduser()
    if raw_path.exists():
        return raw_path.resolve()

    text = str(raw_path)
    replacements = [
        ("/kaggle/input/datasets/raddar/chest-xrays-indiana-university", str(IU_ROOT)),
        ("/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset", str(MIMIC_ROOT)),
        ("/content/drive/MyDrive/IU-Xray", str(IU_ROOT)),
        ("/content/drive/MyDrive/MIMIC", str(MIMIC_ROOT)),
    ]
    for source_prefix, target_prefix in replacements:
        if text.startswith(source_prefix):
            mapped = Path(target_prefix + text[len(source_prefix):]).expanduser()
            if mapped.exists():
                return mapped.resolve()

    return raw_path.resolve()


def remap_iu_study_paths(studies: list[IUStudy]) -> list[IUStudy]:
    remapped: list[IUStudy] = []
    for study in studies:
        remapped.append(
            IUStudy(
                uid=study.uid,
                image_path=str(map_external_dataset_path(study.image_path)),
                findings=study.findings,
                impression=study.impression,
                ground_truth_report=study.ground_truth_report,
            )
        )
    return remapped


def remap_mimic_study_paths(studies: list[MIMICStudy]) -> list[MIMICStudy]:
    remapped: list[MIMICStudy] = []
    for study in studies:
        remapped.append(
            MIMICStudy(
                subject_id=study.subject_id,
                study_id=study.study_id,
                image_path=str(map_external_dataset_path(study.image_path)),
                all_image_paths=[str(map_external_dataset_path(item)) for item in study.all_image_paths],
                report_text=study.report_text,
                split=study.split,
                source_dataset=study.source_dataset,
            )
        )
    return remapped


def validate_unique_values(values: list[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        key = str(value).strip()
        if key in seen:
            duplicates.append(key)
            if len(duplicates) >= 10:
                break
        seen.add(key)
    if duplicates:
        raise RuntimeError(f"Duplicate {label} values detected: {duplicates}")


def validate_eval_manifest_alignment(studies: list[IUStudy]) -> None:
    validate_unique_values([study.uid for study in studies], "IU eval uid")
    for study in studies:
        image_path = _resolve_existing_path(study.image_path)
        external_payload = _load_external_chexone_report_payload(study.uid)
        external_image_path = _resolve_existing_path(str(external_payload.get("image_path", "")))
        if image_path != external_image_path:
            raise RuntimeError(
                f"Eval alignment mismatch for uid={study.uid}: {image_path} vs {external_image_path}"
            )


def validate_retrieval_bank_manifest_alignment(studies: list[MIMICStudy]) -> None:
    validate_unique_values([study.study_id for study in studies], "retrieval bank study_id")
    for study in studies:
        _resolve_existing_path(study.image_path)


def validate_bank_metadata_alignment(
    metadata: list[dict[str, Any]],
    bank_manifest: list[MIMICStudy],
    label: str,
) -> None:
    if len(metadata) != len(bank_manifest):
        raise RuntimeError(
            f"{label} metadata row count {len(metadata)} does not match retrieval bank manifest count {len(bank_manifest)}."
        )
    metadata_ids = [str(row.get("study_id", "")).strip() for row in metadata]
    manifest_ids = [study.study_id for study in bank_manifest]
    if metadata_ids != manifest_ids:
        mismatches: list[str] = []
        for idx, (left, right) in enumerate(zip(metadata_ids, manifest_ids, strict=True)):
            if left != right:
                mismatches.append(f"idx={idx} metadata={left} manifest={right}")
                if len(mismatches) >= 10:
                    break
        raise RuntimeError(f"{label} study_id order mismatch: {mismatches}")


def count_statuses(paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            counts["missing"] = counts.get("missing", 0) + 1
            continue
        try:
            payload = read_json(path)
        except Exception:
            counts["invalid_json"] = counts.get("invalid_json", 0) + 1
            continue
        status = str(payload.get("status", "missing")).strip() if isinstance(payload, dict) else "invalid_payload"
        counts[status] = counts.get(status, 0) + 1
    return counts


def print_stage_summary(stage_label: str, paths: list[Path]) -> dict[str, int]:
    counts = count_statuses(paths)
    ordered = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"{stage_label} summary: {ordered}")
    return counts


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _load_external_chexone_report_payload(uid: str) -> dict[str, Any]:
    source_path = _resolve_existing_path(EXTERNAL_CHEXONE_REPORT_DIR / f"{uid}.json")
    payload = read_json(source_path)
    if not isinstance(payload, dict):
        raise ValueError(f"External CheXOne payload is not a JSON object: {source_path}")
    if str(payload.get("uid", "")).strip() != str(uid):
        raise ValueError(f"External CheXOne payload UID mismatch for {uid}: {source_path}")
    if payload.get("status") != "success":
        raise ValueError(f"External CheXOne payload is not successful for {uid}: {source_path}")
    return payload




_CHEXONE_REPORT_CHEXPERT_LABELS_CACHE: dict[str, Any] | None = None


def load_chexone_report_chexbert_vector(uid: str) -> dict[str, float]:
    global _CHEXONE_REPORT_CHEXPERT_LABELS_CACHE
    if _CHEXONE_REPORT_CHEXPERT_LABELS_CACHE is None:
        payload = read_json(CHEXONE_REPORT_CHEXPERT_LABELS_PATH)
        if not isinstance(payload, dict):
            raise RuntimeError(f"CheXbert labels must be a JSON object: {CHEXONE_REPORT_CHEXPERT_LABELS_PATH}")
        _CHEXONE_REPORT_CHEXPERT_LABELS_CACHE = payload
    row = _CHEXONE_REPORT_CHEXPERT_LABELS_CACHE.get(str(uid))
    if not isinstance(row, dict) or not isinstance(row.get("parsed_vector"), dict):
        raise RuntimeError(f"Missing CheXbert vector for CheXOne report uid={uid}: {CHEXONE_REPORT_CHEXPERT_LABELS_PATH}")
    return {finding: float(row["parsed_vector"].get(finding, 0.0)) for finding in CHEXPERT_FINDINGS}


def _normalize_external_report_payload(uid: str, image_path: str, payload: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    raw_response = str(payload.get("raw_output", "") or payload.get("raw_response", "") or "").strip()
    findings = str(payload.get("findings", "") or "").strip()
    impression = str(payload.get("impression", "") or "").strip()
    report_text = str(payload.get("report_text", "") or "").strip()

    if not report_text:
        parsed = parse_stage5_report(raw_response)
        findings = findings or parsed["findings"]
        impression = impression or parsed["impression"]
        report_text = parsed["report_text"]

    if not report_text:
        raise ValueError(f"External CheXOne payload has empty report text for uid={uid}")

    return {
        "uid": uid,
        "image_path": image_path,
        "model_id": CHEXONE_MODEL_ID,
        "prompt_hash": stable_hash({"model": CHEXONE_MODEL_ID, "prompt": prompt_text, "uid": uid}),
        "prompt_text": prompt_text,
        "cache_hit": True,
        "cache_source": "external_verified_chexone_reports",
        "raw_response": raw_response,
        "findings": findings,
        "impression": impression,
        "report_text": report_text,
        "status": "success",
        "created_at": str(payload.get("updated_at", utc_now())),
        "updated_at": utc_now(),
        "external_source_path": str((EXTERNAL_CHEXONE_REPORT_DIR / f"{uid}.json").resolve()),
        "parse_mode": str(payload.get("parse_mode", "external_import")),
    }


def build_external_synced_iu_eval_manifest(limit: int) -> list[IUStudy]:
    source_path = _resolve_existing_path(EXTERNAL_IU_EVAL_MANIFEST_SOURCE_PATH)
    source_rows = [IUStudy(**row) for row in read_json(source_path)]
    synced: list[IUStudy] = []

    for study in source_rows:
        study_image_path = _resolve_existing_path(study.image_path)
        chexone_payload = _load_external_chexone_report_payload(study.uid)
        chexone_image_path = _resolve_existing_path(str(chexone_payload.get("image_path", "")))
        if study_image_path != chexone_image_path:
            raise ValueError(
                f"Image path mismatch for uid={study.uid}: manifest={study_image_path} chexone={chexone_image_path}"
            )
        synced.append(
            IUStudy(
                uid=study.uid,
                image_path=str(study_image_path),
                findings=study.findings,
                impression=study.impression,
                ground_truth_report=study.ground_truth_report,
            )
        )
        if len(synced) >= limit:
            break

    if len(synced) < limit:
        raise RuntimeError(f"Only found {len(synced)} synced IU studies, expected at least {limit}.")
    return synced


if USE_EXTERNAL_VERIFIED_ARTIFACTS:
    IU_TRAIN_MANIFEST_PATH = SPLITS_DIR / "iu_train_manifest_external_canonical_2069.json"
    IU_EVAL_MANIFEST_PATH = SPLITS_DIR / f"iu_eval_manifest_external_first_{IU_EVAL_LIMIT}.json"

    iu_train_manifest = remap_iu_study_paths([IUStudy(**row) for row in read_json(_resolve_existing_path(EXTERNAL_IU_TRAIN_MANIFEST_PATH))])
    iu_eval_manifest = build_external_synced_iu_eval_manifest(IU_EVAL_LIMIT)
    validate_eval_manifest_alignment(iu_eval_manifest)
    write_json(IU_TRAIN_MANIFEST_PATH, [asdict(item) for item in iu_train_manifest])
    write_json(IU_EVAL_MANIFEST_PATH, [asdict(item) for item in iu_eval_manifest])
else:
    IU_TRAIN_MANIFEST_PATH = SPLITS_DIR / f"iu_train_manifest_holdout_{IU_TEST_COUNT}.json"
    IU_EVAL_MANIFEST_PATH = SPLITS_DIR / f"iu_test_manifest_{IU_TEST_COUNT}.json"

    if IU_TRAIN_MANIFEST_PATH.exists() and IU_EVAL_MANIFEST_PATH.exists():
        iu_train_manifest = [IUStudy(**row) for row in read_json(IU_TRAIN_MANIFEST_PATH)]
        iu_eval_manifest = [IUStudy(**row) for row in read_json(IU_EVAL_MANIFEST_PATH)]
    else:
        all_iu_studies = build_iu_study_table()
        iu_train_manifest, iu_eval_manifest = split_iu_train_test(all_iu_studies, IU_TEST_COUNT)
        write_json(IU_TRAIN_MANIFEST_PATH, [asdict(item) for item in iu_train_manifest])
        write_json(IU_EVAL_MANIFEST_PATH, [asdict(item) for item in iu_eval_manifest])

effective_iu_train_bank_target = (
    len(iu_train_manifest)
    if IU_TRAIN_BANK_TARGET <= 0
    else min(IU_TRAIN_BANK_TARGET, len(iu_train_manifest))
)

print(f"IU train manifest path: {IU_TRAIN_MANIFEST_PATH}")
print(f"IU test manifest path: {IU_EVAL_MANIFEST_PATH}")
print(f"IU train study count: {len(iu_train_manifest)}")
print(f"IU test study count: {len(iu_eval_manifest)}")
print(f"Effective IU train bank target: {effective_iu_train_bank_target}")
print(f"First test UID: {iu_eval_manifest[0].uid if iu_eval_manifest else 'n/a'}")


# %% [markdown]
# ## MIMIC bank manifest code

# %%
def iter_mimic_studies(csv_path: Path, split: str) -> list[MIMICStudy]:
    studies: list[MIMICStudy] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        for row in tqdm(rows, total=len(rows), desc=f"Scanning MIMIC {split}", unit="row"):
            image_paths = parse_list_cell(row.get("image"))
            report_texts = parse_list_cell(row.get("text"))
            ap_paths = set(parse_list_cell(row.get("AP")))
            pa_paths = set(parse_list_cell(row.get("PA")))
            lateral_paths = set(parse_list_cell(row.get("Lateral")))

            study_to_paths: dict[str, list[str]] = {}
            for rel_path in image_paths:
                match = STUDY_ID_PATTERN.search(rel_path)
                if match is None:
                    continue
                study_to_paths.setdefault(match.group(1), []).append(rel_path)

            ordered_ids = sorted(study_to_paths)
            if len(ordered_ids) != len(report_texts):
                continue

            subject_id = str(row["subject_id"])
            for study_id, report_text in zip(ordered_ids, report_texts, strict=True):
                chosen = choose_frontal_path(study_to_paths[study_id], ap_paths, pa_paths, lateral_paths)
                if chosen is None:
                    continue
                chosen_abs = (MIMIC_IMAGE_ROOT / chosen).resolve()
                if not chosen_abs.exists():
                    continue
                studies.append(
                    MIMICStudy(
                        subject_id=subject_id,
                        study_id=study_id,
                        image_path=str(chosen_abs),
                        all_image_paths=[str((MIMIC_IMAGE_ROOT / item).resolve()) for item in sorted(study_to_paths[study_id])],
                        report_text=str(report_text),
                        split=split,
                        source_dataset="mimic",
                    )
                )
    return studies


def build_mimic_bank_manifest(target_count: int) -> list[MIMICStudy]:
    selected: list[MIMICStudy] = []
    for study in iter_mimic_studies(MIMIC_TRAIN_CSV, split="train"):
        selected.append(study)
        if len(selected) >= target_count:
            break
    return selected


def build_iu_train_bank_manifest(train_studies: list[IUStudy], target_count: int) -> list[MIMICStudy]:
    selected: list[MIMICStudy] = []
    for study in train_studies[:target_count]:
        selected.append(
            MIMICStudy(
                subject_id=f"iu_{study.uid}",
                study_id=f"iu_{study.uid}",
                image_path=study.image_path,
                all_image_paths=[study.image_path],
                report_text=study.ground_truth_report,
                split="iu_train",
                source_dataset="iu",
            )
        )
    return selected


# %% [markdown]
# ## Retrieval bank manifest run

# %%
if USE_EXTERNAL_VERIFIED_ARTIFACTS:
    MIMIC_BANK_MANIFEST_PATH = _resolve_existing_path(EXTERNAL_MIMIC_BANK_MANIFEST_PATH)
    IU_TRAIN_BANK_MANIFEST_PATH = _resolve_existing_path(EXTERNAL_IU_BANK_MANIFEST_PATH)
    RETRIEVAL_BANK_MANIFEST_PATH = _resolve_existing_path(EXTERNAL_RETRIEVAL_BANK_MANIFEST_PATH)

    mimic_bank_manifest = remap_mimic_study_paths([MIMICStudy(**row) for row in read_json(MIMIC_BANK_MANIFEST_PATH)])
    iu_train_bank_manifest = remap_mimic_study_paths([MIMICStudy(**row) for row in read_json(IU_TRAIN_BANK_MANIFEST_PATH)])
    retrieval_bank_manifest = remap_mimic_study_paths([MIMICStudy(**row) for row in read_json(RETRIEVAL_BANK_MANIFEST_PATH)])
    validate_retrieval_bank_manifest_alignment(retrieval_bank_manifest)
else:
    MIMIC_BANK_MANIFEST_PATH = BANKS_DIR / f"mimic_bank_manifest_{MIMIC_BANK_TARGET}.json"
    IU_TRAIN_BANK_MANIFEST_PATH = BANKS_DIR / f"iu_train_bank_manifest_{effective_iu_train_bank_target}.json"
    RETRIEVAL_BANK_MANIFEST_PATH = BANKS_DIR / f"retrieval_bank_manifest_mimic_{MIMIC_BANK_TARGET}_iu_{effective_iu_train_bank_target}.json"

    if MIMIC_BANK_MANIFEST_PATH.exists():
        mimic_bank_manifest = [MIMICStudy(**row) for row in read_json(MIMIC_BANK_MANIFEST_PATH)]
    else:
        mimic_bank_manifest = build_mimic_bank_manifest(MIMIC_BANK_TARGET)
        write_json(MIMIC_BANK_MANIFEST_PATH, [asdict(item) for item in mimic_bank_manifest])

    if IU_TRAIN_BANK_MANIFEST_PATH.exists():
        iu_train_bank_manifest = [MIMICStudy(**row) for row in read_json(IU_TRAIN_BANK_MANIFEST_PATH)]
    else:
        iu_train_bank_manifest = build_iu_train_bank_manifest(iu_train_manifest, effective_iu_train_bank_target)
        write_json(IU_TRAIN_BANK_MANIFEST_PATH, [asdict(item) for item in iu_train_bank_manifest])

    if RETRIEVAL_BANK_MANIFEST_PATH.exists():
        retrieval_bank_manifest = [MIMICStudy(**row) for row in read_json(RETRIEVAL_BANK_MANIFEST_PATH)]
    else:
        retrieval_bank_manifest = list(mimic_bank_manifest) + list(iu_train_bank_manifest)
        write_json(RETRIEVAL_BANK_MANIFEST_PATH, [asdict(item) for item in retrieval_bank_manifest])

print(f"MIMIC bank manifest path: {MIMIC_BANK_MANIFEST_PATH}")
print(f"MIMIC bank size: {len(mimic_bank_manifest)}")
print(f"IU train bank manifest path: {IU_TRAIN_BANK_MANIFEST_PATH}")
print(f"IU train bank size: {len(iu_train_bank_manifest)}")
print(f"Combined retrieval bank manifest path: {RETRIEVAL_BANK_MANIFEST_PATH}")
print(f"Combined retrieval bank size: {len(retrieval_bank_manifest)}")
print(f"First retrieval study id: {retrieval_bank_manifest[0].study_id if retrieval_bank_manifest else 'n/a'}")


# %% [markdown]
# ## Shared model and parsing code

# %%
def _column_aliases(name: str) -> tuple[str, ...]:
    compact = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    return (name, name.lower(), name.replace(" ", "_"), name.replace(" ", "-"), compact)


def _resolve_score(row: dict[str, str], finding: str) -> float | None:
    for key in _column_aliases(finding):
        if key in row and row[key] != "":
            try:
                return float(row[key])
            except ValueError:
                return None
    return None


def load_prediction_overrides(model_dir: str | None) -> dict[str, dict[str, float]]:
    if not model_dir:
        return {}
    root = Path(model_dir)
    source = next(
        (candidate for candidate in [root / "predictions.csv", root / "predictions.jsonl", root / "predictions.json"] if candidate.exists()),
        None,
    )
    if source is None:
        raise FileNotFoundError(f"Configured prediction override directory has no predictions file: {root}")

    rows: list[dict[str, str]] = []
    if source.suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        rows = read_json(source)

    overrides: dict[str, dict[str, float]] = {}
    for row in rows:
        study_id = row.get("study_id") or row.get("id") or row.get("case_id")
        if not study_id:
            continue
        overrides[str(study_id)] = {
            finding: _resolve_score(row, finding) or 0.0 for finding in CHEXPERT_FINDINGS
        }
    return overrides


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _lookup_mapping_key(mapping: dict[str, Any], aliases: list[str]) -> str | None:
    alias_keys = {_compact_key(alias) for alias in aliases}
    for key in mapping:
        if _compact_key(key) in alias_keys:
            return key
    return None


def parse_stage7_judge(text: str) -> dict[str, Any]:
    parsed = extract_json_object(text)
    required_scores = [
        "clinical_accuracy_score",
        "groundedness_score",
        "completeness_score",
        "style_score",
        "overall_score",
    ]
    for key in required_scores:
        if key not in parsed:
            raise ValueError(f"Missing judge score: {key}")
        parsed[key] = float(parsed[key])
    flags = parsed.get("hallucination_flags", [])
    if not isinstance(flags, list):
        raise ValueError("hallucination_flags must be a list.")
    parsed["hallucination_flags"] = [str(item) for item in flags]
    parsed["brief_rationale"] = str(parsed.get("brief_rationale", "")).strip()
    return parsed


def parse_stage5_report(text: str) -> dict[str, str]:
    cleaned = str(text).replace("**", "").strip()
    forbidden_patterns = [
        r"\bdisclaimer\b",
        r"\bimportant disclaimer",
        r"\bsample report\b",
        r"\bnot (?:be )?used for actual patient care\b",
        r"\bqualified radiologist\b",
        r"\bmedical advice\b",
        r"\brecommendations?\s*:",
        r"\bpatient\s*:",
        r"\bpatient id\s*:",
        r"\bdate of exam\s*:",
        r"\breferring physician\s*:",
        r"\bradiologist\s*:",
        r"\bcredentials\s*:",
        r"\bclinical indication\s*:",
        r"\btechnique\s*:",
        r"\bexam\s*:",
        r"\bcardiothoracic ratio is estimated\b",
        r"\[[^\]]+\]",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            raise ValueError(f"Stage 5 response contains forbidden non-IU report boilerplate: {pattern}")
    findings_match = re.search(r"(?is)Findings:\s*(.*?)(?=\bImpression:\s*|$)", cleaned)
    impression_match = re.search(r"(?is)Impression:\s*(.*)$", cleaned)
    findings = re.sub(r"\s+", " ", findings_match.group(1)).strip() if findings_match else ""
    impression = re.sub(r"\s+", " ", impression_match.group(1)).strip() if impression_match else ""
    if not findings and not impression:
        raise ValueError("Stage 5 response did not contain Findings/Impression sections.")
    if len(findings.split()) > 120 or len(impression.split()) > 60:
        raise ValueError("Stage 5 response is too long for IU-style output.")
    report_text = "\n".join(
        line for line in [f"Findings: {findings}" if findings else "", f"Impression: {impression}" if impression else ""]
        if line
    )
    return {"findings": findings, "impression": impression, "report_text": report_text}


def normalize_report_text_for_metrics(text: str) -> str:
    cleaned = str(text).replace("**", "").strip()
    if not cleaned:
        return ""

    findings_match = re.search(r"(?is)Findings:\s*(.*?)(?=\bImpression:\s*|$)", cleaned)
    impression_match = re.search(r"(?is)Impression:\s*(.*)$", cleaned)
    findings = re.sub(r"\s+", " ", findings_match.group(1)).strip() if findings_match else ""
    impression = re.sub(r"\s+", " ", impression_match.group(1)).strip() if impression_match else ""

    if findings or impression:
        parts = [part for part in [findings, impression] if part]
        combined = ". ".join(part.rstrip(" .") for part in parts if part).strip()
    else:
        combined = cleaned

    combined = re.sub(r"\[[^\]]+\]\s*", "", combined)
    combined = re.sub(r"(?i)\bfindings:\s*", " ", combined)
    combined = re.sub(r"(?i)\bimpression:\s*", " ", combined)
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("Zero-norm vector cannot be normalized.")
    return [value / norm for value in vector]


class SimpleFaissIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self._faiss_index = None

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)

    def build(self) -> None:
        import faiss  # type: ignore
        import numpy as np  # type: ignore

        if not self.embeddings:
            self._faiss_index = None
            return
        matrix = np.asarray(self.embeddings, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._faiss_index = index

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        if self._faiss_index is None:
            self.build()
        import faiss  # type: ignore

        faiss.write_index(self._faiss_index, str(path))

    @classmethod
    def load(cls, path: Path, ids: list[str], embeddings: list[list[float]]) -> "SimpleFaissIndex":
        index = cls()
        index.add(ids, embeddings)
        if not path.exists():
            index.build()
            return index

        import faiss  # type: ignore

        loaded = faiss.read_index(str(path))
        index._faiss_index = loaded
        return index

    def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        if self._faiss_index is not None:
            import numpy as np  # type: ignore

            query = np.asarray([embedding], dtype="float32")
            scores, indices = self._faiss_index.search(query, top_k)
            hits: list[tuple[str, float]] = []
            for score, idx in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
                if idx < 0:
                    continue
                hits.append((self.ids[idx], float(score)))
            return hits
        scored = [(item_id, cosine_similarity(embedding, candidate)) for item_id, candidate in zip(self.ids, self.embeddings, strict=True)]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def save_numpy(path: Path, rows: list[list[float]]) -> None:
    import numpy as np  # type: ignore

    ensure_dir(path.parent)
    np.save(path, np.asarray(rows, dtype="float32"))


def load_numpy(path: Path) -> list[list[float]]:
    import numpy as np  # type: ignore

    array = np.load(path)
    return array.tolist()


def load_resized_image(image_path: str, max_dim: int) -> Any:
    from PIL import Image  # type: ignore

    image = Image.open(image_path).convert("RGB")
    if max(image.size) <= max_dim:
        return image
    resized = image.copy()
    resized.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return resized


# %% [markdown]
# ## Stage 1 code: unified CheXOne import or generation
#
# Purpose:
# - produce one image-first report per IU study
# - derive one deterministic 14-label proxy from that same report text
#
# Input:
# - one frontal IU image
# - either a verified external CheXOne JSON or a live CheXOne model call
#
# Output:
# - one Stage 1 JSON with `raw_response`, `findings`, `impression`,
#   `report_text`, and `parsed_vector`

# %%
LOCAL_STAGE1_PROMPT = build_stage1_prompt()
CHEXONE_DIRECT_REPORT_PROMPT = LOCAL_STAGE1_PROMPT


class CheXOneModel:
    def __init__(self, model_id: str, device: str, local_only: bool = False) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
        import torch  # type: ignore

        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            LOCAL_CHEXONE_SNAPSHOT_DIR or model_id,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            LOCAL_CHEXONE_SNAPSHOT_DIR or model_id,
            trust_remote_code=True,
            local_files_only=local_only,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map={"": device},
        )

    def infer(self, image_path: str, prompt_text: str) -> str:
        image = load_resized_image(image_path, QUERY_IMAGE_MAX_DIM)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with __import__("torch").inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        input_token_count = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[:, input_token_count:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


_CHEXONE_MODEL_CACHE: dict[tuple[str, str, bool], CheXOneModel] = {}


def get_chexone_model(model_id: str, device: str, local_only: bool) -> CheXOneModel:
    key = (model_id, device, local_only)
    if key not in _CHEXONE_MODEL_CACHE:
        _CHEXONE_MODEL_CACHE[key] = CheXOneModel(model_id=model_id, device=device, local_only=local_only)
    return _CHEXONE_MODEL_CACHE[key]


def stage1_output_path(uid: str) -> Path:
    return STAGE1_DIR / f"{uid}.json"


def chexone_direct_output_path(uid: str) -> Path:
    return CHEXONE_DIRECT_DIR / f"{uid}.json"


def load_stage1_success(uid: str) -> dict[str, Any] | None:
    path = stage1_output_path(uid)
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "success" or payload.get("model_id") != CHEXONE_MODEL_ID:
        return None
    if str(payload.get("cache_source", "")).strip() == "external_verified_chexone_reports":
        return payload
    expected_prompt_hash = stable_hash({"model": CHEXONE_MODEL_ID, "prompt": LOCAL_STAGE1_PROMPT, "uid": uid})
    if payload.get("prompt_hash") == expected_prompt_hash:
        return payload
    return None


def import_external_stage1(studies: list[IUStudy]) -> None:
    if not USE_EXTERNAL_VERIFIED_ARTIFACTS:
        return

    imported_count = 0
    for study in studies:
        external_payload = _load_external_chexone_report_payload(study.uid)
        external_image_path = _resolve_existing_path(str(external_payload.get("image_path", "")))
        study_image_path = _resolve_existing_path(study.image_path)
        if external_image_path != study_image_path:
            raise ValueError(
                f"External CheXOne import image mismatch for uid={study.uid}: {external_image_path} vs {study_image_path}"
            )

        stage1_payload = _normalize_external_report_payload(
            uid=study.uid,
            image_path=str(study_image_path),
            payload=external_payload,
            prompt_text=str(external_payload.get("prompt_text", "")).strip() or LOCAL_STAGE1_PROMPT,
        )
        report_text = str(stage1_payload.get("report_text", "")).strip()
        if not report_text:
            raise RuntimeError(f"Imported Stage 1 report text is empty for uid={study.uid}")
        stage1_payload["parsed_vector"] = load_chexone_report_chexbert_vector(study.uid)
        stage1_payload["parse_diagnostics"] = {
            "source": "chexbert_on_external_chexone_report_prompt",
            "report_text_chars": len(report_text),
            "chexbert_labels_path": str(CHEXONE_REPORT_CHEXPERT_LABELS_PATH),
        }
        stage1_payload["cache_source"] = "external_verified_chexone_reports"

        write_json(stage1_output_path(study.uid), stage1_payload)
        imported_count += 1

    print(f"Imported external Stage 1 payloads: {imported_count}")


import_external_stage1(iu_eval_manifest)


# %% [markdown]
# ## Stage 1 run

# %%
pending_stage1 = [study for study in iu_eval_manifest if FORCE_STAGE1_REFRESH or load_stage1_success(study.uid) is None]

print(f"Stage 1 pending studies: {len(pending_stage1)}")

if USE_EXTERNAL_VERIFIED_ARTIFACTS and pending_stage1:
    missing_ids = [study.uid for study in pending_stage1[:10]]
    raise RuntimeError(
        "External-artifact mode requires Stage 1 to be satisfied from imported artifacts. "
        f"Missing Stage 1 payloads for UIDs: {missing_ids}"
    )

if pending_stage1:
    chexone_model = get_chexone_model(
        model_id=CHEXONE_MODEL_ID,
        device=DEVICE,
        local_only=LOCAL_FILES_ONLY,
    )

    for index, study in enumerate(
        tqdm(pending_stage1, total=len(pending_stage1), desc="Stage 1", unit="study"),
        start=1,
    ):
        prompt_hash = stable_hash({"model": CHEXONE_MODEL_ID, "prompt": LOCAL_STAGE1_PROMPT, "uid": study.uid})
        path = stage1_output_path(study.uid)
        print(f"[Stage 1] {index}/{len(pending_stage1)} uid={study.uid}")
        payload = {
            "uid": study.uid,
            "image_path": study.image_path,
            "model_id": CHEXONE_MODEL_ID,
            "prompt_hash": prompt_hash,
            "prompt_text": LOCAL_STAGE1_PROMPT,
            "cache_hit": False,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        write_json(path, payload)
        raw_response = ""
        try:
            raw_response = chexone_model.infer(study.image_path, LOCAL_STAGE1_PROMPT)
            parsed_report = parse_stage5_report(raw_response)
            report_text = parsed_report["report_text"] or raw_response
            payload.update(
                {
                    "raw_response": raw_response,
                    "findings": parsed_report["findings"],
                    "impression": parsed_report["impression"],
                    "report_text": parsed_report["report_text"],
                    "status": "success",
                    "updated_at": utc_now(),
                }
            )
            payload["parsed_vector"] = load_chexone_report_chexbert_vector(study.uid)
            payload["parse_diagnostics"] = {
                "source": "chexbert_on_external_chexone_report_prompt",
                "report_text_chars": len(report_text),
                "chexbert_labels_path": str(CHEXONE_REPORT_CHEXPERT_LABELS_PATH),
            }
        except Exception as exc:
            payload.update(
                {
                    "status": "failed_permanent",
                    "raw_response": raw_response,
                    "error": repr(exc),
                    "updated_at": utc_now(),
                }
            )
        write_json(path, payload)

print_stage_summary("Stage 1", [stage1_output_path(study.uid) for study in iu_eval_manifest])


# %% [markdown]
# ## CheXOne direct-report baseline code

# %%
pending_chexone_direct = [
    study
    for study in iu_eval_manifest
    if FORCE_CHEXONE_DIRECT_REFRESH or not artifact_has_success_status(chexone_direct_output_path(study.uid))
]

print(f"CheXOne direct-report pending studies: {len(pending_chexone_direct)}")

if USE_EXTERNAL_VERIFIED_ARTIFACTS and pending_chexone_direct:
    unsatisfied_ids = [
        study.uid
        for study in pending_chexone_direct
        if load_stage1_success(study.uid) is None
    ]
    if unsatisfied_ids:
        raise RuntimeError(
            "External-artifact mode requires direct CheXOne reports to be satisfied from Stage 1 reuse or imported artifacts. "
            f"Missing direct-report inputs for UIDs: {unsatisfied_ids[:10]}"
        )

if pending_chexone_direct:
    chexone_model: CheXOneModel | None = None

    for index, study in enumerate(
        tqdm(pending_chexone_direct, total=len(pending_chexone_direct), desc="CheXOne direct", unit="study"),
        start=1,
    ):
        path = chexone_direct_output_path(study.uid)
        print(f"[CheXOne Direct] {index}/{len(pending_chexone_direct)} uid={study.uid}")
        stage1_payload = load_stage1_success(study.uid)
        if stage1_payload is not None:
            payload = {
                "uid": study.uid,
                "image_path": study.image_path,
                "model_id": CHEXONE_MODEL_ID,
                "prompt_hash": stage1_payload["prompt_hash"],
                "prompt_text": CHEXONE_DIRECT_REPORT_PROMPT,
                "cache_hit": True,
                "cache_source": "stage1_reuse",
                "raw_response": stage1_payload.get("raw_response", ""),
                "findings": stage1_payload.get("findings", ""),
                "impression": stage1_payload.get("impression", ""),
                "report_text": stage1_payload.get("report_text", ""),
                "status": "success",
                "created_at": stage1_payload.get("created_at", utc_now()),
                "updated_at": utc_now(),
            }
        else:
            if chexone_model is None:
                chexone_model = get_chexone_model(
                    model_id=CHEXONE_MODEL_ID,
                    device=DEVICE,
                    local_only=LOCAL_FILES_ONLY,
                )
            prompt_hash = stable_hash({"model": CHEXONE_MODEL_ID, "prompt": CHEXONE_DIRECT_REPORT_PROMPT, "uid": study.uid})
            payload = {
                "uid": study.uid,
                "image_path": study.image_path,
                "model_id": CHEXONE_MODEL_ID,
                "prompt_hash": prompt_hash,
                "prompt_text": CHEXONE_DIRECT_REPORT_PROMPT,
                "cache_hit": False,
                "status": "running",
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            write_json(path, payload)
            raw_response = ""
            try:
                raw_response = chexone_model.infer(study.image_path, CHEXONE_DIRECT_REPORT_PROMPT)
                parsed_report = parse_stage5_report(raw_response)
                payload.update(
                    {
                        "raw_response": raw_response,
                        "findings": parsed_report["findings"],
                        "impression": parsed_report["impression"],
                        "report_text": parsed_report["report_text"],
                        "status": "success",
                        "updated_at": utc_now(),
                    }
                )
            except Exception as exc:
                payload.update(
                    {
                        "status": "failed_permanent",
                        "raw_response": raw_response,
                        "error": repr(exc),
                        "updated_at": utc_now(),
                    }
                )
        write_json(path, payload)

print_stage_summary("CheXOne direct", [chexone_direct_output_path(study.uid) for study in iu_eval_manifest])


# %% [markdown]
# ## Stage 2a code: CLFIR-adapter-2 visual retrieval bank
#
# Purpose:
# - embed the query image with BioViL-T
# - project the 128d BioViL-T vector through CLFIR adapter-2
# - retrieve visually/textually aligned studies from the CLFIR FAISS bank
#
# Input:
# - the Stage 1 study image
# - the external CLFIR adapter-2 visual bank
#
# Output:
# - one Stage 2a JSON with top visual hits
#
# In external-artifact mode, this is the first stage that still requires model
# compute. We are no longer running CheXOne here. The only remaining inference
# is query-image embedding for the IU eval image so the pipeline can retrieve
# visually similar bank images. Those query embeddings are cached on disk.

# %%
def build_visual_bank_paths() -> dict[str, Path]:
    root = EXTERNAL_CLFIR_ADAPTER2_VISUAL_BANK_DIR if USE_EXTERNAL_VERIFIED_ARTIFACTS else BANKS_DIR / "visual_clfir"
    return {
        "embeddings": root / "clfir_visual_embeddings.npy",
        "index": root / "clfir_visual.index",
        "metadata": root / "clfir_visual_metadata.json",
        "index_manifest": root / "clfir_visual_manifest.json",
        "adapter_checkpoint": root / "clfir_adapter_best.pt",
    }


def resolve_biovilt_weights_path() -> Path:
    explicit_path = Path(LOCAL_BIOVILT_WEIGHTS_PATH).expanduser().resolve() if LOCAL_BIOVILT_WEIGHTS_PATH else None
    candidates = [candidate for candidate in [explicit_path, MANAGED_BIOVILT_WEIGHTS_PATH] if candidate is not None]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if not AUTO_DOWNLOAD_BIOVILT_WEIGHTS:
        raise FileNotFoundError(
            "BioViL-T weights were not found locally. "
            f"Set `LOCAL_BIOVILT_WEIGHTS_PATH` or place weights at `{MANAGED_BIOVILT_WEIGHTS_PATH}`."
        )

    target = explicit_path or MANAGED_BIOVILT_WEIGHTS_PATH
    print(f"Downloading BioViL-T weights to: {target}")
    try:
        download_to_path(BIOVILT_WEIGHTS_URL, target)
    except Exception as exc:
        raise RuntimeError(
            "BioViL-T weights download failed. "
            f"Tried URL `{BIOVILT_WEIGHTS_URL}` and target `{target}`. "
            "If you already have the checkpoint locally, set `LOCAL_BIOVILT_WEIGHTS_PATH`."
        ) from exc
    print("BioViL-T weights download complete.")
    return target


class BioViLTEmbedder:
    def __init__(self) -> None:
        try:
            from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference  # type: ignore
            from health_multimodal.image.inference_engine import ImageInferenceEngine  # type: ignore
            from health_multimodal.image.model.model import ImageModel  # type: ignore
            from health_multimodal.image.model.pretrained import JOINT_FEATURE_SIZE  # type: ignore
            from health_multimodal.image.model.types import ImageEncoderType  # type: ignore
            from health_multimodal.image.utils import TRANSFORM_RESIZE  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "BioViL-T runtime is unavailable. Stage 2a requires local BioViL-T weights or a working networked setup."
            ) from exc

        try:
            weights_path = resolve_biovilt_weights_path()
            image_model = ImageModel(
                img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                joint_feature_size=JOINT_FEATURE_SIZE,
                pretrained_model_path=weights_path,
            )
            transform = create_chest_xray_transform_for_inference(
                resize=TRANSFORM_RESIZE,
                center_crop_size=448,
            )
            self.engine = ImageInferenceEngine(image_model=image_model, transform=transform)
            self.weights_path = str(weights_path)
        except Exception as exc:
            raise RuntimeError(
                "BioViL-T could not be initialized. In external-artifact mode this is the only remaining required model "
                "compute stage, used to embed the eval query image for visual retrieval."
            ) from exc

    def embed_image(self, image_path: str) -> list[float]:
        import numpy as np  # type: ignore

        embedding = self.engine.get_projected_global_embedding(Path(image_path))
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
        return normalize_vector([float(item) for item in vector])


class CLFIRImageProjector:
    def __init__(self) -> None:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        paths = build_visual_bank_paths()
        if not paths["adapter_checkpoint"].exists():
            raise FileNotFoundError(f"Missing CLFIR adapter checkpoint: {paths['adapter_checkpoint']}")
        manifest = read_json(paths["index_manifest"])
        image_dim = int(manifest.get("image_embedding_dim", 128))
        projected_dim = int(manifest.get("projected_dim", 512))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(image_dim, projected_dim),
            nn.LayerNorm(projected_dim),
        ).to(self.device)

        checkpoint = torch.load(paths["adapter_checkpoint"], map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        projection_state = {
            key.replace("image_projection.", "", 1): value
            for key, value in state.items()
            if str(key).startswith("image_projection.")
        }
        missing, unexpected = self.model.load_state_dict(projection_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Could not load CLFIR image projection cleanly. "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.model.eval()
        self.checkpoint_path = str(paths["adapter_checkpoint"])

    def project(self, biovilt_embedding: list[float]) -> list[float]:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        with torch.inference_mode():
            tensor = torch.tensor([biovilt_embedding], dtype=torch.float32, device=self.device)
            projected = F.normalize(self.model(tensor), dim=-1)
        return projected.squeeze(0).detach().cpu().numpy().astype("float32").tolist()


def stage2a_query_embedding_path(uid: str) -> Path:
    return STAGE2A_QUERY_EMBED_DIR / f"{uid}.json"


def load_cached_stage2a_query_embedding(uid: str) -> list[float] | None:
    path = stage2a_query_embedding_path(uid)
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return None
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        return None
    return [float(item) for item in vector]


def get_or_compute_stage2a_query_embedding(study: IUStudy, embedder: BioViLTEmbedder, projector: CLFIRImageProjector) -> list[float]:
    cached = load_cached_stage2a_query_embedding(study.uid)
    if cached is not None:
        return cached
    raw_biovilt_vector = embedder.embed_image(study.image_path)
    vector = projector.project(raw_biovilt_vector)
    write_json(
        stage2a_query_embedding_path(study.uid),
        {
            "uid": study.uid,
            "image_path": study.image_path,
            "raw_biovilt_embedding_length": len(raw_biovilt_vector),
            "embedding_length": len(vector),
            "embedding": vector,
            "embedding_model": "BioViL-T + CLFIR adapter-2 image projection",
            "adapter_checkpoint": projector.checkpoint_path,
            "status": "success",
            "updated_at": utc_now(),
        },
    )
    return vector


def load_visual_bank() -> tuple[SimpleFaissIndex, list[dict[str, Any]], dict[str, Any]]:
    paths = build_visual_bank_paths()
    manifest = read_json(paths["index_manifest"])
    metadata = read_json(paths["metadata"])
    if isinstance(metadata, list):
        for row in metadata:
            if isinstance(row, dict):
                if "image_path" in row:
                    row["image_path"] = str(map_external_dataset_path(str(row["image_path"])))
                if "all_image_paths" in row and isinstance(row["all_image_paths"], list):
                    row["all_image_paths"] = [str(map_external_dataset_path(str(item))) for item in row["all_image_paths"]]
    validate_bank_metadata_alignment(metadata, retrieval_bank_manifest, "Visual bank")
    embeddings = load_numpy(paths["embeddings"])
    index = SimpleFaissIndex.load(paths["index"], [row["study_id"] for row in metadata], embeddings)
    if not paths["index"].exists():
        index.save(paths["index"])
    return index, metadata, manifest


def load_or_build_visual_bank() -> tuple[SimpleFaissIndex, list[dict[str, Any]], dict[str, Any]]:
    paths = build_visual_bank_paths()
    if USE_EXTERNAL_VERIFIED_ARTIFACTS:
        required = [paths["embeddings"], paths["metadata"], paths["index_manifest"]]
        missing = [str(path) for path in required if not file_is_nonempty(path)]
        if missing:
            raise FileNotFoundError("Missing external visual bank artifacts:\n" + "\n".join(missing))
        manifest = read_json(paths["index_manifest"])
        if not isinstance(manifest, dict) or int(manifest.get("actual_count", 0)) < len(retrieval_bank_manifest):
            raise RuntimeError(
                "External visual bank manifest count is smaller than the retrieval bank manifest count."
            )
        return load_visual_bank()
    if (
        not FORCE_REBUILD_BANKS
        and file_is_nonempty(paths["embeddings"])
        and file_is_nonempty(paths["metadata"])
        and file_is_nonempty(paths["index_manifest"])
    ):
        manifest = read_json(paths["index_manifest"])
        if isinstance(manifest, dict) and int(manifest.get("actual_count", 0)) >= len(retrieval_bank_manifest):
            return load_visual_bank()

    existing_embeddings = load_numpy(paths["embeddings"]) if paths["embeddings"].exists() else []
    existing_metadata = read_json(paths["metadata"]) if paths["metadata"].exists() else []
    if not isinstance(existing_metadata, list):
        existing_metadata = []

    start_index = len(existing_metadata)
    embedder = BioViLTEmbedder()

    pending_bank_studies = retrieval_bank_manifest[start_index:]
    for idx, study in enumerate(
        tqdm(pending_bank_studies, total=len(pending_bank_studies), desc="Building visual bank", unit="study"),
        start=start_index,
    ):
        vector = embedder.embed_image(study.image_path)
        existing_embeddings.append(vector)
        existing_metadata.append(
            {
                "subject_id": study.subject_id,
                "study_id": study.study_id,
                "image_path": study.image_path,
                "all_image_paths": study.all_image_paths,
                "report_text": study.report_text,
                "split": study.split,
                "source_dataset": study.source_dataset,
            }
        )
        if (idx + 1) % BANK_CHECKPOINT_EVERY == 0 or (idx + 1) == len(retrieval_bank_manifest):
            checkpoint_index = SimpleFaissIndex()
            checkpoint_index.add([row["study_id"] for row in existing_metadata], existing_embeddings)
            checkpoint_index.build()
            save_numpy(paths["embeddings"], existing_embeddings)
            checkpoint_index.save(paths["index"])
            write_json(paths["metadata"], existing_metadata)
            write_json(
                paths["index_manifest"],
                {
                    "target_count": len(retrieval_bank_manifest),
                    "actual_count": len(existing_metadata),
                    "model_name": BIOVILT_MODEL_ID,
                    "completed": len(existing_metadata) >= len(retrieval_bank_manifest),
                    "updated_at": utc_now(),
                },
            )

    return load_visual_bank()


def search_visual_bank(index: SimpleFaissIndex, metadata: list[dict[str, Any]], query_embedding: list[float], top_k: int) -> list[RetrievalHit]:
    metadata_by_id = {row["study_id"]: row for row in metadata}
    hits: list[RetrievalHit] = []
    search_k = min(len(metadata), max(top_k, top_k * REPORT_QUALITY_OVERFETCH_FACTOR))
    for rank, (study_id, score) in enumerate(index.search(query_embedding, search_k), start=1):
        row = metadata_by_id[study_id]
        quality = report_quality(str(row["report_text"]))
        hits.append(
            RetrievalHit(
                rank=rank,
                score=float(score),
                subject_id=str(row["subject_id"]),
                study_id=str(row["study_id"]),
                image_path=str(row["image_path"]),
                report_text=str(row["report_text"]),
                split=str(row["split"]),
                source_dataset=str(row.get("source_dataset", "mimic")),
                visual_score=float(score),
                report_quality=quality,
            )
        )
    valid_hits = [hit for hit in hits if hit.report_quality.get("has_clinical_content")]
    invalid_hits = [hit for hit in hits if not hit.report_quality.get("has_clinical_content")]
    return (valid_hits + invalid_hits)[:top_k]


# %% [markdown]
# ## Stage 2a run

# %%
visual_bank_index, visual_bank_metadata, visual_bank_manifest = load_or_build_visual_bank()
print(f"Stage 2a bank size: {visual_bank_manifest['actual_count']}")

def retrieval_hits_have_quality(path: Path) -> bool:
    if not artifact_has_success_status(path):
        return False
    payload = read_json(path)
    hits = payload.get("hits", []) if isinstance(payload, dict) else []
    return isinstance(hits, list) and all(isinstance(row, dict) and isinstance(row.get("report_quality"), dict) for row in hits)


pending_stage2a = [study for study in iu_eval_manifest if not retrieval_hits_have_quality(STAGE2A_DIR / f"{study.uid}.json")]
print(f"Stage 2a pending studies: {len(pending_stage2a)}")

if pending_stage2a:
    query_embedder = BioViLTEmbedder()
    query_projector = CLFIRImageProjector()
    for index, study in enumerate(
        tqdm(pending_stage2a, total=len(pending_stage2a), desc="Stage 2a", unit="study"),
        start=1,
    ):
        print(f"[Stage 2a] {index}/{len(pending_stage2a)} uid={study.uid}")
        path = STAGE2A_DIR / f"{study.uid}.json"
        try:
            query_embedding = get_or_compute_stage2a_query_embedding(study, query_embedder, query_projector)
            hits = search_visual_bank(visual_bank_index, visual_bank_metadata, query_embedding, RETRIEVAL_TOP_K)
            write_json(
                path,
                {
                    "uid": study.uid,
                    "image_path": study.image_path,
                    "retrieval_model": "BioViL-T query embedding + CLFIR adapter-2 projected bank",
                    "query_embedding_length": len(query_embedding),
                    "query_embedding_path": str(stage2a_query_embedding_path(study.uid)),
                    "hits": [asdict(hit) for hit in hits],
                    "status": "success",
                    "updated_at": utc_now(),
                },
            )
        except Exception as exc:
            write_json(path, {"uid": study.uid, "status": "failed_permanent", "error": repr(exc), "updated_at": utc_now()})

print_stage_summary("Stage 2a", [STAGE2A_DIR / f"{study.uid}.json" for study in iu_eval_manifest])


# %% [markdown]
# ## CheXbert label store
#
# Purpose:
# - load CheXbert labels for retrieved reports
# - provide coarse text-side evidence to Stage 3/4
#
# This is not a retrieval stage. It only reads stored labels for reports that
# were already selected by CLFIR visual retrieval.

# %%
CHEXBERT_OVERRIDES = load_prediction_overrides(LOCAL_CHEXBERT_PREDICTIONS_PATH or None)


def _chexbert_raw_value_to_score(value: object) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        numeric = float(text)
    except ValueError:
        lowered = normalize_text(text)
        if lowered in {"positive", "present"}:
            return 1.0
        if lowered in {"uncertain"}:
            return 0.5
        return 0.0
    if numeric > 0:
        return 1.0
    if numeric < 0:
        return 0.5
    return 0.0


def bank_row_chexbert_labels(row: dict[str, Any]) -> dict[str, float]:
    if isinstance(row.get("labels"), dict) and row["labels"]:
        return {finding: float(row["labels"].get(finding, 0.0) or 0.0) for finding in CHEXPERT_FINDINGS}
    if isinstance(row.get("chexbert_raw"), dict) and row["chexbert_raw"]:
        return {finding: _chexbert_raw_value_to_score(row["chexbert_raw"].get(finding, "")) for finding in CHEXPERT_FINDINGS}
    if isinstance(row.get("state_labels"), dict) and row["state_labels"]:
        state_scores = {"positive": 1.0, "uncertain": 0.5, "negative": 0.0, "blank": 0.0}
        return {finding: state_scores.get(str(row["state_labels"].get(finding, "blank")).strip().lower(), 0.0) for finding in CHEXPERT_FINDINGS}
    return {finding: 0.0 for finding in CHEXPERT_FINDINGS}


def chexbert_vector(study_id: str, report_text: str) -> dict[str, float]:
    labels = CHEXBERT_OVERRIDES.get(study_id)
    if labels is None:
        labels = globals().get("PATHOLOGY_BANK_LABELS_BY_ID", {}).get(study_id)
    if labels is None:
        raise KeyError(
            f"No CheXbert labels available for study_id={study_id}. "
            "Provide LOCAL_CHEXBERT_PREDICTIONS_PATH or the external CheXbert metadata file."
        )
    return {finding: float(labels[finding]) for finding in CHEXPERT_FINDINGS}


def load_chexbert_bank_labels() -> dict[str, dict[str, float]]:
    if not file_is_nonempty(EXTERNAL_CHEXBERT_METADATA_PATH):
        raise FileNotFoundError(f"Missing CheXbert metadata: {EXTERNAL_CHEXBERT_METADATA_PATH}")
    metadata = read_json(EXTERNAL_CHEXBERT_METADATA_PATH)
    if not isinstance(metadata, list):
        raise RuntimeError(f"CheXbert metadata must be a list: {EXTERNAL_CHEXBERT_METADATA_PATH}")
    labels_by_id: dict[str, dict[str, float]] = {}
    for row in metadata:
        if not isinstance(row, dict):
            continue
        study_id = str(row.get("study_id", "")).strip()
        if not study_id or not report_quality(str(row.get("report_text", "")))["has_clinical_content"]:
            continue
        labels_by_id[study_id] = bank_row_chexbert_labels(row)
    return labels_by_id


PATHOLOGY_BANK_LABELS_BY_ID = {
    **load_chexbert_bank_labels(),
    **CHEXBERT_OVERRIDES,
}
print(f"CheXbert label store size: {len(PATHOLOGY_BANK_LABELS_BY_ID)}")


# %% [markdown]
# ## Stage 2c code: deterministic reranker
#
# Purpose:
# - rerank CLFIR visual hits into one final shortlist
#
# Input:
# - Stage 2a hits
#
# Output:
# - one Stage 2c JSON with reranked retrieval results

# %%
def rerank_hits(visual_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_hits: list[dict[str, Any]] = []
    for row in visual_hits:
        item = dict(row)
        item["retrieval_source_group"] = "visual"
        item["report_quality"] = item.get("report_quality") or report_quality(str(item.get("report_text", "")))
        item["visual_score"] = float(item.get("visual_score", item.get("score", 0.0)) or 0.0)
        score = item["visual_score"]
        if not hit_has_clinical_content(row):
            score -= 10.0
        item["combined_score"] = score
        scored_hits.append(item)

    return sorted(
        scored_hits,
        key=lambda row: (hit_has_clinical_content(row), float(row.get("combined_score", 0.0))),
        reverse=True,
    )[: RETRIEVAL_TOP_K]


# %% [markdown]
# ## Stage 2c run

# %%
def reranked_hits_have_quality(path: Path) -> bool:
    if not artifact_has_success_status(path):
        return False
    payload = read_json(path)
    hits = payload.get("reranked_hits", []) if isinstance(payload, dict) else []
    return isinstance(hits, list) and all(isinstance(row, dict) and isinstance(row.get("report_quality"), dict) for row in hits)


pending_stage2c = [study for study in iu_eval_manifest if not reranked_hits_have_quality(STAGE2C_DIR / f"{study.uid}.json")]
print(f"Stage 2c pending studies: {len(pending_stage2c)}")

for index, study in enumerate(
    tqdm(pending_stage2c, total=len(pending_stage2c), desc="Stage 2c", unit="study"),
    start=1,
):
    print(f"[Stage 2c] {index}/{len(pending_stage2c)} uid={study.uid}")
    stage2a_path = STAGE2A_DIR / f"{study.uid}.json"
    output_path = STAGE2C_DIR / f"{study.uid}.json"

    if not stage2a_path.exists():
        write_json(output_path, {"uid": study.uid, "status": "skipped_missing_inputs", "updated_at": utc_now()})
        continue

    stage2a_payload = read_json(stage2a_path)
    if stage2a_payload.get("status") != "success":
        write_json(output_path, {"uid": study.uid, "status": "skipped_failed_inputs", "updated_at": utc_now()})
        continue

    reranked = rerank_hits(stage2a_payload["hits"])
    write_json(
        output_path,
        {
            "uid": study.uid,
            "reranked_hits": reranked,
            "status": "success",
            "updated_at": utc_now(),
        },
    )

print_stage_summary("Stage 2c", [STAGE2C_DIR / f"{study.uid}.json" for study in iu_eval_manifest])


# %% [markdown]
# ## Stage 3 code: CheXbert-backed retrieval text evidence extraction
#
# Purpose:
# - summarize what the retrieved reports support in the 14-label space
# - use the CheXbert labels already stored in the external metadata
# - fail loudly if a reranked study has no CheXbert-backed bank label
#
# Input:
# - Stage 2c reranked hits
#
# Output:
# - one Stage 3 JSON with an aggregated `text_vector`

# %%
def aggregate_text_evidence(reranked_hits: list[dict[str, Any]]) -> dict[str, float]:
    merged = {finding: 0.0 for finding in CHEXPERT_FINDINGS}
    for row in reranked_hits:
        if not hit_has_clinical_content(row):
            continue
        study_id = str(row.get("study_id", "")).strip()
        labels = CHEXBERT_OVERRIDES.get(study_id) or PATHOLOGY_BANK_LABELS_BY_ID.get(study_id)
        if not labels:
            raise KeyError(f"No CheXbert labels available for retrieved study_id={study_id}")
        for finding in CHEXPERT_FINDINGS:
            merged[finding] = max(merged[finding], float(labels.get(finding, 0.0)))
    return merged


# %% [markdown]
# ## Stage 3 run

# %%
def stage3_has_quality_gate(path: Path) -> bool:
    if not artifact_has_success_status(path):
        return False
    payload = read_json(path)
    return isinstance(payload, dict) and payload.get("report_quality_gate") == "v1"


pending_stage3 = [study for study in iu_eval_manifest if not stage3_has_quality_gate(STAGE3_DIR / f"{study.uid}.json")]
print(f"Stage 3 pending studies: {len(pending_stage3)}")

for index, study in enumerate(
    tqdm(pending_stage3, total=len(pending_stage3), desc="Stage 3", unit="study"),
    start=1,
):
    print(f"[Stage 3] {index}/{len(pending_stage3)} uid={study.uid}")
    input_path = STAGE2C_DIR / f"{study.uid}.json"
    output_path = STAGE3_DIR / f"{study.uid}.json"
    if not input_path.exists():
        write_json(output_path, {"uid": study.uid, "status": "skipped_missing_stage2c", "updated_at": utc_now()})
        continue
    payload = read_json(input_path)
    if payload.get("status") != "success":
        write_json(output_path, {"uid": study.uid, "status": "skipped_failed_stage2c", "updated_at": utc_now()})
        continue
    aggregated = aggregate_text_evidence(payload["reranked_hits"])
    write_json(
        output_path,
        {
            "uid": study.uid,
            "model_name": STAGE3_MODEL_NAME,
            "report_quality_gate": "v1",
            "text_vector": aggregated,
            "status": "success",
            "updated_at": utc_now(),
        },
    )

print_stage_summary("Stage 3", [STAGE3_DIR / f"{study.uid}.json" for study in iu_eval_manifest])

# %% [markdown]
# ## Stage 4 code: deterministic fusion
#
# Purpose:
# - compare image-first evidence against retrieval-text evidence
# - assign one fused state per CheXpert finding
#
# Input:
# - Stage 1 `parsed_vector`
# - Stage 3 `text_vector`
#
# Output:
# - one Stage 4 JSON with `fusion_table`

# %%
def fuse_vectors(image_scores: dict[str, float], text_scores: dict[str, float]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for finding in CHEXPERT_FINDINGS:
        image_score = float(image_scores.get(finding, 0.0))
        text_score = float(text_scores.get(finding, 0.0))
        image_positive = image_score >= IMAGE_POSITIVE_THRESHOLD
        text_positive = text_score >= TEXT_POSITIVE_THRESHOLD

        if image_positive and text_positive:
            state = "CONFLICT" if abs(image_score - text_score) >= CONFLICT_GAP_THRESHOLD else "CONFIRMED"
        elif image_positive:
            state = "IMAGE-ONLY"
        elif text_positive:
            state = "TEXT-ONLY"
        else:
            state = "ABSENT"

        decisions.append(
            {
                "finding": finding,
                "state": state,
                "image_score": image_score,
                "text_score": text_score,
            }
        )
    return decisions


# %% [markdown]
# ## Stage 4 run

# %%
def stage4_has_quality_gate(path: Path) -> bool:
    if not artifact_has_success_status(path):
        return False
    payload = read_json(path)
    return isinstance(payload, dict) and payload.get("report_quality_gate") == "v1"


pending_stage4 = [study for study in iu_eval_manifest if not stage4_has_quality_gate(STAGE4_DIR / f"{study.uid}.json")]
print(f"Stage 4 pending studies: {len(pending_stage4)}")

for index, study in enumerate(
    tqdm(pending_stage4, total=len(pending_stage4), desc="Stage 4", unit="study"),
    start=1,
):
    print(f"[Stage 4] {index}/{len(pending_stage4)} uid={study.uid}")
    stage1_payload = load_stage1_success(study.uid)
    stage3_path = STAGE3_DIR / f"{study.uid}.json"
    output_path = STAGE4_DIR / f"{study.uid}.json"
    if stage1_payload is None or not stage3_path.exists():
        write_json(output_path, {"uid": study.uid, "status": "skipped_missing_inputs", "updated_at": utc_now()})
        continue
    stage3_payload = read_json(stage3_path)
    if stage3_payload.get("status") != "success":
        write_json(output_path, {"uid": study.uid, "status": "skipped_failed_stage3", "updated_at": utc_now()})
        continue
    fusion_table = fuse_vectors(stage1_payload["parsed_vector"], stage3_payload["text_vector"])
    write_json(
        output_path,
        {
            "uid": study.uid,
            "fusion_table": fusion_table,
            "report_quality_gate": "v1",
            "status": "success",
            "updated_at": utc_now(),
        },
    )

print_stage_summary("Stage 4", [STAGE4_DIR / f"{study.uid}.json" for study in iu_eval_manifest])


def _state_counts(rows: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    return dict(sorted(counts.items()))


def build_pre_gemini_metrics() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage_dirs = {
        "stage1": STAGE1_DIR,
        "chexone_direct": CHEXONE_DIRECT_DIR,
        "stage2a": STAGE2A_DIR,
        "stage2c": STAGE2C_DIR,
        "stage3": STAGE3_DIR,
        "stage4": STAGE4_DIR,
    }
    stage_status = {
        name: count_statuses([directory / f"{study.uid}.json" for study in iu_eval_manifest])
        for name, directory in stage_dirs.items()
    }

    retrieval_top1_sources: list[str] = []
    retrieval_hit_counts: list[int] = []
    fusion_states: list[str] = []
    per_study_rows: list[dict[str, Any]] = []

    for study in iu_eval_manifest:
        stage1_payload = read_json(stage1_output_path(study.uid)) if stage1_output_path(study.uid).exists() else {}
        stage2c_payload = read_json(STAGE2C_DIR / f"{study.uid}.json") if (STAGE2C_DIR / f"{study.uid}.json").exists() else {}
        stage4_payload = read_json(STAGE4_DIR / f"{study.uid}.json") if (STAGE4_DIR / f"{study.uid}.json").exists() else {}

        reranked_hits = list(stage2c_payload.get("reranked_hits", [])) if isinstance(stage2c_payload, dict) and stage2c_payload.get("status") == "success" else []
        retrieval_hit_counts.append(len(reranked_hits))
        top1_source = ""
        top1_study_id = ""
        if reranked_hits:
            top1_source = str(reranked_hits[0].get("source_dataset", ""))
            top1_study_id = str(reranked_hits[0].get("study_id", ""))
            retrieval_top1_sources.append(top1_source)

        fusion_table = list(stage4_payload.get("fusion_table", [])) if isinstance(stage4_payload, dict) and stage4_payload.get("status") == "success" else []
        fusion_states.extend(str(row.get("state", "")) for row in fusion_table)

        direct_report_text = str(stage1_payload.get("report_text", "")) if isinstance(stage1_payload, dict) else ""
        per_study_rows.append(
            {
                "uid": study.uid,
                "image_path": study.image_path,
                "ground_truth_report_chars": len(study.ground_truth_report),
                "direct_report_chars": len(direct_report_text),
                "direct_report_nonempty": bool(direct_report_text),
                "retrieval_hit_count": len(reranked_hits),
                "top1_retrieved_study_id": top1_study_id,
                "top1_retrieved_source": top1_source,
                "stage1_status": str(stage1_payload.get("status", "")) if isinstance(stage1_payload, dict) else "",
                "stage2c_status": str(stage2c_payload.get("status", "")) if isinstance(stage2c_payload, dict) else "",
                "stage4_status": str(stage4_payload.get("status", "")) if isinstance(stage4_payload, dict) else "",
            }
        )

    metrics = {
        "bundle_root": str(BUNDLE_ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "external_bank_root": str(EXTERNAL_BANK_ROOT),
        "external_chexone_report_dir": str(EXTERNAL_CHEXONE_REPORT_DIR),
        "num_eval_studies": len(iu_eval_manifest),
        "stage_status": stage_status,
        "retrieval": {
            "top_k": RETRIEVAL_TOP_K,
            "average_hit_count": (sum(retrieval_hit_counts) / len(retrieval_hit_counts)) if retrieval_hit_counts else 0.0,
            "top1_source_counts": _state_counts(retrieval_top1_sources),
        },
        "fusion": {
            "label_state_counts": _state_counts(fusion_states),
        },
        "direct_reports": {
            "nonempty_count": sum(1 for row in per_study_rows if row["direct_report_nonempty"]),
            "average_chars": (
                sum(int(row["direct_report_chars"]) for row in per_study_rows) / len(per_study_rows)
                if per_study_rows
                else 0.0
            ),
        },
        "updated_at": utc_now(),
    }
    return metrics, per_study_rows


def write_pre_gemini_metrics() -> dict[str, Any]:
    metrics, per_study_rows = build_pre_gemini_metrics()
    metrics_json_path = EVAL_DIR / "pre_gemini_metrics.json"
    per_study_csv_path = EVAL_DIR / "pre_gemini_per_study.csv"
    write_json(metrics_json_path, metrics)
    write_csv_rows(
        per_study_csv_path,
        per_study_rows,
        [
            "uid",
            "image_path",
            "ground_truth_report_chars",
            "direct_report_chars",
            "direct_report_nonempty",
            "retrieval_hit_count",
            "top1_retrieved_study_id",
            "top1_retrieved_source",
            "stage1_status",
            "stage2c_status",
            "stage4_status",
        ],
    )
    print(f"Pre-Gemini metrics JSON: {metrics_json_path}")
    print(f"Pre-Gemini per-study CSV: {per_study_csv_path}")
    print(f"Pre-Gemini eval studies: {metrics['num_eval_studies']}")
    print(f"Direct report nonempty count: {metrics['direct_reports']['nonempty_count']}")
    print(f"Average retrieval hit count: {metrics['retrieval']['average_hit_count']:.2f}")
    print(f"Top-1 retrieval source counts: {metrics['retrieval']['top1_source_counts']}")
    print(f"Label fusion state counts: {metrics['fusion']['label_state_counts']}")
    return metrics


def maybe_stop_before_gemini_stages() -> None:
    if STOP_BEFORE_GEMINI_STAGES:
        write_pre_gemini_metrics()
        print("Stopping before Gemini Stage 5 and Stage 7 as requested by configuration.")
        raise SystemExit(0)


maybe_stop_before_gemini_stages()


# %% [markdown]
# ## Stage 5 code: LLM report composer with persistent retry cache
#
# Purpose:
# - write the final pipeline report from grounded fused evidence;
# - keep the output in IU-style `Findings:` and `Impression:` sections;
# - cache each prompt attempt so failed or interrupted runs can resume.
#
# Input:
# - Stage 4 `fusion_table`, which identifies confirmed findings and guardrails;
# - Stage 2c retrieved reports, treated as supporting context rather than truth;
# - Stage 1 CheXOne report, used as the image-first direct draft.
#
# Output:
# - one Stage 5 JSON with final `findings`, `impression`, `report_text`, prompt
#   metadata, backend metadata, and cache status.
#
# The same wrapper supports Gemini and local Ollama. The active default in this
# notebook is local Ollama unless environment variables disable it.

# %%
FIXED_IU_STYLE_EXAMPLE_REPORT = (
    "Findings: The cardiomediastinal silhouette is within normal size limits. "
    "The lungs are clear without focal consolidation, pleural effusion, or pneumothorax.\n"
    "Impression: No acute cardiopulmonary abnormality."
)


STAGE5_PROMPT_TEMPLATE = """Write a concise chest X-ray report from the approved evidence below.

Rules:
- Output only two sections: Findings: and Impression:
- Findings: 1-4 concise prose sentences.
- Impression: 1-2 concise prose sentences.
- Use the style example for wording and length only.
- Do not add unsupported findings.
- Do not include title, technique, recommendations, disclaimers, bullets, numbering, markdown, or explanations.
- Treat retrieved reports as supporting evidence, not truth.

Required output format:
Findings: <one concise paragraph>
Impression: <one concise paragraph>

Confirmed evidence:
{confirmed_fusion_lines}

Do not state as present:
{guardrail_lines}

Direct image-first CheXOne draft:
{direct_chexone_report}

Balanced retrieved evidence candidates:
{retrieval_evidence_lines}

Style example only:
{style_example_report}
"""


def gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def classify_llm_error(exc: Exception) -> tuple[bool, str]:
    http_status = extract_http_status(exc)
    text = repr(exc).lower()

    if http_status == 400:
        return False, "bad_request"
    if http_status in {401, 403} or "401" in text or "403" in text or "invalid api key" in text or "permission" in text:
        return False, "permanent_auth_failure"
    if http_status == 404 or ("404" in text and "models/" in text):
        return False, "permanent_model_failure"
    if http_status in {429, 500, 502, 503, 504}:
        return True, f"http_{http_status}"
    return True, exc.__class__.__name__


def extract_http_status(exc: Exception) -> int | None:
    if hasattr(exc, "status") and isinstance(getattr(exc, "status"), int):
        return int(getattr(exc, "status"))
    if hasattr(exc, "code") and isinstance(getattr(exc, "code"), int):
        return int(getattr(exc, "code"))
    return None


def stage5_report_evidence_hits(reranked_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in reranked_hits
        if hit_has_clinical_content(row)
        and str(row.get("retrieval_source_group", "") or "") == "visual"
    ]


def format_retrieval_evidence_lines(reranked_hits: list[dict[str, Any]]) -> str:
    evidence_hits = stage5_report_evidence_hits(reranked_hits)
    if not evidence_hits:
        return "- none"
    lines: list[str] = []
    for index, row in enumerate(evidence_hits[:RETRIEVAL_TOP_K], start=1):
        labels = row.get("labels", {})
        positive_labels = []
        if isinstance(labels, dict):
            positive_labels = [
                finding
                for finding, value in labels.items()
                if finding != "No Finding" and safe_float(value, 0.0) >= TEXT_POSITIVE_THRESHOLD
            ]
        visual_score = safe_float(row.get("visual_score", 0.0), 0.0)
        pathology_score = safe_float(row.get("pathology_score", 0.0), 0.0)
        report_text = normalize_space(str(row.get("report_text", "")))[:700]
        lines.append(
            "- "
            f"candidate={index}; source={row.get('retrieval_source_group', 'visual')}; rank={row.get('rank')}; "
            f"study_id={row.get('study_id')}; visual_score={visual_score:.3f}; "
            f"pathology_score={pathology_score:.3f}; "
            f"positive_labels={', '.join(positive_labels) if positive_labels else 'none'}; "
            f"report_excerpt={report_text}"
        )
    return "\n".join(lines)


def build_stage5_prompt(
    direct_chexone_report: str,
    fusion_table: list[dict[str, Any]],
    reranked_hits: list[dict[str, Any]],
) -> str:
    confirmed_fusion_lines = format_confirmed_fusion_lines(fusion_table)
    guardrail_lines = format_fusion_guardrail_lines(fusion_table)
    retrieval_evidence_lines = format_retrieval_evidence_lines(reranked_hits)
    return STAGE5_PROMPT_TEMPLATE.format(
        confirmed_fusion_lines=confirmed_fusion_lines,
        guardrail_lines=guardrail_lines,
        direct_chexone_report=direct_chexone_report.strip() or "No direct draft available.",
        retrieval_evidence_lines=retrieval_evidence_lines,
        style_example_report=FIXED_IU_STYLE_EXAMPLE_REPORT,
    )


def gemini_generate(model_name: str, prompt_text: str) -> str:
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{parse.quote(model_name, safe='')}:generateContent?key={parse.quote(api_key, safe='')}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.1, "topP": 0.8},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=60) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    candidates = parsed.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise ValueError("Gemini returned an empty text response.")
    return text


def ollama_options_for_model(model_name: str) -> dict[str, Any]:
    if model_name == COMPOSER_MODEL_NAME:
        return {
            "temperature": LOCAL_OLLAMA_COMPOSER_TEMPERATURE,
            "top_p": LOCAL_OLLAMA_COMPOSER_TOP_P,
            "num_ctx": LOCAL_OLLAMA_COMPOSER_NUM_CTX,
            "num_predict": LOCAL_OLLAMA_COMPOSER_NUM_PREDICT,
        }
    return {
        "temperature": LOCAL_OLLAMA_JUDGE_TEMPERATURE,
        "top_p": LOCAL_OLLAMA_JUDGE_TOP_P,
        "num_ctx": LOCAL_OLLAMA_JUDGE_NUM_CTX,
        "num_predict": LOCAL_OLLAMA_JUDGE_NUM_PREDICT,
    }


def ollama_think_for_model(model_name: str) -> bool:
    return False


def ollama_generate(model_name: str, prompt_text: str) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "think": ollama_think_for_model(model_name),
        "options": ollama_options_for_model(model_name),
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(LOCAL_OLLAMA_CHAT_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=LOCAL_OLLAMA_TIMEOUT_SECONDS) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    message = parsed.get("message", {}) if isinstance(parsed, dict) else {}
    text = str(message.get("content", "")).strip()
    if not text:
        raise ValueError("Ollama returned an empty text response.")
    return text


def unload_ollama_model(model_name: str) -> None:
    if not USE_LOCAL_OLLAMA_LLM:
        return
    payload = {"model": model_name, "keep_alive": 0}
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        LOCAL_OLLAMA_CHAT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            response.read()
        print(f"Unloaded Ollama model from memory: {model_name}")
    except Exception as exc:
        print(f"Could not unload Ollama model `{model_name}`; Ollama may already be stopped: {repr(exc)}")


def llm_generate(model_name: str, prompt_text: str) -> str:
    if USE_LOCAL_OLLAMA_LLM:
        return ollama_generate(model_name, prompt_text)
    return gemini_generate(model_name, prompt_text)


def run_gemini_cached(
    cache_dir: Path,
    model_name: str,
    prompt_text: str,
    parse_fn,
    force_refresh: bool,
    max_retries: int,
) -> dict[str, Any]:
    provider = "ollama" if USE_LOCAL_OLLAMA_LLM else "gemini"
    prompt_hash = stable_hash({"provider": provider, "model_name": model_name, "prompt_text": prompt_text})
    cache_path = cache_dir / f"{prompt_hash}.json"
    existing = read_json(cache_path) if cache_path.exists() else None

    if isinstance(existing, dict):
        status = existing.get("status")
        if status == "success" and not force_refresh:
            existing["cache_hit"] = True
            return existing
        if status == "failed_permanent" and not force_refresh:
            existing["cache_hit"] = True
            return existing

    if not isinstance(existing, dict) or force_refresh:
        existing = {
            "prompt_hash": prompt_hash,
            "model_name": model_name,
            "prompt_text": prompt_text,
            "status": "pending",
            "attempt_count": 0,
            "attempt_history": [],
            "final_response_text": "",
            "parsed_output": None,
            "last_error": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "cache_hit": False,
        }
        write_json(cache_path, existing)
    else:
        existing["cache_hit"] = False

    start_attempt = int(existing.get("attempt_count", 0))
    for attempt in range(start_attempt, max_retries):
        existing["status"] = "running" if attempt == 0 else "retrying"
        existing["updated_at"] = utc_now()
        write_json(cache_path, existing)

        attempt_record = {
            "attempt_index": attempt,
            "started_at": utc_now(),
            "finished_at": None,
            "status": "running",
            "http_status": None,
            "error_type": None,
            "error_message": None,
            "raw_response_excerpt": None,
        }

        try:
            raw_response = llm_generate(model_name, prompt_text)
            parsed_output = parse_fn(raw_response)
            attempt_record["finished_at"] = utc_now()
            attempt_record["status"] = "success"
            attempt_record["raw_response_excerpt"] = raw_response[:500]
            existing["attempt_count"] = attempt + 1
            existing["attempt_history"].append(attempt_record)
            existing["final_response_text"] = raw_response
            existing["parsed_output"] = parsed_output
            existing["last_error"] = None
            existing["status"] = "success"
            existing["updated_at"] = utc_now()
            write_json(cache_path, existing)
            return existing
        except Exception as exc:
            retryable, error_type = classify_llm_error(exc)
            attempt_record["finished_at"] = utc_now()
            attempt_record["status"] = "retryable_failure" if retryable else "permanent_failure"
            attempt_record["http_status"] = extract_http_status(exc)
            attempt_record["error_type"] = error_type
            attempt_record["error_message"] = repr(exc)
            existing["attempt_count"] = attempt + 1
            existing["attempt_history"].append(attempt_record)
            existing["last_error"] = {"error_type": error_type, "error_message": repr(exc)}
            existing["updated_at"] = utc_now()

            if not retryable or attempt + 1 >= max_retries:
                existing["status"] = "failed_permanent"
                write_json(cache_path, existing)
                return existing

            existing["status"] = "retrying"
            write_json(cache_path, existing)
            time.sleep(retry_sleep_seconds(attempt))

    existing["status"] = "failed_permanent"
    existing["updated_at"] = utc_now()
    write_json(cache_path, existing)
    return existing


# %%
def load_stage5_success(uid: str) -> dict[str, Any] | None:
    target = STAGE5_DIR / f"{uid}.json"
    if not target.exists():
        return None
    payload = read_json(target)
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return None
    return payload


def compute_stage5_prompt_hash_for_study(study: IUStudy) -> str | None:
    stage1_payload = load_stage1_success(study.uid)
    stage4_path = STAGE4_DIR / f"{study.uid}.json"
    stage2c_path = STAGE2C_DIR / f"{study.uid}.json"
    if stage1_payload is None or not stage4_path.exists() or not stage2c_path.exists():
        return None
    stage4_payload = read_json(stage4_path)
    stage2c_payload = read_json(stage2c_path)
    if stage4_payload.get("status") != "success" or stage2c_payload.get("status") != "success":
        return None
    prompt_text = build_stage5_prompt(
        str(stage1_payload.get("raw_response", "") or stage1_payload.get("report_text", "")),
        stage4_payload["fusion_table"],
        stage2c_payload["reranked_hits"],
    )
    return stable_hash({"model_name": COMPOSER_MODEL_NAME, "prompt_text": prompt_text})


def stage5_needs_run(study: IUStudy) -> bool:
    if FORCE_STAGE5_REFRESH:
        return True
    existing = load_stage5_success(study.uid)
    if existing is None:
        return True
    expected_prompt_hash = compute_stage5_prompt_hash_for_study(study)
    if expected_prompt_hash is None:
        return True
    return str(existing.get("prompt_hash", "")).strip() != expected_prompt_hash


def _run_stage5_for_study(study: IUStudy) -> dict[str, Any]:
    stage1_payload = load_stage1_success(study.uid)
    stage4_path = STAGE4_DIR / f"{study.uid}.json"
    stage2c_path = STAGE2C_DIR / f"{study.uid}.json"
    output_path = STAGE5_DIR / f"{study.uid}.json"
    if stage1_payload is None or not stage4_path.exists() or not stage2c_path.exists():
        write_json(output_path, {"uid": study.uid, "status": "skipped_missing_inputs", "updated_at": utc_now()})
        return {"uid": study.uid, "status": "skipped_missing_inputs"}

    stage4_payload = read_json(stage4_path)
    stage2c_payload = read_json(stage2c_path)
    if stage4_payload.get("status") != "success" or stage2c_payload.get("status") != "success":
        write_json(output_path, {"uid": study.uid, "status": "skipped_failed_inputs", "updated_at": utc_now()})
        return {"uid": study.uid, "status": "skipped_failed_inputs"}

    prompt_text = build_stage5_prompt(
        str(stage1_payload.get("raw_response", "") or stage1_payload.get("report_text", "")),
        stage4_payload["fusion_table"],
        stage2c_payload["reranked_hits"],
    )
    llm_payload = run_gemini_cached(
        cache_dir=LLM_FLASH_DIR,
        model_name=COMPOSER_MODEL_NAME,
        prompt_text=prompt_text,
        parse_fn=parse_stage5_report,
        force_refresh=FORCE_STAGE5_REFRESH,
        max_retries=MAX_LLM_RETRIES,
    )

    if llm_payload["status"] == "success":
        write_json(
            output_path,
            {
                "uid": study.uid,
                "model_name": COMPOSER_MODEL_NAME,
                "prompt_hash": llm_payload["prompt_hash"],
                "cache_path": str(LLM_FLASH_DIR / f"{llm_payload['prompt_hash']}.json"),
                "cache_status": llm_payload["status"],
                "cache_hit": bool(llm_payload.get("cache_hit", False)),
                "prompt_text": prompt_text,
                "findings": llm_payload["parsed_output"]["findings"],
                "impression": llm_payload["parsed_output"]["impression"],
                "report_text": llm_payload["parsed_output"]["report_text"],
                "status": "success",
                "updated_at": utc_now(),
            },
        )
        return {"uid": study.uid, "status": "success"}

    write_json(
        output_path,
        {
            "uid": study.uid,
            "model_name": COMPOSER_MODEL_NAME,
            "prompt_hash": llm_payload["prompt_hash"],
            "cache_path": str(LLM_FLASH_DIR / f"{llm_payload['prompt_hash']}.json"),
            "cache_status": llm_payload["status"],
            "status": "failed_permanent",
            "error": llm_payload.get("last_error"),
            "updated_at": utc_now(),
        },
    )
    return {
        "uid": study.uid,
        "status": "failed_permanent",
        "error": llm_payload.get("last_error"),
    }


# %% [markdown]
# ## Stage 5 run

# %%
pending_stage5 = [study for study in iu_eval_manifest if stage5_needs_run(study)]
print(f"Stage 5 pending studies: {len(pending_stage5)}")
print(f"Stage 5 parallel workers: {LLM_PARALLEL_WORKERS}")

stage5_failures: list[dict[str, Any]] = []
if pending_stage5:
    with ThreadPoolExecutor(max_workers=LLM_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_run_stage5_for_study, study): study.uid for study in pending_stage5}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Stage 5", unit="study"):
            result = future.result()
            if result["status"] == "failed_permanent":
                stage5_failures.append(result)

if stage5_failures:
    first_failure = stage5_failures[0]
    raise RuntimeError(
        f"Stage 5 failed permanently for uid={first_failure['uid']}. "
        f"Last error: {first_failure.get('error')}"
    )

if pending_stage5:
    unload_ollama_model(COMPOSER_MODEL_NAME)


# %% [markdown]
# ## Stage 7 code: LLM judge
#
# Purpose:
# - compare generated reports against the IU reference report;
# - score clinical accuracy, groundedness, completeness, style, and overall
#   quality;
# - judge both the pipeline output and the CheXOne direct baseline with the same
#   rubric.
#
# Input:
# - pipeline report or direct CheXOne report;
# - IU reference report.
#
# Output:
# - one judge JSON per candidate with rubric scores, rationale, hallucination
#   flags, backend metadata, and cache status.

# %%
STAGE7_PROMPT_VERSION = "reference_candidate_only_v1"

STAGE7_PROMPT_TEMPLATE = """You are evaluating a generated chest x-ray radiology report.

Compare the generated report against the IU reference report only.

Score the generated report on:
- clinical_accuracy_score
- groundedness_score
- completeness_score
- style_score
- overall_score

Each score must be between 0 and 10.
For groundedness_score, judge whether the generated report is supported by and consistent with the reference report. Do not use pipeline evidence, retrieved reports, labels, or drafts.

Also return:
- hallucination_flags: a list of short strings
- brief_rationale: short paragraph

Output valid JSON only using this exact schema:

{{
  "clinical_accuracy_score": 0,
  "groundedness_score": 0,
  "completeness_score": 0,
  "style_score": 0,
  "overall_score": 0,
  "hallucination_flags": [],
  "brief_rationale": ""
}}

Reference report:
{reference_report}

Generated report:
{generated_report}
"""


def build_stage7_prompt(reference_report: str, generated_report: str) -> str:
    return STAGE7_PROMPT_TEMPLATE.format(
        reference_report=reference_report,
        generated_report=generated_report,
    )


def _run_stage7_pipeline_for_study(study: IUStudy) -> dict[str, Any]:
    stage5_path = STAGE5_DIR / f"{study.uid}.json"
    output_path = JUDGING_PIPELINE_DIR / f"{study.uid}.json"
    if not stage5_path.exists():
        _write_stage7_skip(output_path, study.uid, "skipped_missing_inputs")
        return {"uid": study.uid, "status": "skipped_missing_inputs"}

    stage5_payload = read_json(stage5_path)
    if stage5_payload.get("status") != "success":
        _write_stage7_skip(output_path, study.uid, "skipped_failed_inputs")
        return {"uid": study.uid, "status": "skipped_failed_inputs"}

    prompt_text = build_stage7_prompt(
        study.ground_truth_report,
        stage5_payload["report_text"],
    )
    llm_payload = run_gemini_cached(
        cache_dir=LLM_PRO_DIR,
        model_name=JUDGE_MODEL_NAME,
        prompt_text=prompt_text,
        parse_fn=parse_stage7_judge,
        force_refresh=FORCE_STAGE7_REFRESH,
        max_retries=MAX_LLM_RETRIES,
    )

    if llm_payload["status"] == "success":
        write_json(
            output_path,
            {
                "uid": study.uid,
                "candidate_name": "pipeline",
                "reference_report": study.ground_truth_report,
                "generated_report": stage5_payload["report_text"],
                "model_name": JUDGE_MODEL_NAME,
                "prompt_version": STAGE7_PROMPT_VERSION,
                "prompt_hash": llm_payload["prompt_hash"],
                "cache_path": str(LLM_PRO_DIR / f"{llm_payload['prompt_hash']}.json"),
                "cache_status": llm_payload["status"],
                "cache_hit": bool(llm_payload.get("cache_hit", False)),
                "judge_output": llm_payload["parsed_output"],
                "status": "success",
                "updated_at": utc_now(),
            },
        )
        return {"uid": study.uid, "status": "success"}

    write_json(
        output_path,
        {
            "uid": study.uid,
            "candidate_name": "pipeline",
            "reference_report": study.ground_truth_report,
            "generated_report": stage5_payload["report_text"],
            "model_name": JUDGE_MODEL_NAME,
            "prompt_version": STAGE7_PROMPT_VERSION,
            "prompt_hash": llm_payload["prompt_hash"],
            "cache_path": str(LLM_PRO_DIR / f"{llm_payload['prompt_hash']}.json"),
            "cache_status": llm_payload["status"],
            "cache_hit": bool(llm_payload.get("cache_hit", False)),
            "status": "failed_permanent",
            "error": llm_payload.get("last_error"),
            "updated_at": utc_now(),
        },
    )
    return {
        "uid": study.uid,
        "status": "failed_permanent",
        "candidate_name": "pipeline",
        "error": llm_payload.get("last_error"),
    }


def _run_stage7_chexone_for_study(study: IUStudy) -> dict[str, Any]:
    chexone_direct_path = chexone_direct_output_path(study.uid)
    output_path = JUDGING_CHEXONE_DIR / f"{study.uid}.json"
    if not chexone_direct_path.exists():
        _write_stage7_skip(output_path, study.uid, "skipped_missing_inputs")
        return {"uid": study.uid, "status": "skipped_missing_inputs"}

    chexone_direct_payload = read_json(chexone_direct_path)
    if chexone_direct_payload.get("status") != "success":
        _write_stage7_skip(output_path, study.uid, "skipped_failed_inputs")
        return {"uid": study.uid, "status": "skipped_failed_inputs"}

    prompt_text = build_stage7_prompt(
        study.ground_truth_report,
        chexone_direct_payload["report_text"],
    )
    llm_payload = run_gemini_cached(
        cache_dir=LLM_PRO_DIR,
        model_name=JUDGE_MODEL_NAME,
        prompt_text=prompt_text,
        parse_fn=parse_stage7_judge,
        force_refresh=FORCE_STAGE7_REFRESH,
        max_retries=MAX_LLM_RETRIES,
    )

    if llm_payload["status"] == "success":
        write_json(
            output_path,
            {
                "uid": study.uid,
                "candidate_name": "chexone_direct",
                "reference_report": study.ground_truth_report,
                "generated_report": chexone_direct_payload["report_text"],
                "model_name": JUDGE_MODEL_NAME,
                "prompt_version": STAGE7_PROMPT_VERSION,
                "prompt_hash": llm_payload["prompt_hash"],
                "cache_path": str(LLM_PRO_DIR / f"{llm_payload['prompt_hash']}.json"),
                "cache_status": llm_payload["status"],
                "cache_hit": bool(llm_payload.get("cache_hit", False)),
                "judge_output": llm_payload["parsed_output"],
                "status": "success",
                "updated_at": utc_now(),
            },
        )
        return {"uid": study.uid, "status": "success"}

    write_json(
        output_path,
        {
            "uid": study.uid,
            "candidate_name": "chexone_direct",
            "reference_report": study.ground_truth_report,
            "generated_report": chexone_direct_payload["report_text"],
            "model_name": JUDGE_MODEL_NAME,
            "prompt_version": STAGE7_PROMPT_VERSION,
            "prompt_hash": llm_payload["prompt_hash"],
            "cache_path": str(LLM_PRO_DIR / f"{llm_payload['prompt_hash']}.json"),
            "cache_status": llm_payload["status"],
            "cache_hit": bool(llm_payload.get("cache_hit", False)),
            "status": "failed_permanent",
            "error": llm_payload.get("last_error"),
            "updated_at": utc_now(),
        },
    )
    return {
        "uid": study.uid,
        "status": "failed_permanent",
        "candidate_name": "chexone_direct",
        "error": llm_payload.get("last_error"),
    }


# %% [markdown]
# ## Stage 7 run

# %%
def _write_stage7_skip(output_path: Path, uid: str, status: str) -> None:
    write_json(output_path, {"uid": uid, "status": status, "updated_at": utc_now()})

pending_stage7_pipeline = [
    study
    for study in iu_eval_manifest
    if FORCE_STAGE7_REFRESH
    or not artifact_has_success_status(JUDGING_PIPELINE_DIR / f"{study.uid}.json")
    or str(read_json(JUDGING_PIPELINE_DIR / f"{study.uid}.json").get("model_name", "")).strip() != JUDGE_MODEL_NAME
    or str(read_json(JUDGING_PIPELINE_DIR / f"{study.uid}.json").get("prompt_version", "")).strip() != STAGE7_PROMPT_VERSION
]
pending_stage7_chexone = [
    study
    for study in iu_eval_manifest
    if FORCE_STAGE7_REFRESH
    or not artifact_has_success_status(JUDGING_CHEXONE_DIR / f"{study.uid}.json")
    or str(read_json(JUDGING_CHEXONE_DIR / f"{study.uid}.json").get("model_name", "")).strip() != JUDGE_MODEL_NAME
    or str(read_json(JUDGING_CHEXONE_DIR / f"{study.uid}.json").get("prompt_version", "")).strip() != STAGE7_PROMPT_VERSION
]

print(f"Stage 7 pipeline pending studies: {len(pending_stage7_pipeline)}")
print(f"Stage 7 CheXOne-direct pending studies: {len(pending_stage7_chexone)}")
print(f"Stage 7 parallel workers: {LLM_PARALLEL_WORKERS}")

if not ENABLE_LLM_JUDGE:
    for study in pending_stage7_pipeline:
        _write_stage7_skip(JUDGING_PIPELINE_DIR / f"{study.uid}.json", study.uid, "skipped_disabled")
    for study in pending_stage7_chexone:
        _write_stage7_skip(JUDGING_CHEXONE_DIR / f"{study.uid}.json", study.uid, "skipped_disabled")
else:
    if pending_stage7_pipeline or pending_stage7_chexone:
        if USE_LOCAL_OLLAMA_LLM and COMPOSER_MODEL_NAME != JUDGE_MODEL_NAME:
            unload_ollama_model(COMPOSER_MODEL_NAME)
    stage7_pipeline_failures: list[dict[str, Any]] = []
    if pending_stage7_pipeline:
        with ThreadPoolExecutor(max_workers=LLM_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_run_stage7_pipeline_for_study, study): study.uid for study in pending_stage7_pipeline}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Stage 7 pipeline", unit="study"):
                result = future.result()
                if result["status"] == "failed_permanent":
                    stage7_pipeline_failures.append(result)

    if stage7_pipeline_failures:
        first_failure = stage7_pipeline_failures[0]
        raise RuntimeError(
            f"Stage 7 pipeline judge failed permanently for uid={first_failure['uid']}. "
            f"Last error: {first_failure.get('error')}"
        )

    stage7_chexone_failures: list[dict[str, Any]] = []
    if pending_stage7_chexone:
        with ThreadPoolExecutor(max_workers=LLM_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_run_stage7_chexone_for_study, study): study.uid for study in pending_stage7_chexone}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Stage 7 CheXOne", unit="study"):
                result = future.result()
                if result["status"] == "failed_permanent":
                    stage7_chexone_failures.append(result)

    if stage7_chexone_failures:
        first_failure = stage7_chexone_failures[0]
        raise RuntimeError(
            f"Stage 7 CheXOne direct judge failed permanently for uid={first_failure['uid']}. "
            f"Last error: {first_failure.get('error')}"
        )

    if pending_stage7_pipeline or pending_stage7_chexone:
        unload_ollama_model(JUDGE_MODEL_NAME)


# %% [markdown]
# ## Evaluation summary

# %%
def label_f1(reference: dict[str, float], predicted: dict[str, float], threshold: float = 0.5) -> dict[str, float]:
    ref_positive = {finding for finding, value in reference.items() if value >= threshold}
    pred_positive = {finding for finding, value in predicted.items() if value >= threshold}
    tp = len(ref_positive & pred_positive)
    fp = len(pred_positive - ref_positive)
    fn = len(ref_positive - pred_positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def tokenize_for_metrics(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[^\w\s]", str(text).lower())


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    import nltk  # type: ignore

    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def compute_text_overlap_metrics(reference_text: str, candidate_text: str) -> dict[str, float]:
    if not str(candidate_text).strip():
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "meteor": 0.0,
        }

    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from nltk.translate.meteor_score import single_meteor_score  # type: ignore
    from rouge_score import rouge_scorer  # type: ignore

    ensure_nltk_resource("corpora/wordnet", "wordnet")
    ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")

    ref_tokens = tokenize_for_metrics(reference_text)
    cand_tokens = tokenize_for_metrics(candidate_text)
    smoother = SmoothingFunction().method4
    bleu = float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoother)) if cand_tokens else 0.0
    meteor = float(single_meteor_score(ref_tokens, cand_tokens)) if cand_tokens else 0.0
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(reference_text, candidate_text)
    return {
        "bleu": bleu,
        "rouge1": float(rouge["rouge1"].fmeasure),
        "rouge2": float(rouge["rouge2"].fmeasure),
        "rougeL": float(rouge["rougeL"].fmeasure),
        "meteor": meteor,
    }


def compute_bertscore_batch(references: list[str], candidates: list[str]) -> list[float]:
    if not references or not candidates:
        return []
    active_indices = [idx for idx, text in enumerate(candidates) if str(text).strip()]
    scores = [0.0 for _ in candidates]
    if not active_indices:
        return scores

    from bert_score import score as bertscore_score  # type: ignore

    active_candidates = [candidates[idx] for idx in active_indices]
    active_references = [references[idx] for idx in active_indices]
    device_name = "cuda" if DEVICE == "cuda" else "cpu"
    _, _, f1_scores = bertscore_score(
        active_candidates,
        active_references,
        lang="en",
        model_type=BERTSCORE_MODEL_TYPE,
        verbose=False,
        rescale_with_baseline=False,
        device=device_name,
    )
    for idx, score in zip(active_indices, f1_scores.tolist(), strict=True):
        scores[idx] = float(score)
    return scores


def compute_corpus_generation_metrics(references: list[str], candidates: list[str]) -> dict[str, float]:
    import sacrebleu  # type: ignore

    refs = [str(text or "") for text in references]
    hyps = [str(text or "") for text in candidates]
    if not refs or not hyps:
        return {"sacrebleu": 0.0, "chrf": 0.0, "chrf_pp": 0.0}
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=0)
    chrf_pp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    return {
        "sacrebleu": float(bleu.score),
        "chrf": float(chrf.score),
        "chrf_pp": float(chrf_pp.score),
    }


def aggregate_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    return sum(values) / len(values) if values else 0.0


def aggregate_metric_filtered(rows: list[dict[str, Any]], key: str, filter_key: str) -> float:
    filtered = [row for row in rows if row.get(filter_key)]
    return aggregate_metric(filtered, key)


def print_result_block(
    title: str,
    text_metrics: dict[str, float],
    corpus_metrics: dict[str, float],
    judge_score: float | None,
    completed_reports: int,
    completed_judges: int,
    total: int,
) -> None:
    print("")
    print(title)
    print("=" * len(title))
    print(f"completed_reports: {completed_reports}/{total}")
    print(f"completed_judge_scores: {completed_judges}/{total}")
    if judge_score is not None:
        print(f"judge_overall: {judge_score:.3f}")
    print(f"bleu: {text_metrics['bleu']:.3f}")
    print(f"rouge1: {text_metrics['rouge1']:.3f}")
    print(f"rouge2: {text_metrics['rouge2']:.3f}")
    print(f"rougeL: {text_metrics['rougeL']:.3f}")
    print(f"meteor: {text_metrics['meteor']:.3f}")
    print(f"bertscore_f1: {text_metrics['bertscore_f1']:.3f}")
    print(f"sacrebleu: {corpus_metrics['sacrebleu']:.3f}")
    print(f"chrf: {corpus_metrics['chrf']:.3f}")
    print(f"chrf_pp: {corpus_metrics['chrf_pp']:.3f}")


study_records: list[dict[str, Any]] = []
for study in tqdm(iu_eval_manifest, total=len(iu_eval_manifest), desc="Collecting evaluation rows", unit="study"):
    stage1_path = stage1_output_path(study.uid)
    stage2a_path = STAGE2A_DIR / f"{study.uid}.json"
    stage2c_path = STAGE2C_DIR / f"{study.uid}.json"
    stage3_path = STAGE3_DIR / f"{study.uid}.json"
    stage4_path = STAGE4_DIR / f"{study.uid}.json"
    stage5_path = STAGE5_DIR / f"{study.uid}.json"
    chexone_direct_path = chexone_direct_output_path(study.uid)
    pipeline_judge_path = JUDGING_PIPELINE_DIR / f"{study.uid}.json"
    chexone_judge_path = JUDGING_CHEXONE_DIR / f"{study.uid}.json"

    stage1_payload = read_json(stage1_path) if stage1_path.exists() else {}
    stage5_payload = read_json(stage5_path) if stage5_path.exists() else {}
    chexone_direct_payload = read_json(chexone_direct_path) if chexone_direct_path.exists() else {}
    pipeline_judge_payload = read_json(pipeline_judge_path) if pipeline_judge_path.exists() else {}
    chexone_judge_payload = read_json(chexone_judge_path) if chexone_judge_path.exists() else {}

    top_retrieved_ids: list[str] = []
    if stage2c_path.exists():
        stage2c_payload = read_json(stage2c_path)
        top_retrieved_ids = [str(row["study_id"]) for row in stage2c_payload.get("reranked_hits", [])]

    study_records.append(
        {
            "study": study,
            "stage1_path": stage1_path,
            "stage2a_path": stage2a_path,
            "stage2c_path": stage2c_path,
            "stage3_path": stage3_path,
            "stage4_path": stage4_path,
            "stage5_path": stage5_path,
            "chexone_direct_path": chexone_direct_path,
            "pipeline_judge_path": pipeline_judge_path,
            "chexone_judge_path": chexone_judge_path,
            "stage1_payload": stage1_payload,
            "stage5_payload": stage5_payload,
            "chexone_direct_payload": chexone_direct_payload,
            "pipeline_judge_payload": pipeline_judge_payload,
            "chexone_judge_payload": chexone_judge_payload,
            "reference_report_metric_text": normalize_report_text_for_metrics(study.ground_truth_report),
            "pipeline_report": str(stage5_payload.get("report_text", "")) if isinstance(stage5_payload, dict) else "",
            "pipeline_report_metric_text": normalize_report_text_for_metrics(
                str(stage5_payload.get("report_text", "")) if isinstance(stage5_payload, dict) else ""
            ),
            "chexone_direct_report": str(chexone_direct_payload.get("report_text", "")) if isinstance(chexone_direct_payload, dict) else "",
            "chexone_direct_report_metric_text": normalize_report_text_for_metrics(
                str(chexone_direct_payload.get("report_text", "")) if isinstance(chexone_direct_payload, dict) else ""
            ),
            "top_retrieved_ids": top_retrieved_ids,
        }
    )


reference_reports = [record["reference_report_metric_text"] for record in study_records]
pipeline_reports = [record["pipeline_report_metric_text"] for record in study_records]
chexone_direct_reports = [record["chexone_direct_report_metric_text"] for record in study_records]
pipeline_bertscores = compute_bertscore_batch(reference_reports, pipeline_reports)
chexone_direct_bertscores = compute_bertscore_batch(reference_reports, chexone_direct_reports)

rows: list[dict[str, Any]] = []
pipeline_judge_scores: list[float] = []
chexone_direct_judge_scores: list[float] = []
flash_retry_count = 0
flash_failures = 0
pipeline_judge_retry_count = 0
pipeline_judge_failures = 0
chexone_direct_judge_retry_count = 0
chexone_direct_judge_failures = 0

for idx, record in enumerate(
    tqdm(study_records, total=len(study_records), desc="Scoring evaluation metrics", unit="study")
):
    study = record["study"]
    stage5_payload = record["stage5_payload"]
    chexone_direct_payload = record["chexone_direct_payload"]
    pipeline_judge_payload = record["pipeline_judge_payload"]
    chexone_judge_payload = record["chexone_judge_payload"]

    pipeline_text_metrics = compute_text_overlap_metrics(record["reference_report_metric_text"], record["pipeline_report_metric_text"])
    chexone_direct_text_metrics = compute_text_overlap_metrics(record["reference_report_metric_text"], record["chexone_direct_report_metric_text"])

    flash_cache_path = Path(stage5_payload.get("cache_path", "")) if isinstance(stage5_payload, dict) and stage5_payload.get("cache_path") else None
    pipeline_judge_cache_path = (
        Path(pipeline_judge_payload.get("cache_path", ""))
        if isinstance(pipeline_judge_payload, dict) and pipeline_judge_payload.get("cache_path")
        else None
    )
    chexone_judge_cache_path = (
        Path(chexone_judge_payload.get("cache_path", ""))
        if isinstance(chexone_judge_payload, dict) and chexone_judge_payload.get("cache_path")
        else None
    )
    flash_cache = read_json(flash_cache_path) if flash_cache_path and flash_cache_path.exists() else {}
    pipeline_judge_cache = read_json(pipeline_judge_cache_path) if pipeline_judge_cache_path and pipeline_judge_cache_path.exists() else {}
    chexone_judge_cache = read_json(chexone_judge_cache_path) if chexone_judge_cache_path and chexone_judge_cache_path.exists() else {}

    flash_retry_count += max(0, int(flash_cache.get("attempt_count", 0)) - (1 if flash_cache.get("status") == "success" else 0))
    pipeline_judge_retry_count += max(
        0,
        int(pipeline_judge_cache.get("attempt_count", 0)) - (1 if pipeline_judge_cache.get("status") == "success" else 0),
    )
    chexone_direct_judge_retry_count += max(
        0,
        int(chexone_judge_cache.get("attempt_count", 0)) - (1 if chexone_judge_cache.get("status") == "success" else 0),
    )
    if flash_cache.get("status") == "failed_permanent":
        flash_failures += 1
    if pipeline_judge_cache.get("status") == "failed_permanent":
        pipeline_judge_failures += 1
    if chexone_judge_cache.get("status") == "failed_permanent":
        chexone_direct_judge_failures += 1

    pipeline_judge_output = pipeline_judge_payload.get("judge_output", {}) if isinstance(pipeline_judge_payload, dict) else {}
    chexone_judge_output = chexone_judge_payload.get("judge_output", {}) if isinstance(chexone_judge_payload, dict) else {}
    if isinstance(pipeline_judge_output, dict) and "overall_score" in pipeline_judge_output:
        pipeline_judge_scores.append(float(pipeline_judge_output["overall_score"]))
    if isinstance(chexone_judge_output, dict) and "overall_score" in chexone_judge_output:
        chexone_direct_judge_scores.append(float(chexone_judge_output["overall_score"]))

    rows.append(
        {
            "uid": study.uid,
            "image_path": study.image_path,
            "stage1_path": str(record["stage1_path"]),
            "stage2a_path": str(record["stage2a_path"]),
            "stage2c_path": str(record["stage2c_path"]),
            "stage3_path": str(record["stage3_path"]),
            "stage4_path": str(record["stage4_path"]),
            "stage5_path": str(record["stage5_path"]),
            "chexone_direct_path": str(record["chexone_direct_path"]),
            "pipeline_judge_path": str(record["pipeline_judge_path"]),
            "chexone_direct_judge_path": str(record["chexone_judge_path"]),
            "top_retrieved_ids": "|".join(record["top_retrieved_ids"]),
            "stage1_report_text": str(record["stage1_payload"].get("report_text", "")) if isinstance(record["stage1_payload"], dict) else "",
            "reference_report_metric_text": record["reference_report_metric_text"],
            "pipeline_report_metric_text": record["pipeline_report_metric_text"],
            "chexone_direct_report_metric_text": record["chexone_direct_report_metric_text"],
            "pipeline_report_text": record["pipeline_report"],
            "chexone_direct_report_text": record["chexone_direct_report"],
            "pipeline_judge_overall_score": pipeline_judge_output.get("overall_score", ""),
            "chexone_direct_judge_overall_score": chexone_judge_output.get("overall_score", ""),
            "pipeline_bleu": pipeline_text_metrics["bleu"],
            "pipeline_rouge1": pipeline_text_metrics["rouge1"],
            "pipeline_rouge2": pipeline_text_metrics["rouge2"],
            "pipeline_rougeL": pipeline_text_metrics["rougeL"],
            "pipeline_meteor": pipeline_text_metrics["meteor"],
            "pipeline_bertscore_f1": pipeline_bertscores[idx],
            "chexone_direct_bleu": chexone_direct_text_metrics["bleu"],
            "chexone_direct_rouge1": chexone_direct_text_metrics["rouge1"],
            "chexone_direct_rouge2": chexone_direct_text_metrics["rouge2"],
            "chexone_direct_rougeL": chexone_direct_text_metrics["rougeL"],
            "chexone_direct_meteor": chexone_direct_text_metrics["meteor"],
            "chexone_direct_bertscore_f1": chexone_direct_bertscores[idx],
            "composer_cache_status": flash_cache.get("status", ""),
            "pipeline_judge_cache_status": pipeline_judge_cache.get("status", ""),
            "chexone_direct_judge_cache_status": chexone_judge_cache.get("status", ""),
            "composer_cache_hit": bool(stage5_payload.get("cache_hit", False)) if isinstance(stage5_payload, dict) else False,
            "pipeline_judge_cache_hit": bool(pipeline_judge_payload.get("cache_hit", False)) if isinstance(pipeline_judge_payload, dict) else False,
            "chexone_direct_judge_cache_hit": bool(chexone_judge_payload.get("cache_hit", False)) if isinstance(chexone_judge_payload, dict) else False,
        }
    )

pipeline_completed_rows = [row for row in rows if row.get("pipeline_report_text")]
chexone_completed_rows = [row for row in rows if row.get("chexone_direct_report_text")]
corpus_text_metrics = {
    "pipeline": {
        "all_requested": compute_corpus_generation_metrics(reference_reports, pipeline_reports),
        "completed_only": compute_corpus_generation_metrics(
            [row["reference_report_metric_text"] for row in pipeline_completed_rows],
            [row["pipeline_report_metric_text"] for row in pipeline_completed_rows],
        ),
    },
    "chexone_direct": {
        "all_requested": compute_corpus_generation_metrics(reference_reports, chexone_direct_reports),
        "completed_only": compute_corpus_generation_metrics(
            [row["reference_report_metric_text"] for row in chexone_completed_rows],
            [row["chexone_direct_report_metric_text"] for row in chexone_completed_rows],
        ),
    },
}

summary_json = {
    "artifact_root": str(ARTIFACT_ROOT),
    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
    "device": DEVICE,
    "iu_eval_limit": IU_EVAL_LIMIT,
    "retrieval_top_k": RETRIEVAL_TOP_K,
    "force_rebuild_banks": FORCE_REBUILD_BANKS,
    "force_stage1_refresh": FORCE_STAGE1_REFRESH,
    "force_chexone_direct_refresh": FORCE_CHEXONE_DIRECT_REFRESH,
    "force_stage5_refresh": FORCE_STAGE5_REFRESH,
    "force_stage7_refresh": FORCE_STAGE7_REFRESH,
    "iu_train_manifest_path": str(IU_TRAIN_MANIFEST_PATH),
    "iu_eval_manifest_path": str(IU_EVAL_MANIFEST_PATH),
    "mimic_bank_manifest_path": str(MIMIC_BANK_MANIFEST_PATH),
    "iu_train_bank_manifest_path": str(IU_TRAIN_BANK_MANIFEST_PATH),
    "retrieval_bank_manifest_path": str(RETRIEVAL_BANK_MANIFEST_PATH),
    "visual_bank_manifest": visual_bank_manifest,
    "num_studies": len(iu_eval_manifest),
    "completed_report_count": {
        "pipeline": len(pipeline_completed_rows),
        "chexone_direct": len(chexone_completed_rows),
    },
    "completed_judge_count": {
        "pipeline": len(pipeline_judge_scores),
        "chexone_direct": len(chexone_direct_judge_scores),
    },
    "failed_judge_count": {
        "pipeline": pipeline_judge_failures,
        "chexone_direct": chexone_direct_judge_failures,
    },
    "mean_judge_overall_score": {
        "pipeline": sum(pipeline_judge_scores) / len(pipeline_judge_scores) if pipeline_judge_scores else 0.0,
        "chexone_direct": sum(chexone_direct_judge_scores) / len(chexone_direct_judge_scores) if chexone_direct_judge_scores else 0.0,
    },
    "mean_text_metrics": {
        "pipeline": {
            "all_requested": {
                "bleu": aggregate_metric(rows, "pipeline_bleu"),
                "rouge1": aggregate_metric(rows, "pipeline_rouge1"),
                "rouge2": aggregate_metric(rows, "pipeline_rouge2"),
                "rougeL": aggregate_metric(rows, "pipeline_rougeL"),
                "meteor": aggregate_metric(rows, "pipeline_meteor"),
                "bertscore_f1": aggregate_metric(rows, "pipeline_bertscore_f1"),
            },
            "completed_only": {
                "bleu": aggregate_metric(pipeline_completed_rows, "pipeline_bleu"),
                "rouge1": aggregate_metric(pipeline_completed_rows, "pipeline_rouge1"),
                "rouge2": aggregate_metric(pipeline_completed_rows, "pipeline_rouge2"),
                "rougeL": aggregate_metric(pipeline_completed_rows, "pipeline_rougeL"),
                "meteor": aggregate_metric(pipeline_completed_rows, "pipeline_meteor"),
                "bertscore_f1": aggregate_metric(pipeline_completed_rows, "pipeline_bertscore_f1"),
            },
        },
        "chexone_direct": {
            "all_requested": {
                "bleu": aggregate_metric(rows, "chexone_direct_bleu"),
                "rouge1": aggregate_metric(rows, "chexone_direct_rouge1"),
                "rouge2": aggregate_metric(rows, "chexone_direct_rouge2"),
                "rougeL": aggregate_metric(rows, "chexone_direct_rougeL"),
                "meteor": aggregate_metric(rows, "chexone_direct_meteor"),
                "bertscore_f1": aggregate_metric(rows, "chexone_direct_bertscore_f1"),
            },
            "completed_only": {
                "bleu": aggregate_metric(chexone_completed_rows, "chexone_direct_bleu"),
                "rouge1": aggregate_metric(chexone_completed_rows, "chexone_direct_rouge1"),
                "rouge2": aggregate_metric(chexone_completed_rows, "chexone_direct_rouge2"),
                "rougeL": aggregate_metric(chexone_completed_rows, "chexone_direct_rougeL"),
                "meteor": aggregate_metric(chexone_completed_rows, "chexone_direct_meteor"),
                "bertscore_f1": aggregate_metric(chexone_completed_rows, "chexone_direct_bertscore_f1"),
            },
        },
    },
    "corpus_text_metrics": corpus_text_metrics,
    "llm_cache_hits": {
        "flash": sum(1 for row in rows if row["composer_cache_hit"]),
        "judge_pipeline": sum(1 for row in rows if row["pipeline_judge_cache_hit"]),
        "judge_chexone_direct": sum(1 for row in rows if row["chexone_direct_judge_cache_hit"]),
    },
    "llm_retries": {
        "flash": flash_retry_count,
        "judge_pipeline": pipeline_judge_retry_count,
        "judge_chexone_direct": chexone_direct_judge_retry_count,
    },
    "llm_failures": {
        "flash": flash_failures,
        "judge_pipeline": pipeline_judge_failures,
        "judge_chexone_direct": chexone_direct_judge_failures,
    },
    "rows": rows,
    "updated_at": utc_now(),
}

if ENABLE_LLM_JUDGE:
    pipeline_mean_judge_score = (
        sum(pipeline_judge_scores) / len(pipeline_judge_scores)
        if pipeline_judge_scores
        else 0.0
    )
    chexone_direct_mean_judge_score = (
        sum(chexone_direct_judge_scores) / len(chexone_direct_judge_scores)
        if chexone_direct_judge_scores
        else 0.0
    )
else:
    pipeline_mean_judge_score = None
    chexone_direct_mean_judge_score = None

summary_json["mean_judge_overall_score"] = {
    "pipeline": pipeline_mean_judge_score,
    "chexone_direct": chexone_direct_mean_judge_score,
}
summary_json["completed_judge_count"] = {
    "pipeline": len(pipeline_judge_scores),
    "chexone_direct": len(chexone_direct_judge_scores),
}
summary_json["failed_judge_count"] = {
    "pipeline": pipeline_judge_failures,
    "chexone_direct": chexone_direct_judge_failures,
}


summary_json_path = EVAL_DIR / "summary.json"
write_json(summary_json_path, summary_json)

summary_csv_path = EVAL_DIR / "per_study_summary.csv"
with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)

print(f"Evaluation summary JSON: {summary_json_path}")
print(f"Evaluation summary CSV: {summary_csv_path}")
if "mean_label_f1" in summary_json:
    print(f"Completed-only label F1: pipeline={summary_json['mean_label_f1']['pipeline_completed_only']:.3f}, chexone_direct={summary_json['mean_label_f1']['chexone_direct_completed_only']:.3f}")
pipeline_judge_score = summary_json["mean_judge_overall_score"]["pipeline"] if ENABLE_LLM_JUDGE else None
chexone_judge_score = summary_json["mean_judge_overall_score"]["chexone_direct"] if ENABLE_LLM_JUDGE else None
if not ENABLE_LLM_JUDGE:
    print("LLM judge disabled; judge_overall not computed.")

print_result_block(
    "Pipeline Results",
    summary_json["mean_text_metrics"]["pipeline"]["completed_only"],
    summary_json["corpus_text_metrics"]["pipeline"]["completed_only"],
    pipeline_judge_score,
    len(pipeline_completed_rows),
    len(pipeline_judge_scores),
    len(iu_eval_manifest),
)
print_result_block(
    "CheXOne Direct Baseline",
    summary_json["mean_text_metrics"]["chexone_direct"]["completed_only"],
    summary_json["corpus_text_metrics"]["chexone_direct"]["completed_only"],
    chexone_judge_score,
    len(chexone_completed_rows),
    len(chexone_direct_judge_scores),
    len(iu_eval_manifest),
)
