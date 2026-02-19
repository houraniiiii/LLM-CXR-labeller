# CXR Labeller

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/vLLM-Inference-4B8BBE?logo=nvidia&logoColor=white" alt="vLLM">
  <img src="https://img.shields.io/badge/Hugging%20Face-Models-FFCC4D?logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Task-CXR%20report%20labeling-0A7E8C" alt="Task">
</p>

## Abstract

This pipeline turns free-text chest X-ray reports into the standard 14 CXR finding labels used in clinical benchmarking. It lets you compare different open LLMs, prompt styles, quantization levels, and few-shot setups in one consistent workflow, so you can understand the quality, speed, and efficiency tradeoffs before selecting a deployment configuration.

Text-only chest X-ray (CXR) report labeling pipeline for CheXpert-style 14-label prediction, with zero-shot and retrieval-based few-shot inference using open-weight LLMs.

## What this repo does

- Loads harmonized CXR report datasets (`dev`, `test`, and per-dataset splits).
- Builds MedCPT retrieval archives for few-shot example selection.
- Runs batched vLLM inference across model, prompt, quantization, and k-shot settings.
- Saves prediction JSONL files, logs, and scoring artifacts for publication reporting.

## Repository layout

- `src/core/`: inference engine, model registry, prompts, retrieval, data loading.
- `src/cli/ppe_retrieval.py`: main CLI entrypoint for inference runs.
- `configs/run_defaults.yaml`: default run configuration.
- `models.jsonl`: model catalog and default batch sizes per quant mode and k-shot.
- `results/`: predictions, logs, and derived outputs.

## Quick start

### 1) Environment setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run a smoke-test inference

```bash
python -m src.cli.ppe_retrieval \
  --dataset all \
  --split dev \
  --quant-mode fp16 \
  --model openai/gpt-oss-20b \
  --k-shot 0 \
  --prompt-variants clinical_compact \
  --limit 20
```

## Full pipeline


### 1) Generate MedCPT embeddings

```bash
python src/generate_medcpt_article_embeddings.py --input data/few_shot/few_shot_50.csv --output data/embeddings/few_shot_50_medcpt_article.jsonl --revision <HF_COMMIT> --normalize
python src/generate_medcpt_article_embeddings.py --input data/dev/dev_all.csv --output data/embeddings/dev_all_medcpt_article.jsonl --revision <HF_COMMIT> --normalize
python src/generate_medcpt_article_embeddings.py --input data/test/test_all.csv --output data/embeddings/test_all_medcpt_article.jsonl --revision <HF_COMMIT> --normalize
```

### 2) Build retrieval archives

```bash
python src/build_medcpt_fewshot_retrieval_archives.py --top-k 10
```

### 3) Run inference

Example: all models for dev split, fp16, k=5, one prompt variant.

```bash
python -m src.cli.ppe_retrieval \
  --dataset all \
  --split dev \
  --quant-mode fp16 \
  --all-models \
  --k-shot 5 \
  --prompt-variants clinical_stepwise
```

You can also run test split and INT4:

```bash
python -m src.cli.ppe_retrieval --dataset all --split test --quant-mode int4 --model Qwen/Qwen3-4B-Instruct-2507 --k-shot 5 --prompt-variants clinical_compact
```


## Outputs

- Predictions: `results/preds/<dataset_split>/*.jsonl`
- Logs: `results/logs/*.log`

## Figure

<p align="center">
  <img src="figures/figure2_macro_f1_vs_rpm_test.png" alt="Macro-F1 vs RPM on test split" width="900">
</p>
<p align="center"><em>n=1250</em></p>

---

## Acknowledgement

<p>
  Powered by infrastructure support from <strong>SimplePod.ai</strong>, an HPC GPU hosting company located in Europe (Poland).
</p>

<p>
  Contact: <a href="mailto:help@simplepod.ai">help@simplepod.ai</a>
</p>

<p>
  <img src="figures/simplepod-logo-black.png" alt="SimplePod logo" width="220">
</p>
