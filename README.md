## [WIP] Tini-Omni (Based on SLAM-LLM / SLAM-Omni)

**Tini-Omni** is a customized, lightweight speech-to-speech (S2S) conversational system built on top of the original projects **SLAM-LLM** and **SLAM-Omni**.

Compared to the original SLAM-Omni implementation, this repository:

- Uses a **modified model architecture / configuration**, referred to as **Tini-Omni**, while keeping the overall S2S training and inference pipeline compatible with SLAM-Omni.
- Replaces the original **Qwen2-0.5B** backbone with **Gemma 3 270M**, making the core LLM significantly smaller and lighter.
- Scales down the adapter modules to roughly **1 / 6.5** of the original size, further reducing memory footprint and computation cost.
- Simplifies dependency management via the root-level `requirements.txt` and `pyproject.toml`, so the whole project can be installed as a Python package during development.

Unlike the original `SLAM-LLM` repository, which contains multiple tasks, this repository focuses on a **single end-to-end pipeline for (Tini-)Omni S2S** under `examples/s2s`. Other example recipes from the original project have been removed; only files and scripts that exist in this directory are considered part of the supported pipeline.

---

## Repository Layout

- **Root**
  - `pyproject.toml` – Python package configuration. The package name is `slam-llm`, and dependencies are loaded from `requirements.txt`.
  - `requirements.txt` – Core Python dependencies (e.g. `torch`, `transformers`, `peft`, `datasets`, `hydra-core`).
  - `src/slam_llm/` – Shared training / inference utilities (model construction, configuration management, distributed training helpers, etc.).
  - `examples/` – Example tasks; in this fork only `s2s` (Tini-/SLAM-Omni) is kept.

-- **`examples/s2s/` (Tini-/SLAM-Omni main example)**
  - `finetune_s2s.py` – Python entry point for S2S training / fine-tuning (used by shell scripts under `scripts/finetune/`).
  - `inference_s2s.py` – Python entry point for S2S inference (used by shell scripts under `scripts/inference/`).
  - `s2s_config.py` – Dataclass-style configuration for model, training, data and decoding; this is where the **Tini-Omni model configuration** is defined.
  - `conf/` – Hydra / YAML configs (e.g. `prompt.yaml`) used together with the training and inference scripts.
  - `scripts/`
    - `finetune/` – S2S training script `finetune_s2s_group.sh` (Gemma 3 270M + CosyVoice-based training; Mini-Omni scripts from upstream have been removed).
    - `inference/` – Inference scripts such as `inference_s2s_online.sh` and `inference_s2s_batch.sh` (single-turn online / batch inference).
  - `generate/` – High-level generation interfaces (online dialogue, audio generation, etc.).
  - `model/` – Tini-/S2S-specific model definitions and factory functions.
  - `speech_dataset_s2s.py` – Dataset loading and preprocessing for the S2S task.
  - `audio_prompt/` – Example timbre prompts for conditioning the voice.

The following sections summarize the S2S recipe implemented in this repository (adapted from the original SLAM-Omni documentation). For more extensive background on datasets and training strategies, please refer to the original SLAM-LLM / SLAM-Omni repository.

---

## Installation & Environment

We recommend using **Python 3.10** and a CUDA-enabled PyTorch build if you plan to train models or run inference on GPU.

A minimal setup for working with this repository locally looks like:

```bash
cd /path/to/Tini-Omni   # this repository

# Install the project itself in editable (development) mode
pip install -e .
```
---

## Tini-Omni Recipe

- **Task type**: speech-to-speech (S2S) dialogue with timbre control, covering only English.
- **Data format**: Parquet and JSONL are supported.
- **Training & inference entry points**:
  - Training / fine-tuning: `examples/s2s/scripts/finetune/finetune_s2s_group.sh` + `finetune_s2s.py`.
  - Online / batch inference: `examples/s2s/scripts/inference/inference_s2s_online.sh` and `examples/s2s/scripts/inference/inference_s2s_batch.sh` + `inference_s2s.py` / modules under `generate/`.

This repository aims to provide a **ready-to-use and easy-to-extend engineering template** for Tini-Omni / SLAM-Omni style S2S interaction systems, while remaining compatible with the upstream SLAM-LLM tooling stack.

### Environment setup (S2S example)

After installing the core project (see **Installation & Environment**), the S2S example can use extra dependencies listed under:

```bash
pip install -r ./examples/s2s/requirements.txt
```

### Data preparation

The S2S recipe supports two data formats:

- **Parquet** (recommended; Tini-Omni follows the same Parquet-based data layout as the original SLAM-Omni implementation).
- **JSONL**, with a compact per-utterance structure that contains source audio, source text and target codec tokens.

Tini-Omni experiments in this repository focus on **single-round English dialogue**, using the `worstchan/VoiceAssistant-400K-SLAM-Omni` dataset from the original SLAM-Omni release.  
For more details on available datasets and their exact schemas, please refer to the original SLAM-Omni documentation and data cards.

### Training

The S2S example currently exposes a single fine-tuning entry script (which internally calls `finetune_s2s.py`):

```bash
bash ./examples/s2s/scripts/finetune/finetune_s2s_group.sh
```

This script configures a Gemma 3 270M backbone, CosyVoice-based codec, Whisper encoder and related vocabulary settings via Hydra arguments.

### Inference

For inference (online single-turn or batch processing), the example offers:

```bash
# Online S2S inference (single-turn)
bash ./examples/s2s/scripts/inference/inference_s2s_online.sh

# Batch inference (Parquet / JSONL)
bash ./examples/s2s/scripts/inference/inference_s2s_batch.sh
```

These scripts call into `inference_s2s.py` and helper functions under `generate/`, using Hydra configs under `conf/`. Tini-Omni keeps the same CLI surface as these scripts, with model changes encapsulated in `s2s_config.py` and the model factory.

---

## Development & Extension

- **Reuse the core library**: When adding new tasks or model variants, try to reuse the utilities in `src/slam_llm` (training loop, Hydra + dataclass config system, distributed training helpers, etc.) instead of re-implementing them under `examples/`.
- **Add new examples**: For additional tasks, follow the structure of `examples/s2s`:
  - Place task-specific dataset / model / generation code and scripts under a dedicated subdirectory.
  - Promote generic utilities back into `src/slam_llm` whenever possible, to avoid duplication.

---

## License & Acknowledgements

- The code in this repository follows the original **MIT License** of SLAM-LLM / SLAM-Omni (see `LICENSE`).
- This project heavily builds on the original **SLAM-LLM** and **SLAM-Omni** implementations, as well as related open-source projects such as Mini-Omni and CosyVoice. We gratefully acknowledge their authors and maintainers.

Original SLAM-LLM repository: `https://github.com/X-LANCE/SLAM-LLM`

---

## Citation

```bibtex
@article{chen2024slam,
  title={SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training},
  author={Chen, Wenxi and Ma, Ziyang and Yan, Ruiqi and Liang, Yuzhe and Li, Xiquan and Xu, Ruiyang and Niu, Zhikang and Zhu, Yanqiao and Yang, Yifan and Liu, Zhanxun and others},
  journal={arXiv preprint arXiv:2412.15649},
  year={2024}
}

@article{xie2024mini,
  title={Mini-omni: Language models can hear, talk while thinking in streaming},
  author={Xie, Zhifei and Wu, Changqiao},
  journal={arXiv preprint arXiv:2408.16725},
  year={2024}
}

@article{gemma_2025,
    title={Gemma 3},
    url={https://arxiv.org/abs/2503.19786},
    publisher={Google DeepMind},
    author={Gemma Team},
    year={2025}
}
```

For additional ASR / TTS / audio understanding tasks and their citations, please refer to the original `SLAM-LLM` documentation. 
