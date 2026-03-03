# Probabilistic Multi-View Crime Scene Reconstruction from Natural Language Descriptions

A unified multi-objective generative AI system that reconstructs crime scenes from natural language descriptions. The system implements a **12-stage pipeline** combining NLP parsing, scene graph reasoning, probabilistic spatial inference, depth-conditioned controlled diffusion generation, and a **7-component unified scoring function S(R)**.

> **Best Score:** S(R) = 0.739 on bedroom crime scene (Realistic Vision v5.1 + ControlNet Depth)

---

## System Architecture

```
Text Description
        |
  Stage 0:  Environment Initialization (models, aliases, memory config)
        |
  Stage 1:  NLP Parsing (spaCy en_core_web_sm)
        |
  Stage 2:  Vocabulary Normalization (Visual Genome aliases)
        |
  Stage 3:  Scene Graph Construction (NetworkX)
        |
  Stage 4:  Multi-Hypothesis Generation (probabilistic sampling)
        |
  Stage 5:  Spatial Layout Estimation (2D coords + depth ordering)
        |
  Stage 6:  Depth Map Generation (MiDaS DPT-Hybrid / room-aware synthetic)
        |
  Stage 7:  Two-Pass ControlNet + Diffusion Image Generation
        |
  Stage 8:  Multi-View Rendering (depth perturbation)
        |
  Stage 9:  Explainability Report (JSON reasoning chain)
        |
  Stage 10: Unified 7-Component Scoring (S(R) evaluation)
        |
  Stage 11: Result Packaging (structured output)
```

---

## Unified Scoring Function S(R)

The system evaluates reconstructions using a weighted combination of seven quality components:

| Component | Weight | Method |
|-----------|--------|--------|
| Semantic Alignment | 0.20 | CLIP similarity + object recall + relationship satisfaction |
| Spatial Consistency | 0.15 | Constraint violations, position/depth ordering errors |
| Physical Plausibility | 0.10 | Gravity alignment, support relations, scale realism |
| Visual Realism | 0.15 | Aesthetic CLIP score, noise residual, sharpness |
| Probabilistic Prior | 0.10 | Visual Genome co-occurrence + spatial relation log-likelihood |
| Multi-View Consistency | 0.10 | CLIP similarity across depth-perturbed views |
| Perceptual Believability | 0.20 | CLIP realism probe, scene coherence, uncanny penalty |

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3060 (6 GB VRAM) or equivalent |
| Precision | FP16 |
| Resolution | 512 x 512 |
| Memory Optimizations | Attention slicing, VAE slicing, sequential CPU offload |

---

## Setup

### 1. Create Environment

```bash
conda create -n fed-skel-gpu python=3.10 -y
conda activate fed-skel-gpu
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download Models

```bash
python scripts/download_models.py
```

This downloads the following to `models/` (not tracked by git):

| Model | ID | Purpose |
|-------|----|---------|
| Realistic Vision v5.1 | `SG161222/Realistic_Vision_V5.1_noVAE` | Photorealistic image generation |
| Fine-tuned VAE | `stabilityai/sd-vae-ft-mse` | Improved VAE decoding |
| ControlNet Depth | `lllyasviel/sd-controlnet-depth` | Depth-conditioned generation |
| MiDaS DPT-Hybrid | `Intel/dpt-hybrid-midas` | Monocular depth estimation |
| CLIP ViT-B/32 | OpenAI (via open_clip) | Evaluation metrics |

### 4. Download Datasets

Place datasets in `Data/` (not tracked by git):

- **Visual Genome** — `object_alias.txt`, `relationship_alias.txt`, `objects.json`, `relationships.json`
- **MS COCO 2017** — train/val images + annotations
- **CLEVR v1.0** — images, scenes, questions

### 5. Verify GPU

```bash
python scripts/verify_gpu.py
```

---

## Usage

### Run Full Pipeline

```bash
python -m src.pipeline --input "A dimly lit bedroom with a broken window, a bloodstained mattress on the floor, and a knife near the doorway"
```

### Run Research Pipeline (with optimization + ablation)

```bash
python -m src.research_pipeline --input "Small kitchen with broken glass on the floor, an overturned chair, and a bloodstain on the counter"
```

### Options

```bash
python -m src.pipeline \
  --input "Dark alley with a dumpster against the wall and blood on the ground" \
  --no-multiview    # Skip multi-view generation (faster) \
  --no-eval         # Skip CLIP evaluation (less VRAM) \
  --no-controlnet   # Use plain SD instead of ControlNet
```

---

## Project Structure

```
CrimeSceneReconstruction/
├── configs/
│   ├── default_config.yaml          # Pipeline parameters
│   └── research_config.yaml         # Research framework parameters
├── scripts/
│   ├── download_models.py           # Auto-download pretrained models
│   ├── generate_report.py           # PDF report generator (reportlab)
│   ├── verify_gpu.py                # GPU verification
│   ├── test_diffusion.py            # Diffusion generation test
│   └── test_nlp_pipeline.py         # NLP pipeline test
├── src/
│   ├── pipeline.py                  # Main 12-stage orchestrator
│   ├── research_pipeline.py         # Research pipeline (optimization + ablation)
│   ├── __init__.py
│   ├── stages/
│   │   ├── stage0_init.py           # Environment initialization
│   │   ├── stage1_text_understanding.py
│   │   ├── stage2_vocabulary_normalization.py
│   │   ├── stage3_scene_graph.py
│   │   ├── stage4_hypothesis_generation.py
│   │   ├── stage5_spatial_layout.py
│   │   ├── stage6_depth_map.py
│   │   ├── stage7_image_generation.py
│   │   ├── stage8_multiview.py
│   │   ├── stage9_explainability.py
│   │   ├── stage10_evaluation.py
│   │   └── stage11_packaging.py
│   ├── scoring/
│   │   ├── unified_scorer.py        # Weighted 7-component scorer
│   │   ├── semantic_alignment.py    # CLIP + object recall
│   │   ├── spatial_consistency.py   # Constraint satisfaction
│   │   ├── physical_plausibility.py # Gravity, support, scale
│   │   ├── visual_realism.py        # Aesthetic + noise + sharpness
│   │   ├── probabilistic_prior.py   # VG co-occurrence priors
│   │   ├── multiview_consistency.py # Cross-view CLIP
│   │   └── perceptual_believability.py # Realism probe
│   ├── optimization/
│   │   ├── energy_optimizer.py      # Simulated Annealing + Evolutionary Strategies
│   │   └── weight_calibration.py    # Scoring weight calibration
│   ├── correction/
│   │   └── closed_loop.py           # Iterative weakest-component correction
│   ├── conditioning/
│   │   └── segmentation_layout.py   # Segmentation-based conditioning
│   ├── experiments/
│   │   ├── experiment_runner.py     # Experiment orchestration
│   │   ├── ablation_runner.py       # Ablation study framework
│   │   └── research_logger.py       # Structured experiment logging
│   └── utils/
│       ├── config.py                # Configuration management
│       ├── memory.py                # GPU memory utilities
│       └── logging_utils.py         # Logging setup
├── tests/
│   └── test_research_framework.py   # Framework tests
├── requirements.txt
├── .gitignore
└── README.md
```

### Not Tracked (gitignored)

| Directory | Contents | Reason |
|-----------|----------|--------|
| `Data/` | COCO, Visual Genome, CLEVR datasets | Large size, copyrighted |
| `models/` | Stable Diffusion, ControlNet, MiDaS, VAE weights | Large size, licensed |
| `outputs/` | Generated images, depth maps, reports, logs | Transient pipeline outputs |
| `.venv/` | Python virtual environment | Environment-specific |

---

## Pipeline Stages

| Stage | Module | Purpose |
|-------|--------|---------|
| 0 | `stage0_init.py` | Load models, aliases, spaCy; configure GPU memory |
| 1 | `stage1_text_understanding.py` | Parse text into objects, attributes, relationships |
| 2 | `stage2_vocabulary_normalization.py` | Canonicalize names via Visual Genome aliases |
| 3 | `stage3_scene_graph.py` | Build directed graph (NetworkX) |
| 4 | `stage4_hypothesis_generation.py` | Generate N plausible spatial layouts |
| 5 | `stage5_spatial_layout.py` | Map to 2D pixel coords + depth ordering |
| 6 | `stage6_depth_map.py` | Room-aware synthetic depth / MiDaS estimation |
| 7 | `stage7_image_generation.py` | Two-pass ControlNet + Realistic Vision v5.1 |
| 8 | `stage8_multiview.py` | Multi-view rendering via depth perturbation |
| 9 | `stage9_explainability.py` | Full reasoning chain (JSON) |
| 10 | `stage10_evaluation.py` | Unified 7-component S(R) scoring |
| 11 | `stage11_packaging.py` | Structured output assembly |

---

## Key Results

### Bedroom Scene: S(R) = 0.739

| Component | Score |
|-----------|-------|
| Semantic Alignment | 0.783 |
| Spatial Consistency | 0.999 |
| Physical Plausibility | 0.923 |
| Visual Realism | 0.515 |
| Probabilistic Prior | 0.816 |
| Multi-View Consistency | 0.500 |
| Perceptual Believability | 0.654 |
| **Unified S(R)** | **0.739** |

### Model Comparison

| Metric | SD v1.4 | Realistic Vision v5.1 |
|--------|---------|----------------------|
| Visual Realism | 0.402 | 0.515 |
| Overall S(R) | 0.676 | 0.739 |

---

## Memory Optimizations (RTX 3060 6 GB)

1. **FP16 precision** — halves model memory footprint
2. **Attention slicing** — processes attention in chunks
3. **VAE slicing** — decodes VAE in slices
4. **Sequential CPU offload** — streams model layers from CPU RAM to GPU
5. **Lazy model loading** — loads models only when needed
6. **Model unloading** — frees VRAM between pipeline stages
7. **MiDaS on CPU** — depth estimation offloaded to CPU

---

## Technologies

- **Python 3.10** | PyTorch 2.7.1 | CUDA 11.8
- **Diffusers** (Hugging Face) — Stable Diffusion + ControlNet pipeline management
- **Transformers** — model loading and tokenization
- **open_clip** — CLIP ViT-B/32 for evaluation
- **spaCy** — NLP parsing (en_core_web_sm)
- **NetworkX** — scene graph construction
- **MiDaS** — monocular depth estimation
- **reportlab** — PDF report generation
- **Pillow / matplotlib / scipy / numpy** — image processing and visualization

---

## License

This project is for academic/research purposes. The pretrained models (Stable Diffusion, ControlNet, MiDaS) are subject to their respective licenses. Datasets (COCO, Visual Genome, CLEVR) are subject to their original terms.
