# MUSE Project Rules for Claude Code

## Project Overview
MUSE (Music Unmixing & Score Extraction) is a monorepo at `d:\program_project\MUSE` with 4 workspaces:
- `instrument_recognition/` — **Active.** PyTorch multi-label instrument classifier (10 classes)
- `source_separation/` — Future module (empty skeleton)
- `score_extraction/` — Future module (empty skeleton)
- `frontend/` — Future module (empty)
- `data/` — Shared datasets (MedleyDB, IRMAS, nsynth, moisesdb). NOT in git.
- `music/` — Local test audio. NOT in git.

Current active branch: `ir/ver3.0`. Main branch (`main`) is reserved for stable/complete versions.

## Code Discipline
1. After every code change, run the modified script to verify it executes without errors.
2. Training code must be capable of running at least 1 full epoch stably before being presented.
3. Inference/test code must complete without runtime errors — verify with `python test/infer.py`.
4. Use `python -c "<one-liner>"` for quick sanity checks; do not assume code works.
5. Chinese comments in code are acceptable and expected — the maintainer prefers them for nuanced design notes.

## Model Logging Convention
1. Before modifying model architecture, log the current version's final performance in `markdown/model_log.md`.
2. After logging the old version, write the new architecture design, innovation points, and hyperparameters.
3. Version naming: `VER<MAJOR>.<MINOR>_<FeatureTag>` (e.g., `VER2.3_StrictMetric`, `VER3.0_TransformerFocal`).
4. Each entry must record: architecture description, attempted innovations, and final evaluation metrics.
5. Commit the log after a version is complete and evaluated, before starting the next version.

## Git Workflow (Monorepo)
1. **Branch naming**: `ir/ver<MAJOR>.<MINOR>` for instrument_recognition changes.
2. **Commit format**: `feat|fix|docs|refactor|perf(instrument_recognition): <description>`. Use English.
3. **Staging**: stage specific files with `git add instrument_recognition/<path>`. Never use `git add -A` or `git add .`.
4. **Never push experimental models to main**. Merge to main only when a version is complete and stable.
5. Create a new version branch for each model iteration: `git checkout -b ir/verX.X` from the current branch.
6. At session start, run `git -C d:/program_project/MUSE log --oneline -5` to understand current state.

## Training Requirements
1. **Progress bar**: tqdm is mandatory in all training loops.
2. **Epoch time**: Each epoch must complete within 5 minutes. If it exceeds, optimize (adjust batch size, num_workers, or model size).
3. **Timestamped logs**: Write per-epoch metrics to `model/log/YYYYMMDD-HHMMSS.log`. Format: `Epoch\tTrain_Loss\tTrain_F1\tVal_Loss\tVal_F1`.
4. **Checkpointing**:
   - Save `model/checkpoint_latest.pth` every epoch (contains: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_acc).
   - Save `model/best_model.pth` when validation F1 improves (contains: version, model_state_dict, scheduler_state_dict).
   - Training must support automatic resume from `checkpoint_latest.pth`.
5. **Version guard**: If checkpoint version != current MODEL_VERSION, restart training from scratch (to avoid cross-version weight corruption).

## Environment
- **Python**: Conda environment at `instrument_recognition/env/` (Python 3.10)
- **GPU**: NVIDIA RTX 4060 Laptop 8GB VRAM
- **PyTorch**: CUDA 11.8
- **Working directory**: `d:\program_project\MUSE\instrument_recognition`

---

## Architecture Reference (Current: VER4.0_BinaryEnsemble)

### Model: 10× BinaryInstrumentClassifier (Ensemble)
- **Architecture**: Conv2d(1→16) → BN → ReLU → ResidualBlock(16→32, stride=2) → ResidualBlock(32→64, stride=2) → AdaptiveAvgPool2d → Dropout(0.3) → Linear(64, 1)
- **Params per model**: ~72K (10 models = 0.73M total — 76% smaller than VER3.5's 3.06M)
- **Storage**: 2.9 MB total (92% smaller than 37 MB checkpoint)
- **Training**: BCEWithLogitsLoss, Adam(lr=1e-3), 30 epochs, 1:1 balanced pos/neg sampling
- **Inference**: All 10 models run per window; sigmoid output thresholded independently

### Training Strategy (3-stage)
1. **Clean stem pre-training**: 4185 extracts from MedleyDB (120-720 per class)
2. **Per-instrument validation**: All 10 models >0.94 Val F1 on clean stems
3. **Ensemble inference**: 10 models → merged multi-label output, per-class threshold

### Files
| File | Purpose |
|------|---------|
| `src/binary_model.py` | BinaryInstrumentClassifier definition (~72K params) |
| `src/binary_train.py` | Training script (single instrument) |
| `src/train_all_binary.py` | Batch train all 10 instruments |
| `test/binary_infer.py` | Ensemble inference (load all 10 models) |
| `test/binary_eval_gt.py` | Ground truth evaluation |
| `data/build_clean_stems.py` | Clean stem extraction from MedleyDB |
| `model/binary/*.pth` | 10 trained model checkpoints |

### 10 Classes
acoustic guitar, cello, drum set, electric bass, electric guitar, flute, piano, singer, synthesizer, violin

### Ground Truth Results (vs VER3.5)
| Song | VER3.5 | VER4.0 | Δ |
|------|--------|--------|---|
| Global Micro F1 | 0.527 | **~0.68** | **+29%** |
