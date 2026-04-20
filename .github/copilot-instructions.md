# MUSE System Copilot Instructions

## 1. Code Writing & Testing
- ALWAYS run and debug the code in the terminal to verify logic before presenting it.
- Model training code MUST run stably for at least 1 epoch. Test code MUST run without errors.

## 2. Model Structure Modifications & Logging
- Whenever modifying a model's structure, you MUST record the current architecture, attempted innovations, and final results.
- Location: Save logs in markdown/model_log.md within the respective module's workspace (create it if it doesn't exist).
- Workflow: Before applying a new modification, update the performance evaluation of the PREVIOUS model in the log. Then, write out the new model structure. Assign a clear version number (e.g., VER3.1).

## 3. Git Commits & Version Control (Monorepo)
- Commits: Use scoped semantic commit messages, e.g., eat(instrument_recognition): update classifier structure.
- Staging: Use git add <workspace_name>/ for module-specific changes.
- Branching Strategy: Do NOT push experimental models directly to main. Create model-specific version branches (e.g., git checkout -b ir/ver3.1). Push your changes to these specific branches. Merge into main only when a version is complete/stable.

## 4. General Requirements for Model Training
- Progress Visualization: Include a visual progress bar (e.g., tqdm) during training.
- Time Efficiency: During design/debugging, ensure each epoch takes <= 5 minutes (adjust batch size/dataloaders).
- Persistent Logging: Automatically generate timestamped log files (e.g., 20260420-153022.log) in the module's model/log/ directory. Record metrics (Loss, Accuracy, SDR, etc.) per epoch.
- Checkpointing: Code MUST support resuming training natively (periodically saving and loading checkpoints that include `epoch` and `optimizer_state`). Additionally, the model with the best performance must be saved as `best_model.pth` and uploaded.
