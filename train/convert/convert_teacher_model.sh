#!/bin/bash
set -euo pipefail

# Multi-checkpoint KD-aware conversion for all distillation variations.
# Run this on TPU host 0. This avoids multihost runner and forces the job to use your updated source.

# 1) Activate environment
source "$HOME/maxtext_env/bin/activate"

# 2) Ensure imports use your repo (not a staged snapshot)
export PYTHONPATH="$HOME/maxtext:${PYTHONPATH:-}"

# 3) Configure backend (CPU avoids distributed init)
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_force_host_platform_device_count=1

# 4) User settings
BASE_BUCKET=${BASE_BUCKET:-gs://taiming_us_central2_b}
MODEL_NAME=${MODEL_NAME:-llama3.1-1b}

# 5) Sanity check which converter module is being used
python3.10 -c "import MaxText.generate_param_only_checkpoint as m; print('USING:', m.__file__)"

# Loop settings
SEEDS=(42 43)
CHECKPOINTS=(2500 12500 24999)

for SEED in "${SEEDS[@]}"; do
  for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    LOAD_PATH="$BASE_BUCKET/pretrain/maxtext/llama3.1-1b_finewebedu_pretrain_shuffled_lr_3e-4_seed_${SEED}/checkpoints/${CHECKPOINT}/items"
    OUT_DIR="$BASE_BUCKET/ckpts/pretrain_param_only/llama3.1-1b_finewebedu_pretrain_shuffled_lr_3e-4_seed_${SEED}/checkpoint_${CHECKPOINT}"

    echo "========================"
    echo "Seed: ${SEED}"
    echo "Converting checkpoint ${CHECKPOINT}"
    echo "Model: ${MODEL_NAME}"
    echo "From:  ${LOAD_PATH}"
    echo "To:    ${OUT_DIR}"
    echo "Backend: ${JAX_PLATFORMS}"
    echo "========================"

    # Run converter by absolute path (KD-aware)
    python3.10 -u "/home/terry/gcs-bucket/maxtext/MaxText/generate_param_only_checkpoint.py" /home/terry/maxtext/MaxText/configs/base.yml \
      skip_jax_distributed_system=True enable_single_controller=True hardware=cpu \
      load_full_state_path="$LOAD_PATH" \
      checkpoint_dir="$OUT_DIR" \
      enable_checkpointing=True async_checkpointing=False \
      model_name="$MODEL_NAME"

    echo "Done. Output: ${OUT_DIR}/0/items"
    echo ""
  done
done
