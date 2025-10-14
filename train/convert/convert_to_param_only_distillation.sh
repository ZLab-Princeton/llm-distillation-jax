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

# Define distillation variations
DISTILL_VARIATIONS=("P_1B_T_1B" "P_1B_T_5B" "P_1B_T_50B")

# Define checkpoints to process
CHECKPOINTS=(0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000)

# Loop through each distillation variation
for distill_var in "${DISTILL_VARIATIONS[@]}"; do
    echo "========================================"
    echo "Processing distillation variation: ${distill_var}"
    echo "========================================"
    
    # Loop through each checkpoint for this distillation variation
    for checkpoint in "${CHECKPOINTS[@]}"; do
        LOAD_PATH="$BASE_BUCKET/model_ckpts/maxtext/${MODEL_NAME}_distill_${distill_var}/checkpoints/${checkpoint}/items"
        OUT_DIR="$BASE_BUCKET/model_ckpts/maxtext/maxtext_param_only/${MODEL_NAME}-distill_${distill_var}/checkpoint_${checkpoint}"
        
        echo "========================"
        echo "Converting checkpoint ${checkpoint}"
        echo "Distillation: ${distill_var}"
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
    
    echo "========================================"
    echo "Completed distillation variation: ${distill_var}"
    echo "========================================"
    echo ""
done

echo "All distillation variations and checkpoints converted successfully!"
