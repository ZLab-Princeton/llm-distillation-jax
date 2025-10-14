#!/bin/bash

cd ~/maxtext

# export TPU_PREFIX=llm-pruning-v6e
# export BUCKET_NAME=taiming_us_central2_b
# export TPU_PREFIX=taiming-v4

echo "========================"
echo "environment variables:"
echo "TPU_PREFIX: $TPU_PREFIX"
echo "BUCKET_NAME: $BUCKET_NAME"
echo "========================" 

required_vars=(
    "BUCKET_NAME"
    "TPU_PREFIX"
)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ERROR] $var is not set"
    exit 1
  fi
done

export MODEL_NAME='llama3.1-1b'
export NUM_STEPS=25000
export SEQ_LEN=8192
export BATCH_SIZE=4
export GRAD_ACCUM=1
export LR=3.e-4
export MIN_LR_RATIO=0.1
export WARMUP_RATIO=0.05
export ASYNC_CHECKPOINTING=false
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/pretrain/maxtext"
export DATA_FILES='/home/terry/gcs-bucket/datasets/fineweb-edu/*.array_record'

export RUN_NAME="${MODEL_NAME}_finewebedu_pretrain_shuffled_lr_3e-4_seed_43"
export RUN_ID="llama3-1b_finewebedu_pretrain_shuffled_lr_3e-4_seed_43"

echo "========================"
echo "running llama3_8b_1b_3e-4.sh"
echo "parameters:"
echo "MODEL_NAME: $MODEL_NAME"
echo "SEQ_LEN: $SEQ_LEN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "LR: $LR"
echo "MIN_LR_RATIO: $MIN_LR_RATIO"
echo "WARMUP_RATIO: $WARMUP_RATIO"
echo "ASYNC_CHECKPOINTING: $ASYNC_CHECKPOINTING"
echo "BASE_OUTPUT_DIRECTORY: $BASE_OUTPUT_DIRECTORY"
echo "DATA_FILES: $DATA_FILES"
echo "RUN_NAME: $RUN_NAME"
echo "TPU_PREFIX: $TPU_PREFIX"
echo "BUCKET_NAME: $BUCKET_NAME"
echo "========================"

wandb login --relogin 01126ae90da25bae0d86704140ac978cb9fd9c73

python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/maxtext_env/bin/activate
    export WANDB_API_KEY='01126ae90da25bae0d86704140ac978cb9fd9c73'
    export WANDB_PROJECT=maxtext_1b
    export WANDB_NAME=${RUN_NAME}
    python3.10 -u -m MaxText.train MaxText/configs/base.yml \
        run_name=${RUN_NAME} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY} \
        dataset_type=grain \
        grain_train_files=${DATA_FILES} \
        grain_file_type='arrayrecord' \
        grain_worker_count=1 \
        tokenize_train_data=False \
        tokenize_eval_data=False \
        max_target_length=${SEQ_LEN} \
        async_checkpointing=${ASYNC_CHECKPOINTING} \
        original_max_position_embeddings=${SEQ_LEN} \
        model_name=${MODEL_NAME} \
        steps=${NUM_STEPS} \
        per_device_batch_size=${BATCH_SIZE} \
        gradient_accumulation_steps=${GRAD_ACCUM} \
        learning_rate=${LR} \
        cosine_learning_rate_final_fraction=${MIN_LR_RATIO} \
        warmup_steps_fraction=${WARMUP_RATIO} \
        checkpoint_period=250 \
        checkpoint_max_to_keep=100 \
        gcs_metrics=True \
        use_wandb=True \
        wandb_project=maxtext_1b \
        wandb_run_name=${RUN_NAME} \
        wandb_run_id=${RUN_ID} \
        packing=true \
        enable_data_shuffling=true \
        data_shuffle_seed=43 \
        init_weights_seed=43 \
        wandb_resume=relog \
        wandb_relog_source=auto \
    "
