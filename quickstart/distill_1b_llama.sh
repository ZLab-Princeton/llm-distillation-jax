##

export YOUR_KD_ALPHA=YOUR_KD_ALPHA
export YOUR_KD_TEMPERATURE=YOUR_KD_TEMPERATURE
export YOUR_KD_TEACHER_PARAMETERS_PATH=YOUR_KD_TEACHER_PARAMETERS_PATH
export YOUR_BASE_OUTPUT_DIRECTORY=YOUR_BASE_OUTPUT_DIRECTORY
export YOUR_DATA_FILES=YOUR_DATA_FILES
export YOUR_RUN_NAME=YOUR_RUN_NAME
export YOUR_RUN_ID=YOUR_RUN_ID
export WANDB_PROJECT=YOUR_WANDB_PROJECT

required_vars=(
    "YOUR_KD_ALPHA"
    "YOUR_KD_TEMPERATURE"
    "YOUR_KD_TEACHER_PARAMETERS_PATH"
    "YOUR_BASE_OUTPUT_DIRECTORY"
    "YOUR_DATA_FILES"
    "YOUR_RUN_NAME"
    "YOUR_RUN_ID"
    "WANDB_PROJECT"
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

export USE_KD=true
export KD_ALPHA=YOUR_KD_ALPHA
export KD_TEMPERATURE=YOUR_KD_TEMPERATURE
export KD_TEACHER_PARAMETERS_PATH=YOUR_KD_TEACHER_PARAMETERS_PATH
export BASE_OUTPUT_DIRECTORY=YOUR_BASE_OUTPUT_DIRECTORY
export DATA_FILES=YOUR_DATA_FILES
export RUN_NAME=YOUR_RUN_NAME
export RUN_ID=YOUR_RUN_ID

echo "========================"
echo "running distill_1b_llama.sh"
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
echo "RUN_ID: $RUN_ID"
echo "start time: $(date)"
echo "========================"

python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND="
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
        wandb_project=${WANDB_PROJECT} \
        wandb_run_name=${RUN_NAME} \
        wandb_run_id=${RUN_ID} \
        packing=true \
        enable_data_shuffling=true \
        data_shuffle_seed=43 \
        init_weights_seed=43 \
        wandb_resume=relog \
        wandb_relog_source=auto  \
        use_kd=${USE_KD} \
        kd_alpha=${KD_ALPHA} \
        kd_temperature=${KD_TEMPERATURE} \
        kd_teacher_parameters_path=${KD_TEACHER_PARAMETERS_PATH}
    "
