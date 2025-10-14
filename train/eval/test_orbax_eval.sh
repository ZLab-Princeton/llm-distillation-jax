#!/bin/bash

set +x
set -eo pipefail

export BUCKET_NAME=taiming_us_central2_b
export TPU_PREFIX=taiming-v4-128_000079
# gcloud config set project vision-mix
# gcloud config set compute/zone us-central2-b

# export MODEL='llama3.1-8b'
export MODEL='llama3.1-1b'
export BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/model_ckpts/maxtext/maxtext_param_only"
export MODEL_VARIATION="llama3.1-1b-vanilla"
export STEP="5000"
# Param-only checkpoints produced by MaxText live under "checkpoints/0/items"
export PARAMS_ONLY_CKPT_PATH="${BASE_OUTPUT_DIRECTORY}/${MODEL_VARIATION}/checkpoint_${STEP}/0/items"

echo "loading from PARAMS_ONLY_CKPT_PATH: $PARAMS_ONLY_CKPT_PATH"

export HF_MODEL_PATH='/home/terry/gcs-bucket/model_ckpts/HF_HOME/Llama-3.1-8B'
# export HF_MODEL_PATH='meta-llama/Llama-3.1-8B'

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_DISABLE_MOST_OPTIMIZATIONS=False
export JAX_PLATFORMS=tpu
export JAX_DISTRIBUTED_INIT=false
export JAX_COORDINATOR_IP=""
export JAX_COORDINATOR_PORT=""

cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH

source ~/maxtext_env/bin/activate

cd /home/terry/gcs-bucket/maxtext/
# Add both MaxText and lm-evaluation-harness to PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/lm-evaluation-harness:$PYTHONPATH
python3 -u lm-evaluation-harness/scripts/test_orbax_eval.py \
    MaxText/configs/base.yml \
    load_parameters_path=${PARAMS_ONLY_CKPT_PATH} \
    run_name=forward_pass_test \
    per_device_batch_size=1 \
    model_name=${MODEL} \
    max_prefill_predict_length=4 \
    max_target_length=8192 \
    dataset_type=synthetic \
    dtype=bfloat16 \
    scan_layers=false \
    attention="dot_product" \
    --hf_model_path=${HF_MODEL_PATH} \
    --save_path=/home/terry/gcs-bucket/model_evals/${MODEL_VARIATION}/checkpoint_${STEP} \
    --ppl_tasks= \
    --acc_tasks=mmlu
    # --acc_tasks=hellaswag,arc_challenge,winogrande,mmlu

    # skip_jax_distributed_system=true \
    # dcn_data_parallelism=1 \
    # dcn_fsdp_parallelism=1 \
    # num_slices=1 \
# decode example
# idx=0
# TOKENIZER='/home/terry/maxtext/assets/tokenizer_llama3.tiktoken'
# python3 -m MaxText.decode \
#     /home/terry/gcs-bucket/maxtext/MaxText/configs/base.yml \
#     load_parameters_path=${UNSCANNED_CKPT_PATH} \
#     tokenizer_type=tiktoken \
#     tokenizer_path=$TOKENIZER \
#     per_device_batch_size=1 \
#     run_name=runner_$(date +%Y-%m-%d-%H-%M) \
#     max_prefill_predict_length=4 \
#     max_target_length=16 \
#     model_name=$MODEL \
#     dataset_type=synthetic \
#     async_checkpointing=false \
#     scan_layers=false \
#     attention=dot_product \
#     prompt="I love to" 

# python3 -m MaxText.decode MaxText/configs/base.yml tokenizer_path=assets/tokenizer_llama3.tiktoken tokenizer_type=tiktoken load_parameters_path=${UNSCANNED_CKPT_PATH} per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=4 max_target_length=16 dataset_type=synthetic async_checkpointing=false scan_layers=false model_name=${MODEL_VARIATION} attention=dot_product prompt="I love to"


# gs://taiming_us_central2_b/model_ckpts/maxtext/maxtext_param_only/llama3.1-1b-vanilla/checkpoint_5000/checkpoints/0/items
# taiming_us_central2_b/model_ckpts/maxtext/maxtext_param_only/llama3.1-1b-vanilla/checkpoint_5000/0/items