#!/bin/bash

CONVERTED_CHECKPOINT='gs://taiming_us_central2_b/model_ckpts/maxtext/llama3.1-1b_seqlen_8192_bs_4_grad_accum_1_lr_2.e-4_min_lr_ratio_0.1_warmup_ratio_0.05_quadratic_warmup/checkpoints/49999/items'

JAX_PLATFORMS=cpu python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
    MaxText/configs/base.yml \
    base_output_directory=gs://taiming_us_central2_b/model_ckpts/maxtext \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=convert_to_hf \
    model_name=llama3.1-1b \
    hf_model_path=/home/terry/gcs-bucket/model_ckpts/minitron/llama3_1b
