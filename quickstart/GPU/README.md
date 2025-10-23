### Running on GPU

This guide shows a minimal, reproducible workflow to run MaxText training with a container and a small compatibility patch.

#### 1) Pull the container
```bash
# Optional: put cache on a fast, large filesystem
export APPTAINER_CACHEDIR="$(pwd)/.apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR"

# Pull the NVIDIA MaxText container once (modify tag as needed)
apptainer pull --force quickstart/maxtext.sif docker://nvcr.io/nvidia/maxtext:24.09
```

#### 2) Apply the MaxText compatibility patch
```bash
patch -p0 < quickstart/GPU/multi_gpu.patch
```
If you only have one GPU to run the script, apply
```bash
patch -p0 < quickstart/GPU/single_gpu.patch
```

#### 3) Quick smoke test (synthetic data)
```bash
apptainer exec --nv quickstart/maxtext.sif \
  python3 -u -m MaxText.train MaxText/configs/base.yml \
  run_name=gpu_smoke \
  base_output_directory=./ckpts/gpu_smoke \
  model_name=llama3.1-1b \
  dataset_type=synthetic \
  steps=20 \
  per_device_batch_size=1 \
  use_wandb=False \
  gcs_metrics=False
```

#### 4) Example: local ArrayRecord data
```bash
DATA_GLOB="/path/to/data/*.array_record"

apptainer exec --nv quickstart/maxtext.sif \
  python3 -u -m MaxText.train MaxText/configs/base.yml \
  run_name=gpu_local_arrayrecord \
  base_output_directory=./ckpts/local_run \
  model_name=llama3.1-1b \
  dataset_type=grain \
  grain_train_files="${DATA_GLOB}" \
  grain_file_type=arrayrecord \
  grain_worker_count=1 \
  tokenize_train_data=False \
  tokenize_eval_data=False \
  packing=False \
  max_target_length=8192 \
  steps=1000 \
  per_device_batch_size=1 \
  learning_rate=3e-4 \
  warmup_steps_fraction=0.05 \
  checkpoint_period=250 \
  checkpoint_max_to_keep=3 \
  use_wandb=False \
  gcs_metrics=False
```

Notes for ArrayRecord:
- Use `tokenize_train_data=False` when your records already contain token IDs.
- Set `packing=False` unless your records are pre-packed with positions/segmentations.

#### 5) Example: knowledge distillation
```bash
TEACHER_ITEMS="/path/to/teacher/items"  # e.g., quickstart/teacher/0/items

apptainer exec --nv quickstart/maxtext.sif \
  python3 -u -m MaxText.train MaxText/configs/base.yml \
  run_name=gpu_distill \
  base_output_directory=./ckpts/distill \
  model_name=llama3.1-1b \
  dataset_type=synthetic \
  steps=200 \
  per_device_batch_size=1 \
  use_kd=True \
  kd_alpha=0.5 \
  kd_temperature=2.0 \
  kd_teacher_parameters_path="${TEACHER_ITEMS}" \
  use_wandb=False \
  gcs_metrics=False
```

That’s it. The commands above are self-contained and typically sufficient to validate training and distillation on a GPU node with Apptainer.