# Quick Start: Knowledge Distillation for Llama3.1-1B

This guide walks you through knowledge distillation: data preparation → teacher training → student distillation.

This quickstart example uses the 350BT split of Fineweb-edu by Huggingface, and trained on the Llama3.2 1b model architecture.

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Set environment variables
export BUCKET_NAME="your-gcs-bucket-name"
export TPU_PREFIX="your-tpu-name"
export YOUR_BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts"
export YOUR_DATA_FILES="gs://$BUCKET_NAME/datasets/fineweb-edu/*.array_record"
export YOUR_RUN_NAME="your-run-name"
export YOUR_RUN_ID="your-run-id"
export WANDB_PROJECT="your-wandb-project-name"
```

## Step 1: Data Processing

```bash
source venv/bin/activate
python stream_fineweb_to_tokens.py
```

**What this does:** Downloads FineWeb-Edu dataset, tokenizes with Llama-3.1-8B tokenizer, chunks into 8192-token segments, writes ~10GB ArrayRecord files.

## Step 2: Train Teacher Model

```bash
source venv/bin/activate
export YOUR_BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/teacher"
export YOUR_DATA_FILES="gs://$BUCKET_NAME/datasets/fineweb-edu/*.array_record"
export YOUR_RUN_NAME="llama3.1-1b-teacher"
export YOUR_RUN_ID="teacher-run-001"

bash vanilla_1b_llama.sh
```

**Training:** Llama3.1-1B, 25K steps, 8K seq length, batch size 4, LR 3e-4, checkpoints every 250 steps.

## Step 3: Knowledge Distillation

```bash
source venv/bin/activate
export YOUR_KD_ALPHA="0.6"  # 0.0=no KD, 1.0=only KD
export YOUR_KD_TEMPERATURE="1.0"
export YOUR_KD_TEACHER_PARAMETERS_PATH="gs://$BUCKET_NAME/ckpts/teacher/llama3.1-1b-teacher/checkpoint_12500/0/items"
export YOUR_BASE_OUTPUT_DIRECTORY="gs://$BUCKET_NAME/ckpts/student"
export YOUR_DATA_FILES="gs://$BUCKET_NAME/datasets/fineweb-edu/*.array_record"
export YOUR_RUN_NAME="llama3.1-1b-student-distilled"
export YOUR_RUN_ID="student-distilled-001"

bash distill_1b_llama.sh
```

**Distillation:** Same model as teacher, KD enabled, teacher frozen, combined loss: `α * KD_loss + (1-α) * CrossEntropy_loss`.

## Key Parameters

- **`KD_ALPHA`**: Distillation weight (0.0=no KD, 0.5=balanced, 1.0=only KD)
- **`KD_TEMPERATURE`**: Softmax temperature for teacher outputs (1.0-5.0 typical)

## Monitoring

- **WandB**: Project set by `WANDB_PROJECT` variable, automatic logging
- **Checkpoints**: Every 250 steps, max 100 kept
- **Resume**: `wandb_resume=relog` for automatic resume

## Troubleshooting

- **TPU**: Check `TPU_PREFIX` is set
- **GCS**: Verify bucket permissions
- **Memory**: Reduce batch size if OOM
- **Checkpoints**: Verify teacher path format