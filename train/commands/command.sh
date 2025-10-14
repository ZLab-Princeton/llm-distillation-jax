export BUCKET_NAME=taiming_us_central2_b
export TPU_PREFIX=taiming-v4-128_000079
gcloud config set project vision-mix
gcloud config set compute/zone us-central2-b

export TPU_NAME=taiming-v4-128_000079
export ZONE=us-central2-b
export SSH_KEY_FILE=~/.ssh/google_rsa

# SSH into TPU VM and run commands on all workers
# First, let's check if the directory exists and then remove it with verbose output
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --ssh-key-file=${SSH_KEY_FILE} \
    --worker=all \
    --command "ls -la ~ && echo 'Removing maxtext...' && rm -rf ~/maxtext && echo 'Done. Checking if removed:' && ls -la ~"