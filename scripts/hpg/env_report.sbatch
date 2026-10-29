#!/bin/bash
# Phase 0 environment report on a HiPerGator GPU node.
# Fill in ACCOUNT/QOS with your course or lab allocation (ask the TA or run `showassoc $USER`).
# Submit from the repo root:  mkdir -p logs && sbatch scripts/hpg/env_report.sbatch
# (SLURM won't create logs/ itself; without it the job's output is lost.)
#SBATCH --job-name=gsgpu_env
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:15:00
#SBATCH --output=logs/env_%j.log

set -euo pipefail
module purge
# Load whichever Python + CUDA you'll actually benchmark with, then record it here.
module load conda
conda activate gstatsim-gpu

nvidia-smi
python scripts/env_report.py --out results/env --label "hpg-${SLURM_JOB_PARTITION:-gpu}"
