#!/bin/bash
# Phase 0 environment report on a HiPerGator L4 GPU node.
# Set ACCOUNT and QOS to the course group (the name under /blue/, e.g. /blue/<course>).
# Submit from the repo root:  mkdir -p logs && sbatch scripts/hpg/env_report.sbatch
# (SLURM won't create logs/ itself; without it the job's output is lost.)
#SBATCH --job-name=gsgpu_env
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
# Courses may use ONLY L4 GPUs. The burst QOS (<group>-b) cannot request GPUs.
# Check that the partition holds L4s with `slurmInfo`.
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:15:00
#SBATCH --output=logs/env_%j.log

set -euo pipefail

# Job output goes to blue storage; home is for code only (40 GB, not for job I/O).
OUT=/blue/${SLURM_JOB_ACCOUNT}/${USER}/gstatsim-gpu/results/env
mkdir -p "$OUT"

module purge
module load conda
conda activate gstatsim-gpu

nvidia-smi
python scripts/env_report.py --out "$OUT" --label "hpg-l4"
