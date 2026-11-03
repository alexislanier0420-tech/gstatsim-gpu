#!/bin/bash
# Poster speed benchmarks on a HiPerGator L4: GStatSim and our NumPy version on the node's CPU vs the
# GPU, same workloads as figures/poster/compute_speed.py on the laptop.
# Course group gly4930 (account = QOS = course group).
# Submit from the repo root (cloned on /blue, see docs/hipergator.md):  mkdir -p logs && sbatch scripts/hpg/benchmarks.sbatch
#SBATCH --job-name=gsgpu_bench
#SBATCH --account=gly4930
#SBATCH --qos=gly4930
# Courses may use ONLY L4 GPUs; hpg-turin is the L4 partition.
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
#SBATCH --output=logs/bench_%j.log

set -euo pipefail

# Job output goes to blue storage (home is 40 GB and not for job I/O).
OUT=/blue/${SLURM_JOB_ACCOUNT}/${USER}/gstatsim-gpu/results/hpg
mkdir -p "$OUT"

if [ ! -f data/greenland_test_data.csv ]; then
    echo "data/greenland_test_data.csv is missing: download it on the login node first (docs/hipergator.md)" >&2
    exit 1
fi

module purge
module load conda
conda activate gstatsim-gpu
# NumPy/BLAS threads = the cores Slurm gave this job (the node has 96; more threads than cores skews timings)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nvidia-smi
lscpu | grep -E "Model name|^CPU\(s\)"
echo "Slurm CPUs for this job: ${SLURM_CPUS_PER_TASK}"

python scripts/env_report.py --out "$OUT" --label "hpg-l4"
# Job ID in the name so a rerun never overwrites a record already cited (docs/hpg/speed_hpg_l4.json)
python figures/poster/compute_speed.py --out "$OUT/speed_hpg_l4_${SLURM_JOB_ID}.json"
