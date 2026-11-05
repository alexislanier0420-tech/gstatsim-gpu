#!/bin/bash
# Large-region benchmark, GPU part, on a HiPerGator L4 (benchmarks/bench_region.py).
# Submit from the repo root, with the resolution in metres (default 1000):
#   mkdir -p logs && sbatch scripts/hpg/region_gpu.sbatch 1000
#   mkdir -p logs && sbatch --time=04:00:00 scripts/hpg/region_gpu.sbatch 500
# The course shares 2 L4s: keep --time close to what the run needs.
#SBATCH --job-name=gsgpu_region_gpu
# Another course? Change gly4930 to your course group (id -gn) on these two lines.
#SBATCH --account=gly4930
#SBATCH --qos=gly4930
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
#SBATCH --output=logs/region_gpu_%j.log

set -euo pipefail
RES=${1:-1000}
OUT=/blue/${SLURM_JOB_ACCOUNT}/${USER}/gstatsim-gpu/results/hpg
mkdir -p "$OUT"

module purge
module load conda
conda activate gstatsim-gpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nvidia-smi
# At 500 m one realization alone takes about as long as 32 together, so skip it there.
if [ "$RES" -ge 1000 ]; then BATCHES="1 32 64"; else BATCHES="32"; fi
python benchmarks/bench_region.py --part gpu --res "$RES" --batches $BATCHES \
    --out "$OUT/region_ase_${RES}m_gpu_${SLURM_JOB_ID}.json"
