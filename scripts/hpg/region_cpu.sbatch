#!/bin/bash
# Large-region benchmark, CPU part: our NumPy version and GStatSim itself, no GPU
# (benchmarks/bench_region.py). Runs on the burst QOS so it doesn't hold one of
# the course's GPUs. GStatSim is slow here on purpose: that's the comparison.
# Submit from the repo root, with the resolution in metres (default 1000):
#   mkdir -p logs && sbatch scripts/hpg/region_cpu.sbatch 1000
#   mkdir -p logs && sbatch --time=4-00:00:00 scripts/hpg/region_cpu.sbatch 500
# Extra arguments go to bench_region.py, e.g. only GStatSim's SGS:
#   sbatch --time=4-00:00:00 scripts/hpg/region_cpu.sbatch 500 --only gstatsim_sgs_x1
# The record is rewritten after each step, and GStatSim's progress bar shows its
# rate in the log, so a run that hits the time limit still gives usable numbers.
#SBATCH --job-name=gsgpu_region_cpu
# Another course? Change gly4930 to your course group (id -gn) on these two lines.
#SBATCH --account=gly4930
#SBATCH --qos=gly4930-b
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/region_cpu_%j.log

set -euo pipefail
RES=${1:-1000}
OUT=/blue/${SLURM_JOB_ACCOUNT%-b}/${USER}/gstatsim-gpu/results/hpg
mkdir -p "$OUT"

module purge
module load conda
conda activate gstatsim-gpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# GStatSim's progress bar: one log line a minute instead of several a second
export TQDM_MININTERVAL=60

lscpu | grep -E "Model name|^CPU\(s\)"
python benchmarks/bench_region.py --part cpu --res "$RES" "${@:2}" \
    --out "$OUT/region_ase_${RES}m_cpu_${SLURM_JOB_ID}.json"
