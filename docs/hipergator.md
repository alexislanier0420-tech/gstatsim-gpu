# Running on HiPerGator (course allocation)

Based on the UFIT-RC course orientation. Support questions go to the **course TA**
first, not to RC directly.

## What the course gets

| Resource | Limit | What it means here |
|---|---|---|
| CPU | 32 cores, 250 GB RAM (whole course), 744 h job time limit | CPU baselines: request ≤ 8 cores so classmates can run too |
| GPU | **2 L4s for the whole course** (23 GB each), partition `hpg-turin` | All GPU results are on L4s. Develop on the laptop GPU; use HPG for final runs, and expect queueing near deadlines |
| Blue storage | 2 TB per course at `/blue/gly4930/amcgill` | Data, results, logs |
| Home | 40 GB at `/home/amcgill` | Code and conda envs only, not job input/output |
| QOS | `gly4930` (investment) or `gly4930-b` (burst) | Burst can't request GPUs; use it only for CPU runs |

**Nothing is backed up, and course accounts are deleted at the end of the
semester.** Copy every benchmark record the report cites back to the laptop
(and commit the reviewed ones to `docs/`) before the Nov 30 deadline.

## One-time setup

The repo goes on blue, so code, data, logs and results sit together; the conda
env stays in home (named envs live in `~/.conda`).

```bash
ssh amcgill@hpg.rc.ufl.edu          # or ood.rc.ufl.edu > Clusters > Shell
cd /blue/gly4930/$USER
git clone https://github.com/alexislanier0420-tech/gstatsim-gpu.git
#   while the repo is private, git asks for a username and a password:
#   give a GitHub personal access token (read-only, this repo) as the password
cd gstatsim-gpu
module load conda
conda env create -f environment.yml     # conda, not bare pip (pip --user installs collide)
mkdir -p data logs
wget -O data/greenland_test_data.csv https://raw.githubusercontent.com/GatorGlaciology/GStatSim/main/demos/data/greenland_test_data.csv
```

The Rubble Glacier inputs (`figures/data/*.npz`) come with the clone.

Login nodes have **no GPUs** and are for editing and job management only.
Run tests that need a GPU in a job or an OnDemand session with the `gpu`
partition and `gpu:1`.

## Running

```bash
cd /blue/gly4930/$USER/gstatsim-gpu
git pull                                 # pick up laptop changes
sbatch scripts/hpg/env_report.sbatch     # ~2 min
sbatch scripts/hpg/benchmarks.sbatch     # poster speed numbers, ~30-60 min
squeue --me
jobnvtop <jobid>                         # watch GPU use live
```

Handy: `slurmInfo` (group resources), `blue_quota`, `home_quota`.

Results land in `results/hpg/` (`speed_hpg_l4.json` plus the environment
report). Copy them back to the laptop the same day:

```bash
scp -r amcgill@hpg.rc.ufl.edu:/blue/gly4930/amcgill/gstatsim-gpu/results/hpg results/   # on the laptop
```
