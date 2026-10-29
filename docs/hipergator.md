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

```bash
ssh amcgill@hpg.rc.ufl.edu          # or ood.rc.ufl.edu > Clusters > Shell
mkdir -p /blue/gly4930/$USER/gstatsim-gpu/data
cd ~ && git clone <repo-url> gstatsim-gpu   # needs a remote (e.g. a private GitHub repo);
                                           # otherwise push a git bundle: see below
cd gstatsim-gpu
module load conda
conda env create -f environment.yml     # conda, not bare pip (pip --user installs collide)
```

Without a remote, bundle the repo on the laptop and clone the bundle (keeps LF
line endings on the scripts, unlike copying the Windows working folder):

```bash
git bundle create gsgpu.bundle main          # on the laptop
scp gsgpu.bundle amcgill@hpg.rc.ufl.edu:~
git clone ~/gsgpu.bundle ~/gstatsim-gpu      # on HiPerGator
```

Copy the Tutorial 3 data to blue, not home:
`scp data/greenland_test_data.csv amcgill@hpg.rc.ufl.edu:/blue/gly4930/amcgill/gstatsim-gpu/data/`

Login nodes have **no GPUs** and are for editing and job management only.
Run tests that need a GPU in a job or an OnDemand session with the `gpu`
partition and `gpu:1`.

## Running

```bash
mkdir -p logs
sbatch scripts/hpg/env_report.sbatch    # after setting ACCOUNT/QOS
squeue --me
jobnvtop <jobid>                         # watch GPU use live
```

Handy: `slurmInfo` (group resources), `blue_quota`, `home_quota`.
