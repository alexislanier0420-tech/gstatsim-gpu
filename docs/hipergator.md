# Running on HiPerGator (course allocation)

Based on the UFIT-RC course orientation. Support questions go to the **course TA**
first, not to RC directly.

## What the course gets

| Resource | Limit | What it means here |
|---|---|---|
| CPU | 32 cores, 256 GB RAM (whole course) | CPU baselines: request ≤ 8 cores so classmates can run too |
| GPU | "as needed", **L4 only** (24 GB) | All GPU results are on L4s. No B200/A100 numbers |
| Blue storage | 2 TB per course at `/blue/<course>/<gatorlink>` | Data, results, logs |
| Home | 40 GB at `/home/<gatorlink>` | Code and conda envs only, not job input/output |
| QOS | `<course>` (investment) or `<course>-b` (burst) | Burst can't request GPUs; use it only for CPU runs |

**Nothing is backed up, and course accounts are deleted at the end of the
semester.** Copy every benchmark record the report cites back to the laptop
(and commit the reviewed ones to `docs/`) before the Nov 30 deadline.

## One-time setup

```bash
ssh <gatorlink>@hpg.rc.ufl.edu          # or ood.rc.ufl.edu > Clusters > Shell
mkdir -p /blue/<course>/$USER/gstatsim-gpu/data
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
scp gsgpu.bundle <gatorlink>@hpg.rc.ufl.edu:~
git clone ~/gsgpu.bundle ~/gstatsim-gpu      # on HiPerGator
```

Copy the Tutorial 3 data to blue, not home:
`scp data/greenland_test_data.csv <gatorlink>@hpg.rc.ufl.edu:/blue/<course>/<gatorlink>/gstatsim-gpu/data/`

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
