# Project tracker

Class final project. Internal POC target **Nov 20**, class deadline **Nov 30**.
Budget 5–10 h/week. Plan: *Gator Glaciology GPU System Design RFC* and *Implementation Timeline*.

## Phase gates

### Phase 0: Scope & environment (Oct 29 – Nov 1) · Gate 1 Nov 1
- [x] Repo created with CPU/GPU dispatcher, validation, tests (37 passing), benchmark harness
- [x] Isolated dev env (`environment.yml`); course env untouched
- [x] Laptop environment report: RTX 5060 Laptop 8 GiB, CC 12.0, driver 592.01, CUDA runtime 12.9, CuPy 14.2; GPU smoke test passed
- [x] Tutorial 3 runs through the harness (500-point smoke run, laptop)
- [x] Prior work reviewed: [docs/existing_port_review.md](docs/existing_port_review.md)
- [ ] Confirm deliverable expectations with instructor (see open questions)
- [ ] HiPerGator: set account/QOS (course group) in `scripts/hpg/env_report.sbatch`, build env, run report on an L4 ([docs/hipergator.md](docs/hipergator.md))
- [ ] Pick a small, repeatable Tutorial 4 (SGS) case

### Phase 1: CPU baseline (Nov 2 – 8) · Gate 2 Nov 8
- [ ] Full Tutorial 3 baseline, SK and OK, 3 repeats (laptop + HiPerGator CPU)
- [ ] Profile by component (early result: neighbor search ~79%; see review doc)
- [ ] Tutorial 4 SGS baseline at a manageable size
- [ ] **Write tolerances before Phase 2** (see Decisions)
- [ ] Decide whether to also benchmark an optimized CPU path (KD-tree)

### Phase 2: GPU prototype (Nov 9 – 15) · Gate 3 Nov 15
- [ ] Batched GPU kriging with GStatSim octant search, returning estimate + variance
- [ ] CPU-vs-GPU tests on fixtures incl. spherical model, nugget > 0, points on data
- [ ] Runs on laptop GPU (RTX 5060) and a HiPerGator L4

### Phase 3: Validate & deliver (Nov 16 – 20)
- [ ] Benchmarks across sizes, incl. cases where GPU is slower; memory + transfer costs
- [ ] Report, setup guide, demo notebook, expansion roadmap
- [ ] Copy all cited results off HiPerGator (no backups; course accounts deleted at semester end)
- [ ] Buffer Nov 21 – 30 for the class deadline

## Open questions
| Question | For | Status |
|---|---|---|
| Is the deliverable a report + code, or must it be a PR to GStatSim? | Instructor | open |
| Does the project have to use custom CUDA, or is CuPy acceptable? | Instructor | open |
| Which HiPerGator account/QOS and GPU partition can the class use? | RC orientation | Account = QOS = course group; L4 GPUs only. Still need the group name (TA) and to confirm the L4 partition (`slurmInfo`) |
| May we reuse code from badger-lord/gstatsim_gpu (no license)? Contact author or reimplement | Author | open; plan to reimplement |
| Should the Rubble Glacier data also be benchmarked, or only the tutorial data? | Instructor | open |
| Is SGS in scope, or only a feasibility discussion? | Instructor | open |

## Decisions
| Date | Decision | Why |
|---|---|---|
| 10-29 | Code lives in `dev/gstatsim-gpu` (own repo), not inside HardTune | Keep histories separate |
| 10-29 | CPU backend = GStatSim called unchanged | Trusted reference; outputs must match exactly (tested) |
| 10-29 | GPU request never silently falls back to CPU | RFC §9: fallback would invalidate comparisons |
| 10-29 | CuPy first; custom CUDA only for a measured bottleneck | RFC §10 |
| 10-29 | GStatSim's octant search is the target semantics for the GPU path | Prior port uses plain K-nearest and doesn't match |
| 10-29 | GPU target hardware: HiPerGator L4 (24 GB) + laptop RTX 5060 (8 GB) | Courses may use only L4s |
| 10-29 | HPG job I/O in `/blue/<course>/<user>`, code + env in home | RC storage policy |
| — | Kriging tolerance (normal-score units): est ___, var ___ | Set in Phase 1, before GPU results exist |

## Benchmark log
| Date | Machine | Workload | Backend | Points | Median | Per point | Correctness | Record |
|---|---|---|---|---|---|---|---|---|
| 10-29 | Laptop (Intel, 24 thr) | Tut3 SK, k=100, r=50 km | cpu | 500 | 4.85 s | 9.7 ms | reference | smoke test, 1 repeat |

## Risks to watch
- Data: `glacier_lab1.nc` may still hold the Marie Byrd Land crop; check extent before using lab outputs.
- Laptop GPU has 8 GiB, the L4 24 GB; size batches for the laptop first.
- Course shares 32 cores / 256 GB: keep CPU baseline jobs small (≤ 8 cores) and short.
- HiPerGator storage isn't backed up and course accounts are deleted after the semester.
- A GPU speedup over pandas-bound GStatSim may mostly measure Python overhead; report against an optimized CPU path too.
