<!-- 한국어: [KR.md](KR.md) -->

<div align="center">

<img src="assets/logo.svg" width="88" alt="Daedalus">

# Daedalus Engineering AI

**An autonomous engineering design agent.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![GPU: NVIDIA Warp](https://img.shields.io/badge/GPU-NVIDIA%20Warp-76b900.svg)](https://github.com/NVIDIA/warp)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-281%20passing-brightgreen.svg)](#status)
[![Status](https://img.shields.io/badge/status-phase%200--6%20complete-orange.svg)](#status)

</div>

Give it an engineering goal and it loops (reason, design, simulate on the GPU,
optimize, learn) to design a robot part, accumulating what it learns in an
**Engineering Brain** where every statement carries an explicit evidence level.

The agent and its CLI are branded **Daedalus**.

---

## Status

Phases 0-6 are implemented and verified.

| phase | what it does | state |
|---|---|---|
| 0 | venv, project skeleton, GPU profiles, Warp/torch sanity | done |
| 1 | Engineering IR (the problem), materials DB, Design Genome (the variables) | done |
| 2 | Differentiable GPU beam physics via NVIDIA Warp | done |
| 3 | Constrained mass optimization (SLSQP + differential evolution) | done |
| 4 | Autonomous design loop: state machine, episodes, budget, explore/exploit | done |
| 5 | Engineering Brain: episodic/semantic memory, evidence levels, retrieval | done |
| 6 | PyTorch surrogate + two-stage screen-and-verify | done |
| 7 | High-fidelity 3D FEM verification gate (Warp, matrix-free CG) | done |
| 8 | Material expansion (15 materials) and orthotropic elasticity | done |
| 9 | Parametric CAD B-rep and STEP export, mass-consistent | done |
| 10 | Assemblies and kinematics (FK, Jacobian, IK, statics, assembly STEP) | done |
| 11 | Rigid-body dynamics (inertia, M/C/G, load cases, torque and power) | done |
| 12 | Motor and gearbox selection (margins, reflected inertia, alternatives) | done |
| 13 | Topology optimization (SIMP on the GPU FEM, sensitivity-verified) | done |

**MVP problem:** minimize the mass of a single hollow-rectangular robot link,
cantilevered, carrying a 196.2 N tip load (a 20 kg payload), in aluminium
7075-T6, subject to a stress ceiling, a tip-deflection limit and a safety
factor.

**Result:** **1.686 kg → 0.250 kg, an 85.2% mass reduction.** Two independent
optimizers agree to 1.3×10⁻⁵ relative. The design is **deflection-limited**:
tip deflection sits exactly on its 1 mm cap while the stress constraint keeps
over 70% margin.

**281 tests pass**, including independent verification of every critical
calculation against a separately-derived reference.

---

## Architecture

<p align="center"><img src="assets/architecture.svg" alt="Daedalus architecture" width="840"></p>

A goal enters as an **Engineering IR** (the problem: geometry, material, load,
constraints, objectives, all fixed). The **Design Genome** holds only what a
search may change, currently the cross-section. Candidates run through the
engine (geometry, GPU physics, optimization) and down a **multi-fidelity
funnel**: a surrogate screens thousands, beam theory evaluates the shortlist,
and 3D FEM is the final gate. Results are judged against the constraints, and
every iteration is written to the **Engineering Brain** with an evidence level
attached, which is what the reasoner reads on the next pass.

The key separation is that physics never sees a genome without a problem
attached, so a design variable can never quietly become a requirement.

## Hardware & GPU profiles

Developed and verified on an **RTX 3050 Laptop GPU (4 GB)**. Scale is separated
from the code: every size-dependent setting lives in `configs/profiles/*.yaml`.

| profile | VRAM | character |
|---|---|---|
| `laptop_4gb` | 4 GB | current target; small candidate pools, AMP + gradient checkpointing |
| `desktop_16gb` | 16 GB | mid-range workstation |
| `rtx5090_32gb` | 32 GB | high bandwidth; fastest per-solve for this workload |
| `dgx_spark_128gb` | 128 GB | huge capacity, low bandwidth: big models, slow solves |
| `cloud_a100` | 80 GB | large-scale parallel |

Selection order: `--profile` → `ENG_PROFILE` env var → VRAM auto-detection →
fallback.

```bash
python -m interfaces.cli.main info                     # auto-detect
python -m interfaces.cli.main info --profile cloud_a100
ENG_PROFILE=desktop_16gb python -m interfaces.cli.main info
```

**No system CUDA toolkit is required**: NVIDIA Warp JITs its own kernels, and
torch ships its CUDA runtime in the wheel.

---

## System requirements

Grounded in what has actually been run and in the shipped profiles: not
aspirational.

### Software

| | requirement |
|---|---|
| OS | **Linux**: the development and verification platform. Windows/macOS: **TBD / experimental**, not yet validated. |
| Python | **3.10+** (verified on 3.10.12) |
| GPU driver | An NVIDIA driver new enough for the bundled torch/warp CUDA runtime. **A system CUDA toolkit is *not* required.** |
| Disk | **≈ 6 GB**: the venv is ~5 GB (torch + Warp CUDA wheels), plus room for `datasets/` and `runs/` |

### Hardware

| tier | GPU | CPU / RAM | profile |
|---|---|---|---|
| **Minimum** (MVP, development) | NVIDIA, 4 GB VRAM (e.g. RTX 3050) | 8 cores / 16 GB | `laptop_4gb` |
| **Recommended** | 16 GB+ VRAM (RTX 4070 Ti / 4080, used 3090 24 GB) | 8+ cores / 16-32 GB | `desktop_16gb`, `rtx5090_32gb` |
| **Large scale** | 24-48 GB+ (4090 / 5090 / A6000) or cloud A100 80 GB | 16+ cores / 64 GB+ | `cloud_a100` |
| **CPU only** | none, Warp has a CPU device |, | works, but **slow and limited**; a GPU is strongly recommended |

---

## Install (development)

```bash
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/python -m pip install -U pip wheel
env -u PYTHONPATH .venv/bin/pip install -r requirements.txt
env -u PYTHONPATH .venv/bin/python scripts/gpu_sanity.py
```

> **Run the venv with a clean `PYTHONPATH`.** A sourced shell environment can
> export `PYTHONPATH` and shadow the venv's packages with older ones: a
> different `numpy` silently winning, for example. Prefixing with
> `env -u PYTHONPATH` avoids it. The planned bootstrap installer (below) is
> intended to handle this automatically so end users never think about it.

`gpu_sanity.py` initializes Warp, compiles and runs a real kernel, checks the
result, verifies torch CUDA, and prints the resolved profile.

---

## Usage

All commands are `python -m interfaces.cli.main <command>` today; the packaged
CLI (below) will expose them as `dae <command>`.

### `evaluate`: one design, on the GPU

```bash
python -m interfaces.cli.main evaluate --width 50 --height 80 --thickness 5
```

```
              evaluated metrics (Euler-Bernoulli beam theory)
┃ quantity              ┃             SI ┃   readable ┃   limit ┃ verdict ┃
│ mass                  │       1.686 kg │  1.6860 kg │       - │    -    │
│ max bending stress    │ 3.96364e+06 Pa │  3.964 MPa │ 120 MPa │  PASS   │
│ tip deflection        │  0.000115168 m │  0.1152 mm │ 1.00 mm │  PASS   │
│ safety factor         │        126.904 │     126.90 │     2.0 │  PASS   │
│ 1st natural frequency │     324.761 Hz │   324.8 Hz │       - │    -    │
```

### `optimize`: minimize mass, cross-verified by two methods

```bash
python -m interfaces.cli.main optimize --method both
```

```
┃ quantity           ┃    baseline ┃       SLSQP ┃ DifferentialEvolution ┃
│ b (mm)             │      50.000 │      10.000 │                10.000 │
│ h (mm)             │      80.000 │      80.960 │                80.958 │
│ t (mm)             │       5.000 │       1.000 │                 1.000 │
│ mass (kg, SI)      │    1.686000 │    0.249977 │              0.249973 │
│ delta_tip (mm)     │     0.11517 │     1.00000 │               1.00004 │
│ mass reduction     │           - │       85.2% │                 85.2% │
│ active constraint  │        none │  deflection │            deflection │
cross-verification: |SLSQP - DE| / SLSQP = 1.341e-05 (0.0013%)  AGREE
```

### `run`: the autonomous design loop

```bash
python -m interfaces.cli.main run --iterations 6 --seed 1          # live TUI
python -m interfaces.cli.main run --no-tui --target-mass 0.30      # headless
```

```
 # │ action  │ strategy          │ mass (kg) │ feasible │ best │ evals
 0 │ exploit │ initial-exploit   │  0.249977 │   yes    │ NEW  │   196
 1 │ explore │ explore-scheduled │  0.249976 │   yes    │ NEW  │   349
 4 │ explore │ explore-on-stall  │  0.249977 │   yes    │  -   │    14

  termination  converged
       detail  4 consecutive iterations improved by less than 0.100%
       budget  964/20000 evaluations, 10.2/300 s
```

Options: `--iterations`, `--seed`, `--target-mass`, `--max-evaluations`,
`--max-seconds`, `--profile`, `--tui/--no-tui`.

### `brain`: inspect accumulated experience

```bash
python -m interfaces.cli.main brain --generalize
```

```
┃ level    ┃  conf ┃ evidence ┃ runs ┃ statement                          ┃
│ repeated │ 0.692 │        9 │    3 │ For cantilever_link designs, the   │
│          │       │          │      │ binding constraint is 'deflection' │
│          │       │          │      │ (active in 9/9 feasible episodes). │
```

---

## Fidelity & safety: read before trusting any number

This is the part that distinguishes the project. Every layer states what it
does **not** know.

**Physics (Phase 2) is Euler-Bernoulli beam theory.** It ignores root stress
concentration, ignores transverse shear deformation, and does not check
buckling. **Real peak stress at the root will be higher than reported**: treat
the reported stress as a lower bound. A design that passes here is a
**candidate, not a verified part**.

**Phase 7 added a 3D FEM gate, and it immediately caught something.** The
Phase 3 optimum passed beam theory at exactly 1.00000 mm tip deflection, on its
1 mm limit. Under 3D FEM the same design deflects **1.019 mm and violates the
constraint**, because Euler-Bernoulli omits shear deformation and this link is
not slender (L/h is about 6). The optimizer had found a design sitting precisely
on a blind spot of the cheap model.

**Phase 7.5 closed that loop.** The beam model gained a Timoshenko shear term,
calibrated against 3D FEM across L/h from 4 to 20: mean error fell from 2.07% to
0.35%. Re-optimizing under the corrected model gives a design **0.74% heavier**
(0.2518 kg) that **passes 3D FEM at 0.9975 mm**. Learning that a cheap model is
wrong, fixing it, and re-deriving a design that survives the gate is the point
of the funnel.

**Even 3D FEM is still a simulation**: linear elastic, small strain, and an
idealised fully clamped root. That idealisation is a **stress singularity**, so
the peak stress it reports **does not converge under mesh refinement** and must
not be used to certify anything. A gauge measure offset from the support is
reported alongside it, and that one does converge.

**The surrogate (Phase 6) approximates that beam evaluator, not 3D FEM.** Its
error stacks on top of beam theory's own. It never decides: `screen_and_verify`
ranks thousands of candidates with the model but returns a design the **solver**
evaluated. There is also **no speedup today**: the beam kernel is closed-form
arithmetic, so the surrogate measures ~0.38× the batched solver's throughput.
The value is deferred to Phase 7.

**The Brain (Phase 5) stores evidence-graded experience, not facts.**
`EXPERIMENTALLY_VALIDATED` is reachable **only** with physical-test evidence: 
no volume of simulation, no passing test suite, no analytical derivation can
promote a claim to it. Independence is counted per *run*, not per episode, so
one long search yields at most `SIMULATED`.

**The reasoner (Phase 4) is a rule-based heuristic, not a language model.**
Calling it AI reasoning would be an overclaim. `Reasoner` is a one-method ABC: 
that is the documented seam where an LLM policy plugs in.

**A topology result is a design concept, not a part.** SIMP leaves intermediate densities that have to be thresholded, so the shape you get is not the field that was optimized. It exports as a blocky voxel STL rather than a clean STEP, for the same reason organic geometry always does. And it minimises **compliance, not stress**: it carries no stress constraint and says nothing about peak stress. It still has to pass the 3D FEM gate.

**The motor and gearbox catalogues are illustrative archetypes, not real parts.** No vendor part numbers were invented, because a fabricated catalogue gets read later as if it had been sourced. The selection logic is the deliverable: replace the catalogue with datasheet values before ordering. The thermal check is a continuous-torque proxy, so results are labelled subject to thermal validation, and when nothing meets the requirement the selector reports infeasible rather than returning the least bad option. First-pass screening, not a final component decision.

**Dynamics gives required torque, not a motor choice.** Phase 11 adds inertia, Coriolis and acceleration terms, and reports peak and continuous (duty-weighted RMS) requirements separately, because a motor has both ratings and they differ by about 2.5x here. Friction, backlash and joint compliance remain **zero**: the terms exist, the data does not, and inventing it would put fabricated numbers into a torque an actuator gets selected from. Selecting the motor and gearbox is a later phase.

**Assembly analysis was statics only through Phase 10.** Phase 10 computes the joint torques needed to hold a pose against gravity and a payload, and feeds each link's root bending moment into the structural stack. There is no inertia, no Coriolis or acceleration torque, no friction, no backlash and no joint compliance: rigid bodies on ideal joints. Those torques size a **link**; they do **not** size a motor or a gearbox, which needs the dynamic terms and is a later phase.

**Exported CAD is analysis geometry, not a manufacturing-ready part.** STEP export is exact for parametric solids and is refused outright if the B-rep volume disagrees with the mass the physics used, so the file is always the part that was analysed. But it has no fillets, no fastener features and no tolerances, and its sharp root corner is precisely where Phase 7 found the stress concentration. Organic and topology-optimized shapes do **not** get a clean STEP: that needs surface reconstruction, and the mesh path says so rather than emitting geometry that looks manufacturable.

**Material values carry their own caveats.** The database holds 15 materials, all `reference_typical` rather than certified datasheet values. Polymer properties depend strongly on temperature, strain rate and process, and a printed part is not isotropic, so the bulk values stored here are an upper bound on a printed one. Alumina has **no ductile yield point**, so a yield-based safety factor is the wrong failure criterion for it. CFRP is orthotropic with a 30x ratio between fibre-direction and transverse strength, which is why a single yield number is not offered for it. Derived values (such as G from E and nu) are exact and marked as derived; estimated values carry an uncertainty and force the material's status down to `ASSUMED`.

**The optimum depends on an assumed manufacturing bound.** Two of three design
variables land on their bounds, and the 1 mm minimum wall thickness is an
**[ASSUMED]** CNC-aluminium limit, not a derived one. Change the process and
the achievable mass changes with it.

---

## Installation & distribution (standalone CLI): open design

The intent is to package this as a **self-contained, installable CLI tool**: 
the kind of experience where a user installs once and runs a single clean
command. Proposed direction:

- A **console entry point** in `pyproject.toml` so installing provides a single
  `dae` command (with `daedalus` available as a longer alias). The CLI is
  already Typer-based, so this is a natural fit.
- **`pipx` for isolated global install**, or a **bootstrap install script** that
  creates the venv, installs the GPU dependencies (Warp / torch), and **wraps
  the clean-`PYTHONPATH` invocation automatically** so users never have to think
  about environment shadowing.
- The existing `interfaces/cli` commands become subcommands of that single
  entry point:

  ```bash
  dae evaluate --width 50 --height 80 --thickness 5
  dae optimize --method both
  dae run --iterations 6
  dae brain --generalize
  ```

- A PyInstaller single binary is **low priority**: torch and Warp CUDA wheels
  make it impractical for now.

> **This is not settled. The packaging and installation UX is still being
> designed, and proposals and opinions on the approach are welcome.**

### Release installation methods

> Installation methods will be finalized at release. **Proposals welcome.**

| method | status |
|---|---|
| pip | `TBD, to be provided at release` |
| pipx | `TBD, to be provided at release` |
| PowerShell (Windows) | `TBD, to be provided at release` |
| cmd (Windows) | `TBD, to be provided at release` |
| bash / curl (Linux, macOS) | `TBD, to be provided at release` |
| Node / npx | `TBD, to be provided at release` |
| Docker | `TBD, to be provided at release` |

**pip**
```
# TBD, to be provided at release
```

**pipx**
```
# TBD, to be provided at release
```

**PowerShell (Windows)**
```
# TBD, to be provided at release
```

**cmd (Windows)**
```
# TBD, to be provided at release
```

**bash / curl (Linux, macOS)**
```
# TBD, to be provided at release
```

**Node / npx**
```
# TBD, to be provided at release
```

**Docker**
```
# TBD, to be provided at release
```

---

## Roadmap

- **Phase 7: high-fidelity 3D FEM**: stress concentration and buckling, the
  gate a candidate must clear to be treated as a real part. This is also where
  the surrogate infrastructure finally pays off.
- **Anisotropic materials**: CFRP and 3D-printed plastics need direction-
  dependent property fields **and** an anisotropic solver before they can be
  added: forcing them into the current single-E schema would produce confident
  wrong answers.
- **Topology optimization**, implicit/SDF geometry, lattice structures.
- **LLM-backed reasoner** plugged into the existing `Reasoner` ABC.
- **Multi-GPU device pool**: independent candidates shard linearly.
- **CAD / STEP export** for manufacturing handoff.
- **Part scope**: link → joint → gearbox → leg → humanoid.
- **Text-embedding semantic retrieval + ANN indexing** in the Brain (today's
  retrieval is numeric feature similarity, deliberately not called semantic).
- **Fine-tuning** a domain-specialized model on accumulated experience.

---

## Contributors

<p align="center">
<a href="https://github.com/SeongminJaden"><img src="https://github.com/SeongminJaden.png?size=100" width="48" alt="SeongminJaden"/></a>
</p>

<!-- To add a contributor, add another <a><img></a> beside the one above.
     For an auto-updating grid with circular avatars, replace the block with:
[![Contributors](https://contrib.rocks/image?repo=SeongminJaden/daedalus-engineering-ai)](https://github.com/SeongminJaden/daedalus-engineering-ai/graphs/contributors)
     Note: GitHub strips style attributes from README HTML, so hand-written
     avatars render square. contrib.rocks is what produces round ones. -->

Contributions are welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)**. The
project cares less about volume than about a specific habit: every layer states
what it does *not* know, critical calculations are verified against an
independent method, and nothing is overclaimed. A change that keeps that
property is the kind this project wants.

---

## Sponsors

*No sponsors yet: this section is waiting for its first.*

If this work is useful to you and you would like to support it, sponsorship
options will be listed here once they are set up. `.github/FUNDING.yml` is
present as a commented template; nothing is enabled yet, because a funding link
that does not work is worse than none.

---

## Community

[![Discord](https://img.shields.io/badge/Discord-server%20not%20yet%20created-5865F2.svg)](#community)

**Discord: `TBD, link to be added once the server is created.`**

There is no invite link yet, and one will not be invented here. Until the server
exists, GitHub **Issues** and **Discussions** are the place for questions,
proposals and design debate: particularly on the two open questions flagged
above: the **packaging / installation UX**, and **anything in the fidelity
model you think is wrong**. Being told a number is misleading is the most useful
contribution this project can receive.

---

## License

[Apache License 2.0](LICENSE) © 2026 SeongminJaden. See also [NOTICE](NOTICE).

Apache-2.0 was chosen over a permissive-only licence deliberately, for patent
defence. It grants an explicit patent licence from every contributor, and it
terminates that grant for anyone who initiates patent litigation over the work.
A licence without a patent clause leaves both contributors and users exposed to
exactly that.

### Defensive publication

Publishing the methodology openly (`DESIGN.md`, this README, and the source) is
also intended to establish prior art. Once a technique is described publicly and
datably, it is substantially harder for a third party to patent the same
approach later and assert it against the community that already uses it. The
combination is the point: prior art makes the ideas hard to claim, and
Apache-2.0's patent grant plus retaliation clause protects the people who build
on them.

---

## Repository

`SeongminJaden/daedalus-engineering-ai`
