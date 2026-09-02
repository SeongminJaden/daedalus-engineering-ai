<!-- English: [README.md](README.md) -->

<div align="center">

<img src="assets/logo.svg" width="88" alt="Daedalus">

# Daedalus Engineering AI

**자율 엔지니어링 설계 에이전트**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![GPU: NVIDIA Warp](https://img.shields.io/badge/GPU-NVIDIA%20Warp-76b900.svg)](https://github.com/NVIDIA/warp)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-1574%20passing-brightgreen.svg)](#현재-상태)
[![Capabilities](https://img.shields.io/badge/capabilities-54%20registered-orange.svg)](#현재-상태)
[![External solvers](https://img.shields.io/badge/external%20solvers-7%20cross--checking-blue.svg)](#현재-상태)
[![Evidence](https://img.shields.io/badge/evidence-simulated%2C%20not%20validated-lightgrey.svg)](#충실도와-안전성-숫자를-믿기-전에-읽을-것)

</div>

공학 목표만 주면 추론 → 설계 → GPU 물리 → 최적화 → 학습을 반복해 로봇 부품을
설계하고, 배운 것을 **Engineering Brain**에 축적한다. Brain의 모든 주장에는
**증거수준**이 명시된다.

에이전트와 CLI의 브랜드는 **Daedalus**이다.

---

## 현재 상태

아래 전부는 구현되고, 독립 레퍼런스와 대조 테스트되고, 개발 머신에서 실행된
것이다. 아래 어느 것도 물리시험은 거치지 않았고, 코드는 자기 출력을 그에 맞게
등급 매긴다.

**12개 노드 위에 등록된 능력 54개.** 능력은 방법 하나와 그것을 돌리는 노드 하나다.
라우팅 규칙은 하나뿐이다: 방법이 문제에 적용 가능하고 노드가 살아 있을 때만 후보가
된다. 제외된 방법은 이유를 말한다.

| 실행 위치 | 수 | 내용 |
|---|---|---|
| 프로세스 내 엔진, GPU | 42 | 보·Timoshenko 이론, matrix-free 3D FEM, 피로(S-N, Goodman, Miner), 오일러 좌굴, 샤프트, 베어링, 볼트, 나사, 기어, 키, 용접, 압입, ISO 286 끼워맞춤, Hertz 접촉, 열저항망과 과도열, 관유동, 항력, 유체 액추에이터, 적층판(CLT), 정역학, 강체 동역학, 모터·감속기 선정, SLSQP, 차분진화, NSGA-II, SIMP 토폴로지(컴플라이언스·응력), 최소치수, 다중설계검토 |
| 외부 솔버 노드, stdio | 7 | CalculiX(FEA, 일반 형상), Code_Aster(소성), Elmer(정자기), OpenFOAM(CFD), Gmsh(메싱), Pinocchio(다물체), MuJoCo(접촉) |
| CAD 지식 계층 | 3 | STEP 분석기, 규칙기반 피처 인식, 벽두께·구배 검사; build123d 파라메트릭 형상의 STEP 출력 |
| 스텁, 정직하게 미가용 | 2 | Fusion 라운드트립(Windows 호스트와 엔타이틀먼트 필요), 외부 LLM reasoner |

두 방법이 겹치는 곳은 교차검증이지 두 번째 능력이 아니다. CalculiX 는 두 메셔가
모두 덮는 한 형상에서 자체 hex FEM 과 0.5 퍼센트 안에서 일치했고, OpenFOAM 은 유체
상관식의 첫 독립 검증이었으며, Pinocchio 와 MuJoCo 는 연속체 스택 밖에서 동역학을
검사한다.

**모든 것에 하나의 근거 사다리.** Brain 이 저장하는 모든 진술, 어셈블리 판정의 모든
체크, 모든 서로게이트 예측이 등급을 달고 있다:

```
UNVERIFIED  <  SURROGATE  <  SIMULATED  <  REPEATED  <  HIGH_CONFIDENCE  <  EXPERIMENTALLY_VALIDATED
   0.20         0.40          0.60          0.80           0.95                   0.99
```

이 저장소의 모든 것은 `SIMULATED` 이하에 있다. `SURROGATE` 는 학습된 모델의 출력이며
스크리닝은 해도 판정은 절대 못 한다. `REPEATED` 이상은 서로 일치하는 독립 실행이
필요하다. `EXPERIMENTALLY_VALIDATED` 는 물리시험이 필요하고, 시뮬레이션은 아무리
많아도 그 문을 열지 못한다. 이 규칙들은 각각 관례가 아니라 테스트다.

**이 프로젝트가 오르는 검증 사다리, 순서대로:**

| 단 | 상태 |
|---|---|
| 닫힌 형태 해와 독립 솔버에 대한 검증 | 완료, 모든 방법에서 계속 |
| 생성설계 트랙(합성데이터, 분류, 임베딩, 서로게이트, 설계의도, 생성) | 진행중 |
| 하드웨어: 출력된 STEP 으로 제조한 부품 | 로드맵 |
| 실측: 물리시험 근거, `EXPERIMENTALLY_VALIDATED` 의 유일한 열쇠 | 로드맵 |

**MVP 문제**: 중공 사각단면 로봇 링크 1개의 질량 최소화. 근부 고정 캔틸레버,
팁에 196.2 N(20 kg 페이로드), 알루미늄 7075-T6, 응력 상한·팁 처짐 한계·안전율
제약.

**결과: 1.686 kg → 0.250 kg, 질량 85.2% 감소.** 서로 독립적인 두 최적화기가
상대차 1.3×10⁻⁵로 일치. 이 설계는 **처짐 지배(deflection-limited)** 다.
팁 처짐이 1 mm 한계에 정확히 붙는 반면 응력 제약은 70% 넘는 여유가 남는다.

**테스트 1574개 통과.** 중요 계산은 전부 별도로 유도한 독립 레퍼런스와 대조
검증. 한계도 테스트로 박혀 있다: 방법이 못 하는 것은 못 한다고 말하는지를 테스트가
확인한다.

---

## 아키텍처

<p align="center"><img src="assets/architecture.svg" alt="Daedalus architecture" width="840"></p>

목표는 **Engineering IR**(문제: 형상·재료·하중·제약·목표, 전부 고정)로 들어온다.
**Design Genome**은 탐색이 바꿔도 되는 것만 담는다: 단면 치수, 토폴로지 필드, CAD
파라미터. 후보는 **capability registry** 로 가고, 레지스트리는 각 파손 모드를 적용
가능하고 노드가 살아 있는 등록 방법으로 보낸다. GPU 위 프로세스 내에서든, stdio 너머
외부 솔버에서든 규칙은 같다. 결과는 **다중 충실도 깔때기**를 내려간다: surrogate가
수천 개를 걸러내고, 보 이론이 후보군을 평가하며, 3D FEM이 관문이고, 독립 솔버가
가능한 것을 교차검증한다. 적용 가능한 모든 파손 모드가 **결합 판정**을 통과해야 하며,
평가되지 않은 모드나 서로게이트만 본 모드는 통과가 아니라 공백이다. 매 이터레이션은
증거수준이 붙은 채 **Engineering Brain**에 기록되어 다음 회차에 reasoner가 읽는다.

두 가지 분리가 설계를 받친다. 물리가 문제 없이 genome만 보는 일은 절대 없어서 설계
변수가 조용히 요구사항으로 둔갑할 수 없다. 그리고 판정이 서로게이트 위에 서는 일은
절대 없어서 학습된 모델이 조용히 솔버로 둔갑할 수 없다.

## 하드웨어 & GPU 프로파일

**RTX 3050 Laptop GPU (4 GB)** 에서 개발·검증했다. 규모는 코드에서 분리되어
`configs/profiles/*.yaml` 에 산다.

| 프로파일 | VRAM | 성격 |
|---|---|---|
| `laptop_4gb` | 4 GB | 현재 목표. 후보풀 작게, AMP + gradient checkpointing |
| `desktop_16gb` | 16 GB | 중급 워크스테이션 |
| `rtx5090_32gb` | 32 GB | 고대역폭. 이 워크로드에서 solve가 가장 빠름 |
| `dgx_spark_128gb` | 128 GB | 대용량·저대역폭. 큰 모델용, solve는 느림 |
| `cloud_a100` | 80 GB | 대규모 병렬 |

선택 우선순위: `--profile` → `ENG_PROFILE` 환경변수 → VRAM 자동감지 → fallback.

```bash
python -m interfaces.cli.main info                     # 자동감지
python -m interfaces.cli.main info --profile cloud_a100
ENG_PROFILE=desktop_16gb python -m interfaces.cli.main info
```

**시스템 CUDA 툴킷은 불필요하다**: Warp가 커널을 자체 JIT하고, torch는 CUDA
런타임을 휠에 포함해 배포한다.

---

## 시스템 요구사양

실제로 돌려본 것과 제공 프로파일에 근거한 값이다. 희망사항이 아니다.

### 소프트웨어

| | 요구사항 |
|---|---|
| OS | **Linux**: 개발·검증 플랫폼. Windows/macOS는 **TBD / 실험적**, 아직 검증 안 됨 |
| Python | **3.10+** (3.10.12에서 검증) |
| GPU 드라이버 | 번들된 torch/warp의 CUDA 런타임을 지원하는 NVIDIA 드라이버. **시스템 CUDA 툴킷은 필요 없음** |
| 디스크 | **약 6 GB**: venv가 약 5 GB(torch + Warp CUDA 휠), 여기에 `datasets/`·`runs/` 여유분 |

### 하드웨어

| 티어 | GPU | CPU / RAM | 프로파일 |
|---|---|---|---|
| **최소** (MVP·개발) | NVIDIA VRAM 4 GB (예: RTX 3050) | 8코어 / 16 GB | `laptop_4gb` |
| **권장** | VRAM 16 GB 이상 (RTX 4070 Ti / 4080, 중고 3090 24 GB) | 8코어+ / 16~32 GB | `desktop_16gb`, `rtx5090_32gb` |
| **대규모** | VRAM 24~48 GB+ (4090 / 5090 / A6000) 또는 클라우드 A100 80 GB | 16코어+ / 64 GB+ | `cloud_a100` |
| **CPU 전용** | 없음, Warp에 CPU 디바이스가 있음 |, | 동작은 하나 **느리고 제한적**. GPU를 강력히 권장 |

---

## 설치 (개발용)

```bash
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/python -m pip install -U pip wheel
env -u PYTHONPATH .venv/bin/pip install -r requirements.txt
env -u PYTHONPATH .venv/bin/python scripts/gpu_sanity.py
```

> **venv는 깨끗한 `PYTHONPATH`에서 실행할 것.** 셸에 source된 다른 환경이
> `PYTHONPATH`를 내보내면 그쪽의 오래된 패키지가 venv 패키지를 가릴 수 있다
> (예를 들어 다른 버전의 `numpy`가 조용히 먼저 잡힌다). `env -u PYTHONPATH`를
> 앞에 붙이면 피할 수 있다. 아래의 부트스트랩 설치 스크립트는 이 래핑을 자동
> 처리해서 최종 사용자가 신경 쓸 필요 없게 하는 것이 목표다.

`gpu_sanity.py`는 Warp를 초기화하고, 실제 커널을 컴파일·실행해 결과를 검증하고,
torch CUDA를 확인하고, 선택된 프로파일을 출력한다.

---

## 사용법

현재는 전부 `python -m interfaces.cli.main <command>` 형태다. 아래의 패키징된
CLI에서는 `dae <command>` 로 노출된다.

### `evaluate`: 설계 하나를 GPU에서 평가

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

### `optimize`: 질량 최소화, 두 방법으로 교차검증

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

### `run`: 자율 설계 루프

```bash
python -m interfaces.cli.main run --iterations 6 --seed 1          # 라이브 TUI
python -m interfaces.cli.main run --no-tui --target-mass 0.30      # 헤드리스
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

옵션: `--iterations`, `--seed`, `--target-mass`, `--max-evaluations`,
`--max-seconds`, `--profile`, `--tui/--no-tui`.

### `brain`: 축적된 경험 조회

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

## 충실도와 안전성: 숫자를 믿기 전에 읽을 것

이 프로젝트를 다른 것과 구분 짓는 부분이다. 각 계층은 자신이 **모르는 것**을
명시한다.

**물리(Phase 2)는 Euler-Bernoulli 보 이론이다.** 근부 응력집중을 무시하고,
전단 변형을 무시하며, 좌굴을 검사하지 않는다. **실제 근부 최대응력은 보고값보다
높다**: 보고된 응력은 하한으로 취급할 것. 여기를 통과한 설계는 **후보이지
검증된 부품이 아니다.**

**Phase 7의 3D FEM 관문은 곧바로 하나를 잡아냈다.** Phase 3 최적해는 보 이론에서
팁 처짐 1.00000 mm로 1 mm 한계에 정확히 걸쳐 통과했다. 같은 설계를 3D FEM으로
풀면 **1.019 mm로 제약을 위반한다.** Euler-Bernoulli가 전단 변형을 빼먹는데 이
링크는 세장하지 않기 때문이다(L/h 약 6). 최적화기가 값싼 모델의 맹점 위에 정확히
올라앉은 해를 찾아냈던 것이다.

**Phase 7.5가 그 루프를 닫았다.** 보 모델에 Timoshenko 전단 항을 넣고 L/h 4~20
구간에서 3D FEM으로 검증했다. 평균 오차 2.07% → 0.35%. 보정된 모델로 재최적화한
설계는 **0.74% 무겁지만**(0.2518 kg) **3D FEM을 0.9975 mm로 통과한다.** 값싼
모델이 틀렸음을 배우고, 고치고, 관문을 견디는 설계를 다시 내놓는 것이 깔때기의
존재 이유다.

**3D FEM조차 여전히 시뮬레이션이다**: 선형탄성, 소변형, 이상화된 완전고정 근부.
그 이상화는 **응력 특이점**이라 FEM이 보고하는 peak 응력은 **메시를 조여도
수렴하지 않으며** 부품 인증에 쓰면 안 된다. 지지부에서 떨어진 게이지 측도를 함께
보고하며 그쪽은 수렴한다.

**surrogate(Phase 6)가 근사하는 것은 그 보 이론 평가기이지 3D FEM이 아니다.**
그래서 오차가 보 이론 오차 위에 더해진다. 그리고 surrogate는 결정하지 않는다: 
`screen_and_verify`는 수천 후보를 모델로 순위 매기지만, 반환하는 설계는 항상
**솔버가 실제로 평가한** 것이다. 이 규칙은 이제 제어 흐름이 아니라 근거등급
사다리에 박혀 있다: 모든 예측은 스스로를 `SIMULATED` **아래**의 `SURROGATE`
등급으로 매기고, 판정 계층은 그 위에 pass/fail 을 세우기를 거부한다. 또한
**현재 속도 이득은 없다**: 보 커널이 닫힌 형태 산술이라 surrogate 처리량은 배치
솔버의 약 0.4배다. 가치는 기본 평가기가 비싼 솔브가 될 때 나온다.

**Brain(Phase 5)은 사실이 아니라 증거수준이 표시된 경험을 저장한다.**
`EXPERIMENTALLY_VALIDATED`는 **물리시험 증거로만** 도달 가능하다: 시뮬레이션을
아무리 많이 돌려도, 테스트 스위트를 전부 통과해도, 해석적으로 유도해도 그
관문은 열리지 않는다. 독립성은 에피소드가 아니라 **run 단위**로 세므로, 한 번의
긴 탐색은 최대 `SIMULATED`에 머문다. 서로게이트 근거는 세기 전에 따로 떼어 놓으므로,
예측 천 개는 `SURROGATE`를 낳고 그 위로는 올라가지 못한다.

**reasoner(Phase 4)는 규칙기반 휴리스틱이지 언어모델이 아니다.** 이걸 AI 추론이라
부르면 과대주장이다. `Reasoner`는 메서드 하나짜리 ABC이며, 그것이 LLM 정책을
끼워 넣도록 문서화된 이음매다.

**토폴로지 결과는 설계 개념이지 부품이 아니다.** SIMP는 임계처리해야 하는 중간 밀도를 남기므로, 얻는 형상은 최적화한 밀도장과 다르다. 유기 형상이 늘 그렇듯 깨끗한 STEP이 아니라 각진 복셀 STL로 나온다. 그리고 **응력이 아니라 컴플라이언스**를 최소화한다. 응력 제약이 없고 최대응력에 대해 아무 말도 하지 않는다. 3D FEM 관문을 여전히 통과해야 한다.

**모터·감속기 카탈로그는 실제 부품이 아니라 illustrative 아키타입이다.** 벤더 부품번호를 지어내지 않았다. 가짜 카탈로그는 나중에 출처가 있는 것처럼 읽히기 때문이다. 선정 로직이 deliverable이며, 주문 전에 데이터시트 값으로 교체해야 한다. 열 검사는 연속토크 프록시라 결과에 thermal validation 조건을 붙이고, 충족 조합이 없으면 가장 덜 나쁜 것을 고르지 않고 infeasible로 보고한다. 1차 스크리닝이지 최종 부품 결정이 아니다.

**동역학은 필요 토크를 줄 뿐 모터를 골라주지 않는다.** Phase 11은 관성·코리올리·가속 항을 더하고, peak와 continuous(duty 가중 RMS)를 따로 보고한다. 모터에는 두 정격이 다 있고 여기서는 약 2.5배 차이가 나기 때문이다. 마찰·백래시·관절 컴플라이언스는 여전히 **0**이다. 항은 있고 데이터가 없으며, 지어내면 액추에이터를 고르는 근거 토크에 조작된 숫자가 들어간다. 모터와 감속기 선정은 후속 단계다.

**어셈블리 해석은 Phase 10까지는 정역학이었다.** Phase 10은 중력과 페이로드에 맞서 자세를 유지하는 관절 토크를 구하고, 각 링크의 근부 굽힘모멘트를 구조 해석에 넘긴다. 관성도, 코리올리·가속 토크도, 마찰도, 백래시도, 관절 컴플라이언스도 없다. 강체와 이상 관절 가정이다. 이 토크는 **링크**를 사이징하지 **모터나 감속기를 사이징하지 못한다**. 그건 동역학 항이 필요하며 후속 단계다.

**내보낸 CAD는 해석 형상이지 제조 준비 형상이 아니다.** STEP 출력은 파라메트릭 솔리드에 대해 정확하고, B-rep 부피가 물리가 쓴 질량과 어긋나면 아예 거부하므로 파일은 항상 해석한 그 부품이다. 하지만 필렛도 체결 피처도 공차도 없고, 날카로운 근부 모서리는 Phase 7이 응력집중을 찾아낸 바로 그 위치다. 유기·토폴로지 형상은 깨끗한 STEP을 받지 **못한다**. 표면 재구성이 필요하며, 메시 경로는 제조 가능해 보이는 지오메트리를 뱉는 대신 그렇다고 말한다.

**재료 값에는 각자의 단서가 붙는다.** DB의 15종은 전부 인증된 데이터시트가 아니라 `reference_typical`이다. 고분자 물성은 온도·변형률·공정 의존이 크고 프린팅 부품은 등방이 아니므로, 여기 저장된 벌크 값은 프린팅 부품의 **상한**이다. 알루미나는 **연성 항복점이 없어서** 항복 기준 안전율이 틀린 파괴 기준이다. CFRP는 직교이방이며 섬유방향과 횡방향 강도비가 30배라 단일 항복값을 제공하지 않는다. 유도값(E와 ν에서 얻는 G 등)은 정확하며 유도됨으로 표시되고, 추정값은 불확실성을 동반하며 재료 status를 `ASSUMED`로 강등시킨다.

**최적해는 가정된 제조 bound에 의존한다.** 설계변수 3개 중 2개가 bound에 붙어서
끝나고, 그중 1 mm 최소 벽 두께는 유도된 값이 아니라 **[ASSUMED]** CNC 알루미늄
한계다. 공정이 바뀌면 달성 가능한 질량도 함께 바뀐다.

---

## 설치·배포 (설치형 CLI): 열린 설계

이 프로젝트를 **자체적으로 완결된 설치형 CLI 도구**로 패키징하는 것이 목표다.
사용자가 한 번 설치하면 깔끔한 단일 명령으로 쓰는 형태. 제안하는 방향:

- `pyproject.toml`에 **console entry point**를 정의해 설치하면 `dae` 단일
  명령이 생기게 한다(긴 별칭으로 `daedalus`도 제공). 이미 Typer 기반이라
  자연스럽다.
- **`pipx`로 격리 설치**하거나, **부트스트랩 설치 스크립트**가 venv 생성 + GPU
  의존성(Warp / torch) 설치 + **깨끗한 `PYTHONPATH` 래핑까지 자동 처리**해서
  사용자가 환경 오염을 신경 쓰지 않게 한다.
- 현재 `interfaces/cli`의 명령들이 그 단일 진입점의 서브커맨드가 된다:

  ```bash
  dae evaluate --width 50 --height 80 --thickness 5
  dae optimize --method both
  dae run --iterations 6
  dae brain --generalize
  ```

- PyInstaller 단일 바이너리는 **후순위**: torch·Warp CUDA 휠 때문에 현실적이지
  않다.

> **이건 확정안이 아니다. 패키징·설치 UX는 아직 설계 중이며, 접근법에 대한
> 제안과 의견을 환영한다.**

### 릴리즈 설치 방법

> 설치 방법은 릴리즈 시 확정된다. **제안 환영.**

| 방법 | 상태 |
|---|---|
| pip | `TBD, 릴리즈 시 제공` |
| pipx | `TBD, 릴리즈 시 제공` |
| PowerShell (Windows) | `TBD, 릴리즈 시 제공` |
| cmd (Windows) | `TBD, 릴리즈 시 제공` |
| bash / curl (Linux, macOS) | `TBD, 릴리즈 시 제공` |
| Node / npx | `TBD, 릴리즈 시 제공` |
| Docker | `TBD, 릴리즈 시 제공` |

**pip**
```
# TBD, 릴리즈 시 제공
```

**pipx**
```
# TBD, 릴리즈 시 제공
```

**PowerShell (Windows)**
```
# TBD, 릴리즈 시 제공
```

**cmd (Windows)**
```
# TBD, 릴리즈 시 제공
```

**bash / curl (Linux, macOS)**
```
# TBD, 릴리즈 시 제공
```

**Node / npx**
```
# TBD, 릴리즈 시 제공
```

**Docker**
```
# TBD, 릴리즈 시 제공
```

---

## 로드맵

사다리 둘을 순서대로 오른다. 어느 것도 측정되기 전에는 주장하지 않는다.

**생성설계 트랙(진행중).** 순서는 서로게이트 근거 게이트가 들어간 뒤에 확정했다.
그 게이트 없이 학습 모델이 루프에 들어오면 "모델이 그렇게 말했다"가 "솔버가 그렇게
말했다"에 섞이기 때문이다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 게이트 | `SIMULATED` 아래의 `SURROGATE` 근거등급; 서로게이트는 스크리닝만 하고 판정은 못 한다, 코드와 테스트로 강제 | 완료 |
| P5 | 합성데이터 엔진: 닫힌 형태 부피를 가진 build123d 패밀리 다섯, 모든 레코드를 자기 파라미터와 대조, Gmsh 와 CalculiX 로 라벨링하며 모든 솔버 라벨에 메시 민감도 기록; 라벨은 구조적으로 `SIMULATED` | 완료 |
| P3 | 형상 기술자와 분류 | 진행중 |
| P6 | CAD 임베딩 | 계획 |
| P7 | CAD 형상 서로게이트 예측, 탐색 가속 전용, 게이트 뒤에서 | 계획 |
| P8 | 설계의도, 주장이 아니라 실제 솔버 대비 절제 실험으로 측정 | 계획 |
| P9 | 생성설계와 STEP 을 내보내는 자율 CAD 루프 | 계획 |

**검증 사다리(로드맵).** 시뮬레이션 검증이 지금 프로젝트가 있는 자리다. 다음은
하드웨어: 출력된 STEP 파일로 제조한 부품. 마지막은 실측: 진술을
`EXPERIMENTALLY_VALIDATED` 로 올릴 수 있는 유일한 것인 물리시험. 하드웨어와 실측은
이 소프트웨어가 아니라 사람이 수행한다.

**그 외 열려 있는 것**, 순서 약속 없이: 기존 `Reasoner` 이음매 뒤의 LLM 기반
reasoner; 멀티 GPU device pool; 분석기를 대조할 독립 CAD 커널로서 Windows 호스트의
Fusion 라운드트립; Brain 의 텍스트 임베딩 검색(현재는 수치 특징 유사도이며 의도적으로
의미검색이라 부르지 않는다).

---

## 함께한 사람들

<p align="center">
<a href="https://github.com/SeongminJaden"><img src="https://github.com/SeongminJaden.png?size=100" width="48" alt="SeongminJaden"/></a>
</p>

<!-- To add a contributor, add another <a><img></a> beside the one above.
     For an auto-updating grid with circular avatars, replace the block with:
[![Contributors](https://contrib.rocks/image?repo=SeongminJaden/daedalus-engineering-ai)](https://github.com/SeongminJaden/daedalus-engineering-ai/graphs/contributors)
     Note: GitHub strips style attributes from README HTML, so hand-written
     avatars render square. contrib.rocks is what produces round ones. -->

기여를 환영한다. **[CONTRIBUTING.md](CONTRIBUTING.md)** 참고. 이 프로젝트가
중요하게 여기는 건 기여의 양이 아니라 하나의 습관이다: 각 계층이 자신이
**모르는 것**을 명시하고, 중요한 계산은 독립적인 방법으로 검증하며, 과대주장을
하지 않는 것. 그 성질을 지키는 변경이 이 프로젝트가 원하는 기여다.

---

## 후원

*아직 후원자가 없다: 이 자리는 첫 후원자를 기다린다.*

이 작업이 도움이 되어 후원하고 싶다면, 후원 경로가 준비되는 대로 여기에 표시된다.
`.github/FUNDING.yml`은 주석 처리된 템플릿으로만 들어 있다. 동작하지 않는 후원
링크는 없느니만 못하기 때문에 아직 아무것도 활성화하지 않았다.

---

## 커뮤니티

[![Discord](https://img.shields.io/badge/Discord-server%20not%20yet%20created-5865F2.svg)](#커뮤니티)

**Discord: `TBD, 서버 개설 후 링크 삽입`**

아직 초대 링크가 없고, 여기서 지어내지도 않는다. 서버가 생기기 전까지는 GitHub
**Issues**와 **Discussions**가 질문·제안·설계 토론의 장소다. 특히 위에서 열어둔
두 가지 질문에 대한 의견을 환영한다: **패키징·설치 UX**, 그리고 **충실도 모델에서
틀렸다고 생각하는 부분**. 어떤 숫자가 오해를 부른다고 지적받는 것이 이 프로젝트가
받을 수 있는 가장 유용한 기여다.

---

## 라이선스

[Apache License 2.0](LICENSE) © 2026 SeongminJaden. [NOTICE](NOTICE) 참고.

허용적 라이선스 대신 Apache-2.0을 고른 것은 **특허 방어**를 위한 의도적 선택이다.
모든 기여자로부터 명시적인 특허 실시권을 부여받고, 이 저작물에 대해 특허 소송을
제기하는 쪽에는 그 실시권이 종료된다. 특허 조항이 없는 라이선스는 기여자와
사용자를 바로 그 위험에 노출시킨다.

### 방어적 공개 (defensive publication)

방법론을 공개하는 것(`DESIGN.md`, 이 문서, 그리고 소스) 자체가 **prior art**를
성립시키려는 목적도 갖는다. 어떤 기법이 공개적으로, 날짜를 특정할 수 있게
기술되고 나면, 제3자가 같은 접근을 나중에 특허화해 이미 그것을 쓰고 있는
커뮤니티에 주장하기가 훨씬 어려워진다. 둘의 조합이 핵심이다: prior art가 아이디어를
선점당하지 않게 하고, Apache-2.0의 특허 grant와 보복 종료 조항이 그 위에서
개발하는 사람들을 보호한다.

---

## 저장소

`SeongminJaden/daedalus-engineering-ai`
