# Autonomous Engineering AI — 개념설계 (v0)

> 원전: `autonomous_engineering_ai_brain.md`. 이 문서는 그 비전을 `ros` 머신(RTX 3050 4GB, 확장 예정)의 실제 실행 계획으로 내려앉힌 것.

## 0. 설계 원칙
1. **비전은 그대로, 하드웨어는 현실**: 3050 4GB에서 "지금 돌아가는 최대치"로 시작하되, 큰 GPU로 옮기면 코드 수정 없이 규모만 커지도록 **프로파일 옵션**으로 스케일을 분리.
2. **시스템 불가침**: 모든 파이썬·CUDA 의존성은 `.venv` 안에만. 시스템 파이썬(3.10.12)·드라이버 건드리지 않음.
3. **CLI 우선**: 자율 루프·모니터링은 터미널(rich TUI + JSONL 로그). 웹 GUI는 `interfaces/api/` 자리만 비워두고 후순위.

## 1. 기술 스택 (하드웨어 반영)
| 계층 | 선택 | 이유 |
|---|---|---|
| 물리 코어 | **NVIDIA Warp (warp-lang)** | pip·venv 설치, 자체 JIT(시스템 CUDA 툴킷 불필요), 미분가능, sm_86 지원, 확장성 |
| ML surrogate | **PyTorch (cu13x)** | CUDA 동작 확인, Warp↔torch 텐서 연동 |
| 최적화 | Warp autodiff + 파이썬 진화/베이지안 | 4GB에선 GPU 배치평가 + CPU 메타옵티마이저 |
| Geometry | implicit/SDF(파이썬+Warp) 우선, CAD 후순위 | 초기 topology엔 voxel/level-set이 적합 |
| Orchestration | Python + Typer CLI | |
| 모니터링 | rich/textual TUI + JSONL | CLI-only, tmux attach로 관찰 |
| 추론(LLM) | pa-a2 ↔ ros-brain 세션 루프 | 이미 구축된 자율 루프가 MD의 상위 오케스트레이션 |

## 2. GPU 프로파일 시스템 (핵심)
스케일을 코드에서 분리 → `configs/profiles/*.yaml`. 선택 우선순위: `--profile` → `ENG_PROFILE` → VRAM 자동감지 → default fallback. 로더: `core/profile.py`.

| 프로파일 | VRAM | 성격 |
|---|---|---|
| laptop_4gb | 4GB | 현재. 후보풀 작게, 저해상, AMP+grad checkpoint |
| desktop_16gb | 16GB | 중형 확장 |
| rtx5090_32gb | 32GB | 고대역폭·빠른 solve. 이 워크로드 속도 최적 |
| dgx_spark_128gb | 128GB | 대용량·저대역폭. 큰 문제/로컬LLM용, solve 속도는 낮음 |
| cloud_a100 | 80GB | 대량 병렬 |

## 3. 하드웨어 사이징 가이드 (참고)
VRAM을 잡아먹는 3요인: ①3D 토폴로지 해상도(res³×필드×4B) ②동시 후보 수 ③surrogate 모델. 병목은 항상 "후보수×해상도"이며 프로파일이 상한을 건다.
- **4GB(현재)**: 전 단계 개발 + MVP 가능. ~256³, 후보 ~수백~1천.
- **32GB(5090)**: 이 프로젝트 대부분을 빠르게. 대역폭 1.8TB/s로 solve가 빠름.
- **128GB(DGX Spark)**: 용량은 크나 대역폭 273GB/s로 solve 느림. 로컬 대형 LLM·초대형 단일문제에 유리.
- **멀티GPU**: 독립 후보 병렬은 선형 확장(주 워크로드에 최적). 단일 거대문제 분산은 NVLink 없는 5090에선 PCIe 병목. → 설계에 device-pool(후보 배치 샤딩) 개념 예정.

**판단 기준**: "solve를 빨리·많이" = 5090. "거대한 걸 통째로 담기/로컬 LLM 두뇌" = DGX Spark. 지금은 하드웨어 구매 불필요 — 3050으로 전 파이프라인 구축 후 천장에 닿을 때 선택. 어느 쪽이든 `--profile`로 흡수.

## 4. 자율 루프 매핑
MD의 REASON→PLAN→DESIGN→ACT→SIMULATE→EVALUATE→LEARN→UPDATE_BRAIN을 현 구조에 매핑:
- 추론/계획/관찰/학습(LLM층) = pa-a2 ↔ ros-brain 대화 루프
- 설계생성/물리/최적화/평가(엔진층) = venv 파이썬+Warp 프로세스
- Brain = 레포 내 구조화 저장소(초기 SQLite+JSONL, 후에 벡터+그래프)

## 5. 단계 계획 (물리엔진 조기투자 반영)
- **Phase 0 (완료)**: venv + 스켈레톤 + 프로파일 + Warp/torch GPU 새너티.
- **Phase 1**: Engineering IR + Design Genome (단일 링크 파라메트릭).
- **Phase 2**: Warp GPU 물리 코어 — 보/링크 응력·처짐·질량, 미분가능. 해석해 대조 검증.
- **Phase 3**: 최적화 루프 — Warp autodiff 경사 + 진화탐색, 4GB 내 다후보 배치.
- **Phase 4**: 자율 루프 — 세션이 엔진 구동·관찰·재계획.
- **Phase 5**: Brain — 에피소드/전략/지식.
- **Phase 6**: Surrogate(torch) — 데이터 축적 후.
- **Phase 7**: 고정밀 검증 + 확장 프로파일.

MVP 목표(MD §34): "하중 받는 단일 로봇 링크의 최소 질량 설계" 루프 1회 완주.

## 재료 확장 로드맵
재료 확장은 **물리가 감당할 수 있는 순서**로 진행한다. 스키마와 솔버가 못 받는 재료를 먼저 넣으면 값은 들어가도 계산이 틀린다.

- **1단계 (현재/MVP)**: 등방성 Al 2종만 — `al_7075_t6`, `al_6061_t6`.
- **2단계 — 등방성 금속 세트 확장**: 6061 / 7075 / 2024, S45C / SCM440, SS316, Ti-6Al-4V, AZ31 등. **현재 스키마·선형탄성 물리를 그대로 수용**하므로 코드 변경 없이 데이터만 추가하면 됨. 각 항목은 출처 있는 값 + `status` 태그 필수.
- **3단계 — 이방성 재료 추가**: CFRP(방향별 물성·적층 구성), 3D프린팅 플라스틱(공정 의존·직교이방성).
  **선행 조건: 스키마 확장(방향별 물성 필드) + 이방성 솔버 지원.** 단일 E/항복/포아송 스키마에 이방성 재료를 밀어넣으면 물리가 틀리므로, 선행 조건이 갖춰지기 전에는 추가 금지.

## 6. 운영 규칙 (필독)
- **venv 실행은 항상 `env -u PYTHONPATH` 로 감쌀 것.** 이 셸은 ROS 2가 source돼 PYTHONPATH에 /opt/ros/humble 등이 있어, 안 벗기면 ROS numpy 등이 venv 패키지를 가림.
- venv는 sudo 없이 `--without-pip` + get-pip 부트스트랩으로 구성됨(시스템 미변경).
- git init만 됨, 커밋은 사용자 확인 후.

## 7. 현재 상태
Phase 0 완료. Warp 1.16.0 + torch 2.13.0+cu130, RTX 3050(sm_86) 인식·커널 검증 통과. 자동감지 → laptop_4gb.
