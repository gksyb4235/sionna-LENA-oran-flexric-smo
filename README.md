# KHU NR / Sionna / O-RAN 실험 환경

ns-3.48, 5G-LENA, Sionna RT 캐시와 FlexRIC을 결합한 KHU 캠퍼스 연구용 테스트베드입니다.
두 대역은 모두 **NR**입니다. `gNB_4G_*`는 1.8 GHz 셀의 기존 식별자이며 LTE 구현을 뜻하지 않습니다.
이 README가 저장소 구조, 설치, 데이터 수집, 성능평가의 기준 문서입니다.

## 저장소 구조

```text
experiment/khu/
  datasets/          UE 이동 경로, 기지국 좌표, 고정 파라미터 CSV, scene
    ue_position/     여러 seed의 이동 경로
    conflict_500/    원본 변경 시나리오 manifest/CSV로 연결
    conflict_reserve50/  예비 501~550 설계로 연결
  caches/            주파수/안테나/RET별 Sionna HDF5와 생성 manifest
    ret_caches/      기본 RET 캐시와 재생성 가능한 merged_percell
  results/           원시 KPI, 로그, 실패 기록, 학습 산출물, 집계 결과
  tools/             고정/변경 파라미터 sweep, 캐시 생성 도구
  studies/           실험 설계, 학습/평가 코드, 격리된 backend 원본
near-rt-ric/          FlexRIC, 실행 xApp, 로컬 설치
  monitoring/        InfluxDB/Grafana Compose, provisioning, dashboards
non-rt-ric/           Non-RT RIC 서버와 정책 구현
  non_rt_ric/        A1 정책 검증/저장, rApp 처리 Python 패키지
smo/                 GNN, Agent, 제어 게이트웨이
scratch/             ns-3 시나리오 및 생성된 실험별 빌드 타깃
scripts/             공통 준비/실행/검증/평가 진입점
src/, contrib/       ns-3 및 NR/O-RAN 모듈
build-support/, bindings/, examples/, doc/, utils/  ns-3 기반 구성
deploy/conflict-sweep/  CPU-only Docker 배포 도구
```

`scenarios/khu-real`, `scenarios/results`, `scenes/khu-real`,
`monitoring`은 기존 경로를 위한 호환 링크입니다. 데이터는 위 새 위치에 한 벌만 있습니다.
동결한 manifest의 경로·바이트·해시는 변경하지 않았습니다. 과거 결과의 재개와 입력 검증에 필요하므로
호환 링크를 임의로 삭제하지 마세요. 새 코드와 문서는 `experiment/khu`를 사용합니다.
루트 `experiments` 링크는 제거했습니다. 기존 conflict 설정의 `experiments/...` 값은
실행기가 새 studies 위치로 해석하며, 원본 설정/manifest 해시를 바꾸지 않습니다.

원본 실험 코드의 역할은 다음과 같습니다.

| studies 하위 폴더 | 역할 |
|---|---|
| `conflict_scenarios_500` | 900초 도중 예약된 파라미터 변경을 재생하는 원본 500개 설계 |
| `conflict_scenarios_reserve50` | 별도 manifest의 예비 501~550 설계 |
| `two_switch_900s` | 이전 두 번 변경 실험의 설계 |
| `drl_xapp_offline` | 고정 실행 데이터로 학습한 정책 및 평가 코드 |
| `drl_xapp_conflict` | 별도 온라인 DRL 실험 |
| `ho-sweep-20260804` | 과거 handover 실험 코드 |

이 실험들은 데이터 생성 방식과 학습 조건이 다르므로 하나의 학습 세트나 성능 수치로 자동 병합하지 않습니다.
과거 DRL 코드를 새 checkout에서 사용할 때는 `python scripts/setup_study_outputs.py`로
출력 디렉터리와 호환 링크를 준비합니다. 이 명령은 기존 체크포인트를 생성하거나 대체하지 않습니다.

## 설치와 빌드

기준 환경은 Linux/WSL2 Ubuntu 24.04, Python 3.12, GCC 13입니다.
Python 패키지 버전은 [requirements.txt](requirements.txt)에 고정돼 있습니다.
WSL에서는 저장소를 Linux 파일시스템에 두고 실행하세요.

```bash
git clone https://github.com/gksyb4235/sionna-LENA-oran-flexric-smo.git
cd sionna-LENA-oran-flexric-smo

sudo apt update
sudo apt install -y build-essential gcc-13 g++-13 cmake ninja-build ccache \
  git pkg-config python3.12-dev python3.12-venv llvm \
  libboost-all-dev libgsl-dev libgtk-3-dev libpcre2-dev \
  libsctp-dev lksctp-tools libsqlite3-dev libxml2-dev \
  autoconf automake libtool bison flex

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

이 빌드는 e2sim 개발 라이브러리를 필요로 합니다. 외부 의존성은 저장소 밖에 준비합니다.

```bash
git clone https://github.com/MinaYonan123/e2sim-kpmv3.git ../e2sim-kpmv3
git -C ../e2sim-kpmv3 checkout --detach acf4f6b2baa8c645af566ea210146abd97de1f48
(
  cd ../e2sim-kpmv3/e2sim
  mkdir -p build
  ./build_e2sim.sh 2
)
test -f /usr/local/lib/libe2sim.a
test -d /usr/local/include/e2sim

python scripts/check_environment.py --require-e2
./ns3 configure --enable-examples --enable-tests --enable-python-bindings
./ns3 build khu-real-dual-nr-sionna-final -j 2
```

`git pull`은 실행 파일을 재빌드하지 않습니다. C++ 변경 뒤에는 사용하는 타깃을 다시 빌드합니다.
conflict backend는 `studies/conflict_scenarios_500/sweep/backend/reviewed-base.cc`의
고정 해시 원본을 보존합니다. 현재 scratch 시나리오가 달라졌어도 사용자 수정은 건드리지 않고
검증된 원본으로 격리 타깃을 생성합니다. 기존 소유권 marker의 `base_source`는 이 원본의
과거 출처를 뜻하며, 실제 적용 코드는 marker의 `files_sha256`으로 검증합니다.
기존 HDF5 캐시를 사용하는 데이터 수집에는 새 ray tracing이나 RIC 실행이 필요하지 않습니다.
새 캐시 생성과 GPU/GUI 구성은 별도 작업입니다. 기존 GPU/WSL 설치 기록은
[이전 설치 기록](doc/project-archive/README.before-layout.txt)에 보존했습니다.

## 공통 실험 실행

모든 명령은 저장소 루트에서 가상환경을 활성화한 후 실행합니다.

```bash
source .venv/bin/activate
python scripts/run_experiments.py KIND FIRST LAST CPUS_PER_JOB PARALLEL_JOBS
```

| KIND | 번호 | 입력 | 변경 방식 |
|---|---|---|---|
| `fixed` | CSV에 존재하는 0-based `gnn_*` 번호 | `datasets/gnn_dataset_params.csv` | 실행 내 고정 |
| `conflict` | 1~500 또는 501~550 | 해당 구간의 frozen manifest | 180/540초 또는 180/360/540/720초 변경 |

번호는 양끝을 포함합니다. 같은 501이어도 `fixed`와 `conflict`는 다른 실험입니다.
1~500과 501~550을 한 명령으로 섞지 않습니다. PC마다 겹치지 않는 구간을 배정합니다.

### 고정 파라미터 데이터

```bash
# 실행 없이 범위와 배정을 확인
python scripts/run_experiments.py fixed 0 99 1 12 --dry-run

# 100개 실험, 작업당 논리 CPU 1개, 최대 12개 동시 실행
python scripts/run_experiments.py fixed 0 99 1 12
```

공통 진입점의 기본 KPI 간격은 1초이고 InfluxDB/live-tail은 끕니다.
기존 `experiment/khu/tools/run_gnn_dataset_sweep.sh` 직접 실행 방식과
`GNN_SWEEP_*` 환경변수는 유지합니다. 직접 실행의 기존 10초 KPI 기본값을 쓸 경우
1초 실험과 동일 조건이라고 보고하지 마세요. 설정과 성공/실패는 `run_status.json`으로 남깁니다.
실패한 기존 실행은 삭제하지 않고 보관한 뒤 재시도합니다.

### 시간별 변경 파라미터 데이터

```bash
# 준비/빌드. 시뮬레이션은 시작하지 않음
python scripts/conflict_sweep.py setup --build

# 예비 501~512, 12개를 CPU 1개씩 실행
python scripts/run_experiments.py conflict 501 512 1 12 --dry-run
python scripts/run_experiments.py conflict 501 512 1 12 --mode preflight
python scripts/run_experiments.py conflict 501 512 1 12

# 같은 범위 재개: 검증된 성공은 건너뛰고, 실패 시도는 보존
python scripts/run_experiments.py conflict 501 512 1 12 --retry-failed

# 진행/결과 검증
python scripts/run_experiments.py conflict 501 512 1 12 --mode status
python scripts/run_experiments.py conflict 501 512 1 12 --mode collect \
  --report experiment/khu/results/verified_501_512.json
```

기존 네 인자 명령도 유지됩니다.

```bash
./experiment/khu/tools/run_conflict_dataset_sweep.sh 501 512 1 12
```

conflict는 실제 Agent/DRL/RIC을 돌리는 것이 아니라 사전 정의한 설정을 ns-3 내부에서 예약 적용합니다.
원본 500개와 예비 50개는 별도 결과 루트를 사용합니다. 기존 로컬 설정은 덮어쓰지 않으므로
실제 저장 위치는 시작 시 출력되는 `output_root`를 확인하세요.
900초/1초 KPI, 파라미터 적용 readback, cache/trace 해시 및 성공 여부를 검증합니다.
예비 501~550은 기존 seed1000~1049 이동 경로를 재사용하므로 신규 독립 trace 50개가 아닙니다.

작업 하나가 끝나면 다음 작업이 빈 CPU 배정에 들어갑니다.
conflict의 초기 시작 간격과 메모리 여유 검사 때문에 잠시 기다릴 수 있습니다.
`1 12`는 논리 CPU 12개에 최대 12개 작업을 배정한다는 의미이며 CPU 사용률 1200% 보장은 아닙니다.
GPU, RAM, 디스크 여유가 다르면 병렬 수를 줄이세요. 기존 프로세스에 스크립트 변경은 자동 적용되지 않습니다.

### 결과 위치

새 실행의 기본 위치는 `experiment/khu/results/` 아래입니다.

- `fixed/`: 고정 파라미터 데이터와 실패 보관본
- `conflict/`: 원본/예비 변경 실험, attempt별 KPI·명령·상태·적용 기록
- `parameter_sweeps/`: HYS/TTT/TxP/CIO/RET 진단 sweep
- `evaluation/`: 원시 결과에서 계산한 집계 보고서
- `agora_runtime/`: 서비스 상태, RIC 제어 로그, 실시간 시뮬레이션
- `legacy_pooled/`, 기존 이름의 결과 폴더: 이전 실험 보존본

다른 PC에서 `git pull`해도 로컬 KPI/로그/학습 결과까지 내려오지는 않습니다.
검증된 collect 보고서와 원시 artifact를 함께 전달하고 해시를 확인하세요.
기존 결과를 새 설정으로 덮어쓰거나 실패한 시도를 성공 사례로 교체하지 않습니다.

## 성능평가 기준

1. 데이터 생성 목적, fixed/scheduled 구분, trace seed, RNG seed/run, 코드 commit,
   dirty 상태, 바이너리·입력 해시, KPI 간격과 변경 일정을 기록합니다.
2. 학습/검증/평가를 trace 단위로 분리합니다. 같은 trace의 다른 파라미터나 예비 시나리오가
   서로 다른 split에 들어가면 누수가 발생할 수 있습니다.
3. 비교안은 동일 trace·초기 조건·관측 구간을 사용합니다. 성공률과 실패 사유를 함께 보고하며
   성공/좋은 사례만 골라 평균을 내지 않습니다.
4. fixed와 scheduled를 따로 집계합니다. 60/180초 구간은 기록된 `interval_s` 기반이며
   변경 직전/직후 구간과 전체 900초 결과를 구분합니다.

```bash
# 검증한 변경 실험 결과를 180초 단위로 집계
python scripts/evaluate_kpis.py --experiment-kind scheduled \
  --collection experiment/khu/results/verified_501_512.json \
  --window-seconds 180 \
  --output-dir experiment/khu/results/evaluation/conflict_501_512_v1

# 고정 실행 한 개 집계
python scripts/evaluate_kpis.py --experiment-kind fixed \
  --run experiment/khu/results/fixed/gnn_0000 \
  --window-seconds 60 \
  --output-dir experiment/khu/results/evaluation/fixed_0000_v1
```

`cell_summary.csv`, `cell_windows.csv`, `evaluation.json`이 새 출력 폴더에 생성됩니다.
collect artifact는 해시를 다시 확인합니다. 고정 실행의 성공 receipt만 있는 경우는
`completion_receipt_only`로 구분합니다. 과거 raw CSV는 `--include-unverified`로
명시할 때만 포함하며 실패 receipt가 있는 실행은 이 옵션으로도 포함하지 않습니다.

| 지표 | 집계식 |
|---|---|
| Cell goodput (Mbit/s) | `8 * sum(rx_bytes_interval) / (1e6 * sum(interval_s))` |
| Byte delivery ratio | `sum(rx_bytes_interval) / sum(tx_bytes_interval)` |
| PRB 사용률 | `sum(prb_used_reg) / sum(prb_capacity_reg)` |
| 평균 지연 (ms) | `sum(delay_sum_ms_interval) / sum(delay_samples_interval)` |
| HO 성공률 / ping-pong 비율 | 각각의 interval 건수 합 / 대응하는 시도·성공 건수 합 |
| 에너지 효율 (Mbit/J) | `8 * sum(rx_bytes_interval) / (1e6 * sum(interval_energy_j))` |

분모가 0이거나 필드가 없으면 null입니다. 누적 카운터를 매 초 더하지 않습니다.
구간 송수신량 비율은 같은 패킷 집단의 손실률이 아니며, HO/패킷이 구간 경계를 넘을 수 있습니다.
지연 P95/P99나 RSRP/SINR P5의 초별 값들을 평균 내어 전체 percentile이라고 부르지 않습니다.
그런 통계에는 원시 표본/히스토그램이 필요합니다. 에너지는 BS 모델 계산값이지 실제 PC 소비전력이 아닙니다.

집계기는 행의 `interval_s`를 그대로 사용합니다. 최초 KPI가 0.5초이고 interval이 1초이면
명목 시작점은 -0.5초입니다. 따라서 이를 정확한 패킷 발생 시각 기준 `[0,60)` 집계라고
주장하지 않습니다. 경계를 가로지르는 행을 임의로 분할하지 않습니다.
UE 위치와 활성 logical UE는 원시 `ue_kpi.csv`에 보존되며, 현재 공통 집계기는 cell 지표를 계산합니다.

## RIC / 모니터링 / AGORA

일반 데이터 수집에는 아래 서비스를 시작할 필요가 없습니다.
`non-rt-ric/A1_Mediator`가 `non-rt-ric/non_rt_ric` 패키지를 로드해 한 서버로 실행합니다.
SMO는 GNN/Agent의 추천과 공통 제어 계약을 제공하고, Non-RT 정책 API와 저장 로직은
`non-rt-ric`에서 관리합니다. 실행 게이트웨이와의 통신은 기존 HTTP API를 유지합니다.

```bash
FLEXRIC_BUILD_JOBS=4 ./near-rt-ric/build-local.sh
docker compose --project-directory near-rt-ric/monitoring up -d
python scripts/agora.py up
python scripts/agora.py status
```

AGORA UI는 `http://127.0.0.1:18081`, 게이트웨이는 `:18082`, Grafana는 `:3000`,
InfluxDB는 `:8086`, Non-RT A1 API는 `:9000/docs`를 사용합니다.
실제 후보 적용은 운영자 선택 후 RIC/E2로 전송하며 ns-3 readback으로 적용/복원을 확인합니다.
연결 테스트나 적용 성공 자체가 성능 개선을 뜻하지 않습니다.

```bash
python scripts/agora.py simulate --from-session SESSION_ID
python scripts/agora_replay.py verify
```

재생 NPZ와 과거 학습 산출물은 모두 Git으로 배포되는 것이 아닙니다.
다른 PC에 재생 데이터를 전달할 때는 `scripts/agora_replay.py export/import`를 사용합니다.
예약 시각은 시뮬레이션 시간이며 Grafana의 벽시계 수신 시간과 구분합니다.
[SMO 구성](smo/README.md), [Near-RT 로컬 빌드](near-rt-ric/LOCAL_NS3_SETUP.md),
[CPU-only Docker 배포](deploy/conflict-sweep/README.md)를 참고하세요.

## Git 관리와 정리 원칙

소스, 테스트, frozen 실험 설계, 재현에 필요한 기본 입력은 버전 관리합니다.
`.venv`, build, DB/서비스 상태, 신규 KPI/로그, 재생성 가능한 혼합 캐시는 제외합니다.
이미 추적되던 일부 과거 결과는 삭제 승인 전까지 보존합니다.
큰 HDF5의 배포 방식(LFS 또는 외부 artifact)은 별도 결정 사항이며 기존 Git 이력은 재작성하지 않았습니다.

루트의 AGORA/SWEEP 설명은 이 README로 통합하고 원문을 `doc/project-archive/*.txt`에 보존했습니다.
작업 지침은 [.agents/AGENTS.md](.agents/AGENTS.md)로 이동했습니다. 코드 수정 전에 이 지침을 확인하세요.
이전 경로에서 사용자 파일이 남아 있는 checkout은 `python scripts/reorganize_khu.py`로
이동 계획을 확인할 수 있습니다. `--apply`는 기존 파일을 덮어쓰지 않고 옮긴 뒤 호환 링크를 남깁니다.
동일한 목적지 파일이 이미 있으면 자동 병합하지 않고 중단합니다.

주요 검증 명령:

```bash
python -m unittest discover -s scripts -p 'test_*.py'
python -m unittest discover -s experiment/khu/studies/conflict_scenarios_500/sweep/tests
python scripts/run_experiments.py conflict 501 512 1 12 --dry-run
python scripts/run_experiments.py fixed 0 11 1 12 --dry-run
```

폴더 정리는 원시 결과 삭제나 ns-3 계산 모델 변경을 뜻하지 않습니다.
cache/source 변경 뒤의 성능 재현은 별도의 smoke 및 전체 실행 검증이 필요합니다.

## 라이선스

ns-3 본체는 GPL-2.0-only입니다. [LICENSE](LICENSE), [LICENSES](LICENSES) 및
각 vendored 구성 요소의 라이선스를 유지합니다. upstream ns-3 설명은 [doc](doc)에 있습니다.
