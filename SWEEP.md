# 500-case conflict sweep after git pull

이 실험은 사전 결정한 설정을 ns3 내부에서 예약 적용한다. 실제 Agent/DRL/RIC을
실행하지 않는다. 900초 동안 초기 설정과 180·540초 또는
180·360·540·720초 변경을 재생한다. 500개는 확정 충돌이 아니라 데이터 수집 후보다.

## 다른 PC에서 준비

기존 테스트베드와 `.venv` 준비는 루트 README.md를 따른다. 아래 명령은 저장소
루트에서 실행한다. 저장소 위치와 Python site-packages는 자동으로 찾는다.

```bash
git pull
python3 scripts/conflict_sweep.py setup --build --jobs 2
python3 scripts/conflict_sweep.py preflight --shard-index 0 --shard-count 9 --workers 16 --pilot20
```

setup은 개인별 `experiments/conflict_scenarios_500/sweep/config.local.json`을
만들고, 격리된 `scratch/agora-conflict-sweep` 타깃을 설치한다. 원본 시나리오와
공유 NR 코드를 수정하지 않는다. 이미 있는 설정·캐시는 덮어쓰지 않는다.
Git에 포함된 기본 RET 캐시에서 누락된 20가지 혼합 저대역 캐시를 생성한다.
새 ray tracing은 필요하지 않다. 캐시·site 좌표·라이브러리·500개 이동 경로는
preflight에서 검증한다. **setup/preflight는 시뮬레이션을 시작하지 않는다.**

이미 빌드한 PC도 pull 후 setup/build를 실행한다. 설치기가 원본 또는 생성 타깃의
수동 변경을 발견하면 덮어쓰지 않고 중단한다. 자동 강제 설치로 우회하지 않는다.
TTT 340/720 ms를 그대로 쓰기 위해 Ideal RRC를 사용한다.

## 시작/끝 번호와 CPU 수로 실행하기 (기존 sweep과 같은 형식)

```bash
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 시작번호 끝번호 실행당CPU 동시실행수

# conflict_0001~conflict_0100: 100개, 실행당 논리 CPU 1개, 최대 12개 동시 실행
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 1 100 1 12

# conflict_0101~conflict_0200: 100개, 실행당 논리 CPU 2개, 최대 6개 동시 실행
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 101 200 2 6

# 실행 없이 범위/CPU 할당만 확인
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 1 100 1 12 --dry-run

# 준비·빌드·검사까지만 수행 (시뮬레이션 없음)
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 1 100 1 12 --mode preflight
```

범위는 양끝을 포함하며 **1~500의 시나리오 이름 번호**다. CSV 행 번호,
우선순위, UE trace seed가 아니다. 기존 고정 설정 GNN sweep은 원래
`run_gnn_dataset_sweep.sh`에 그대로 남아 있다. 새 실행의 `500 599`는 유효하지 않다.
범위 안에서는 번호순으로 실행한다. 각 PC에 서로 겹치지 않는 범위를 배정한다.

네 인자 명령은 PC별 설정·혼합 캐시를 준비하고 격리 타깃을 빌드한 뒤 실행한다.
Linux의 허용된 논리 CPU 집합에서 서로 겹치지 않는 슬롯을 배정하고 `taskset`으로
실제 프로세스를 고정한다. 계산 라이브러리 스레드 상한도 실행당 CPU 수로 설정한다.
`실행당CPU × 동시실행수`가 허용 CPU 수를 넘으면 실행 전에 거절한다.
CPU 수는 할당량이며 ns3가 그 수에 비례하여 빨라진다는 보장은 아니다.
가용 RAM에 따라 실제 동시 실행 수는 지정 상한보다 작을 수 있다.

```bash
# 상태 및 결과 검증
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 1 100 1 12 --mode status
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 1 100 1 12 --mode collect --report verified_pc0.json

# 실패 원인 해결 후, 기존 시도 보존하면서 재시도
./scenarios/khu-real/tools/run_conflict_dataset_sweep.sh 1 100 1 12 --retry-failed
```

동일 명령을 다시 실행하면 검증된 완료 실행을 건너뛴다. 범위를 확장해도 완료
작업을 재사용한다. 실패/중단 작업은 위의 명시적 재시도 옵션이 필요하다.
동일 PC·동일 출력 경로에는 runner 하나만 허용한다. 다른 실험·프로그램과의
CPU 독점까지 보장하는 것은 아니므로 한 PC에서 여러 sweep을 겹쳐 실행하지 않는다.

번호 범위 방식의 설정은 `sweep/config.range.local.json`, 결과는
`experiments/agora_conflict_sweep/range_results_v1/jobs/`에 저장한다.
아래 modulo shard 방식과 출력 경로를 분리하므로 두 방식을 혼합하지 않는다.

9대에 500개를 나누는 예시(모두 실행당 1 CPU):

| PC | 시작 | 끝 | 실행 수 | 동시 실행 상한 예시 |
|---|---:|---:|---:|---:|
| 1 | 1 | 56 | 56 | 16 |
| 2 | 57 | 112 | 56 | 12 |
| 3 | 113 | 168 | 56 | 12 |
| 4 | 169 | 224 | 56 | 12 |
| 5 | 225 | 280 | 56 | 12 |
| 6 | 281 | 335 | 55 | 12 |
| 7 | 336 | 390 | 55 | 12 |
| 8 | 391 | 445 | 55 | 12 |
| 9 | 446 | 500 | 55 | 12 |

다른 PC의 실제 CPU/RAM에 맞게 마지막 두 인자를 줄여도 시나리오 정의는 같다.

## 9대에 분배하여 실행

각 PC에 서로 다른 번호 0..8을 부여한다. 전부 `--shard-count 9`를 사용한다.
`--workers`는 해당 PC의 동시 실행 상한이며 가용 메모리가 부족하면 낮게 실행한다.

```bash
# 예비 20개 중 PC 0에 배정된 작업
python3 scripts/conflict_sweep.py run --shard-index 0 --shard-count 9 --workers 16 --pilot20 --resume

# 상태 확인
python3 scripts/conflict_sweep.py status --shard-index 0 --shard-count 9

# 예비 실행의 실제 KPI를 검토한 뒤 전체 수집으로 확장
python3 scripts/conflict_sweep.py run --shard-index 0 --shard-count 9 --workers 16 --resume

# 각 PC에서 검증 보고서 생성
python3 scripts/conflict_sweep.py collect --shard-index 0 --shard-count 9 --report verified_pc0.json
```

다른 PC는 번호와 workers를 바꾼다. 전체 500개는 55~56개씩 배정된다.
pilot을 전체 실행으로 확장해도 담당 PC는 그대로이고, 완료 데이터는 검증 후
재사용한다. 실패 원인을 해결한 뒤에만 `--retry-failed`를 붙인다. 실패 시도와
기존 결과는 보존된다. report 파일은 이미 있으면 덮어쓰지 않는다.

결과는 기본적으로 `experiments/agora_conflict_sweep/results_v1/jobs/`에 저장된다.
`cell_kpi.csv`, `ue_kpi.csv`, `parameter_changes.csv`, 실행 명령과 입력 해시를
함께 보존한다. 취합 시 공통 소스·캐시·이동 경로가 서로 다르면 거절한다.

## 파일 위치

- `experiments/conflict_scenarios_500/sweep/data/`: 고정 실행표, 20개 pilot, 사전 검토
- `experiments/conflict_scenarios_500/sweep/REVIEW_ko.txt`: 실측/GNN 근거와 A~D 등급
- `experiments/conflict_scenarios_500/sweep/README_ko.txt`: 저수준 실행·취합 명령
- `experiments/README.md`: 다른 실험과 소스/산출물 관리 기준

모델 선별에 통과한 사례만 최종 성능으로 제시하면 선택 편향이 생긴다. 유지/원안/
완화안의 동일 trace·공통 이력 분기는 별도 성능평가로 수행한다. 이 500개는
변화 데이터 학습과 시나리오 발굴을 위한 수집이다.

## 전송·보존 범위

Git에는 소스·테스트·고정 설계 데이터가 들어간다. `.venv`, 빌드 바이너리,
개인 API 키, 실행 로그, 원시 KPI, 학습 산출물과 혼합 캐시는 로컬에 보존한다.
기본 RET 캐시와 UE seed1000..1499 CSV는 기존 저장소에 포함되어 있다.
새 컴퓨터에서 로컬 실행 결과까지 git pull로 가져오는 것은 아니다.
