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
