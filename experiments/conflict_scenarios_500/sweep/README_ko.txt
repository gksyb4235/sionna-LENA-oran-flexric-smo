AGORA 500개 고정 설정·다중 변경 시나리오 분산 sweep
===================================================

이 묶음은 이미 정한 파라미터를 ns3의 지정 시각에 적용한다.
실제 xApp/DRL, LLM, RIC, GUI, InfluxDB, Grafana는 실행하지 않는다.
새 원본 데이터 수집용이며, 모든 후보에서 충돌이 발생한다고 보장하지 않는다.

1. 입력과 해석
--------------
data/conflict_scenario_params_reviewed_500.csv: 500개 전체 후보와 사전 검토 등급.
data/execution_manifest.json: 실제 실행에 쓰는 고정 JSON. 최초 설정 11개,
  이후 180·540초 또는 180·360·540·720초 변경, 이동 경로를 담고 있다.
data/pilot_ids.json: 첫 앱→다음 앱 순서쌍 20종을 대표하는 예비 실행 20개.

A: 모든 후속 변경에 목표 KPI 개선 근거가 있고, 하나 이상 상충 근거가 있음.
B: 상충 근거가 있지만 일부 요청의 목표 개선 근거가 약함.
C: 일부 목표 개선 근거만 있고 상충 근거는 부족함.
D: 현재 예측 기반 근거가 약함.
이 등급은 사전 모델 검토이며 실제 KPI 측정/충돌 확률/정답이 아니다.
일부 A만 골라 최종 논문 성능을 제시하면 선택 편향이 생긴다.
전체 500개는 보존하고 실행 순서만 screening_priority로 정렬했다.

초기 상태는 0초부터 사용한다. 이후 이벤트는 바뀌는 필드만 적용하며
다음 이벤트가 덮어쓰기 전까지 유지된다. 자동 복원은 없다.
한 이벤트의 여러 필드는 동일 ns3 콜백에서 적용한다.
공통 TTT/HYS는 세 기지국에 적용된다. 4g1/4g2는 1.8GHz NR 셀 이름이다.

이동 경로는 원본 CSV 행 순서에 따라 seed1000..1499로 고정했다.
이 번호들이 과거 모든 실험에서 미사용이었다고 주장하지 않는다.
향후 학습/평가 분할은 실행·이동 경로·시나리오 전체 단위로 해야 한다.
같은 실행의 서로 다른 시간 창을 train/test에 나누면 누출 위험이 있다.

2. 다른 컴퓨터로 가져갈 것
--------------------------
각 Linux/WSL 컴퓨터에는 동일한 수정 ns3 저장소와 Python 환경이 필요하다.
다른 PC의 바이너리를 그대로 복사하기보다 각 PC에서 같은 소스로 빌드한다.
다음 자료를 가져간다:
  - 이 sweep_bundle 폴더 전체 (개인 API 키/LLM 모델 불필요)
  - 기존 LENA-oran-flexric-smo 저장소의 동일 수정 버전
  - scenarios/khu-real/gnbs-ret.csv
  - scenarios/khu-real/ue_position/ue_positions_seed1000.csv .. seed1499.csv
  - scenarios/khu-real/ret_caches/ 전체: tilt_3/5/7/9/12deg의 두 대역,
    merged_percell 디렉터리 포함. HDF5 외부 링크의 상대 경로를 유지한다.
캐시는 묶음에 중복 포함하지 않는다. 기존 테스트베드 데이터를 복사해야 한다.
runner는 표준 라이브러리만 사용하지만 ns3의 embedded Python 및 캐시 검사는
해당 .venv의 h5py/numpy가 필요하다. CPU·Python ABI가 맞아야 한다.

3. 설정과 독립 backend 빌드
---------------------------
이하 명령은 Linux/WSL에서 sweep_bundle을 현재 디렉터리로 두고 실행한다.

  cp config.example.json config.json

config.json의 ns3_root를 해당 PC 경로로 바꾼다.
python_site_packages와 python_executable도 그 PC의 venv 버전에 맞춘다.
output_root는 ns3_root 내부의 새 결과 경로를 정한다.
다른 실험의 결과 디렉터리를 사용하지 않는다.

  python3 backend/install_backend.py --repo /home/user/LENA-oran-flexric-smo
  python3 backend/install_backend.py --repo /home/user/LENA-oran-flexric-smo --install

ns3 저장소 디렉터리에서:

  ./ns3 build khu-agora-conflict-sweep -j 2

설치기는 원래 시나리오와 공유 NR/channel 소스를 수정하지 않는다.
일치하는 원본 해시와 필요한 제어 API가 없으면 중단한다.
기존 독립 대상의 수동 수정도 덮어쓰지 않는다.
UseIdealRrc=true를 강제한다. TTT 340/720ms는 Real RRC에서 지원하지 않는
연구용 제어 범위이므로, 이 실행을 Real RRC 기반 표준 TTT 검증으로 해석하지 않는다.

4. 9대 분할과 예비 실행
-----------------------
PC별 shard-index를 0..8로 하나씩 배정하고, 모두 shard-count=9를 사용한다.
전체 순위 i인 작업은 i % 9인 PC에 배정된다. 전체는 56개×5대+55개×4대다.
20개 예비 실행의 PC별 개수는 5,2,2,4,0,3,2,1,1개다. PC4는 예비 실행이 없다.
workers는 그 PC의 동시 실행 상한이다. 아래는 첫 PC에서 16개 상한의 예다.
다른 PC에서는 shard-index만 자기 번호로 바꾼다.

  python3 sweep.py plan --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --pilot20 --report pilot_plan_pc0.json
  python3 sweep.py preflight --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --workers 16 --pilot20 --report pilot_preflight_pc0.json

위 명령은 시뮬레이션을 시작하지 않는다. preflight는 바이너리/라이브러리 ABI,
소스/캐시/이동 경로 SHA256, 정확한 RET 캐시 메타데이터·외부 링크,
디스크와 메모리 계획을 확인한다. 전체 캐시 해시 계산에는 시간이 걸릴 수 있다.

실제 900초 예비 실행을 시작하려면:

  python3 sweep.py run --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --workers 16 --pilot20 --resume

각 PC에서 완료 후 다음으로 결과를 검증한다:

  python3 sweep.py collect --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --pilot20 --report pilot_verified_pc0.json

처음부터 한 PC에서 20개 예비 실행을 모두 하려면 별도 output_root와 shard-count=1을
사용할 수 있다. 하지만 본 실행에 shard-count=9로 바꿀 때 같은 output_root를
재사용할 수 없다. 동일 9분할을 유지하면 예비 실행 결과를 그대로 재사용한다.

5. 전체 실행, 중단과 재개
-------------------------
예비 실행에서 실제 목표 개선·상충과 메모리 사용을 검토한 후 --pilot20만 뺀다.

  python3 sweep.py preflight --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --workers 16 --report full_preflight_pc0.json
  python3 sweep.py run --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --workers 16 --resume

완료한 작업은 결과를 다시 검증한 후 건너뛴다. 실패한 작업은 자동으로 덮어쓰지 않는다.
실패 원인을 확인한 뒤 보존된 기존 시도와 별도로 재시도하려면:

  python3 sweep.py run --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --workers 16 --resume --retry-failed

--max-jobs 2는 현재 선택된 shard의 상위 두 작업만 처리한다.
--tier A는 사전 검토 A등급만 선택한다(파일 자체를 바꾸지 않음).
분할 기준은 필터링 전에 고정하므로 pilot/tier 필터에 따라 PC가 바뀌지 않는다.
수집 다양성을 위해 전체 실행 기본값은 --tier all이다.

Ctrl+C/SIGTERM은 이번 runner가 시작한 프로세스 그룹만 종료한다.
killall, pkill ns3처럼 다른 실험까지 종료하는 명령을 사용하지 않는다.
동일 결과·동일 shard에는 한 runner만 접근하도록 파일 잠금을 사용한다.
하드 종료 후 이전 PID가 살아 있으면 중복 실행을 막고 수동 확인을 요청한다.
소스·바이너리·입력 해시, manifest, 설정 또는 shard-count가 달라지면 같은
결과 경로의 재개를 거절한다. 변경 실험은 새 output_root로 실행한다.

메모리 예약 기본 4GiB + 새 작업 1.75GiB, 출력 크기 추정은 실행당 0.25GiB다.
이 값은 안전 상한/측정 보장이 아니다. 첫 실행의 RSS·실제 파일 크기를 보고
새 실험 설정을 조정한다. 메모리 부족 시 workers 한도보다 적게 실행한다.
OpenMP/MKL/OpenBLAS 등 계산 스레드는 실행당 1개로 제한한다.

6. 상태와 결과 취합
-------------------
  python3 sweep.py status --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9
  python3 sweep.py collect --manifest data/execution_manifest.json --config config.json --shard-index 0 --shard-count 9 --report verified_pc0.json

주요 결과 위치:
  <ns3_root>/<output_root>/jobs/conflict_XXXX/attempt0001/
    job.json, parameter_schedule.csv, execution.json, stdout.log, stderr.log
    data/conflict_XXXX/cell_kpi.csv
    data/conflict_XXXX/ue_kpi.csv
    data/conflict_XXXX/parameter_changes.csv

마지막 KPI 시간은 899.5초가 정상이다. 3셀×900개=2700행 전체 격자,
각 행의 11개 파라미터 상태, 모든 변경의 지정/실제 적용 시각과 값,
정상 종료와 비어 있지 않은 UE 파일을 모두 확인해야 complete가 된다.
메타데이터와 원본 결과는 각 PC별 디렉터리로 복사해 보존한다.
검증 보고서만 중앙 PC에 모아 확인하려면:

  python3 combine_reports.py --manifest data/execution_manifest.json --reports verified_pc0.json verified_pc1.json verified_pc2.json verified_pc3.json verified_pc4.json verified_pc5.json verified_pc6.json verified_pc7.json verified_pc8.json --output combined_verified_500.json

이 명령은 중복/미등록 작업·manifest 불일치를 거절하며 원본 데이터를 이동하지 않는다.
PC 간 겹치는 소스·캐시·이동 경로·gNB 위치 파일의 해시가 다르면 취합을 거절한다.
PC별 컴파일 바이너리와 공유 라이브러리는 해시를 기록하지만 동일 바이트를 요구하지 않는다.
실행 전 입력 해시 기록이 없으면 data_review_required로 표시하고 최종 검증 완료로 보지 않는다.
기존 --report/--output 파일을 덮어쓰지 않는다. 재검증에는 새 파일명을 사용한다.

7. 평가 시 주의
---------------
한 경로에서 변경 전후만 비교하면 시간대별 UE/트래픽 변화가 섞인다.
실제 충돌 효과와 mitigation 개선량은 이후 동일 trace/초기 상태의
무변경·원안·완화안 분기 실행을 추가해서 평가해야 한다.
이 500개는 변화 상황 학습/탐색 데이터 수집이며 완성된 인과 성능평가가 아니다.
180.5초 같은 경계 KPI 한 행은 변경 전후 측정 구간이 섞일 수 있다.
시간 구간 겹침 규칙을 적용하거나 변경 경계 행을 제외한다.

검증 명령(실제 무선 시뮬레이션 아님):
  python3 -m unittest discover -s tests -v
backend/의 smoke_report.json은 별도의 짧은 실제 backend 검증 결과다.
900초 전체 후보 실행 여부와 혼동하지 않는다.
