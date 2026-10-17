# 900초 ns3 파라미터 변경 실험

이 실험은 사전에 고정한 설정을 ns3 시뮬레이션 시각 **180초와 540초**에 한 번씩 변경한다. 판단 모델이나 실행 중 에이전트 없이, 변경된 네트워크의 실제 KPI를 수집한다. 변경값은 유지되며 자동 복구하지 않는다.

## 실험 구성

9개 이동 trace마다 두 실행을 짝지어 총 18개를 실행한다. 쌍 안에서는 초기 설정, 이동 trace, RNG 설정, 변경 대상 셀과 값, 트래픽 설정이 같다.

| 시뮬레이션 구간 | TxP 먼저 변경 | RET 먼저 변경 |
|---|---|---|
| 0–180초 | 초기 설정 | 초기 설정 |
| 180–540초 | 한 셀의 TxP 감소 | 한 셀의 RET 증가 |
| 540–900초 | 기존 TxP 감소 유지 + RET 증가 | 기존 RET 증가 유지 + TxP 감소 |

최종 파라미터는 쌍 안에서 같지만, 네트워크 상태는 앞선 동작 순서에 따라 달라질 수 있다. 변경 시각은 wall-clock이 아닌 **ns3 simulation time**이다.

## 초기 설정

| 제어값 | 5g | 4g1 | 4g2 |
|---|---:|---:|---:|
| TxP, dBm | 43 | 43 | 43 |
| RET, ° | 9 | 9 | 9 |
| CIO, dB | 6 | 0 | −1 |

TTT는 모든 셀 512 ms, HYS는 1 dB이다. 셀 이름은 기존 데이터 식별자이며 4g1/4g2도 실제 시뮬레이터에서는 1.8 GHz NR 셀이다.

초기값은 API `gnn_0197` / 원본 `Final_Dataset/gnn_0184`의 학습 실행에서 관측한 설정이다. 미래 KPI를 사용하지 않고 `[30,180)` 관측값으로 초기 후보를 선정했다. 이 구간의 셀별 평균 PRB는 21.156/20.280/13.361%, 이 평균들의 Jain 지수는 **0.964846**이다. 변경 직전 `[150,180)` 구간평균 Jain은 **0.978520**이었다.

시간 평균의 균형과 순간 균형은 다르다. `[30,180)`의 초별 Jain 평균은 **0.898337**이다. 새로운 이동 trace에서도 같은 균형이 보장되지는 않으므로 각 새 실행의 변경 전 KPI를 확인해야 한다. 이 설정은 Load Balancing 앱이 유지하는 상태로 가정하며, 실제 xApp 제어 루프를 실행하지 않는다.

근거: [초기 상태 감사](initial_state_audit.json), [재현 코드](audit_initial_state.py).

## 사전에 고정한 변경표

RET는 선택된 셀 하나만 9°에서 **12°**로 높인다. 요청을 각도 증가로 해석했으며, SINR이나 지리적 커버리지 개선을 보장하지 않는다. 28° 캐시는 없고 이번 실험은 존재하는 12° 캐시를 사용한다.

| 쌍 | 이동 trace seed | TxP 변경 셀 | TxP 변경, dBm | RET 변경 셀 |
|---|---:|---|---|---|
| 01 | 1405 | 5g | 43→37 | 4g1 |
| 02 | 1406 | 5g | 43→33 | 4g2 |
| 03 | 1407 | 4g1 | 43→40 | 4g2 |
| 04 | 1408 | 4g1 | 43→33 | 4g1 |
| 05 | 1409 | 5g | 43→40 | 5g |
| 06 | 1410 | 4g2 | 43→33 | 5g |
| 07 | 1411 | 4g2 | 43→40 | 4g1 |
| 08 | 1412 | 4g2 | 43→37 | 4g2 |
| 09 | 1413 | 4g1 | 43→37 | 5g |

9가지 셀 조합을 고정 난수 seed `20261015`로 섞어 이동 trace에 배정했다. 각 TxP 셀은 40/37/33 dBm 목표를 한 번씩 갖는다. 위 각 행을 두 순서로 실행하므로 총 18개다. 상세 입력·구간별 전체 11개 파라미터는 [manifest.json](manifest.json)에 고정했다.

선택 trace는 기존 주 데이터, reserve 데이터, 앞선 1401–1404 개입 실험과 겹치지 않는다. [캐시·이동 경로 감사](cache_trace_audit.json)는 1405–1422를 포함하는 더 넓은 후보 집합을 조사했다. 해당 집합에서 실제 lookup 규칙의 위치 오차 최대는 약 1.344 m이며, 격자 키가 없는 0.060%의 표본은 nearest 조회를 사용했다. 실제 사용 trace는 1405–1413이다. 각도별 HDF5의 위치 격자도 대조했다.

## 실행 방식과 출력

대상 저장소: `/home/user/LENA-oran-flexric-smo`

시나리오: `scratch/khu-real-dual-nr-sionna-final.cc`. 기존 1,000개 데이터와 같은 dual-band 계열을 사용한다. 각 실행은 다음 CSV를 시작 시 읽는다.

```csv
time_s,parameter,value
180,txp_5g,37
540,ret_4g1,12
```

CLI는 `--parameterSchedule=<CSV 경로>`와 `--parameterScheduleRetCacheRoot=<캐시 루트>`이다. 시뮬레이터가 `Simulator::Schedule`로 해당 시점에 영구 변경을 적용한다. RET는 안테나 각도뿐 아니라 대상 셀의 전파 채널 캐시도 전환한다. 지원하지 않는 캐시를 다른 각도로 대신 사용하지 않는다.

실행기 설치 후 WSL 저장소 루트에서 사용하는 명령:

```bash
.venv/bin/python scripts/two_switch_batch.py preflight --manifest experiments/two_switch_900s/manifest.json --workers 18
.venv/bin/python scripts/two_switch_batch.py run --manifest experiments/two_switch_900s/manifest.json --workers 18
.venv/bin/python scripts/two_switch_batch.py status --manifest experiments/two_switch_900s/manifest.json
.venv/bin/python scripts/two_switch_batch.py verify --manifest experiments/two_switch_900s/manifest.json
```

`run`은 실제 장시간 작업을 시작한다. 현재 결과 디렉터리를 덮어쓰는 재실행은 거부한다. 실행 중인지 먼저 `status`로 확인한다.

**2026-10-15에 본 배치 실행을 시작했다.** 다시 `run`을 실행하지 말고 상태를 조회한다. Windows의 현재 데이터셋 폴더에서도 다음 명령으로 조회할 수 있다.

```powershell
.\experiments\two_switch_900s\status.ps1
```

[launch_record.json](launch_record.json)은 실제 시작 기록이고, [latest_status.json](latest_status.json)은 마지막으로 조회한 상태의 사본이다. 실시간 파일은 WSL 결과 폴더의 `batch_status.json`이다. 900초 완료 여부는 실제 실행 상태와 완료 검증 결과로 판단해야 한다.

기본 결과 위치는 `scenarios/results/two_switch_900s_20261015/`이다. 작업별 원본 `cell_kpi.csv`, `ue_kpi.csv`, `parameter_changes.csv`와 실행기의 입력·해시·PID·시간 기록을 분리해 보관한다. 파라미터 변경 로그에는 예정 시각, 실제 적용 시각, 변경 전후 값과 readback 성공 여부를 남긴다. 완료 검증은 900초 전체 KPI, 정확히 두 번의 변경, 세 구간의 누적 설정을 확인한다.

최대 동시 실행은 18개이며, 기본적으로 가용 메모리 3 GiB를 남기고 시작 간격을 둔다. 실제 동시 실행 수는 메모리 상황에 따라 작을 수 있다. 이전 실행의 peak RSS는 대략 1.0–1.75 GiB/작업이므로 18개 실행은 CPU뿐 아니라 RAM 제한도 고려해야 한다. 각 작업의 완료 시간과 전체 묶음의 완료 시간을 따로 기록한다.

정확한 기본 투입 조건은 현재 가용 RAM이 `3 + 1.75 = 4.75 GiB` 이상인 경우다. 1.75 GiB는 신규 작업의 예상 사용량이며, 실행 도중 사용량 증가까지 보장하는 강제 메모리 제한은 아니다. 기본 시작 간격은 10초, 개별 작업 제한 시간은 48시간이다. SIGINT/SIGTERM 또는 시간 초과 시 실행기가 자신이 시작한 작업 그룹을 정리한다.

사전 검증은 실행표 검사 4개, 병렬 실행기 검사 20개, 실제 ns3 및 잘못된 입력 검사 15개가 통과했다. 짧은 실제 시나리오에서는 세 셀 모두의 TxP/RET 변경과 정순·역순의 누적 설정을 확인했다. [ns3_smoke_report.json](ns3_smoke_report.json)에 실제 명령, 적용 로그, 소스 해시와 검증 결과를 보관했다. 이는 기능 검증이며 18개 900초 실험의 성능 결과를 의미하지 않는다.

## 초기 실패와 동일 입력 재실행

`pair07_seed1411_txp_then_ret`의 첫 시도는 약 20초에서 SIGSEGV(종료 코드 -11)로 종료됐다. 첫 예약 변경인 180초 이전이었으며, 원인은 아직 특정하지 못했다. 동일 입력에 진단용 backtrace 수집만 추가한 별도 실행은 35초까지 정상 진행한 뒤 의도적으로 종료했다. 원래 실패 전 KPI 60행의 모든 필드가 재현 실행과 일치했다. 자세한 기록은 [seed1411_crash_diagnostic.json](seed1411_crash_diagnostic.json)에 있다.

원본 실패 폴더는 보존하고, 같은 trace·RNG·초기값·변경값·시각으로 해당 작업만 한 번 다시 시작했다. [retry01_manifest.json](retry01_manifest.json)은 원본 입력 파일 10개의 해시를 고정하며, 실행기 해시도 원본과 일치함을 확인했다. [retry01_launch_record.json](retry01_launch_record.json)에 시작 기록을 남겼다. 운영 소스·바이너리·공유 라이브러리는 재현 점검과 재실행 과정에서 변경하지 않았다.

재실행 결과는 별도 `scenarios/results/two_switch_900s_20261015_retry01/`에 저장한다. `status.ps1`은 본 배치와 재실행을 함께 표시하되 원래 실패도 계속 표시한다. 재실행이 성공하기 전에는 이 실험을 18개 완료로 집계하면 안 된다. 본 배치 자체는 원래 실패 때문에 최종 상태가 `failed`로 남을 수 있으며, 대체 실행의 정상 종료 및 전체 900초 검증을 별도로 확인해야 한다.

```bash
.venv/bin/python scripts/two_switch_batch.py verify --manifest experiments/two_switch_900s/retry01_manifest.json
```

## 결과 해석

1초 KPI를 원본으로 보관하며, 나중에는 `[0,180)`, `[180,360)`, `[360,540)`, `[540,720)`, `[720,900)` 구간으로 집계할 수 있다. 180·540초 경계를 가로질러 서로 다른 설정의 KPI를 평균하지 않는다.

이 18개는 순서가 다른 두 시퀀스의 관측이다. 변화를 주지 않은 동일 trace 대조군이나 한 번만 변경한 대조군은 포함하지 않으므로, 시간에 따른 이동·부하 변화와 개별 행동의 인과 효과를 완전히 분리하는 실험은 아니다. 같은 seed라도 행동 이후의 내부 이벤트·트래픽 난수 소비가 갈라질 수 있다. 이번 데이터는 요청한 충돌 상황과 전환 동작을 관찰하는 데 사용하고, 정식 방법 비교와 단독 효과 검증의 대조군은 별도로 추가할 수 있다.
