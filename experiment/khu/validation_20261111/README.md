# 실행 폴더: 개인 PC 4대 + 컨테이너 서버

PC1이 현재 사용자 컴퓨터입니다. 기존 미완료 16건을 1코어씩 실행하는 작업은 PC1의 첫 사이클에 포함합니다.
PC1~PC4와 서버, 총 다섯 장비에 배정합니다. 서로 다른 폴더를 사용하세요.

| 폴더 | 동시 실행 코어 | 전체 작업 수 | 이미 시작 | 새 실행 목록 | 전체 사이클 |
|---|---:|---:|---:|---:|---:|
| PC1 (현재 PC) | 16 | 96 | 16 | 80 | 6 (기존 1 + 추가 5) |
| PC2 | 22 | 110 | 0 | 110 | 5 |
| PC3 | 14 | 42 | 0 | 42 | 3 |
| PC4 | 12 | 36 | 0 | 36 | 3 |
| Server | 28 | 84 | 0 | 84 | 3 |

각 시뮬레이션에는 정확히 1코어가 할당됩니다.
PC3와 PC4는 코어 수에 비례해 배분하므로 각각 3사이클로 동일합니다.
직전 배분에서 PC3의 28건을 PC4에 12건, PC1에 16건 이동했습니다. PC2와 서버 목록은 유지했습니다.
PC 3사이클 = 서버 2사이클이라는 속도 가정에서 처음부터의 전체 예상 완료 시간은 PC 6사이클입니다.
PC2는 5사이클, 서버는 3사이클(PC 4.5사이클 환산)을 유지합니다.
PC2/서버 배정과 코어 배수를 유지하면서 PC3/PC4의 사이클을 같게 하려면 PC1의 추가 1사이클이 필요합니다.
PC1의 이미 진행한 시간은 남은 완료 시간에서 줄어듭니다. 위 표는 처음부터의 전체 배정량입니다.
실제 개별 시뮬레이션 실행 시간의 편차는 별도로 생길 수 있습니다.

기존 재실행 16건 + 신규 345건 + 제어 전 상태 재검증 6건 = 367건입니다.
모든 코어 수가 짝수여서 367건을 전부 정수 배수로 나눌 수 없습니다.
Case 5의 승인 대조군 rq3_0009를 PC4에 추가하여 전체 368건으로 구성했습니다.
Case 5의 재검증 계획 5건과 같은 PC에서 대조군을 비교하는 목적이며,
독립 표본 또는 추가 AGORA 성공으로 중복 집계하지 않습니다.
PC4에는 Case 3 재검증 rq3_0054도 포함됩니다.

총 449개 고유 궤적 중 기존 81개를 재사용하고,
PC1 96건(기존 16 + 추가 80) + 다른 PC/서버 272건 = 368건을 실행하는 구성입니다.
대상 방법은 Full AGORA ReAct 2/3/4/5, LLM-only, SHAP Greedy이며,
방법별 120조건(총 720조건), 대조군을 합친 매핑은 912항목입니다.
위반을 예측한 LLM-only 계획도 제외하지 않았습니다.

## 개인 PC 사용법

기존 LENA-oran-flexric-smo 저장소에 이 구성 파일이 반영되어 있어야 합니다.
해당 PC에 맞는 폴더 하나만 사용합니다. 예: PC2

```bash
cd ~/LENA-oran-flexric-smo
bash experiment/khu/validation_20261111/PC2/preflight.sh
bash experiment/khu/validation_20261111/PC2/run.sh
bash experiment/khu/validation_20261111/PC2/status.sh
```

사전 검증이 통과한 뒤 run.sh를 실행합니다. 백그라운드 실행 PID와 로그 경로가 출력됩니다.
PC3/PC4에서는 폴더명만 바꿉니다. 병렬 수는 PC1/PC2/PC3/PC4 각각 16/22/14/12로 고정돼 있습니다.
각 폴더의 jobs.json에 정확한 실행 목록이 있습니다.

PC1은 기존 16건의 실행이 끝난 뒤 run.sh를 실행합니다. 기존 재실행 프로세스가 살아 있으면
run.sh가 시작을 거부하므로 16코어를 중복 사용하지 않습니다. 이 스크립트는 추가 80건만 실행하고,
기존 16건은 다시 실행하지 않습니다. PC1/status.sh는 두 출력 폴더의 상태를 나누어 표시합니다.

## 서버 사용법

기존에 전달한 agora-rq3:cpu-v1 이미지를 사용하며 재빌드는 필요 없습니다.
Git 체크아웃의 새 계획과 실행기를 읽기 전용으로 마운트합니다.

```bash
cd ~/LENA-oran-flexric-smo
bash experiment/khu/validation_20261111/Server/preflight.sh
bash experiment/khu/validation_20261111/Server/run.sh
docker logs -f agora-validation-server
```

서버 컨테이너는 28개 작업을 동시에 실행하며 SSH가 끊겨도 계속됩니다.
기본 캐시 경로는 ~/xapp_conflict_experiments/inputs/scenarios/khu-real/ret_caches,
기본 결과 경로는 ~/xapp_conflict_experiments/validation_20261111 입니다.
다르면 RQ3_CACHE_SOURCE, VALIDATION_OUTPUT 환경 변수로 지정합니다.
서버에 할당된 CPU affinity/quota가 28 미만이면 실행기는 오류를 내며 과할당하지 않습니다.

## 결과 보존과 재시작

PC1 기존 16건: experiment/khu/results/rq3_retry_20261111
PC1 추가 80건 및 다른 PC: experiment/khu/results/validation_20261111
서버: 위 VALIDATION_OUTPUT

중단/실패 로그를 확인한 후 같은 run.sh에 --retry-failed를 붙여 재시작할 수 있습니다.
동일 입력·출력 해시의 완료 작업은 건너뛰고, 미완료 이전 시도는 보관합니다.
실패를 자동으로 무제한 재시도하지 않습니다. 원본 rq3_final은 덮어쓰지 않습니다.
프로세스별 주소 공간 상한은 8GiB이며, 미리 할당하는 메모리 용량은 아닙니다.

결과를 가져올 때 각 rq3_XXXX 폴더 전체와 _batches를 장비별로 묶어 보존하세요.
특히 execution.json, cell_kpi.csv, ue_kpi.csv, parameter_changes.csv, run.log가 필요합니다.
제어 전 상태의 동등성과 GNN 예측 대비 관측 KPI는 완료 후 다시 확인해야 합니다.
재실행 결과가 더 좋아 보인다는 이유로 시도를 선택하지 않습니다.

이 패키지는 새 LLM 호출이나 재학습을 수행하지 않습니다.
사용자가 Git 전송 및 각 원격 장비 실행을 담당하는 구성입니다.
