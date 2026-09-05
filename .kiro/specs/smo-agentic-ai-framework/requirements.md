# Requirements Document

## Introduction

본 기능은 `/home/user/LENA-oran-flexric-smo/smo` 에서 동작하는 **SMO Agentic AI Framework** 를 구축한다.

이 프레임워크는 세 가지 축으로 구성된다.

1. **최소 AIMLFW 계층** — ns-3(LENA-oran-flexric) 시뮬레이션이 생성한 `cell_kpi.csv` 결과를 3분(180초) 간격 피처로 집계하고, GNN 모델을 학습·등록·서빙하며, 예측 결과를 Agent 에게 전달하고, 성능 저하 시 재학습을 트리거한다. AIMLFW 참조 구현(Kubernetes/Kubeflow/KServe/InfluxDB/Cassandra)의 전체 스택은 도입하지 않고, 동일한 계층 분리 철학(Data/Feature ≠ Training ≠ Model Management ≠ Inference ≠ RAN Control)만 최소 구현으로 재현한다.
2. **Agentic AI 판단 계층** — 기존 `smo/agentic_ai`(deep-agents 기반 Decomposition / Planning / Probe / Monitoring / Representative 에이전트, MongoDB Knowledge DB, Dashboard, MCP 서버) 위에 GNN 질의 능력을 추가한다. Agent 는 xApp 이 요청한 5개 파라미터(TxP, RET, CIO, HYS, TTT) 조합에 대해 GNN 을 **배치(batch)** 로 프로빙하여, (a) 임계 KPI 위반 여부 판정, (b) 파라미터 간 **간접 충돌(indirect conflict)** 탐지, (c) 900초 시뮬레이션을 5개 시간 스텝(t=0..4)으로 나눈 **시간 구간별 xApp 실행 계획** 수립을 데이터 근거 기반으로 수행한다.
3. **A1 인터페이스 연동** — 판단 결과를 `A1_Mediator_standalone`(FastAPI, O-RAN.WG2.A1AP-R004-v04.03 부분 준수, 포트 9000)의 Policy Type / Policy Instance 로 발행하여 Near-RT RIC(FlexRIC)의 xApp 에 전달한다.

Non-RT RIC 은 SMO 내부에 위치하며, RAN 제어 권한은 Near-RT RIC/xApp(E2)에 남고 본 프레임워크는 정책(A1)과 AI 능력(AIMLFW)만 제공한다.

## Glossary

- **SMO_Framework**: `smo/` 하위에서 동작하는 본 기능의 전체 시스템. Non-RT RIC 과 최소 AIMLFW 를 포함한다.
- **Data_Extractor**: `scenarios/results/**/cell_kpi.csv` 를 읽어 3분 간격 피처 레코드로 집계하는 구성요소.
- **Feature_Group**: 학습 데이터셋의 논리적 정의(피처 이름 목록, 집계 규칙, 집계 주기, 대상 셀 집합, 타깃 KPI 목록)를 담는 선언적 명세.
- **Feature_Store**: 집계된 피처 레코드를 영속 저장하고 Feature_Group 단위로 조회를 제공하는 구성요소.
- **Feature_Record**: 하나의 (interval_index, cell_id) 에 대한 5개 제어 파라미터 값과 집계된 KPI 값의 집합.
- **Control_Parameter**: TxP, RET, CIO, HYS, TTT 중 하나.
- **TxP**: 셀 송신 전력, 단위 dBm, 허용 범위 30~46, 스텝 1.
- **RET**: 원격 전자 틸트 각도, 단위 degree, 허용 범위 0~15, 스텝 1.
- **CIO**: Cell Individual Offset, 단위 dB, 허용 범위 -6.0~+6.0, 스텝 0.5.
- **HYS**: 핸드오버 히스테리시스, 단위 dB, 허용 범위 0.0~10.0, 스텝 0.5.
- **TTT**: Time To Trigger, 단위 ms, 허용 값 집합 {0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120}.
- **Parameter_Set**: 대상 셀 집합 전체에 대해 5개 Control_Parameter 값을 모두 지정한 하나의 완전한 조합.
- **Target_KPI**: GNN 이 예측하는 KPI. 기본 집합은 `cell_goodput_mbps`, `avg_ue_goodput_mbps`, `ue_goodput_p5_mbps`, `sinr_p50_db`, `prb_utilization_pct`, `delay_p95_ms`, `ho_failure_count`, `pingpong_count`, `rlf_count`, `interval_energy_j`.
- **Training_Manager**: 학습 작업의 생성, 실행, 상태 추적, 산출물 등록을 담당하는 제어면 구성요소.
- **Model_Registry**: 모델 이름, 버전, 메타데이터, 평가 지표, 산출물 URI 를 저장하고 조회를 제공하는 구성요소.
- **Model_Storage**: 학습된 모델 산출물 파일을 저장하는 구성요소.
- **Inference_Service**: 등록된 GNN 모델을 로드하여 예측 API 를 제공하는 구성요소.
- **GNN_Model**: 3분 간격 Feature_Record 로 학습되어 Parameter_Set 으로부터 Target_KPI 를 예측하는 그래프 신경망 모델.
- **Batch_Prediction_Request**: 하나의 요청에 다수 Parameter_Set 을 담아 Inference_Service 에 전달하는 예측 요청.
- **GNN_MCP_Server**: Inference_Service 능력을 MCP 도구로 노출하는 서버.
- **KPI_Advisor_Agent**: xApp 파라미터 요청을 받아 GNN 프로빙과 판정, 대안 추천을 수행하는 신규 sub-agent.
- **Probe_Plan**: KPI_Advisor_Agent 가 생성하는 프로빙 명세. 기준 Parameter_Set(baseline)과 변형 Parameter_Set 목록, 변형 규칙, 평가 기준을 포함한다.
- **Baseline_Parameter_Set**: xApp 이 원래 요청한 Parameter_Set.
- **Probe_Variant**: Baseline_Parameter_Set 에서 지정된 Control_Parameter 를 지정된 델타만큼 변형한 Parameter_Set.
- **Threshold_KPI**: 각 Target_KPI 에 대해 정의된 허용 하한 또는 허용 상한 값의 집합. 위반 시 요청은 열화(degrading)로 판정된다.
- **Degradation_Verdict**: `acceptable`, `degrading`, `unknown` 중 하나의 판정 결과.
- **Conflict_Analyzer**: 두 Control_Parameter 가 동일 Target_KPI 에 대해 반대 방향의 유의미한 한계 효과를 갖는지 판정하는 구성요소.
- **Marginal_Effect**: 하나의 Control_Parameter 를 한 스텝 변형했을 때 특정 Target_KPI 의 예측 변화율, 단위 percent, 값 범위 -100.0~100.0. 음수 값은 해당 Control_Parameter 가 해당 Target_KPI 를 악화시키는 방향임을 의미한다.
- **Indirect_Conflict**: 동일 Target_KPI 에 대해 한 Control_Parameter 의 Marginal_Effect 가 +Conflict_Threshold 이상이고 다른 Control_Parameter 의 Marginal_Effect 가 -Conflict_Threshold 이하인 상태.
- **Conflict_Threshold**: Indirect_Conflict 판정 기준값, 단위 percent, 기본값 5.0.
- **Time_Step**: 1초부터 900초까지의 시뮬레이션 구간을 180초 단위로 분할한 구간 인덱스. 값 범위 0~4. Time_Step k 는 초 구간 `180*k < time_s <= 180*(k+1)` 에 대응한다.
- **xApp_Objective**: xApp 의 최적화 목표. 기본 집합은 `energy_saving`, `throughput_maximization`, `mobility_robustness`.
- **Temporal_Scheduler**: Time_Step 별로 실행할 xApp_Objective 를 배정하는 구성요소.
- **Temporal_Plan**: Time_Step 0~4 각각에 대해 xApp_Objective 와 Parameter_Set 을 배정한 계획.
- **Policy_Manager**: Temporal_Plan 과 판정 결과를 A1 Policy Type / Policy Instance 로 변환하고 A1_Mediator 에 발행하는 Non-RT RIC 구성요소.
- **A1_Mediator**: `A1_Mediator_standalone/A1_Mediator/app/main.py` 로 실행되는 기존 FastAPI 서비스.
- **Policy_Type**: A1 Policy Type. `policy_type_id`, `name`, `description`, `create_schema` 를 갖는다.
- **Policy_Instance**: Policy_Type 의 구체 인스턴스. `data` 객체를 갖는다.
- **Retraining_Controller**: 예측 정확도 저하를 감지하여 재학습을 트리거하는 구성요소.
- **Prediction_Error**: 동일 Parameter_Set 에 대한 GNN 예측값과 ns-3 실측 Feature_Record 값 사이의 평균 절대 백분율 오차, 단위 percent.
- **Retraining_Threshold**: Retraining_Controller 가 재학습을 트리거하는 Prediction_Error 기준값, 단위 percent, 기본값 15.0.
- **Evidence_Record**: 판정 근거를 담은 구조화 기록. 사용한 모델 이름과 버전, Probe_Plan, 각 Probe_Variant 의 예측값, 적용된 Threshold_KPI 값, 최종 판정을 포함한다.
- **SMO_Dashboard**: 기존 `smo/agentic_ai/dashboard/server.py` 기반 웹 대시보드.
- **Multi_Agent_Runtime**: 기존 `smo/agentic_ai` 의 Decomposition Agent, Planning Agent, Probe Agent, Monitoring Agent, Representative Agent 및 MongoDB Knowledge DB 로 구성된 실행 환경.

## Requirements

### Requirement 1: KPI 결과 데이터 추출 및 3분 간격 집계

**User Story:** 개발자로서, ns-3 시뮬레이션 결과 CSV 를 GNN 학습에 바로 쓸 수 있는 3분 간격 피처로 변환하고 싶다. 그래야 시뮬레이션 결과와 모델 입력 사이의 수동 가공 단계를 없앨 수 있다.

#### Acceptance Criteria

1. WHEN 존재하고 읽기 가능한 `cell_kpi.csv` 경로와 Feature_Group 정의가 주어지면, THE Data_Extractor SHALL 해당 CSV 의 데이터 행 중 `time_s` 값이 1 이상 900 이하인 모든 행을 읽어 `180*k < time_s <= 180*(k+1)` 규칙에 따라 Time_Step k 로 매핑하고, 각 (Time_Step, cell_id) 키마다 정확히 1개의 Feature_Record 를 생성한다. 지원 최대 입력 규모는 1,000,000 데이터 행이며, 최대 입력 규모에서의 처리 완료 시간은 300초 이내이다.
2. WHEN Data_Extractor 가 하나의 (Time_Step, cell_id) 그룹을 집계하면, THE Data_Extractor SHALL Feature_Group 이 피처 이름별로 지정한 집계 규칙(합계 또는 평균)을 해당 피처 열에만 적용하고, 집계 결과 실수 값을 소수점 6자리로 반올림하여 Feature_Record 에 기록한다.
3. WHEN 집계 대상 값이 `nan` 문자열이거나 빈 문자열이거나 실수로 해석되지 않는 문자열이면, THE Data_Extractor SHALL 해당 값을 집계 모집단에서 제외하고 해당 (Time_Step, cell_id, 피처) 단위의 제외 표본 수와 유효 표본 수를 Feature_Record 에 기록한다.
4. IF 하나의 (Time_Step, cell_id) 그룹에서 어떤 요구 피처의 유효 표본 수가 0 이면, THEN THE Data_Extractor SHALL 해당 Feature_Record 의 그 피처 값을 미정의 표식으로 설정하고 `data_quality` 필드를 `insufficient` 로 설정하며, 나머지 (Time_Step, cell_id) 그룹의 처리를 중단 없이 계속한다.
5. WHEN Data_Extractor 가 Feature_Record 를 생성하면, THE Data_Extractor SHALL 원본 결과 디렉터리의 실행 메타데이터에 기록된 5개 Control_Parameter 값(`tx_power_dbm`, `ret_tilt_deg`, `cio_bias_db`, `hysteresis_db`, `ttt_ms`)과 원본 결과 디렉터리 경로 및 seed 값을 각 Feature_Record 에 기록한다.
6. WHEN `time_s` 최대값이 900 인 900초 길이 결과가 입력되면, THE Data_Extractor SHALL Feature_Group 의 대상 셀 집합에 속한 각 cell_id 마다 Time_Step 인덱스 0 부터 4 까지 5개 구간의 Feature_Record 를 생성하여 총 (대상 셀 수 × 5) 개의 Feature_Record 를 산출한다.
7. IF `time_s` 값이 1 미만이거나 900 을 초과하거나 실수로 해석되지 않으면, THEN THE Data_Extractor SHALL 해당 행을 집계에서 제외하고 제외된 행 수를 처리 결과에 반환하며 나머지 행의 처리를 계속한다.
8. IF CSV 헤더에 Feature_Group 이 요구하는 열 이름이 없거나 Feature_Group 에 어떤 요구 피처의 집계 규칙이 정의되어 있지 않으면, THEN THE Data_Extractor SHALL 누락된 열 이름 목록과 집계 규칙이 정의되지 않은 피처 이름 목록을 담은 오류를 반환하고, Feature_Record 를 하나도 생성하지 않으며 Feature_Store 를 변경하지 않는다.
9. WHEN 동일한 결과 디렉터리에 대해 Data_Extractor 가 두 번 실행되면, THE Feature_Store SHALL 레코드 개수, (Time_Step, cell_id) 키 집합, 모든 필드 값이 첫 번째 실행 결과와 동일한 Feature_Record 집합을 보유하고 동일 키의 중복 레코드를 보유하지 않는다.
10. IF 입력 `cell_kpi.csv` 경로가 존재하지 않거나 읽을 수 없거나 헤더 이후 데이터 행이 0개이면, THEN THE Data_Extractor SHALL 해당 원인을 식별하는 오류를 반환하고 Feature_Store 를 변경하지 않는다.
11. WHEN Data_Extractor 가 어떤 Feature_Record 의 `data_quality` 필드를 설정하면, THE Data_Extractor SHALL 모든 요구 피처의 유효 표본 수가 1 이상이고 제외 표본 수 합이 0 인 경우 `complete`, 모든 요구 피처의 유효 표본 수가 1 이상이고 제외 표본 수 합이 1 이상인 경우 `partial`, 어떤 요구 피처의 유효 표본 수가 0 인 경우 `insufficient` 중 하나의 값만 설정한다.
12. IF 결과 디렉터리 실행 메타데이터의 Control_Parameter 값 중 하나 이상이 용어집에 정의된 허용 범위 또는 허용 값 집합을 벗어나면, THEN THE Data_Extractor SHALL 위반된 Control_Parameter 이름과 해당 값을 담은 오류를 반환하고 Feature_Store 를 변경하지 않는다.

### Requirement 2: 피처 레코드 직렬화 왕복

**User Story:** 개발자로서, 집계된 피처를 파일과 학습 파이프라인 사이에서 손실 없이 주고받고 싶다. 그래야 저장 형식 버그가 모델 품질 문제로 위장되지 않는다.

#### Acceptance Criteria

1. WHEN Feature_Record 집합과 대상 파일 경로를 담은 직렬화 요청이 도착하면, THE Feature_Store SHALL 최대 100,000개의 Feature_Record 를 (Time_Step, cell_id) 오름차순으로 단일 파일에 기록하고 기록된 레코드 개수를 반환한다.
2. WHEN 직렬화된 파일 경로를 담은 파싱 요청이 도착하면, THE Feature_Store SHALL 해당 파일의 모든 데이터 레코드를 Feature_Record 집합으로 복원하고 복원된 레코드 개수를 반환한다.
3. WHEN 1개 이상 100,000개 이하의 유효한 Feature_Record 집합이 직렬화된 후 파싱되면, THE Feature_Store SHALL 원본과 동일한 필드 이름 집합, 동일한 레코드 개수, 각 수치 필드 값이 원본 대비 절대 오차 1e-9 이내인 값, 각 문자열 필드 값이 원본과 완전히 일치하는 값을 갖는 집합을 반환한다.
4. IF 파싱 대상 파일의 필드 이름 집합이 Feature_Group 이 정의한 피처 이름 목록 및 타깃 KPI 목록과 일치하지 않으면, THEN THE Feature_Store SHALL 누락된 필드 이름 목록과 정의되지 않은 필드 이름 목록을 포함한 오류를 반환하고 Feature_Record 를 하나도 반환하지 않는다.
5. IF 직렬화는 성공했으나 이후 파싱이 실패하면, THEN THE Feature_Store SHALL 해당 왕복 연산을 `roundtrip_violation` 오류로 실패 처리하고, 불일치가 발생한 레코드 인덱스와 필드 이름을 오류에 포함하며, 부분 복원 결과를 반환하지 않는다.
6. IF 직렬화 중 파일 기록이 실패하면, THEN THE Feature_Store SHALL 부분 기록된 파일을 남기지 않고 기록 실패를 나타내는 오류를 반환하며 입력 Feature_Record 집합을 변경하지 않는다.
7. IF 파싱 대상 파일이 잘린 레코드 또는 헤더에 선언된 필드 개수와 다른 필드 개수를 갖는 레코드를 포함하면, THEN THE Feature_Store SHALL 최초 위반 레코드 인덱스를 포함한 오류를 반환하고 Feature_Record 를 하나도 반환하지 않는다.
8. WHEN Feature_Record 개수가 0 인 집합이 직렬화된 후 파싱되면, THE Feature_Store SHALL 오류 없이 레코드 개수가 0 인 Feature_Record 집합을 반환한다.

### Requirement 3: 최소 학습 파이프라인

**User Story:** 연구자로서, 수집된 결과로 GNN 을 학습시키고 결과를 추적하고 싶다. 그래야 Kubeflow 전체 스택 없이도 재현 가능한 학습을 돌릴 수 있다.

#### Acceptance Criteria

1. WHEN 학습 요청이 Feature_Group 이름, 모델 이름, 난수 시드(정수, 값 범위 0~4294967295)를 담고 도착하면, THE Training_Manager SHALL 기존 작업과 중복되지 않는 작업 식별자를 생성하고 해당 작업의 상태를 `pending` 으로 초기화한 뒤 5초 이내에 작업 식별자를 호출자에게 반환한다.
2. THE Training_Manager SHALL 학습 파이프라인을 피처 추출 단계, 모델 학습 단계, 모델 저장 단계, 모델 지표 저장 단계의 순서로 실행하고 각 단계의 시작 시각, 종료 시각, 단계 결과(`succeeded` 또는 `failed`)를 작업 메타데이터에 기록하며, 어느 단계가 `failed` 이면 후속 단계를 실행하지 않는다.
3. WHILE 학습 작업이 실행 중이면, THE Training_Manager SHALL 해당 작업의 상태를 `running` 으로, 현재 실행 중인 단계 이름을 함께 조회 가능하게 유지하고 상태 조회 요청에 1초 이내로 응답한다.
4. WHEN 학습 작업이 완료되면, THE Training_Manager SHALL 학습 데이터 분할 비율(기본값 학습 0.8, 검증 0.2), 학습 표본 수, 검증 표본 수, 난수 시드, Feature_Group 이름, Target_KPI 별 평균 절대 백분율 오차(단위 percent, 소수점 둘째 자리까지)를 Model_Registry 에 기록한다.
5. IF 학습 작업이 실패하면, THEN THE Training_Manager SHALL 해당 작업의 상태를 `failed` 로 설정하고 실패한 단계 이름과 실패 유형(`data_error`, `training_error`, `storage_error` 중 하나)을 작업 메타데이터에 기록하며, 해당 작업의 모델 산출물을 Model_Registry 에 등록하지 않는다.
6. WHEN 동일한 Feature_Group 과 동일한 난수 시드로 학습 작업이 두 번 실행되면, THE Training_Manager SHALL 두 작업의 각 Target_KPI 별 평가 지표 절대 차이가 1.0 percentage point 이내이고 학습 표본 수가 동일한 결과를 산출한다.
7. IF 학습은 성공했으나 Model_Registry 지표 기록이 실패하면, THEN THE Training_Manager SHALL 최대 3회까지 기록을 재시도하고 3회 모두 실패한 경우 해당 작업의 상태를 `completed` 로 유지하며 지표 기록 실패 사유를 작업 메타데이터에 기록한다.
8. IF 요청된 Feature_Group 이름이 Feature_Store 에 없거나 해당 Feature_Group 의 유효 Feature_Record 수가 최소 학습 표본 수(기본값 200) 미만이면, THEN THE Training_Manager SHALL 학습 작업을 생성하지 않고 `insufficient_training_data` 오류 코드와 부족한 표본 수를 담은 오류 응답을 호출자에게 반환한다.
9. IF 하나의 학습 작업의 실행 시간이 최대 허용 실행 시간(기본값 3600초)을 초과하면, THEN THE Training_Manager SHALL 해당 작업을 중단하고 상태를 `failed` 로 설정하며 시간 초과를 실패 사유로 기록한다.

### Requirement 4: 모델 등록 및 조회

**User Story:** 운영자로서, 어떤 모델 버전이 어떤 데이터로 학습되었고 성능이 어떤지 조회하고 싶다. 그래야 추론 결과의 출처를 신뢰할 수 있다.

#### Acceptance Criteria

1. WHEN 모델 이름, Feature_Group 이름, Target_KPI 별 평가 지표, Model_Storage 내 산출물 URI 를 담은 등록 요청이 도착하면, THE Model_Registry SHALL 해당 모델 이름의 첫 등록에는 버전 번호 1 을, 이후 등록에는 해당 이름의 기존 최대 버전 번호에 1 을 더한 값을 부여하고, 부여된 버전 번호와 UTC 기준 생성 시각을 포함한 레코드를 저장한 뒤 저장된 모델 이름과 버전 번호를 호출자에게 반환한다.
2. WHEN 모델 이름으로 버전 목록 조회 요청이 도착하면, THE Model_Registry SHALL 해당 이름의 모든 버전을 버전 번호 오름차순으로 정렬하여 한 응답에 최대 500 개까지 반환하며, 각 항목에 버전 번호, 생성 시각, Feature_Group 이름, Target_KPI 별 평가 지표, 산출물 URI 를 포함하고, 요청 수신 시점부터 2 초 이내에 응답한다.
3. WHEN 모델 이름만으로 최신 버전 조회 요청이 도착하면, THE Model_Registry SHALL 해당 이름의 버전 중 버전 번호가 가장 큰 단일 항목을 반환하며, 그 항목에 버전 번호, 생성 시각, Feature_Group 이름, Target_KPI 별 평가 지표, 산출물 URI 를 포함한다.
4. IF 요청된 모델 이름이 Model_Registry 에 없거나 요청된 버전 번호가 해당 이름에 저장된 버전 집합에 속하지 않으면, THEN THE Model_Registry SHALL `model_not_found` 오류 코드와 요청된 모델 이름 및 버전 번호를 포함한 오류 응답 본문을 호출자에게 반환하고 저장된 레코드를 변경하지 않는다.
5. IF 버전 레코드는 존재하지만 연결된 모델 이름 레코드가 없으면, THEN THE Model_Registry SHALL `model_not_found` 오류 코드와 오류 응답 본문을 호출자에게 반환하고 해당 버전 레코드의 부분 메타데이터를 응답에 포함하지 않는다.
6. THE Model_Storage SHALL 등록된 각 (모델 이름, 버전 번호) 조합의 산출물 파일을 서로 겹치지 않는 고유 위치에 보관하고, 다른 버전의 산출물 파일을 덮어쓰거나 삭제하지 않는다.
7. IF 등록 요청에 모델 이름, Feature_Group 이름, Target_KPI 별 평가 지표, 산출물 URI 중 하나 이상이 누락되었거나 모델 이름 길이가 1 자 미만 또는 128 자 초과이면, THEN THE Model_Registry SHALL 등록을 거부하고 `invalid_model_metadata` 오류 코드와 누락 또는 위반 항목 이름 목록을 포함한 오류 응답 본문을 반환하며 새 버전 번호를 부여하지 않는다.
8. IF 등록 요청의 산출물 URI 가 Model_Storage 에서 조회되지 않으면, THEN THE Model_Registry SHALL 등록을 거부하고 `artifact_unavailable` 오류 코드와 해당 URI 를 포함한 오류 응답 본문을 반환하며 부분 레코드를 남기지 않는다.
9. IF 모델 이름은 Model_Registry 에 존재하지만 저장된 버전이 0 개이면, THEN THE Model_Registry SHALL 항목 수가 0 인 빈 버전 목록을 반환하고 오류 코드를 반환하지 않는다.

### Requirement 5: 배치 추론 서비스

**User Story:** Agent 개발자로서, 여러 파라미터 조합의 KPI 예측을 한 번의 호출로 받고 싶다. 그래야 프로빙 비용을 낮추고 비교 가능한 결과를 얻을 수 있다.

#### Acceptance Criteria

1. WHEN Inference_Service 가 시작되면, THE Inference_Service SHALL Model_Registry 에서 지정된 모델 이름과 버전의 산출물 로드를 60 초 이내에 완료하고, 상태 조회 응답에 모델 로드 상태 `loaded` 와 로드된 모델 이름 및 버전을 포함한다.
2. WHEN Batch_Prediction_Request 가 1 개 이상 64 개 이하인 N 개의 Parameter_Set 을 담고 도착하면, THE Inference_Service SHALL 요청과 동일한 순서로 N 개의 예측 결과를 반환하고 각 결과에 요청 내 0 기반 인덱스를 포함한다.
3. WHEN Inference_Service 가 하나의 Parameter_Set 을 예측하면, THE Inference_Service SHALL Glossary 에 정의된 모든 Target_KPI 각각에 대해 하나의 유한 실수 예측값과, 요청된 대상 셀 집합의 각 cell_id 에 대해 하나의 유한 실수 셀별 예측값을 누락 없이 반환한다.
4. WHEN Batch_Prediction_Request 가 0 이상 4 이하의 정수 Time_Step 값을 포함하면, THE Inference_Service SHALL 해당 Time_Step 조건에서의 예측값을 반환하고 응답에 적용된 Time_Step 값을 포함한다.
5. IF Parameter_Set 의 어떤 Control_Parameter 값이 Glossary 에 정의된 허용 범위 또는 허용 값 집합을 벗어나면, THEN THE Inference_Service SHALL 요청 전체에 대한 예측을 수행하지 않고, 위반한 모든 (Parameter_Set 인덱스, 위반 파라미터 이름, 위반 값) 항목 목록을 담은 `parameter_out_of_range` 오류를 반환한다.
6. THE Inference_Service SHALL Batch_Prediction_Request 당 1 개 이상 64 개 이하의 Parameter_Set 을 수용하고, 64 개 Parameter_Set 요청에 대한 응답을 요청 수신 후 5 초 이내에 반환한다.
7. IF Batch_Prediction_Request 의 Parameter_Set 개수가 0 이거나 64 를 초과하면, THEN THE Inference_Service SHALL 예측을 수행하지 않고, 수신된 Parameter_Set 개수를 포함한 오류를 반환하며, 개수가 0 인 경우 `empty_batch` 오류 코드를, 64 를 초과하는 경우 `batch_too_large` 오류 코드를 사용한다.
8. WHEN 동일한 Batch_Prediction_Request 가 동일한 모델 이름과 버전에 두 번 전송되면, THE Inference_Service SHALL 두 응답의 모든 Target_KPI 예측값과 셀별 예측값이 동일하고 오류 코드가 동일한 응답을 반환한다.
9. WHEN 예측 응답이 생성되면, THE Inference_Service SHALL 사용된 모델 이름과 버전을 응답에 포함한다.
10. IF 예측값이 생성된 후 오류가 발생하면, THEN THE Inference_Service SHALL 오류 응답에 사용된 모델 이름과 버전 및 오류 코드를 포함하고 부분 예측 결과를 포함하지 않는다.
11. IF 모델 산출물 로드가 실패하거나 60 초 이내에 완료되지 않으면, THEN THE Inference_Service SHALL 상태 조회 응답의 모델 로드 상태를 `not_loaded` 로 노출하고, 이후 도착하는 모든 Batch_Prediction_Request 에 대해 예측을 수행하지 않고 실패 사유를 나타내는 `model_not_loaded` 오류를 반환한다.
12. IF Batch_Prediction_Request 의 Time_Step 값이 정수가 아니거나 0 이상 4 이하 범위를 벗어나면, THEN THE Inference_Service SHALL 예측을 수행하지 않고 수신된 Time_Step 값을 포함한 `invalid_time_step` 오류를 반환한다.
13. WHERE Batch_Prediction_Request 에 Time_Step 값이 생략되면, THE Inference_Service SHALL Time_Step 0 을 적용한 예측값을 반환하고 응답에 적용된 Time_Step 값 0 을 포함한다.

### Requirement 6: GNN 능력의 MCP 도구 노출

**User Story:** Agent 개발자로서, GNN 질의를 기존 MCP 도구 방식으로 호출하고 싶다. 그래야 `smo/agentic_ai` 의 기존 도구 사용 패턴을 그대로 재사용할 수 있다.

#### Acceptance Criteria

1. THE GNN_MCP_Server SHALL 1개 이상 64개 이하의 Parameter_Set 과 선택적 Time_Step 값(0~4)을 입력으로 받아 Batch_Prediction_Request 를 수행하고 입력과 동일한 순서의 예측 결과 목록을 반환하는 MCP 도구를 제공한다.
2. THE GNN_MCP_Server SHALL Baseline_Parameter_Set 1개, 1개 이상 5개 이하의 Control_Parameter 이름, 1개 이상의 Target_KPI 이름을 입력으로 받아 각 (Control_Parameter, Target_KPI) 쌍의 Marginal_Effect 값을 -100.0 이상 100.0 이하의 percent 값으로 반환하는 MCP 도구를 제공한다.
3. THE GNN_MCP_Server SHALL 입력 인자 없이 호출되어 Model_Registry 의 현재 서빙 모델 이름과 버전 번호를 반환하는 MCP 도구를 제공한다.
4. WHEN MCP 도구가 호출되고 Inference_Service 가 성공 응답을 반환하면, THE GNN_MCP_Server SHALL Inference_Service 응답의 필드 이름, 필드 값, 결과 순서를 변경하지 않고 사용된 모델 이름과 버전을 포함한 상태로 도구 결과에 포함한다.
5. IF Inference_Service 가 오류를 반환하면, THEN THE GNN_MCP_Server SHALL 동일한 오류 코드와 오류 메시지를 도구 결과로 반환하고 부분 예측값을 도구 결과에 포함하지 않는다.
6. THE GNN_MCP_Server SHALL 읽기 전용 예측 및 조회 도구만 제공하며, Parameter_Set, Model_Registry 항목, RAN 구성 중 어느 것도 변경하는 도구를 노출하지 않는다.
7. WHEN RAN 파라미터 변경이 필요하면, THE SMO_Framework SHALL Policy_Manager 를 통한 A1 정책 발행 경로로 Near-RT RIC 의 xApp 에 변경을 위임하고 xApp 이 E2 로 RAN 을 제어하게 한다.
8. WHEN MCP 클라이언트가 도구 목록 조회를 요청하면, THE GNN_MCP_Server SHALL 제공하는 각 도구의 이름, 입력 인자 이름, 각 인자의 필수 여부와 허용 범위를 반환한다.
9. IF 도구 호출 인자에 필수 인자가 누락되었거나 Parameter_Set 개수가 64 를 초과하거나 Time_Step 값이 0~4 범위를 벗어나면, THEN THE GNN_MCP_Server SHALL Inference_Service 를 호출하지 않고 위반된 인자 이름을 담은 입력 검증 오류를 도구 결과로 반환한다.
10. IF Inference_Service 가 도구 호출 후 30초 이내에 응답하지 않거나 연결이 불가하면, THEN THE GNN_MCP_Server SHALL 재시도 없이 추론 서비스 이용 불가를 나타내는 오류 코드와 오류 메시지를 도구 결과로 반환하고 호출자가 전달한 입력 인자를 변경하지 않는다.

### Requirement 7: xApp 파라미터 요청에 대한 배치 프로빙

**User Story:** Non-RT RIC 운영자로서, xApp 이 요청한 파라미터 조합이 실제로 좋은 선택인지 GNN 으로 검증받고 싶다. 그래야 추측이 아닌 데이터 근거로 정책을 승인할 수 있다.

#### Acceptance Criteria

1. WHEN xApp 파라미터 요청이 도착하면, THE KPI_Advisor_Agent SHALL 요청에 지정되지 않은 Control_Parameter 를 Feature_Store 의 가장 최근 Feature_Record 값으로 채워 5개 Control_Parameter 가 모두 지정된 Baseline_Parameter_Set 1개를 구성하고, 대상 셀 집합을 요청에 명시된 1개 이상 16개 이하의 cell_id 로 확정한다.
2. WHEN Baseline_Parameter_Set 이 확정되면, THE KPI_Advisor_Agent SHALL Glossary 에 정의된 각 Control_Parameter 의 스텝 크기(TxP 1, RET 1, CIO 0.5, HYS 0.5, TTT 는 허용 값 집합의 인접 값)를 변형 단위로 사용하여 Probe_Plan 을 생성하고, Probe_Plan 에 Baseline_Parameter_Set 식별자, Probe_Variant 목록, 각 Probe_Variant 의 변형 대상 cell_id, 변형 파라미터 이름, 변형 방향을 기록한다.
3. WHEN Probe_Plan 이 생성되면, THE KPI_Advisor_Agent SHALL Probe_Plan 에 Baseline_Parameter_Set 1개와 각 대상 셀의 각 Control_Parameter 에 대해 +1 스텝 방향 1개 및 -1 스텝 방향 1개의 Probe_Variant 를 포함시키고, 각 Probe_Variant 는 Baseline_Parameter_Set 과 정확히 하나의 (cell_id, Control_Parameter) 값만 다르게 하며, 변형 결과가 Glossary 의 허용 범위 또는 허용 값 집합을 벗어나는 방향은 Probe_Variant 를 생성하지 않는다.
4. IF 모든 대상 셀의 모든 Control_Parameter 에 대해 +1 스텝 및 -1 스텝 변형 결과가 Glossary 의 허용 범위 또는 허용 값 집합을 벗어나면, THEN THE KPI_Advisor_Agent SHALL Baseline_Parameter_Set 1개만 담고 Probe_Variant 개수 0 을 기록한 Probe_Plan 을 생성한다.
5. WHEN KPI_Advisor_Agent 가 Parameter_Set 개수가 64 이하인 Probe_Plan 을 실행하면, THE KPI_Advisor_Agent SHALL Probe_Plan 의 모든 Parameter_Set 을 Probe_Plan 에 기록된 순서대로 하나의 Batch_Prediction_Request 에 담아 GNN_MCP_Server 에 전달한다.
6. IF Probe_Plan 의 Parameter_Set 개수가 64 를 초과하면, THEN THE KPI_Advisor_Agent SHALL Probe_Plan 의 순서를 유지한 채 각 배치의 Parameter_Set 개수가 64 이하가 되도록 분할하고, Baseline_Parameter_Set 을 첫 배치의 첫 Parameter_Set 으로 배치하여 모든 배치를 순차 전달한다.
7. WHEN Probe_Plan 의 모든 배치에 대한 예측 결과가 수신되면, THE KPI_Advisor_Agent SHALL 각 Probe_Variant 의 각 Target_KPI 에 대해 `(Probe_Variant 예측값 - Baseline_Parameter_Set 예측값) / |Baseline_Parameter_Set 예측값| * 100` 을 소수점 첫째 자리까지 반올림한 변화율을 percent 단위로 산출한다.
8. IF xApp 요청이 어떤 Control_Parameter 값도 지정하지 않거나 대상 셀 집합이 비어 있으면, THEN THE KPI_Advisor_Agent SHALL Probe_Plan 을 생성하지 않고 `empty_request` 오류를 반환한다.
9. THE KPI_Advisor_Agent SHALL Policy_Manager 의 A1 정책 발행 경로만 사용하여 파라미터 변경을 요청한다.
10. IF Baseline_Parameter_Set 의 어떤 Target_KPI 예측값의 절대값이 0 이면, THEN THE KPI_Advisor_Agent SHALL 해당 Target_KPI 의 변화율 값을 `undefined` 로 표기하고 Baseline_Parameter_Set 예측값과 Probe_Variant 예측값을 원시값으로 반환한다.
11. IF GNN_MCP_Server 가 오류를 반환하거나 하나의 배치 전달 후 30 초 이내에 응답하지 않으면, THEN THE KPI_Advisor_Agent SHALL 해당 배치를 최대 2 회 재전송하고, 모든 재전송이 실패하면 Probe_Plan 실행을 중단하며 실패한 배치 인덱스와 원 오류 사유를 담은 오류를 반환하고 부분 예측 결과를 반환하지 않는다.
12. IF 분할 전달된 배치들의 응답에 포함된 모델 이름 또는 모델 버전이 서로 다르면, THEN THE KPI_Advisor_Agent SHALL 해당 Probe_Plan 실행 결과를 폐기하고 배치 간 모델 버전 불일치를 지시하는 오류를 반환한다.

### Requirement 8: 임계 KPI 판정 및 대안 추천

**User Story:** Non-RT RIC 운영자로서, 요청이 KPI 를 임계값 이상으로 열화시키는지, 더 나은 조합이 있는지 알고 싶다. 그래야 xApp 요청을 승인하거나 대안을 제시할 수 있다.

#### Acceptance Criteria

1. WHEN KPI_Advisor_Agent 가 초기화되면, THE KPI_Advisor_Agent SHALL 설정 파일에서 Threshold_KPI 정의를 로드하고, 각 Target_KPI 에 대해 허용 하한 값, 허용 상한 값, 개선 방향(값이 클수록 개선 또는 값이 작을수록 개선) 중 하나 이상이 정의되어 있는지 확인한다.
2. WHEN Baseline_Parameter_Set 의 예측값이 Threshold_KPI 가 정의된 모든 Target_KPI 에 대해 허용 하한 값 이상이고 허용 상한 값 이하이면(경계값 포함), THE KPI_Advisor_Agent SHALL Degradation_Verdict 를 `acceptable` 로 설정한다.
3. WHEN Baseline_Parameter_Set 의 예측값이 하나 이상의 Target_KPI 에서 허용 하한 값 미만 또는 허용 상한 값 초과이면, THE KPI_Advisor_Agent SHALL Degradation_Verdict 를 `degrading` 으로 설정하고, 위반된 모든 Target_KPI(최대 10개)에 대해 Target_KPI 이름, 예측값, 적용된 임계값, 위반 방향(하한 미만 또는 상한 초과)을 반환한다.
4. IF Threshold_KPI 위반이 없고 Feature_Store 에 Baseline_Parameter_Set 과 동일 구간의 실측 근거가 없고 예측 신뢰 구간 폭이 예측값의 50 percent 를 초과하면, THEN THE KPI_Advisor_Agent SHALL Degradation_Verdict 를 `unknown` 으로 설정하고 해당 Target_KPI 이름과 신뢰 구간 폭 비율을 반환한다.
5. WHEN 어떤 Probe_Variant 가 Threshold_KPI 가 정의된 모든 Target_KPI 에 대해 허용 범위(경계값 포함)를 만족하고, xApp_Objective 에 대해 설정 파일에 지정된 주 Target_KPI 의 예측값이 설정된 개선 방향 기준으로 Baseline_Parameter_Set 대비 1.0 percent 이상 개선되면, THE KPI_Advisor_Agent SHALL 해당 Probe_Variant 를 추천 후보에 포함한다.
6. WHEN 추천 후보가 2개 이상이면, THE KPI_Advisor_Agent SHALL 주 Target_KPI 개선폭(단위 percent) 내림차순으로 후보를 정렬하고, 개선폭 차이가 0.1 percent 이하로 동률인 후보는 Probe_Variant 식별자 오름차순으로 정렬하여 상위 최대 5개를 반환한다.
7. WHEN 추천 후보가 정확히 1개이면, THE KPI_Advisor_Agent SHALL 해당 후보를 최상위 추천으로 반환한다.
8. WHEN 추천 후보가 0개이면, THE KPI_Advisor_Agent SHALL 추천 목록을 빈 목록으로 반환하고, 각 Probe_Variant 별로 후보 제외 사유(Threshold_KPI 위반, 주 Target_KPI 개선폭 1.0 percent 미달, 예측값 누락 중 하나)를 포함한 Baseline_Parameter_Set 유지 사유를 반환한다.
9. WHEN Probe_Plan 의 모든 Parameter_Set 에 대한 예측 결과 처리가 완료되면, THE KPI_Advisor_Agent SHALL 30초 이내에 Degradation_Verdict, 추천 목록, Evidence_Record 를 하나의 응답으로 함께 반환한다.
10. IF Threshold_KPI 정의 로드가 실패하거나 어떤 Target_KPI 의 임계값이 숫자가 아니거나 허용 하한 값이 허용 상한 값보다 크면, THEN THE KPI_Advisor_Agent SHALL 판정을 수행하지 않고 해당 Target_KPI 이름과 실패 원인을 나타내는 오류를 반환한다.
11. IF 어떤 Probe_Variant 의 Target_KPI 예측값이 누락되면, THEN THE KPI_Advisor_Agent SHALL 해당 Probe_Variant 를 추천 후보에서 제외하고 제외 사유를 표시하며, 나머지 Probe_Variant 에 대한 판정과 추천을 계속한다.
12. IF 예측 결과에 신뢰 구간 정보가 포함되지 않으면, THEN THE KPI_Advisor_Agent SHALL Degradation_Verdict 를 `unknown` 으로 설정하고 신뢰 구간 정보 부재를 사유로 반환한다.

### Requirement 9: 간접 충돌 분석

**User Story:** Non-RT RIC 운영자로서, 서로 다른 파라미터가 같은 KPI 를 반대 방향으로 끌어당기는 상황을 미리 알고 싶다. 그래야 상충하는 정책을 동시에 내리는 실수를 막을 수 있다.

#### Acceptance Criteria

1. WHEN Probe_Plan 실행 결과가 주어지면, THE Conflict_Analyzer SHALL 동일 대상 셀에 속한 각 (Control_Parameter, Target_KPI) 쌍에 대한 Marginal_Effect 를 최대 20개 Control_Parameter, 최대 10개 Target_KPI, 최대 200개 쌍의 범위에서 계산하고, 각 값을 -100.0 이상 100.0 이하 범위로 제한하여 소수점 첫째 자리까지 반올림한 percent 값으로 산출한다.
2. WHEN 동일 대상 셀과 동일 Target_KPI 에 대해 서로 다른 두 Control_Parameter 중 한쪽의 Marginal_Effect 가 `+Conflict_Threshold` 이상이고 다른 쪽의 Marginal_Effect 가 `-Conflict_Threshold` 이하이면, THE Conflict_Analyzer SHALL 해당 쌍을 Indirect_Conflict 로 판정한다. 경계값 `+Conflict_Threshold` 와 `-Conflict_Threshold` 는 판정에 포함한다.
3. WHEN Indirect_Conflict 가 판정되면, THE Conflict_Analyzer SHALL 대상 셀 식별자, 두 Control_Parameter 이름, 대상 Target_KPI 이름, 두 Marginal_Effect 값, 사용된 Conflict_Threshold 값을 하나의 항목으로 Indirect_Conflict 목록에 반환한다.
4. WHEN 동일 Target_KPI 에 대한 두 Control_Parameter 의 Marginal_Effect 가 기준 2 의 조건을 만족하지 않으면, THE Conflict_Analyzer SHALL 해당 쌍을 Indirect_Conflict 목록에서 제외한다. 두 값의 절대값이 모두 Conflict_Threshold 미만인 경우, 두 값의 부호가 같은 경우, 한쪽만 절대값이 Conflict_Threshold 이상인 경우를 모두 제외 대상으로 한다.
5. WHERE Conflict_Threshold 가 설정 파일에 0.1 이상 100.0 이하의 값으로 지정되면, THE Conflict_Analyzer SHALL 지정된 값을 사용한다.
6. WHERE Conflict_Threshold 가 설정 파일에 지정되지 않으면, THE Conflict_Analyzer SHALL 5.0 percent 를 사용한다.
7. WHEN Conflict_Analyzer 가 동일한 예측 결과 집합과 동일한 Conflict_Threshold 값에 두 번 적용되면, THE Conflict_Analyzer SHALL Target_KPI 이름 오름차순, 이어서 두 Control_Parameter 이름의 사전순 오름차순으로 정렬된 동일한 Indirect_Conflict 목록을 항목 순서까지 동일하게 반환한다.
8. IF 어떤 Control_Parameter 의 Probe_Variant 예측이 누락되면, THEN THE Conflict_Analyzer SHALL 해당 Control_Parameter 를 `effect_unavailable` 로 표시하고, 해당 Control_Parameter 가 포함된 쌍만 Indirect_Conflict 판정에서 제외하며, 나머지 쌍에 대한 분석을 완료하여 목록을 반환한다.
9. IF Conflict_Threshold 가 설정 파일에 지정되었으나 숫자가 아니거나 0.1 이상 100.0 이하 범위를 벗어나면, THEN THE Conflict_Analyzer SHALL 5.0 percent 를 사용하여 분석을 계속하고, 설정값이 허용 범위를 벗어났음을 나타내는 경고를 분석 결과에 포함한다.
10. IF Marginal_Effect 계산이 가능한 Control_Parameter 가 동일 Target_KPI 에 대해 2개 미만이면, THEN THE Conflict_Analyzer SHALL 해당 Target_KPI 에 대해 빈 Indirect_Conflict 목록을 반환하고 판정 불가 사유를 결과에 포함한다.
11. WHEN 분석 대상 쌍이 200개 이하이면, THE Conflict_Analyzer SHALL Probe_Plan 실행 결과 수신 후 2000 ms 이내에 Indirect_Conflict 목록을 반환한다.

### Requirement 10: 시간 구간별 xApp 실행 계획

**User Story:** Non-RT RIC 운영자로서, 900초 구간 안에서 어떤 시간대에 어떤 xApp 을 돌릴지 계획하고 싶다. 그래야 에너지 절감과 처리량 목표를 시간축으로 나눠 함께 달성할 수 있다.

#### Acceptance Criteria

1. WHEN 1개 이상 3개 이하의 xApp_Objective 와 각 xApp_Objective 별 1개 이상 64개 이하의 후보 Parameter_Set 이 주어지면, THE Temporal_Scheduler SHALL Time_Step 0 부터 4 까지 5개 구간 각각에 대해 해당 구간 예측 결과에서 xApp_Objective 주 KPI 값이 가장 좋은(주 KPI 가 최대화 대상이면 최대값, 최소화 대상이면 최소값) (xApp_Objective, Parameter_Set) 조합 하나를 배정한 Temporal_Plan 을 생성한다.
2. IF xApp_Objective 가 주어지지 않고 후보 Parameter_Set 만 주어지면, THEN THE Temporal_Scheduler SHALL 각 Time_Step 의 배정 xApp_Objective 를 `unspecified` 로 설정하고 해당 Time_Step 에서 Threshold_KPI 를 위반한 Target_KPI 개수가 가장 적은 후보 Parameter_Set 을 배정한 Temporal_Plan 을 생성한다.
3. WHEN Temporal_Scheduler 가 Temporal_Plan 을 평가하면, THE Temporal_Scheduler SHALL 5개 Time_Step 과 후보 Parameter_Set 의 모든 조합을 Batch_Prediction_Request 당 최대 64개 Parameter_Set 이하가 되도록 연속 배치로 분할하여 GNN_MCP_Server 에 전달한다.
4. WHEN Temporal_Plan 생성이 완료되면, THE Temporal_Scheduler SHALL Temporal_Plan 의 각 Time_Step 항목에 배정된 xApp_Objective 이름, 적용 Parameter_Set 의 5개 Control_Parameter 값, 예측된 각 Target_KPI 값, 예측에 사용된 모델 이름과 버전을 포함한다.
5. WHEN Temporal_Plan 생성이 완료되면, THE Temporal_Scheduler SHALL 5개 Time_Step 전체에 대한 Target_KPI 누적값을 반환하되, 처리량·에너지·카운트 계열 Target_KPI 는 5개 구간 값의 합계로, 비율 및 분위 계열 Target_KPI 는 5개 구간 값의 산술 평균으로 산출한다.
6. IF 어떤 Time_Step 의 배정 예측값이 1개 이상의 Threshold_KPI 를 위반하면, THEN THE Temporal_Scheduler SHALL 해당 Time_Step 인덱스를 위반 항목으로 표시하고 위반된 Target_KPI 이름과 예측값과 적용된 임계값을 반환하며, 5개 Time_Step 배정을 모두 포함한 Temporal_Plan 반환을 계속한다.
7. WHEN Temporal_Plan 생성이 완료되면, THE Temporal_Scheduler SHALL 직전 Time_Step 과 배정 xApp_Objective 가 다른 모든 Time_Step 인덱스(1 이상 4 이하)를 Temporal_Plan 의 전환 목록에 오름차순으로 기록하고, 그러한 인덱스가 없으면 빈 전환 목록을 기록한다.
8. WHEN 주어진 후보 Parameter_Set 개수가 0 이면, THE Temporal_Scheduler SHALL Feature_Store 의 Feature_Record 중 `data_quality` 가 `insufficient` 가 아닌 레코드만 대상으로 xApp_Objective 주 KPI 값이 가장 좋은(최대화 대상이면 최대값, 최소화 대상이면 최소값) Parameter_Set 을 Time_Step 별 후보로 도출하고 도출 근거 Feature_Record 식별자를 반환한다.
9. IF 후보 Parameter_Set 개수가 0 이고 Feature_Store 에 `data_quality` 가 `insufficient` 가 아닌 Feature_Record 가 0개이면, THEN THE Temporal_Scheduler SHALL `no_candidate` 오류를 반환하고 Temporal_Plan 을 생성하지 않는다.
10. THE Temporal_Scheduler SHALL Temporal_Plan 에 Time_Step 인덱스 0 부터 4 까지 정확히 5개 항목만 오름차순으로 포함한다.
11. IF 동일 Time_Step 에서 배정 기준값이 동일한 후보 조합이 2개 이상이면, THEN THE Temporal_Scheduler SHALL xApp_Objective 이름 오름차순으로, 이름이 동일하면 입력 순서가 앞선 Parameter_Set 을 배정하여 동일 입력에 대해 동일한 Temporal_Plan 을 반환한다.
12. IF 요청이 0 미만 또는 4 초과의 Time_Step 인덱스를 지정하면, THEN THE Temporal_Scheduler SHALL 범위를 벗어난 인덱스 값을 담은 `time_step_out_of_range` 오류를 반환하고 Temporal_Plan 을 생성하지 않는다.
13. IF GNN_MCP_Server 가 오류를 반환하거나 요청 전송 후 30초 이내에 응답하지 않으면, THEN THE Temporal_Scheduler SHALL 수신한 오류 코드 또는 시간 초과 사유를 담은 `prediction_unavailable` 오류를 반환하고 부분 Temporal_Plan 을 반환하지 않는다.

### Requirement 11: A1 정책 발행

**User Story:** Non-RT RIC 운영자로서, 승인된 계획을 A1 정책으로 xApp 에 내려보내고 싶다. 그래야 Agent 판단이 실제 RAN 제어로 이어진다.

#### Acceptance Criteria

1. WHEN Temporal_Plan 또는 추천 Parameter_Set 에 대한 명시적 승인 기록(승인자 식별자와 승인 시각을 포함)이 생성되면, THE Policy_Manager SHALL 5개 Control_Parameter 각각의 허용 범위와 스텝(TxP 30~46 스텝 1, RET 0~15 스텝 1, CIO -6.0~+6.0 스텝 0.5, HYS 0.0~10.0 스텝 0.5, TTT 허용 값 집합)을 `create_schema` 속성에 포함한 Policy_Type 을 A1_Mediator 에 등록한다.
2. WHERE 동일한 `policy_type_id` 의 Policy_Type 이 A1_Mediator 에 이미 등록되어 있으면, THE Policy_Manager SHALL 재등록을 수행하지 않고 기존 Policy_Type 식별자를 반환한다.
3. WHEN Policy_Type 이 등록된 상태에서 대상 셀 개수가 1 이상인 정책 발행이 요청되면, THE Policy_Manager SHALL 각 대상 셀의 5개 Control_Parameter 값을 모두 담은 Policy_Instance 를 A1_Mediator 에 생성한다.
4. WHEN 발행 대상이 Temporal_Plan 이면, THE Policy_Manager SHALL Time_Step 0 부터 4 까지 각 구간에 대해 정확히 하나씩 총 5개의 Policy_Instance 를 생성한다.
5. WHEN 유효한 Parameter_Set 이 직렬화 요청으로 주어지면, THE Policy_Manager SHALL 모든 대상 셀 식별자와 각 셀의 5개 Control_Parameter 값을 포함한 A1 Policy_Instance JSON 을 반환한다.
6. WHEN A1 Policy_Instance JSON 이 파싱 요청으로 주어지면, THE Policy_Manager SHALL 해당 JSON 에 포함된 대상 셀 식별자와 5개 Control_Parameter 값으로 구성된 Parameter_Set 을 반환한다.
7. WHEN 임의의 유효한 Parameter_Set 이 Policy_Instance JSON 으로 직렬화된 후 파싱되면, THE Policy_Manager SHALL 원본과 동일한 대상 셀 식별자 집합 및 셀별로 정확히 일치하는 5개 Control_Parameter 값을 갖는 Parameter_Set 을 반환하며, 셀 나열 순서 차이는 불일치로 판정하지 않는다.
8. IF 직렬화 후 파싱 결과가 원본 Parameter_Set 과 불일치하면, THEN THE Policy_Manager SHALL 해당 발행 연산을 `roundtrip_violation` 오류로 실패 처리하고 Policy_Instance 생성을 수행하지 않는다.
9. IF A1_Mediator 가 HTTP 4xx 또는 5xx 응답을 반환하면, THEN THE Policy_Manager SHALL 응답 상태 코드와 응답 본문을 포함한 오류를 반환하고, 재시도 없이 해당 발행 연산을 실패로 처리하며, 호출한 엔드포인트 이름과 시도 시각을 발행 시도 기록에 남긴다.
10. IF Degradation_Verdict 가 `degrading` 또는 `unknown` 이고 명시적 사용자 승인 기록이 존재하지 않으면, THEN THE Policy_Manager SHALL Policy_Instance 를 생성하지 않고 보류 상태 `pending_approval` 과 판정 결과 및 위반된 Threshold_KPI 이름을 반환한다.
11. WHEN Policy_Instance 가 생성되면, THE Policy_Manager SHALL 5초 이내에 A1_Mediator 의 정책 인스턴스 상태 조회를 1회 수행하여 Policy_Instance 식별자와 조회된 상태 값을 반환한다.
12. IF Policy_Instance 생성은 성공했으나 상태 조회가 실패하거나 5초 이내에 응답하지 않으면, THEN THE Policy_Manager SHALL 생성된 Policy_Instance 를 삭제하지 않고 유지하며 상태 값을 `status_unavailable` 로 반환한다.
13. IF 발행 요청 시점에 대상 Policy_Type 이 A1_Mediator 에 등록되어 있지 않으면, THEN THE Policy_Manager SHALL `policy_type_not_registered` 오류를 반환하고 Policy_Instance 생성을 수행하지 않는다.
14. IF 발행 대상 Parameter_Set 에서 5개 Control_Parameter 중 하나라도 누락되거나 값이 해당 파라미터의 허용 범위, 스텝, 허용 값 집합을 벗어나면, THEN THE Policy_Manager SHALL 위반된 Control_Parameter 이름과 입력 값을 포함한 `parameter_out_of_range` 오류를 반환하고 Policy_Instance 생성을 수행하지 않는다.
15. IF A1_Mediator 연결이 실패하거나 10초 이내에 HTTP 응답을 반환하지 않으면, THEN THE Policy_Manager SHALL 동일 요청을 최대 2회까지 재시도하고, 3회 시도 모두 실패하면 `a1_unreachable` 오류를 반환하며 해당 Policy_Instance 를 생성 성공으로 기록하지 않는다.
16. THE Policy_Manager SHALL 포트 9000 의 A1_Mediator 가 이미 제공하는 8개 Policy_Type / Policy_Instance 엔드포인트만 호출하고 그 외 엔드포인트를 호출하지 않는다.

### Requirement 12: 기존 멀티 에이전트 런타임 통합

**User Story:** 개발자로서, 신규 GNN 판단 능력을 기존 deep-agents 구조 안에서 쓰고 싶다. 그래야 새 오케스트레이션 계층을 중복 구축하지 않는다.

#### Acceptance Criteria

1. THE KPI_Advisor_Agent SHALL `smo/agentic_ai/agents/` 하위에 기존 sub-agent 와 동일한 파일 구성(`agent.md`, `agent.py`, `schemas.py`, `server.py`, `pyproject.toml`) 5개 파일을 모두 포함하여 배치되고, `smo/agentic_ai/agents/` 외부에 새 오케스트레이션 계층 디렉터리를 추가하지 않는다.
2. THE KPI_Advisor_Agent SHALL 외부 인터페이스로 `GET /health` 와 `POST /invoke` 2개 엔드포인트만 제공한다.
3. WHEN `GET /health` 요청이 도착하면, THE KPI_Advisor_Agent SHALL 3초 이내에 `ready` 또는 `not_ready` 상태 값과 현재 로드된 GNN_Model 이름 및 버전을 담은 응답을 반환한다.
4. WHEN KPI_Advisor_Agent 가 Multi_Agent_Runtime 에 등록되면, THE Multi_Agent_Runtime SHALL Knowledge DB 의 `planner_manifest` 에 에이전트 이름, 능력 설명, 입출력 스키마 참조를 포함한 항목 1건을 기록하고, Planning Agent 의 후보 조회 결과에 KPI_Advisor_Agent 를 포함하여 반환한다.
5. WHEN Planning Agent 가 RAN 파라미터 영향 예측을 요구하는 subtask 를 처리하면, THE Planning Agent SHALL KPI_Advisor_Agent 노드를 1개 이상 포함하고 해당 노드 입력에 Baseline_Parameter_Set 과 대상 Time_Step 을 지정한 `langgraph_spec` 을 생성한다.
6. WHEN `POST /invoke` 요청의 사용자 의도 텍스트가 한국어이면, THE KPI_Advisor_Agent SHALL 판정 근거 요약을 1자 이상 2000자 이하의 한국어 텍스트 필드로 포함한 응답을 반환한다.
7. WHEN `POST /invoke` 요청이 도착하면, THE KPI_Advisor_Agent SHALL 60초 이내에 Degradation_Verdict, 추천 Parameter_Set 목록(0개 이상 10개 이하), Evidence_Record 식별자를 담은 단일 JSON 객체를 반환한다.
8. THE KPI_Advisor_Agent SHALL 응답 객체와 생성에 관여한 `langgraph_spec` 의 어떤 필드에도 셸 명령 문자열과 kubectl 명령 문자열을 포함하지 않는다.
9. IF Knowledge DB 의 `planner_manifest` 항목 등록 또는 조회가 실패하면, THEN THE Multi_Agent_Runtime SHALL 실패 사유를 담은 오류를 반환하고 KPI_Advisor_Agent 를 선택 후보로 노출하지 않는다.
10. IF KPI_Advisor_Agent 가 60초 이내에 응답하지 않거나 오류 응답을 반환하면, THEN THE Planning Agent SHALL 해당 subtask 를 `advisor_unavailable` 로 표시하고 `langgraph_spec` 의 나머지 노드 실행을 계속한다.
11. IF `POST /invoke` 요청 본문이 KPI_Advisor_Agent 입력 스키마 검증에 실패하면, THEN THE KPI_Advisor_Agent SHALL 검증에 실패한 필드 이름 목록을 담은 오류를 반환하고 GNN 프로빙을 수행하지 않는다.

### Requirement 13: 재학습 루프

**User Story:** 연구자로서, GNN 예측이 실제 시뮬레이션 결과와 벌어지면 자동으로 재학습되게 하고 싶다. 그래야 모델이 시나리오 변경을 따라갈 수 있다.

#### Acceptance Criteria

1. WHEN 새 결과 디렉터리가 Feature_Store 에 추가되면, THE Retraining_Controller SHALL 추가 감지 후 60 seconds 이내에 해당 Feature_Record 의 Parameter_Set 에 대한 GNN_Model 예측값을 Inference_Service 에서 조회하고 각 Target_KPI 마다 Prediction_Error 를 percent 단위로 계산한다.
2. WHEN Target_KPI 별 Prediction_Error 가 계산되면, THE Retraining_Controller SHALL 대상 Target_KPI 이름, Prediction_Error 값을 소수점 첫째 자리까지 percent 단위로, 예측에 사용된 GNN_Model 버전, 계산 시각을 기록한다.
3. IF 어떤 Target_KPI 의 Prediction_Error 가 Retraining_Threshold 를 초과하면, THEN THE Retraining_Controller SHALL 해당 Target_KPI 마다 별개의 재학습 작업을 Training_Manager 에 생성하고 각 작업 메타데이터에 대상 Target_KPI 이름과 트리거 시점의 Prediction_Error 값을 기록한다.
4. WHEN 재학습 작업이 성공 상태로 완료되면, THE Retraining_Controller SHALL 신규 모델 버전과 현재 서빙 모델 버전을 동일한 평가 데이터셋으로 산출한 Target_KPI 별 Prediction_Error 값을 비교한다.
5. WHEN 신규 모델 버전의 모든 Target_KPI Prediction_Error 가 현재 서빙 모델 버전의 동일 Target_KPI Prediction_Error 보다 크지 않고 재학습을 트리거한 Target_KPI 의 Prediction_Error 가 1.0 percentage point 이상 낮으면, THE Retraining_Controller SHALL Model_Registry 에서 신규 버전을 서빙 대상으로 표시하고 Inference_Service 의 서빙 모델 버전을 신규 버전으로 전환한다.
6. IF 신규 모델 버전의 어떤 Target_KPI Prediction_Error 가 현재 서빙 모델 버전의 동일 Target_KPI Prediction_Error 보다 높거나 재학습을 트리거한 Target_KPI 의 Prediction_Error 감소 폭이 1.0 percentage point 미만이면, THEN THE Retraining_Controller SHALL 현재 서빙 모델 버전을 유지하고 신규 버전을 Model_Registry 에 미승격 상태로 보존하며 Target_KPI 별 Prediction_Error 비교값을 포함한 전환 거부 사유를 기록한다.
7. IF 설정 파일에 Retraining_Threshold 가 지정되지 않았거나 지정된 값이 0.1 percent 이상 100.0 percent 이하 범위를 벗어나면, THEN THE Retraining_Controller SHALL 기본값 15.0 percent 를 적용하고 기본값을 적용한 사유를 기록한다.
8. WHILE 재학습 작업이 실행 중이면, THE Inference_Service SHALL 기존 서빙 모델 버전으로 예측 응답을 계속 반환한다.
9. IF GNN_Model 예측값 조회가 실패하거나 Feature_Record 에 해당 Target_KPI 실측값이 없으면, THEN THE Retraining_Controller SHALL 해당 Target_KPI 의 Prediction_Error 계산을 건너뛰고 건너뛴 대상과 사유를 기록하며 서빙 모델 버전을 변경하지 않는다.
10. IF 동일 Target_KPI 에 대한 재학습 작업이 이미 실행 중인 상태에서 해당 Target_KPI 의 Prediction_Error 가 Retraining_Threshold 를 다시 초과하면, THEN THE Retraining_Controller SHALL 추가 재학습 작업을 생성하지 않고 중복 트리거 사유를 기록한다.
11. IF 재학습 작업이 실패 상태로 종료되거나 생성 후 24 hours 이내에 완료되지 않으면, THEN THE Retraining_Controller SHALL 해당 작업을 실패로 표시하고 현재 서빙 모델 버전을 유지하며 실패 사유를 기록한다.

### Requirement 14: 대시보드 가시화

**User Story:** 연구자로서, Agent 의 프로빙 결과와 충돌, 시간 계획을 화면에서 보고 싶다. 그래야 판단 과정을 검토하고 발표할 수 있다.

#### Acceptance Criteria

1. WHEN 사용자가 프로빙 결과 화면을 요청하면, THE SMO_Dashboard SHALL Baseline_Parameter_Set 1행과 각 Probe_Variant 1행으로 구성된 표에 각 행의 Target_KPI 예측값과 Baseline_Parameter_Set 대비 변화율을 요청 시점부터 3초 이내에 표시한다.
2. WHEN Degradation_Verdict 가 수신되면, THE SMO_Dashboard SHALL Degradation_Verdict 값(`acceptable`, `degrading`, `unknown` 중 하나)과 위반된 각 Threshold_KPI 의 Target_KPI 이름, 예측값, 임계값을 표시한다.
3. WHEN Indirect_Conflict 목록이 수신되면, THE SMO_Dashboard SHALL 각 Indirect_Conflict 항목에 대해 두 Control_Parameter 이름, 대상 Target_KPI 이름, 두 Marginal_Effect 값, 사용된 Conflict_Threshold 값을 표시한다.
4. IF Indirect_Conflict 개수가 0 이면, THEN THE SMO_Dashboard SHALL 충돌 목록 영역을 0건 상태 메시지와 함께 빈 목록으로 표시한다.
5. WHEN Temporal_Plan 이 수신되면, THE SMO_Dashboard SHALL Time_Step 0 부터 4 까지 오름차순 5행으로 각 Time_Step 의 배정 xApp_Objective 이름, 적용 Parameter_Set, 예측 Target_KPI 값, 위반 표시 여부를 표시한다.
6. WHEN 화면이 표시되면, THE SMO_Dashboard SHALL 현재 서빙 GNN 모델 이름, 모델 버전, 데이터 조회 시각을 초 단위로 표시한다.
7. WHEN 발행된 Policy_Instance 정보가 수신되면, THE SMO_Dashboard SHALL 각 Policy_Instance 의 정책 유형 식별자, 인스턴스 식별자, 상태 값을 표시하며 상태 값이 `status_unavailable` 인 항목도 동일 목록에 포함하여 표시한다.
8. IF KPI_Advisor_Agent 응답이 요청 시점부터 10초 이내에 수신되지 않거나 오류 응답이 수신되면, THEN THE SMO_Dashboard SHALL 모든 데이터 영역을 `데이터 없음` 상태 메시지로 대체하여 표시하고 마지막 성공 조회 시각을 함께 표시하며 화면 영역 구성은 유지한다.
9. IF KPI_Advisor_Agent 응답의 일부 데이터 영역만 누락되면, THEN THE SMO_Dashboard SHALL 누락된 영역만 `데이터 없음` 상태 메시지로 표시하고 수신된 나머지 영역은 해당 영역의 표시 규칙대로 표시한다.
10. IF Probe_Variant 개수가 100 을 초과하면, THEN THE SMO_Dashboard SHALL Probe_Plan 순서상 앞선 100개 Probe_Variant 행만 표시하고 표시되지 않은 잔여 건수를 표시한다.
11. WHILE KPI_Advisor_Agent 응답 수신을 대기 중이면, THE SMO_Dashboard SHALL 각 데이터 영역에 조회 진행 중 상태를 표시하고 직전에 표시된 데이터를 변경하지 않는다.

### Requirement 15: 근거 추적성

**User Story:** 검토자로서, Agent 의 모든 판단이 어떤 데이터와 모델에서 나왔는지 추적하고 싶다. 그래야 결과를 검증하고 재현할 수 있다.

#### Acceptance Criteria

1. WHEN KPI_Advisor_Agent 가 판정을 생성하면, THE Evidence_Record SHALL 사용된 모델 이름, 모델 버전, Probe_Plan, 각 Parameter_Set 의 예측값, 적용된 Threshold_KPI 값, Degradation_Verdict 를 포함한다.
2. WHEN Conflict_Analyzer 가 Indirect_Conflict 를 판정하면, THE Evidence_Record SHALL 해당 판정에 사용된 Marginal_Effect 값과 Conflict_Threshold 값을 포함한다.
3. WHEN Temporal_Scheduler 가 Temporal_Plan 을 생성하면, THE Evidence_Record SHALL 각 Time_Step 배정에 대해 그 배정의 근거가 된 Target_KPI 예측값과 적용된 Control_Parameter 값을 포함한다.
4. WHEN Evidence_Record 가 생성되면, THE SMO_Framework SHALL Evidence_Record 를 MongoDB 에 저장하고, 기존에 저장된 모든 Evidence_Record 의 식별자와 중복되지 않는 고유한 조회용 식별자를 반환한다.
5. IF Evidence_Record 의 MongoDB 저장이 실패하면, THEN THE SMO_Framework SHALL 해당 Evidence_Record 를 부분적으로 저장하지 않고, 저장 실패를 나타내는 오류를 반환한다.
6. IF 예측값 없이 판정이 요청되면, THEN THE KPI_Advisor_Agent SHALL `no_prediction_evidence` 오류를 반환한다.
7. THE KPI_Advisor_Agent SHALL Evidence_Record 에 없는 KPI 수치를 응답 본문에 포함하지 않는다.
5. IF 예측값 없이 판정이 요청되면, THEN THE KPI_Advisor_Agent SHALL `no_prediction_evidence` 오류를 반환한다.
6. THE KPI_Advisor_Agent SHALL Evidence_Record 에 없는 KPI 수치를 응답 본문에 포함하지 않는다.
