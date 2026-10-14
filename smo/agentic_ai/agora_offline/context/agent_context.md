# Agent context: RCP–KPI intervention evidence DAG

이 문서는 Agentic AI에 직접 넣을 수 있는 연구 근거 context다. 그래프의 노드는 RCP 11개와 셀별 KPI 42개이며 462개 시험 관계를 담는다.
**화살표는 시험한 설정 변경의 결과 요약이다. 직접 인과 구조나 제어 실행 허가를 나타내지 않는다.**
본문은 요약이다. 전체 KPI에 대한 판단에는 `dag.json` 또는 `query_dag.py`로 관련 관계를 조회하고, `edge_evidence.json`으로 근거를 확인한다. 파일을 제공하지 않거나 조회 도구를 연결하지 않은 에이전트는 이 문서에 없는 값을 알 수 없다.

## 공통 기준과 적용 범위

`{"txp_5g": 33.0, "txp_4g1": 37.0, "txp_4g2": 40.0, "ret_5g": 7.0, "ret_4g1": 9.0, "ret_4g2": 7.0, "cio_5g": 8.0, "cio_4g1": 0.0, "cio_4g2": 0.0, "ttt_ms": 512.0, "hys_db": 1.5}`

세 셀 모두 NR이다. 5g=3.5 GHz, 4g1·4g2=1.8 GHz이며 파일 이름을 LTE라는 뜻으로 읽지 않는다.
개입은 t=0부터 고정한 하나의 RCP 변경, 기간은 900초다. 관측은 150초×6구간과 900초로 집계했다. RngSeed=1, RngRun=1. 입력 구성을 맞췄지만 동일 내부 상태 snapshot과 완전한 사전 외생 수요 manifest는 없다.
원래 trace는 1401·1402·1403. TxP(4g2)만 실패 후 보완한 1401·1403·1404이며, 원래 trace1402의 실패는 보존했다. 세 trace는 모집단 신뢰구간을 제공하지 않고 재사용 기준/시간 구간도 독립 표본이 아니다.

## 에이전트의 근거 사용 규칙

1. Edges are tested finite intervention contrasts at one joint RCP baseline, not structural direct causal links.
2. An edge exists even for numerical zero. Absence from a filtered view means hidden/not retrieved, not no effect.
3. Use observed deltas for effect magnitude; SHAP is model dependence and MAE is predictive error. Never interchange them.
4. Sign means increase/decrease of the measured KPI, not benefit/harm without an explicit objective.
5. Three trace scenarios are not three independent radio-seed repeats; six windows and reused controls are dependent.
6. All-positive/all-negative traces are descriptive repeatability, not statistical significance or calibrated confidence.
7. TxP(4g2) uses 1401,1403,1404 after failure on 1402. Keep original-only and supplemental summaries separate.
8. Do not extrapolate to another joint baseline, reverse/change the step, add simultaneous/sequential effects, or infer online control response.
9. Reference scope is t=0 fixed settings, 900s, audited simulator/trace/cache recipe; matching RCPs alone does not establish transportability.
10. No xApp identities/objectives, operational harm thresholds, or combined-action evidence are supplied. Identify tradeoffs, not confirmed xApp conflicts.
11. Do not use this summary adjacency for d-separation, mediation identification, or causal adjustment-set selection.
12. Derived KPI outputs are labelled and are not independent corroboration. Operational effect thresholds and population confidence intervals are unspecified.
13. Some loss_pct 150s windows were invalid under the existing 0..100% rule. Null deltas are missing, not zero; flag those relations for data-quality review even if their 900s aggregates are valid.
14. Cite edge IDs, action before/after, units and trace ranges when answering. For a decision, report applicability and unresolved context.

## 시험한 행동

| RCP | 변경 | 단위 | trace |
|---|---|---|---|
| txp_5g | 33→37 | dBm | 1401,1402,1403 |
| txp_4g1 | 37→40 | dBm | 1401,1402,1403 |
| txp_4g2 | 40→43 | dBm | 1401,1403,1404 |
| cio_5g | 8→9 | dB | 1401,1402,1403 |
| cio_4g1 | 0→1 | dB | 1401,1402,1403 |
| cio_4g2 | 0→1 | dB | 1401,1402,1403 |
| ttt_ms | 512→720 | ms | 1401,1402,1403 |
| hys_db | 1.5→2 | dB | 1401,1402,1403 |
| ret_5g | 7→9 | degree | 1401,1402,1403 |
| ret_4g1 | 9→12 | degree | 1401,1402,1403 |
| ret_4g2 | 7→9 | degree | 1401,1402,1403 |

## 처리량 영향 요약

단위 Mbit/s. 각 칸은 평균 [최소, 최대] / 방향 분류다. 범위는 신뢰구간이 아니다. 해당 칸의 근거 ID는 `{RCP}__{cell}__goodput_mbps`다.

| 행동 | 5g | 4g1 | 4g2 |
|---|---|---|---|
| txp_5g | +3.3506 [+3.0850, +3.7705] / positive_all_traces | +0.0498 [-0.4502, +0.4481] / mixed_sign | -2.0490 [-2.8461, -1.4795] / negative_all_traces |
| txp_4g1 | -0.0872 [-0.1457, -0.0544] / negative_all_traces | +0.5067 [+0.3815, +0.6532] / positive_all_traces | -0.4421 [-0.5712, -0.3618] / negative_all_traces |
| txp_4g2 | -0.6049 [-0.6959, -0.4636] / negative_all_traces | -0.4038 [-0.9375, +0.0908] / mixed_sign | +0.2335 [-1.6626, +1.2922] / mixed_sign |
| cio_5g | +0.4450 [+0.3116, +0.5564] / positive_all_traces | +0.1144 [-0.1427, +0.4816] / mixed_sign | -0.3990 [-0.4106, -0.3786] / negative_all_traces |
| cio_4g1 | -0.0324 [-0.1092, +0.0330] / mixed_sign | +0.3068 [+0.0857, +0.6843] / positive_all_traces | -0.3689 [-1.0562, +0.0013] / mixed_sign |
| cio_4g2 | -0.2628 [-0.3158, -0.2154] / negative_all_traces | +0.1143 [-0.2188, +0.6878] / mixed_sign | -0.1126 [-0.7851, +0.2998] / mixed_sign |
| ttt_ms | -0.0511 [-0.1120, -0.0053] / negative_all_traces | +0.1556 [+0.1088, +0.2338] / positive_all_traces | -0.1031 [-0.3134, +0.0913] / mixed_sign |
| hys_db | +0.0135 [-0.0767, +0.0880] / mixed_sign | +0.3067 [+0.0017, +0.7356] / positive_all_traces | +0.2152 [-0.1685, +0.5497] / mixed_sign |
| ret_5g | +0.3033 [+0.2608, +0.3361] / positive_all_traces | +0.1614 [-0.0837, +0.4259] / mixed_sign | +0.0925 [-0.1253, +0.4303] / mixed_sign |
| ret_4g1 | +0.0590 [-0.0181, +0.1154] / mixed_sign | +0.1594 [-0.4150, +0.6213] / mixed_sign | +0.1960 [-0.3534, +0.5953] / mixed_sign |
| ret_4g2 | -0.0527 [-0.1487, +0.0250] / mixed_sign | +0.2094 [+0.0397, +0.4783] / positive_all_traces | +0.1798 [-0.1559, +0.4573] / mixed_sign |

## 처리량 외 사전 주 관심 효과

| 행동 | 대상 셀 | KPI | 평균 [최소, 최대] / 방향 | 효과 단위 |
|---|---|---|---|---|
| cio_5g | 5g | ue_share_pct | +2.4207 [+2.1038, +2.8226] / positive_all_traces | pp |
| cio_4g1 | 4g1 | ue_share_pct | +0.5580 [+0.2598, +0.8965] / positive_all_traces | pp |
| cio_4g2 | 4g2 | ue_share_pct | +2.0111 [+1.8029, +2.2702] / positive_all_traces | pp |
| ttt_ms | 5g | ho_attempts | -110.0000 [-121.0000, -100.0000] / negative_all_traces | count/900s |
| ttt_ms | 5g | pingpongs | -42.0000 [-43.0000, -41.0000] / negative_all_traces | count/900s |
| ttt_ms | 5g | rlfs | +1.3333 [+0.0000, +3.0000] / nonnegative_with_zero | count/900s |
| ttt_ms | 4g1 | ho_attempts | -116.0000 [-122.0000, -108.0000] / negative_all_traces | count/900s |
| ttt_ms | 4g1 | pingpongs | -10.0000 [-13.0000, -7.0000] / negative_all_traces | count/900s |
| ttt_ms | 4g1 | rlfs | -0.6667 [-2.0000, +0.0000] / nonpositive_with_zero | count/900s |
| ttt_ms | 4g2 | ho_attempts | -177.6667 [-190.0000, -163.0000] / negative_all_traces | count/900s |
| ttt_ms | 4g2 | pingpongs | -102.3333 [-117.0000, -91.0000] / negative_all_traces | count/900s |
| ttt_ms | 4g2 | rlfs | -0.6667 [-4.0000, +2.0000] / mixed_sign | count/900s |
| hys_db | 5g | ho_attempts | -99.3333 [-106.0000, -95.0000] / negative_all_traces | count/900s |
| hys_db | 5g | pingpongs | -23.6667 [-39.0000, -11.0000] / negative_all_traces | count/900s |
| hys_db | 5g | rlfs | +0.3333 [-1.0000, +1.0000] / mixed_sign | count/900s |
| hys_db | 4g1 | ho_attempts | -61.3333 [-67.0000, -57.0000] / negative_all_traces | count/900s |
| hys_db | 4g1 | pingpongs | -3.3333 [-5.0000, -2.0000] / negative_all_traces | count/900s |
| hys_db | 4g1 | rlfs | -1.0000 [-2.0000, +0.0000] / nonpositive_with_zero | count/900s |
| hys_db | 4g2 | ho_attempts | -137.6667 [-149.0000, -126.0000] / negative_all_traces | count/900s |
| hys_db | 4g2 | pingpongs | -38.6667 [-49.0000, -25.0000] / negative_all_traces | count/900s |
| hys_db | 4g2 | rlfs | +0.3333 [-5.0000, +5.0000] / mixed_sign | count/900s |

## 응답 형식

1. 사용자의 목적 KPI·대상 셀·현재 전체 RCP·제안 변경값·시간 조건을 확인한다.
2. 조건이 맞는 실험인지 `reference_conditions_match / out_of_scope / insufficient_context`로 표시한다. 조건 일치는 새 환경 일반화의 증명이 아니다.
3. 관련 edge ID와 관측된 평균·범위·부호 혼재 여부를 인용한다. 이웃 셀과 다른 목적 KPI의 상충 가능성도 조회한다.
4. 적용 범위를 벗어나면 저장된 결과는 참고 근거로만 제시하고 새 효과 값을 만들어내지 않는다.
5. 목적 함수·허용 저하량·AB 실험이 없으면 최종 xApp 충돌 판정은 미정이라고 기록한다.

권장 응답 필드: `applicability`, `evidence_ids`, `observed_effects`, `tradeoffs`, `uncertainties`, `missing_information`, `decision_status`. 이 context에는 실제 RAN 제어 명령을 수행하는 기능이 없다.
