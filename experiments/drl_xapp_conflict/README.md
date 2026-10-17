# DRL Mobility / Load Balancing xApp 충돌 실험

기존 AGORA, GNN, 시나리오와 18개 two-switch 결과를 보존하는 별도 연구 실험이다.
원본 시나리오를 복사한 신규 타깃에서 실제 ns3 관측을 받아 실제 CIO/TTT/HYS를
변경한다. 실제 ns3 다음 구간 KPI가 강화학습 보상이 된다. GNN 예측이나 사전
고정 스케줄을 DRL의 실행 결과로 사용하지 않는다.

**현재 설계는 파일럿이다. 학습 수렴이나 성능 향상을 미리 보장하지 않는다.**
실행 상태는 `campaign_status.json`, 각 실행의 실제 결과는 `results/`에서 확인한다.
단위 테스트 및 25초 기능 테스트 결과를 900초 성능평가 결과와 구분한다.

## xApp과 강도

| xApp | 제어 | 강도 / 행동 |
|---|---|---|
| Load Balancing | 셀별 CIO | 셀 하나를 골라 현재 값에서 +/-1,2,3,4,5,6 dB 변경; 절대 범위 [-6,6] dB |
| Mobility Optimization | 모든 셀 공통 TTT와 HYS | TTT=[40,64,128,256,512,1024] ms, HYS=[0,1,2,3,4,6] dB의 36개 조합 |

각 xApp은 hold를 포함해 37개 행동을 가진다. CIO는 3셀 * 2방향 * 6강도다.
범위 끝에서 중복되는 CIO 행동 및 실제 변화 없는 행동은 마스킹한다.
두 정책은 같은 관측에서 독립적으로 행동을 계산한다. 제어 파라미터가 겹치지 않아도
HO 판정을 통해 충돌할 수 있다. TTT와 HYS의 큰 값은 HO를 더 보수적으로 만드는
강도이며, 항상 더 좋은 성능을 의미하지 않는다.

## 신경망과 학습

- 각 xApp: 상태 54차원, 은닉층 64/64, 출력 37의 MLP; 학습 가중치 10,085개.
- Double DQN: replay buffer, online network의 행동 선택, target network의 가치 평가,
  terminal bootstrap 제거, Huber loss, gradient clipping.
- 관측은 직전 30초 및 90초의 셀 PRB/UE/Goodput와 네트워크 Jain, HO, 핑퐁,
  RLF, 전달량/발생량, 지연 및 현재 CIO/TTT/HYS/시각이다. 미래 행을 읽지 않는다.
- 실제 제어 간격은 30초이고, 다음 30초의 실제 KPI로 직전 행동을 학습한다.
- 체크포인트는 가중치뿐 아니라 optimizer, replay, 난수 상태, 탐색 진행도를 저장한다.
- 파일럿은 정책당 3개 900초 episode, 실제 전이 72개다. 배치 16, warm-up 16,
  전이당 gradient update 4회를 사용한다. 재사용한 replay update를 새 환경 표본으로 세지 않는다.
- 파일럿 epsilon은 0.9에서 시작해 96전이에 걸쳐 0.1까지 감소한다. 72전이 후 0.3이다.
  이는 장기 학습의 기본 설정이 아니라 작은 파일럿에 고정한 설정이다.
- 평가는 epsilon=0, optimizer/replay update 없이 동결된 가중치로만 실행한다.

## 비교 구현

`local`과 `guarded`는 같은 신경망/행동 공간을 사용하는 별도 보상 설계다.
LB의 기본 목표는 PRB Jain, Mobility의 기본 목표는 핑퐁/HO 시도 비율이다.
두 버전 모두 최소 RLF/전달 성능/HO 실패 벌점을 둔다. `guarded`는 벌점을 강화하고
상대 xApp의 KPI도 고려한다. 완전히 무제약인 정책을 일부러 만들어 충돌을 유도하지 않는다.
정확한 계수와 모든 항목은 `python/drl_xapps/rewards.py` 및 실행 로그에 남긴다.

추가로 PRB 불균형에 따라 CIO를 조절하고 핑퐁/RLF에 따라 TTT/HYS를 조절하는
명시적 휴리스틱 구현을 둔다. 휴리스틱 결과를 DRL 결과로 표시하지 않는다.

## 64개 파일럿 실행 계획

| 단계 | 구성 | ns3 실행 수 |
|---|---|---:|
| 학습 | 2 xApps * 2 보상 * 2 학습 seed * 3 이동 trace | 24 |
| 정책 선택 | 8 학습 정책 * 2 validation trace | 16 |
| 동결 평가 | 8 비교군 * 3 test trace | 24 |

학습 trace는 1414~1416, validation은 1417~1418, test는 1419~1421이다.
이미 충돌을 분석했던 1406/1410/1412는 최종 test로 재활용하지 않는다.
모든 실행은 900초, 1초 KPI, 동일 트래픽 설정, TxP=43 dBm/RET=9도를 유지한다.
초기 CIO는 [0,0,0] dB, TTT=128 ms, HYS=1 dB다. 최대 동시 ns3 수는 8개이며
가용 RAM 5.5 GiB 미만에서는 새 실행을 보류한다. 해당 메모리 기준은 추정치다.

평가 비교군은 none, LB-local, MO-local, both-local, LB-guarded, MO-guarded,
both-guarded, heuristic-both이다. **LB는 180초부터, MO는 540초부터** 동작한다.
단독 MO도 540초에 시작한다. 학습/validation 단독 앱은 180초부터 실행하므로
MO의 배포 시점 차이에 대한 일반화는 이 파일럿의 제한으로 함께 보고한다.

정책은 validation의 공통 app별 목표/벌점 점수로 학습 seed 두 개 중 하나를 고른다.
trace별 전체 평가 구간의 합계 기반 지표를 먼저 구하고, 두 trace의 점수를 같은
비중으로 평균한다. test 결과를 보고 정책을 선택하지 않는다. 선택한 파일의 SHA256은
`selection.json`에 기록한다. 어느 버전이 더 좋은지는 test의 원래 KPI를 함께 보고 판단한다.

## 충돌 판정

1. 주요 평가 구간은 [540,720), [720,900)이다. 30초 제어와 180초 공식 집계를 구분한다.
2. 같은 trace/시각의 both vs LB-only, both vs MO-only를 비교한다.
3. 2x2 상호작용은 `both - LB - MO + none`으로 보고한다. Jain의 음수 상호작용은
   불리한 방향, 핑퐁 비율의 양수 상호작용은 불리한 방향이다.
4. 540초 이전 none/MO와 LB/both의 원본 KPI 일치 여부도 확인한다.
5. 핑퐁 비율은 분자/분모를 합친 뒤 계산하며 HO 시도 10회 미만에서는 공식 비율을
   불충분 표본으로 표시한다. 학습에는 Beta(1,9) 추정과 표본 신뢰도를 사용한다.
6. RLF 원시 횟수와 1,000 UE-seconds당 비율, HO 성공/실패, Goodput, 전달량/발생량,
   지연도 함께 제시한다. HO 자체를 막거나 실패시켜 비율만 낮춘 정책을 좋은 정책으로 보지 않는다.

현재 ns3 핑퐁 정의는 성공한 A-B-A 복귀의 간격이 **1초 미만**인 경우다.
최대 TTT가 1,024 ms이므로 이 정의에 대한 민감도를 보고해야 한다. 핑퐁 하나만으로
일반적인 이동성 품질 개선을 주장하지 않는다. 전달량/발생량은 같은 구간의 집계 proxy로,
동일 패킷 cohort의 손실률이나 AGORA의 미래 기준 대비 처리량 보존율과 다르다.

원본 KPI 보고 시각은 0.5, 1.5, ...초다. 정수 초 제어에 대한 보상 창은 이 원본 행
시각으로 집계하므로 물리적 interval counter 구간과 약 0.5초 경계 차이가 있다.
제어 후 첫 보고 행에 변경 전 최대 0.5초가 섞일 수 있다. 미래 관측을 상태에 넣지는
않으며, 1초 이하의 정밀한 제어 효과를 이 집계로 주장하지 않는다.

## 실제 적용 범위

연구용 제어 전송은 실험별 로컬 파일 lockstep이다. E2/RIC 전송 성능을 검증하는 실험은
아니다. 다른 시뮬레이션이나 실행 중인 RIC에 제어를 전송하지 않는다.
CIO는 모든 UE RRC 객체에 적용·readback한다. TTT/HYS는 실제 gNB A3 설정을 변경하고
연결 UE의 RRC 재구성을 예약한다. readback은 gNB 설정과 예약 확인이며, 모든 UE의
무선 메시지 수신이 같은 순간에 완료됐다는 뜻은 아니다. 동일 값에는 재설정을 하지 않는다.

기존 원본 시나리오/라이브러리는 수정하지 않는다. 신규 실행 타깃은
`scratch/drl-xapp-conflict/`, 스냅샷 실행 파일은 이 폴더의 `bin/`이다.

## 실행과 결과 확인

WSL 위치: `/home/user/LENA-oran-flexric-smo/experiments/drl_xapp_conflict`

```bash
cd /home/user/LENA-oran-flexric-smo
.venv/bin/python experiments/drl_xapp_conflict/campaign.py status
.venv/bin/python experiments/drl_xapp_conflict/analysis/report.py --root experiments/drl_xapp_conflict
```

Windows에서는 이 폴더의 `status.ps1`로 조회한다. 이미 시작한 실험은 `launch.py`를
재실행하지 않는다. 실행 결과를 덮어쓰거나 자동으로 기존 작업을 재시작하지 않는다.

각 `results/<job_id>/`에는 `execution.json`, 1초 `cell_kpi.csv`, `ue_kpi.csv`,
`control/actions.jsonl`, `control/receipts.jsonl`, `control/summary.json`, 학습 체크포인트,
`ns3/simulation/run.log`가 있다. 코드 테스트, 제어 기능 테스트, 학습, validation,
test를 별도 식별자로 보존한다. 한 실행이 실패하면 관련 오류와 부분 결과를 보존하고
캠페인을 중지하므로 실패 이유를 확인한 후 새 계획으로 재개해야 한다.

Double DQN 알고리즘 근거: https://arxiv.org/abs/1509.06461
