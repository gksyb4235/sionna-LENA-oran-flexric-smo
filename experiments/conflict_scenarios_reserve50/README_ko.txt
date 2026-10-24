추가 예비 시나리오 501~550 (총50개)
기존 1~500의 CSV 및 실행 manifest는 변경하지 않았습니다.
900초, 2회 변경(180/540초)25개, 4회 변경(180/360/540/720초)25개.
초기 앱 ES/TM/CV/LB/MO 각각10개. 기존500 및 추가50 사이 시퀀스 중복 없음.
같은 파라미터 허용값, 앱별 소유권, 강한 요청 생성 규칙을 사용했습니다.
GNN 선별과 실제 ns3 검증은 아직 하지 않아 screening_tier=unreviewed입니다.
이동 경로는 이미 배포된 seed1000~1049를 재사용합니다. 독립적인 신규 이동 경로50개가 아닙니다.
파라미터 시퀀스는 새로 만들었습니다. 학습/평가 분할 시 같은 trace가 양쪽에 섞이지 않게 하세요.
실패한 기존 실행은 실패 원인과 기록을 보존하고, 추가분을 성공한 대체 결과로 몰래 치환하지 마세요.

서버 xapp_conflict_experiments 폴더에 archive를 두고:
  tar -xzf agora-reserve50.tar.gz
  chmod +x reserve50/run_reserve50.sh
  ./reserve50/run_reserve50.sh 501 550 1 32 --dry-run

기존32병렬 실행이 끝난 후:
  AGORA_CPUS=32 AGORA_MEMORY=80g AGORA_NAME=agora-reserve50 AGORA_DETACH=1 ./reserve50/run_reserve50.sh 501 550 1 32

진행 확인:
  docker logs --tail 50 -f agora-reserve50
  ./reserve50/run_reserve50.sh 501 550 1 32 --mode status
완료 후 검증:
  ./reserve50/run_reserve50.sh 501 550 1 32 --mode collect --report /opt/agora/results/verified_501_550.json

추가 결과는 results-reserve50/jobs/conflict_0501/... 에 저장됩니다.
기존 run.sh/image에 직접 501 550을 넣으면 기존 manifest가 500까지여서 거부됩니다.
반드시 reserve50/run_reserve50.sh를 사용하세요. 기존 이미지/입력 재전송은 필요 없습니다.
통합550 CSV는 열람·분석용이며 실행은 독립 manifest를 사용합니다.
