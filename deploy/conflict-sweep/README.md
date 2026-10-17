# CPU-only conflict sweep Docker

목적: 500개 conflict 시나리오를 900초씩 실행합니다. E2/RIC, GUI, InfluxDB,
Agent/LLM, GPU 및 실시간 ray tracing은 사용하지 않습니다. 미리 정한 시각에
ns3 내부 스케줄이 TxP/RET/CIO/TTT/HYS를 변경합니다.

## 패키지 구성

- Ubuntu 24.04 Linux amd64 이미지: 현재 로컬에서 빌드된 ns3 실행 파일,
  연결된 ns3 공유 라이브러리, Python NumPy/h5py, sweep 코드와 고정 manifest.
- `inputs.tar`: RET 캐시 30개, UE trace 500개, gNB 위치 1개. 이미지와 분리.
- `bundle_manifest.json`: 소스 commit/dirty 상태와 바이너리·라이브러리·입력 SHA256.
- `results/`: 호스트에 보존되는 결과. 컨테이너를 지워도 유지됩니다.

이는 **검증된 기존 빌드를 배포하는 runtime 이미지**입니다. Docker 안에서
ns3 전체 소스를 처음부터 재빌드하는 개발 이미지가 아닙니다. 서버가 ARM이면
이 amd64 이미지 대신 해당 아키텍처용 별도 빌드가 필요합니다.
Ubuntu Server라는 이유만으로 Git을 사용할 수 없는 것은 아니지만,
아래 전달 방식은 서버의 Git/Python/ns3 설치 없이 사용할 수 있습니다.

## 제작 PC: 준비 및 export

WSL 저장소에서 실행합니다. 기존 실행 중인 시뮬레이션은 종료하지 않습니다.

```bash
cd /home/user/LENA-oran-flexric-smo
python3 scripts/conflict_sweep.py setup --build
python3 deploy/conflict-sweep/prepare.py --output /home/user/agora-cpu-export-v1
docker build -t agora-conflict-sweep:cpu-v1 /home/user/agora-cpu-export-v1/context
cp deploy/conflict-sweep/run.sh deploy/conflict-sweep/README.md /home/user/agora-cpu-export-v1/
docker save -o /home/user/agora-cpu-export-v1/agora-cpu-v1.tar agora-conflict-sweep:cpu-v1
cd /home/user/agora-cpu-export-v1
sha256sum agora-cpu-v1.tar inputs.tar bundle_manifest.json run.sh > SHA256SUMS
```

`prepare.py`는 새 export 디렉터리를 요구하며 기존 결과를 덮어쓰지 않습니다.
원본 build/cache가 없으면 우선 setup을 수행해야 합니다. 비밀 키, .env,
AGORA 모델, 로컬 Python 환경 전체는 이미지로 복사하지 않습니다.
서버에는 `context/`를 제외한 위 파일들을 전달하면 됩니다.

## 서버: Docker 사용 권한과 사양 확인

```bash
uname -m
nproc
free -h
df -h .
docker version
docker info
```

`docker version`에서 Server 부분까지 나와야 합니다. 권한 오류 또는 명령 없음은
서버 관리자에게 Docker 설치/실행 권한을 요청합니다. GPU는 필요 없습니다.

```bash
cd ~/agora-cpu-export-v1
sha256sum -c SHA256SUMS
docker load -i agora-cpu-v1.tar
mkdir -p inputs results
tar --no-same-owner -xf inputs.tar -C inputs
chmod +x run.sh
./run.sh --resources
```

## 실행 순서

먼저 CPU/GPU/메모리 인식과 짧은 기능 테스트를 확인합니다.
smoke는 실제 ns3 20초, 8개 변경 시점, 11개 제어 파라미터를 검사합니다.
`results/smoke`가 이미 있으면 새 결과 폴더를 지정합니다.

```bash
AGORA_RESULTS=./smoke-results ./run.sh --smoke
./run.sh 1 500 1 8 --dry-run
./run.sh 1 500 1 8 --mode preflight --report /opt/agora/results/preflight.json
```

네 인자는 기존과 동일: **시작 ID / 종료 ID / 실행당 논리 CPU 수 / 최대 동시 실행 수**.
ID는 1~500이며 양 끝을 포함합니다. CPU 40개만으로 40병렬을 결정하지 않습니다.
RAM 미확인 상태에서는 8병렬부터 시작하고 peak 구간의 메모리 사용량을 관찰합니다.
초기 20초 smoke만으로 500초 peak 메모리를 보장할 수 없습니다.
기존 실행기의 추정치는 작업당 1.75 GiB + 예비 4 GiB이며 실측 보장이 아닙니다.
예를 들어 32병렬은 추정상 약60 GiB가 필요하므로 64GB 서버에서도 바로 권하지 않습니다.
자원 검사는 affinity와 cgroup CPU quota, host 및 cgroup 잔여 메모리를 함께 확인합니다.
요청 CPU가 quota보다 크면 거부하며, 실행 중 메모리 부족 시 새 작업 시작을 대기합니다.
메모리 증가 중인 기존 작업의 OOM까지 막아주는 보장은 없습니다.

```bash
# 첫 8건으로 900초 전체 구간의 시간/RAM 확인 (이 명령은 실제로 장시간 실행)
AGORA_NAME=agora-pilot AGORA_DETACH=1 ./run.sh 1 8 1 8
docker logs -f agora-pilot
docker stats agora-pilot

# 이후 확인된 사양에 맞춰 병렬 수 지정. 아래는 16병렬 예시.
AGORA_NAME=agora-sweep AGORA_DETACH=1 ./run.sh 1 500 1 16
docker logs -f agora-sweep

# 다른 터미널: 각 실행의 최근 simulation time 확인
./run.sh 1 500 1 16 --mode status
# 완료 후 KPI/제어 영수증 검증 보고서
./run.sh 1 500 1 16 --mode collect --report /opt/agora/results/verified_server.json
```

같은 결과 폴더에서는 검증된 성공 건을 자동으로 건너뜁니다. 실패/중단 건을
재실행하려면 원인을 확인한 뒤 `--retry-failed`를 추가합니다. 동일 출력 폴더에
두 sweep을 동시에 시작하지 마세요. 파일 잠금이 중복 실행을 거부합니다.
중지는 `docker stop -t 60 agora-sweep`. 재개는 동일 명령 + `--retry-failed`.
`--rm` 컨테이너는 종료 후 사라지지만 각 작업 로그/상태는 results에 남습니다.

원한다면 `AGORA_CPUS=16 AGORA_MEMORY=40g`처럼 Docker 상한도 지정합니다.
메모리 값은 서버 용량을 확인한 후 정해야 합니다. 기본적으로 임의 상한을 넣지 않습니다.
프로세스는 호스트 사용자 UID/GID로 실행되고, 입력과 이미지 파일시스템은 읽기 전용이며
네트워크 접근/GPU 장치 전달 없이 동작합니다. 결과 폴더에 쓰기 권한이 있어야 합니다.

결과: `results/jobs/conflict_XXXX/attempt0001/data/conflict_XXXX/`의
`cell_kpi.csv`, `ue_kpi.csv`, `parameter_changes.csv`.
전체 500건은 기본 사전 검사에서 여유 디스크135GiB를 요구합니다(추정치).
입력/export 파일 저장 공간은 별도입니다. 캐시는 이미지에 내장하지 않았으므로
항상 입력 mount가 필요합니다.
