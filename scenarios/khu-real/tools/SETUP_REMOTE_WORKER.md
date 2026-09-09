# GNN 데이터셋 스윕용 원격 컴퓨터 세팅 (최소 버전)

GPU 드라이버/OptiX, GUI, Docker/Grafana, FlexRIC/e2sim 빌드는 전부 건너뜁니다.
RET 캐시가 이미 Sionna 레이트레이싱으로 사전 계산되어 git에 커밋돼 있고,
ns-3는 그 캐시를 `h5py`/`numpy`로 조회만 하기 때문입니다 (GPU도, 실시간
Sionna RT 실행도 필요 없습니다).

## 1) WSL2 배포판 준비 (Windows에서)

```powershell
wsl --install Ubuntu-24.04 --name sionna
wsl -d sionna
```

D 드라이브에 설치하려면:

```powershell
mkdir D:\WSL
wsl --install Ubuntu-24.04 --name sionna --location D:\WSL\sionna
wsl -d sionna
```

## 2) 시스템 패키지 (WSL "sionna" 배포판 안에서)

```bash
sudo apt update
sudo apt install -y \
  build-essential gcc-13 g++-13 cmake cmake-curses-gui ninja-build ccache \
  git pkg-config python3.12 python3.12-dev python3.12-venv \
  llvm \
  libboost-all-dev libgsl-dev libgtk-3-dev libpcre2-dev \
  libsctp-dev lksctp-tools libsqlite3-dev libxml2-dev \
  autoconf automake libtool bison flex
```

## 3) 저장소 clone

저장소가 private이면 clone 시 GitHub 사용자명 + 개인 액세스 토큰(PAT)이
필요합니다 -- **본인 계정에서 직접 발급받은 토큰을 쓰세요, 다른 사람과 토큰을
공유하지 마세요** (Settings -> Developer settings -> Personal access tokens,
`repo` 권한만 체크).

```bash
export REPO_ROOT="$HOME/LENA-oran-flexric-smo"
git clone https://github.com/gksyb4235/sionna-LENA-oran-flexric-smo.git "$REPO_ROOT"
cd "$REPO_ROOT"
```

### 3.1) e2sim (컴파일에 필요 -- 런타임에는 안 씀)

`contrib/nr`가 `contrib/oran-interface`를 무조건 include하고, 거기서 e2sim이
생성하는 ASN.1 헤더(`E2SM-KPM-RANfunction-Description.h` 등)를 참조합니다.
GNN 스윕 자체는 이 RIC/E2 코드를 실행하지 않지만, e2sim 헤더/라이브러리가
`/usr/local`에 없으면 **빌드 자체가 실패**합니다. FlexRIC(실제 RIC 프로세스)는
필요 없지만 e2sim은 건너뛸 수 없습니다.

```bash
export DEPS_ROOT="$HOME/LENA-oran-flexric-smo-deps"
mkdir -p "$DEPS_ROOT"

git clone https://github.com/MinaYonan123/e2sim-kpmv3.git "$DEPS_ROOT/e2sim-kpmv3"
git -C "$DEPS_ROOT/e2sim-kpmv3" checkout --detach acf4f6b2baa8c645af566ea210146abd97de1f48

cd "$DEPS_ROOT/e2sim-kpmv3/e2sim"
mkdir -p build
./build_e2sim.sh 2   # /usr/local에 설치하므로 sudo 비밀번호를 물어볼 수 있음

test -f /usr/local/lib/libe2sim.a && test -d /usr/local/include/e2sim && echo OK
cd "$REPO_ROOT"
```

## 4) Python 가상환경

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/check_environment.py --require-e2
```

## 5) ns-3 설정 및 빌드

`--enable-python-bindings`가 **필요합니다**. 이름과 달리 ns-3 자체 SWIG Python
바인딩을 쓰려는 게 아니라, 이 플래그가 켜지면 ns-3 최상위 CMake 코드가
`find_package(Python3 COMPONENTS Interpreter Development)`를 먼저 실행해서
Development(헤더/라이브러리) 탐지 결과를 캐시에 성공적으로 채워둡니다. 이
플래그 없이 진행하면 나중에 pybind11이 Python을 찾을 때 이 캐시가 없어서
`Python.h: No such file or directory`로 빌드가 실패하는 사례가 실제로
확인됐습니다 (venv에 pip으로 pybind11만 설치돼 있고 시스템에 `python3.12-dev`가
멀쩡히 있어도 이 순서 문제 때문에 실패함).

```bash
./ns3 configure --enable-examples --enable-tests --enable-python-bindings
./ns3 build khu-real-dual-nr-sionna-final
```

출력에 다음 문구가 보이면 정상입니다:

```text
Sionna-RT support enabled: all required dependencies were found.
```

## 6) RET 캐시 확인 (clone만 해도 이미 존재해야 함)

```bash
ls scenarios/khu-real/ret_caches/
# tilt_1deg tilt_3deg tilt_5deg tilt_7deg tilt_9deg tilt_12deg
```

## 7) 스모크 테스트 (짧은 simTime으로 파이프라인 전체 검증)

수동으로 바이너리 인자를 다 채우는 대신, 실제 스윕 스크립트를 아주 작은
범위 + 짧은 simTime으로 한 번 돌려보는 쪽이 훨씬 안전합니다 (인자 하나만
빠져도 바로 실패하니 실전과 동일한 검증이 됩니다):

```bash
GNN_SWEEP_SIM_TIME=15 ./scenarios/khu-real/tools/run_gnn_dataset_sweep.sh 0 0 2 1
```

`scenarios/results/GNN_dataset_sweep/gnn_0000/run.log`에 `FATAL`/`aborted.`가
없고 `cell_kpi.csv`가 만들어졌으면 정상입니다.

## 8) 실제 스윕 실행 (배정받은 구간만 사람마다 다르게)

```bash
./scenarios/khu-real/tools/run_gnn_dataset_sweep.sh <start_idx> <end_idx> <cores_per_job> <parallel_jobs>

# 예: 0~99번(100개), 시뮬레이션당 2코어, 5개 동시 실행(총 10코어)
./scenarios/khu-real/tools/run_gnn_dataset_sweep.sh 0 99 2 5
```

- `cores_per_job * parallel_jobs`가 이 컴퓨터의 논리 코어 수를 넘지 않게 조절하세요.
- 결과는 `scenarios/results/GNN_dataset_sweep/gnn_XXXX/`에 시나리오별로 쌓입니다.
- 중간에 끊겨도 같은 명령을 다시 실행하면 완료된 시나리오는 자동으로 건너뜁니다.
- 끝나면 `scenarios/results/GNN_dataset_sweep/` 폴더를 통째로 보내주세요.
