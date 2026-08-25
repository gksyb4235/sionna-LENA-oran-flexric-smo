# The Network Simulator, Version 3

[![codecov](https://codecov.io/gh/nsnam/ns-3-dev-git/branch/master/graph/badge.svg)](https://codecov.io/gh/nsnam/ns-3-dev-git/branch/master/)
[![Gitlab CI](https://gitlab.com/nsnam/ns-3-dev/badges/master/pipeline.svg)](https://gitlab.com/nsnam/ns-3-dev/-/pipelines)
[![Github CI](https://github.com/nsnam/ns-3-dev-git/actions/workflows/per_commit.yml/badge.svg)](https://github.com/nsnam/ns-3-dev-git/actions)

[![Latest Release](https://gitlab.com/nsnam/ns-3-dev/-/badges/release.svg)](https://gitlab.com/nsnam/ns-3-dev/-/releases)

## License

This software is licensed under the terms of the GNU General Public License v2.0 only (GPL-2.0-only).
See the LICENSE file for more details.

## sionna-LENA-oran-flexric-smo (this fork)

> 이 절은 이 저장소 전용 문서입니다. 아래 [Overview](#overview-an-open-source-project)부터는
> 원본 ns-3 문서입니다.

이 저장소는 ns-3.48, 5G-LENA NR, Sionna RT 2.0.1, O-RAN E2,
Polyscope GUI, InfluxDB/Grafana를 결합한 KHU 캠퍼스 이동 UE 시뮬레이터입니다.
Sionna RT는 별도 서버가 아니라 pybind11을 통해 ns-3 프로세스 안에서 실행됩니다.
현재 `scratch/khu-real-nr-sionna.cc`에서 GUI, E2 접속, KPI 저장,
A3 RSRP handover 및 TTT/HYS sweep을 함께 사용할 수 있습니다.

### 재현 범위와 고정 버전

검증 기준은 Ubuntu 24.04 WSL2, Python 3.12, GCC 13입니다. 현재 이 문서의
GPU 검증 기준은 NVIDIA GeForce RTX 4090입니다. GPU에 따라 실행 시간은 달라지며,
부동소수점/JIT 연산 차이로 임계점 근처의 handover 시각이 완전히 bit-identical하다고
보장할 수는 없습니다. 논문용 결과에는 `git rev-parse HEAD`, `nvidia-smi`,
`python -m pip freeze`와 원본 로그를 같이 보관하십시오.

| 구성 요소 | 재현 버전 |
|---|---|
| ns-3 | 3.48 (이 저장소에 포함) |
| 5G-LENA / O-RAN 연동 코드 | `contrib/`에 포함 |
| Python | 3.12 |
| Sionna RT / Mitsuba / Dr.Jit | 2.0.1 / 3.8.0 / 1.3.1 |
| e2sim source | Orange 통합 저장소가 고정한 `MinaYonan123/e2sim-kpmv3` commit `acf4f6b2` |
| FlexRIC | commit `307e1d0a`, E2AP v1.01, KPM v3.00 |
| Grafana / InfluxDB | 10.4.2 / 1.8-alpine (`monitoring/docker-compose.yml`) |

`.venv`, `build`, InfluxDB 볼륨과 Grafana 런타임 데이터는 Git에 포함되지 않습니다.
반면 KHU XML/mesh, UE/gNB CSV, Grafana provisioning과 dashboard는 포함되어 있습니다.
따라서 이 저장소만 내려받은 뒤 아래 외부 의존성 두 개(e2sim, FlexRIC)를 고정
commit으로 빌드하면 동일한 기능 구성을 재현할 수 있습니다.

### 1. Windows와 WSL2 준비

Windows 11 PowerShell에서 WSL을 설치/갱신하고 Ubuntu 24.04를 사용합니다.

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

Windows 쪽 NVIDIA Game Ready 또는 Studio 드라이버를 최신 버전으로 설치하십시오.
WSL에는 `nvidia-driver-*`나 Linux display driver를 설치하지 않습니다. Windows
드라이버가 CUDA를 WSL로 전달합니다. 이 프로젝트는 CUDA 소스를 컴파일하지 않으므로
별도 CUDA Toolkit (`nvcc`)도 필수 사항이 아닙니다. 자세한 내용은
[NVIDIA CUDA on WSL 공식 가이드](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)를
참조하십시오.

WSL 터미널에서 먼저 GPU가 보이는지 확인합니다.

```bash
nvidia-smi
```

#### WSL2에서 RTX 4090으로 OptiX 사용

`nvidia-smi`가 RTX 4090을 보여도 OptiX는 자동으로 사용할 수 없을 수 있습니다.
WSL2에서 Mitsuba/Dr.Jit가 사용하는 OptiX runtime은 별도의 Ubuntu 패키지나
OptiX SDK가 아니라 NVIDIA driver에서 제공됩니다. 따라서 WSL 안에
`nvidia-driver-*`를 설치하지 않습니다.

이 프로젝트에서 확인한 조합은 RTX 4090, Windows NVIDIA driver 591.86,
Linux driver runtime 590.48.01입니다. OptiX는 GPU 모델별로 별도 버전을 선택하는
방식이 아니므로, 새 환경에서는 Windows driver와 같거나 가까운 Linux driver
runtime을 사용하십시오. WSL2 OptiX는 공식 지원 경로가 아니므로 native Linux보다
재현성이 낮을 수 있습니다. 자세한 배경은 [Mitsuba의 WSL2 OptiX 안내](https://mitsuba.readthedocs.io/en/v3.6.3/src/optix_setup.html)를
참고하십시오.

Linux driver `.run` 파일은 설치하지 않고 압축만 풉니다. 아래 예시는 이 프로젝트의
검증에 사용한 590.48.01 runtime입니다.

```bash
cd "$HOME"
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/590.48.01/NVIDIA-Linux-x86_64-590.48.01.run
bash NVIDIA-Linux-x86_64-590.48.01.run -x --target "$HOME/driver"

mkdir -p "$HOME/driver-dist"
cp "$HOME/driver/libnvoptix.so."* "$HOME/driver-dist/libnvoptix.so.1"
cp "$HOME/driver/libnvidia-ptxjitcompiler.so."* "$HOME/driver-dist/libnvidia-ptxjitcompiler.so.1"
cp "$HOME/driver/libnvidia-rtcore.so."* "$HOME/driver-dist/"
cp "$HOME/driver/libnvidia-gpucomp.so."* "$HOME/driver-dist/"
cp "$HOME/driver/nvoptix.bin" "$HOME/driver-dist/"
explorer.exe "$HOME/driver-dist"
explorer.exe 'C:\Windows\System32\lxss\lib'
```

`driver-dist`의 파일들을 Windows의 `C:\Windows\System32\lxss\lib`에 관리자
권한으로 복사한 뒤 PowerShell에서 WSL을 재시작합니다.

```powershell
wsl --shutdown
```

WSL을 다시 시작한 뒤 `DRJIT_LIBOPTIX_PATH`에는 디렉터리가 아니라 실제
`libnvoptix.so.1` 파일을 지정해야 합니다.

```bash
export DRJIT_LIBOPTIX_PATH="$HOME/driver-dist/libnvoptix.so.1"
export LD_LIBRARY_PATH="$HOME/driver-dist:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

매번 설정하지 않으려면 위 두 `export`를 `~/.bashrc`에 추가합니다. OptiX까지
확인하려면 CUDA backend 검사만으로 충분하지 않으므로 실제 KHU scene을 로드합니다.

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
python - <<'PY'
from sionna.rt import load_scene

load_scene("scenes/khu-real/KHU_Cropped_Sionna_RT.xml")
print("OptiX scene load: OK")
PY
```

NVIDIA GPU가 없는 시스템도 사용할 수 있습니다. 이 경우 Sionna RT는
Mitsuba/Dr.Jit의 LLVM CPU backend를 자동으로 선택합니다. CPU 실행 방법은
[CPU-only 실행](#cpu-only-실행)을 참고하십시오.

GUI는 Windows 11의 WSLg를 사용합니다. `echo "$DISPLAY"`가 비어 있으면 WSL을
업데이트한 뒤 `wsl --shutdown`하고 다시 시작하십시오.

### 2. Ubuntu 패키지 설치

아래 목록은 [ns-3 공식 prerequisite](https://www.nsnam.org/docs/installation/html/system.html)와
e2sim/FlexRIC 빌드 의존성을 합친 것입니다.

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

#### WSL2 Docker Engine 또는 Docker Desktop 사용

이 프로젝트는 WSL2의 Ubuntu 안에 Docker Engine을 직접 설치하거나, Docker Desktop의
WSL integration을 사용하는 두 구성을 지원합니다. WSL2 안에서 Docker Engine을
직접 실행하려면 `/etc/wsl.conf`에 다음 설정이 있어야 systemd가 Docker daemon을
자동으로 시작할 수 있습니다.

```ini
[boot]
systemd=true
```

설정을 새로 추가했다면 Windows PowerShell에서 `wsl --shutdown`을 실행한 뒤 WSL을
다시 시작합니다. 그다음 [Docker 공식 Ubuntu 설치 절차](https://docs.docker.com/engine/install/ubuntu/)에
따라 Docker 저장소와 패키지를 설치합니다.

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<'EOF'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

일반 사용자로 Docker를 실행하려면 사용자를 `docker` 그룹에 추가합니다. 이 그룹은
root 수준 권한을 부여한다는 점에 유의하십시오.

```bash
sudo usermod -aG docker "$USER"
```

명령 후 WSL 터미널을 완전히 닫았다가 다시 연 다음 설치를 확인합니다.

```bash
docker version
docker compose version
docker run --rm hello-world
```

### 3. 소스 내려받기

경로는 자유롭습니다. 이후 명령은 다음 변수를 사용하므로 `/home/user`라는 계정명이
아니어도 됩니다.

```bash
export REPO_ROOT="$HOME/LENA-oran-flexric-smo"
export DEPS_ROOT="$HOME/LENA-oran-flexric-smo-deps"
export FLEXRIC_ROOT="$DEPS_ROOT/flexric"

git clone https://github.com/gksyb4235/sionna-LENA-oran-flexric-smo.git "$REPO_ROOT"
mkdir -p "$DEPS_ROOT"
```

`REPO_ROOT`는 이 프로젝트이고, `DEPS_ROOT`는 Git에 포함되지 않는 e2sim/FlexRIC
외부 소스 전용 디렉터리입니다. 새 터미널에서도 편하게 쓰려면 위 세 `export`를
`~/.bashrc`에 추가합니다.

### 4. e2sim 빌드 및 설치

ns-3의 `contrib/oran-interface`는 `/usr/local/include/e2sim`과
`/usr/local/lib/libe2sim.a`를 링크하므로 ns-3보다 먼저 설치해야 합니다.

```bash
git clone https://github.com/MinaYonan123/e2sim-kpmv3.git \
  "$DEPS_ROOT/e2sim-kpmv3"
git -C "$DEPS_ROOT/e2sim-kpmv3" checkout --detach \
  acf4f6b2baa8c645af566ea210146abd97de1f48

cd "$DEPS_ROOT/e2sim-kpmv3/e2sim"
mkdir -p build
./build_e2sim.sh 2
```

설치 확인:

```bash
test -f /usr/local/lib/libe2sim.a
test -d /usr/local/include/e2sim
```

### 5. FlexRIC nearRT-RIC 빌드 및 설치

e2sim과 프로토콜 버전을 맞추기 위해 반드시 E2AP v1.01, KPM v3.00으로 구성합니다.
아래 commit은 현재 검증된 nearRT-RIC 코어와 동일합니다. 로컬 RET xApp 개발분은
nearRT-RIC 실행에 필요하지 않습니다.

FlexRIC의 RRC monitor 예제는 시스템 `asn1c` 0.9.24가 제공하는 구형 옵션으로는
빌드되지 않습니다. 특히 `-gen-UPER`가 지원되지 않아 `gen-UPER: Invalid argument`
및 `ANY_aper.c: No such file or directory`가 발생합니다. FlexRIC가 사용하는
`mouse07410/asn1c` fork를 먼저 설치합니다.

```bash
if [ ! -d "$DEPS_ROOT/asn1c/.git" ]; then
  git clone https://github.com/mouse07410/asn1c.git "$DEPS_ROOT/asn1c"
fi
git -C "$DEPS_ROOT/asn1c" checkout 940dd5fa9f3917913fd487b13dfddfacd0ded06e
cd "$DEPS_ROOT/asn1c"
autoreconf -iv
./configure --prefix=/opt/asn1c
make -j"$(nproc)"
sudo make install
```

그다음 FlexRIC를 빌드합니다. 이미 `FLEXRIC_ROOT`에 clone이 있으면 `git clone`은
다시 실행하지 말고 `fetch`부터 실행합니다.

```bash
if [ ! -d "$FLEXRIC_ROOT/.git" ]; then
  git clone https://gitlab.eurecom.fr/mosaic5g/flexric.git "$FLEXRIC_ROOT"
fi
git -C "$FLEXRIC_ROOT" fetch origin
git -C "$FLEXRIC_ROOT" checkout --detach \
  307e1d0a5c26751c9e5595805b668a4f91d09550

cmake -S "$FLEXRIC_ROOT" -B "$FLEXRIC_ROOT/build" \
  -DASN1C_EXEC=/opt/asn1c/bin/asn1c \
  -DCMAKE_BUILD_TYPE=Debug \
  -DE2AP_VERSION=E2AP_V1 \
  -DKPM_VERSION=KPM_V3_00 \
  -DXAPP_MULTILANGUAGE=OFF
cmake --build "$FLEXRIC_ROOT/build" -j"$(nproc)" && \
  sudo cmake --install "$FLEXRIC_ROOT/build" && \
  sudo ldconfig
```

`cmake --build`가 실패하면 설치 단계도 실행하지 않습니다. 이전에 실패한 빌드의
오래된 산출물을 설치하지 않도록 `&&`를 유지하십시오.

설치 후 `/usr/local/etc/flexric/flexric.conf`의 `NEAR_RIC_IP`가
`127.0.0.1`인지 확인합니다. RIC를 다른 호스트에서 실행할 때만 해당 IP와 ns-3의
`E2TermIp`를 함께 변경합니다.

### 6. Python 가상환경과 ns-3 빌드

Sionna 공식 문서도 Python 가상환경 사용을 권장합니다. 이 저장소는
[Sionna RT 2.0.1](https://nvlabs.github.io/sionna/installation.html)의 알려진 정상
조합을 `requirements.txt`에 고정합니다.

```bash
cd "$REPO_ROOT"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/check_environment.py --require-e2

./ns3 configure --enable-examples --enable-tests --enable-python-bindings
./ns3 build khu-real-nr-sionna
```

GUI 설정 파일은 upstream 패키지의 `data/` 경로가 Git에서 제외된 경우가 있어,
소스 실행에서는 기본 `base.yaml`을 찾지 못할 수 있습니다. 이 저장소에서는
간단한 로컬 설정 파일을 만들어 명시적으로 전달합니다.

```bash
if [ ! -f "$REPO_ROOT/gui/local.yaml" ]; then
  cat > "$REPO_ROOT/gui/local.yaml" <<'EOF'
rendering:
  envmap: null
EOF
fi
```

configure 출력에 다음 문구가 있어야 합니다.

```text
Sionna-RT support enabled: all required dependencies were found.
```

NVIDIA GPU 사용자는 CUDA backend까지 필수 조건으로 검사할 수 있습니다.

```bash
python scripts/check_environment.py --require-gpu --require-e2
```

이 검사가 실패하면 ns-3 빌드 문제가 아니라 Windows 드라이버/WSL GPU 전달 문제부터
해결해야 합니다. CPU 동작만 확인할 때는 `--require-gpu`를 사용하지 않습니다.

#### CPU-only 실행

NVIDIA GPU가 없어도 같은 `requirements.txt`와 ns-3 바이너리를 사용합니다. 별도의
`--cpu` 시나리오 옵션은 없습니다. Sionna RT가 CUDA를 찾지 못하면
`llvm_ad_mono_polarized` Mitsuba variant를 자동으로 선택합니다. Dr.Jit LLVM
backend에는 LLVM 11 이상이 필요하며, 위 Ubuntu 패키지 단계의 `llvm`이 이를
설치합니다.

CPU 환경에서는 GPU 검사를 강제하지 않고 다음과 같이 확인합니다.

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
python scripts/check_environment.py --require-e2

python - <<'PY'
import sionna.rt
import mitsuba as mi
print("Mitsuba variant:", mi.variant())
PY
```

정상적인 CPU 출력은 `Mitsuba variant: llvm_ad_mono_polarized`입니다. 이후 build와
시나리오 실행 명령은 GPU 환경과 같습니다. 다만 KHU 전체 scene의 ray tracing은
CPU에서 훨씬 느릴 수 있으므로 사전 동작 확인에는 `--sionnaUpdatePeriod=5s` 또는
`10s`를 권장합니다. GPU 실험과 KPI를 비교할 때는 update period, path solver 옵션,
seed를 양쪽에서 동일하게 유지해야 합니다.

간단한 headless smoke test:

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
./ns3 run --no-build "khu-real-nr-sionna \
  --sumoTrace=scenarios/khu-real/ue-handover-1.csv \
  --N_Ues=1 --numerologyBwp1=0 --sionnaUpdatePeriod=1s --simTime=2"
```

### 7. 전체 시뮬레이션 실행

먼저 Grafana/InfluxDB를 시작합니다.

```bash
cd "$REPO_ROOT/monitoring"
docker compose up -d
```

- InfluxDB: <http://localhost:8086>
- Grafana: <http://localhost:3000> (`admin` / `admin`)

터미널 1 — GUI:

```bash
cd "$REPO_ROOT/gui"
source "$REPO_ROOT/.venv/bin/activate"
python scripts/run.py \
  --config "$REPO_ROOT/gui/local.yaml" \
  "$REPO_ROOT/scenes/khu-real/KHU_Cropped_Sionna_RT.xml"
```

터미널 2 — nearRT-RIC:

```bash
cd "$FLEXRIC_ROOT"
./build/examples/ric/nearRT-RIC \
  -c /usr/local/etc/flexric/flexric.conf \
  -p /usr/local/lib/flexric/
```

터미널 3 — UE 1대, A3 handover, GUI/E2/KPI 통합 시나리오:

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
./build/scratch/ns3.48-khu-real-nr-sionna-default \
  --gnbPositions=scenarios/khu-real/gnbs-ret.csv \
  --sumoTrace=scenarios/khu-real/ue-handover-1.csv \
  --N_Ues=1 \
  --numerologyBwp1=0 --sionnaUpdatePeriod=1s --simTime=250 \
  --ns3::NrHelper::HandoverAlgorithm=ns3::NrA3RsrpHandoverAlgorithm \
  --handoverTtt=256 --handoverHysteresis=6 \
  --guiSrc="$REPO_ROOT/scratch" --guiHost=localhost \
  --influxSrc="$REPO_ROOT/scratch" \
  --influxHost=localhost --influxPort=8086 --influxDb=nr_kpi \
  --kpiReportInterval=1.0 \
  --ns3::NrHelper::E2ModeNr=true \
  --ns3::NrHelper::E2TermIp=127.0.0.1 \
  --ns3::NrHelper::E2LocalPort=39100
```

현재 sweep에서 안정성이 좋았던 기준값은 TTT 256 ms, HYS 6 dB입니다. 비교
실험에서는 `--handoverTtt`와 `--handoverHysteresis`만 바꾸고 나머지 입력, seed,
시뮬레이션 시간을 고정하십시오. `experiments/ho-sweep-20260804/`에 로그와 분석
스크립트가 있습니다.

### 문제 해결

- `libe2sim.a` 또는 E2 헤더를 찾지 못함: 4단계를 먼저 실행하고 ns-3를 다시
  configure합니다.
- `Cannot assign requested address`: 이전 SCTP 연결이 남아 있을 수 있습니다.
  nearRT-RIC/ns-3를 종료한 뒤 잠시 기다리거나 `E2LocalPort`를 다른 값으로 바꿉니다.
- GUI 창이 안 뜸: WSLg의 `DISPLAY`와 Windows 방화벽, ZMQ 포트 5600/5601을
  확인합니다.
- `Could not initialize OptiX`: `nvidia-smi`가 GPU를 보여도 OptiX runtime이
  준비되지 않았을 수 있습니다. `DRJIT_LIBOPTIX_PATH`가
  `/home/.../driver-dist/libnvoptix.so.1`처럼 실제 파일을 가리키는지, 그리고
  `LD_LIBRARY_PATH`에 `driver-dist`가 포함되는지 확인한 뒤 WSL을 재시작합니다.
- `Config file not found: .../base.yaml`: 소스 GUI의 기본 `data/` 설정이 없을 수
  있습니다. 위의 `gui/local.yaml`을 만들고 `--config "$REPO_ROOT/gui/local.yaml"`
  을 사용합니다.
- `ModuleNotFoundError: No module named 'zmq_bridge'`: `--guiSrc`는 GUI 디렉터리가
  아니라 `"$REPO_ROOT/scratch"`여야 합니다. `--influxSrc`도 같은 디렉터리를
  사용해야 하며, `/home/user/...`처럼 다른 계정의 경로를 하드코딩하지 않습니다.
- Docker pull 중 `docker-credential-desktop.exe: exec format error`: WSL의
  `~/.docker/config.json`이 Windows credential helper를 가리키는 경우입니다.
  공개 이미지 사용 시 파일을 `{}`로 바꾸고 `docker compose pull`을 다시 실행합니다.
- Grafana가 비어 있음: 시나리오의 `influxDb`와 dashboard datasource/database가
  같은지 확인하고 `docker compose logs influxdb grafana`를 봅니다.
- 재빌드 후에도 예전 동작: 가상환경을 활성화한 상태에서 `./ns3 configure`를 다시
  실행한 다음 목표 바이너리를 빌드합니다.

### 구성 요소 연결

- ns-3 ↔ Sionna RT: 같은 프로세스 안의 pybind11 호출입니다.
- ns-3 ↔ GUI: ZMQ 5600/5601로 위치를 전달합니다.
- ns-3 ↔ nearRT-RIC: SCTP 기반 E2AP/E2SM 연결입니다.
- ns-3 → InfluxDB → Grafana: KPI 저장 및 dashboard 표시 경로입니다.

## Table of Contents

* [sionna-LENA-oran-flexric-smo (this fork)](#sionna-lena-oran-flexric-smo-this-fork)
* [Overview](#overview-an-open-source-project)
* [Software overview](#software-overview)
* [Getting ns-3](#getting-ns-3)
* [Building ns-3](#building-ns-3)
* [Testing ns-3](#testing-ns-3)
* [Running ns-3](#running-ns-3)
* [ns-3 Documentation](#ns-3-documentation)
* [Working with the Development Version of ns-3](#working-with-the-development-version-of-ns-3)
* [Contributing to ns-3](#contributing-to-ns-3)
* [Reporting Issues](#reporting-issues)
* [Asking Questions](#asking-questions)
* [ns-3 App Store](#ns-3-app-store)

> **NOTE**: Much more substantial information about ns-3 can be found at
<https://www.nsnam.org>

## Overview: An Open Source Project

ns-3 is a free open source project aiming to build a discrete-event
network simulator targeted for simulation research and education.
This is a collaborative project; we hope that
the missing pieces of the models we have not yet implemented
will be contributed by the community in an open collaboration
process. If you would like to contribute to ns-3, please check
the [Contributing to ns-3](#contributing-to-ns-3) section below.

This README excerpts some details from a more extensive
tutorial that is maintained at:
<https://www.nsnam.org/documentation/latest/>

## Software overview

From a software perspective, ns-3 consists of a number of C++
libraries organized around different topics and technologies.
Programs that actually run simulations can be written in
either C++ or Python; the use of Python is enabled by
[runtime C++/Python bindings](https://cppyy.readthedocs.io/en/latest/).  Simulation programs will
typically link or import the ns `core` library and any additional
libraries that they need.  ns-3 requires a modern C++ compiler
installation (g++ or clang++) and the [CMake](https://cmake.org) build system.
Most ns-3 programs are single-threaded; there is some limited
support for parallelization using the [MPI](https://www.nsnam.org/docs/models/html/distributed.html) framework.
ns-3 can also run in a real-time emulation mode by binding to an
Ethernet device on the host machine and generating and consuming
packets on an actual network.  The ns-3 APIs are documented
using [Doxygen](https://www.doxygen.nl).

The code for the framework and the default models provided
by ns-3 is built as a set of libraries. The libraries maintained
by the open source project can be found in the `src` directory.
Users may extend ns-3 by adding libraries to the build;
third-party libraries can be found on the [ns-3 App Store](https://www.nsnam.org)
or elsewhere in public Git repositories, and are usually added to the `contrib` directory.

## Getting ns-3

ns-3 can be obtained by either downloading a released source
archive, or by cloning the project's
[Git repository](https://gitlab.com/nsnam/ns-3-dev.git).

Starting with ns-3 release version 3.45, there are two versions
of source archives that are published with each release:

1. ns-3.##.tar.bz2
1. ns-allinone-3.##.tar.bz2

The first archive is simply a compressed archive of the same code
that one can obtain by checking out the release tagged code from
the ns-3-dev Git repository.  The second archive consists of
ns-3 plus additional contributed modules that are maintained outside
of the main ns-3 open source project but that have been reviewed
by maintainers and lightly tested for compatibility with the
release.  The contributed modules included in the `allinone` release
will change over time as new third-party libraries emerge while others
may lose compatibility with the ns-3 mainline (e.g., if they become
unmaintained).

## Building ns-3

As mentioned above, ns-3 uses the CMake build system, but
the project maintains a customized wrapper around CMake
called the `ns3` tool.  This tool provides a
[Waf-like](https://waf.io) API
to the underlying CMake build manager.
To build the set of default libraries and the example
programs included in this package, you need to use the
`ns3` tool. This tool provides a Waf-like API to the
underlying CMake build manager.
Detailed information on how to use `ns3` is included in the
[quick start guide](doc/installation/source/quick-start.rst).

Before building ns-3, you must configure it.
This step allows the configuration of the build options,
such as whether to enable the examples, tests and more.

To configure ns-3 with examples and tests enabled,
run the following command on the ns-3 main directory:

```shell
./ns3 configure --enable-examples --enable-tests
```

Then, build ns-3 by running the following command:

```shell
./ns3 build
```

By default, the build artifacts will be stored in the `build/` directory.

### Supported Platforms

The current codebase is expected to build and run on the
set of platforms listed in the [release notes](RELEASE_NOTES.md)
file.

Other platforms may or may not work: we welcome patches to
improve the portability of the code to these other platforms.

## Testing ns-3

ns-3 contains test suites to validate the models and detect regressions.
To run the test suite, run the following command on the ns-3 main directory:

```shell
./test.py
```

More information about ns-3 tests is available in the
[test framework](doc/manual/source/test-framework.rst) section of the manual.

## Running ns-3

On recent Linux systems, once you have built ns-3 (with examples
enabled), it should be easy to run the sample programs with the
following command, such as:

```shell
./ns3 run simple-global-routing
```

That program should generate a `simple-global-routing.tr` text
trace file and a set of `simple-global-routing-xx-xx.pcap` binary
PCAP trace files, which can be read by `tcpdump -n -tt -r filename.pcap`.
The program source can be found in the `examples/routing` directory.

## Running ns-3 from Python

If you do not plan to modify ns-3 upstream modules, you can get
a pre-built version of the ns-3 python bindings. It is recommended
to create a python virtual environment to isolate different application
packages from system-wide packages (installable via the OS package managers).

```shell
python3 -m venv ns3env
source ./ns3env/bin/activate
pip install ns3
```

If you do not have `pip`, check their documents
on [how to install it](https://pip.pypa.io/en/stable/installation/).

After installing the `ns3` package, you can then create your simulation python script.
Below is a trivial demo script to get you started.

```python
from ns import ns

ns.LogComponentEnable("Simulator", ns.LOG_LEVEL_ALL)

ns.Simulator.Stop(ns.Seconds(10))
ns.Simulator.Run()
ns.Simulator.Destroy()
```

The simulation will take a while to start, while the bindings are loaded.
The script above will print the logging messages for the called commands.

Use `help(ns)` to check the prototypes for all functions defined in the
ns3 namespace. To get more useful results, query specific classes of
interest and their functions e.g., `help(ns.Simulator)`.

Smart pointers `Ptr<>` can be differentiated from objects by checking if
`__deref__` is listed in `dir(variable)`. To dereference the pointer,
use `variable.__deref__()`.

Most ns-3 simulations are written in C++ and the documentation is
oriented towards C++ users. The ns-3 tutorial programs (`first.cc`,
`second.cc`, etc.) have Python equivalents, if you are looking for
some initial guidance on how to use the Python API. The Python
API may not be as full-featured as the C++ API, and an API guide
for what C++ APIs are supported or not from Python do not currently exist.
The project is looking for additional Python maintainers to improve
the support for future Python users.

## ns-3 Documentation

Once you have verified that your build of ns-3 works by running
the `simple-global-routing` example as outlined in the [running ns-3](#running-ns-3)
section, it is quite likely that you will want to get started on reading
some ns-3 documentation.

All of that documentation should always be available from
the ns-3 website: <https://www.nsnam.org/documentation/>.

This documentation includes:

* a tutorial
* a reference manual
* models in the ns-3 model library
* a wiki for user-contributed tips: <https://www.nsnam.org/wiki/>
* API documentation generated using doxygen: this is
  a reference manual, most likely not very well suited
  as introductory text:
  <https://www.nsnam.org/doxygen/index.html>

## Working with the Development Version of ns-3

If you want to download and use the development version of ns-3, you
need to use the tool `git`. A quick and dirty cheat sheet is included
in the manual, but reading through the Git
tutorials found in the Internet is usually a good idea if you are not
familiar with it.

If you have successfully installed Git, you can get
a copy of the development version with the following command:

```shell
git clone https://gitlab.com/nsnam/ns-3-dev.git
```

However, we recommend to follow the GitLab guidelines for starters,
that includes creating a GitLab account, forking the ns-3-dev project
under the new account's name, and then cloning the forked repository.
You can find more information in the [manual](https://www.nsnam.org/docs/manual/html/working-with-git.html).

## Contributing to ns-3

The process of contributing to the ns-3 project varies with
the people involved, the amount of time they can invest
and the type of model they want to work on, but the current
process that the project tries to follow is described in the
[contributing code](https://www.nsnam.org/developers/contributing-code/)
website and in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Reporting Issues

If you would like to report an issue, you can open a new issue in the
[GitLab issue tracker](https://gitlab.com/nsnam/ns-3-dev/-/issues).
Before creating a new issue, please check if the problem that you are facing
was already reported and contribute to the discussion, if necessary.

## Asking Questions

ns-3 has an official [ns-3-users message board](https://groups.google.com/g/ns-3-users)
where the community asks questions and share helpful advice.
Additionally, ns-3 has the [ns-3 Zulip chat](https://ns-3.zulipchat.com/), used to discuss
development issues and questions among maintainers and the community.

Please use the above resources to ask questions about ns-3, rather than creating issues.

## ns-3 App Store

The official [ns-3 App Store](https://apps.nsnam.org/) is a centralized directory
listing third-party modules for ns-3 available on the Internet.

More information on how to submit an ns-3 module to the ns-3 App Store is available
in the [ns-3 App Store documentation](https://www.nsnam.org/docs/contributing/html/external.html).
