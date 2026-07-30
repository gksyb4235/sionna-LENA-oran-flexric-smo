# The Network Simulator, Version 3

[![codecov](https://codecov.io/gh/nsnam/ns-3-dev-git/branch/master/graph/badge.svg)](https://codecov.io/gh/nsnam/ns-3-dev-git/branch/master/)
[![Gitlab CI](https://gitlab.com/nsnam/ns-3-dev/badges/master/pipeline.svg)](https://gitlab.com/nsnam/ns-3-dev/-/pipelines)
[![Github CI](https://github.com/nsnam/ns-3-dev-git/actions/workflows/per_commit.yml/badge.svg)](https://github.com/nsnam/ns-3-dev-git/actions)

[![Latest Release](https://gitlab.com/nsnam/ns-3-dev/-/badges/release.svg)](https://gitlab.com/nsnam/ns-3-dev/-/releases)

## License

This software is licensed under the terms of the GNU General Public License v2.0 only (GPL-2.0-only).
See the LICENSE file for more details.

## sionna-LENA-oran-flexric-smo (this fork)

> Everything in this section is specific to this project. The rest of the
> README (starting at [Overview](#overview-an-open-source-project) below) is
> the stock upstream ns-3 documentation, kept as-is for reference.

### What this is

NR-based rebuild of `ns-O-RAN-flexric` (a sibling local project, at
`/home/user/ns-O-RAN-flexric`) on top of
**ns-3.48**, which is the first ns-3 release with an **official, upstream
Sionna RT channel model** (`src/spectrum/model/sionna-rt-channel-model.*`,
merged via GitLab MR `!2608`), plus the [`nr`](https://github.com/cttc-lena/nr)
module at branch `5g-lena-v5.0.y`, which already ships
`NrChannelHelper::ChannelModel::SionnaRT` and an example
(`contrib/nr/examples/cttc-nr-demo-sionna-rt.cc`) wiring the two together.

The old project (`ns-O-RAN-flexric`) used the legacy `mmwave` module plus a
hand-rolled ZMQ/protobuf bridge to a separate Sionna Python server process.
Here, Sionna RT runs **embedded inside the ns-3 process itself** via
pybind11 (`py::scoped_interpreter`) — no separate server, no network
protocol for the channel path. On top of that NR+Sionna base, this fork
adds back everything the old project had:

- The real KHU campus scene and workload (2 gNBs, 15 moving pedestrians),
  ported from `ns-O-RAN-flexric`'s CSV-driven scenario.
- The Polyscope 3D GUI for watching gNB/UE positions live.
- `contrib/oran-interface` (`ns3-o-ran-e2`) + a patch to `contrib/nr` wiring
  `NrGnbNetDevice`/`NrHelper` to an `E2Termination`, so gNBs can register
  with FlexRIC's nearRT-RIC over E2AP/E2SM, ported from the old project's
  fork of the legacy `mmwave` module.

**Status**: the GUI+real-workload path and the RIC/E2 handshake path have
each been verified independently (see below), but not yet run together in
one scenario — `scratch/khu-real-nr-sionna.cc` (the real moving-UE workload)
doesn't set up `E2Termination` yet. Actual KPM report generation and
RIC-Control message dispatch on the NR side are also not implemented yet
(the old project's legacy `mmwave` net device had this; the `nr` module
equivalent doesn't exist upstream and hasn't been written here).

### Layout

```
LENA-oran-flexric-smo/
├── src/spectrum/model/sionna-rt-channel-model.{h,cc}  # official Sionna RT channel model
├── contrib/
│   ├── nr/            # 5g-lena-v5.0.y + local E2Termination/NrHelper patch, vendored directly
│   └── oran-interface/  # ns3-o-ran-e2 (E2AP/E2SM), vendored directly
├── scratch/
│   ├── khu-real-nr-sionna.cc  # CSV-driven real KHU workload (moving UEs, GUI bridge)
│   └── zmq_bridge.py          # standalone ZMQ client used by khu-real-nr-sionna.cc
├── gui/                # standalone copy of the Polyscope GUI (own venv-installable package)
├── scenes/khu-real/    # KHU campus Sionna RT scene (XML + meshes)
├── scenarios/khu-real/ # gNB positions + UE movement traces (CSV)
└── .venv/              # project Python env (sionna-rt 2.0.1, gitignored)
```

`contrib/nr` and `contrib/oran-interface` were originally separate git
clones; they're vendored here as plain tracked directories (their prior
standalone histories, including the `local/oran-e2-integration` E2 patch
branch, are preserved outside this tree, not inside it). The `origin` remote
points at this GitHub repo; `upstream` points at the real
`nsnam/ns-3-dev` GitLab repo this fork started from.

### Prerequisites

```bash
cd /home/user/LENA-oran-flexric-smo
python3 -m venv .venv
source .venv/bin/activate
pip install -r gui/requirements.txt   # sionna-rt, polyscope, matplotlib, omegaconf, etc.
./ns3 configure --enable-examples --enable-tests --enable-python-bindings
./ns3 build
```
Look for `Sionna-RT support enabled: all required dependencies were found.`
in the configure output. GPU acceleration (Dr.Jit/Mitsuba CUDA backend)
works with just the NVIDIA driver — the CUDA toolkit (`nvcc`) is **not**
required despite a cosmetic configure-time warning about it.

FlexRIC's `nearRT-RIC` and the xApps are **not** vendored into this repo —
they're reused from the old project's build:
`/home/user/ns-O-RAN-flexric/flexric/build/examples/...`.

### Running it — GUI, ns-3/Sionna RT, and the O-RAN RIC

There are two independently-verified paths today; they haven't been
combined into a single run yet.

#### Path A — GUI + real KHU workload (no RIC)

Terminal 1, the Polyscope GUI:
```bash
cd /home/user/LENA-oran-flexric-smo
source .venv/bin/activate
cd gui
python scripts/run.py /home/user/LENA-oran-flexric-smo/scenes/khu-real/KHU_Cropped_Sionna_RT.xml
```

Terminal 2, the ns-3 scenario:
```bash
cd /home/user/LENA-oran-flexric-smo
source .venv/bin/activate
./ns3 run --no-build "khu-real-nr-sionna \
  --numerologyBwp1=0 --sionnaUpdatePeriod=10s --simTime=60 \
  --guiSrc=/home/user/LENA-oran-flexric-smo/scratch --guiHost=localhost"
```
`--guiSrc` makes the ns-3 process itself act as the GUI's position
publisher (via `scratch/zmq_bridge.py`, ports 5600/5601) — the role
`kyunghee_server.py` used to play in the old project, now absorbed into
the ns-3 process's embedded Python interpreter. Drop `--guiSrc` to run
headless. `numerologyBwp1=0` + a 10s `sionnaUpdatePeriod` (vs. the
defaults of `1`/50ms) cut a 180s run from ~6h42m down to ~4m29s — see
`NR_ROADMAP.md`-equivalent notes for why (channel recompute cadence
dominates; numerology roughly doubles/halves PHY event density on top of
that).

#### Path B — RIC/E2 handshake (official static-UE demo, no GUI/no real workload yet)

Terminal 1, nearRT-RIC:
```bash
cd /home/user/ns-O-RAN-flexric/flexric/build/examples/ric
./nearRT-RIC
```

Terminal 2, ns-3 with E2 enabled:
```bash
cd /home/user/LENA-oran-flexric-smo
source .venv/bin/activate
./ns3 run --no-build "cttc-nr-demo-sionna-rt \
  --Scenario=/home/user/LENA-oran-flexric-smo/scenes/khu-real/KHU_Cropped_Sionna_RT.xml \
  --gNbNum=2 --ueNumPergNb=2 --simTime=500ms \
  --centralFrequencyBand1=3.5e9 --bandwidthBand1=20e6 \
  --ns3::NrHelper::E2ModeNr=true \
  --ns3::NrHelper::E2TermIp=127.0.0.1 \
  --ns3::NrHelper::E2LocalPort=38480"
```
Pick a fresh `E2LocalPort` each run — a previous run's still-ESTABLISHED
SCTP association squatting on the computed local port (`E2LocalPort` +
cellId) causes `Cannot assign requested address` otherwise.

Terminal 3 (optional), an xApp once E2-SETUP succeeds:
```bash
cd /home/user/ns-O-RAN-flexric/flexric/build/examples/xApp/c/monitor
./xapp_kpm_moni
```

### What talks to what

- **ns-3 <-> Sionna RT**: not a network protocol — pybind11 calls within
  one process. Every `SionnaRtChannelModel::UpdatePeriod`, ns-3 reads each
  node pair's `MobilityModel` position and `PhasedArrayModel` antenna
  geometry, hands them to Sionna's ray-tracing `PathSolver` (GPU-accelerated
  via Dr.Jit/Mitsuba), and gets back per-path delay/angle/Doppler/complex
  gain, which becomes an ns-3 `ChannelMatrix` feeding the NR PHY's SINR
  calculation directly.
- **ns-3 <-> GUI**: separate from the above — the scenario pushes plain
  `(x, y, z)` positions over ZMQ (ports 5600/5601) so the GUI can render
  live movement; the GUI does **not** redo any channel computation itself
  (`paths.auto_update` is off by default in `gui/scripts/run.py` for
  exactly this reason — recomputing paths a second time per position update
  was previously causing ~2s GUI frame times).
- **ns-3 <-> nearRT-RIC**: real E2AP/E2SM over SCTP, via `oran-interface`'s
  `E2Termination` (registered per-gNB in `NrHelper::InstallSingleGnbDevice`
  when `E2ModeNr=true`). Verified through E2-SETUP-REQUEST/RESPONSE only so
  far — KPM report bodies and RIC-Control dispatch on the NR side are still
  unimplemented stubs.

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
