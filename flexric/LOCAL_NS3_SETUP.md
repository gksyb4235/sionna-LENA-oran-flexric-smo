# Local FlexRIC setup for the ns-3 scenario

This source snapshot is based on FlexRIC commit
`307e1d0a5c26751c9e5595805b668a4f91d09550` and includes the local
`xapp_energy_off_demo` changes used by this workspace.

The ns-3 integration uses these protocol versions:

- E2AP: `E2AP_V1`
- KPM: `KPM_V3_00`
- build type: `Debug`
- multi-language xApps: disabled

## First build on another machine

Install a C/C++ toolchain, CMake, and SCTP development headers first. On an
Ubuntu-compatible system, the SCTP packages are normally `libsctp-dev` and
`lksctp-tools`.

From the repository root, run:

```bash
FLEXRIC_BUILD_JOBS=4 ./flexric/build-local.sh
```

The script builds only the nearRT-RIC, the service models required by this
scenario, and `xapp_energy_off_demo`. It installs the configuration and
service-model shared libraries below `flexric/install`, so root access and a
global `/usr/local` installation are not required.

## Terminal 2: nearRT-RIC

```bash
cd /home/user/LENA-oran-flexric-smo/flexric
./build/examples/ric/nearRT-RIC \
  -c ./install/etc/flexric/flexric.conf \
  -p ./install/lib/flexric/
```

When the repository is cloned to a different path, only the `cd` path changes.
The `-p` argument must retain its trailing slash.

The legacy `/usr/local` command also works on a machine where FlexRIC has
already been installed globally, but it is not self-contained and therefore
is not recommended for reproducing this workspace.
