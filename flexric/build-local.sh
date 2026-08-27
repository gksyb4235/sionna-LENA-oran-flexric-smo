#!/usr/bin/env bash
set -euo pipefail

flexric_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
flexric_build_dir="${flexric_source_dir}/build"
flexric_install_dir="${flexric_source_dir}/install"
flexric_build_jobs="${FLEXRIC_BUILD_JOBS:-$(nproc)}"

cmake \
  -S "${flexric_source_dir}" \
  -B "${flexric_build_dir}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DE2AP_VERSION=E2AP_V1 \
  -DKPM_VERSION=KPM_V3_00 \
  -DXAPP_MULTILANGUAGE=OFF \
  -DCMAKE_INSTALL_PREFIX="${flexric_install_dir}"

# The full default target also builds the optional NR-RRC monitor, which
# requires an external asn1c executable. Build only the components used by
# this ns-3/FlexRIC experiment.
cmake --build "${flexric_build_dir}" \
  --parallel "${flexric_build_jobs}" \
  --target \
    nearRT-RIC \
    mac_sm \
    rlc_sm \
    pdcp_sm \
    slice_sm \
    tc_sm \
    gtp_sm \
    kpm_sm \
    rc_sm \
    xapp_energy_off_demo

cmake --install "${flexric_build_dir}"

printf '%s\n' \
  "FlexRIC local build and install completed." \
  "Run the nearRT-RIC with:" \
  "  cd ${flexric_source_dir}" \
  "  ./build/examples/ric/nearRT-RIC -c ./install/etc/flexric/flexric.conf -p ./install/lib/flexric/"
