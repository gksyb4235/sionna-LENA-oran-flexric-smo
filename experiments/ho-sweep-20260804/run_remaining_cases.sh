#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
scenario="${repo_root}/build/scratch/ns3.48-khu-real-nr-sionna-default"
result_dir="${script_dir}"

cd "${repo_root}"

run_case() {
    local ttt="$1"
    local hys="$2"
    local ttt_label
    local hys_label
    ttt_label=$(printf '%04d' "$ttt")
    hys_label=$(printf '%02d' "$hys")
    local case_name="ttt${ttt_label}_hys${hys_label}"

    echo "START ${case_name} $(date --iso-8601=seconds)"
    /usr/bin/time -f 'WALL_SECONDS=%e' "$scenario" \
        --gnbPositions="${repo_root}/scenarios/khu-real/gnbs-ret.csv" \
        --sumoTrace="${repo_root}/scenarios/khu-real/ue-handover-1.csv" \
        --N_Ues=1 \
        --numerologyBwp1=0 \
        --sionnaUpdatePeriod=1s \
        --simTime=200 \
        --ns3::NrHelper::HandoverAlgorithm=ns3::NrA3RsrpHandoverAlgorithm \
        --influxSrc="${repo_root}/scratch" \
        --influxHost=localhost \
        --influxPort=8086 \
        --influxDb="nr_kpi_${case_name}" \
        --kpiReportInterval=1.0 \
        --handoverTtt="$ttt" \
        --handoverHysteresis="$hys" \
        > "${result_dir}/${case_name}.log" 2>&1
    echo "END   ${case_name} $(date --iso-8601=seconds)"
}

run_case 0 0
run_case 0 3
run_case 0 6
run_case 256 0
# ttt=256, hys=3 completed separately as the baseline.
run_case 256 6
run_case 1024 0
run_case 1024 3
run_case 1024 6
