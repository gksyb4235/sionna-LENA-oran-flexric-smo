#!/usr/bin/env bash
set -euo pipefail

cd /home/user/LENA-oran-flexric-smo
source .venv/bin/activate

exec stdbuf -oL -eL ./build/scratch/ns3.48-khu-real-dual-nr-sionna-pooled-default \
  --centralFrequency18=1.8e9 \
  --bandwidth18=10e6 \
  --centralFrequency35=3.5e9 \
  --bandwidth35=15e6 \
  --numerology=0 \
  --sionnaCacheFile18=scenarios/khu-real/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5 \
  --sionnaCacheFile35=scenarios/khu-real/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5 \
  --sionnaUpdatePeriod=1s \
  --udpBaseIntervalMs=200 \
  --trafficMultiplierEnabled=true \
  --simTime=900 \
  --guiSrc=/home/user/LENA-oran-flexric-smo/scratch \
  --guiHost=localhost \
  --influxSrc=/home/user/LENA-oran-flexric-smo/scratch \
  --influxHost=localhost \
  --influxPort=8086 \
  --influxDb=nr_kpi \
  --kpiReportInterval=1.0 \
  --sumoTrace=scenarios/khu-real/ue_positions_seed12.csv \
  --N_Ues=300 \
  --ueKpiCsvPath=scenarios/khu-real/runs/seed12_dual_nr_900s_20260827_014436/ue_kpi.csv \
  --cellKpiCsvPath=scenarios/khu-real/runs/seed12_dual_nr_900s_20260827_014436/cell_kpi.csv
