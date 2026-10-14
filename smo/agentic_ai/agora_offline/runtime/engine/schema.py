"""One source of truth for the v16 physical and application contracts."""
CELLS = ['5g', '4g1', '4g2']
PARAMETERS = [f'{p}_{c}' for p in ('txp','ret','cio') for c in CELLS] + ['ttt_ms','hys_db']
TARGETS = ['goodput_mbps','prb_pct','delay_ms','loss_pct','bler_pct',
           'rsrp_interval_mean_dbm','sinr_interval_mean_db','ho_attempts','pingpongs','rlfs',
           'power_w','energy_eff_mbit_per_j','ue_share_pct','traffic_share_pct',
           'rsrp_p10_dbm','ho_successes','delay_over_100ms_pct']
CORE = [0,1,4,5,6,7,8,9,14,15,16]
COUNTS = [7,8,9,15]
INPUTS = ['num_ues'] + [TARGETS[k] for k in CORE] + ['delay_ms','loss_pct','offered_load_mbps']
HORIZONS = [90,180]
DELAYS = [0,30,60,90]
ISSUES = list(range(30,721,10))
APPS = [
    dict(id='energy_saving', label='Energy Saving', controls=['txp'], targets=['goodput_mbps'], goal='network_energy_eff_mbit_per_j', direction='increase'),
    dict(id='throughput', label='Throughput Maximization', controls=['txp','ret'], targets=['goodput_mbps','bler_pct'], goal='goodput_mbps', direction='increase'),
    dict(id='coverage', label='Coverage Optimization', controls=['ret'], targets=['rsrp_interval_mean_dbm','rsrp_p10_dbm','sinr_interval_mean_db'], goal='rsrp_p10_dbm', direction='increase'),
    dict(id='load_balancing', label='Load Balancing', controls=['cio'], targets=['prb_pct'], goal='prb_imbalance', direction='decrease'),
    dict(id='handover_control', label='Mobility Optimization', controls=['ttt','hys'], targets=['ho_attempts','ho_successes','pingpongs','rlfs','delay_over_100ms_pct'], goal='pingpong_per_success_pct', direction='decrease'),
]
