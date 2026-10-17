/* AGORA offline workspace. Every displayed evaluation originates from the API. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const AGENTS = {
    intent: {name: 'Intent Agent', short: 'IN', description: '의도와 목표·제약을 정리합니다.'},
    conflict: {name: 'Conflict Agent', short: 'CF', description: '예측 결과를 목표·제약과 비교합니다.'},
    analyst: {name: 'Analyst Agent', short: 'AN', description: 'KPI와 제어 영향을 분석합니다.'},
    mitigation: {name: 'Mitigation Agent', short: 'MT', description: '대안을 생성하고 다시 평가합니다.'},
    deployment: {name: 'Deployment Agent', short: 'DP', description: '오프라인 실행안을 정리합니다.'}
  };
  const CELLS = ['Cell5G', 'Cell4G1', 'Cell4G2'];
  const KPI = [
    {key: 'throughput_mbps', name: '처리량', unit: 'Mbit/s', aliases: ['goodput_mbps', 'throughput_mbps', 'throughput_mbit_s', 'throughput', 'dl_throughput_mbps']},
    {key: 'prb_util_pct', name: 'PRB', unit: '%', aliases: ['prb_util_pct', 'prb_utilization_pct', 'prb_usage_pct', 'prb_used_pct', 'prb_pct']},
    {key: 'delay_ms', name: '지연', unit: 'ms', aliases: ['delay_ms', 'latency_ms', 'mean_delay_ms', 'avg_delay_ms']},
    {key: 'loss_pct', name: '손실률', unit: '%', aliases: ['loss_pct', 'packet_loss_pct', 'packet_loss_rate_pct']},
    {key: 'bler_pct', name: 'BLER', unit: '%', aliases: ['bler_pct', 'dl_bler_pct', 'bler']},
    {key: 'rsrp_dbm', name: 'RSRP', unit: 'dBm', aliases: ['rsrp_interval_mean_dbm', 'rsrp_dbm', 'rsrp_db', 'mean_rsrp_dbm']},
    {key: 'sinr_db', name: 'SINR', unit: 'dB', aliases: ['sinr_interval_mean_db', 'sinr_db', 'mean_sinr_db']},
    {key: 'ho_attempts', name: 'HO 시도', unit: 'count', aliases: ['ho_attempts', 'ho_attempt_count', 'ho_attempts_count', 'handover_attempts']},
    {key: 'pingpong_count', name: 'Ping-pong', unit: 'count', aliases: ['pingpongs', 'pingpong_count', 'ping_pong_count', 'pingpong', 'ping_pong']},
    {key: 'rlf_count', name: 'RLF', unit: 'count', aliases: ['rlfs', 'rlf_count', 'rlf', 'radio_link_failure_count']},
    {key: 'power_w', name: '전력', unit: 'W', aliases: ['power_w', 'power']},
    {key: 'energy_eff_mbit_per_j', name: '에너지 효율', unit: 'Mbit/J', aliases: ['energy_eff_mbit_per_j', 'energy_efficiency_mbit_j']},
    {key: 'ue_share_pct', name: 'UE 점유율', unit: '%', aliases: ['ue_share_pct', 'ue_share']},
    {key: 'traffic_share_pct', name: '트래픽 점유율', unit: '%', aliases: ['traffic_share_pct', 'traffic_share']},
    {key: 'rsrp_p10_dbm', name: 'P10 RSRP', unit: 'dBm', aliases: ['rsrp_p10_dbm']},
    {key: 'ho_successes', name: 'HO 성공', unit: '회', aliases: ['ho_successes']},
    {key: 'delay_over_100ms_pct', name: '100ms 초과', unit: '%', aliases: ['delay_over_100ms_pct']}
  ];
  const EXTRA = {num_ues: {key: 'num_ues', name: '서비스 UE', unit: '명', aliases: ['num_ues', 'num_ues_last', 'ue_count', 'n_ues']},
    rsrp_gap: {key: 'rsrp_gap', name: '평균 − P10 RSRP', unit: 'dB', aliases: ['rsrp_mean_p10_gap_db']}};
  const state = {
    status: null, runs: [], snapshot: null, settings: null, promptDraft: {}, promptAgent: 'intent',
    historyCell: 0, sessionId: null, running: false, source: null, poll: null, seen: new Set(),
    candidates: new Map(), summaries: new Map(), lastAgent: null, snapshotRequest: 0, runsRequest: 0,
    promptTouched: false, promptFamily: 'txp', finishing: false, restoring: false, activeGraphNode: null
  };
  let toastTimer;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function clear(node) { node.replaceChildren(); }
  function setText(id, text) { $(id).textContent = text == null ? '—' : String(text); }
  function showNotice(message, error = false) {
    const node = $('global-notice'); node.hidden = !message; node.textContent = message || '';
    node.classList.toggle('error', error);
  }
  function toast(message, error = false) {
    const node = $('toast'); node.textContent = message; node.classList.toggle('error', error); node.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => {node.hidden = true;}, 4200);
  }
  function number(value, digits = 2) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return '—';
    const n = Number(value); return Number.isFinite(n) ? new Intl.NumberFormat('ko-KR', {maximumFractionDigits: digits}).format(n) : '—';
  }
  function finite(value) { return value !== null && value !== undefined && value !== '' && typeof value !== 'boolean' && Number.isFinite(Number(value)); }
  function metric(id, value, unit) {
    const node = $(id); clear(node); node.append(document.createTextNode(number(value)));
    if (finite(value)) node.append(element('small', '', unit));
  }
  function first(object, names) {
    if (!object || typeof object !== 'object') return undefined;
    for (const name of names) if (object[name] !== undefined && object[name] !== null) return object[name];
    const normalized = Object.fromEntries(Object.keys(object).map(key => [key.toLowerCase().replace(/[^a-z0-9]/g, ''), key]));
    for (const name of names) { const key = normalized[name.toLowerCase().replace(/[^a-z0-9]/g, '')]; if (key !== undefined) return object[key]; }
    return undefined;
  }
  function jsonSafe(value) {
    const seen = new WeakSet();
    return JSON.stringify(value, (key, val) => {
      if (/^(api_?key|authorization|access_token|secret|password)$/i.test(key)) return '[REDACTED]';
      if (val && typeof val === 'object') { if (seen.has(val)) return '[Repeated reference]'; seen.add(val); }
      return val;
    }, 2);
  }
  function asText(value) { return typeof value === 'string' ? value : value == null ? '' : jsonSafe(value); }
  async function api(path, options = {}) {
    const response = await fetch(`api/${path}`, {cache: 'no-store', ...options,
      headers: {'Accept': 'application/json', ...(options.body ? {'Content-Type': 'application/json'} : {}), ...options.headers}});
    const body = await response.text(); let data;
    try { data = body ? JSON.parse(body) : {}; } catch (_) { throw new Error(`서버 응답 형식이 올바르지 않습니다 (HTTP ${response.status}).`); }
    if (!response.ok) {
      const message = data.detail || data.error || data.message || `HTTP ${response.status}`;
      throw new Error(typeof message === 'string' ? message : asText(message));
    }
    return data;
  }
  function badge(node, text, tone = 'neutral') { node.className = `pill ${tone}`; node.textContent = text; }
  function activeAgent(value) {
    const normalized = String(value || '').toLowerCase().replace(/[^a-z]/g, '');
    return Object.keys(AGENTS).find(key => normalized.includes(key)) || null;
  }
  function roster(agent, status) {
    const li = document.querySelector(`[data-agent="${agent}"]`); if (!li) return;
    li.className = status; li.querySelector('.roster-state').textContent = ({running: '실행', complete: '완료', error: '오류'})[status] || '대기';
  }
  function cellAliases(index) { return [CELLS[index], ['5g', '4g1', '4g2'][index], ['cell_5g', 'cell_4g1', 'cell_4g2'][index], String(index + 1)]; }
  function isCell(value, index) {
    if (value === undefined || value === null) return false;
    const s = String(value).toLowerCase().replace(/[^a-z0-9]/g, '');
    return cellAliases(index).some(a => a.toLowerCase().replace(/[^a-z0-9]/g, '') === s);
  }
  function txp(rcp, index = 0) {
    if (Array.isArray(rcp)) return rcp[index];
    const names = [['txp_5g', 'tx_power_5g', 'tx_power_dbm_5g', 'tx_power_dbm_cell1', 'txp_cell1'],
      ['txp_4g1', 'tx_power_4g1', 'tx_power_dbm_4g1', 'tx_power_dbm_cell2'],
      ['txp_4g2', 'tx_power_4g2', 'tx_power_dbm_4g2', 'tx_power_dbm_cell3']][index];
    const direct = first(rcp, names); if (direct !== undefined) return direct;
    const cell = first(rcp, cellAliases(index)); if (cell) return first(cell, ['txp', 'tx_power_dbm', 'tx_power']);
    const group = first(rcp, ['txp', 'tx_power_dbm']);
    if (Array.isArray(group)) return group[index];
    if (group && typeof group === 'object') return first(group, cellAliases(index));
    return index === 0 && finite(group) ? group : undefined;
  }
  function currentRcp() { return state.snapshot?.current_rcp || state.runs.find(run => run.run_id === $('run-select').value)?.current_rcp || {}; }
  function runControlValues(rcp) {
    const cell = $('example-cell').value;
    return [[`TxP`, rcp?.[`txp_${cell}`], 'dBm'], ['RET', rcp?.[`ret_${cell}`], '°'],
      ['CIO', rcp?.[`cio_${cell}`], 'dB'], ['TTT', rcp?.ttt_ms, 'ms'], ['HYS', rcp?.hys_db, 'dB']]
      .map(([name, value, unit]) => `${name} ${number(value)}${unit === '°' ? '' : ' '}${unit}`);
  }
  function renderRunControls() {
    const cell = $('example-cell').value;
    setText('run-control-label', `기준 실행 · Cell${cell.toUpperCase()}`);
    for (const option of $('run-select').options) {
      const run = state.runs.find(item => item.run_id === option.value);
      if (run) option.textContent = `${run.run_id} (${runControlValues(run.current_rcp).join(', ')})`;
    }
    const selected = state.runs.find(run => run.run_id === $('run-select').value);
    const summary = $('run-control-values'); clear(summary);
    if (selected) {
      runControlValues(selected.current_rcp).forEach(value => summary.append(element('span', '', value)));
      summary.append(element('small', '', 'TTT·HYS는 공통 값'));
      $('run-select').title = $('run-select').selectedOptions[0]?.textContent || '';
    }
  }
  function selectedCellData(raw, index) {
    if (!raw) return null;
    if (Array.isArray(raw)) {
      const match = raw.find(row => row && !Array.isArray(row) && isCell(first(row, ['cell_id', 'cell', 'cell_name']), index));
      return match || raw[index];
    }
    const cell = first(raw, cellAliases(index)); if (cell !== undefined) return cell;
    if (raw.cells) return selectedCellData(raw.cells, index);
    return raw;
  }
  function kpiValue(raw, index, descriptor) {
    if (!raw) return undefined;
    const cell = selectedCellData(raw, index);
    if (Array.isArray(cell)) {
      const found = cell.find(v => v && typeof v === 'object' && descriptor.aliases.includes(v.name || v.kpi));
      if (found) return first(found, ['value', 'mean', 'latest']);
      if (cell.every(v => finite(v) || v == null) && cell.length === 14) return cell[KPI.indexOf(descriptor)];
    }
    const value = first(cell, descriptor.aliases);
    if (finite(value)) return value;
    if (value && !Array.isArray(value) && typeof value === 'object') {
      const nested = first(value, ['value', 'mean', 'latest']); if (finite(nested)) return nested;
    }
    if (Array.isArray(value) && finite(value[index])) return value[index];
    if (raw.kpis && raw.kpis !== raw) return kpiValue(raw.kpis, index, descriptor);
    if (cell?.kpis) return kpiValue(cell.kpis, 0, descriptor);
    return undefined;
  }

  async function loadStatus() {
    try {
      state.status = await api('status');
      const provider = state.status.provider || {};
      const model = state.status.model || {};
      const label = typeof model === 'string' ? model : model.model_name || model.name || model.model_id || 'Compact Temporal GNN';
      setText('model-name', label);
      $('connection-status').replaceChildren(element('span', 'status-dot'), document.createTextNode('로컬 서비스 연결됨'));
      const llmOption = $('mode-select').querySelector('option[value="llm"]');
      llmOption.disabled = !provider.configured;
      if (!provider.configured && $('mode-select').value === 'llm') $('mode-select').value = 'deterministic';
      modeLabel();
      const limits = state.status.limits || {};
      const times = first(limits, ['issue_times_s', 'issue_times']);
      if (Array.isArray(times) && times.length) { $('issue-time').min = Math.min(...times); $('issue-time').max = Math.max(...times); }
      if (finite(limits.min_issue_time_s)) $('issue-time').min = limits.min_issue_time_s;
      if (finite(limits.max_issue_time_s)) $('issue-time').max = limits.max_issue_time_s;
      updateRunButton();
    } catch (error) {
      $('connection-status').replaceChildren(element('span', 'status-dot error'), document.createTextNode('서비스 연결 실패'));
      showNotice(`로컬 API에 연결하지 못했습니다. ${error.message}`, true);
      state.status = null; updateRunButton();
    }
  }
  function modeLabel() {
    const provider = state.status?.provider || {};
    if ($('mode-select').value === 'deterministic') {
      setText('provider-status', provider.configured ? '규칙 기반 데모 · LLM 호출 없이 실제 GNN 도구를 평가합니다.' : 'LLM 미설정 · 규칙 기반 데모를 선택했습니다. 실제 LLM 실행이 아닙니다.');
    } else setText('provider-status', `LLM · ${provider.model || '서버 설정 모델'} · 실행 시 OpenAI API로 선택한 관측 요약·예측·DAG 문맥·프롬프트를 전송합니다.`);
    updateRunButton();
  }
  function updateRunButton() {
    renderPromptExamples();
    $('run-button').disabled = state.running || state.restoring || !window.AgoraPolicyEditor?.ready() || !state.status || !$('run-select').value || !state.snapshot || !$('intent-prompt').value.trim() ||
      ($('mode-select').value === 'llm' && !state.status?.provider?.configured);
    $('run-button').replaceChildren(element('span', '', state.running ? '◌' : '▷'), document.createTextNode(state.running ? '평가 진행 중' : '의도 평가 시작'));
  }
  async function loadRuns() {
    const request = ++state.runsRequest;
    state.snapshot = null; updateRunButton();
    const select = $('run-select'), previous = select.value;
    select.replaceChildren(new Option('실행 목록 불러오는 중', '')); select.disabled = true;
    try {
      const result = await api(`runs?split=${encodeURIComponent($('split-select').value)}`);
      if (request !== state.runsRequest) return;
      state.runs = Array.isArray(result) ? result : result.runs || [];
      const choices = state.runs.map(run => new Option(`${run.run_id} (${runControlValues(run.current_rcp).join(', ')})`, run.run_id));
      select.replaceChildren(...(choices.length ? choices : [new Option('사용할 실행이 없습니다', '')]));
      const preferred = state.runs.find(run => run.run_id === previous) || state.runs.find(run => run.run_id === 'gnn_0130' && Number(txp(run.current_rcp)) === 43) || state.runs.find(run => Number(txp(run.current_rcp)) === 43) || state.runs[0];
      if (preferred) select.value = preferred.run_id;
      select.disabled = state.running;
      if (preferred) await loadSnapshot(); else {setText('snapshot-status', '해당 분할에 실행이 없습니다.'); updateRunButton();}
    } catch (error) {
      if (request !== state.runsRequest) return;
      select.replaceChildren(new Option('목록을 불러오지 못했습니다', ''));
      setText('snapshot-status', error.message); showNotice(error.message, true); updateRunButton();
    }
  }
  async function loadSnapshot() {
    const run = $('run-select').value; if (!run) return;
    renderRunControls();
    const time = Number($('issue-time').value);
    if (!$('issue-time').checkValidity()) { $('issue-time').reportValidity(); return; }
    const request = ++state.snapshotRequest;
    state.snapshot = null; setText('snapshot-status', '기준 관측 불러오는 중…'); updateRunButton();
    try {
      const snapshot = await api(`snapshot?run_id=${encodeURIComponent(run)}&t=${encodeURIComponent(time)}`);
      if (request !== state.snapshotRequest) return;
      state.snapshot = snapshot;
      const actual = txp(currentRcp());
      metric('metric-txp', actual, 'dBm');
      const latest = snapshot.latest_kpis || snapshot.latest || snapshot.current_kpis;
      metric('metric-throughput', kpiValue(latest, 0, KPI[0]), 'Mbit/s');
      metric('metric-prb', kpiValue(latest, 0, KPI[1]), '%');
      setText('metric-txp-note', `${run} · 실제 관측 설정`);
      const horizon = Number($('horizon-select').value);
      setText('snapshot-status', `${run} · [${snapshot.history_start_s ?? Math.max(0,time-180)}, ${time})초 관측 → [${time}, ${time + horizon})초 예측`);
      setText('history-forecast-label', `과거 관측만 입력 · 미래 ${horizon}초 예측`);
      renderHistory(); renderGraph(); updateRunButton();
    } catch (error) {
      if (request !== state.snapshotRequest) return;
      setText('snapshot-status', error.message); showNotice(`스냅샷을 불러오지 못했습니다. ${error.message}`, true); updateRunButton();
    }
  }

  function renderPromptExamples() {
    const family = state.promptFamily, shared = family === 'mobility';
    const unavailable = state.running || state.restoring || !state.snapshot;
    $$('[data-intent-family]').forEach(button => {
      button.disabled = unavailable;
      button.setAttribute('aria-pressed', String(!state.promptTouched && button.dataset.intentFamily === family));
    });
    $('example-cell').disabled = unavailable || shared;
    if (state.promptTouched) {
      setText('example-help', '직접 입력한 프롬프트를 유지합니다. 제어 예시를 선택하면 현재값을 반영한 예시로 바뀝니다.');
      return;
    }
    if (unavailable) return;
    const cell = $('example-cell').value;
    const control = family === 'throughput' ? 'txp' : shared ? 'ttt' : family;
    const key = shared ? 'ttt_ms' : `${control}_${cell}`;
    const current = Number(currentRcp()[key]);
    if (!finite(currentRcp()[key])) {
      $('intent-prompt').value = ''; setText('example-help', '현재 제어값을 확인할 수 없어 예시를 만들 수 없습니다.'); return;
    }
    const config = {
      txp: {unit: 'dBm', purpose: 'Energy Saving rApp (네트워크 에너지 효율 개선)', target: current > 33 ? 33 : 30},
      throughput: {unit: 'dBm', purpose: 'Throughput Maximization xApp (처리량 최대화)', target: 43},
      ret: {unit: '도', purpose: 'Coverage Optimization rApp (평균 RSRP와 P10 개선)', target: current > 3 ? 3 : 12},
      cio: {unit: 'dB', purpose: 'Load Balancing xApp (PRB 불균형 완화)', target: current < (cell === '5g' ? 9 : 2) ? (cell === '5g' ? 9 : 2) : (cell === '5g' ? 5 : -2)},
      mobility: {unit: 'ms', purpose: 'Mobility Optimization xApp (성공 HO 대비 핑퐁 비율 감소)', target: current < 720 ? 720 : 512}
    }[family];
    if (family === 'txp' && current <= 30) {
      $('intent-prompt').value = '';
      setText('example-help', `${cell.toUpperCase()}의 현재 TxP는 ${current} dBm으로 모델 하한입니다. 에너지 절감 예시는 다른 셀을 선택해 주세요.`);
      return;
    }
    const time = Number($('issue-time').value), horizon = Number($('horizon-select').value);
    const scope = shared ? '네트워크 공통' : `Cell${cell.toUpperCase()}`;
    const axes = shared ? '적용 시점·제어 강도와 그 조합' : '적용 시점·제어 강도·대상 셀과 그 조합';
    let action = `${control.toUpperCase()} ${current} ${config.unit}에서 ${config.target} ${config.unit}`;
    if (family === 'throughput' && current >= 43) {
      const ret = Number(currentRcp()[`ret_${cell}`]);
      action = `RET ${ret} 도에서 ${ret > 3 ? 3 : 5} 도`;
    }
    if (shared) {
      const hys = Number(currentRcp().hys_db);
      action += `, HYS ${hys} dB에서 ${hys < 5 ? Math.min(5,hys+1) : 4} dB`;
    }
    $('intent-prompt').value = `최근 ${Math.min(180,time)}초 관측 데이터를 기준으로, ${config.purpose} 목적으로 ${scope} ${action}로 변경하고 ${horizon}초간 유지하려고 합니다. 아래 편집기에서 확정한 KPI 개선 목표와 threshold 제약으로 먼저 원안 1개를 평가해 주세요. 목표 또는 제약을 충족하지 못하면 ${axes}을 배치로 비교해 예측상 목표와 제약을 충족하는 대안을 제안해 주세요.${shared ? ' TTT와 HYS는 공통 제어이므로 셀 이동은 적용하지 마세요.' : ''}${family === 'throughput' || family === 'ret' ? ' RET는 틸트 각도이며, 증감 방향의 효과는 예측으로 확인하세요.' : ''}`;
    $('intent-prompt').value += ' 각 허용 셀의 유효 중간값과 적용 지연을 넓은 배치로 빠짐없이 확인하세요. TxP 단일 요청은 그 결과를 읽고 유망 후보를 기준으로 여러 셀에 변경량을 분산하는 조합을 후순위 배치에서 검토하세요.';
    setText('example-help', `제어 요청은 이 입력에서, KPI 목표와 threshold는 아래 편집기에서 지정합니다. ${shared ? '공통 제어 · 시간·강도 탐색.' : '셀별 제어 · 시간·강도·셀 탐색.'} 수정한 목표·제약은 예시를 바꿔도 유지합니다.`);
  }

  function historyUnit(descriptor) {
    return descriptor.unit === 'count' ? '회/1초 구간' : descriptor.unit;
  }
  function populateHistoryKpis() {
    const metrics = element('optgroup'); metrics.label = '관측 KPI 17개 (예측 출력과 구분)';
    KPI.forEach(kpi => metrics.append(new Option(`${kpi.name} · ${historyUnit(kpi)}`, kpi.key)));
    const extra = element('optgroup'); extra.label = '보조 관측';
    Object.values(EXTRA).forEach(kpi => extra.append(new Option(`${kpi.name} · ${kpi.unit}`, kpi.key)));
    $('history-kpi').replaceChildren(metrics, extra);
  }
  function historyPoints() {
    const descriptor = KPI.find(k => k.key === $('history-kpi').value) || EXTRA[$('history-kpi').value];
    const cell = state.historyCell, snapshot = state.snapshot;
    if (!snapshot) return {points: [], descriptor};
    const allKpis = snapshot.kpi_history;
    if (descriptor.key === 'rsrp_gap' && allKpis) {
      const a = allKpis.metric_names.indexOf('rsrp_interval_mean_dbm'), b = allKpis.metric_names.indexOf('rsrp_p10_dbm');
      const points = allKpis.times_s.map((t,i) => {const row=allKpis.values[i]?.[cell], valid=allKpis.valid?.[i]?.[cell];
        return {t, v:a>=0 && b>=0 && valid?.[a]!==false && valid?.[b]!==false && finite(row?.[a]) && finite(row?.[b]) ? row[a]-row[b] : null};});
      return {points,descriptor};
    }
    let source = allKpis?.metric_names?.some(name => descriptor.aliases.includes(name)) ? allKpis : snapshot.history || snapshot.history_1s || snapshot.timeline || [];
    if (source.rows) source = source.rows;
    if (source.records) source = source.records;
    let points = [];
    const names = source.metric_names || source.input_names;
    if (Array.isArray(source.times_s) && Array.isArray(source.values) && Array.isArray(names)) {
      const k = names.findIndex(name => descriptor.aliases.includes(name));
      if (k >= 0) points = source.times_s.map((t, i) => {
        const v = source.values[i]?.[cell]?.[k], valid = source.valid?.[i]?.[cell]?.[k];
        return {t: Number(t), v: valid !== false && finite(v) ? Number(v) : null};
      });
    } else if (Array.isArray(source)) {
      for (const row of source) {
        if (!row || typeof row !== 'object') continue;
        const rowCell = first(row, ['cell_id', 'cell', 'cell_name']);
        if (rowCell !== undefined && !isCell(rowCell, cell)) continue;
        const t = first(row, ['time_s', 't', 'timestamp_s', 'time']);
        let value = first(row, descriptor.aliases);
        if (!finite(value)) value = kpiValue(row.cells || row.kpis || row.values || row, cell, descriptor);
        if (finite(t)) points.push({t: Number(t), v: finite(value) ? Number(value) : null});
      }
    } else if (source && typeof source === 'object') {
      const cellSource = first(source, cellAliases(cell));
      if (Array.isArray(cellSource)) {
        for (const row of cellSource) {
          const t = first(row, ['time_s', 't', 'timestamp_s', 'time']);
          const v = first(row, descriptor.aliases); if (finite(t)) points.push({t: Number(t), v: finite(v) ? Number(v) : null});
        }
      } else {
        const data = cellSource || source;
        const times = first(data, ['time_s', 'times_s', 't', 'times']) || first(source, ['time_s', 'times_s', 't', 'times']);
        const values = first(data, descriptor.aliases) || first(data.kpis, descriptor.aliases);
        if (Array.isArray(times) && Array.isArray(values)) points = times.map((t, i) => {
          const v = Array.isArray(values[i]) ? values[i][cell] : values[i]; return {t: Number(t), v: finite(v) ? Number(v) : null};
        });
      }
    }
    const issue = Number(snapshot.issue_time_s ?? $('issue-time').value);
    points = points.filter(point => Number.isFinite(point.t) && point.t >= issue - 180 && point.t < issue).sort((a, b) => a.t - b.t);
    return {points, descriptor};
  }
  function svgNode(name, attrs = {}, text) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function renderHistory() {
    const container = $('history-chart'); clear(container);
    const {points, descriptor} = historyPoints();
    const unit = historyUnit(descriptor);
    $('history-kpi').title = `${descriptor.name} · ${unit}`;
    const values = points.filter(point => point.v !== null);
    const issue = Number(state.snapshot?.issue_time_s ?? $('issue-time').value);
    setText('history-range', `${CELLS[state.historyCell]} · ${state.snapshot?.history_start_s ?? Math.max(0,issue-180)}–${issue}초`);
    if (!values.length) {
      const empty = element('div', 'empty-state small'); empty.append(element('span', '', '⌁'), element('p', '', '이 스냅샷에 해당 KPI의 1초 관측 데이터가 없습니다.')); container.append(empty); return;
    }
    const width = 640, height = 210, left = 48, right = 13, top = 18, bottom = 30;
    const lo = Math.min(...values.map(p => p.v)), hi = Math.max(...values.map(p => p.v));
    const padding = Math.max((hi - lo) * .14, Math.abs(hi) * .015, .1);
    const min = lo >= 0 ? Math.max(0, lo - padding) : lo - padding, max = hi + padding;
    const x = t => left + (t - (issue - 180)) / 180 * (width - left - right);
    const y = v => top + (max - v) / (max - min) * (height - top - bottom);
    const svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${CELLS[state.historyCell]}의 과거 180초 ${descriptor.name} 실제 관측`});
    const defs = svgNode('defs'), gradient = svgNode('linearGradient', {id: 'history-fill', x1: 0, y1: 0, x2: 0, y2: 1});
    gradient.append(svgNode('stop', {offset: '0%', 'stop-color': '#7ce5c2', 'stop-opacity': .18}), svgNode('stop', {offset: '100%', 'stop-color': '#7ce5c2', 'stop-opacity': 0})); defs.append(gradient); svg.append(defs);
    for (let i = 0; i < 4; i++) {
      const value = min + (max - min) * i / 3, yy = y(value);
      svg.append(svgNode('line', {x1: left, y1: yy, x2: width - right, y2: yy, stroke: '#29354a', 'stroke-width': .7, 'stroke-dasharray': '3 5'}));
      svg.append(svgNode('text', {x: left - 10, y: yy + 3, class: 'chart-axis-label', 'text-anchor': 'end'}, number(value, 1)));
    }
    for (let i = 0; i <= 4; i++) svg.append(svgNode('text', {x: x(issue - 180 + 45 * i), y: height - 7, class: 'chart-axis-label', 'text-anchor': 'middle'}, `${issue - 180 + 45 * i}s`));
    svg.append(svgNode('text', {x: left, y: 9, class: 'chart-axis-label'}, unit));
    let segment = [], segments = [];
    for (const p of points) {
      if (p.v === null || (segment.length && p.t - segment[segment.length - 1].t > 2)) { if (segment.length) segments.push(segment); segment = []; }
      if (p.v !== null) segment.push(p);
    }
    if (segment.length) segments.push(segment);
    for (const items of segments) {
      const path = items.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(2)},${y(p.v).toFixed(2)}`).join(' ');
      svg.append(svgNode('path', {d: `${path} L${x(items[items.length - 1].t)},${height-bottom} L${x(items[0].t)},${height-bottom} Z`, fill: 'url(#history-fill)'}));
      svg.append(svgNode('path', {d: path, fill: 'none', stroke: '#7ce5c2', 'stroke-width': 1.8, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}));
    }
    const guide = svgNode('line', {y1: top, y2: height - bottom, stroke: '#8ea1bc', 'stroke-dasharray': '3 4', visibility: 'hidden'});
    const dot = svgNode('circle', {r: 3, fill: '#c0f5e2', visibility: 'hidden'});
    const tip = svgNode('text', {x: width - right, y: 10, 'text-anchor': 'end', class: 'chart-axis-label'});
    const overlay = svgNode('rect', {x: left, y: top, width: width-left-right, height: height-top-bottom, fill: 'transparent'});
    overlay.addEventListener('pointermove', event => {
      const rect = svg.getBoundingClientRect(); const localX = (event.clientX - rect.left) / rect.width * width;
      const t = issue - 180 + (localX - left) / (width-left-right) * 180;
      const nearest = values.reduce((best, p) => Math.abs(p.t-t) < Math.abs(best.t-t) ? p : best, values[0]);
      guide.setAttribute('x1', x(nearest.t)); guide.setAttribute('x2', x(nearest.t)); guide.setAttribute('visibility', 'visible');
      dot.setAttribute('cx', x(nearest.t)); dot.setAttribute('cy', y(nearest.v)); dot.setAttribute('visibility', 'visible');
      tip.textContent = `${number(nearest.t, 1)}s · ${number(nearest.v)} ${unit}`;
    });
    overlay.addEventListener('pointerleave', () => {guide.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); tip.textContent = '';});
    svg.append(guide, dot, tip, overlay); container.append(svg);
  }

  function renderGraph() {
    const container = $('graph-canvas'); clear(container);
    const svg = svgNode('svg', {viewBox: '0 0 1120 770', 'aria-hidden': 'true'});
    const nodes = [], edges = [], byId = {};
    const add = n => {nodes.push(n); byId[n.id] = n;};
    const apps = [{label: 'Energy Saving', sub: '에너지 효율 · TxP', p: [0]}, {label: 'Throughput Max.', sub: '처리량 · TxP / RET', p: [0,1]}, {label: 'Coverage Opt.', sub: '평균·P10 RSRP · RET', p: [1]}, {label: 'Load Balancing', sub: 'PRB 균형 · CIO', p: [2]}, {label: 'Mobility Opt.', sub: '핑퐁 / HO · TTT / HYS', p: [3, 4]}];
    apps.forEach((app, i) => {add({id: `a${i}`, type: 'app', x: 15, y: 94 + i*137, w: 165, h: 51, label: app.label, sub: app.sub, data: app}); app.p.forEach(p => edges.push([`a${i}`, `p${p}`]));});
    ['TxP', 'RET', 'CIO', 'TTT', 'HYS'].forEach((label, i) => add({id: `p${i}`, type: 'param', x: 266, y: 94+i*137, w: 116, h: 49, label, sub: i < 3 ? '셀별 제어값' : '공통 제어값', data: {index: i}}));
    CELLS.forEach((label, c) => {
      add({id: `c${c}`, type: 'cell', x: 515, y: 147+c*238, w: 118, h: 52, label, sub: finite(txp(currentRcp(), c)) ? `${number(txp(currentRcp(), c))} dBm · 현재 TxP` : '관측값 대기', data: {cell: c}});
      for (let p = 0; p < 5; p++) edges.push([`p${p}`, `c${c}`]);
      KPI.filter(kpi => !['delay_ms','loss_pct','power_w','energy_eff_mbit_per_j','ue_share_pct','traffic_share_pct'].includes(kpi.key)).forEach((kpi, k) => {
        const value = kpiValue(state.snapshot?.latest_kpis, c, kpi);
        const id = `k${c}_${k}`;
        add({id, type: 'kpi', x: 770 + Math.floor(k / 7)*172, y: 65+c*238+(k%7)*32, w: 153, h: 27, label: kpi.name, sub: finite(value) ? `${number(value, 2)} ${kpi.unit}` : '—', data: {cell: c, kpi, value}});
        edges.push([`c${c}`, id]);
      });
    });
    [['APPLICATION', 15], ['RCP TYPE', 266], ['CELL', 515], ['KEY PERFORMANCE INDICATOR', 770]].forEach(([label, x]) => svg.append(svgNode('text', {x, y: 25, class: 'graph-column-label'}, label)));
    edges.forEach(([from, to]) => {
      const a = byId[from], b = byId[to], ax = a.x+a.w, ay = a.y+a.h/2, bx = b.x, by = b.y+b.h/2;
      const path = svgNode('path', {d: `M${ax},${ay} C${ax+(bx-ax)*.5},${ay} ${bx-(bx-ax)*.5},${by} ${bx},${by}`, class: 'graph-edge', 'data-from': from, 'data-to': to}); svg.append(path);
    });
    const palette = {app: ['#26253d', '#5a5080'], param: ['#302b25', '#756040'], cell: ['#1e2d44', '#49648d'], kpi: ['#172e2b', '#36534c']};
    nodes.forEach(n => {
      const group = svgNode('g', {class: `graph-node ${n.type}`, tabindex: 0, role: 'button', 'aria-label': `${n.label} ${n.sub}`, 'data-node': n.id});
      group.append(svgNode('rect', {x: n.x, y: n.y, width: n.w, height: n.h, rx: n.type === 'kpi' ? 5 : 8, fill: palette[n.type][0], stroke: palette[n.type][1], 'stroke-width': 1, class: 'node-box'}));
      if (n.type === 'kpi') {
        group.append(svgNode('text', {x: n.x+9, y: n.y+17, class: 'node-label'}, n.label));
        group.append(svgNode('text', {x: n.x+n.w-7, y: n.y+17, 'text-anchor': 'end', class: 'node-subtitle'}, n.sub));
      } else {
        group.append(svgNode('text', {x: n.x+12, y: n.y+21, class: 'node-label'}, n.label));
        group.append(svgNode('text', {x: n.x+12, y: n.y+37, class: 'node-subtitle'}, n.sub));
      }
      group.append(svgNode('title', {}, `${n.label} · ${n.sub}`));
      const select = () => {
        state.activeGraphNode = n.id;
        svg.querySelectorAll('.graph-node').forEach(node => node.classList.toggle('selected', node.getAttribute('data-node') === n.id));
        svg.querySelectorAll('.graph-edge').forEach(edge => edge.classList.toggle('highlight', edge.getAttribute('data-from') === n.id || edge.getAttribute('data-to') === n.id));
        graphDetails(n);
      };
      group.addEventListener('click', select); group.addEventListener('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); select();}});
      svg.append(group);
    });
    container.append(svg);
    if (state.activeGraphNode) svg.querySelector(`[data-node="${state.activeGraphNode}"]`)?.dispatchEvent(new Event('click'));
  }
  function graphDetails(node) {
    const detail = $('graph-detail'); clear(detail);
    detail.append(element('div', 'eyebrow', `${node.type.toUpperCase()} INSPECTOR`), element('h3', '', node.label));
    let text;
    if (node.type === 'app') text = `${node.sub}\n정의된 App–제어 파라미터 연결입니다. 이 화면의 에이전트 5개와는 다른 네트워크 애플리케이션 노드입니다.`;
    else if (node.type === 'param') {
      const key = ['txp', 'ret', 'cio', 'ttt', 'hys'][node.data.index];
      const rcp = currentRcp();
      const entries = typeof rcp === 'object' && !Array.isArray(rcp) ? Object.entries(rcp).filter(([name]) => name.toLowerCase().includes(key) || (key === 'hys' && name.toLowerCase().includes('hysteresis'))) : [];
      text = node.data.index === 0 ? CELLS.map((cell, i) => `${cell}: ${number(txp(rcp, i))} dBm`).join('\n') : entries.length ? entries.map(([name, value]) => `${name}: ${asText(value)}`).join('\n') : '셀별 또는 공통 제어 파라미터 유형입니다. 실제 설정 값은 스냅샷의 current_rcp에 저장됩니다.';
    } else if (node.type === 'cell') text = `현재 TxP: ${number(txp(currentRcp(), node.data.cell))} dBm\n11개 기초 KPI를 미래 90초 또는 180초 구간에 대해 예측합니다. 에너지 효율·PRB 균형·평균−P10 편차는 예측에서 계산합니다.\n기준 실행: ${$('run-select').value || '선택 전'}`;
    else text = `${CELLS[node.data.cell]} · ${node.label}\n현재 스냅샷 관측: ${number(node.data.value)} ${node.data.kpi.unit}\n예측값과 현재 관측값을 구분해 평가합니다.`;
    detail.append(element('p', '', text), element('div', 'detail-note', '이 도식은 선언된 관계를 요약합니다. SHAP이나 도구 응답에 없는 인과 관계를 생성하지 않습니다.'));
  }

  function setRunning(value) {
    state.running = value;
    if (value) window.AgoraOperatorReview.setCompleted(false);
    window.AgoraPolicyEditor.setBusy(value);
    $('intent-prompt').disabled = value;
    ['split-select', 'run-select', 'issue-time', 'mode-select', 'horizon-select'].forEach(id => {$(id).disabled = value;});
    $('cancel-button').hidden = !value; $('live-indicator').hidden = !value;
    $('session-dot').className = `status-dot${value ? '' : ' pending'}`;
    updateRunButton();
  }
  function resetSessionView() {
    window.AgoraOperatorReview.reset();
    state.seen.clear(); state.candidates.clear(); state.summaries.clear(); state.lastAgent = null; state.finishing = false;
    state.source?.close(); state.source = null; clearInterval(state.poll); state.poll = null;
    Object.keys(AGENTS).forEach(key => roster(key, ''));
    clear($('event-stream')); renderCandidates(); $('export-link').hidden = true;
    ['search-coverage', 'mitigation-context'].forEach(id => {if ($(id)) {clear($(id)); $(id).hidden = true;}});
    badge($('decision-badge'), '진행 중', 'purple');
    const content = $('decision-content'); clear(content);
    content.append(element('div', 'decision-symbol', '◌'), element('h3', '', '원안을 평가하고 있습니다'), element('p', '', '에이전트 요약과 GNN 도구 호출을 아래 스트림에서 확인하세요.'));
  }
  async function startSession(event) {
    event.preventDefault(); if (state.running || $('run-button').disabled) return;
    const prompt = $('intent-prompt').value.trim(); if (!prompt) return;
    let intentSpec;
    try {
      intentSpec = window.AgoraPolicyEditor.getSpec();
    } catch (error) {toast(`목표·제약을 확인하세요: ${error.message}`, true); return;}
    resetSessionView(); showNotice(''); setRunning(true); setText('session-status', '세션 생성 중…');
    const body = {prompt, run_id: $('run-select').value, issue_time_s: Number($('issue-time').value), mode: $('mode-select').value, horizon_s: Number($('horizon-select').value)};
    body.intent_spec = intentSpec;
    try {
      const result = await api('sessions', {method: 'POST', body: JSON.stringify(body)});
      if (!result.session_id) throw new Error('서버가 세션 ID를 반환하지 않았습니다.');
      state.sessionId = result.session_id;
      rememberSession(result.session_id);
      setText('session-status', `${body.mode === 'llm' ? 'LLM 에이전트' : '규칙 기반 데모'} · ${result.session_id}`);
      connectEvents();
    } catch (error) {showNotice(error.message, true); badge($('decision-badge'), '시작 실패', 'bad'); setText('session-status', '세션을 시작하지 못했습니다'); setRunning(false);}
  }
  function rememberSession(id) {
    const url = new URL(window.location.href);
    url.searchParams.set('session', id);
    window.history.replaceState(window.history.state, '', url);
  }
  async function restoreSession(id) {
    if (!/^[A-Za-z0-9_-]{1,128}$/.test(id)) {
      state.restoring = false; updateRunButton(); showNotice('세션 링크의 ID 형식이 올바르지 않습니다.', true); return;
    }
    state.restoring = true; resetSessionView(); setRunning(true);
    state.sessionId = id; setText('session-status', '저장된 세션을 불러오는 중…');
    try {
      const session = await api(`sessions/${encodeURIComponent(id)}`);
      const events = Array.isArray(session.events) ? session.events : [];
      const started = events.find(event => event.type === 'status' && event.data?.run_id)?.data || {};
      const observed = events.find(event => event.type === 'tool_result' && event.data?.tool === 'get_network_state')?.data?.result || {};
      const request = session.request || session.body || {};
      const runId = request.run_id || started.run_id || observed.run_id;
      const issue = request.issue_time_s ?? started.issue_time_s ?? observed.issue_time_s;
      const horizon = request.horizon_s ?? session.result?.requested_plan?.horizon_s;
      const mode = request.mode || session.result?.mode || started.mode;
      if (finite(issue)) $('issue-time').value = issue;
      if ([90,180].includes(Number(horizon))) $('horizon-select').value = String(horizon);
      await window.AgoraPolicyEditor.load(Number($('horizon-select').value));
      window.AgoraPolicyEditor.restore(request.intent_spec || session.result?.intent?.editor_spec);
      if (['llm','deterministic'].includes(mode)) { $('mode-select').value = mode; modeLabel(); }
      if (request.prompt) { $('intent-prompt').value = request.prompt; state.promptTouched = true; }
      if (runId) {
        let run = state.runs.find(item => item.run_id === runId);
        if (!run) {
          const split = request.split || observed.split;
          if (['train','val','test'].includes(split)) $('split-select').value = split;
          else {
            const all = await api('runs');
            const found = (Array.isArray(all) ? all : all.runs || []).find(item => item.run_id === runId);
            if (found) $('split-select').value = found.split;
          }
          await loadRuns(); run = state.runs.find(item => item.run_id === runId);
        }
        if (run) { $('run-select').value = runId; await loadSnapshot(); }
        else showNotice(`저장된 결과를 표시합니다. 기준 실행 ${runId}의 관측 데이터는 현재 서버에서 찾을 수 없습니다.`, true);
      }
      // Replaying a stored done event must not start a second asynchronous finish.
      const originalPrompt = session.request?.prompt || session.result?.user_prompt;
      if (originalPrompt) $('intent-prompt').value = originalPrompt;
      else $('intent-prompt').value = '이 이전 검증 기록에는 원문 프롬프트가 저장되지 않았습니다. 요청한 파라미터·기간·제약은 아래 submit_intent 도구 기록에서 확인할 수 있습니다.';
      events.forEach(consumeEvent);
      if (session.result) { collectCandidates(session.result, {}); renderDecision(session.result); }
      state.restoring = false;
      if (isDone(session.status)) {
        await finishSession(session);
        setText('session-status', `${mode === 'llm' ? 'LLM' : mode === 'deterministic' ? '규칙 기반 데모' : '저장된 세션'} · 기록 열람 · ${id}`);
        if (!$('global-notice').classList.contains('error')) showNotice(request.intent_spec || session.result?.intent?.editor_spec
          ? '저장된 세션의 목표·제약과 결과를 복원했습니다. LLM이나 GNN 평가를 다시 실행하지 않았습니다.'
          : '이 기록에는 편집기 설정이 저장되어 있지 않습니다. 위 편집기는 다음 실행용 기준이며, 저장된 판정 기준은 후보의 목표·제약 상세에서 확인하세요. 평가를 다시 실행하지 않았습니다.');
      } else {
        setRunning(true); setText('session-status', `진행 중인 세션에 다시 연결합니다 · ${id}`); connectEvents();
      }
    } catch (error) {
      state.restoring = false; setRunning(false); state.sessionId = null;
      badge($('decision-badge'), '불러오기 실패', 'bad'); setText('session-status', '저장된 세션을 불러오지 못했습니다');
      showNotice(`세션 기록을 불러오지 못했습니다. ${error.message}`, true);
    }
  }
  function connectEvents() {
    const id = state.sessionId;
    const source = new EventSource(`api/sessions/${encodeURIComponent(id)}/events`); state.source = source;
    const listener = event => { if (!event.data || id !== state.sessionId) return; try {consumeEvent(JSON.parse(event.data));} catch (error) {console.warn('AGORA event display error:', error.message);} };
    source.onmessage = listener;
    ['status', 'agent_start', 'summary_delta', 'tool_call', 'tool_result', 'candidates', 'search_coverage', 'decision', 'error', 'done'].forEach(type => source.addEventListener(type, listener));
    source.onerror = event => {
      if (event.data || !state.running) return;
      setText('session-status', '스트림 재연결 중 · 서버의 세션 상태를 확인합니다');
    };
    state.poll = setInterval(() => refreshSession(false), 5000);
    refreshSession(false);
  }
  function eventTime(value) {
    if (!value) return '';
    const parsed = new Date(typeof value === 'number' && value < 1e12 ? value*1000 : value);
    return Number.isNaN(parsed.getTime()) ? '' : parsed.toLocaleTimeString('ko-KR', {hour12: false});
  }
  function eventContent(event) {
    if (typeof event.content === 'string') return event.content;
    if (typeof event.data?.summary === 'string') return event.data.summary;
    if (typeof event.data?.message === 'string') return event.data.message;
    return event.content ? asText(event.content) : '';
  }
  function consumeEvent(event) {
    if (!event || typeof event !== 'object') return;
    if (event.seq !== undefined) { const key = String(event.seq); if (state.seen.has(key)) return; state.seen.add(key); }
    const agent = activeAgent(event.agent), type = event.type || 'status';
    if (type === 'agent_start' && agent) {
      if (state.lastAgent && state.lastAgent !== agent) roster(state.lastAgent, 'complete');
      roster(agent, 'running'); state.lastAgent = agent;
    }
    if (type === 'candidates') collectCandidates(event.data || event.content, event);
    if (type === 'tool_result') collectCandidates(event.data, event);
    if (type === 'tool_result' && event.data?.tool === 'prepare_mitigation_context' && event.data.result) renderContextAudit(event.data.result);
    if (type === 'search_coverage') renderSearchCoverage(event.data?.search_coverage);
    if (type === 'decision') {renderDecision(event.data || event.content, eventContent(event)); collectCandidates(event.data, event);}
    if (type === 'summary_delta') {
      const text = eventContent(event); if (!text) return;
      const key = `${agent || event.agent || 'system'}:${event.data?.message_id || event.data?.turn || 'summary'}`;
      let entry = state.summaries.get(key);
      if (!entry) {entry = appendEvent(event, ''); state.summaries.set(key, entry);}
      entry.textContent += text; scrollStream();
    } else {
      // A new agent turn or tool boundary starts a new readable summary block.
      if (type === 'agent_start' || type === 'tool_call' || type === 'tool_result') state.summaries.clear();
      appendEvent(event);
    }
    if (type === 'error') {
      if (event.data?.recoverable) {
        showNotice(`도구 요청을 보정하여 재시도합니다: ${eventContent(event)}`);
      } else {
        if (agent) roster(agent, 'error');
        badge($('decision-badge'), '오류', 'bad'); showNotice(eventContent(event) || '세션 실행 중 오류가 발생했습니다.', true);
      }
    }
    if (type === 'done' && !state.restoring) finishSession(event.data || {});
  }
  function appendEvent(event, customText) {
    $('stream-empty')?.remove();
    const type = event.type || 'status', agent = activeAgent(event.agent);
    const agentInfo = AGENTS[agent] || {name: event.agent || 'Orchestrator', short: 'SY'};
    const isTool = type === 'tool_call' || type === 'tool_result';
    const row = element('article', `stream-event${isTool ? ' tool' : ''}${type === 'error' ? ' error' : ''}`);
    row.append(element('div', 'event-avatar', isTool ? '⌘' : agentInfo.short));
    const body = element('div', 'event-body'), heading = element('div', 'event-header');
    const types = {status: '상태', agent_start: '시작', summary_delta: '판단 요약', tool_call: '도구 요청', tool_result: '도구 응답', candidates: '후보 평가', search_coverage: '탐색 범위', decision: '판단', error: '오류', done: '종료'};
    heading.append(element('span', 'event-name', agentInfo.name), element('span', 'event-type', types[type] || type), element('time', 'event-time', eventTime(event.ts)));
    body.append(heading);
    const defaults = {agent_start: AGENTS[agent]?.description || '에이전트를 시작합니다.', tool_call: '실제 도구 요청을 전송합니다.', tool_result: '실제 도구 응답을 받았습니다.', candidates: '후보 평가 결과를 반영했습니다.', done: '세션 처리가 종료되었습니다.'};
    const text = element('p', 'event-content', customText !== undefined ? customText : eventContent(event) || defaults[type] || '');
    body.append(text);
    const data = event.data;
    if (data && typeof data === 'object' && Object.keys(data).length && (isTool || type === 'decision' || type === 'candidates' || type === 'search_coverage' || type === 'error')) {
      const details = element('details', 'event-details');
      const name = first(data, ['tool_name', 'tool', 'name']);
      details.append(element('summary', '', isTool ? `${name || 'GNN / 분석 도구'} · ${type === 'tool_call' ? '요청 JSON' : '응답 JSON'}` : '반환 데이터 보기'));
      details.append(element('pre', '', jsonSafe(data))); body.append(details);
    }
    row.append(body); $('event-stream').append(row);
    while ($('event-stream').children.length > 350) $('event-stream').firstElementChild.remove();
    scrollStream(); return text;
  }
  function scrollStream() { if ($('auto-scroll').checked) $('event-stream').scrollTop = $('event-stream').scrollHeight; }
  async function refreshSession(force) {
    if (!state.sessionId || (!state.running && !force)) return;
    const id = state.sessionId;
    try {
      const session = await api(`sessions/${encodeURIComponent(id)}`);
      if (id !== state.sessionId) return;
      if (Array.isArray(session.events)) session.events.forEach(consumeEvent);
      if (session.result) {collectCandidates(session.result, {}); if (!state.running || isDone(session.status)) renderDecision(session.result);}
      if (isDone(session.status)) await finishSession(session);
    } catch (error) {if (force) toast(`세션 결과 조회 실패: ${error.message}`, true);}
  }
  function isDone(status) {return ['done', 'completed', 'complete', 'failed', 'error', 'cancelled', 'canceled'].includes(String(status).toLowerCase());}
  async function finishSession(initial) {
    if (state.finishing) return; state.finishing = true;
    const id = state.sessionId;
    state.source?.close(); state.source = null; clearInterval(state.poll); state.poll = null;
    setRunning(false);
    if (state.lastAgent) roster(state.lastAgent, initial?.status === 'error' ? 'error' : 'complete');
    const failed = ['error', 'failed'].includes(initial?.status), cancelled = ['cancelled', 'canceled'].includes(initial?.status);
    setText('session-status', cancelled ? '세션을 중지했습니다' : failed ? '오류로 종료되었습니다' : '평가 완료');
    if ($('decision-badge').textContent === '진행 중') badge($('decision-badge'), cancelled ? '중지' : failed ? '오류' : '완료', failed ? 'bad' : 'neutral');
    $('export-link').href = `api/sessions/${encodeURIComponent(state.sessionId)}/export`; $('export-link').hidden = false;
    $('export-link').setAttribute('download', `agora-${state.sessionId}.json`);
    try {
      const session = initial.result || Array.isArray(initial.events) ? initial : await api(`sessions/${encodeURIComponent(id)}`);
      if (id !== state.sessionId) return;
      if (session.result) {renderDecision(session.result); collectCandidates(session.result, {});}
      else $('export-link').hidden = true;
      window.AgoraOperatorReview.setCompleted(session.status === 'completed');
      if (['failed', 'error'].includes(session.status)) {badge($('decision-badge'), '오류', 'bad'); setText('session-status', '오류로 종료되었습니다');}
    } catch (error) {toast(`종료 기록 조회 실패: ${error.message}`, true);}
  }
  async function cancelSession() {
    if (!state.running || !state.sessionId) return;
    $('cancel-button').disabled = true;
    try {await api(`sessions/${encodeURIComponent(state.sessionId)}/cancel`, {method: 'POST'}); setText('session-status', '중지 요청을 전달했습니다'); await refreshSession(true);}
    catch (error) {toast(error.message, true);} finally {$('cancel-button').disabled = false;}
  }

  function candidateId(candidate, index, round) {return String(first(candidate, ['candidate_id', 'id', 'name', 'label']) || `${round || 'candidate'}-${index}`);}
  function findCandidateArrays(value, depth = 0) {
    if (!value || typeof value !== 'object' || depth > 5) return [];
    if (Array.isArray(value)) return value.length && value.every(v => v && typeof v === 'object' && typeof v.candidate_id === 'string') ? [value] : [];
    const arrays = [];
    for (const key of ['all_candidates', 'candidates', 'top_candidates', 'infeasible_diagnostics', 'evaluations']) if (Array.isArray(value[key])) arrays.push(value[key]);
    for (const key of ['result', 'results', 'output', 'data', 'evaluation', 'initial_evaluation', 'alternative_evaluation', 'combined_evaluation', 'rounds']) {
      if (value[key]) {
        if (Array.isArray(value[key]) && key === 'rounds') value[key].forEach(item => arrays.push(...findCandidateArrays(item, depth+1)));
        else arrays.push(...findCandidateArrays(value[key], depth+1));
      }
    }
    return arrays;
  }
  function collectCandidates(value, event) {
    let changed = false;
    const round = first(value, ['round', 'phase', 'stage']) || event.data?.round || event.agent;
    for (const candidates of findCandidateArrays(value)) candidates.forEach((candidate, index) => {
      if (!candidate || typeof candidate !== 'object' || !candidate.candidate_id) return;
      const id = candidateId(candidate, index, round);
      const existing = state.candidates.get(id) || {};
      const inferred = id === 'requested' ? 'original' : /^b2_/.test(id) ? 'alternative' : /^b[3-9]_/.test(id) ? 'refinement' : '';
      state.candidates.set(id, {...existing, ...candidate, _ui_id: id, _ui_round: candidate.round || candidate.phase || candidate.stage || existing._ui_round || inferred || round || ''}); changed = true;
    });
    if (changed) renderCandidates();
  }
  function candidateStatus(candidate) {
    const conflict = first(candidate, ['conflict', 'has_conflict', 'is_conflict']);
    const feasible = first(candidate, ['feasible', 'is_feasible', 'constraints_satisfied']);
    const violations = first(candidate, ['violations', 'constraint_violations']);
    const status = String(first(candidate, ['status', 'decision', 'verdict']) || '');
    if (status === 'invalid') return {text: '입력 범위 오류', tone: 'bad'};
    if (candidate.conflict_status === 'insufficient_evidence') return {text: '판정 불가 · 근거 부족', tone: 'warn'};
    if (feasible === true && candidate.goals_met === false) return {text: '제약 충족 · 목표 미달', tone: 'warn'};
    if (conflict === true || feasible === false || (Array.isArray(violations) && violations.length) || /infeasible|conflict|reject|violation/i.test(status)) return {text: '충돌 / 제약 위반', tone: 'bad'};
    if (conflict === false || feasible === true || /feasible|pass|safe|accept/i.test(status)) return {text: '제약 충족', tone: 'good'};
    return {text: status || '평가됨', tone: 'neutral'};
  }
  function roundLabel(value) {
    const string = String(value).toLowerCase();
    if (string === '0' || /original|initial|exact|requested/.test(string)) return '01 · 원안';
    if (/combin|second|refinement|round.?2/.test(string) || string === '2') return '03 · 재평가';
    if (/alternative|mitigation|first|round.?1/.test(string) || string === '1') return '02 · 대안';
    return value ? String(value) : '—';
  }
  function formatActions(candidate) {
    const source = first(candidate, ['action', 'actions', 'changes', 'control', 'candidate_rcp', 'rcp', 'parameters', 'candidate_params']);
    if (!source) return first(candidate, ['description', 'label', 'name']) || '상세 데이터 참조';
    if (typeof source === 'string') return source;
    if (Array.isArray(source)) return source.map(item => typeof item === 'object' ? formatActions({action: item}) : String(item)).join(' · ');
    const parameter = first(source, ['parameter', 'rcp', 'param', 'name']);
    const value = first(source, ['value', 'target_value', 'to']);
    if (parameter && value !== undefined) return `${first(source, ['cell', 'cell_id']) || ''} ${parameter} → ${asText(value)}`.trim();
    return Object.entries(source).slice(0, 5).map(([key, value]) => `${key}: ${typeof value === 'object' ? asText(value) : value}`).join(' · ');
  }
  function candidateSummary(candidate) {
    const summary = first(candidate, ['summary', 'reason', 'explanation', 'constraint_summary']);
    if (typeof summary === 'string') return summary;
    const violations = first(candidate, ['violations', 'constraint_violations']) || (candidate.constraints || []).filter(item => item.satisfied === false);
    if (Array.isArray(violations) && violations.length) return violations.map(item => typeof item === 'string' ? item : item.evaluation_status === 'insufficient_evidence' ? `${item.label || item.metric}: 판정 불가` : `${item.label || item.metric}: ${number(item.predicted_value)} / 기준 ${item.operator} ${number(item.effective_threshold)}`).join(' / ');
    const metrics = first(candidate, ['objective_values', 'metrics', 'scores', 'kpi_summary']);
    if (metrics && typeof metrics === 'object') return Object.entries(metrics).slice(0, 3).map(([key, value]) => `${key}: ${finite(value) ? number(value, 3) : asText(value)}`).join(' · ');
    if (candidate.predictions) {
      const gp = kpiValue(candidate.predictions, 0, KPI[0]), prb = kpiValue(candidate.predictions, 0, KPI[1]);
      if (finite(gp) || finite(prb)) return `5G 처리량 ${number(gp)} Mbit/s · PRB ${number(prb)}%`;
    }
    const score = first(candidate, ['score', 'utility', 'objective_score']);
    return finite(score) ? `평가 점수 ${number(score, 4)}` : '세부 KPI와 제약 판정은 상세 기록에서 확인';
  }
  function showCandidateDetails(candidate) {
    setText('candidate-detail-title', candidate._ui_id || candidate.candidate_id);
    setText('candidate-detail-json', jsonSafe(Object.fromEntries(Object.entries(candidate).filter(([key]) => !key.startsWith('_ui_')))));
    let panel = $('candidate-detail-kpis');
    if (!panel) {panel = element('div', 'policy-result'); panel.id = 'candidate-detail-kpis'; $('candidate-detail-json').before(panel);}
    clear(panel);
    for (const [title, rows] of [['KPI 목표 달성', candidate.goals || []], ['Threshold 제약 판정', candidate.constraints || []]]) {
      panel.append(element('h3', '', title));
      const table = element('table'), head = element('tr');
      ['KPI / 범위', '유지안', '후보', '변화', '기준', '판정'].forEach(label => head.append(element('th', '', label))); table.append(head);
      for (const row of rows) {
        const tr = element('tr');
        const unit = row.unit || '';
        const status = row.evaluation_status === 'insufficient_evidence' ? '판정 불가' : row.satisfied ? '충족' : '미달';
        const tone = status === '판정 불가' ? 'unknown' : row.satisfied ? 'pass' : 'fail';
        const group = ({base_constraints:'기본 · ', additional_constraints:'추가 · '})[row.constraint_group] || '';
        const change = finite(row.actual_change_pct) ? `${number(row.actual_change_pct)}%` :
          ['rsrp_interval_mean_dbm', 'sinr_interval_mean_db'].includes(row.metric) && finite(row.predicted_value) && finite(row.reference_value) ? `${number(row.predicted_value-row.reference_value)} dB` : '—';
        tr.append(element('td', '', `${group}${row.label || row.metric} · ${(row.cells || []).join('/')}`), element('td', '', number(row.reference_value, 3)),
          element('td', '', `${number(row.predicted_value, 3)} ${unit}`), element('td', '', change), element('td', '', `${row.operator} ${number(row.effective_threshold, 3)} ${unit}`), element('td', tone, status));
        if (row.reason) tr.title = row.reason;
        table.append(tr);
      }
      panel.append(table);
    }
    if (candidate.power_source?.startsWith('analytic')) panel.append(element('p', '', '전력·에너지는 ON 상태 전력 수식으로 계산했습니다. 처리량·이동성·부하는 GNN 예측입니다.'));
    $('candidate-dialog').showModal();
  }
  function renderCandidates() {
    const body = $('candidate-rows'); clear(body);
    const candidates = Array.from(state.candidates.values());
    metric('metric-candidates', candidates.length, '개'); setText('candidate-count', `${candidates.length}개`);
    setText('metric-candidate-note', candidates.length ? '도구가 반환한 평가 후보' : '아직 평가를 시작하지 않았습니다');
    if (!candidates.length) {const row = element('tr', 'empty-table'); const cell = element('td', '', 'GNN 도구가 반환한 후보만 이 표에 표시합니다.'); cell.colSpan = 7; row.append(cell); body.append(row); return;}
    candidates.forEach(candidate => {
      const row = element('tr', candidate.selected || candidate.recommended ? 'selected-candidate' : '');
      const name = element('td', ''); name.append(element('div', 'candidate-name', candidate._ui_id));
      const label = first(candidate, ['label', 'name', 'description']); if (label && label !== candidate._ui_id) name.append(element('small', '', label));
      const action = element('td', 'result-cell', formatActions(candidate));
      const delay = first(candidate, ['delay_s', 'start_delay_s', 'apply_delay_s', 'start_s', 'apply_at_s']);
      const duration = first(candidate, ['duration_s', 'horizon_s']);
      const timing = element('td', '', finite(delay) ? Number(delay) === 0 ? '즉시' : `${number(delay)}초` : '상세 참조');
      if (finite(duration)) timing.append(element('small', '', `${number(duration)}초 유지`));
      const status = candidateStatus(candidate), statusCell = element('td'), pill = element('span', `pill ${status.tone}`, status.text); statusCell.append(pill);
      const details = element('td'), button = element('button', 'text-button', '목표·제약 ↗'); button.type = 'button'; button.addEventListener('click', () => showCandidateDetails(candidate)); details.append(button);
      row.append(name, element('td', '', roundLabel(candidate._ui_round)), action, timing, element('td', 'result-cell', candidateSummary(candidate)), statusCell, details); body.append(row);
    });
  }
  function auditPanel(id) {
    let panel = $(id);
    if (!panel) {
      panel = element('div', 'search-audit'); panel.id = id;
      const table = document.querySelector('.candidate-table').parentElement;
      table.parentElement.insertBefore(panel, table);
    }
    clear(panel); panel.hidden = false; return panel;
  }
  function renderSearchCoverage(coverage) {
    if (!coverage) return;
    const panel = auditPanel('search-coverage');
    panel.append(element('strong', '', '세 축 탐색 현황'));
    const row = element('div', 'coverage-axes');
    const labels = {time: '시간', intensity: '강도', cell: '대상 셀'};
    for (const [key, info] of Object.entries(coverage.axes || {})) {
      const chip = element('span', `pill ${info.status === 'complete' ? 'good' : info.status === 'pending' ? 'warn' : 'neutral'}`,
        `${labels[key] || key} · ${info.status === 'not_applicable' ? '적용 가능한 대안 없음' : `${info.evaluated}/${info.planned} 평가 · feasible ${info.feasible}`}`);
      chip.title = info.reason || ''; row.append(chip);
    }
    panel.append(row);
    if (coverage.single_cell_grid_candidates != null) panel.append(element('p', '', `단일 제어 격자 ${coverage.single_cell_grid_evaluated}/${coverage.single_cell_grid_candidates} · 다중 셀 추가 확인 필요 ${coverage.joint_refinement_remaining || 0}개 · 배치 최대 128개`));
    for (const target of coverage.cell_targets || []) {
      if (!['blocked', 'not_applicable'].includes(target.status)) continue;
      const reason = String(target.reason || '');
      const translated = reason.includes('supported bound') ? `현재 ${number(target.current_value)}으로, 요청 방향의 추가 변경이 모델 범위를 벗어남` :
        reason.includes('shared controls') ? 'TTT/HYS는 공통 제어이므로 셀 이동 축을 적용할 수 없음' :
        reason.includes('excluded') ? '의도에서 허용하지 않은 셀' : reason.includes('RIC domain') ? '허용된 RIC 영역 밖의 셀' : reason;
      panel.append(element('p', '', `${target.source}${target.target ? ` → ${target.target}` : ''}: ${translated}`));
    }
    const reasons = {original_request_met_all_expectations: '원안이 목표와 제약을 충족하여 대안 탐색이 필요하지 않습니다.',
      goal_met_after_all_axes_checked: '세 축의 기본 후보를 확인했고 목표와 제약을 충족하는 대안을 찾았습니다.',
      goal_met_after_broad_and_joint_search: '셀별 전체 강도·시간 격자와 후순위 다중 셀 검토를 마치고 목표·제약을 충족하는 대안을 찾았습니다.',
      declared_candidate_grid_exhausted: '정의한 유한 후보 범위의 탐색을 완료했습니다. 전체 연속 공간의 불가능성을 뜻하지 않습니다.',
      budget_exhausted_with_unsearched_candidates: '탐색 예산에 도달했습니다. 아직 평가하지 않은 후보가 남아 있습니다.',
      search_incomplete: '미평가 후보를 이어서 검증합니다.'};
    panel.append(element('p', '', `${reasons[coverage.stop_reason] || coverage.stop_reason} ${coverage.combination_candidates != null ? `조합 후보 ${coverage.combinations_evaluated}/${coverage.combination_candidates} 평가.` : ''}`));
  }
  function renderContextAudit(context) {
    const provenance = context.context_provenance; if (!provenance) return;
    const panel = auditPanel('mitigation-context');
    panel.append(element('strong', '', 'Mitigation Agent에 전달한 DAG 문맥'));
    panel.append(element('p', '', `${provenance.dag_features.join(', ')} · 관계 ${provenance.dag_relations}개 · SHAP/DAG는 Agent 문맥이며 GNN 입력이 아닙니다.`));
    const details = element('details', 'event-details');
    details.append(element('summary', '', '실제 전달 문맥과 적용 범위 확인'));
    details.append(element('pre', '', jsonSafe({context_provenance: provenance, shared_numeric_and_scoped_context: context.shared_numeric_and_scoped_context})));
    panel.append(details);
  }
  function renderDecision(payload, fallbackText = '') {
    if (payload === undefined || payload === null) return;
    const data = typeof payload === 'object' ? payload : {summary: String(payload)};
    const decision = data.decision && typeof data.decision === 'object' ? data.decision : data;
    window.AgoraOperatorReview.render(decision, state.sessionId, showCandidateDetails);
    renderSearchCoverage(decision.search_coverage);
    const summary = first(decision, ['summary', 'recommendation', 'message', 'explanation', 'reason', 'final_summary']) || first(data, ['summary', 'final_summary']) || fallbackText;
    const selected = first(decision, ['selected', 'selected_candidate', 'recommended_candidate', 'best_candidate', 'candidate']);
    const verdict = first(decision, ['verdict', 'status', 'decision']);
    const conflicts = first(decision, ['conflict', 'has_conflict']);
    const feasible = first(decision, ['feasible', 'is_feasible']);
    let tone = 'neutral', title = '평가 결과가 도착했습니다';
    if (feasible === false || conflicts === true || /reject|infeasible|conflict|no_feasible/i.test(String(verdict))) {tone = 'warn'; title = '원안 또는 대안의 제약을 확인하세요';}
    if (feasible === true || conflicts === false || selected || /approve|feasible|recommend|accept/i.test(String(verdict))) {tone = 'good'; title = '오프라인 실행안이 정리되었습니다';}
    const titles = {recommended: '요청을 조정한 실행안을 제안합니다', partial_goal: '제약은 충족하지만 목표를 모두 달성하지 못했습니다', no_feasible_candidate: decision.operator_review?.candidates?.length ? '위반이 적은 대안을 운영자 검토안으로 제시합니다' : '탐색한 후보 중 제약을 만족하는 후보가 없습니다', search_incomplete: '탐색 예산에 도달했습니다 · 평가된 후보 중 검토안을 제시합니다'};
    if (verdict && typeof verdict === 'string') title = titles[verdict] || verdict;
    if (['partial_goal', 'no_feasible_candidate', 'search_incomplete'].includes(verdict)) tone = 'warn';
    const content = $('decision-content'); clear(content);
    content.append(element('div', 'decision-symbol', tone === 'good' ? '✓' : tone === 'warn' ? '△' : '◇'), element('h3', '', title));
    const explanation = selected && typeof selected === 'object' ? `${formatActions(selected)} · ${Number(selected.delay_s) === 0 ? '즉시' : `${number(selected.delay_s)}초 뒤`} · ${number(selected.horizon_s)}초 유지\n${candidateSummary(selected)}` : '';
    content.append(element('p', '', summary ? asText(summary) : explanation || '세부 판정 근거와 도구 응답은 에이전트 스트림 및 내보내기 기록에서 확인할 수 있습니다.'));
    if (selected) content.append(element('div', 'decision-meta', `선택 후보: ${typeof selected === 'object' ? first(selected, ['candidate_id', 'id', 'name']) || '상세 기록 참조' : selected}`));
    if (decision.operator_review?.candidates?.length) {const link = element('a', 'text-button', `하단 검토안 ${decision.operator_review.candidates.length}개 비교 · RIC 배포 선택 ↓`); link.href = '#operator-section'; content.append(link);}
    badge($('decision-badge'), tone === 'good' ? '실행안 검토' : tone === 'warn' ? '제약 확인' : '평가 결과', tone);
  }

  async function loadSettings() {
    const settings = await api('settings'); state.settings = settings; return settings;
  }
  async function openSettings() {
    $('settings-dialog').showModal(); $('save-settings').disabled = true; settingsFeedback('설정을 불러오는 중…');
    try {
      const settings = await loadSettings(); state.promptDraft = {...settings.prompts};
      $('prompt-editor').value = state.promptDraft[state.promptAgent] || '';
      $('policy-editor').value = JSON.stringify(settings.policy || {}, null, 2);
      $('provider-model').value = settings.provider?.model || state.status?.provider?.model || '';
      setText('provider-settings-status', (settings.provider?.configured ?? state.status?.provider?.configured) ? '서버의 LLM 인증 설정이 연결되어 있습니다.' : 'LLM 인증이 설정되지 않았습니다. 서버 환경 설정이 필요합니다.');
      settingsFeedback(''); $('save-settings').disabled = false;
    } catch (error) {settingsFeedback(error.message, 'error');}
  }
  function settingsFeedback(message, tone = '') {setText('settings-feedback', message); $('settings-feedback').className = tone;}
  async function saveSettings(event) {
    event.preventDefault(); state.promptDraft[state.promptAgent] = $('prompt-editor').value;
    let policy;
    try {policy = JSON.parse($('policy-editor').value || '{}'); if (!policy || Array.isArray(policy) || typeof policy !== 'object') throw new Error('JSON 객체가 필요합니다.');}
    catch (error) {settingsFeedback(`정책 JSON 오류: ${error.message}`, 'error'); return;}
    $('save-settings').disabled = true; settingsFeedback('저장 중…');
    try {
      const payload = {prompts: state.promptDraft, policy, provider: {model: $('provider-model').value.trim()}};
      const result = await api('settings', {method: 'PUT', body: JSON.stringify(payload)});
      state.settings = result; settingsFeedback('저장했습니다. 다음 세션부터 적용됩니다.', 'good'); toast('프롬프트와 설정을 저장했습니다.'); await loadStatus();
    } catch (error) {settingsFeedback(error.message, 'error');} finally {$('save-settings').disabled = false;}
  }
  function setupEvents() {
    $('intent-form').addEventListener('submit', startSession);
    $('intent-prompt').addEventListener('input', () => {state.promptTouched = true; updateRunButton();});
    $$('[data-intent-family]').forEach(button => button.addEventListener('click', () => {
      state.promptFamily = button.dataset.intentFamily; state.promptTouched = false;
      window.AgoraPolicyEditor.selectPurpose(({txp:'energy_saving', throughput:'throughput', ret:'coverage', cio:'load_balancing', mobility:'handover_control'})[state.promptFamily]); updateRunButton();
    }));
    $('example-cell').addEventListener('change', () => {state.promptTouched = false; renderRunControls(); updateRunButton();});
    $('split-select').addEventListener('change', loadRuns); $('run-select').addEventListener('change', loadSnapshot); $('issue-time').addEventListener('change', loadSnapshot);
    $('mode-select').addEventListener('change', modeLabel); $('horizon-select').addEventListener('change', async () => {
      updateRunButton();
      await Promise.allSettled([loadSnapshot(), window.AgoraPolicyEditor.load(Number($('horizon-select').value)).catch(error => showNotice(error.message, true))]);
    });
    $('refresh-button').addEventListener('click', async () => {showNotice(''); await loadStatus(); if (!state.running) await loadRuns();});
    $('history-kpi').addEventListener('change', renderHistory);
    $$('#history-cells button').forEach(button => button.addEventListener('click', () => {state.historyCell = Number(button.dataset.cell); $$('#history-cells button').forEach(b => b.classList.toggle('selected', b === button)); renderHistory();}));
    $('graph-toggle').addEventListener('click', () => {const body = $('graph-body'); body.hidden = !body.hidden; $('graph-toggle').textContent = body.hidden ? '펼치기 ↓' : '접기 ↑'; $('graph-toggle').setAttribute('aria-expanded', String(!body.hidden));});
    $('clear-stream').addEventListener('click', () => {clear($('event-stream')); state.summaries.clear(); toast('화면을 비웠습니다. 전체 기록은 내보내기에 유지됩니다.');});
    $('cancel-button').addEventListener('click', cancelSession);
    $('open-settings').addEventListener('click', openSettings); $('close-settings').addEventListener('click', () => $('settings-dialog').close()); $('settings-form').addEventListener('submit', saveSettings);
    $('close-candidate').addEventListener('click', () => $('candidate-dialog').close());
    $$('[data-settings-tab]').forEach(button => button.addEventListener('click', () => {
      $$('[data-settings-tab]').forEach(b => b.setAttribute('aria-selected', String(b === button)));
      ['prompts', 'policy', 'provider'].forEach(name => {$(`settings-${name}`).hidden = name !== button.dataset.settingsTab;});
    }));
    $$('[data-prompt-agent]').forEach(button => button.addEventListener('click', () => {
      state.promptDraft[state.promptAgent] = $('prompt-editor').value; state.promptAgent = button.dataset.promptAgent;
      $$('[data-prompt-agent]').forEach(b => b.setAttribute('aria-selected', String(b === button)));
      $('prompt-editor').value = state.promptDraft[state.promptAgent] || ''; setText('prompt-editor-label', `${AGENTS[state.promptAgent].name} · 시스템 프롬프트`);
    }));
    $$('.navigation a').forEach(link => link.addEventListener('click', () => {$$('.navigation a').forEach(a => a.classList.toggle('active', a === link));}));
    window.addEventListener('beforeunload', () => state.source?.close());
  }
  populateHistoryKpis(); setupEvents(); renderGraph();
  const requestedSession = new URL(window.location.href).searchParams.get('session');
  state.restoring = Boolean(requestedSession);
  Promise.allSettled([loadStatus(), loadRuns(), window.AgoraPolicyEditor.init(updateRunButton).catch(error => showNotice(error.message, true))]).then(() => {
    if (requestedSession) return restoreSession(requestedSession);
  });
})();
