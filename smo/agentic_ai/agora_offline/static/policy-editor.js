/* Structured, user-owned intent policy. The server validates every active row. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const clone = value => JSON.parse(JSON.stringify(value));
  const groups = {goals: 'policy-goals', base_constraints: 'policy-base', additional_constraints: 'policy-additional'};
  const cells = ['5g', '4g1', '4g2'];
  const scopes = [['5g,4g1,4g2', '네트워크 전체'], ['5g', 'Cell5G'], ['4g1', 'Cell4G1'], ['4g2', 'Cell4G2'],
    ['5g,4g1', '5G + 4G1'], ['5g,4g2', '5G + 4G2'], ['4g1,4g2', '4G1 + 4G2']];
  let catalog, spec, pending = 0, busy = false, changed = false, notify = () => {};
  function node(tag, className, text) { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; }
  function unique(prefix) { return `${prefix}_${crypto.randomUUID().slice(0, 8)}`; }
  function option(row) { return catalog.metrics.find(m => m.key === row.option) || catalog.metrics.find(m => m.metric === row.metric && m.aggregation === row.aggregation); }
  function recommended(row) {
    const key = option(row)?.key;
    if (!key) return row.recommendation || {target:null, note:'이전 정의의 KPI입니다. 새 모델의 예측 지원 여부를 확인하세요.'};
    if (['rlf', 'pp_ratio', 'pp_success', 'rsrp_p10', 'rsrp_gap', 'bler', 'delay_tail'].includes(key) && row.cells.length !== 3) return {target: null, note: '셀별 / 부분 셀 그룹에는 기본 상한이 없습니다. 허용값을 직접 입력하세요.', source: 'user_required'};
    return option(row)?.recommendation || {target: null, note: '기준값을 직접 입력하세요.', source: 'user_required'};
  }
  function refreshSummary() {
    if (!spec) return;
    const goalCount = spec.goals.filter(r => r.enabled !== false).length;
    const limits = [...spec.base_constraints, ...spec.additional_constraints];
    const active = limits.filter(r => r.enabled !== false).length;
    $('policy-summary').textContent = `${spec.horizon_s}초 평가 · 목표 ${goalCount}개 · 활성 제약 ${active}개 · 비활성 제약 ${limits.length-active}개${active === 0 ? ' · 제약 없는 평가입니다' : ''}`;
    $('policy-json').textContent = JSON.stringify(spec, null, 2);
    notify();
  }
  function field(label, input) { const wrap = node('label', 'policy-field'); wrap.append(node('span', '', label), input); return wrap; }
  function select(options, value, handler) {
    const input = node('select'); for (const [v, label] of options) input.add(new Option(label, v)); input.value = value;
    input.addEventListener('change', () => { changed = true; handler(input.value); }); return input;
  }
  function numeric(value, handler, min) {
    const input = node('input'); input.type = 'number'; input.step = 'any'; input.value = value ?? ''; if (min !== undefined) input.min = min;
    input.addEventListener('input', () => { changed = true; handler(input.value === '' ? null : Number(input.value)); refreshSummary(); }); return input;
  }
  function applyOption(row, descriptor, goal) {
    Object.assign(row, {option: descriptor.key, metric: descriptor.metric, aggregation: descriptor.aggregation,
      label: descriptor.label, unit: descriptor.unit, note: descriptor.note,
      reference_kind: 'keep_forecast_same_window', operator: descriptor.direction === 'decrease' ? '<=' : '>='});
    if (goal) {
      row.target_kind = ['rsrp', 'sinr', 'rsrp_p10', 'rsrp_gap'].includes(descriptor.key) ? 'absolute_delta' : 'relative_fraction';
      row.target = row.target_kind === 'absolute_delta' ? (row.operator === '<=' ? -3 : 3) : row.operator === '<=' ? -.05 : .05;
    } else {
      row.target_kind = 'absolute'; const rec = recommended(row);
      row.target = rec.target; if (rec.target !== null) row.operator = rec.operator;
      row.follows_recommendation = true;
    }
  }
  function renderRow(row, group) {
    const goal = group === 'goals';
    const card = node('div', `policy-row${row.enabled === false ? ' is-disabled' : ''}`); card.dataset.policyId = row.id;
    const line = node('div', `policy-row-fields${goal ? ' goal-fields' : ''}`);
    const enabled = node('input'); enabled.type = 'checkbox'; enabled.checked = row.enabled !== false; enabled.setAttribute('aria-label', goal ? '목표 활성화' : '제약 활성화');
    enabled.addEventListener('change', () => { changed = true; row.enabled = enabled.checked; render(); });
    const toggle = node('label', 'policy-enabled'); toggle.append(enabled, node('span', '', '사용')); line.append(toggle);
    const currentOption = option(row), choices = catalog.metrics.map(m => [m.key, m.label]);
    if (!currentOption) choices.push([row.option || row.metric, `${row.label || row.metric} · 이전 정의`]);
    line.append(field('KPI', select(choices, currentOption?.key || row.option || row.metric, value => {
      const descriptor=catalog.metrics.find(m=>m.key===value); if (descriptor) applyOption(row,descriptor,goal); render();
    })));
    line.append(field('평가 범위', select(scopes, cells.filter(c => row.cells.includes(c)).join(','), value => {
      row.cells = value.split(','); if (!goal && row.follows_recommendation) row.target = recommended(row).target; render();
    })));
    if (goal) {
      const logMetric = ['rsrp_interval_mean_dbm', 'sinr_interval_mean_db', 'rsrp_p10_dbm', 'rsrp_mean_p10_gap_db'].includes(row.metric);
      const choices = [['relative_fraction', '% 개선'], ['absolute_delta', '변화량'], ['absolute', '도달값']].filter(([v]) => !logMetric || v !== 'relative_fraction');
      line.append(field('목표 방식', select(choices, row.target_kind, value => {row.target_kind = value; row.target = null; render();})));
      line.append(field(row.target_kind === 'absolute' ? '조건' : '방향', select([['<=', row.target_kind === 'absolute' ? '이하' : '감소'], ['>=', row.target_kind === 'absolute' ? '이상' : '증가']], row.operator, value => {
        row.operator = value; if (row.target !== null && row.target_kind !== 'absolute') row.target = Math.abs(row.target)*(value === '<=' ? -1 : 1); render();
      })));
    } else line.append(field('조건', select([['<=', '이하 (≤)'], ['>=', '이상 (≥)']], row.operator, value => {row.operator = value; row.follows_recommendation = false; render();})));
    const percent = goal && row.target_kind === 'relative_fraction';
    const signed = goal && row.target_kind !== 'absolute';
    const value = row.target == null ? null : percent ? Math.abs(row.target)*100 : signed ? Math.abs(row.target) : row.target;
    const input = numeric(value, value => {row.target = value === null ? null : (signed ? (row.operator === '<=' ? -1 : 1) : 1)*value/(percent ? 100 : 1); row.follows_recommendation = false;}, signed ? 0 : undefined);
    input.required = row.enabled !== false; input.disabled = row.enabled === false;
    const targetUnit = percent ? '%' : goal && row.target_kind === 'absolute_delta' && row.unit === 'dBm' ? 'dB' : row.unit || '';
    line.append(field(`${goal ? '목표' : 'Threshold'} (${targetUnit})`, input));
    if (goal) { const weight = numeric(row.weight ?? 1, value => {row.weight = value;}, .001); weight.required = row.enabled !== false; weight.disabled = row.enabled === false; line.append(field('가중치', weight)); }
    const remove = node('button', 'policy-remove', '×'); remove.type = 'button'; remove.setAttribute('aria-label', `${row.label} 삭제`);
    remove.addEventListener('click', () => {changed = true; spec[group] = spec[group].filter(r => r.id !== row.id); render();}); line.append(remove);
    card.append(line);
    const notes = node('div', 'policy-row-note');
    if (goal) notes.append(node('span', '', `${row.note || ''} · 기준: 같은 미래 구간의 설정 유지 예측`));
    else {
      const rec = recommended(row), defined = rec.target !== null && rec.target !== undefined;
      notes.append(node('span', '', defined ? `추천 ${Number(rec.target.toFixed(4))} ${row.unit} ${rec.operator === '<=' ? '이하' : '이상'} · ${rec.note}` : rec.note));
      if (defined) {
        const restore = node('button', '', '추천값 적용'); restore.type = 'button'; restore.addEventListener('click', () => {changed = true; row.target = rec.target; row.operator = rec.operator; row.follows_recommendation = true; render();}); notes.append(restore);
      }
      if (row.note) notes.append(node('small', '', row.note));
    }
    card.append(notes); return card;
  }
  function render() {
    if (!spec || !catalog) return;
    $('policy-purpose').value = spec.objective;
    for (const [group, id] of Object.entries(groups)) {
      $(id).replaceChildren(...spec[group].map(row => renderRow(row, group)));
      if (!spec[group].length) $(id).append(node('p', 'policy-empty', group === 'goals' ? '평가할 목표를 하나 이상 추가하세요.' : '등록한 제약이 없습니다.'));
    }
    $('intent-policy-editor').disabled = busy; refreshSummary();
  }
  async function load(horizon) {
    const request = ++pending;
    const response = await fetch(`api/intent-policy?horizon_s=${horizon}`, {cache: 'no-store'});
    if (!response.ok) throw new Error('목표·제약 기준을 불러오지 못했습니다.');
    const data = await response.json(); if (request !== pending) return;
    catalog = data;
    if (!spec) spec = clone(catalog.templates[$('policy-purpose').value]);
    else {
      spec.horizon_s = horizon;
      for (const row of [...spec.base_constraints, ...spec.additional_constraints]) if (row.follows_recommendation) {
        const rec = recommended(row); row.target = rec.target; if (rec.target !== null) row.operator = rec.operator;
      }
    }
    render();
  }
  function getSpec() {
    if (!spec || !catalog) throw new Error('목표·제약 기준을 먼저 불러와야 합니다.');
    if (spec.horizon_s !== Number($('horizon-select').value)) throw new Error('평가 구간의 기준을 불러오는 중입니다.');
    if (!spec.goals.some(r => r.enabled !== false)) throw new Error('활성 KPI 개선 목표가 하나 이상 필요합니다.');
    for (const group of Object.keys(groups)) for (const row of spec[group]) if (row.enabled !== false) {
      if (row.target === null || !Number.isFinite(row.target)) throw new Error(`${row.label}의 ${group === 'goals' ? '목표' : 'threshold'}를 입력하세요.`);
      if (group === 'goals' && (!(row.weight > 0) || !Number.isFinite(row.weight))) throw new Error('목표 가중치는 0보다 커야 합니다.');
    }
    return clone(spec);
  }
  function add(group) {
    const row = {id: unique(group), cells: [...cells], enabled: true, weight: 1};
    applyOption(row, catalog.metrics.find(m => m.key === (group === 'goals' ? 'goodput' : 'prb_max')), group === 'goals');
    spec[group].push(row); changed = true; render();
  }
  window.AgoraPolicyEditor = {
    async init(onChange) {
      notify = onChange;
      $('policy-purpose').addEventListener('change', () => {if (spec) {spec.objective = $('policy-purpose').value; changed = true; refreshSummary();}});
      $('policy-preset').addEventListener('click', () => {
        spec.goals = clone(catalog.templates[spec.objective].goals);
        for (const row of clone(catalog.templates[spec.objective].additional_constraints)) if (!spec.additional_constraints.some(r => r.option === row.option)) {row.id = unique('extra'); spec.additional_constraints.push(row);}
        changed = true; render();
      });
      $('reset-base-constraints').addEventListener('click', () => {spec.base_constraints = clone(catalog.templates[spec.objective].base_constraints); changed = true; render();});
      $('add-goal').addEventListener('click', () => add('goals'));
      $('add-base-constraint').addEventListener('click', () => add('base_constraints'));
      $('add-constraint').addEventListener('click', () => add('additional_constraints'));
      await load(Number($('horizon-select').value));
    },
    load, getSpec,
    ready: () => Boolean(spec && catalog && spec.horizon_s === Number($('horizon-select').value)),
    setBusy(value) {busy = value; $('intent-policy-editor').disabled = value;},
    selectPurpose(purpose) {if (!spec || !catalog) return; if (!changed) spec = clone(catalog.templates[purpose]); else spec.objective = purpose; render();},
    restore(value) {if (['threshold-v2','threshold-v3-app5'].includes(value?.version)) {spec = clone(value); changed = true; render();}},
  };
})();
