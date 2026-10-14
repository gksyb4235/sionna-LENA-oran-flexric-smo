/* Explicit operator selection can dispatch the immutable evaluated plan through real RIC/E2. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const names = {non_rt: 'Non-RT RIC', near_rt: 'Near-RT RIC', split: '제어 종류별 RIC 분리'};
  let model = null, selected = null, completed = false, busy = false, receipt = null, details;
  let liveReceipt = null, liveTimer = null;
  const isLive = () => $('operator-mode').value === 'live';
  async function refreshRuns() {
    const select = $('operator-live-run'), previous = select.value;
    try {
      const response = await fetch('api/live/runs', {cache:'no-store'}), data = await response.json();
      if (!response.ok) throw Error(data.detail || '실행 조회 실패');
      select.replaceChildren(el('option','', '연결할 실행을 선택하세요')); select.firstChild.value='';
      for (const run of data.runs.filter(r => r.process_alive && !r.finished)) {
        const option = el('option','', `${run.replay_run_id || '기준 실행 미지정'} · t=${num(run.sim_time_s)} / ${num(run.stop_time_s)}s · ${run.run_id.slice(0,8)}`);
        option.value=run.run_id; select.append(option);
      }
      if ([...select.options].some(o => o.value === previous)) select.value=previous;
      $('operator-live-help').textContent=(select.options.length>1 ? '같은 기준 실행의 ns3를 선택하세요. ' : '실행 중인 ns3가 없습니다. ')+(model ? `시작 명령: python scripts/agora.py simulate --from-session ${model.sessionId}` : '');
    } catch(error) {select.replaceChildren(el('option','', '연결된 실행 없음'));select.firstChild.value='';$('operator-live-help').textContent=error.message;}
    if(model)updateControls();
  }
  function paintLive(saved) {
    liveReceipt=saved;
    selected=model.review.candidates.find(c=>c.candidate_id===saved.candidate_id)||selected;
    $('operator-mode').value='live';$('operator-ric').value=saved.target_ric;
    const runSelect=$('operator-live-run');
    if(![...runSelect.options].some(o=>o.value===saved.live_run_id)) {
      const option=el('option','',`${saved.live_run_id.slice(0,8)} · 실행 기록`);option.value=saved.live_run_id;runSelect.append(option);
    }
    runSelect.value=saved.live_run_id;
    const labels={accepted:'전송 접수',transmitting:'RIC 전송 중',submitted:'RIC 전송 · ns3 응답 대기',scheduled:'ns3 예약 확인',applied:'ns3 적용 확인',restored:'원래 값 복원 확인',rejected:'ns3 적용 거부',transport_failed:'RIC 전송 실패',restore_conflict:'복원 충돌',unconfirmed_simulation_stopped:'적용 미확인 · 시뮬레이션 종료',restore_unconfirmed_simulation_stopped:'적용 후 시뮬레이션 종료 · 복원 미확인'};
    const box=$('operator-feedback');box.replaceChildren();box.className=['rejected','transport_failed','restore_conflict'].includes(saved.status)?'operator-error':'operator-receipt';
    box.append(el('strong','',labels[saved.status]||saved.status),el('p','',`${names[saved.target_ric]} · ${saved.live_run_id.slice(0,8)} · 시뮬레이션 t=${num(saved.binding.start_s)}~${num(saved.binding.end_s)}s`));
    box.append(el('p','',`정책 저장 ${saved.policy_stored?'완료':'해당 없음'} · RIC 전송 ${saved.forwarded?'확인':'대기'} · ns3 적용 ${saved.applied?'확인':'미확인'} · 복원 ${saved.restored?'확인':'미확인'}`));
    for(const event of saved.receipts||[])box.append(el('p','',`t=${num(event.sim_time_s)}s · ${labels[event.status]||event.status}${event.reason?' · '+event.reason:''}`));
    if(saved.transport_note && !saved.applied)box.append(el('p','',saved.transport_note));
    if(saved.live)box.append(el('p','',`현재 ns3 t=${num(saved.live.sim_time_s)}s · ${saved.live.process_alive?'실행 중':'종료'}`));
    const readback=el('details','operator-ranking');readback.append(el('summary','','ns3 실제 값 · 전달 경로 상세'),el('pre','',JSON.stringify({path:saved.path,binding:saved.binding,receipts:saved.receipts},null,2)));box.append(readback);
    updateControls();
  }
  async function pollLive() {
    if(!model)return;
    const sessionId=model.sessionId;
    try {
      const response=await fetch(`api/sessions/${sessionId}/deployment-live`,{cache:'no-store'});
      if(response.ok && model?.sessionId===sessionId)paintLive(await response.json());
    } catch(error) { /* Keep the last acknowledged state during a reconnect. */ }
  }

  const el = (tag, cls, text) => {const n = document.createElement(tag); n.className = cls || ''; if (text != null) n.textContent = text; return n;};
  const num = v => typeof v === 'number' && Number.isFinite(v) ? new Intl.NumberFormat('ko-KR', {maximumFractionDigits: 3}).format(v) : '—';
  function rowEvidence(entry, goal) {
    const row = el('li', entry.satisfied ? 'operator-pass' : 'operator-fail');
    row.append(el('strong', '', entry.label || entry.metric));
    const known = entry.evaluation_status !== 'insufficient_evidence' && entry.predicted_value != null;
    row.append(el('span', '', known ? `${num(entry.predicted_value)} ${entry.unit || ''} / 기준 ${entry.operator} ${num(entry.effective_threshold)} ${entry.unit || ''}` : '판정 불가 · 근거 부족'));
    if (goal && entry.actual_change_pct != null) row.append(el('small', '', `유지 예측 대비 ${entry.actual_change_pct >= 0 ? '+' : ''}${num(entry.actual_change_pct)}% · ${entry.satisfied ? '목표 달성' : '목표 미달'}`));
    else if (!entry.satisfied) row.append(el('small', '', known ? `위반량 ${num(entry.violation)} ${entry.unit === '%' ? '%p' : entry.unit === 'dBm' ? 'dB' : entry.unit || ''}` : entry.reason || '추가 관측이 필요합니다.'));
    else row.append(el('small', '', goal ? '목표 달성' : '제약 충족'));
    return row;
  }
  function updateControls() {
    const hasReceipt = !!receipt;
    const sent=!!liveReceipt;
    $('operator-ric').disabled = busy || hasReceipt || sent || !selected;
    const invalidLive=isLive() && (!$('operator-live-run').value || $('operator-ric').value==='split');
    $('operator-deploy').disabled = busy || (isLive()?sent:hasReceipt) || invalidLive || !completed || !selected?.deployment_eligible;
    $('operator-deploy').textContent = busy ? '요청 중…' : isLive() ? sent ? '실제 배포 요청 완료' : selected?.requires_constraint_override ? '위반 감수하고 실제 적용' : '선택안 실제 RIC으로 전송' : hasReceipt ? '대기열 등록 완료' : '배포 대기열 등록';
    $('operator-selection').textContent = selected ? `선택: 후보 ${selected.operator_rank} · ${selected.candidate_id}${selected.requires_constraint_override ? ' · 제약 위반 포함' : ''}` : '후보를 선택하세요';
    $('operator-route-note').textContent = isLive() ? $('operator-ric').value==='split' ? '실제 적용은 Near-RT 또는 Non-RT 한 경로를 선택하세요.' : '선택한 ns3에 적용하고 구간 종료 후 변경한 값을 복원합니다.' : '실제 전송 없이 오프라인 대기열에 저장합니다.';
    $('operator-live-run').disabled=busy || sent || !isLive();
    document.querySelectorAll('.operator-choice').forEach(input => {input.disabled = busy || hasReceipt || sent;});
    document.querySelectorAll('.operator-card').forEach(card => {
      const chosen = card.dataset.candidateId === selected?.candidate_id;
      card.classList.toggle('selected', chosen); card.querySelector('input').checked = chosen;
    });
  }
  function paintReceipt(saved) {
    receipt = saved;
    selected = model.review.candidates.find(c => c.candidate_id === saved.candidate_id) || selected;
    $('operator-ric').value = saved.target_ric;
    const box = $('operator-feedback'); box.replaceChildren(); box.className = 'operator-receipt';
    box.append(el('strong', '', `${saved.routes.map(route => names[route.ric]).join(' + ')} 배포 대기열 등록 완료`),
      el('p', '', `${saved.candidate_id} · RIC 전달 전 · 실제 적용 전 · ns3 검증 전`));
    const link = el('a', 'text-button', '등록 실행안 JSON 내려받기 ↗');
    link.href = `api/sessions/${model.sessionId}/deployment`; link.download = `agora-queue-${saved.deployment_id}.json`; box.append(link);
    updateControls();
  }
  function render(result, sessionId, onDetails) {
    if (!result?.operator_review || !sessionId) return;
    const review = result.operator_review;
    if (model?.sessionId === sessionId && model.review.review_sha256 === review.review_sha256) {updateControls(); return;}
    model = {result, sessionId, review}; selected = null; receipt = null; busy = false; details = onDetails;
    liveReceipt=null;clearInterval(liveTimer);liveTimer=setInterval(pollLive,2000);refreshRuns();pollLive();
    $('operator-section').hidden = false;
    $('operator-feedback').replaceChildren(); $('operator-feedback').className = '';
    const feasible = review.candidates.filter(c => c.feasible && c.deployment_eligible).length;
    $('operator-intro').textContent = feasible
      ? `평가된 ${review.evaluated_unique_count}개 중 검토안 ${review.candidates.length}개입니다. 제약 충족 후보를 우선 제시합니다. 목표와 남은 위반을 비교해 하나를 선택하세요.`
      : `제약을 모두 만족하는 후보는 없습니다. 평가된 ${review.evaluated_unique_count}개 중 위반이 적은 검토안 ${review.candidates.length}개를 제시합니다. 기준치는 유지되며, 아래 위반을 감수할지는 운영자가 결정합니다.`;
    const cards = $('operator-cards'); cards.replaceChildren();
    for (const row of review.candidates) {
      const card = el('article', 'operator-card'); card.dataset.candidateId = row.candidate_id;
      const label = el('label', 'operator-card-heading'), radio = el('input', 'operator-choice');
      radio.type = 'radio'; radio.name = 'operator-candidate'; radio.value = row.candidate_id;
      radio.addEventListener('change', () => {selected = row; $('operator-ric').value = row.suggested_ric; updateControls();});
      label.append(radio, el('strong', '', `후보 ${row.operator_rank}`), el('span', `pill ${row.feasible ? 'good' : 'warn'}`, !row.deployment_eligible ? '판정 불가' : row.feasible ? '제약 충족' : '최소 위반 검토안'));
      card.append(label, el('div', 'operator-id', row.candidate_id));
      const current = row.current_parameters || result.execution_plan?.current_parameters || {};
      for (const [key, value] of Object.entries(row.changes)) {
        const unit = ({txp:'dBm', ret:'°', cio:'dB', ttt:'ms', hys:'dB'})[key.split('_')[0]] || '';
        card.append(el('p', 'operator-action', `${key}: ${num(current[key])} → ${num(value)} ${unit}`));
      }
      const interpolated = [...new Set((row.warnings || []).filter(w => w.code === 'interpolated_unvalidated').map(w => `${w.parameter}=${w.value}`))];
      if (interpolated.length) card.append(el('p', 'operator-interpolation', `학습값 사이 보간 · 개별 정확도 미검증: ${interpolated.join(', ')}`));
      card.append(el('p', 'operator-time', `t=${num(row.interval_start_s)}~${num(row.interval_end_s)}s · ${num(row.delay_s)}초 뒤 · ${num(row.horizon_s)}초 유지`));
      const score = row.operator_score;
      card.append(el('div', 'operator-score', row.deployment_eligible ? `제약 위반 ${score.violation_count}개 · 위반 점수 ${num(score.normalized_violation_sum)}` : `판정 근거 부족 · 제약 ${score.unknown_constraints}개 / 목표 ${score.unknown_goals}개`));
      const goals = el('ul', 'operator-evidence'), constraints = el('ul', 'operator-evidence');
      row.goals.forEach(entry => goals.append(rowEvidence(entry, true)));
      row.constraints.forEach(entry => constraints.append(rowEvidence(entry, false)));
      card.append(el('h3', '', 'KPI 향상 목표'), goals, el('h3', '', 'Threshold 제약'), constraints);
      const button = el('button', 'text-button', '전체 예측·목표·제약 상세 ↗'); button.type = 'button';
      button.addEventListener('click', () => details(row)); card.append(button); cards.append(card);
    }
    updateControls();
    fetch(`api/sessions/${sessionId}/deployment`, {cache: 'no-store'}).then(async response => {
      if (response.status === 404) return;
      if (!response.ok) throw Error('이전 배포 등록 내역 조회 실패');
      const saved = await response.json(); if (model?.sessionId === sessionId) {if(saved.live_run_id)paintLive(saved);else if(!liveReceipt)paintReceipt(saved);}
    }).catch(error => {if (model?.sessionId === sessionId) $('operator-feedback').textContent = error.message;});
  }
  async function deploy() {
    if (!model || !selected || busy || (isLive()?liveReceipt:receipt) || !completed) return;
    const sessionId = model.sessionId;
    const body = {candidate_id: selected.candidate_id, target_ric: $('operator-ric').value,
      review_sha256: model.review.review_sha256, accept_constraint_violations: selected.requires_constraint_override};
    if(isLive())body.live_run_id=$('operator-live-run').value;
    busy = true; updateControls(); $('operator-feedback').textContent = '';
    try {
      const response = await fetch(`api/sessions/${sessionId}/${isLive()?'deploy-live':'deploy'}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const saved = await response.json(); if (!response.ok) throw Error(typeof saved.detail === 'string' ? saved.detail : `등록 실패 (HTTP ${response.status})`);
      if (model?.sessionId === sessionId) {if(saved.live_run_id)paintLive(saved);else if(!liveReceipt)paintReceipt(saved);}
    } catch (error) {if (model?.sessionId === sessionId) {$('operator-feedback').textContent = error.message; $('operator-feedback').className = 'operator-error';}}
    finally {if (model?.sessionId === sessionId) {busy = false; updateControls();}}
  }
  document.addEventListener('DOMContentLoaded', () => {
    $('operator-deploy').addEventListener('click',deploy);
    $('operator-refresh-runs').addEventListener('click',refreshRuns);
    for(const id of ['operator-mode','operator-live-run','operator-ric'])$(id).addEventListener('change',()=>{if(model)updateControls();});
  });
  window.AgoraOperatorReview = {render, setCompleted(value) {completed = value; if (model) updateControls();}, reset() {clearInterval(liveTimer);liveReceipt=null;model = selected = receipt = null; completed = busy = false; $('operator-section').hidden = true;}};
})();
