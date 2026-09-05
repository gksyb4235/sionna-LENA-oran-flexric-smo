const runsEl = document.getElementById('runs');
const timelineEl = document.getElementById('timeline');
const detailEl = document.getElementById('detail');
const titleEl = document.getElementById('run-title');
const metaEl = document.getElementById('run-meta');
const approvalEl = document.getElementById('approval');
const refreshBtn = document.getElementById('refresh');
const intentForm = document.getElementById('intent-form');
const intentInput = document.getElementById('intent-input');
const intentSubmitBtn = document.getElementById('intent-submit');
const intentStatusEl = document.getElementById('intent-status');
const graphSvg = document.getElementById('graph-svg');
const graphEmpty = document.getElementById('graph-empty');
const graphHint = document.getElementById('graph-hint');
const graphZoomInBtn = document.getElementById('graph-zoom-in');
const graphZoomOutBtn = document.getElementById('graph-zoom-out');
const graphFitBtn = document.getElementById('graph-fit');
const graphResetBtn = document.getElementById('graph-reset');
const graphPanLeftBtn = document.getElementById('graph-pan-left');
const graphPanRightBtn = document.getElementById('graph-pan-right');
const graphPanUpBtn = document.getElementById('graph-pan-up');
const graphPanDownBtn = document.getElementById('graph-pan-down');
const SVG_NS = 'http://www.w3.org/2000/svg';
let selectedRunId = null;
let selectedEventId = null;
let selectedGraphNodeId = null;
let currentRun = null;
let currentGraph = { nodes: [], edges: [] };
let dragState = null;
let graphPanState = null;
let graphViewport = null;
let graphBounds = { x: 0, y: 0, width: 1280, height: 340 };
let intentInvokeInFlight = false;
let runEventSource = null;
let runStreamConnected = false;
let streamReloadTimer = null;

function statusClass(status) { return String(status || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_'); }
function pretty(value) { return JSON.stringify(value, null, 2); }
function short(text, n = 120) { text = String(text || ''); return text.length > n ? `${text.slice(0, n)}...` : text; }
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch]));
}
function svgEl(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) el.setAttribute(key, String(value));
  }
  return el;
}
function appendText(parent, text, x, y, className) {
  const node = svgEl('text', { x, y, class: className });
  node.textContent = text;
  parent.appendChild(node);
  return node;
}
function graphPositionKey(runId) { return `agent-ops-graph-positions:${runId || 'default'}`; }
function graphViewportKey(runId) { return `agent-ops-graph-viewport:${runId || 'default'}`; }
function loadGraphPositions(runId) {
  try {
    const raw = localStorage.getItem(graphPositionKey(runId));
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (err) {
    return {};
  }
}
function saveGraphPositions(runId, positions) {
  try { localStorage.setItem(graphPositionKey(runId), JSON.stringify(positions)); } catch (err) { /* ignore */ }
}
function graphPointFromEvent(event) {
  const point = graphSvg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(graphSvg.getScreenCTM().inverse());
}
function applyStoredPositions(graph, runId) {
  const positions = loadGraphPositions(runId);
  graph.nodes.forEach(node => {
    const saved = positions[node.id];
    if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) {
      node.x = saved.x;
      node.y = saved.y;
    }
  });
}
function persistNodePosition(runId, node) {
  const positions = loadGraphPositions(runId);
  positions[node.id] = { x: node.x, y: node.y };
  saveGraphPositions(runId, positions);
}
function runIdFor(run) {
  return (run && run.summary && run.summary.run_id) || (run && run.state && run.state.run_id) || selectedRunId || 'default';
}

function isValidViewport(viewport) {
  return viewport && Number.isFinite(viewport.x) && Number.isFinite(viewport.y) && Number.isFinite(viewport.width) && Number.isFinite(viewport.height) && viewport.width > 0 && viewport.height > 0;
}
function loadGraphViewport(runId) {
  try {
    const raw = localStorage.getItem(graphViewportKey(runId));
    const parsed = raw ? JSON.parse(raw) : null;
    return isValidViewport(parsed) ? parsed : null;
  } catch (err) {
    return null;
  }
}
function saveGraphViewport(runId, viewport) {
  if (!isValidViewport(viewport)) return;
  try { localStorage.setItem(graphViewportKey(runId), JSON.stringify(viewport)); } catch (err) { /* ignore */ }
}
function graphSvgSize() {
  const rect = graphSvg.getBoundingClientRect();
  return { width: Math.max(rect.width || 1280, 320), height: Math.max(rect.height || 340, 240) };
}
function computeGraphBounds(graph) {
  if (!graph.nodes.length) return { x: 0, y: 0, width: 1280, height: 340 };
  const minX = Math.min(...graph.nodes.map(node => node.x)) - 90;
  const minY = Math.min(...graph.nodes.map(node => node.y)) - 90;
  const maxX = Math.max(...graph.nodes.map(node => node.x + 240)) + 90;
  const maxY = Math.max(...graph.nodes.map(node => node.y + 120)) + 90;
  return { x: minX, y: minY, width: Math.max(maxX - minX, 520), height: Math.max(maxY - minY, 280) };
}
function fitViewportForGraph(graph) {
  const bounds = computeGraphBounds(graph);
  const size = graphSvgSize();
  const aspect = size.width / size.height;
  let width = bounds.width;
  let height = bounds.height;
  if (width / height > aspect) height = width / aspect;
  else width = height * aspect;
  return {
    x: bounds.x - Math.max(0, width - bounds.width) / 2,
    y: bounds.y - Math.max(0, height - bounds.height) / 2,
    width,
    height,
  };
}
function setGraphViewport(viewport, persist = true) {
  if (!isValidViewport(viewport)) return;
  const runId = runIdFor(currentRun);
  graphViewport = { ...viewport, runId };
  graphSvg.setAttribute('viewBox', `${graphViewport.x} ${graphViewport.y} ${graphViewport.width} ${graphViewport.height}`);
  graphSvg.setAttribute('preserveAspectRatio', 'xMinYMin meet');
  if (persist) saveGraphViewport(runId, graphViewport);
  updateGraphHint();
}
function ensureGraphViewport(graph, run) {
  const runId = runIdFor(run);
  graphBounds = computeGraphBounds(graph);
  if (!graphViewport || graphViewport.runId !== runId) {
    const stored = loadGraphViewport(runId);
    graphViewport = stored ? { ...stored, runId } : { ...fitViewportForGraph(graph), runId };
  }
  setGraphViewport(graphViewport, false);
}
function graphZoomPercent() {
  if (!graphViewport || !currentGraph.nodes.length) return 100;
  const fit = fitViewportForGraph(currentGraph);
  return Math.max(10, Math.round((fit.width / graphViewport.width) * 100));
}
function updateGraphHint() {
  if (!currentGraph.nodes.length) {
    graphHint.textContent = 'LangGraph-style runtime view';
    return;
  }
  graphHint.textContent = `${currentGraph.nodes.length} nodes / ${currentGraph.edges.length} edges / ${graphZoomPercent()}%`;
}
function zoomGraph(factor, center) {
  if (!graphViewport || !currentGraph.nodes.length) return;
  const fit = fitViewportForGraph(currentGraph);
  const minWidth = Math.max(240, fit.width * 0.18);
  const maxWidth = Math.max(fit.width * 4, graphBounds.width * 4);
  const nextWidth = Math.min(maxWidth, Math.max(minWidth, graphViewport.width * factor));
  const ratio = nextWidth / graphViewport.width;
  const pivot = center || { x: graphViewport.x + graphViewport.width / 2, y: graphViewport.y + graphViewport.height / 2 };
  setGraphViewport({
    x: pivot.x - (pivot.x - graphViewport.x) * ratio,
    y: pivot.y - (pivot.y - graphViewport.y) * ratio,
    width: graphViewport.width * ratio,
    height: graphViewport.height * ratio,
  });
}
function panGraphBy(dxRatio, dyRatio) {
  if (!graphViewport) return;
  setGraphViewport({
    x: graphViewport.x + graphViewport.width * dxRatio,
    y: graphViewport.y + graphViewport.height * dyRatio,
    width: graphViewport.width,
    height: graphViewport.height,
  });
}
function fitGraphViewport() {
  if (!currentGraph.nodes.length) return;
  setGraphViewport(fitViewportForGraph(currentGraph));
}
function resetGraphLayout() {
  if (!currentRun) return;
  const runId = runIdFor(currentRun);
  try {
    localStorage.removeItem(graphPositionKey(runId));
    localStorage.removeItem(graphViewportKey(runId));
  } catch (err) { /* ignore */ }
  selectedGraphNodeId = null;
  dragState = null;
  graphPanState = null;
  graphViewport = null;
  graphSvg.classList.remove('dragging', 'panning');
  renderGraph(currentRun);
}


function setIntentStatus(message, status = '') {
  if (!intentStatusEl) return;
  intentStatusEl.textContent = message;
  intentStatusEl.className = `intent-status ${status}`.trim();
}

function hashRunId() {
  return location.hash ? decodeURIComponent(location.hash.slice(1)) : '';
}

function setRunHash(runId) {
  if (!runId || hashRunId() === runId) return;
  history.replaceState(null, '', `#${encodeURIComponent(runId)}`);
}

function closeRunStream() {
  if (runEventSource) runEventSource.close();
  runEventSource = null;
  runStreamConnected = false;
}

function scheduleRunReload(delay = 120) {
  if (streamReloadTimer) clearTimeout(streamReloadTimer);
  streamReloadTimer = setTimeout(async () => {
    streamReloadTimer = null;
    if (dragState || graphPanState) {
      scheduleRunReload(500);
      return;
    }
    try {
      await loadRun();
      await loadRuns();
    } catch (err) {
      console.error(err);
    }
  }, delay);
}

function connectRunStream(runId) {
  if (!window.EventSource || !runId) return;
  if (runEventSource && runEventSource.runId === runId) return;
  closeRunStream();
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);
  source.runId = runId;
  source.onopen = () => { if (runEventSource === source) runStreamConnected = true; };
  source.onmessage = () => { if (runEventSource === source) scheduleRunReload(); };
  source.onerror = () => { if (runEventSource === source) runStreamConnected = false; };
  runEventSource = source;
}

async function selectRun(runId, run = null) {
  if (!runId) return;
  const changed = selectedRunId !== runId;
  selectedRunId = runId;
  if (changed) {
    selectedEventId = null;
    selectedGraphNodeId = null;
    graphViewport = null;
  }
  setRunHash(runId);
  connectRunStream(runId);
  if (run) renderTimeline(run);
  await loadRuns();
  if (!run) await loadRun();
}

async function invokeIntent(event) {
  if (event) event.preventDefault();
  if (!intentInput) return;
  const intent = intentInput.value.trim();
  if (!intent) {
    setIntentStatus('Intent is required.', 'error');
    intentInput.focus();
    return;
  }
  intentInvokeInFlight = true;
  if (intentSubmitBtn) intentSubmitBtn.disabled = true;
  setIntentStatus('Running decomposition -> planning...', 'running');
  try {
    const payload = await fetchJson('/api/invoke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ intent }),
    });
    const runId = (payload.result && payload.result.run_id) || (payload.run && payload.run.summary && payload.run.summary.run_id);
    if (runId) await selectRun(runId, payload.run || null);
    else {
      await loadRuns();
      await loadRun();
    }
    const status = payload.result && payload.result.overall_status ? payload.result.overall_status : 'submitted';
    setIntentStatus(`Run ${short(runId || '', 32)} ${status}.`, status === 'pending_approval' ? 'pending' : 'success');
  } catch (err) {
    setIntentStatus(formatFetchError(err), 'error');
  } finally {
    intentInvokeInFlight = false;
    if (intentSubmitBtn) intentSubmitBtn.disabled = false;
  }
}

function formatFetchError(error) {
  const detail = error && error.detail;
  if (detail && typeof detail === 'object') {
    const stage = detail.stage ? `stage=${detail.stage}` : 'stage=unknown';
    const callId = detail.call_id ? `call_id=${detail.call_id}` : 'call_id=unknown';
    const message = detail.message || error.message || 'Request failed.';
    return `${error.status || 'error'} ${stage} ${callId}: ${message}`;
  }
  return String((error && error.message) || error);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.text();
    let detail = null;
    try {
      const parsed = JSON.parse(body);
      detail = parsed.detail || parsed;
    } catch (err) {
      detail = body;
    }
    const message = typeof detail === 'string' ? detail : (detail.message || body);
    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return response.json();
}

function renderRuns(payload) {
  const runs = payload.runs || [];
  if (!selectedRunId && runs.length) selectedRunId = runs[0].run_id;
  runsEl.innerHTML = '';
  for (const run of runs) {
    const button = document.createElement('button');
    button.className = `run-item ${run.run_id === selectedRunId ? 'active' : ''}`;
    button.innerHTML = `
      <div class="status ${statusClass(run.overall_status)}">${escapeHtml(run.overall_status || 'unknown')}</div>
      <div class="intent">${escapeHtml(short(run.intent || run.run_id))}</div>
      <div class="meta">${escapeHtml(run.current_agent || 'no active agent')} -> ${Number(run.event_count || 0)} events</div>`;
    button.onclick = () => { selectRun(run.run_id).catch(err => { detailEl.textContent = String(err); }); };
    runsEl.appendChild(button);
  }
}

function renderApproval(run) {
  const pending = run.state && run.state.pending_approval;
  if (!pending || !pending.report) {
    approvalEl.classList.add('hidden');
    approvalEl.innerHTML = '';
    return;
  }
  const report = pending.report;
  approvalEl.classList.remove('hidden');
  approvalEl.innerHTML = `
    <div class="approval-title">Pending approval - ${escapeHtml(report.risk_level || 'unknown')} risk</div>
    <pre class="code">${escapeHtml((report.kubectl_preview || []).join('\n'))}</pre>
    <div class="approval-actions">
      <button class="approve">Approve</button>
      <button class="reject">Reject</button>
    </div>`;
  approvalEl.querySelector('.approve').onclick = () => approveRun(true);
  approvalEl.querySelector('.reject').onclick = () => approveRun(false);
}

function nodeKind(raw) {
  if (!raw) return 'agent';
  if (raw.id === 'planning-agent') return 'planning';
  if (raw.role === 'completion_checker' || raw.name === 'planning-completion-check') return 'completion';
  if (raw.type === 'tool') return 'tool';
  return 'agent';
}
function nodeColumn(kind) {
  if (kind === 'planning') return 2;
  if (kind === 'agent') return 4;
  if (kind === 'tool') return 5;
  if (kind === 'completion') return 6;
  return 4;
}
function planStatus(plan) { return plan && plan.status ? plan.status : 'pending'; }

function buildGraph(run) {
  const state = run.state || {};
  const plans = Array.isArray(state.subtask_plans) ? state.subtask_plans : [];
  const events = Array.isArray(run.events) ? run.events : [];
  const nodes = new Map();
  const edges = [];
  const addNode = (id, node) => {
    if (!nodes.has(id)) nodes.set(id, { id, ...node });
    return nodes.get(id);
  };
  const addEdge = (from, to, edge = {}) => {
    if (from && to && from !== to) edges.push({ from, to, ...edge });
  };

  const overall = (run.summary && run.summary.overall_status) || state.overall_status || 'running';
  addNode('user-intent', { label: 'User Intent', subtitle: short(state.intent || (run.summary && run.summary.intent) || 'Intent', 34), kind: 'user', status: overall, x: 50, y: 130, detail: { type: 'intent', intent: state.intent || (run.summary && run.summary.intent) } });
  addNode('decomposition-agent', { label: 'Decomposition Agent', subtitle: 'intent -> subtasks', kind: 'agent', status: 'completed', x: 370, y: 130, detail: { agent: 'decomposition-agent', events: events.filter(e => e.agent === 'decomposition-agent') } });
  addNode('planning-agent', { label: 'Planning Agent', subtitle: 'orchestrator loop', kind: 'planning', status: overall, x: 690, y: 130, detail: { agent: 'planning-agent', state } });
  addEdge('user-intent', 'decomposition-agent', { label: 'intent', status: 'completed' });
  addEdge('decomposition-agent', 'planning-agent', { label: 'handoff', status: overall });

  plans.forEach((plan, index) => {
    const subtaskId = plan.subtask_id || `subtask-${index + 1}`;
    const baseY = 50 + index * 175;
    const status = planStatus(plan);
    const subtaskNodeId = `subtask:${subtaskId}`;
    addNode(subtaskNodeId, {
      label: subtaskId,
      subtitle: short(plan.subtask || 'subtask', 32),
      kind: 'subtask',
      status,
      x: 1010,
      y: baseY,
      detail: plan,
    });
    addEdge('planning-agent', subtaskNodeId, { label: 'select', status });

    const spec = plan.langgraph_spec || {};
    const specNodes = Array.isArray(spec.nodes) ? spec.nodes : [];
    const perColumnCount = {};
    const idForSpecNode = raw => `${subtaskId}:${raw.id || raw.name || Math.random()}`;
    const specIdMap = new Map();
    specNodes.forEach(raw => {
      const kind = nodeKind(raw);
      const col = nodeColumn(kind);
      const offset = perColumnCount[col] || 0;
      perColumnCount[col] = offset + 1;
      const id = idForSpecNode(raw);
      specIdMap.set(raw.id, id);
      addNode(id, {
        label: raw.name || raw.id || kind,
        subtitle: short((raw.capabilities || []).join(', ') || raw.source || raw.type || kind, 34),
        kind,
        status,
        x: 1320 + (col - 4) * 300,
        y: baseY + offset * 72,
        detail: { subtask_id: subtaskId, node: raw, plan_status: status },
      });
    });

    const entryId = specIdMap.get(spec.entrypoint || spec.representative_agent);
    if (entryId) addEdge(subtaskNodeId, entryId, { label: 'entry', status });
    if (Array.isArray(spec.edges)) {
      spec.edges.forEach(edge => {
        const from = specIdMap.get(edge.source);
        const to = specIdMap.get(edge.target);
        if (from && to) addEdge(from, to, { label: edge.condition || '', status });
      });
    }
    if (!specNodes.length && plan.representative_agent) {
      const rep = plan.representative_agent;
      const repId = `${subtaskId}:representative`;
      addNode(repId, { label: rep.name || rep.id || 'Representative', subtitle: rep.id || 'agent', kind: 'agent', status, x: 1320, y: baseY, detail: rep });
      addEdge(subtaskNodeId, repId, { label: 'entry', status });
    }
  });

  if (!plans.length) {
    const targets = [...new Set(events.map(e => e.target_agent).filter(Boolean))];
    targets.forEach((target, index) => {
      addNode(`event-target:${target}`, { label: target, subtitle: 'event target', kind: 'agent', status: overall, x: 1010 + index * 300, y: 130, detail: events.filter(e => e.target_agent === target) });
      addEdge('planning-agent', `event-target:${target}`, { label: 'event', status: overall });
    });
  }

  const graph = { nodes: [...nodes.values()], edges };
  applyStoredPositions(graph, state.run_id || (run.summary && run.summary.run_id));
  return graph;
}

function renderGraph(run) {
  const graph = buildGraph(run);
  drawGraph(graph, run);
}

function drawGraph(graph, run) {
  currentGraph = graph;
  graphSvg.replaceChildren();
  if (!graph.nodes.length) {
    graphEmpty.classList.remove('hidden');
    return;
  }
  graphEmpty.classList.add('hidden');
  ensureGraphViewport(graph, run);

  const defs = svgEl('defs');
  const marker = svgEl('marker', { id: 'arrow', markerWidth: 10, markerHeight: 10, refX: 9, refY: 3, orient: 'auto', markerUnits: 'strokeWidth' });
  marker.appendChild(svgEl('path', { d: 'M0,0 L0,6 L9,3 z', fill: '#8aa2b7' }));
  defs.appendChild(marker);
  graphSvg.appendChild(defs);

  const byId = new Map(graph.nodes.map(n => [n.id, n]));
  const edgeLayer = svgEl('g', { class: 'edge-layer' });
  graph.edges.forEach(edge => {
    const from = byId.get(edge.from);
    const to = byId.get(edge.to);
    if (!from || !to) return;
    const x1 = from.x + 180;
    const y1 = from.y + 34;
    const x2 = to.x;
    const y2 = to.y + 34;
    const bend = Math.max(60, Math.abs(x2 - x1) * 0.35);
    edgeLayer.appendChild(svgEl('path', {
      d: `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`,
      class: `graph-edge ${statusClass(edge.status)}`,
      'marker-end': 'url(#arrow)',
    }));
    if (edge.label) {
      const label = short(edge.label, 18);
      const labelX = (x1 + x2) / 2 - Math.max(28, label.length * 3.2);
      const labelY = (y1 + y2) / 2 - 12;
      edgeLayer.appendChild(svgEl('rect', {
        x: labelX - 8,
        y: labelY - 13,
        width: Math.max(58, label.length * 7.5 + 16),
        height: 20,
        rx: 10,
        ry: 10,
        class: 'graph-edge-label-bg',
      }));
      appendText(edgeLayer, label, labelX, labelY + 2, 'graph-edge-label');
    }
  });
  graphSvg.appendChild(edgeLayer);

  const nodeLayer = svgEl('g', { class: 'node-layer' });
  graph.nodes.forEach(node => {
    const group = svgEl('g', { class: `graph-node ${node.kind} ${statusClass(node.status)} ${node.id === selectedGraphNodeId ? 'selected' : ''}`, transform: `translate(${node.x},${node.y})`, 'data-node-id': node.id });
    group.appendChild(svgEl('rect', { width: 180, height: 68, rx: 8, ry: 8 }));
    appendText(group, short(node.label, 24), 14, 25, 'graph-node-title');
    appendText(group, short(node.subtitle || '', 28), 14, 45, 'graph-node-subtitle');
    group.appendChild(svgEl('rect', { x: 12, y: 52, width: 86, height: 18, rx: 9, ry: 9, class: 'graph-node-badge' }));
    appendText(group, short(node.status || node.kind, 13), 22, 65, 'graph-node-badge-text');
    group.addEventListener('pointerdown', event => {
      event.preventDefault();
      selectedGraphNodeId = node.id;
      const point = graphPointFromEvent(event);
      dragState = {
        runId: runIdFor(run),
        nodeId: node.id,
        offsetX: point.x - node.x,
        offsetY: point.y - node.y,
        moved: false,
      };
      graphSvg.classList.add('dragging');
      detailEl.textContent = pretty(node.detail || node);
      drawGraph(graph, run);
    });
    group.addEventListener('click', () => {
      selectedGraphNodeId = node.id;
      detailEl.textContent = pretty(node.detail || node);
      drawGraph(graph, run);
    });
    nodeLayer.appendChild(group);
  });
  graphSvg.appendChild(nodeLayer);
}

function renderTimeline(run) {
  currentRun = run;
  const summary = run.summary || {};
  titleEl.textContent = summary.intent || summary.run_id || 'Run detail';
  metaEl.textContent = `${summary.overall_status || 'unknown'} -> ${summary.current_agent || 'idle'} -> ${summary.event_count || 0} events`;
  renderApproval(run);
  renderGraph(run);
  const events = run.events || [];
  timelineEl.innerHTML = '';
  if (!events.length) {
    timelineEl.innerHTML = '<div class="event"><div class="event-title">No telemetry events yet</div><div class="event-meta">Planning state is still available in the detail panel.</div></div>';
    detailEl.textContent = pretty(run.state || run.summary);
    return;
  }
  for (const event of events) {
    const node = document.createElement('div');
    node.className = `event ${statusClass(event.status)}`;
    node.innerHTML = `
      <div class="event-title">${escapeHtml(event.agent)} -> ${escapeHtml(event.event_type)}</div>
      <div class="event-meta">${escapeHtml(event.timestamp || '')} -> ${escapeHtml(event.subtask_id || '')} ${event.target_agent ? '-> ' + escapeHtml(event.target_agent) : ''}</div>`;
    node.onclick = () => {
      selectedEventId = event.event_id;
      detailEl.textContent = pretty(event);
    };
    timelineEl.appendChild(node);
  }
  const selected = events.find(item => item.event_id === selectedEventId) || events[events.length - 1];
  detailEl.textContent = pretty(selected);
}

async function loadRuns() {
  const payload = await fetchJson('/api/runs');
  renderRuns(payload);
}

async function loadRun() {
  if (!selectedRunId) return;
  const run = await fetchJson(`/api/runs/${selectedRunId}`);
  renderTimeline(run);
}

async function approveRun(approved) {
  if (!selectedRunId) return;
  const reason = approved ? 'Approved from dashboard.' : 'Rejected from dashboard.';
  const payload = await fetchJson(`/api/runs/${selectedRunId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved, approved_by: 'dashboard', reason })
  });
  await selectRun(selectedRunId, payload.run || null);
}

function updateDraggedNode(event) {
  if (!dragState) return;
  event.preventDefault();
  const node = currentGraph.nodes.find(item => item.id === dragState.nodeId);
  if (!node) return;
  const point = graphPointFromEvent(event);
  const nextX = Math.max(10, point.x - dragState.offsetX);
  const nextY = Math.max(10, point.y - dragState.offsetY);
  if (Math.abs(node.x - nextX) < 0.5 && Math.abs(node.y - nextY) < 0.5) return;
  node.x = nextX;
  node.y = nextY;
  dragState.moved = true;
  persistNodePosition(dragState.runId, node);
  drawGraph(currentGraph, currentRun);
}

function finishDraggedNode() {
  if (!dragState) return;
  const node = currentGraph.nodes.find(item => item.id === dragState.nodeId);
  if (node) persistNodePosition(dragState.runId, node);
  dragState = null;
  graphSvg.classList.remove('dragging');
}

function startGraphPan(event) {
  if (!graphViewport || dragState) return;
  if (event.button !== 0) return;
  if (event.target && event.target.closest && event.target.closest('.graph-node')) return;
  event.preventDefault();
  graphPanState = {
    startClientX: event.clientX,
    startClientY: event.clientY,
    startViewport: { ...graphViewport },
  };
  graphSvg.classList.add('panning');
}

function updateGraphPan(event) {
  if (!graphPanState || !graphViewport) return;
  event.preventDefault();
  const size = graphSvgSize();
  const dx = (event.clientX - graphPanState.startClientX) * (graphPanState.startViewport.width / size.width);
  const dy = (event.clientY - graphPanState.startClientY) * (graphPanState.startViewport.height / size.height);
  setGraphViewport({
    x: graphPanState.startViewport.x - dx,
    y: graphPanState.startViewport.y - dy,
    width: graphPanState.startViewport.width,
    height: graphPanState.startViewport.height,
  }, false);
}

function finishGraphPan() {
  if (!graphPanState) return;
  graphPanState = null;
  graphSvg.classList.remove('panning');
  if (graphViewport) saveGraphViewport(runIdFor(currentRun), graphViewport);
}

window.addEventListener('pointermove', event => {
  updateDraggedNode(event);
  updateGraphPan(event);
});
window.addEventListener('pointerup', () => {
  finishDraggedNode();
  finishGraphPan();
});
window.addEventListener('pointercancel', () => {
  finishDraggedNode();
  finishGraphPan();
});
graphSvg.addEventListener('pointerdown', startGraphPan);
graphSvg.addEventListener('wheel', event => {
  if (!currentGraph.nodes.length) return;
  event.preventDefault();
  const center = graphPointFromEvent(event);
  zoomGraph(event.deltaY < 0 ? 0.88 : 1.14, center);
}, { passive: false });
if (graphZoomInBtn) graphZoomInBtn.onclick = () => zoomGraph(0.82);
if (graphZoomOutBtn) graphZoomOutBtn.onclick = () => zoomGraph(1.22);
if (graphFitBtn) graphFitBtn.onclick = fitGraphViewport;
if (graphResetBtn) graphResetBtn.onclick = resetGraphLayout;
if (graphPanLeftBtn) graphPanLeftBtn.onclick = () => panGraphBy(-0.18, 0);
if (graphPanRightBtn) graphPanRightBtn.onclick = () => panGraphBy(0.18, 0);
if (graphPanUpBtn) graphPanUpBtn.onclick = () => panGraphBy(0, -0.18);
if (graphPanDownBtn) graphPanDownBtn.onclick = () => panGraphBy(0, 0.18);

refreshBtn.onclick = async () => { await loadRuns(); await loadRun(); };
if (intentForm) intentForm.addEventListener('submit', invokeIntent);
window.addEventListener('beforeunload', closeRunStream);
window.addEventListener('hashchange', () => {
  const runId = hashRunId();
  if (runId && runId !== selectedRunId) selectRun(runId).catch(err => { detailEl.textContent = String(err); });
});
setInterval(async () => {
  try {
    if (dragState || graphPanState || intentInvokeInFlight || runStreamConnected) return;
    await loadRuns();
    await loadRun();
    if (selectedRunId) connectRunStream(selectedRunId);
  } catch (err) {
    console.error(err);
  }
}, 3000);

async function initDashboard() {
  const initialRunId = hashRunId();
  if (initialRunId) selectedRunId = initialRunId;
  await loadRuns();
  if (selectedRunId) {
    connectRunStream(selectedRunId);
    await loadRun();
  }
}

initDashboard().catch(err => { detailEl.textContent = String(err); });
