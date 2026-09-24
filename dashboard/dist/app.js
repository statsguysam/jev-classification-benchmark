const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
})[c]);
const names = {
  sst2: 'SST-2',
  trec: 'TREC',
  imdb: 'IMDb',
  ag_news: 'AG News',
  banking77: 'Banking77',
  titanic: 'Titanic',
  breast_cancer: 'Breast Cancer',
  wine: 'Wine'
};
const familyNames = {
  jev: 'Jev',
  open: 'Open LLM',
  hosted: 'Hosted LLM',
  classical: 'Classical ML'
};
const methodNames = {
  zero_shot: 'Zero-shot',
  few_shot: 'Few-shot',
  lora: 'LoRA',
  qlora: 'QLoRA',
  classical: 'Classical fit'
};
const colors = {
  jev: '#008d99',
  open: '#8570c4',
  hosted: '#3866da',
  classical: '#c17b36'
};
const metricNames = {
  accuracy: 'Accuracy',
  macro_f1: 'Macro-F1',
  balanced_accuracy: 'Balanced accuracy',
  failure_rate: 'Failure rate',
  log_loss: 'Log loss',
  brier_score: 'Brier score',
  ece: 'Calibration error'
};
const defaults = {
  type: 'all',
  task: 'all',
  dataset: 'all',
  family: 'all',
  model: 'all',
  method: 'all',
  budget: 'matched',
  seed: '42',
  coverage: 'all',
  replicas: false
};
const filters = {
  ...defaults
};
let DATA,
  filtered = [],
  focus = 'titanic',
  metric = 'macro_f1',
  tab = 'compare',
  page = 0,
  chartSvg = '',
  search = '';
const number = x => Number.isFinite(Number(x)) && x !== null && x !== undefined ? Number(x) : null;
const percent = x => number(x) === null ? 'N/A' : `${(Number(x) * 100).toFixed(1)}%`;
const score = x => number(x) === null ? 'N/A' : Number(x).toFixed(3);
const count = x => Number(x || 0).toLocaleString('en-US');
const displayModel = r => r.display_model || r.model_display || String(r.model).replace('Qwen/', '').replace('-Instruct-2507', '').replace('-Instruct', '').replace('typesafe/', '').replaceAll('_', ' ');
const dataType = x => ({
  text: 'Text',
  numeric: 'Numeric tabular',
  mixed: 'Mixed tabular'
})[x] || x;
const training = r => r.train_per_class === null ? 'Full training' : r.train_per_class === 0 ? '0 labels' : `${r.train_per_class} / class`;
function getMetric(r, m = metric) {
  if (m === 'failure_rate') return r.n_test ? r.n_failures / r.n_test : null;
  if (['log_loss', 'brier_score', 'ece'].includes(m)) {
    const p = r.probability_metrics || {};
    return number(p[m] ?? (m === 'brier_score' ? p.brier_sum : m === 'ece' ? p.ece_15_equal_width : null));
  }
  return number(r[m]);
}
function formatMetric(value, m = metric) {
  return m === 'accuracy' || m === 'balanced_accuracy' || m === 'failure_rate' ? percent(value) : score(value);
}
function budgetMatches(r, b) {
  if (b === 'all') return true;
  if (b === 'full') return r.train_per_class === null;
  if (b === 'matched') return r.train_per_class === 0 || r.train_per_class === 4;
  return r.train_per_class === Number(b);
}
function match(r, f = filters) {
  return (f.replicas || r.canonical !== false) && (f.type === 'all' || r.data_type === f.type) && (f.task === 'all' || r.task === f.task) && (f.dataset === 'all' || r.dataset === f.dataset) && (f.family === 'all' || r.family === f.family) && (f.model === 'all' || r.model === f.model) && (f.method === 'all' || r.method === f.method) && budgetMatches(r, f.budget) && (f.seed === 'all' || r.seed === Number(f.seed)) && (f.coverage === 'all' || (f.coverage === 'probabilities' ? r.probability_coverage > 0 : r.n_failures > 0));
}
function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $('toast').hidden = true, 3000);
}
function fillSelect(id, values, current, allLabel) {
  const select = $(id);
  select.innerHTML = (allLabel ? `<option value="all">${esc(allLabel)}</option>` : '') + values.map(([value, label]) => `<option value="${esc(value)}">${esc(label)}</option>`).join('');
  select.value = values.some(([v]) => v === current) || current === 'all' ? current : allLabel ? 'all' : values[0]?.[0] || '';
  return select.value;
}
function updateChoices() {
  const datasetRows = DATA.runs.filter(r => (filters.type === 'all' || r.data_type === filters.type) && (filters.task === 'all' || r.task === filters.task));
  const ds = [...new Set(datasetRows.map(r => r.dataset))];
  filters.dataset = fillSelect('dataset', ds.map(x => [x, names[x] || x]), filters.dataset, 'All datasets');
  const models = [...new Map(DATA.runs.filter(r => filters.family === 'all' || r.family === filters.family).map(r => [r.model, displayModel(r)])).entries()].sort((a, b) => a[1].localeCompare(b[1]));
  filters.model = fillSelect('model', models, filters.model, 'All models');
}
function encodeState() {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) if (v !== defaults[k]) p.set(k, String(v));
  if (focus !== 'titanic') p.set('focus', focus);
  if (metric !== 'macro_f1') p.set('metric', metric);
  if (tab !== 'compare') p.set('view', tab);
  history.replaceState(null, '', location.pathname + (p.size ? '?' + p : '') + location.hash);
}
function render() {
  filtered = DATA.runs.filter(r => match(r));
  const ds = [...new Set(filtered.map(r => r.dataset))];
  focus = fillSelect('focus', ds.map(d => [d, names[d] || d]), focus);
  $('metric').value = metric;
  $('kpi-runs').textContent = count(filtered.length);
  $('kpi-predictions').textContent = count(filtered.reduce((s, r) => s + r.n_test, 0));
  $('kpi-datasets').textContent = ds.length;
  $('kpi-failures').textContent = count(filtered.reduce((s, r) => s + r.n_failures, 0));
  $('kpi-runs').previousElementSibling.textContent = filters.replicas ? 'Executions in view' : 'Conditions in view';
  $('kpi-scope').textContent = filters.replicas ? 'Executions, including replicas' : 'Distinct measured conditions';
  $('kpi-types').textContent = [...new Set(filtered.map(r => dataType(r.data_type)))].join(' · ') || 'No matching datasets';
  const canonical = DATA.runs.filter(r => r.canonical !== false).length;
  $('status').textContent = `${count(filtered.length)} ${filters.replicas ? 'executions' : 'conditions'} in view · ${count(canonical)} distinct conditions in the study · ${filters.seed === 'all' ? 'all selection seeds' : 'seed ' + filters.seed}`;
  $('jev-only').classList.toggle('active', filters.family === 'jev');
  $('all-models').classList.toggle('active', filters.family === 'all' && filters.model === 'all');
  renderChart();
  renderDataset();
  renderJev();
  renderTable();
  renderAudit();
  encodeState();
}
function renderChart() {
  const rows = filtered.filter(r => r.dataset === focus).filter(r => getMetric(r) !== null);
  const low = ['failure_rate', 'log_loss', 'brier_score', 'ece'].includes(metric);
  const sorted = [...rows].sort((a, b) => (low ? 1 : -1) * (getMetric(a) - getMetric(b)) || displayModel(a).localeCompare(displayModel(b)));
  const allFocus = filtered.filter(r => r.dataset === focus);
  const task = allFocus[0];
  $('chart-context').textContent = task ? `${names[focus] || focus} · ${dataType(task.data_type)} · ${task.n_classes || task.labels?.length || ''} classes · ${count(task.n_test)} held-out rows${rows.length < allFocus.length ? ` · ${allFocus.length - rows.length} conditions have no ${metricNames[metric].toLowerCase()} value` : ''}` : 'No dataset matches the selected filters.';
  if (!rows.length) {
    $('main-chart').innerHTML = '<div class="empty"><strong>No measured values here</strong>Change a filter or select a metric available for these models.</div>';
    chartSvg = '';
    $('export-chart').disabled = true;
    return;
  }
  $('export-chart').disabled = false;
  const width = Math.max(620, Math.min(1200, $('main-chart').clientWidth - 36)),
    left = Math.max(205, Math.min(290, width * .34)),
    right = 65,
    rowH = 47,
    top = 28,
    bottom = 43,
    height = top + bottom + rowH * sorted.length;
  const plotW = width - left - right;
  const isBounded = !['log_loss', 'brier_score'].includes(metric);
  const max = isBounded ? 1 : Math.max(...rows.map(r => getMetric(r)), .1) * 1.12;
  const X = v => left + Math.max(0, Math.min(v / max, 1)) * plotW;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="group" aria-label="${esc(metricNames[metric] + ' by model on ' + names[focus])}" style="font-family:system-ui,sans-serif"><title>${esc(names[focus])}: ${esc(metricNames[metric])}</title><rect width="100%" height="100%" fill="white"/>`;
  for (let i = 0; i <= 4; i++) {
    const value = max * i / 4,
      x = X(value);
    svg += `<line x1="${x}" x2="${x}" y1="${top - 7}" y2="${height - bottom + 4}" stroke="#e7edf5"/><text x="${x}" y="${height - 15}" text-anchor="middle" fill="#8390a3" font-size="11">${['accuracy', 'balanced_accuracy', 'failure_rate'].includes(metric) ? (value * 100).toFixed(0) + '%' : value.toFixed(isBounded ? 2 : 1)}</text>`;
  }
  sorted.forEach((r, i) => {
    const y = top + i * rowH,
      color = colors[r.family],
      value = getMetric(r),
      ci = r.ci?.[metric],
      label = displayModel(r),
      detail = `${methodNames[r.method] || r.method} · ${training(r)}${filters.seed === 'all' ? ' · s' + r.seed : ''}${!r.canonical ? ' · replica' : ''}`;
    svg += `<g class="chart-run" data-id="${esc(r.id)}" tabindex="0" role="button" aria-label="${esc(label + ', ' + detail + ', ' + metricNames[metric] + ' ' + formatMetric(value) + '. View run details')}" style="cursor:pointer"><title>${esc(label + ' / ' + detail + ' / ' + formatMetric(value) + (ci ? ' / 95% CI ' + ci.map(x => formatMetric(x)).join(' to ') : ' / no interval available'))}</title><rect x="0" y="${y - 14}" width="${width}" height="${rowH}" fill="${i % 2 ? '#fafbfd' : '#fff'}"/><circle cx="10" cy="${y - 1}" r="3.5" fill="${color}"/><text x="23" y="${y + 1}" fill="#2b3c55" font-size="14" font-weight="600">${esc(label.length > Math.floor((left - 30) / 7.5) ? label.slice(0, Math.floor((left - 30) / 7.5) - 1) + "…" : label)}</text><text x="23" y="${y + 18}" fill="#77859a" font-size="12">${esc(detail.length > Math.floor((left - 25) / 6.2) ? detail.slice(0, Math.floor((left - 25) / 6.2) - 1) + "…" : detail)}</text><rect x="${left}" y="${y - 5}" width="${Math.max(1, X(value) - left)}" height="11" rx="3" fill="${color}" opacity=".86"/>`;
    if (ci && ci.length === 2 && ci.every(v => number(v) !== null)) svg += `<line x1="${X(ci[0])}" x2="${X(ci[1])}" y1="${y + .5}" y2="${y + .5}" stroke="#172b46" stroke-width="1.1"/><path d="M${X(ci[0])} ${y - 3}v7 M${X(ci[1])} ${y - 3}v7" stroke="#172b46" stroke-width="1.1"/>`;
    svg += `<text x="${width - 14}" y="${y + 5}" text-anchor="end" fill="${r.family === 'jev' ? color : '#43536b'}" font-family="ui-monospace,monospace" font-size="14" font-weight="600">${esc(formatMetric(value))}</text></g>`;
  });
  svg += '</svg>';
  chartSvg = svg;
  $('main-chart').innerHTML = svg;
  const regimes = [...new Set(rows.map(r => training(r)))];
  $('chart-note').textContent = `${rows.some(r => r.ci?.[metric]) ? 'Whiskers: recorded 95% conditional intervals.' : 'No confidence intervals were computed for this metric.'} ${regimes.length > 1 ? 'These conditions use different numbers of training labels.' : 'Training-label budget: ' + regimes[0] + '.'} Click a row for its source record. Results are compared within each dataset.`;
}
function renderDataset() {
  const rows = filtered.filter(r => r.dataset === focus),
    r = rows[0];
  $('dataset-title').textContent = names[focus] || 'Dataset snapshot';
  $('dataset-badge').textContent = r ? dataType(r.data_type) : 'No selection';
  if (!r) {
    $('dataset-info').innerHTML = '';
    $('comparison-notes').innerHTML = '<p>No matching conditions. Reset filters to explore the study.</p>';
    return;
  }
  const budgets = [...new Set(rows.map(x => x.train_labels))].sort((a, b) => a - b);
  $('dataset-info').innerHTML = [['Task', r.task === 'binary' ? 'Binary classification' : 'Multiclass classification'], ['Classes', r.n_classes || r.labels?.length || 'N/A'], ['Held-out rows', count(r.n_test)], ['Training labels', budgets.join(', ')], ['Conditions shown', rows.length], ['Input', r.data_type === 'text' ? 'Text, capped at 2,000 characters' : 'Named numerical / categorical features']].map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
  $('comparison-notes').innerHTML = `<p><strong>Matched comparisons:</strong> few-shot, adapters and classical 4/class runs use the same selected labels at seed 42. Zero-shot uses no new labels.</p>
<p><strong>Uncertainty:</strong> ${r.data_type === 'text' ? 'stratified test-row bootstrap' : 'whole-feature-group bootstrap'}, conditional on the fixed split and fitted model. Overlapping bars alone do not tell us whether the methods differ.</p><p><strong>Coverage:</strong> ${focus === 'wine' ? 'Wine has only 36 test rows; a perfect score here does not establish perfect accuracy on new data.' : focus === 'titanic' ? 'Titanic3 includes mixed numeric and categorical features. Identical feature vectors stay together across splits.' : 'Missing model / method combinations mean “not measured”, never zero.'} ${rows.some(x => x.dev_labels > 0) ? 'Full text classical fits also use development labels.' : ''}</p>`;
}
function renderJev() {
  const rows = filtered.filter(r => r.family === 'jev' && r.canonical !== false);
  if (!rows.length) {
    $('jev-content').innerHTML = '<div class="empty"><strong>No Jev results in this selection</strong>Include Jev in the model filters and use seed 42 with zero-shot or four examples per class. Jev was measured on SST-2, TREC, Titanic, Breast Cancer and Wine.</div>';
    return;
  }
  let html = '<div class="notice">This view follows your filters. Changes below are observed accuracy differences, not paired significance tests. Four examples per class use additional labels compared with zero-shot. Jev adapter training was not available in this study.</div>';
  for (const ds of [...new Set(rows.map(r => r.dataset))]) {
    const zero = rows.find(r => r.dataset === ds && r.method === 'zero_shot'),
      few = rows.find(r => r.dataset === ds && r.method === 'few_shot'),
      base = zero || few;
    const delta = zero && few ? (few.accuracy - zero.accuracy) * 100 : null;
    const ci = r => r?.ci?.accuracy ? `95% CI ${r.ci.accuracy.map(percent).join(' to ')}` : r ? 'No interval' : 'Excluded by filters';
    html += `<div class="jev-row"><div><h3>${esc(names[ds] || ds)}</h3><p>${esc(dataType(base.data_type))} · ${base.n_classes} classes · ${count(base.n_test)} test rows</p><p>${few ? `${few.train_labels} few-shot labels` : 'Few-shot condition excluded'} · ${rows.filter(r => r.dataset === ds).reduce((s, r) => s + r.n_failures, 0)} retained failures</p></div><div class="jev-values"><div><span>Zero-shot</span><strong>${percent(zero?.accuracy)}</strong><p>${esc(ci(zero))}</p></div><div><span>4 / class</span><strong>${percent(few?.accuracy)}</strong><p>${esc(ci(few))}</p></div><div><span>Observed change</span><strong class="jev-delta ${delta < 0 ? 'negative' : ''}">${delta === null ? 'N/A' : (delta > 0 ? '+' : '') + delta.toFixed(1) + ' pp'}</strong><p>${delta === null ? 'Select both methods' : 'Unequal label budgets'}</p></div></div></div>`;
    const peers = filtered.filter(r => r.canonical !== false && r.dataset === ds && r.family !== 'jev' && r.train_per_class === 4 && r.seed === 42 && r.dev_labels === 0 && ['few_shot', 'classical'].includes(r.method));
    if (few && peers.length) {
      html += `<div class="peer-strip"><span>Matched labels · observed accuracy</span>${peers.map(r => `<button data-detail="${esc(r.id)}"><i style="background:${colors[r.family]}"></i>${esc(displayModel(r))} <b>${percent(r.accuracy)}</b></button>`).join('')}</div>`;
    }
  }
  html += '<div class="notice">Astra had higher few-shot accuracy than Jev on all five measured tasks. Jev had higher observed few-shot accuracy than Qwen3 4B on all five, but several paired differences remain uncertain. Classical models remain competitive on numerical data. The public datasets may have appeared in model training.</div>';
  $('jev-content').innerHTML = html;
  bindDetails($('jev-content'));
}
function renderTable() {
  const query = search.trim().toLowerCase();
  const rows = filtered.filter(r => !query || [displayModel(r), r.dataset, names[r.dataset], r.run_id, r.method, r.family].join(' ').toLowerCase().includes(query));
  const pageSize = 20;
  page = Math.max(0, Math.min(page, Math.max(0, Math.ceil(rows.length / pageSize) - 1)));
  const shown = rows.slice(page * pageSize, (page + 1) * pageSize);
  $('prev').disabled = page === 0;
  $('next').disabled = (page + 1) * pageSize >= rows.length;
  $('table-count').textContent = rows.length ? `${page * pageSize + 1} to ${Math.min((page + 1) * pageSize, rows.length)} of ${count(rows.length)} filtered measurements` : 'No matching measurements';
  if (!shown.length) {
    $('run-table').innerHTML = '<div class="empty"><strong>No matching measurements</strong>Try another filter or search phrase.</div>';
    return;
  }
  $('run-table').innerHTML = `<div class="table-wrap">
<table>
<thead>
<tr><th>Model / method</th><th>Dataset</th><th>Labels · seed</th><th class="num">Accuracy</th><th class="num">Macro-F1</th><th class="num">Balanced acc.</th><th class="num">Failures</th><th class="num">Probability coverage</th>
</tr>
</thead>
<tbody>${shown.map(r => `<tr><td><button class="row-model" data-detail="${esc(r.id)}"><span class="family-dot" style="--c:${colors[r.family]}"></span>${esc(displayModel(r))}</button><small>${esc(methodNames[r.method])}${r.canonical === false ? ' · replication' : ''}${r.repetition_count > 1 && r.canonical ? ' · ' + r.repetition_count + ' executions' : ''}</small></td><td>${esc(names[r.dataset] || r.dataset)}<small>${esc(dataType(r.data_type))}</small></td><td><span class="pill">${esc(training(r))}</span><small>${count(r.train_labels)} train + ${count(r.dev_labels)} dev · seed ${r.seed}</small></td><td class="num">${percent(r.accuracy)}</td><td class="num">${score(r.macro_f1)}</td><td class="num">${percent(r.balanced_accuracy)}</td><td class="num ${r.n_failures ? 'failure' : ''}">${r.n_failures} / ${r.n_test}</td><td class="num">${percent(r.probability_coverage)}</td></tr>`).join('')}</tbody></table></div>`;
  bindDetails($('run-table'));
}
function dollars(value, digits = 2) {
  return number(value) === null ? 'N/A' : '$' + Number(value).toFixed(digits);
}
function renderAudit() {
  const c = DATA.costs;
  if (!c) {
    $('audit-content').innerHTML = '<div class="empty">Cost reconciliation unavailable.</div>';
    return;
  }
  const summary = DATA.summary;
  const providerRows = (c.providers || []).map(p => {
    const apiKnown = p.reported_api_cost_coverage > 0;
    const api = apiKnown ? p.reported_api_cost_known_subtotal_usd : p.reported_api_cost_usd;
    const estimate = p.token_rate_estimate_known_subtotal_usd ?? p.token_rate_estimate_usd;
    const estAvailable = estimate !== undefined && estimate !== null && (p.token_rate_estimate_coverage === undefined || p.token_rate_estimate_coverage > 0);
    return `<tr><td>${esc(p.model)}<small>${esc(p.study)} study · ${count(p.requests ?? p.reservations)} calls</small></td><td class="num">${api === null || api === undefined ? 'N/A' : dollars(api, 6)}<small>${apiKnown ? percent(p.reported_api_cost_coverage) + ' cost coverage' : 'Not reported'}</small></td><td class="num">${estAvailable ? dollars(estimate, 6) : 'N/A'}<small>${estAvailable ? p.token_rate_estimate_coverage !== undefined ? percent(p.token_rate_estimate_coverage) + ' usage coverage' : 'Complete reported usage' : 'Not provided here'}</small></td><td class="num">${dollars(p.accounted_conservative_usd, 6)}</td></tr>`;
  }).join('');
  $('audit-content').innerHTML = `<div class="notice" style="margin:0 0 20px"><strong>Historical accounting snapshot.</strong> This section covers the original text and tabular studies, before the numerical/text review extensions, controls and retries. It does not change with the sidebar filters. These are earlier phase totals, not the final study totals. Local and Colab compute costs were not measured in dollars. <a href="https://github.com/statsguysam/jev-classification-benchmark/blob/main/results/completion_20260923/COSTS.md">Final study accounting ↗</a>
</div>
<div class="audit-grid">
<div class="panel audit-stat"><span>Conservative API accounting</span><strong>${dollars(c.conservative_accounted_usd)}</strong><small>Includes settlements and retained reserves</small>
</div>
<div class="panel audit-stat"><span>Authorization at this snapshot</span><strong>${dollars(c.authorized_usd)}</strong><small>Text and tabular studies combined</small></div><div class="panel audit-stat"><span>Remaining at this snapshot</span><strong>${dollars(c.remaining_capacity_usd)}</strong><small>Not a provider account balance</small>
</div>
</div>
<div class="panel">
<div class="panel-heading">
<div>
<div class="eyebrow">API ACCOUNTING</div>
<h2>Cost, coverage & reservations</h2>
</div><span class="badge">USD · recorded rates</span>
</div>
<div class="table-wrap">
<table>
<thead>
<tr><th>Provider / study</th><th class="num">Known API-reported subtotal</th><th class="num">Known token-rate estimate</th><th class="num">Conservative accounting</th>
</tr>
</thead>
<tbody>${providerRows}</tbody>
</table>
</div>
<div class="notice" style="margin-top:18px">Reported Jev dollars, OpenAI token-rate estimates and retained reservations are different accounting bases. Unknown costs stay unknown. These are not invoices; missing dollars are never treated as free calls.</div>
</div>
<div class="panel" style="margin-top:20px">
<div class="panel-heading">
<div>
<div class="eyebrow">STUDY DESIGN</div>
<h2>Understand the evidence</h2>
</div><span class="badge">${summary?.execution_records || 318} executions · ${summary?.canonical_configurations || 290} conditions</span>
</div>
<div class="methodology">
<h3>Shared rows and explicit label budgets</h3>
<p>Few-shot, LoRA/QLoRA and matched classical arms use the same ordered training examples within each dataset and selection seed. Zero-shot uses no new labels. Full classical models use additional training labels; the text full-training track also uses development labels for selection. The default comparison includes zero-shot as a separate reference alongside four labels per class.</p>
<h3>Text, numeric and mixed tabular data</h3>
<p>SST-2, TREC, IMDb, AG News and Banking77 contain text. Breast Cancer and Wine are numeric. Titanic contains both numeric and categorical features. LLMs see named feature=value serialization for tabular rows; classical models use native columns with preprocessing fit on training rows. Only SST-2 and TREC have text neural/hosted measurements.</p>
<h3>Uncertainty and repeated executions</h3>
<p>Text intervals use 1,000 stratified test-row bootstrap replicates. Tabular intervals use 2,000 whole-identical-feature-group replicates. They measure uncertainty for this test set with the split and models held fixed. They do not measure variation across prompts, training examples or future data. The 28 repeated classical executions are retained for reproducibility and hidden by default. They are not independent experiments.</p>
<h3>Probability quality and timing</h3>
<p>Jev returns native Choice probabilities. Qwen probabilities normalize restricted class-ID-plus-EOS likelihoods. Classical probabilities are uncalibrated; SVM variants provide none. OpenAI generated-label runs provide no class distribution. Probability metrics cover only valid available distributions. Brier scores sum across classes; ECE uses 15 bins. ROC-AUC and average precision apply only to binary tasks, with class ID 1 as positive. Timing combines different hardware and protocols and does not establish a speed ranking.</p>
<h3>Study limitations</h3>
<p>These public datasets may have appeared in model training. Wine has only 36 test rows and arbitrary cultivar IDs. A perfect score here does not establish perfect accuracy on new data. Failures remain in accuracy and macro-F1. The Breast Cancer task is a research benchmark, not clinical validation. No test-guided retuning was used.</p>
<h3>Source records</h3>
<p>The study repository is private; source links require permission. The dashboard contains aggregate measurements, not API credentials or original dataset rows.</p>
<p>${(c.sources || []).map(s => `<a href="${esc(s.url)}" target="_blank" rel="noreferrer">${esc(s.path)} ↗</a>`).join('<br>')}</p></div></div>`;
}
function bindDetails(element) {
  element.querySelectorAll('[data-detail]').forEach(button => button.onclick = () => openDetail(button.dataset.detail));
}
function openDetailBase(id) {
  const r = DATA.runs.find(x => x.id === id);
  if (!r) return;
  $('detail-content').innerHTML = `<h2>${esc(displayModel(r))}</h2><p>${esc(names[r.dataset] || r.dataset)} · ${esc(methodNames[r.method])} · ${esc(training(r))}</p><dl class="detail-grid">${[['Accuracy', percent(r.accuracy)], ['Macro-F1', score(r.macro_f1)], ['Balanced accuracy', percent(r.balanced_accuracy)], ['Test rows', count(r.n_test)], ['Training labels', count(r.train_labels)], ['Failures', r.n_failures], ['Probability coverage', percent(r.probability_coverage)], ['Seed', r.seed], ['Development labels', r.dev_labels ?? 0]].map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl><h3>Source record</h3><a class="source-link" target="_blank" rel="noreferrer" href="${esc(r.source_url || 'https://github.com/statsguysam/jev-classification-benchmark/blob/main/' + r.source_path)}">${esc(r.run_id || r.source_path)} ↗</a><p class="notice" style="margin:18px 0">Source records are in the study's private GitHub repository. Access requires repository permission.</p>`;
  $('detail').showModal();
}
function download(filename, text, type) {
  const holder = $('download-ready');
  if (download.previousURL) URL.revokeObjectURL(download.previousURL);
  const url = URL.createObjectURL(new Blob([text], {
    type
  }));
  download.previousURL = url;
  holder.hidden = false;
  holder.innerHTML = `Export ready: <a href="${url}" download="${esc(filename)}">${esc(filename)} ↓</a><span>If the automatic download does not start, use this link.</span>`;
  holder.querySelector('a').click();
}
function setTab(next) {
  tab = next;
  document.querySelectorAll('[data-tab]').forEach(b => {
    const active = b.dataset.tab === tab;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', String(active));
  });
  for (const t of ['compare', 'jev', 'runs', 'audit']) $(t + '-view').hidden = t !== tab;
  encodeState();
}
function applyFilters(next) {
  validateFilters(next);
  Object.assign(filters, next);
  for (const [k, v] of Object.entries(filters)) if ($(k)) {
    if (k === 'replicas') $(k).checked = v;else $(k).value = v;
  }
  updateChoices();
  page = 0;
  render();
}
async function boot() {
  try {
    const response = await fetch('data.json', {
      cache: 'no-cache'
    });
    if (!response.ok) throw Error('Data file unavailable');
    DATA = await response.json();
    if (!Array.isArray(DATA.runs) || !DATA.runs.length) throw Error('No measured records available');
    const query = new URLSearchParams(location.search);
    for (const key of Object.keys(defaults)) {
      const value = query.get(key);
      if (value !== null) {
        const proposed = key === 'replicas' ? value === 'true' : value;
        try {
          validateFilters({
            [key]: proposed
          });
          filters[key] = proposed;
        } catch {}
      }
    }
    if (names[query.get('focus')]) focus = query.get('focus');
    if (metricNames[query.get('metric')]) metric = query.get('metric');
    for (const key of Object.keys(defaults)) {
      $(key).addEventListener('change', () => applyFilters({
        [key]: key === 'replicas' ? $(key).checked : $(key).value
      }));
    }
    applyFilters(filters);
    setTab(['compare', 'jev', 'runs', 'audit'].includes(query.get('view')) ? query.get('view') : 'compare');
    $('toggle-filters').onclick = () => {
      const sidebar = document.querySelector('.sidebar');
      const open = sidebar.classList.toggle('filters-open');
      $('toggle-filters').setAttribute('aria-expanded', String(open));
      $('toggle-filters').textContent = open ? 'Hide filters' : 'Show filters';
    };
    $('reset').onclick = () => {
      focus = 'titanic';
      metric = 'macro_f1';
      search = '';
      $('search').value = '';
      applyFilters({
        ...defaults
      });
    };
    $('jev-only').onclick = () => applyFilters({
      family: 'jev',
      model: 'all'
    });
    $('all-models').onclick = () => applyFilters({
      family: 'all',
      model: 'all'
    });
    $('focus').onchange = () => {
      focus = $('focus').value;
      renderChart();
      renderDataset();
      encodeState();
    };
    $('metric').onchange = () => {
      metric = $('metric').value;
      renderChart();
      encodeState();
    };
    document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => setTab(b.dataset.tab));
    $('export').onclick = exportCSV;
    $('export-chart').onclick = () => {
      if (chartSvg) exportChart();
    };
    $('main-chart').addEventListener('click', e => {
      const item = e.target.closest('[data-id]');
      if (item) openDetail(item.dataset.id);
    });
    $('main-chart').addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        const item = e.target.closest('[data-id]');
        if (item) {
          e.preventDefault();
          openDetail(item.dataset.id);
        }
      }
    });
    $('close-detail').onclick = () => $('detail').close();
    $('detail').addEventListener('click', e => {
      if (e.target === $('detail')) $('detail').close();
    });
    $('search').oninput = () => {
      search = $('search').value;
      page = 0;
      renderTable();
    };
    $('prev').onclick = () => {
      page = Math.max(0, page - 1);
      renderTable();
    };
    $('next').onclick = () => {
      page++;
      renderTable();
    };
    window.addEventListener('resize', () => {
      clearTimeout(render.resizeTimer);
      render.resizeTimer = setTimeout(renderChart, 120);
    });
    registerAgentTools();
    window.benchmarkDashboard = {
      getState: () => ({
        filters: {
          ...filters
        },
        focus,
        metric,
        tab,
        count: filtered.length
      }),
      setFilters: applyFilters,
      getRows: () => filtered,
      openDetail
    };
  } catch (error) {
    $('status').textContent = 'The measured data could not be loaded. Reload the page or check the data file.';
    $('main-chart').innerHTML = `<div class="empty"><strong>Results unavailable</strong>${esc(error.message)}</div>`;
  }
}
function openDetail(id) {
  openDetailBase(id);
  const r = DATA.runs.find(x => x.id === id);
  if (!r) return;
  const content = $('detail-content');
  const ciRows = ['accuracy', 'macro_f1', 'balanced_accuracy'].map(m => `<tr><td>${esc(metricNames[m])}</td><td class="num">${formatMetric(getMetric(r, m), m)}</td><td class="num">${r.ci?.[m] ? r.ci[m].map(v => formatMetric(v, m)).join(' to ') : 'Not estimated'}</td></tr>`).join('');
  content.insertAdjacentHTML('beforeend', `<h3>Metrics & uncertainty</h3>
<div class="table-wrap">
<table>
<thead>
<tr><th>Metric</th><th class="num">Score</th><th class="num">95% interval</th>
</tr>
</thead>
<tbody>${ciRows}</tbody></table></div><p class="detail-note">${esc(r.ci_method || 'Interval method not recorded')} · ${count(r.ci_samples)} replicates${r.ci_n_groups ? ' · ' + r.ci_n_groups + ' feature groups' : ''}. These intervals hold the split and model fixed; they do not cover variation across prompts or training seeds.</p>
<h3>Probability quality</h3>
<p class="detail-note">${esc((r.probability_kinds || []).join(', ') || 'No class distribution supplied')} · ${percent(r.probability_coverage)} coverage</p><dl class="detail-grid">${[['Log loss', score(r.probability_metrics?.log_loss)], ['Brier sum', score(r.probability_metrics?.brier_sum)], ['ECE · 15 bins', score(r.probability_metrics?.ece_15_equal_width)], ['ROC-AUC', score(r.probability_metrics?.roc_auc)], ['Average precision', score(r.probability_metrics?.average_precision)], ['Valid probability rows', count(r.n_probability_rows)]].map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>
<p class="detail-note">Probability metrics use available valid distributions only. They do not include rows with missing distributions; failures remain in classification scores.</p>`);
  if (r.per_class?.length) {
    content.insertAdjacentHTML('beforeend', `<h3>Per-class performance</h3>
<div class="table-wrap detail-table">
<table>
<thead>
<tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th>
</tr>
</thead>
<tbody>${r.per_class.map(c => `<tr><td>${esc(c.name || r.labels?.[c.label] || c.label)}</td><td class="num">${score(c.precision)}</td><td class="num">${score(c.recall)}</td><td class="num">${score(c.f1)}</td><td class="num">${count(c.support)}</td></tr>`).join('')}</tbody></table></div>`);
  }
  if (r.confusion_matrix?.length) {
    const cols = r.confusion_matrix_columns || r.labels.map((_, i) => i);
    const matrix = `<p class="detail-note">Rows: actual class. Columns: predicted class. Recorded failures have their own column.</p>
<div class="table-wrap detail-table">
<table class="confusion">
<thead>
<tr><th>Actual ↓ / predicted →</th>${cols.map(c => `<th>${esc(c === 'failure' ? 'Failure' : r.labels?.[c] ?? c)}</th>`).join('')}</tr></thead><tbody>${r.confusion_matrix.map((row, i) => `<tr><th>${esc(r.labels?.[i] ?? i)}</th>${row.map((v, j) => `<td style="background:${v ? 'rgba(0,141,153,' + Math.min(.5, .06 + v / Math.max(...row) * .24) + ')' : 'white'}">${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    content.insertAdjacentHTML('beforeend', r.n_classes > 12 ? `<details><summary>Full confusion matrix · ${r.n_classes} classes</summary>${matrix}</details>` : `<h3>Confusion matrix</h3>${matrix}`);
  }
  const latency = r.latency || {};
  const devices = r.hardware?.inference_devices || [];
  content.insertAdjacentHTML('beforeend', `<h3>Execution context</h3><dl class="detail-grid">${[['p50 latency', latency.p50_s == null ? 'N/A' : Number(latency.p50_s).toFixed(3) + ' s'], ['p95 latency', latency.p95_s == null ? 'N/A' : Number(latency.p95_s).toFixed(3) + ' s'], ['Training time', r.training_time_s == null ? 'Not measured' : Number(r.training_time_s).toFixed(1) + ' s'], ['Inference device', devices.join(', ') || 'Not disclosed'], ['Execution status', r.canonical ? 'Canonical' : 'Replication'], ['Equivalent executions', r.repetition_count || 1]].map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl><p class="detail-note">${esc(latency.basis || 'Timing basis not recorded')}. Hardware and timing protocols differ across model families; these timings do not establish a speed ranking.</p>
<h3>Model provenance</h3>
<p class="detail-note">Requested model: ${esc(r.model)}<br>Returned model: ${esc((r.resolved_models || []).join(', ') || 'Not recorded')}<br>Resolved revision: ${esc((r.resolved_revisions || []).join(', ') || r.requested_revision || 'No immutable revision recorded')}<br>Inference source: ${esc(r.metric_source || 'Measured run record')}</p>`);
  const replicas = DATA.runs.filter(x => x.duplicate_group_id === r.duplicate_group_id && x.id !== r.id);
  if (replicas.length) content.insertAdjacentHTML('beforeend', `<h3>Equivalent executions</h3><p class="detail-note">Replicas are retained separately. Their timings and probability roundoff are not averaged.</p>${replicas.map(x => `<p><button class="row-model" data-detail="${esc(x.id)}">${esc(x.source_path)} ↗</button></p>`).join('')}`);
  bindDetails(content);
}
function exportCSV() {
  const query = search.trim().toLowerCase();
  const rows = tab === 'runs' && query ? filtered.filter(r => [displayModel(r), r.dataset, names[r.dataset], r.run_id, r.method, r.family].join(' ').toLowerCase().includes(query)) : filtered;
  const fields = ['dataset', 'data_type', 'task', 'n_classes', 'model', 'family', 'method', 'input_representation', 'train_per_class', 'train_labels', 'dev_labels', 'seed', 'canonical', 'accuracy', 'accuracy_ci95_low', 'accuracy_ci95_high', 'macro_f1', 'macro_f1_ci95_low', 'macro_f1_ci95_high', 'balanced_accuracy', 'n_test', 'n_failures', 'probability_coverage', 'log_loss', 'brier_sum', 'ece_15_equal_width', 'ci_method', 'ci_samples', 'latency_p50_s', 'latency_p95_s', 'latency_basis', 'source_path'];
  const cell = x => `"${String(x ?? '').replaceAll('"', '""')}"`;
  const mapped = rows.map(r => ({
    ...r,
    accuracy_ci95_low: r.ci?.accuracy?.[0],
    accuracy_ci95_high: r.ci?.accuracy?.[1],
    macro_f1_ci95_low: r.ci?.macro_f1?.[0],
    macro_f1_ci95_high: r.ci?.macro_f1?.[1],
    ...r.probability_metrics,
    latency_p50_s: r.latency?.p50_s,
    latency_p95_s: r.latency?.p95_s,
    latency_basis: r.latency?.basis
  }));
  download('jev-benchmark-filtered.csv', [fields.join(','), ...mapped.map(r => fields.map(k => cell(r[k])).join(','))].join('\r\n'), 'text/csv;charset=utf-8');
  toast(`${rows.length} filtered rows exported with uncertainty and provenance`);
}
function exportChart() {
  const parsed = new DOMParser().parseFromString(chartSvg, 'image/svg+xml').documentElement;
  const [,, width, height] = parsed.getAttribute('viewBox').split(' ').map(Number);
  const shown = filtered.filter(r => r.dataset === focus);
  const caption = `${names[focus]} · ${metricNames[metric]} · ${shown[0]?.n_test || 0} held-out rows · seed ${filters.seed}`;
  const output = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height + 130}" font-family="system-ui,sans-serif"><rect width="100%" height="100%" fill="white"/><text x="20" y="27" font-size="19" font-weight="700" fill="#14223b">Jev Benchmark Observatory</text><text x="20" y="50" font-size="13" fill="#627089">${esc(caption)}</text><g transform="translate(0 68)">${parsed.innerHTML}</g><text x="20" y="${height + 98}" font-size="11" fill="#627089">95% conditional intervals where available. Label budgets and methods are shown per row.</text><text x="20" y="${height + 118}" font-size="11" fill="#627089">Public datasets may have appeared in model training. September 2026 results for this setup.</text></svg>`;
  download(`jev-${focus}-${metric}.svg`, output, 'image/svg+xml');
  toast('Chart exported with dataset context and limitations');
}
function validateFilters(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw Error('Filters must be an object');
  const choices = {
    type: ['all', 'text', 'numeric', 'mixed'],
    task: ['all', 'binary', 'multiclass'],
    dataset: ['all', ...Object.keys(names)],
    family: ['all', ...Object.keys(familyNames)],
    model: ['all', ...new Set(DATA.runs.map(r => r.model))],
    method: ['all', ...Object.keys(methodNames)],
    budget: ['all', 'matched', '0', '1', '4', '8', 'full'],
    seed: ['all', '13', '42', '87'],
    coverage: ['all', 'probabilities', 'failures']
  };
  for (const [k, v] of Object.entries(input)) {
    if (k === 'replicas') {
      if (typeof v !== 'boolean') throw Error('replicas must be boolean');
    } else if (!choices[k] || !choices[k].includes(v)) throw Error('Unsupported filter ' + k);
  }
  return input;
}
function registerAgentTools() {
  const context = document.modelContext;
  if (!context?.registerTool) {
    document.documentElement.dataset.webmcp = 'unavailable';
    return;
  }
  const lifecycle = new AbortController();
  window.addEventListener('pagehide', () => lifecycle.abort(), {
    once: true
  });
  const definitions = [{
    name: 'read_benchmark_view',
    title: 'Read benchmark view',
    description: 'Read active dashboard filters, focus dataset and aggregate counts. Does not change the dashboard.',
    inputSchema: {
      type: 'object',
      properties: {},
      additionalProperties: false
    },
    annotations: {
      readOnlyHint: true
    },
    execute(input) {
      if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).length) throw Error('No input fields accepted');
      return {
        filters: {
          ...filters
        },
        focus,
        metric,
        view: tab,
        conditions: filtered.length,
        predictions: filtered.reduce((s, r) => s + r.n_test, 0),
        failures: filtered.reduce((s, r) => s + r.n_failures, 0)
      };
    }
  }, {
    name: 'configure_benchmark_filters',
    title: 'Configure benchmark filters',
    description: 'Apply supported dashboard filters to the visible comparison, summaries and run explorer. Does not run or change benchmark measurements.',
    inputSchema: {
      type: 'object',
      properties: {
        type: {
          type: 'string',
          enum: ['all', 'text', 'numeric', 'mixed']
        },
        task: {
          type: 'string',
          enum: ['all', 'binary', 'multiclass']
        },
        dataset: {
          type: 'string',
          enum: ['all', ...Object.keys(names)]
        },
        family: {
          type: 'string',
          enum: ['all', 'jev', 'open', 'hosted', 'classical']
        },
        model: {
          type: 'string',
          enum: ['all', ...new Set(DATA.runs.map(r => r.model))]
        },
        method: {
          type: 'string',
          enum: ['all', ...Object.keys(methodNames)]
        },
        budget: {
          type: 'string',
          enum: ['all', 'matched', '0', '1', '4', '8', 'full']
        },
        seed: {
          type: 'string',
          enum: ['all', '13', '42', '87']
        },
        coverage: {
          type: 'string',
          enum: ['all', 'probabilities', 'failures']
        },
        replicas: {
          type: 'boolean'
        }
      },
      additionalProperties: false
    },
    annotations: {
      readOnlyHint: false
    },
    execute(input) {
      validateFilters(input);
      applyFilters(input);
      return {
        filters: {
          ...filters
        },
        conditions: filtered.length,
        focus
      };
    }
  }];
  Promise.all(definitions.map(tool => Promise.resolve(context.registerTool(tool, {
    signal: lifecycle.signal
  })))).then(() => {
    document.documentElement.dataset.webmcp = 'registered';
  }).catch(() => {
    document.documentElement.dataset.webmcp = 'registration-unavailable';
  });
}
boot();
