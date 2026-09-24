const el = id => document.getElementById(id);
const escape = v => String(v).replace(/[&<>"']/g, c => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
})[c]);
const models = {
  'Qwen/Qwen2.5-0.5B-Instruct': 'Qwen2.5 0.5B',
  'Qwen/Qwen3-4B-Instruct-2507': 'Qwen3 4B',
  'HuggingFaceTB/SmolLM2-1.7B-Instruct': 'SmolLM2 1.7B',
  'ibm-granite/granite-3.3-2b-instruct': 'Granite 3.3 2B',
  'gpt-5.6-luna': 'GPT Luna',
  'gpt-6-astra': 'GPT Astra'
};
const datasets = {
  breast_cancer: 'Breast Cancer · binary',
  wine: 'Wine · multiclass',
  sst2: 'SST-2 · binary',
  trec: 'TREC · multiclass',
  titanic: 'Titanic · historical mixed tabular'
};
const armNames = {
  actual: 'Actual proposal',
  no_proposal: 'No proposal',
  shuffled: 'Shuffled proposal'
};
const name = c => `${datasets[c.dataset]} / ${models[c.source_model] || c.source_model} / ${c.shots_per_class ? '4 examples per class' : 'zero-shot'}`;
const table = (head, rows) => `<table><thead><tr>${head.map(s => `<th scope="col">${escape(s)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(s => `<td>${escape(s)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
let data, gates;
const fmt = (v, metric = el('metric').value) => v == null ? 'Unavailable' : metric === 'macro_f1' ? v.toFixed(3) : `${(v * 100).toFixed(1)}%`;
const count = v => v == null ? 'Pending' : String(v);
const delta = (v, metric) => v == null ? 'Pending' : metric === 'macro_f1' ? `${v > 0 ? '+' : ''}${v.toFixed(3)}` : `${v > 0 ? '+' : ''}${(100 * v).toFixed(1)} pp`;
const interval = (value, metric, paired = false) => value == null ? 'Unavailable' : value.map(v => paired ? delta(v, metric) : fmt(v, metric)).join(' to ');
function renderControls() {
  const c = data.controls,
    dataset = el('control-dataset').value,
    metric = el('metric').value === 'micro_accuracy' ? 'accuracy' : el('metric').value;
  const complete = c.runs.filter(r => r.status === 'complete').length,
    expected = c.expected_primary_arms ?? c.runs.length;
  const saved = c.saved_requests == null ? 'Request progress not reported' : `${c.saved_requests.toLocaleString()}/${(c.expected_requests ?? c.planned_primary_requests + c.planned_repeats).toLocaleString()} requests saved`;
  el('control-status').textContent = `${complete}/${expected} primary conditions complete · ${saved}. ${complete === 0 ? 'No controlled result is available.' : 'Scores appear only for complete conditions and comparisons.'}`;
  el('control-note').textContent = c.scope_note ?? '550 fixed cases across Breast Cancer, Wine, SST-2 and TREC. The same prompt changes only the proposal slot. Planned requests are not results.';
  const runs = c.runs.filter(r => r.dataset === dataset);
  el('control-arms').innerHTML = table(['Jev input', 'Saved / planned', 'Status', 'Selected metric', '95% interval', 'Failures'], runs.map(r => {
    const done = r.status === 'complete',
      m = done ? r.metrics : null,
      ci = done ? r.group_bootstrap?.metrics?.[metric]?.ci95 : null;
    return [armNames[r.arm] ?? r.arm, r.saved_requests == null ? 'Not reported' : `${r.saved_requests}/${r.expected_requests}`, done ? 'Complete' : 'Pending', done ? fmt(m?.[metric], metric) : 'Pending', done ? interval(ci, metric) : 'Pending', done ? count(m?.n_failures) : 'Pending'];
  }));
  const comparisons = c.comparisons?.filter(r => r.dataset === dataset) ?? ['no_proposal', 'shuffled'].map(b => ({
    a: 'actual',
    b,
    status: 'pending'
  }));
  el('control-contrasts').innerHTML = table(['Paired contrast: A − B', 'Status', 'Difference', '95% paired interval', 'Corrected / harmed'], comparisons.map(r => {
    const done = r.status === 'complete',
      estimate = metric === 'balanced_accuracy' ? r.balanced_accuracy_delta : r.paired_group_bootstrap?.metrics?.[metric]?.estimate;
    const ci = r.paired_group_bootstrap?.metrics?.[metric]?.ci95,
      t = r.transitions_B_to_A;
    return [`${armNames[r.a]} − ${armNames[r.b]}`, done ? 'Complete' : 'Pending', done ? delta(estimate, metric) : 'Pending', done ? interval(ci, metric, true) : 'Pending', done ? `${count(t?.wrong_to_correct)} / ${count(t?.correct_to_wrong)}` : 'Pending'];
  }));
  el('control-interval-note').textContent = c.interval_note ?? '95% paired intervals will appear only after both arms are complete. Balanced-accuracy contrasts have no interval. This follow-up is exploratory on previously examined holdouts.';
  const r = c.serving_repeat_diagnostic;
  if (!r || r.status !== 'complete') {
    el('repeat-status').textContent = `${r?.available_complete_pairs == null ? 'Not reported' : r.available_complete_pairs}/${r?.expected_pairs ?? c.planned_repeats} repeat pairs available · Pending. Agreement and failure counts are withheld until all declared pairs are present.`;
    el('repeat-summary').textContent = 'Agreement: Pending';
    el('repeat-table').innerHTML = '';
  } else {
    const n = r.counts;
    el('repeat-status').textContent = `${r.available_complete_pairs}/${r.expected_pairs} identical-prompt pairs, selected before execution, are complete. Pairs with two failed responses do not count as label agreement.`;
    el('repeat-summary').textContent = `${fmt(r.valid_pair_agreement, 'accuracy')} agreement among ${n.both_valid} pairs with two valid responses (${n.valid_label_agreement} agree; ${n.valid_label_disagreement} disagree).`;
    el('repeat-table').innerHTML = table(['Dataset', 'Both valid', 'Agree / disagree', 'Reference-only failure', 'Repeat-only failure', 'Both failed'], Object.entries(r.by_dataset ?? {}).map(([d, v]) => [datasets[d] ?? d, count(v.both_valid), `${count(v.valid_label_agreement)} / ${count(v.valid_label_disagreement)}`, count(v.reference_failed_only), count(v.repeat_failed_only), count(v.both_failed)]));
  }
  renderControlRecovery();
}
function renderControlRecovery() {
  const r = data.control_recovery;
  if (!r || r.status !== 'complete') {
    el('control-recovery-status').textContent = 'Post-control recovery pending. The primary results above remain unchanged.';
    el('control-recovery-arms').innerHTML = '';
    el('control-recovery-contrasts').innerHTML = '';
    return;
  }
  const dataset = el('control-dataset').value,
    metric = el('metric').value === 'micro_accuracy' ? 'accuracy' : el('metric').value;
  const c = r.recovery_counts;
  el('control-recovery-status').textContent = r.no_op ? 'Every original control and repeat response was valid, so no additional calls were needed and scores are unchanged.' : `${r.new_calls} additional calls. Primary failures: ${c.primary.original_failures} → ${c.primary.remaining_failures}; repeat failures: ${c.repeat.original_failures} → ${c.repeat.remaining_failures}. This secondary view does not replace the original serving-repeat agreement.`;
  el('control-recovery-arms').innerHTML = table(['Selected dataset · Jev input', 'Full rows', 'First-attempt metric', 'Recovered metric', 'First / remaining failures'], r.runs.filter(v => v.dataset === dataset).map(v => [armNames[v.arm], v.n_rows, fmt(v.first_attempt.metrics[metric], metric), fmt(v.recovered.metrics[metric], metric), `${v.first_attempt.metrics.n_failures} / ${v.recovered.metrics.n_failures}`]));
  const display = view => {
    const ci = view.paired_group_bootstrap?.metrics?.[metric],
      point = metric === 'balanced_accuracy' ? view.balanced_accuracy_delta : ci?.estimate;
    return `${delta(point, metric)} · ${interval(ci?.ci95, metric, true)}`;
  };
  el('control-recovery-contrasts').innerHTML = table(['A − B', 'First attempt · 95% CI', 'Recovered sensitivity · 95% CI'], r.comparisons.filter(v => v.dataset === dataset).map(v => [`${armNames[v.a]} − ${armNames[v.b]}`, display(v.first_attempt), display(v.recovered)]));
  el('control-recovery-note').textContent = `${r.note ?? ''} ${r.interval_note ?? ''}`;
}
function renderRecovery() {
  const r = data.recovery;
  if (!r || r.status === 'not_available') {
    el('recovery-status').textContent = 'Recovery report pending. The comparisons above retain the original first-attempt failures.';
    el('recovery-table').innerHTML = '';
    return;
  }
  const metric = el('metric').value === 'micro_accuracy' ? 'accuracy' : el('metric').value;
  const api = r.api_outcomes;
  el('recovery-status').textContent = (r.note ?? `Recovery status: ${r.status}. Original evidence is retained separately.`) + (api ? ` ${api.new_calls} new calls: ${api.ordinary_paid_failure_retry_calls} retries of paid failures and ${api.dependent_review_calls} dependent-review calls, including ${api.dependent_first_calls} first calls.` : '');
  el('recovery-table').innerHTML = table(['Condition · full rows', 'Status', 'Original snapshot', 'First actual call', 'Recovered', 'Recovered − first · 95% CI', 'Failures: original / first / recovered', 'Corrected / harmed'], r.conditions.map(c => {
    const done = c.status === 'complete',
      score = m => m == null ? 'Pending' : fmt(m[metric], metric);
    const ci = c.paired_group_bootstrap?.metrics?.[metric],
      point = metric === 'balanced_accuracy' ? c.balanced_accuracy_delta : ci?.estimate;
    const model = c.source_model ? `${models[c.source_model] ?? c.source_model} → Jev` : models[c.model] ?? c.model;
    return [`${datasets[c.dataset] ?? c.dataset} / ${model} / ${c.shots_per_class ?? 'N/A'} examples per class / n=${c.n_rows ?? 'N/A'}`, done ? c.same_request_as_original_snapshot === false ? 'Complete · source recovered before review' : 'Complete' : 'Incomplete results', score(c.original_snapshot), score(c.first_attempt), done ? score(c.recovered) : 'Pending', done ? `${delta(point, metric)} · ${interval(ci?.ci95, metric, true)}` : 'Pending', `${count(c.original_snapshot?.n_failures)} / ${count(c.first_attempt?.n_failures)} / ${done ? count(c.recovered?.n_failures) : 'Pending'}`, done ? `${count(c.corrected)} / ${count(c.harmed)}` : 'Pending'];
  }));
  el('recovery-interval-note').textContent = r.interval_note ?? 'Paired intervals require complete first-call and recovered views. Balanced-accuracy changes have no interval.';
}
function renderComparison() {
  const c = data.conditions[Number(el('condition').value)],
    metric = el('metric').value;
  for (const [id, key] of [['source', 'never_review'], ['review', 'always_review'], ['direct', 'direct_jev']]) el(`${id}-score`).textContent = fmt(c.metrics[key][metric]);
  el('source-calls').textContent = `${c.n_rows} source inferences`;
  el('review-calls').textContent = `${c.n_rows} source + ${c.required_inferences.always_review.review_api_requests} review requests`;
  el('direct-calls').textContent = `${c.n_rows} direct Jev requests`;
  el('transitions').textContent = `${c.review_minus_base.wrong_to_correct} / ${c.review_minus_base.correct_to_wrong}`;
  el('comparison-title').textContent = name(c);
  el('comparison-note').textContent = `${c.n_rows} held-out rows. Corrections and harms count decisions, independent of the selected metric.`;
  el('comparison').innerHTML = table(['Comparison', 'Corrected', 'Harmed', 'Net correct change'], [['Review vs LLM alone', c.review_minus_base.wrong_to_correct, c.review_minus_base.correct_to_wrong, c.review_minus_base.net_correct_change], ['Review vs Jev alone', c.review_minus_direct.wrong_to_correct, c.review_minus_direct.correct_to_wrong, c.review_minus_direct.net_correct_change]]);
}
function renderGate() {
  const c = gates[Number(el('gate').value)],
    metric = el('metric').value,
    points = c.curve;
  const x = p => 72 + 650 * p.actual_review_fraction,
    y = v => 275 - 225 * v;
  const poly = key => points.map(p => `${x(p)},${y(key === 'metrics' ? p.metrics[metric] : p.random_matched_rate[metric])}`).join(' ');
  const grid = [0, .25, .5, .75, 1].map(v => `<line x1="72" x2="722" y1="${y(v)}" y2="${y(v)}" stroke="#e2e7f0"/><text x="60" y="${y(v) + 5}" text-anchor="end" fill="#627089" font-size="14">${metric === 'macro_f1' ? v.toFixed(2) : Math.round(v * 100) + '%'}</text>`).join('');
  const ticks = [0, .25, .5, .75, 1].map(v => `<text x="${72 + 650 * v}" y="300" text-anchor="middle" fill="#627089" font-size="14">${Math.round(v * 100)}%</text>`).join('');
  el('curve').innerHTML = `<svg viewBox="0 0 770 340" role="img" aria-label="${escape(name(c))}: selected metric by fraction of rows reviewed. Exact values in following table.">${grid}${ticks}<text x="400" y="330" text-anchor="middle" fill="#627089" font-size="14">Fraction of rows reviewed</text><line x1="72" x2="722" y1="${y(c.metrics.direct_jev[metric])}" y2="${y(c.metrics.direct_jev[metric])}" stroke="#c17b36" stroke-width="2" stroke-dasharray="4 5"/><polyline points="${poly('random')}" fill="none" stroke="#8570c4" stroke-width="2" stroke-dasharray="7 5"/><polyline points="${poly('metrics')}" fill="none" stroke="#008d99" stroke-width="3"/>${points.map(p => `<circle cx="${x(p)}" cy="${y(p.metrics[metric])}" r="4" fill="#008d99"/>`).join('')}</svg>`;
  el('gate-table').innerHTML = table(['Reviewed', 'Least-confident first', 'Random expectation', 'Corrected / harmed', 'Source + review calls'], points.map(p => [`${p.selected_rows}/${c.n_rows}`, fmt(p.metrics[metric]), fmt(p.random_matched_rate[metric]), `${p.transitions_from_never_review.wrong_to_correct} / ${p.transitions_from_never_review.correct_to_wrong}`, `${c.n_rows} + ${p.required_inferences.review_api_requests}`]));
  el('gate-context').textContent = `${name(c)}. Direct Jev: ${fmt(c.metrics.direct_jev[metric])}, using ${c.n_rows} requests and no source inference.`;
}
async function init() {
  try {
    const response = await fetch('review-data.json');
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    data = await response.json();
    gates = data.conditions.filter(c => c.curve);
    if (!data.conditions.length || !gates.length) throw Error('No completed conditions available');
    el('condition').innerHTML = data.conditions.map((c, i) => `<option value="${i}">${escape(name(c))}</option>`).join('');
    el('gate').innerHTML = gates.map((c, i) => `<option value="${i}">${escape(name(c))}</option>`).join('');
    el('condition').value = String(Math.max(0, data.conditions.findIndex(c => c.dataset === 'breast_cancer' && c.source_model.includes('Qwen3') && c.shots_per_class === 4)));
    el('gate').value = String(Math.max(0, gates.findIndex(c => c.dataset === 'breast_cancer' && c.source_model.includes('Qwen3') && c.shots_per_class === 4)));
    el('status').textContent = `${data.complete_review_conditions}/${data.expected_review_conditions} numerical review conditions complete · ${gates.length} eligible selective-review conditions · exploratory, one split`;
    const v = data.identical_prompts;
    el('prompt-variability').textContent = `${v.valid_output_disagreements} different final labels among ${v.both_valid_rows.toLocaleString()} same-proposal comparisons with identical prompt hashes and two valid review responses. Another ${v.rows_with_either_review_failure} comparisons included a failure.`;
    el('gate-description').textContent = `All ${gates.length} eligible conditions are available below, including those where this strategy underperformed random selection.`;
    el('control-dataset').innerHTML = [...new Set(data.controls.runs.map(r => r.dataset))].map(d => `<option value="${escape(d)}">${escape(datasets[d] ?? d)}</option>`).join('');
    el('condition').addEventListener('change', renderComparison);
    el('gate').addEventListener('change', renderGate);
    el('control-dataset').addEventListener('change', renderControls);
    el('metric').addEventListener('change', () => {
      renderComparison();
      renderGate();
      renderControls();
      renderRecovery();
    });
    renderComparison();
    renderGate();
    renderControls();
    renderRecovery();
  } catch (error) {
    el('status').textContent = `Evidence unavailable: ${error.message}. No scores have been substituted.`;
  }
}
init();
