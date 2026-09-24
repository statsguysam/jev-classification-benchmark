#!/usr/bin/env node
'use strict';

// No browser, network, inference, or writes. Copy into SITE/tests/, then run:
// node tests/text-dashboard.smoke.cjs
// While this file is staged elsewhere, pass the site root as the first argument.
// --synthetic-only exercises complete and 36/68 or 44/68 partial UI fixtures without a
// text-data.json. These invented scores are UI tests, never measurements.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {
  loadDecisionPageScripts,
  bootDecisionPage
} = require('./decision-page-scripts.cjs');
const root = path.resolve(process.argv.slice(2).find(arg => !arg.startsWith('--')) || path.join(__dirname, '..'));
const dist = path.join(root, 'dist');
const html = fs.readFileSync(path.join(dist, 'text.html'), 'utf8');
const pageScripts = loadDecisionPageScripts(dist, html, 'text.js');
const syntheticOnly = process.argv.includes('--synthetic-only');
const asset = syntheticOnly ? null : JSON.parse(fs.readFileSync(path.join(dist, 'text-data.json'), 'utf8'));
const originalAsset = JSON.stringify(asset);
const metrics = ['accuracy', 'macro_f1'];
const metricNames = {
  accuracy: 'Accuracy',
  macro_f1: 'Macro-F1'
};
const modelIds = ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen3-4B-Instruct-2507', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'ibm-granite/granite-3.3-2b-instruct', 'gpt-5.6-luna', 'gpt-6-astra'];
const sourceModels = modelIds.map(id => ({
  id
}));
const pct = v => v == null ? 'N/A' : `${(100 * v).toFixed(1)}%`;
const signed = v => v == null ? 'N/A' : `${v >= 0 ? '+' : ''}${(100 * v).toFixed(1)}`;
const ci = v => v == null ? 'N/A' : `[${v.map(signed).join(', ')}]`;
const close = (actual, expected, message) => assert.ok(Math.abs(actual - expected) <= 1e-11, `${message}: ${actual} != ${expected}`);
const decode = value => value.replace(/&(?:amp|lt|gt|quot|#39);/g, c => ({
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'"
})[c]);
const plain = value => decode(value.replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();
function tableRows(value) {
  const body = value.match(/<tbody>([\s\S]*?)<\/tbody>/);
  assert.ok(body, 'Expected a rendered table body');
  return [...body[1].matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(r => [...r[1].matchAll(/<td>([\s\S]*?)<\/td>/g)].map(c => plain(c[1])));
}
function csvRows(value) {
  const rows = [],
    row = [];
  let field = '',
    quoted = false;
  for (let i = 0; i < value.length; i++) {
    const c = value[i];
    if (c === '"') {
      if (quoted && value[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (!quoted && (c === ',' || c === '\n')) {
      row.push(field);
      field = '';
      if (c === '\n') {
        rows.push(row.splice(0));
      }
    } else if (!quoted && c === '\r') {/* CRLF separator */} else field += c;
  }
  assert.equal(quoted, false, 'Unclosed CSV field');
  row.push(field);
  rows.push(row);
  const head = rows.shift();
  assert.equal(new Set(head).size, head.length, 'Duplicate CSV column');
  return rows.map(r => {
    assert.equal(r.length, head.length, 'CSV width');
    return Object.fromEntries(head.map((h, i) => [h, r[i]]));
  });
}
class Element {
  constructor(id) {
    this.id = id;
    this.value = '';
    this.textContent = '';
    this.options = [];
    this.listeners = new Map();
    this.disabled = false;
    this._html = '';
  }
  set innerHTML(value) {
    this._html = value;
    this.options = [...value.matchAll(/<option\b[^>]*value="([^"]*)"[^>]*>/g)].map(m => ({
      value: decode(m[1])
    }));
  }
  get innerHTML() {
    return this._html;
  }
  addEventListener(name, handler) {
    this.listeners.set(name, handler);
  }
}
async function harness(data, query = '') {
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element(m[1])]));
  for (const match of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) elements.get(match[1]).innerHTML = match[2];
  const legend = new Element('source-dot');
  const capture = {
    blob: null,
    anchor: null,
    fetches: 0,
    history: ''
  };
  const context = vm.createContext({
    URLSearchParams,
    Blob,
    console,
    location: {
      search: query
    },
    history: {
      replaceState(_a, _b, url) {
        capture.history = url;
      }
    },
    document: {
      getElementById(id) {
        assert.ok(elements.has(id), `Unknown DOM id ${id}`);
        return elements.get(id);
      },
      querySelector(selector) {
        assert.equal(selector, '.source-dot');
        return legend;
      },
      createElement(tag) {
        assert.equal(tag, 'a');
        const anchor = {
          click() {
            capture.anchor = anchor;
          }
        };
        return anchor;
      }
    },
    URL: {
      createObjectURL(blob) {
        capture.blob = blob;
        return 'blob:smoke-fixture';
      },
      revokeObjectURL(url) {
        assert.equal(url, 'blob:smoke-fixture');
      }
    },
    setTimeout(callback) {
      callback();
    },
    async fetch(url) {
      assert.equal(url, 'text-data.json');
      capture.fetches++;
      return {
        ok: true,
        async json() {
          return data;
        }
      };
    }
  });
  const dashboard = await bootDecisionPage(pageScripts, context);
  assert.equal(capture.fetches, 1, 'Only one local data fetch is allowed');
  assert.notEqual(elements.get('export').disabled, true, `Boot failed: ${elements.get('status').textContent}`);
  return {
    api: dashboard,
    elements,
    legend,
    capture
  };
}
function expectedPairs(data, state) {
  const matches = data.runs.filter(r => r.dataset === state.dataset && r.train_per_class === Number(state.shots));
  return matches.filter(r => r.arm === 'base' && (state.model === 'all' || r.model === state.model)).map(base => {
    const review = matches.find(r => r.arm === 'review' && r.source_run_id === base.run_id);
    const direct = matches.find(r => r.arm === 'direct');
    assert.ok(review && direct, 'Every source requires a review and direct control');
    const reference = state.view === 'pairs' ? base : direct;
    const kind = state.view === 'pairs' ? 'review_minus_source' : 'review_minus_jev_alone';
    const comparisons = data.comparisons.filter(c => c.kind === kind && c.a === review.run_id && c.b === reference.run_id);
    assert.equal(comparisons.length, 1, 'Exactly one paired contrast for each plotted row');
    return {
      base,
      review,
      direct,
      reference,
      comparison: comparisons[0]
    };
  });
}
async function exported(h) {
  h.capture.blob = null;
  h.capture.anchor = null;
  h.api.exportCSV();
  assert.ok(h.capture.anchor && h.capture.blob, 'Export must create and activate a CSV download');
  assert.equal(h.capture.blob.type, 'text/csv;charset=utf-8');
  return csvRows(await h.capture.blob.text());
}
async function verifyView(h, data, state) {
  Object.assign(h.api.state, state);
  h.api.render();
  const expected = expectedPairs(data, state),
    m = state.metric;
  assert.equal(expected.length, state.model === 'all' ? 6 : 1);
  const actual = tableRows(h.elements.get('pair-table').innerHTML);
  assert.equal(actual.length, expected.length);
  assert.match(h.elements.get('pair-table').innerHTML, /Pipeline failures/);
  assert.match(h.elements.get('pair-table').innerHTML, /A source failure counts as a pipeline failure, and no Jev call is made for that row/);
  const chart = h.elements.get('chart').innerHTML;
  const circles = [...chart.matchAll(/<circle cx="([^"]+)" cy="([^"]+)"/g)];
  const diamonds = [...chart.matchAll(/<path d="M([^,]+),([^ ]+) /g)];
  assert.equal(circles.length, expected.filter(p => p.reference[m] != null).length);
  assert.equal(diamonds.length, expected.filter(p => p.review[m] != null).length);
  const csv = await exported(h);
  assert.equal(csv.length, expected.length);
  assert.equal(h.capture.anchor.download, `jev-text-${state.dataset}-${state.shots}-${m}.csv`);
  let referenceMarker = 0,
    reviewMarker = 0;
  for (const [i, item] of expected.entries()) {
    const {
        base,
        review,
        reference,
        comparison: c
      } = item,
      t = review.review_transitions;
    assert.equal(c.dataset, state.dataset);
    assert.equal(c.equal_new_label_budget, true);
    assert.equal(c.train_per_class_a, Number(state.shots));
    assert.equal(c.train_per_class_b, Number(state.shots));
    if (review[m] != null && reference[m] != null) close(c[`${m}_delta`], review[m] - reference[m], 'Paired A−B difference');else {
      assert.equal(c[`${m}_delta`], null);
      assert.equal(c[`${m}_delta_ci95`], null);
    }
    assert.deepEqual(actual[i], [base.display_model, pct(reference[m]), pct(review[m]), `${signed(c[`${m}_delta`])} ${ci(c[`${m}_delta_ci95`])}`, t ? `${t.wrong_to_correct} / ${t.correct_to_wrong}` : 'N/A', review.n_failures == null ? 'N/A' : `${review.n_failures} / ${review.n_test}`, review[m] == null ? 'Pending' : 'Complete']);
    if (reference[m] != null) {
      close(Number(circles[referenceMarker][1]), 205 + 480 * reference[m], 'Reference marker');
      close(Number(circles[referenceMarker++][2]), 52 + i * 52, 'Chart row');
    }
    if (review[m] != null) {
      close(Number(diamonds[reviewMarker][1]), 205 + 480 * review[m], 'Review marker');
      close(Number(diamonds[reviewMarker++][2]), 46 + i * 52, 'Diamond row');
    }
    if (review[m] == null || reference[m] == null) assert.ok(chart.includes(`${pct(reference[m])} → ${pct(review[m])} (unavailable)`));
    assert.ok(chart.includes(`${pct(reference[m])} → ${pct(review[m])}`));
    assert.deepEqual(csv[i], Object.fromEntries(Object.entries({
      dataset: state.dataset,
      metric: m,
      shots_per_class: state.shots,
      source_model: base.model,
      comparison: c.kind,
      review_status: review[m] == null ? 'pending' : 'complete',
      reference_score: reference[m],
      review_score: review[m],
      delta_pp: c[`${m}_delta`] == null ? null : 100 * c[`${m}_delta`],
      delta_ci95_low_pp: c[`${m}_delta_ci95`]?.[0] == null ? null : 100 * c[`${m}_delta_ci95`][0],
      delta_ci95_high_pp: c[`${m}_delta_ci95`]?.[1] == null ? null : 100 * c[`${m}_delta_ci95`][1],
      corrected_from_source: t?.wrong_to_correct,
      harmed_from_source: t?.correct_to_wrong,
      pipeline_failures: review.n_failures,
      source_failure_rows: t?.source_failure_rows,
      review_stage_failure_rows: t?.review_stage_failure_rows,
      test_rows: review.n_test
    }).map(([k, v]) => [k, String(v ?? '')])));
    if (t) assert.equal(review.n_failures, t.source_failure_rows + t.review_stage_failure_rows);
  }
  const ds = data.datasets.find(d => d.id === state.dataset),
    direct = expected[0].direct;
  assert.equal(String(h.elements.get('rows').textContent), String(ds.n_test));
  assert.equal(String(h.elements.get('labels').textContent), String(Number(state.shots) * ds.n_classes));
  assert.equal(h.elements.get('direct-score').textContent, pct(direct[m]));
  assert.match(chart, new RegExp(`x1="${205 + 480 * direct[m]}" x2="${205 + 480 * direct[m]}"[^>]*stroke-dasharray="4 4"`));
  const ups = expected.filter(p => p.comparison[`${m}_delta`] > 0).length;
  const downs = expected.filter(p => p.comparison[`${m}_delta`] < 0).length;
  assert.equal(h.elements.get('changes').textContent, expected.some(p => p.comparison[`${m}_delta`] != null) ? `${ups} ↑ / ${downs} ↓` : 'Pending');
  assert.equal(h.elements.get('changes-note').textContent, `${expected.filter(p => p.comparison[`${m}_delta`] != null).length}/${expected.length} comparisons complete; ties omitted`);
  assert.ok(h.elements.get('status').textContent.startsWith(`${data.completion.complete_runs}/68 conditions complete`));
  assert.equal(h.elements.get('status').textContent.includes('awaits API credits'), ['blocked_no_credit', 'halted', 'blocked_billing', 'billing_paused'].includes(data.costs.status));
  assert.ok(h.elements.get('chart-context').textContent.includes(ds.feature_description));
  assert.ok(!h.elements.get('chart-context').textContent.includes('numeric features'));
  assert.equal(h.legend.textContent, state.view === 'pairs' ? '● LLM alone' : '● Jev alone');
  assert.equal(h.elements.get('chart-title').textContent, state.view === 'pairs' ? 'Before and after Jev' : 'Does the source proposal add value?');
  const native = data.runs.filter(r => r.dataset === state.dataset && r.arm === 'classical' && (state.nativeBudget === 'full' ? r.label_budget === 'full_training' : r.train_per_class === 4));
  assert.equal(native.length, 4);
  const fewDirect = data.runs.find(r => r.dataset === state.dataset && r.arm === 'direct' && r.train_per_class === 4);
  const nativeRows = [fewDirect, ...native];
  const nativeActual = tableRows(h.elements.get('native-table').innerHTML);
  assert.deepEqual(nativeActual, nativeRows.map(r => [r.display_model, String(r.train_labels), `${pct(r[m])} [${r[`${m}_ci95`].map(pct).join(', ')}]`, `${r.n_failures} / ${r.n_test}`]));
  assert.match(h.elements.get('native-table').innerHTML, new RegExp(metricNames[m]));
  const nativeContext = h.elements.get('native-context').textContent;
  assert.ok(nativeContext.includes(`Jev below always has ${4 * ds.n_classes} labels.`));
  assert.ok(nativeContext.includes(`Classical models have ${state.nativeBudget === 'full' ? ds.full_training_labels : 4 * ds.n_classes} labels.`));
  assert.equal(nativeContext.includes('different label budget from the zero-shot chart'), state.shots === '0');
  assert.equal(nativeContext.includes('Full training uses more labels and is reported separately'), state.nativeBudget === 'full');
  const url = new URLSearchParams(h.capture.history.slice(1));
  for (const [key, value] of Object.entries(state)) assert.equal(url.get(key), value, `URL state ${key}`);
}
async function nullChecks(fixture) {
  const baseState = {
    dataset: 'sst2',
    shots: '4',
    model: sourceModels[0].id,
    metric: 'accuracy',
    view: 'pairs',
    nativeBudget: '4'
  };
  for (const which of ['review', 'source', 'direct']) {
    const data = structuredClone(fixture),
      state = {
        ...baseState,
        view: which === 'direct' ? 'direct' : 'pairs'
      };
    const item = expectedPairs(data, state)[0];
    (which === 'review' ? item.review : which === 'source' ? item.base : item.direct).accuracy = null;
    item.comparison.accuracy_delta = null;
    item.comparison.accuracy_delta_ci95 = null;
    const h = await harness(data);
    Object.assign(h.api.state, state);
    h.api.render();
    const graphic = h.elements.get('chart').innerHTML;
    assert.ok(graphic.includes('(unavailable)'));
    assert.equal((graphic.match(/<circle /g) || []).length, which === 'review' ? 1 : 0, 'Plot a measured reference independently; never replace null with zero');
    assert.equal((graphic.match(/<path /g) || []).length, which === 'review' ? 0 : 1, 'Plot a measured review independently; never replace null with zero');
    if (which === 'direct') assert.ok(!graphic.includes('stroke-dasharray'));
    const row = (await exported(h))[0];
    assert.equal(row.delta_pp, '');
    assert.equal(row.delta_ci95_low_pp, '');
    assert.equal(row.delta_ci95_high_pp, '');
    assert.equal(which === 'review' ? row.review_score : row.reference_score, '');
    assert.equal(h.elements.get('changes').textContent, 'Pending');
  }
  const data = structuredClone(fixture),
    item = expectedPairs(data, baseState)[0];
  item.base.accuracy = 0;
  item.review.accuracy = 0;
  item.comparison.accuracy_delta = 0;
  item.comparison.accuracy_delta_ci95 = [null, 0.2];
  const h = await harness(data);
  Object.assign(h.api.state, baseState);
  h.api.render();
  let row = (await exported(h))[0];
  assert.equal(row.reference_score, '0');
  assert.equal(row.review_score, '0');
  assert.equal(row.delta_pp, '0');
  assert.equal(row.delta_ci95_low_pp, '');
  assert.equal(row.delta_ci95_high_pp, '20');
  assert.match(h.elements.get('chart').innerHTML, /<circle cx="205"/);
  // Percentile endpoints must remain as supplied even when they exclude the estimate.
  item.comparison.accuracy_delta = 0.1;
  item.comparison.accuracy_delta_ci95 = [0.2, 0.3];
  h.api.render();
  row = (await exported(h))[0];
  assert.equal(row.delta_pp, '10');
  assert.equal(row.delta_ci95_low_pp, '20');
  assert.equal(row.delta_ci95_high_pp, '30');
  assert.ok(plain(h.elements.get('pair-table').innerHTML).includes('+10.0 [+20.0, +30.0]'));
}
function validateAsset(data) {
  assert.equal(data.schema_version, 1);
  assert.ok(['complete', 'in_progress_or_incomplete'].includes(data.completion.status));
  assert.equal(data.completion.expected_runs, 68);
  assert.equal(data.completion.expected_comparisons, 72);
  assert.equal(data.runs.length, 68);
  assert.equal(data.comparisons.length, 72);
  assert.deepEqual(data.models.filter(m => ['open_weight_llm', 'hosted_llm'].includes(m.family)).map(m => m.id).sort(), [...modelIds].sort());
  assert.equal(data.datasets.length, 2);
  const byId = new Map(data.runs.map(r => [r.run_id, r]));
  assert.equal(byId.size, 68);
  assert.equal(new Set(data.comparisons.map(c => JSON.stringify([c.kind, c.a, c.b]))).size, 72);
  assert.equal(data.completion.complete_runs, data.runs.filter(r => r.status === 'complete').length);
  assert.equal(data.completion.complete_comparisons, data.comparisons.filter(c => c.status === 'complete').length);
  if (data.completion.status === 'complete') {
    assert.equal(data.completion.complete_runs, 68);
    assert.equal(data.completion.complete_comparisons, 72);
    assert.equal(data.costs.status, 'complete');
  } else {
    assert.ok(data.completion.complete_runs < 68);
    assert.ok(data.completion.complete_comparisons < 72);
    assert.notEqual(data.costs.status, 'complete');
  }
  for (const r of data.runs) {
    assert.equal(typeof r.status, 'string');
    if (r.arm === 'review') {
      const base = byId.get(r.source_run_id);
      assert.ok(base, 'Pending and complete reviews retain their source link');
      assert.equal(base.arm, 'base');
      assert.equal(base.dataset, r.dataset);
      assert.equal(base.train_per_class, r.train_per_class);
      assert.equal(base.model, r.source_model);
    }
    if (r.status !== 'complete') {
      assert.ok(['base', 'review'].includes(r.arm), 'Pending source and review conditions are allowed');
      for (const field of ['accuracy', 'macro_f1', 'accuracy_ci95', 'macro_f1_ci95', 'review_transitions', 'n_failures']) assert.equal(r[field], null, `Pending ${r.run_id} ${field} must be null`);
    } else for (const m of metrics) {
      assert.equal(typeof r[m], 'number');
      assert.ok(r[m] >= 0 && r[m] <= 1);
      assert.equal(r[`${m}_ci95`].length, 2);
    }
  }
  for (const c of data.comparisons) {
    const a = byId.get(c.a),
      b = byId.get(c.b);
    assert.ok(a && b, 'Contrast endpoints must resolve');
    const complete = a.status === 'complete' && b.status === 'complete';
    assert.equal(c.status, complete ? 'complete' : 'pending');
    for (const m of metrics) if (complete) close(c[`${m}_delta`], a[m] - b[m], 'All contrast values agree with endpoint scores');else {
      assert.equal(c[`${m}_delta`], null);
      assert.equal(c[`${m}_delta_ci95`], null);
    }
    if (!complete) assert.equal(c.transitions, null, 'Pending contrasts cannot claim transition counts');
  }
}

// Entirely invented test data. These scores do not supplement the real matrix.
function completeFixture() {
  const data = {
    schema_version: 1,
    completion: {
      status: 'complete',
      complete_runs: 68,
      expected_runs: 68,
      complete_comparisons: 72,
      expected_comparisons: 72
    },
    models: modelIds.map((id, i) => ({
      id,
      label: id,
      family: i < 4 ? 'open_weight_llm' : 'hosted_llm'
    })),
    datasets: [{
      id: 'sst2',
      label: 'Synthetic binary fixture',
      n_test: 200,
      n_classes: 2,
      n_features: null,
      feature_description: 'Text · TF-IDF for classical models',
      full_training_labels: 10000
    }, {
      id: 'trec',
      label: 'Synthetic multiclass fixture',
      n_test: 200,
      n_classes: 6,
      n_features: null,
      feature_description: 'Text · TF-IDF for classical models',
      full_training_labels: 4886
    }],
    runs: [],
    comparisons: [],
    costs: {
      status: 'complete',
      new_review_calls: 1500,
      known_reported_api_usd: '1.2',
      unknown_cost_calls: 1,
      cumulative_reserved_usd: '23.35',
      authorized_usd: '25'
    },
    limitations: ['SYNTHETIC UI FIXTURE: these invented scores are not scientific measurements.']
  };
  function add(ds, model, arm, shots, correct) {
    const accuracy = correct / ds.n_test,
      macro_f1 = accuracy - .04;
    const r = {
      run_id: `fixture::${ds.id}::${model}::${arm}::${shots}`,
      dataset: ds.id,
      model,
      display_model: model,
      arm,
      status: 'complete',
      train_per_class: shots,
      train_labels: shots === null ? ds.full_training_labels : shots * ds.n_classes,
      label_budget: shots === null ? 'full_training' : shots === 0 ? 'zero_shot' : 'four_per_class',
      accuracy,
      macro_f1,
      accuracy_ci95: [Math.max(0, accuracy - .05), Math.min(1, accuracy + .05)],
      macro_f1_ci95: [Math.max(0, macro_f1 - .05), Math.min(1, macro_f1 + .05)],
      n_test: ds.n_test,
      n_failures: 0,
      review_transitions: null
    };
    data.runs.push(r);
    return r;
  }
  function contrast(kind, a, b, equal) {
    const c = {
      kind,
      dataset: a.dataset,
      a: a.run_id,
      b: b.run_id,
      status: 'complete',
      equal_new_label_budget: equal,
      train_per_class_a: a.train_per_class,
      train_per_class_b: b.train_per_class,
      transitions: kind === 'review_minus_source' ? a.review_transitions : null
    };
    for (const m of metrics) {
      c[`${m}_delta`] = a[m] - b[m];
      c[`${m}_delta_ci95`] = [Math.max(-1, c[`${m}_delta`] - .1), Math.min(1, c[`${m}_delta`] + .1)];
    }
    data.comparisons.push(c);
  }
  for (const ds of data.datasets) {
    for (const shots of [0, 4]) {
      const direct = add(ds, 'typesafe/jev-1.13', 'direct', shots, Math.floor(ds.n_test * .7) + shots);
      for (const [i, model] of modelIds.entries()) {
        const correct = Math.floor(ds.n_test * .55) + i;
        const base = add(ds, model, 'base', shots, correct),
          delta = [-3, -1, 0, 1, 2, 3][i];
        const review = add(ds, model + '+jev_review', 'review', shots, correct + delta);
        review.source_model = model;
        review.source_run_id = base.run_id;
        const fixed = Math.max(0, delta),
          harmed = Math.max(0, -delta);
        const inherited = ds.id === 'trec' && model === 'gpt-6-astra' && shots === 4 ? 1 : 0;
        base.n_failures = inherited;
        review.n_failures = inherited;
        review.review_transitions = {
          wrong_to_correct: fixed,
          correct_to_wrong: harmed,
          correct_to_wrong_label: harmed,
          correct_to_failure: 0,
          both_correct: correct - harmed,
          both_wrong: ds.n_test - correct - fixed,
          changed_predictions: fixed + harmed,
          source_failure_rows: inherited,
          review_stage_failure_rows: 0,
          net_correct_change: delta,
          accuracy_delta_pp: 100 * delta / ds.n_test
        };
        contrast('review_minus_source', review, base, true);
        contrast('review_minus_jev_alone', review, direct, true);
      }
    }
    for (const model of modelIds) for (const arm of ['base', 'review']) {
      const selected = data.runs.filter(r => r.dataset === ds.id && r.arm === arm && (arm === 'base' ? r.model : r.source_model) === model);
      contrast('few_minus_zero_descriptive', selected.find(r => r.train_per_class === 4), selected.find(r => r.train_per_class === 0), false);
    }
    for (const shots of [4, null]) for (const [i, model] of ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'].entries()) add(ds, model, 'classical', shots, Math.floor(ds.n_test * .8) + i);
  }
  return data;
}
function partialFixture(complete, sourceReady = false) {
  const data = structuredClone(complete);
  const pending = data.runs.filter(r => r.arm === 'review' || !sourceReady && r.arm === 'base' && modelIds.slice(2, 4).includes(r.model));
  assert.equal(pending.length, sourceReady ? 24 : 32);
  for (const r of pending) {
    r.status = 'pending: no complete audited artifact';
    for (const field of ['accuracy', 'macro_f1', 'accuracy_ci95', 'macro_f1_ci95', 'review_transitions', 'n_failures', 'n_test', 'train_labels']) r[field] = null;
  }
  const pendingIds = new Set(pending.map(r => r.run_id));
  for (const c of data.comparisons) if (pendingIds.has(c.a) || pendingIds.has(c.b)) {
    c.status = 'pending';
    c.transitions = null;
    for (const m of metrics) {
      c[`${m}_delta`] = null;
      c[`${m}_delta_ci95`] = null;
    }
  }
  data.completion.status = 'in_progress_or_incomplete';
  data.completion.complete_runs = sourceReady ? 44 : 36;
  data.completion.complete_comparisons = data.comparisons.filter(c => c.status === 'complete').length;
  data.costs.status = 'blocked_no_credit';
  data.costs.new_review_calls = 0;
  data.costs.known_reported_api_usd = '0';
  data.costs.unknown_cost_calls = 0;
  return data;
}
function knownMeasuredReferences(data) {
  const nativeCounts = {
    sst2: [[99, 158], [106, 152], [89, 138], [95, 135]],
    trec: [[100, 173], [98, 162], [70, 141], [75, 145]]
  };
  const nativeModels = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'];
  for (const ds of data.datasets) {
    assert.equal(ds.n_test, 200);
    assert.equal(ds.n_features, null);
    assert.equal(ds.n_classes, ds.id === 'sst2' ? 2 : 6);
    assert.equal(ds.full_training_labels, ds.id === 'sst2' ? 10000 : 4886);
    assert.equal(ds.feature_description, 'Text · TF-IDF for classical models');
    for (const [i, model] of nativeModels.entries()) for (const [j, shots] of [4, null].entries()) {
      const r = data.runs.find(r => r.dataset === ds.id && r.arm === 'classical' && r.model === model && r.train_per_class === shots);
      close(r.accuracy, nativeCounts[ds.id][i][j] / ds.n_test, 'Frozen measured classical accuracy');
    }
    for (const [j, shots] of [0, 4].entries()) {
      const r = data.runs.find(r => r.dataset === ds.id && r.arm === 'direct' && r.train_per_class === shots);
      close(r.accuracy, (ds.id === 'sst2' ? [187, 193] : [67, 171])[j] / ds.n_test, 'Frozen measured direct Jev accuracy');
    }
  }
}
async function verifyDataset(data, label) {
  validateAsset(data);
  const original = JSON.stringify(data),
    h = await harness(data);
  assert.equal(h.elements.get('model').options.length, 7);
  assert.ok(plain(h.elements.get('scope').innerHTML).includes(`${data.completion.complete_runs}/68 conditions complete.`));
  const scope = h.elements.get('scope').innerHTML;
  assert.match(scope, /Conservative accounting through this phase:/);
  assert.match(scope, /not the final study total/);
  assert.match(scope, /excludes later controls and retries/);
  assert.match(scope, /results\/completion_20260923\/COSTS\.md/);
  assert.ok(scope.includes(`$${Number(data.costs.cumulative_reserved_usd).toFixed(2)}`), 'Keep the recorded phase subtotal');
  assert.ok(scope.includes(`$${Number(data.costs.authorized_usd).toFixed(2)}`), 'Keep the recorded phase authorization');
  let views = 0;
  for (const dataset of data.datasets.map(d => d.id)) for (const shots of ['0', '4']) for (const model of ['all', ...sourceModels.map(m => m.id)]) for (const view of ['pairs', 'direct']) for (const metric of metrics) for (const nativeBudget of ['4', 'full']) {
    const state = {
      dataset,
      shots,
      model,
      view,
      metric,
      nativeBudget
    };
    try {
      await verifyView(h, data, state);
    } catch (e) {
      e.message = `${label} ${JSON.stringify(state)}: ${e.message}`;
      throw e;
    }
    views++;
  }
  assert.equal(views, 224);
  // A completed synthetic case verifies that inherited failures are distinct
  // from review-stage failures. Pending or real runs need not have this failure.
  if (label.startsWith('SYNTHETIC COMPLETE')) {
    const inherited = data.runs.find(r => r.dataset === 'trec' && r.arm === 'review' && r.source_model === 'gpt-6-astra' && r.train_per_class === 4);
    assert.equal(inherited.n_failures, 1);
    assert.equal(inherited.review_transitions.source_failure_rows, 1);
    assert.equal(inherited.review_transitions.review_stage_failure_rows, 0);
    Object.assign(h.api.state, {
      dataset: 'trec',
      shots: '4',
      model: 'gpt-6-astra',
      view: 'pairs',
      metric: 'accuracy',
      nativeBudget: '4'
    });
    h.api.render();
    const inheritedCSV = (await exported(h))[0];
    assert.equal(inheritedCSV.pipeline_failures, '1');
    assert.equal(inheritedCSV.source_failure_rows, '1');
    assert.equal(inheritedCSV.review_stage_failure_rows, '0');
  }
  // Exercise actual event handlers, reset, and URL restoration, not only render().
  h.elements.get('metric').value = 'macro_f1';
  h.elements.get('metric').listeners.get('change')();
  assert.equal(h.api.state.metric, 'macro_f1');
  h.elements.get('reset').listeners.get('click')();
  assert.deepEqual(JSON.parse(JSON.stringify(h.api.state)), {
    dataset: 'sst2',
    shots: '4',
    model: 'all',
    metric: 'accuracy',
    view: 'pairs',
    nativeBudget: '4'
  });
  const urlState = {
    dataset: 'trec',
    shots: '0',
    model: sourceModels[1].id,
    metric: 'macro_f1',
    view: 'direct',
    nativeBudget: 'full'
  };
  const restored = await harness(data, '?' + new URLSearchParams(urlState));
  assert.deepEqual(JSON.parse(JSON.stringify(restored.api.state)), urlState);
  assert.equal(JSON.stringify(data), original, 'Dashboard must not mutate its input asset');
  console.log(`PASS: ${label}: ${views} filter views and CSV exports; ${data.completion.complete_runs}/68 conditions, ${data.completion.complete_comparisons}/72 comparisons; pending/null handling; paired/direct controls; classical budgets; inherited failure; URL/events/reset.`);
}
async function main() {
  if (asset) {
    knownMeasuredReferences(asset);
    await verifyDataset(asset, 'REAL EXPORTED ASSET');
  }
  const complete = completeFixture(),
    partial = partialFixture(complete),
    sourceReady = partialFixture(complete, true);
  await verifyDataset(complete, 'SYNTHETIC COMPLETE UI FIXTURE (not measurements)');
  await verifyDataset(partial, 'SYNTHETIC PARTIAL 36/68 UI FIXTURE (not measurements)');
  await verifyDataset(sourceReady, 'SYNTHETIC PARTIAL 44/68 UI FIXTURE (not measurements)');
  const otherStatus = structuredClone(partial);
  otherStatus.costs.status = 'waiting_for_local_execution';
  const statusHarness = await harness(otherStatus);
  assert.match(statusHarness.elements.get('status').textContent, /study in progress; unfinished scores are unavailable/);
  await nullChecks(complete);
  assert.equal(JSON.stringify(asset), originalAsset, 'Dashboard must not mutate its input asset');
  console.log(`PASS: null versus zero and non-bracketing CI fixtures. ${asset ? 'Real data verified; ' : 'No real data asset verified; '}synthetic fixtures cover UI behavior only. No network, inference, or writes.`);
}
main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
