#!/usr/bin/env node
'use strict';

// Offline DOM/VM smoke test. Uses the aggregate export; no browser, network,
// inference, or file writes. Optional argument: dashboard directory.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(process.argv[2] || path.join(__dirname, '..'));
const dist = path.join(root, 'dist');
const html = fs.readFileSync(path.join(dist, 'review.html'), 'utf8');
const source = fs.readFileSync(path.join(dist, 'review.js'), 'utf8');
const asset = JSON.parse(fs.readFileSync(path.join(dist, 'review-data.json'), 'utf8'));
const metrics = ['micro_accuracy', 'balanced_accuracy', 'macro_f1'];
const fmt = (v, metric) => v == null ? 'Unavailable' : metric === 'macro_f1' ? v.toFixed(3) : `${(100 * v).toFixed(1)}%`;
const decode = value => value.replace(/&(?:amp|lt|gt|quot|#39);/g, c => ({
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'"
})[c]);
const plain = value => decode(value.replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();
const close = (a, b, label) => assert.ok(Math.abs(a - b) < 1e-9, `${label}: ${a} != ${b}`);
function tableRows(value) {
  const body = value.match(/<tbody>([\s\S]*?)<\/tbody>/);
  assert.ok(body, 'Expected table body');
  return [...body[1].matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(r => [...r[1].matchAll(/<td>([\s\S]*?)<\/td>/g)].map(c => plain(c[1])));
}
class Element {
  constructor(id) {
    this.id = id;
    this.value = '';
    this.textContent = '';
    this.options = [];
    this.listeners = new Map();
    this._html = '';
  }
  set innerHTML(value) {
    this._html = value;
    this.options = [...value.matchAll(/<option\b[^>]*value="([^"]*)"[^>]*>/g)].map(m => ({
      value: decode(m[1])
    }));
    if (this.options.length && !this.options.some(o => o.value === this.value)) this.value = this.options[0].value;
  }
  get innerHTML() {
    return this._html;
  }
  addEventListener(name, handler) {
    this.listeners.set(name, handler);
  }
  change(value) {
    this.value = String(value);
    assert.ok(this.listeners.has('change'), `Missing change listener: ${this.id}`);
    this.listeners.get('change')();
  }
}
async function harness(data, failure = null) {
  const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element(m[1])]));
  for (const match of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) elements.get(match[1]).innerHTML = match[2];
  for (const id of ['source-score', 'review-score', 'direct-score', 'transitions']) elements.get(id).textContent = 'N/A';
  let fetches = 0;
  const context = vm.createContext({
    console,
    document: {
      getElementById(id) {
        assert.ok(elements.has(id), `Unknown DOM ID ${id}`);
        return elements.get(id);
      }
    },
    async fetch(url) {
      fetches++;
      assert.equal(url, 'review-data.json');
      if (failure === 'network') throw Error('simulated network failure');
      return {
        ok: !failure,
        status: 503,
        async json() {
          return data;
        }
      };
    }
  });
  assert.match(source, /init\(\);\s*$/, 'Expected single trailing init call');
  vm.runInContext(source.replace(/init\(\);\s*$/, '') + '\n;globalThis.smoke={init,renderComparison,renderGate,renderControls,renderRecovery};', context, {
    timeout: 1000
  });
  await context.smoke.init();
  assert.equal(fetches, 1, 'Only the aggregate asset may be fetched');
  return {
    elements,
    api: context.smoke
  };
}
function validateAsset(data) {
  assert.equal(data.expected_review_conditions, 24);
  assert.equal(data.conditions.length, data.complete_review_conditions);
  assert.equal(data.pending_conditions.length, 24 - data.complete_review_conditions);
  assert.equal(new Set(data.conditions.map(c => JSON.stringify([c.dataset, c.source_model, c.shots_per_class]))).size, data.conditions.length);
  assert.equal(data.controls.runs.length, 12);
  assert.ok(data.controls.runs.every(r => r.status === 'complete' || r.metrics === null), 'Planned controls must not be numeric zeros');
  const forbidden = new Set(['row_id', 'row_ids', 'selected_row_ids', 'ranked_row_ids', 'prompt', 'source_max_probabilities', 'request_id', 'config']);
  function visit(value) {
    if (Array.isArray(value)) {
      value.forEach(visit);
    } else if (value && typeof value === 'object') {
      for (const [key, child] of Object.entries(value)) {
        assert.ok(!forbidden.has(key), `Raw field published: ${key}`);
        visit(child);
      }
    }
  }
  visit(data);
}
async function main() {
  validateAsset(asset);
  const before = JSON.stringify(asset),
    h = await harness(asset),
    e = h.elements;
  assert.equal(e.get('condition').options.length, asset.conditions.length);
  assert.equal(e.get('gate').options.length, asset.conditions.filter(c => c.curve).length);
  assert.ok(e.get('status').textContent.includes(asset.complete_review_conditions + '/24'));
  const completed = asset.controls.runs.filter(r => r.status === 'complete').length;
  assert.ok(e.get('control-status').textContent.includes(completed + '/12'));
  if (!completed) assert.match(e.get('control-status').textContent, /No controlled result is available/);
  assert.ok(e.get('prompt-variability').textContent.includes(asset.identical_prompts.valid_output_disagreements + ' different final labels'));
  assert.ok(e.get('prompt-variability').textContent.includes(asset.identical_prompts.both_valid_rows.toLocaleString()));
  let comparisons = 0,
    gatesChecked = 0;
  for (const [index, c] of asset.conditions.entries()) for (const metric of metrics) {
    e.get('condition').change(index);
    e.get('metric').change(metric);
    for (const [id, arm] of [['source', 'never_review'], ['review', 'always_review'], ['direct', 'direct_jev']]) assert.equal(e.get(`${id}-score`).textContent, fmt(c.metrics[arm][metric], metric));
    assert.equal(e.get('source-calls').textContent, `${c.n_rows} source inferences`);
    assert.equal(e.get('review-calls').textContent, `${c.n_rows} source + ${c.required_inferences.always_review.review_api_requests} review requests`);
    assert.equal(e.get('direct-calls').textContent, `${c.n_rows} direct Jev requests`);
    assert.equal(e.get('transitions').textContent, `${c.review_minus_base.wrong_to_correct} / ${c.review_minus_base.correct_to_wrong}`);
    const actual = tableRows(e.get('comparison').innerHTML);
    assert.deepEqual(actual, [['Review vs LLM alone', c.review_minus_base.wrong_to_correct, c.review_minus_base.correct_to_wrong, c.review_minus_base.net_correct_change], ['Review vs Jev alone', c.review_minus_direct.wrong_to_correct, c.review_minus_direct.correct_to_wrong, c.review_minus_direct.net_correct_change]].map(r => r.map(String)));
    comparisons++;
  }
  const gates = asset.conditions.filter(c => c.curve);
  for (const [index, c] of gates.entries()) for (const metric of metrics) {
    e.get('gate').change(index);
    e.get('metric').change(metric);
    const rows = tableRows(e.get('gate-table').innerHTML);
    assert.equal(rows.length, 6);
    assert.deepEqual(c.curve.map(p => p.requested_review_percent), [0, 10, 25, 50, 75, 100]);
    for (const [i, p] of c.curve.entries()) assert.deepEqual(rows[i], [`${p.selected_rows}/${c.n_rows}`, fmt(p.metrics[metric], metric), fmt(p.random_matched_rate[metric], metric), `${p.transitions_from_never_review.wrong_to_correct} / ${p.transitions_from_never_review.correct_to_wrong}`, `${c.n_rows} + ${p.required_inferences.review_api_requests}`]);
    const svg = e.get('curve').innerHTML;
    assert.ok(!svg.includes('NaN') && !svg.includes('undefined'));
    const circles = [...svg.matchAll(/<circle cx="([^"]+)" cy="([^"]+)"/g)];
    assert.equal(circles.length, 6);
    for (const [i, p] of c.curve.entries()) {
      close(Number(circles[i][1]), 72 + 650 * p.actual_review_fraction, 'Actual coverage x');
      close(Number(circles[i][2]), 275 - 225 * p.metrics[metric], 'Selected metric y');
    }
    const polylines = [...svg.matchAll(/<polyline points="([^"]+)"/g)];
    assert.equal(polylines.length, 2);
    const random = polylines[0][1].split(' ').map(pair => pair.split(',').map(Number));
    for (const [i, p] of c.curve.entries()) {
      close(random[i][0], 72 + 650 * p.actual_review_fraction, 'Random same-coverage x');
      close(random[i][1], 275 - 225 * p.random_matched_rate[metric], 'Random expectation y');
    }
    const direct = svg.match(/<line x1="72" x2="722" y1="([^"]+)" y2="([^"]+)" stroke="#c17b36"/);
    assert.ok(direct);
    close(Number(direct[1]), 275 - 225 * c.metrics.direct_jev[metric], 'Direct Jev baseline');
    const axis = [...svg.matchAll(/<text x="60"[^>]*>([^<]+)<\/text>/g)].map(m => m[1]);
    assert.equal(axis.length, 5);
    if (metric === 'macro_f1') assert.ok(axis.every(label => !label.includes('%')), 'Macro-F1 axis must match 0 to 1 score formatting');else assert.ok(axis.every(label => label.includes('%')));
    gatesChecked++;
  }
  for (const failure of ['http', 'network']) {
    const failed = await harness(asset, failure);
    assert.match(failed.elements.get('status').textContent, /Evidence unavailable:/);
    assert.match(failed.elements.get('status').textContent, /No scores have been substituted/);
    for (const id of ['source-score', 'review-score', 'direct-score']) assert.equal(failed.elements.get(id).textContent, 'N/A');
    assert.equal(failed.elements.get('comparison').innerHTML, '');
    assert.equal(failed.elements.get('curve').innerHTML, '');
  }
  assert.equal(JSON.stringify(asset), before, 'Dashboard must not mutate the aggregate input');
  assert.equal(comparisons, asset.conditions.length * 3);
  assert.equal(gatesChecked, gates.length * 3);
  await checkPublishedControlRecovery(asset);
  await checkControls();
  console.log('PASS: every published condition and gate × 3 metrics; all six coverages; published control recovery × 4 datasets × 3 metrics; complete/pending synthetic controls and paired CIs; explicit missing/pending recovery; repeat agreement/failure denominators; first-attempt/recovery display; HTTP/network failures. No network, inference or writes.');
}
async function checkPublishedControlRecovery(data) {
  const original = JSON.stringify(data),
    h = await harness(data),
    e = h.elements,
    r = data.control_recovery;
  if (!r || r.status !== 'complete') {
    assert.match(e.get('control-recovery-status').textContent, /Post-control recovery pending/);
    assert.equal(e.get('control-recovery-arms').innerHTML, '');
    assert.equal(e.get('control-recovery-contrasts').innerHTML, '');
    return;
  }
  assert.equal(r.runs.length, 12);
  assert.equal(r.comparisons.length, 8);
  const counts = r.recovery_counts;
  if (r.no_op) assert.match(e.get('control-recovery-status').textContent, /Every original control and repeat response was valid, so no additional calls were needed and scores are unchanged/);else {
    assert.ok(e.get('control-recovery-status').textContent.includes(`${r.new_calls} additional calls`));
    assert.ok(e.get('control-recovery-status').textContent.includes(`Primary failures: ${counts.primary.original_failures} → ${counts.primary.remaining_failures}; repeat failures: ${counts.repeat.original_failures} → ${counts.repeat.remaining_failures}`));
  }
  const armNames = {
    actual: 'Actual proposal',
    no_proposal: 'No proposal',
    shuffled: 'Shuffled proposal'
  };
  for (const dataset of ['breast_cancer', 'wine', 'sst2', 'trec']) for (const metric of metrics) {
    e.get('control-dataset').change(dataset);
    e.get('metric').change(metric);
    const m = metric === 'micro_accuracy' ? 'accuracy' : metric,
      runs = r.runs.filter(v => v.dataset === dataset),
      contrasts = r.comparisons.filter(v => v.dataset === dataset);
    assert.equal(runs.length, 3);
    assert.equal(contrasts.length, 2);
    assert.deepEqual(tableRows(e.get('control-recovery-arms').innerHTML), runs.map(v => [armNames[v.arm], String(v.n_rows), fmt(v.first_attempt.metrics[m], metric), fmt(v.recovered.metrics[m], metric), `${v.first_attempt.metrics.n_failures} / ${v.recovered.metrics.n_failures}`]));
    const diff = v => m === 'macro_f1' ? `${v > 0 ? '+' : ''}${v.toFixed(3)}` : `${v > 0 ? '+' : ''}${(100 * v).toFixed(1)} pp`;
    const display = view => {
      const ci = view.paired_group_bootstrap?.metrics?.[m];
      const point = m === 'balanced_accuracy' ? view.balanced_accuracy_delta : ci?.estimate;
      assert.equal(typeof point, 'number', 'A completed contrast must contain the selected metric');
      return `${diff(point)} · ${ci?.ci95 ? ci.ci95.map(diff).join(' to ') : 'Unavailable'}`;
    };
    assert.deepEqual(tableRows(e.get('control-recovery-contrasts').innerHTML), contrasts.map(v => [`${armNames[v.a]} − ${armNames[v.b]}`, display(v.first_attempt), display(v.recovered)]));
  }
  assert.equal(JSON.stringify(data), original, 'Rendering recovered controls must not mutate primary or recovered evidence');
}
function controlFixture() {
  const d = JSON.parse(JSON.stringify(asset)),
    expected = {
      breast_cancer: 114,
      wine: 36,
      sst2: 200,
      trec: 200
    };
  d.recovery = {
    status: 'not_available',
    conditions: []
  };
  d.control_recovery = {
    status: 'not_available',
    no_op: null,
    new_calls: null,
    recovery_counts: null,
    runs: [],
    comparisons: []
  };
  const arm = {
    actual: {
      accuracy: .875,
      balanced_accuracy: .84,
      macro_f1: .83
    },
    no_proposal: {
      accuracy: .75,
      balanced_accuracy: .73,
      macro_f1: .70
    },
    shuffled: {
      accuracy: .5,
      balanced_accuracy: .47,
      macro_f1: .42
    }
  };
  const ci = (a, b = 0) => ({
    metrics: Object.fromEntries(['accuracy', 'macro_f1'].map(m => [m, {
      estimate: a[m] - (b ? b[m] : 0),
      ci95: [a[m] - (b ? b[m] : 0) - .05, a[m] - (b ? b[m] : 0) + .05]
    }]))
  });
  d.controls = {
    status: 'complete',
    planned_cases: 550,
    planned_primary_requests: 1650,
    planned_repeats: 64,
    expected_requests: 1714,
    saved_requests: 1714,
    expected_primary_arms: 12,
    complete_primary_arms: 12,
    expected_primary_contrasts: 8,
    complete_primary_contrasts: 8,
    runs: Object.keys(expected).flatMap(dataset => Object.keys(arm).map((a, i) => ({
      dataset,
      arm: a,
      status: 'complete',
      saved_requests: expected[dataset],
      expected_requests: expected[dataset],
      training_examples: 8,
      metrics: {
        ...arm[a],
        n_rows: expected[dataset],
        n_failures: i + 1
      },
      group_bootstrap: ci(arm[a])
    }))),
    comparisons: Object.keys(expected).flatMap(dataset => ['no_proposal', 'shuffled'].map(b => ({
      dataset,
      a: 'actual',
      b,
      status: 'complete',
      paired_group_bootstrap: ci(arm.actual, arm[b]),
      balanced_accuracy_delta: arm.actual.balanced_accuracy - arm[b].balanced_accuracy,
      transitions_B_to_A: {
        wrong_to_correct: 23,
        correct_to_wrong: 11
      }
    })))
  };
  const per = {
    both_valid: 15,
    valid_label_agreement: 14,
    valid_label_disagreement: 1,
    reference_failed_only: 0,
    repeat_failed_only: 0,
    both_failed: 1
  };
  d.controls.serving_repeat_diagnostic = {
    status: 'complete',
    expected_pairs: 64,
    available_complete_pairs: 64,
    valid_pair_agreement: 56 / 60,
    valid_agreement_fraction_of_all_pairs: 56 / 64,
    counts: Object.fromEntries(Object.entries(per).map(([k, v]) => [k, v * 4])),
    by_dataset: Object.fromEntries(Object.keys(expected).map(k => [k, {
      ...per
    }]))
  };
  return d;
}
async function checkControls() {
  const fixture = controlFixture(),
    original = JSON.stringify(fixture),
    h = await harness(fixture),
    e = h.elements;
  assert.match(e.get('control-status').textContent, /12\/12/);
  assert.match(e.get('control-status').textContent, /1,714\/1,714/);
  for (const dataset of ['breast_cancer', 'wine', 'sst2', 'trec']) for (const metric of metrics) {
    e.get('control-dataset').change(dataset);
    e.get('metric').change(metric);
    const m = metric === 'micro_accuracy' ? 'accuracy' : metric,
      rows = tableRows(e.get('control-arms').innerHTML),
      contrasts = tableRows(e.get('control-contrasts').innerHTML);
    assert.equal(rows.length, 3);
    assert.equal(contrasts.length, 2);
    const expected = fixture.controls.runs.filter(r => r.dataset === dataset);
    for (const [i, r] of expected.entries()) {
      assert.equal(rows[i][1], r.saved_requests + '/' + r.expected_requests);
      assert.equal(rows[i][2], 'Complete');
      assert.equal(rows[i][3], fmt(r.metrics[m], metric));
      assert.equal(rows[i][5], String(r.metrics.n_failures));
      assert.equal(rows[i][4], m === 'balanced_accuracy' ? 'Unavailable' : r.group_bootstrap.metrics[m].ci95.map(v => fmt(v, metric)).join(' to '));
    }
    const diff = v => m === 'macro_f1' ? `${v > 0 ? '+' : ''}${v.toFixed(3)}` : `${v > 0 ? '+' : ''}${(100 * v).toFixed(1)} pp`;
    for (const [i, c] of fixture.controls.comparisons.filter(r => r.dataset === dataset).entries()) {
      assert.match(contrasts[i][0], /^Actual proposal − (No proposal|Shuffled proposal)$/);
      assert.equal(contrasts[i][2], diff(m === 'balanced_accuracy' ? c.balanced_accuracy_delta : c.paired_group_bootstrap.metrics[m].estimate));
      assert.equal(contrasts[i][3], m === 'balanced_accuracy' ? 'Unavailable' : c.paired_group_bootstrap.metrics[m].ci95.map(diff).join(' to '));
      assert.equal(contrasts[i][4], '23 / 11');
    }
  }
  assert.match(e.get('repeat-summary').textContent, /93.3% agreement among 60 pairs/);
  assert.match(e.get('repeat-summary').textContent, /56 agree; 4 disagree/);
  assert.equal(tableRows(e.get('repeat-table').innerHTML).length, 4);
  assert.ok(tableRows(e.get('repeat-table').innerHTML).every(r => r.slice(1).join('|') === '15|14 / 1|0|0|1'));
  assert.match(e.get('recovery-status').textContent, /Recovery report pending/);
  assert.match(e.get('control-recovery-status').textContent, /Post-control recovery pending/);
  assert.equal(e.get('recovery-table').innerHTML, '');
  assert.equal(e.get('control-recovery-arms').innerHTML, '');
  assert.equal(e.get('control-recovery-contrasts').innerHTML, '');
  for (const status of ['absent', 'pending']) {
    const missing = controlFixture();
    if (status === 'absent') delete missing.control_recovery;else missing.control_recovery.status = 'pending';
    const mh = await harness(missing),
      me = mh.elements;
    assert.match(me.get('control-recovery-status').textContent, /Post-control recovery pending/);
    assert.equal(me.get('control-recovery-arms').innerHTML, '');
    assert.equal(me.get('control-recovery-contrasts').innerHTML, '');
    assert.equal(tableRows(me.get('control-arms').innerHTML).length, 3, 'Missing recovery must preserve the primary results');
  }
  const partial = controlFixture();
  partial.controls.runs[0].status = 'pending';
  partial.controls.runs[0].saved_requests = 113;
  partial.controls.runs[0].metrics = null;
  partial.controls.runs[0].group_bootstrap = null;
  partial.controls.comparisons.filter(c => c.dataset === 'breast_cancer').forEach(c => {
    c.status = 'pending';
    c.paired_group_bootstrap = null;
    c.balanced_accuracy_delta = null;
    c.transitions_B_to_A = null;
  });
  partial.controls.serving_repeat_diagnostic = {
    status: 'pending',
    expected_pairs: 64,
    available_complete_pairs: 63,
    counts: null,
    by_dataset: null
  };
  const pending = await harness(partial),
    p = pending.elements;
  assert.match(p.get('control-status').textContent, /11\/12/);
  assert.deepEqual(tableRows(p.get('control-arms').innerHTML)[0], ['Actual proposal', '113/114', 'Pending', 'Pending', 'Pending', 'Pending']);
  assert.ok(tableRows(p.get('control-contrasts').innerHTML).every(r => r.slice(1).every(v => v === 'Pending')));
  assert.match(p.get('repeat-status').textContent, /63\/64/);
  assert.equal(p.get('repeat-summary').textContent, 'Agreement: Pending');
  assert.equal(p.get('repeat-table').innerHTML, '');
  const recovered = controlFixture();
  recovered.recovery = {
    status: 'complete',
    note: 'Audited recovery overlay; original first attempts remain.',
    conditions: [{
      dataset: 'wine',
      model: 'gpt-6-astra',
      shots_per_class: 4,
      n_rows: 36,
      status: 'complete',
      same_request_as_original_snapshot: true,
      original_snapshot: {
        accuracy: .75,
        n_failures: 1
      },
      first_attempt: {
        accuracy: .75,
        n_failures: 1
      },
      recovered: {
        accuracy: .7777777778,
        n_failures: 0
      },
      paired_group_bootstrap: {
        metrics: {
          accuracy: {
            estimate: 1 / 36,
            ci95: [0, 1 / 12]
          }
        }
      },
      corrected: 1,
      harmed: 0
    }]
  };
  const recovery = await harness(recovered),
    rr = tableRows(recovery.elements.get('recovery-table').innerHTML)[0];
  assert.deepEqual(rr.slice(1), ['Complete', '75.0%', '75.0%', '77.8%', '+2.8 pp · 0.0 pp to +8.3 pp', '1 / 1 / 0', '1 / 0']);
  const dependent = JSON.parse(JSON.stringify(recovered));
  dependent.recovery.conditions[0].same_request_as_original_snapshot = false;
  dependent.recovery.conditions[0].first_attempt = {
    accuracy: .7777777778,
    n_failures: 0
  };
  dependent.recovery.conditions[0].paired_group_bootstrap = {
    metrics: {
      accuracy: {
        estimate: 0,
        ci95: [0, 0]
      }
    }
  };
  dependent.recovery.conditions[0].corrected = 0;
  const dh = await harness(dependent),
    dr = tableRows(dh.elements.get('recovery-table').innerHTML)[0];
  assert.equal(dr[1], 'Complete · source recovered before review');
  assert.equal(dr[2], '75.0%');
  assert.equal(dr[3], '77.8%');
  assert.equal(dr[6], '1 / 0 / 0');
  assert.equal(dr[7], '0 / 0');
  const secondary = controlFixture(),
    primarySnapshot = JSON.stringify(secondary.controls);
  secondary.control_recovery = {
    status: 'complete',
    no_op: false,
    new_calls: 16,
    recovery_counts: {
      primary: {
        original_failures: 12,
        remaining_failures: 0
      },
      repeat: {
        original_failures: 4,
        remaining_failures: 0
      }
    },
    runs: secondary.controls.runs.map(v => ({
      dataset: v.dataset,
      arm: v.arm,
      n_rows: v.metrics.n_rows,
      first_attempt: {
        metrics: v.metrics
      },
      recovered: {
        metrics: {
          ...v.metrics,
          accuracy: 1,
          balanced_accuracy: 1,
          macro_f1: 1,
          n_failures: 0
        }
      }
    })),
    comparisons: secondary.controls.comparisons.map(v => ({
      dataset: v.dataset,
      a: v.a,
      b: v.b,
      first_attempt: v,
      recovered: {
        paired_group_bootstrap: {
          metrics: {
            accuracy: {
              estimate: 0,
              ci95: [0, 0]
            },
            macro_f1: {
              estimate: 0,
              ci95: [0, 0]
            }
          }
        },
        balanced_accuracy_delta: 0
      }
    })),
    note: 'Separate secondary sensitivity.',
    interval_note: 'Balanced-accuracy differences have no interval.'
  };
  const sh = await harness(secondary),
    se = sh.elements;
  assert.match(se.get('control-recovery-status').textContent, /16 additional calls/);
  assert.match(se.get('control-recovery-status').textContent, /Primary failures: 12 → 0; repeat failures: 4 → 0/);
  for (const dataset of ['breast_cancer', 'wine', 'sst2', 'trec']) for (const metric of metrics) {
    se.get('control-dataset').change(dataset);
    se.get('metric').change(metric);
    const m = metric === 'micro_accuracy' ? 'accuracy' : metric,
      rows = tableRows(se.get('control-recovery-arms').innerHTML);
    const old = secondary.controls.runs.filter(v => v.dataset === dataset);
    assert.equal(rows.length, 3);
    rows.forEach((r, i) => {
      assert.equal(r[2], fmt(old[i].metrics[m], metric));
      assert.equal(r[3], fmt(1, metric));
      assert.equal(r[4], old[i].metrics.n_failures + ' / 0');
    });
    const contrasts = tableRows(se.get('control-recovery-contrasts').innerHTML);
    assert.equal(contrasts.length, 2);
    assert.ok(contrasts.every(r => r[2] === (m === 'macro_f1' ? '0.000 · 0.000 to 0.000' : m === 'balanced_accuracy' ? '0.0 pp · Unavailable' : '0.0 pp · 0.0 pp to 0.0 pp')));
  }
  assert.equal(JSON.stringify(secondary.controls), primarySnapshot);
  secondary.control_recovery.no_op = true;
  secondary.control_recovery.new_calls = 0;
  const noop = await harness(secondary);
  assert.match(noop.elements.get('control-recovery-status').textContent, /Every original control and repeat response was valid, so no additional calls were needed and scores are unchanged/);
  assert.equal(JSON.stringify(fixture), original, 'Rendering controls must not mutate the evidence');
}
main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
