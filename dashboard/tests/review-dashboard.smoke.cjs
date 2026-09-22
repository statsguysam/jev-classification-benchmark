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
const fmt = (v, metric) => v == null ? 'Unavailable' : metric === 'macro_f1' ? v.toFixed(3) : `${(100*v).toFixed(1)}%`;
const decode = value => value.replace(/&(?:amp|lt|gt|quot|#39);/g, c => ({'&amp;':'&','&lt;':'<','&gt;':'>','&quot;':'"','&#39;':"'"}[c]));
const plain = value => decode(value.replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();
const close = (a,b,label) => assert.ok(Math.abs(a-b)<1e-9, `${label}: ${a} != ${b}`);
function tableRows(value) {
  const body = value.match(/<tbody>([\s\S]*?)<\/tbody>/);
  assert.ok(body, 'Expected table body');
  return [...body[1].matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(r => [...r[1].matchAll(/<td>([\s\S]*?)<\/td>/g)].map(c => plain(c[1])));
}
class Element {
  constructor(id) { this.id=id; this.value=''; this.textContent=''; this.options=[]; this.listeners=new Map(); this._html=''; }
  set innerHTML(value) {
    this._html=value;
    this.options=[...value.matchAll(/<option\b[^>]*value="([^"]*)"[^>]*>/g)].map(m=>({value:decode(m[1])}));
    if(this.options.length&&!this.options.some(o=>o.value===this.value))this.value=this.options[0].value;
  }
  get innerHTML(){return this._html;}
  addEventListener(name,handler){this.listeners.set(name,handler);}
  change(value){this.value=String(value);assert.ok(this.listeners.has('change'),`Missing change listener: ${this.id}`);this.listeners.get('change')();}
}
async function harness(data, failure=null) {
  const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element(m[1])]));
  for(const match of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g))elements.get(match[1]).innerHTML=match[2];
  for(const id of ['source-score','review-score','direct-score','transitions'])elements.get(id).textContent='—';
  let fetches=0;
  const context=vm.createContext({console, document:{getElementById(id){assert.ok(elements.has(id),`Unknown DOM ID ${id}`);return elements.get(id);}},
    async fetch(url){fetches++;assert.equal(url,'review-data.json');if(failure==='network')throw Error('simulated network failure');return {ok:!failure,status:503,async json(){return data;}};}});
  assert.match(source,/init\(\);\s*$/,'Expected single trailing init call');
  vm.runInContext(source.replace(/init\(\);\s*$/,'')+'\n;globalThis.smoke={init,renderComparison,renderGate};',context,{timeout:1000});
  await context.smoke.init();
  assert.equal(fetches,1,'Only the aggregate asset may be fetched');
  return {elements,api:context.smoke};
}
function validateAsset(data) {
  assert.equal(data.complete_review_conditions,17);assert.equal(data.expected_review_conditions,24);
  assert.equal(data.conditions.length,17);assert.equal(data.pending_conditions.length,7);
  assert.equal(new Set(data.conditions.map(c=>JSON.stringify([c.dataset,c.source_model,c.shots_per_class]))).size,17);
  assert.equal(data.conditions.filter(c=>c.curve).length,4);
  assert.equal(data.controls.runs.length,12);
  assert.ok(data.controls.runs.every(r=>r.status==='pending'&&r.metrics===null),'Planned controls must not be numeric zeros');
  const forbidden=new Set(['row_id','row_ids','selected_row_ids','ranked_row_ids','prompt','source_max_probabilities','request_id','config']);
  function visit(value){if(Array.isArray(value)){value.forEach(visit);}else if(value&&typeof value==='object'){for(const [key,child] of Object.entries(value)){assert.ok(!forbidden.has(key),`Raw field published: ${key}`);visit(child);}}}
  visit(data);
}
async function main(){
  validateAsset(asset);const before=JSON.stringify(asset),h=await harness(asset),e=h.elements;
  assert.equal(e.get('condition').options.length,17);assert.equal(e.get('gate').options.length,4);
  assert.match(e.get('status').textContent,/17\/24/);assert.match(e.get('control-status').textContent,/0\/12/);
  assert.match(e.get('control-status').textContent,/No controlled result is available/);
  assert.match(e.get('prompt-variability').textContent,/41 different final labels/);
  assert.match(e.get('prompt-variability').textContent,/1,255/);
  let comparisons=0,gatesChecked=0;
  for(const [index,c] of asset.conditions.entries())for(const metric of metrics){
    e.get('condition').change(index);e.get('metric').change(metric);
    for(const [id,arm] of [['source','never_review'],['review','always_review'],['direct','direct_jev']])
      assert.equal(e.get(`${id}-score`).textContent,fmt(c.metrics[arm][metric],metric));
    assert.equal(e.get('source-calls').textContent,`${c.n_rows} source inferences`);
    assert.equal(e.get('review-calls').textContent,`${c.n_rows} source + ${c.required_inferences.always_review.review_api_requests} review requests`);
    assert.equal(e.get('direct-calls').textContent,`${c.n_rows} direct Jev requests`);
    assert.equal(e.get('transitions').textContent,`${c.review_minus_base.wrong_to_correct} / ${c.review_minus_base.correct_to_wrong}`);
    const actual=tableRows(e.get('comparison').innerHTML);
    assert.deepEqual(actual,[['Review vs LLM alone',c.review_minus_base.wrong_to_correct,c.review_minus_base.correct_to_wrong,c.review_minus_base.net_correct_change],
      ['Review vs Jev alone',c.review_minus_direct.wrong_to_correct,c.review_minus_direct.correct_to_wrong,c.review_minus_direct.net_correct_change]].map(r=>r.map(String)));
    comparisons++;
  }
  const gates=asset.conditions.filter(c=>c.curve);
  for(const [index,c] of gates.entries())for(const metric of metrics){
    e.get('gate').change(index);e.get('metric').change(metric);
    const rows=tableRows(e.get('gate-table').innerHTML);assert.equal(rows.length,6);
    assert.deepEqual(c.curve.map(p=>p.requested_review_percent),[0,10,25,50,75,100]);
    for(const [i,p] of c.curve.entries())assert.deepEqual(rows[i],[`${p.selected_rows}/${c.n_rows}`,fmt(p.metrics[metric],metric),fmt(p.random_matched_rate[metric],metric),
      `${p.transitions_from_never_review.wrong_to_correct} / ${p.transitions_from_never_review.correct_to_wrong}`,`${c.n_rows} + ${p.required_inferences.review_api_requests}`]);
    const svg=e.get('curve').innerHTML;assert.ok(!svg.includes('NaN')&&!svg.includes('undefined'));
    const circles=[...svg.matchAll(/<circle cx="([^"]+)" cy="([^"]+)"/g)];assert.equal(circles.length,6);
    for(const [i,p] of c.curve.entries()){close(Number(circles[i][1]),72+650*p.actual_review_fraction,'Actual coverage x');close(Number(circles[i][2]),275-225*p.metrics[metric],'Selected metric y');}
    const polylines=[...svg.matchAll(/<polyline points="([^"]+)"/g)];assert.equal(polylines.length,2);
    const random=polylines[0][1].split(' ').map(pair=>pair.split(',').map(Number));
    for(const [i,p] of c.curve.entries()){close(random[i][0],72+650*p.actual_review_fraction,'Random same-coverage x');close(random[i][1],275-225*p.random_matched_rate[metric],'Random expectation y');}
    const direct=svg.match(/<line x1="72" x2="722" y1="([^"]+)" y2="([^"]+)" stroke="#c17b36"/);
    assert.ok(direct);close(Number(direct[1]),275-225*c.metrics.direct_jev[metric],'Direct Jev baseline');
    const axis=[...svg.matchAll(/<text x="60"[^>]*>([^<]+)<\/text>/g)].map(m=>m[1]);assert.equal(axis.length,5);
    if(metric==='macro_f1')assert.ok(axis.every(label=>!label.includes('%')),'Macro-F1 axis must match 0–1 score formatting');
    else assert.ok(axis.every(label=>label.includes('%')));
    gatesChecked++;
  }
  for(const failure of ['http','network']){
    const failed=await harness(asset,failure);
    assert.match(failed.elements.get('status').textContent,/Evidence unavailable:/);
    assert.match(failed.elements.get('status').textContent,/No scores have been substituted/);
    for(const id of ['source-score','review-score','direct-score'])assert.equal(failed.elements.get(id).textContent,'—');
    assert.equal(failed.elements.get('comparison').innerHTML,'');assert.equal(failed.elements.get('curve').innerHTML,'');
  }
  assert.equal(JSON.stringify(asset),before,'Dashboard must not mutate the aggregate input');
  assert.equal(comparisons,51);assert.equal(gatesChecked,12);
  console.log('PASS: 17 conditions × 3 metrics; 4 eligible gates × 3 metrics; all six coverages and exact chart/table values; 12 pending controls; skipped-source request count; HTTP/network failure; aggregate-only export. No network, inference or writes.');
}
main().catch(error=>{console.error(error.stack||error);process.exitCode=1;});
