const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'dist/data.json'), 'utf8'));
const elements = new Map();
const document = {
  getElementById: id => {
    if (!elements.has(id)) elements.set(id, {
      textContent: '',
      hidden: true,
      disabled: false,
      innerHTML: '',
      clientWidth: 850
    });
    return elements.get(id);
  }
};
const captured = [];
const sandbox = {
  document,
  console,
  setTimeout: () => 0,
  clearTimeout: () => {},
  capture: (...args) => captured.push(args),
  fixture: data
};
vm.createContext(sandbox);
let source = fs.readFileSync(path.join(root, 'dist/app.js'), 'utf8');
source = source.replace(/\nboot\(\);\s*$/, '');
vm.runInContext(source, sandbox);
vm.runInContext('DATA=fixture;download=(...args)=>capture(...args);', sandbox);
const evaluate = code => vm.runInContext(code, sandbox);
assert.equal(evaluate('DATA.runs.filter(r=>match(r,{...defaults,budget:"all",seed:"all"})).length'), 290);
assert.equal(evaluate('DATA.runs.filter(r=>match(r,{...defaults,budget:"all",seed:"all",replicas:true})).length'), 318);
assert.equal(evaluate('DATA.runs.filter(r=>match(r,{...defaults,type:"numeric",task:"multiclass",family:"jev"})).length'), 2);
assert.equal(evaluate('getMetric(DATA.runs.find(r=>r.family==="hosted"),"log_loss")'), null);
assert.equal(evaluate('getMetric(DATA.runs.find(r=>r.family==="jev"),"failure_rate")'), .01);
assert.throws(() => evaluate('validateFilters({budget:"999"})'), /Unsupported/);
evaluate('filtered=DATA.runs.filter(r=>match(r));tab="runs";search="Wine";exportCSV();');
const [filename, csv, mime] = captured[0];
assert.equal(filename, 'jev-benchmark-filtered.csv');
assert.equal(csv.split('\r\n').length, 18);
assert.match(csv, /accuracy_ci95_low,accuracy_ci95_high/);
assert.match(csv, /macro_f1_ci95_low,macro_f1_ci95_high/);
assert.ok(csv.split('\r\n').slice(1).every(row => row.startsWith('"wine",')));
assert.equal(mime, 'text/csv;charset=utf-8');
evaluate('focus="titanic";metric="macro_f1";renderChart();');
const svg = elements.get('main-chart').innerHTML;
assert.match(svg, /Macro-F1 by model on Titanic/);
assert.match(svg, /0\.25/);
assert.ok(!svg.includes('NaN'));
assert.match(svg, /0\.846/);
evaluate('renderAudit();');
const accounting = elements.get('audit-content').innerHTML;
assert.match(accounting, /Historical accounting snapshot/);
assert.match(accounting, /earlier phase totals, not the final study totals/);
assert.match(accounting, /before the numerical\/text review extensions, controls and retries/);
assert.match(accounting, /Authorization at this snapshot/);
assert.match(accounting, /Remaining at this snapshot/);
assert.match(accounting, /results\/completion_20260923\/COSTS\.md/);
assert.ok(accounting.includes('$' + Number(data.costs.conservative_accounted_usd).toFixed(2)));
assert.ok(accounting.includes('$' + Number(data.costs.authorized_usd).toFixed(2)));
assert.ok(accounting.includes('$' + Number(data.costs.remaining_capacity_usd).toFixed(2)));
console.log('Dashboard smoke checks passed: filters, replicas, missing probabilities, failures, invalid inputs, filtered CSV, CI columns and chart values.');
