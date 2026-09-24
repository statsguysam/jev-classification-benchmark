'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Execute the exact classic-script order declared by each page. Do not remove
// boot calls or replace imports: the production initialization is under test.
function loadDecisionPageScripts(dist, html, pageScript) {
  const scripts = [...html.matchAll(/<script\b([^>]*)\bsrc="([^"]+)"([^>]*)><\/script>/g)];
  assert.deepEqual(scripts.map(match => match[2]), ['decision-dashboard.js', pageScript]);
  return scripts.map(match => {
    const attributes = `${match[1]} ${match[3]}`;
    assert.match(attributes, /\bdefer\b/, 'Shared code and page config must load in order after the DOM');
    assert.doesNotMatch(attributes, /\basync\b|\btype\s*=/, 'These are ordered classic scripts');
    return {
      name: match[2],
      code: fs.readFileSync(path.join(dist, match[2]), 'utf8')
    };
  });
}
async function bootDecisionPage(scripts, context) {
  for (const script of scripts) {
    vm.runInContext(script.code, context, {
      filename: script.name,
      timeout: 1000
    });
  }
  const dashboard = vm.runInContext('decisionDashboard', context);
  await dashboard.ready;
  return dashboard;
}
module.exports = {
  loadDecisionPageScripts,
  bootDecisionPage
};
