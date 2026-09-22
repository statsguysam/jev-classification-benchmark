const el = id => document.getElementById(id);
const escape = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const models = {'Qwen/Qwen2.5-0.5B-Instruct':'Qwen2.5 0.5B','Qwen/Qwen3-4B-Instruct-2507':'Qwen3 4B','HuggingFaceTB/SmolLM2-1.7B-Instruct':'SmolLM2 1.7B','ibm-granite/granite-3.3-2b-instruct':'Granite 3.3 2B','gpt-5.6-luna':'GPT Luna','gpt-6-astra':'GPT Astra'};
const datasets = {breast_cancer:'Breast Cancer · binary',wine:'Wine · multiclass'};
const name = c => `${datasets[c.dataset]} / ${models[c.source_model]||c.source_model} / ${c.shots_per_class ? '4 examples per class':'zero-shot'}`;
const table = (head, rows) => `<table><thead><tr>${head.map(s=>`<th scope="col">${escape(s)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(s=>`<td>${escape(s)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
let data, gates;
const fmt = (v, metric=el('metric').value) => v == null ? 'Unavailable' : metric==='macro_f1' ? v.toFixed(3) : `${(v*100).toFixed(1)}%`;
function renderComparison(){
 const c=data.conditions[Number(el('condition').value)], metric=el('metric').value;
 for(const [id,key] of [['source','never_review'],['review','always_review'],['direct','direct_jev']]) el(`${id}-score`).textContent=fmt(c.metrics[key][metric]);
 el('source-calls').textContent=`${c.n_rows} source inferences`;
 el('review-calls').textContent=`${c.n_rows} source + ${c.required_inferences.always_review.review_api_requests} review requests`;
 el('direct-calls').textContent=`${c.n_rows} direct Jev requests`;
 el('transitions').textContent=`${c.review_minus_base.wrong_to_correct} / ${c.review_minus_base.correct_to_wrong}`;
 el('comparison-title').textContent=name(c);
 el('comparison-note').textContent=`${c.n_rows} held-out rows. Corrections and harms count decisions, independent of the selected metric.`;
 el('comparison').innerHTML=table(['Comparison','Corrected','Harmed','Net correct change'],[['Review vs LLM alone',c.review_minus_base.wrong_to_correct,c.review_minus_base.correct_to_wrong,c.review_minus_base.net_correct_change],['Review vs Jev alone',c.review_minus_direct.wrong_to_correct,c.review_minus_direct.correct_to_wrong,c.review_minus_direct.net_correct_change]]);
}
function renderGate(){
 const c=gates[Number(el('gate').value)], metric=el('metric').value, points=c.curve;
 const x=p=>72+650*p.actual_review_fraction, y=v=>275-225*v;
 const poly=(key)=>points.map(p=>`${x(p)},${y(key==='metrics'?p.metrics[metric]:p.random_matched_rate[metric])}`).join(' ');
 const grid=[0,.25,.5,.75,1].map(v=>`<line x1="72" x2="722" y1="${y(v)}" y2="${y(v)}" stroke="#e2e7f0"/><text x="60" y="${y(v)+5}" text-anchor="end" fill="#627089" font-size="14">${metric==='macro_f1'?v.toFixed(2):Math.round(v*100)+'%'}</text>`).join('');
 const ticks=[0,.25,.5,.75,1].map(v=>`<text x="${72+650*v}" y="300" text-anchor="middle" fill="#627089" font-size="14">${Math.round(v*100)}%</text>`).join('');
 el('curve').innerHTML=`<svg viewBox="0 0 770 340" role="img" aria-label="${escape(name(c))}: selected metric by fraction of rows reviewed. Exact values in following table.">${grid}${ticks}<text x="400" y="330" text-anchor="middle" fill="#627089" font-size="14">Fraction of rows reviewed</text><line x1="72" x2="722" y1="${y(c.metrics.direct_jev[metric])}" y2="${y(c.metrics.direct_jev[metric])}" stroke="#c17b36" stroke-width="2" stroke-dasharray="4 5"/><polyline points="${poly('random')}" fill="none" stroke="#8570c4" stroke-width="2" stroke-dasharray="7 5"/><polyline points="${poly('metrics')}" fill="none" stroke="#008d99" stroke-width="3"/>${points.map(p=>`<circle cx="${x(p)}" cy="${y(p.metrics[metric])}" r="4" fill="#008d99"/>`).join('')}</svg>`;
 el('gate-table').innerHTML=table(['Reviewed','Least-confident first','Random expectation','Corrected / harmed','Source + review calls'],points.map(p=>[`${p.selected_rows}/${c.n_rows}`,fmt(p.metrics[metric]),fmt(p.random_matched_rate[metric]),`${p.transitions_from_never_review.wrong_to_correct} / ${p.transitions_from_never_review.correct_to_wrong}`,`${c.n_rows} + ${p.required_inferences.review_api_requests}`]));
 el('gate-context').textContent=`${name(c)}. Direct Jev: ${fmt(c.metrics.direct_jev[metric])}, using ${c.n_rows} requests and no source inference.`;
}
async function init(){
 try{
 const response=await fetch('review-data.json'); if(!response.ok)throw Error(`HTTP ${response.status}`); data=await response.json();
 gates=data.conditions.filter(c=>c.curve); if(!data.conditions.length||!gates.length)throw Error('No audited complete conditions');
 el('condition').innerHTML=data.conditions.map((c,i)=>`<option value="${i}">${escape(name(c))}</option>`).join('');
 el('gate').innerHTML=gates.map((c,i)=>`<option value="${i}">${escape(name(c))}</option>`).join('');
 el('condition').value=String(Math.max(0,data.conditions.findIndex(c=>c.dataset==='breast_cancer'&&c.source_model.includes('Qwen3')&&c.shots_per_class===4)));
 el('gate').value=String(Math.max(0,gates.findIndex(c=>c.dataset==='breast_cancer'&&c.source_model.includes('Qwen3')&&c.shots_per_class===4)));
 el('status').textContent=`${data.complete_review_conditions}/${data.expected_review_conditions} numerical review conditions complete · ${gates.length} eligible selective-review conditions · exploratory, one split`;
 const v=data.identical_prompts;
 el('prompt-variability').textContent=`${v.valid_output_disagreements} different final labels among ${v.both_valid_rows.toLocaleString()} same-proposal comparisons with identical prompt hashes and two valid review responses. Another ${v.rows_with_either_review_failure} comparisons included a failure.`;
 const completed=data.controls.runs.filter(r=>r.status==='complete').length;
 el('control-status').textContent=`${completed}/12 primary conditions complete; 1,650 primary requests plus 64 exact-prompt repeats planned. ${completed===0?'No controlled result is available.':'Only complete conditions receive metrics.'}`;
 el('condition').addEventListener('change',renderComparison); el('gate').addEventListener('change',renderGate); el('metric').addEventListener('change',()=>{renderComparison();renderGate();});renderComparison();renderGate();
 }catch(error){el('status').textContent=`Evidence unavailable: ${error.message}. No scores have been substituted.`;}
}
init();
