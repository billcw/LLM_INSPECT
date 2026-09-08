/* No external scripts, fonts, analytics, or model-service calls. */
'use strict';
const $ = id => document.getElementById(id);
const M = LabMath, esc = M.escape;
const state = {loaded:false, busy:false, prediction:null, attention:null, verification:null, records:[], notes:[], draws:new Map(), names:new Map(), tab:0};
const tabs = Array.from(document.querySelectorAll('#navigation button'));
const num = (x, digits=5) => x == null ? '—' : Number(x).toLocaleString('en-US', {maximumSignificantDigits:digits});
const pct = x => x == null ? '—' : `${num(100*x,6)}%`;
const str = x => esc(M.label(x));
const int = id => {const v=Number($(id).value); if (!Number.isSafeInteger(v)) throw Error(`${id}: enter an integer within the safe numeric range`); return v;};
const optional = id => $(id).value.trim()==='' ? null : int(id);
const prompt = () => $('prompt').value;
const temperature = () => Number($('temperature').value);
const table = (headers,rows) => `<div class="table-scroll"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const detail = (title,data) => `<details><summary>${esc(title)}</summary><pre>${esc(JSON.stringify(data,null,2))}</pre></details>`;
const finiteVector = values => Array.isArray(values) && values.length > 0 && values.every(Number.isFinite);
const finiteMatrix = values => Array.isArray(values) && values.length > 0 && values.every(row=>finiteVector(row) && row.length===values[0].length);
function valueStats(values){
  const flat=values.flat(Infinity),count=flat.length,min=Math.min(...flat),max=Math.max(...flat),mean=flat.reduce((s,x)=>s+x,0)/count;
  const sd=Math.sqrt(flat.reduce((s,x)=>s+(x-mean)**2,0)/count);
  return {count,min,max,mean,sd};
}
function statsLine(values){const s=valueStats(values);return `<p class="chart-stats">${s.count.toLocaleString()} values · min ${num(s.min,7)} · max ${num(s.max,7)} · mean ${num(s.mean,7)} · standard deviation ${num(s.sd,7)}</p>`;}
function vectorProfile(values){
  if(!finiteVector(values))return '';
  const width=760,height=235,left=54,right=16,top=18,bottom=35,innerW=width-left-right,innerH=height-top-bottom;
  const bound=Math.max(...values.map(Math.abs),1e-12),zero=top+innerH/2,y=v=>zero-v/bound*(innerH/2),groups=Math.min(values.length,240);
  let marks='';
  for(let g=0;g<groups;g++){
    const start=Math.floor(g*values.length/groups),end=Math.max(start+1,Math.floor((g+1)*values.length/groups)),slice=values.slice(start,end);
    const low=Math.min(...slice),high=Math.max(...slice),avg=slice.reduce((a,b)=>a+b,0)/slice.length,x=left+(g+.5)*innerW/groups;
    marks+=`<line x1="${x}" x2="${x}" y1="${y(low)}" y2="${y(high)}" class="range-mark"><title>Coordinate${end-start>1?'s':''} ${start}${end-start>1?'–'+(end-1):''}: min ${num(low,7)}, max ${num(high,7)}, mean ${num(avg,7)}</title></line><line x1="${x}" x2="${x}" y1="${zero}" y2="${y(avg)}" class="mean-mark"/>`;
  }
  return `<figure class="numeric-chart"><figcaption>Coordinate profile</figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Signed values by vector coordinate"><line x1="${left}" x2="${width-right}" y1="${zero}" y2="${zero}" class="zero-line"/><text x="${left-7}" y="${top+5}" text-anchor="end">+${num(bound,4)}</text><text x="${left-7}" y="${zero+4}" text-anchor="end">0</text><text x="${left-7}" y="${top+innerH+4}" text-anchor="end">−${num(bound,4)}</text>${marks}<text x="${left}" y="${height-8}">coordinate 0</text><text x="${width-right}" y="${height-8}" text-anchor="end">coordinate ${values.length-1}</text></svg><p class="chart-note">Blue bars show signed coordinate values from zero. Grey ranges preserve local minima and maxima when many coordinates are grouped. Coordinate order is the model’s learned dimension order; adjacent bars are not necessarily semantically related.</p></figure>`;
}
function histogram(values){
  if(!finiteVector(values))return '';
  const width=420,height=235,left=48,right=14,top=18,bottom=38,innerW=width-left-right,innerH=height-top-bottom,min=Math.min(...values),max=Math.max(...values),bins=Math.min(20,Math.max(1,Math.ceil(Math.sqrt(values.length))));
  const counts=Array(bins).fill(0);for(const v of values){const i=max===min?0:Math.min(bins-1,Math.floor((v-min)/(max-min)*bins));counts[i]++;}
  const peak=Math.max(...counts,1),bars=counts.map((c,i)=>{const w=innerW/bins-2,h=c/peak*innerH,x=left+i*innerW/bins,y=top+innerH-h;return `<rect x="${x}" y="${y}" width="${Math.max(1,w)}" height="${h}" class="hist-bar"><title>Bin ${i+1}: ${c} values</title></rect>`;}).join('');
  return `<figure class="numeric-chart histogram"><figcaption>Value distribution</figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Histogram of numeric values"><line x1="${left}" x2="${width-right}" y1="${top+innerH}" y2="${top+innerH}" class="zero-line"/>${bars}<text x="${left}" y="${height-9}">${num(min,4)}</text><text x="${width-right}" y="${height-9}" text-anchor="end">${num(max,4)}</text><text x="${left-7}" y="${top+5}" text-anchor="end">${peak}</text><text x="${left-7}" y="${top+innerH+4}" text-anchor="end">0</text></svg><p class="chart-note">The histogram ignores coordinate order and shows how the values are distributed. It is useful for spotting concentration, spread, skew, and outliers—not for assigning meaning to individual dimensions.</p></figure>`;
}
function matrixHeatmap(values){
  if(!finiteMatrix(values))return '';
  const rows=values.length,cols=values[0].length,rGroups=Math.min(rows,32),cGroups=Math.min(cols,96),width=760,height=300,left=66,right=16,top=18,bottom=42,innerW=width-left-right,innerH=height-top-bottom;
  const cells=[];let bound=1e-12;
  for(let rg=0;rg<rGroups;rg++)for(let cg=0;cg<cGroups;cg++){
    const r0=Math.floor(rg*rows/rGroups),r1=Math.max(r0+1,Math.floor((rg+1)*rows/rGroups)),c0=Math.floor(cg*cols/cGroups),c1=Math.max(c0+1,Math.floor((cg+1)*cols/cGroups));let sum=0,n=0;
    for(let r=r0;r<r1;r++)for(let c=c0;c<c1;c++){sum+=values[r][c];n++;}const mean=sum/n;bound=Math.max(bound,Math.abs(mean));cells.push({rg,cg,r0,r1,c0,c1,mean});
  }
  const rects=cells.map(cell=>{const alpha=.08+.82*Math.abs(cell.mean)/bound,color=cell.mean>=0?`rgba(33,94,150,${alpha})`:`rgba(179,84,0,${alpha})`,x=left+cell.cg*innerW/cGroups,y=top+cell.rg*innerH/rGroups;return `<rect x="${x}" y="${y}" width="${innerW/cGroups+.2}" height="${innerH/rGroups+.2}" fill="${color}"><title>Row${cell.r1-cell.r0>1?'s':''} ${cell.r0}${cell.r1-cell.r0>1?'–'+(cell.r1-1):''}, coordinate${cell.c1-cell.c0>1?'s':''} ${cell.c0}${cell.c1-cell.c0>1?'–'+(cell.c1-1):''}: mean ${num(cell.mean,7)}</title></rect>`;}).join('');
  return `<figure class="numeric-chart heatmap"><figcaption>Matrix heatmap</figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Diverging heatmap of matrix values">${rects}<text x="${left}" y="${height-11}">coordinate 0</text><text x="${width-right}" y="${height-11}" text-anchor="end">coordinate ${cols-1}</text><text x="${left-8}" y="${top+5}" text-anchor="end">row 0</text><text x="${left-8}" y="${top+innerH}" text-anchor="end">row ${rows-1}</text></svg><div class="legend"><span><i class="negative-swatch"></i>negative</span><span><i class="positive-swatch"></i>positive</span><span>stronger colour = larger magnitude</span></div><p class="chart-note">Each cell is one value when the matrix is small. Large matrices are grouped and each displayed cell is the group mean. Use the exact values below for calculations.</p></figure>`;
}
function numericView(values){
  if(finiteVector(values))return statsLine(values)+`<div class="chart-grid">${vectorProfile(values)}${histogram(values)}</div>`;
  if(finiteMatrix(values))return statsLine(values)+matrixHeatmap(values);
  return '<p class="hint">No graph is shown because this value is not a one- or two-dimensional finite numeric array.</p>';
}
function namedNumericViews(data,names){return names.filter(name=>finiteVector(data[name])||finiteMatrix(data[name])).map(name=>`<details class="numeric-field"><summary>${esc(name)} — visual view</summary>${numericView(data[name])}</details>`).join('');}
const stamp = d => `<p class="stamp">Run ${esc(d.run_id || d.probe_id || 'toy')} · ${esc(d.created_utc || '')}${d.prompt!==undefined ? `<br>Prompt: <code>${str(d.prompt)}</code>` : ''}${d.settings ? `<br>Settings: ${esc(JSON.stringify(d.settings))}` : ''}</p>`;
const checks = list => table(['Check','Result','Max absolute error','Tolerance (absolute / relative)'],(list||[]).map(c=>[esc(c.name),`<span class="${c.passed?'pass':'fail'}">${c.passed?'PASS':'FAIL'}</span>`,num(c.max_abs_error),c.atol===undefined?'—':`${num(c.atol)} / ${num(c.rtol)}`]));
const chips = tokens => tokens.map(t=>`<span class="token" title="${esc(t.piece||'')}">#${t.position} · ID ${t.id} <code>${str(t.text)}</code>${t.special?' [special]':''}</span>`).join('');
function showTab(index) {
  state.tab=Math.max(0,Math.min(tabs.length-1,index));
  tabs.forEach((b,i)=>{ $(b.dataset.tab).hidden=i!==state.tab; if(i===state.tab)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current'); });
  $('previous').disabled=state.tab===0; $('next').disabled=state.tab===tabs.length-1;
}
function provenance(p) {
  if(!p)return;
  state.loaded=true;
  $('model-status').textContent=`${p.checkpoint} · ${p.model_kind} · ${p.layers} blocks · ${p.width} dimensions · ${p.query_heads} Q / ${p.kv_heads} KV heads · ${num(p.parameter_count,12)} parameters · ${p.dtype} ${p.device}`;
  $('provenance').innerHTML=detail('Complete runtime identity and hashes',p);
  for(const id of ['inspect-layer','ko-layer','compare-layer','jac-layer']) {$(id).max=p.layers; if(Number($(id).value)>p.layers)$(id).value=p.layers;}
  for(const id of ['inspect-head','compare-head'])$(id).max=p.query_heads;
  $('jac-coordinate').max=p.width-1;
}
function record(d) {
  provenance(d.provenance);
  state.records.push(d);
  if(state.records.length>10)state.records.shift();
  stale();
}
function stale(){ $('stale').hidden=!state.records.some(d=>d.prompt!==undefined && d.prompt!==prompt()); }
async function request(path,body) {
  const response=await fetch(`/api/${path}`,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const text=await response.text(); let data;
  try{data=JSON.parse(text);}catch{throw Error(`Server returned ${response.status}: ${text.slice(0,180)}`);}
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));
  return data;
}
async function action(fn) {
  if(state.busy)return;
  state.busy=true; $('error').textContent='';
  document.querySelectorAll('[data-model]').forEach(b=>b.disabled=true);
  document.body.setAttribute('aria-busy','true');
  try {await fn();} catch(e){$('error').textContent=e.message;} finally {
    state.busy=false; document.querySelectorAll('[data-model]').forEach(b=>b.disabled=false); document.body.removeAttribute('aria-busy');
  }
}
function on(id,fn){$(id).addEventListener('click',()=>action(fn));}
function download(filename,data){
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function tokenize(){const p=prompt();const d=await request(`tokenize?prompt=${encodeURIComponent(p)}`);if(p===prompt())$('tokens').innerHTML=chips(d.tokens);}
async function loadModel(){const d=await request('health?load=true');provenance(d.provenance);await tokenize();}
async function predict(){const d=await request('predict',{prompt:prompt()});state.prediction=d;record(d);if(d.prompt===prompt())$('tokens').innerHTML=chips(d.tokens);d.named_tokens.forEach(t=>state.names.set(t.id,t.text));state.draws.clear();renderDistribution();}
function distributions(){
  if(!state.prediction)throw Error('Run Analyze next-token distribution first.');
  const z=state.prediction.logits;
  return {raw:M.softmax(z,1),sample:M.softmax(z,temperature()),order:M.order(z)};
}
function tokenLabel(id){return state.names.has(id)?str(state.names.get(id)):`ID ${id}`;}
function renderDistribution(){
  $('temperature-value').textContent=temperature().toFixed(2);
  $('generation-temp').textContent=`New generation will use T=${temperature()}. Existing generation results keep the temperature recorded in their settings.`;
  if(!state.prediction)return;
  const {raw,sample,order}=distributions(),d=state.prediction;
  $('distribution').innerHTML=stamp(d)+`<p>Full vocabulary: ${d.vocab_size.toLocaleString()} tokens · sampling entropy ${num(M.entropy(sample))} nats · uniform reference ${pct(1/d.vocab_size)} per token.</p>`+table(['Rank','Vocabulary ID','Token','Logit','Model p (T=1)','Sampling p (current T)'],order.slice(0,20).map((id,i)=>[i+1,id,tokenLabel(id),num(d.logits[id],8),pct(raw[id]),pct(sample[id])]))+`<p class="hint">Top 20 shown. Sampling includes all ${d.vocab_size.toLocaleString()} tokens. Display rounding never feeds back into the calculation.</p>`;
  const palette=['#175b9b','#8f4c00','#427e39','#924a92','#3d7680','#775394','#647631','#ad3943'];
  const top=order.slice(0,8),tail=Math.max(0,1-top.reduce((s,id)=>s+sample[id],0));
  $('number-line').innerHTML=`<div class="number-line" aria-label="Sampling probability intervals">${top.map((id,i)=>`<span style="width:${sample[id]*100}%;background:${palette[i]}" title="${esc(state.names.get(id)||String(id))}: ${pct(sample[id])}"></span>`).join('')}<span style="width:${tail*100}%;background:#84919d" title="Remaining vocabulary: ${pct(tail)}"></span></div><div class="legend">${top.map((id,i)=>`<span><i style="background:${palette[i]}"></i>${tokenLabel(id)} ${pct(sample[id])}</span>`).join('')}<span><i style="background:#84919d"></i>Tail ${pct(tail)}</span></div>`;
  renderDraws();
}
function renderDraws(){const rows=Array.from(state.draws).sort((a,b)=>b[1]-a[1]),total=rows.reduce((s,r)=>s+r[1],0);$('dart-result').innerHTML=total?`<p>${total} draws at T=${temperature()}. Expected frequencies are probabilities, not guaranteed counts.</p>`+table(['ID','Token','Count','Observed frequency'],rows.map(([id,n])=>[id,tokenLabel(id),n,pct(n/total)])):'';}
async function darts(count){const {sample,order}=distributions(),c=M.cumulative(sample,order);for(let i=0;i<count;i++){const id=order[M.draw(c,Math.random())];state.draws.set(id,(state.draws.get(id)||0)+1);}for(const id of state.draws.keys())if(!state.names.has(id)){const d=await request(`decode?id=${id}`);state.names.set(id,d.text);}renderDraws();}
function vector(name,v){return `<details class="vector"><summary>${esc(name)} · shape [${v.shape.join(', ')}] · L2 norm ${num(v.l2_norm)}</summary><p class="hint"><strong>Shape</strong> gives the array dimensions. <strong>L2 norm</strong> is its Euclidean magnitude, √Σx²; it does not measure meaning or importance.</p>${numericView(v.values)}<details class="raw-values"><summary>Exact numeric values</summary><pre>${esc(JSON.stringify(v.values,null,2))}</pre></details></details>`;}
function renderInspection(d){
  const u=d.unembedding;
  $('inspection').innerHTML=stamp(d)+`<p>Query head ${d.query_head} shares KV head ${d.shared_kv_head}. Head dimension ${d.head_dim}; score divisor √d = ${num(d.scale_divisor)}.</p>`+checks(d.checks)+
  '<h3>Residual stream and gated MLP</h3>'+Object.entries(d.vectors).map(([k,v])=>vector(k,v)).join('')+
  '<h3>Full-input projection and RoPE</h3>'+namedNumericViews(d.q_coordinate_0_projection,['weights','input','products'])+detail('Every input × weight product for coordinate 0 of this query',d.q_coordinate_0_projection)+
  ['query_before_rope','query_after_rope','keys_before_rope','keys_after_rope','values','rope_cos','rope_sin'].map(k=>vector(k,d[k])).join('')+
  '<h3>One attention row, before and after masking</h3>'+table(['Key position / token','q · k','Scaled score','Mask','Weight','× uniform','||V||','||weight × V||','||projected contribution||'],d.attention_row.map(r=>[`${r.position} ${str(r.token)}`,num(r.dot_product),num(r.scaled_score),r.masked?'Future: −∞ conceptually':'Allowed',pct(r.weight),num(r.multiple_of_uniform),num(r.value_norm),num(r.weighted_value_norm),num(r.projected_contribution_norm)]))+
  '<p>Projected contributions are vectors: their norms need not add. Their vector sum is this head’s residual write, not the complete multi-head write.</p>'+['weighted_values','head_output','projected_position_contributions','head_residual_write'].map(k=>vector(k,d[k])).join('')+
  '<h3>Final output readout at this query position</h3>'+`<p>Token ${str(u.target_text)} (ID ${u.target_id}) · dot product ${num(u.dot_product,8)} · model logit ${num(u.model_logit,8)} · cosine ${num(u.cosine)} · model probability ${pct(u.probability_T1)}.</p>`+['raw_final_state','normalized_final_state','weight_row','dot_terms'].map(k=>vector(k,u[k])).join('');
}
function renderGeneration(d){$('generation-result').innerHTML=stamp(d)+`<p>${d.forward_passes} forward passes · ${d.unique_outputs} distinct outputs across ${d.results.length} runs. Effective base seed: ${d.settings?.base_seed ?? 'not reported'}.</p>`+d.results.map((r,i)=>`<h3>Run ${i+1} · seed ${r.seed}</h3><pre>${esc(r.full_text)}</pre><p>Full sequence decoded together to preserve byte-level boundaries. Continuation IDs: ${esc(JSON.stringify(r.generated_ids||[]))}.</p><p>${num(r.seconds)} s · stopped: ${esc(r.stop_reason)}</p>`+(r.steps.length?table(['Step','Prefix length / processed positions','Selected ID / token','Rank','Model p (T=1)','Sampling p','Prefix and five leading candidates'],r.steps.map(s=>[s.step,`${s.context_length} / ${s.processed_positions}`,`${s.chosen.id} ${str(s.chosen.text)}`,s.chosen.rank,pct(s.chosen.model_probability),pct(s.chosen.sampling_probability),detail('Exact prefix and candidate probabilities',{context:s.context,context_ids:s.context_ids,candidates:s.candidates})])):'<p class="hint">Detailed per-step trace is recorded for run 1; all runs preserve their generated token IDs in the export.</p>')).join('');}
function attentionRow(tokens,weights){return table(['Position','Token / ID','Attention weight'],tokens.map((t,i)=>[i,`${str(t.text)} / ${t.id}`,pct(weights[i])]));}
function renderComparison(d){$('comparison-result').innerHTML=stamp(d)+`<p>Prompt B: <code>${str(d.settings.prompt_b)}</code></p><h3>Tokenization</h3><p>A (${d.tokens.length} positions):</p><div class="tokens">${chips(d.tokens)}</div><p>B (${d.tokens_b.length} positions):</p><div class="tokens">${chips(d.tokens_b)}</div><p>Full-distribution TV ${pct(d.metrics.tv)} · JS ${num(d.metrics.js_nats)} nats. ${esc(d.warning)}</p>`+table(['ID','Token','p(A)','p(B)','B − A (pp)','Logit A','Logit B'],d.candidates.map(r=>[r.id,str(r.text),pct(r.probability_a),pct(r.probability_b),num(r.delta_pp),num(r.logit_a),num(r.logit_b)]))+detail('Last-position residual norms and cosine by raw stage',d.layers)+'<div class="two"><div><h3>A: selected head’s final query</h3>'+attentionRow(d.tokens,d.last_row_a)+'</div><div><h3>B: selected head’s final query</h3>'+attentionRow(d.tokens_b,d.last_row_b)+'</div></div>';}
function renderAttention(){
  const d=state.attention;if(!d)return;
  const li=Number($('attn-layer').value),mode=$('attn-head').value,ratio=$('attn-scale').value==='ratio',a=M.aggregate(d.attentions[li],mode),n=a.length;
  const values=a.flatMap((r,i)=>r.slice(0,i+1).map(x=>ratio?x*(i+1):x)),max=Math.max(...values,1e-12);
  const rows=a.map((r,i)=>`<tr><th>#${i} ${str(d.tokens[i].text)}</th>${r.map((w,j)=>j>i?'<td class="masked" aria-label="Causally masked">—</td>':`<td><button class="cell" data-i="${i}" data-j="${j}" style="background:rgba(34,113,179,${0.08+0.67*(ratio?w*(i+1):w)/max})" title="Query ${i}, key ${j}; ${pct(w)}; ${num(w*(i+1))}× uniform">${ratio?num(w*(i+1),3)+'×':num(100*w,3)+'%'}</button></td>`).join('')}</tr>`);
  $('attention-result').innerHTML=stamp(d)+`<p>Block ${li+1} · ${mode==='mean'?'Mean of heads':mode==='max'?'Cellwise maximum envelope (not a distribution)':`Head ${Number(mode)+1}`} · ${ratio?'Multiple of uniform':'Raw probability'} · colour range 0–${ratio?num(max)+'×':pct(max)} (autoscaled for this grid).</p><div class="table-scroll"><table class="matrix"><thead><tr><th>Query ↓ / Key →</th>${d.tokens.map(t=>`<th>#${t.position}<br>${str(t.text)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div><p class="hint">Click a cell for exact values. Axes keep repeated token positions separate.</p>`;
  $('cell-detail').textContent='';
  const profile=d.attentions.map((heads,l)=>{const row=M.aggregate(heads,mode).at(-1);return {layer:l+1,first:row[0],self:row[n-1],other:n>2?Math.max(...row.slice(1,-1)):null};});
  renderProfile(profile,n,ratio);
}
function renderProfile(rows,n,ratio){
  const series=[['first','Position 0','#175b9b'],['self','Self position','#b35400'],['other','Largest other','#4e7b31']];
  const factor=ratio?n:100,positive=rows.flatMap(r=>series.map(([k])=>r[k]===null?0:r[k]*factor)).filter(x=>x>0);
  const lo=Math.min(-2,Math.floor(Math.log10(Math.min(...positive)))),hi=Math.max(lo+1,Math.ceil(Math.log10(Math.max(...positive))));
  const x=i=>70+(rows.length===1?0:i/(rows.length-1)*600),y=v=>260-(Math.log10(v)-lo)/(hi-lo)*230;
  const ticks=[];for(let e=lo;e<=hi;e++)ticks.push(`<line x1="70" x2="670" y1="${y(10**e)}" y2="${y(10**e)}" stroke="#dae2e8"/><text x="62" y="${y(10**e)+4}" text-anchor="end">${num(10**e)}${ratio?'×':'%'}</text>`);
  const paths=series.map(([key,,color])=>{let path='',active=false;rows.forEach((r,i)=>{const v=r[key]===null?0:r[key]*factor;if(v<=0){active=false;return;}path+=`${active?'L':'M'}${x(i)},${y(v)} `;active=true;});return `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"/>`+rows.map((r,i)=>r[key]>0?`<circle cx="${x(i)}" cy="${y(r[key]*factor)}" r="2" fill="${color}"><title>Block ${r.layer}: ${num(r[key]*factor)}${ratio?'×':'%'}</title></circle>`:'').join('');});
  $('sink-profile').innerHTML=`<svg viewBox="0 0 730 310" role="img" aria-label="Logarithmic bottom-row attention profile across blocks">${ticks.join('')}${paths.join('')}<text x="70" y="280">Block 1</text><text x="670" y="280" text-anchor="end">Block ${rows.length}</text><text x="370" y="303" text-anchor="middle">${ratio?'Multiple of uniform':'Raw probability'} · logarithmic vertical scale</text></svg><div class="legend">${series.map(([,name,color])=>`<span><i style="background:${color}"></i>${name}</span>`).join('')}</div>${n===1?'<p>With one token, position 0 is also self; those lines coincide.</p>':''}`;
  $('sink-table').innerHTML='<details><summary>Exact bottom-row profile values</summary>'+table(['Block','Position 0 (raw)','Self (raw)','Largest other (raw)','Position 0 / uniform'],rows.map(r=>[r.layer,pct(r.first),pct(r.self),pct(r.other),num(r.first*n)+'×']))+'</details>';
}
function renderIntervention(d){$('intervention-result').innerHTML=stamp(d)+`<p>Target ${str(d.target_text)} vs ${str(d.contrast_text)}. Baseline target p ${pct(d.baseline.target_probability)}; baseline logit difference ${num(d.baseline.logit_difference)}.</p>`+checks(d.checks)+table(['Head','Baseline position-0 share','New top token','Top changed?','Target p','Δ target (pp)','Target rank','TV','JS (nats)','Δ logit difference'],d.rows.map(r=>[esc(r.head),pct(r.sink_share),str(r.new_top[0].text),r.top1_changed?'Yes':'No',pct(r.target_probability),num(r.delta_probability_pp),r.target_rank,pct(r.tv),num(r.js_nats),num(r.delta_logit_difference)]))+`<p>${esc(d.definition)} ${d.forward_passes} forward passes.</p>`+detail('Full intervention values, including KL and candidates',d.rows);}
function renderLenses(d){$('lenses-result').innerHTML=stamp(d)+`<p>Tracking final top token ${str(d.target_text)} (ID ${d.target_id}). All probabilities T=1; final-target ranks use logits.</p>`+checks(d.checks)+table(['Stage','Raw / post-norm L2','Top candidate','p(top)','Final target rank','p(final target)','KL(final || lens)','Affine probe KL, if fitted'],d.stages.map(s=>[esc(s.label),num(s.vector_norm),str(s.top[0].text),pct(s.top[0].model_probability),s.target_rank,pct(s.target_probability),num(s.metrics_to_final.kl_base_to_changed_nats),s.probe?num(s.probe.metrics_to_final.kl_base_to_changed_nats):'Not fitted / not applicable']))+detail('Full candidates and readout metrics',d.stages);}
tabs.forEach((b,i)=>b.addEventListener('click',()=>showTab(i)));
$('previous').addEventListener('click',()=>showTab(state.tab-1));$('next').addEventListener('click',()=>showTab(state.tab+1));
$('mode').addEventListener('change',()=>document.body.classList.toggle('guided',$('mode').value==='guided'));
let tokenTimer;
$('prompt').addEventListener('input',()=>{stale();clearTimeout(tokenTimer);$('tokens').textContent='Tokenization pending; load or predict to refresh.';if(state.loaded)tokenTimer=setTimeout(()=>{if(!state.busy)action(tokenize);},450);});
$('temperature').addEventListener('input',()=>{state.draws.clear();renderDistribution();$('rank-result').textContent='';});
on('load-model',loadModel);on('predict',predict);
on('find-rank',async()=>{const {raw,sample,order}=distributions(),rank=int('rank');if(rank<1||rank>order.length)throw Error(`Rank must be 1–${order.length}`);const id=order[rank-1];if(!state.names.has(id)){const d=await request(`decode?id=${id}`);state.names.set(id,d.text);}$('rank-result').innerHTML=`<p>Rank ${rank}: ${tokenLabel(id)} · ID ${id} · logit ${num(state.prediction.logits[id],9)} · model ${pct(raw[id])} · sampling ${pct(sample[id])}</p>`;});
on('dart',()=>darts(1));on('dart100',()=>darts(100));on('clear-darts',()=>{state.draws.clear();renderDraws();});
on('toy-attention',async()=>{const d=await request('toy-attention');record(d);$('toy-attention-result').innerHTML=`<h3>${esc(d.label)}</h3><p>This toy uses deliberately small invented matrices so every multiplication can be checked by hand. The heatmaps show patterns; the table and exact values remain authoritative.</p>`+namedNumericViews(d,['q','k','v','scores','weights','output'])+detail('All inputs, scores, causal mask, weights, and weighted sums',d)+table(['Position','Attention row','Output vector','Row sum'],d.weights.map((r,i)=>[i,esc(r.map(v=>num(v)).join(', ')),esc(d.output[i].map(v=>num(v)).join(', ')),num(d.row_sums[i])]))+'<p>For query position 0, the causal mask allows only key/value position 0, so the output equals that position’s value vector: [2, 0]. For later rows, multiply each value vector by its displayed weight and sum coordinate by coordinate.</p>';});
on('inspect',async()=>{const d=await request('inspect',{prompt:prompt(),layer:int('inspect-layer'),head:int('inspect-head'),position:optional('inspect-position'),target_id:optional('inspect-target')});record(d);renderInspection(d);});
on('generate',async()=>{const d=await request('generate',{prompt:prompt(),n_tokens:int('n-tokens'),runs:int('runs'),temperature:temperature(),seed:optional('seed'),use_cache:$('use-cache').checked,stop_eos:$('stop-eos').checked});record(d);renderGeneration(d);});
const examples={country:['The capital of Japan is called','The capital of France is called'],fiction:['The capital of Japan is called','In this fictional world, the capital of Japan is Osaka. The capital of Japan is called'],order:['The dog chased the cat because','The cat chased the dog because'],space:['Paris',' Paris'],distractor:['The capital of Japan is called','Bananas are yellow. The capital of Japan is called']};
on('set-example',()=>{const [a,b]=examples[$('comparison-example').value];$('prompt').value=a;$('prompt-b').value=b;stale();$('tokens').textContent='Prompt changed; load or predict to refresh tokenization.';});
on('compare',async()=>{const ids=$('compare-targets').value.trim()?$('compare-targets').value.split(',').map(s=>{if(!/^\d+$/.test(s.trim()))throw Error('Use comma-separated nonnegative token IDs');return Number(s.trim());}):[];const d=await request('compare',{prompt_a:prompt(),prompt_b:$('prompt-b').value,target_ids:ids,layer:int('compare-layer'),head:int('compare-head')});record(d);renderComparison(d);});
on('load-attention',async()=>{const d=await request('attention',{prompt:prompt()});record(d);state.attention=d;$('attn-layer').innerHTML=Array.from({length:d.n_layers},(_,i)=>`<option value="${i}">Block ${i+1}</option>`).join('');$('attn-head').innerHTML='<option value="mean">Mean of heads</option><option value="max">Maximum envelope</option>'+Array.from({length:d.n_heads},(_,i)=>`<option value="${i}">Head ${i+1}</option>`).join('');renderAttention();});
for(const id of ['attn-layer','attn-head','attn-scale'])$(id).addEventListener('change',renderAttention);
$('attention-result').addEventListener('click',event=>{const b=event.target.closest('.cell');if(!b||!state.attention)return;const i=Number(b.dataset.i),j=Number(b.dataset.j),heads=state.attention.attentions[Number($('attn-layer').value)],row=M.aggregate(heads,$('attn-head').value)[i],w=row[j];$('cell-detail').textContent=`Query #${i} → key #${j}: raw ${pct(w)}; uniform reference ${pct(1/(i+1))}; ${num(w*(i+1),9)}× uniform; displayed row sum ${num(row.reduce((a,b)=>a+b,0),9)}. ${$('attn-head').value==='max'?'Maximum envelope rows need not sum to one.':''}`;});
on('intervene',async()=>{const d=await request('intervene',{prompt:prompt(),layer:int('ko-layer'),scope:$('ko-scope').value,target_id:optional('ko-target'),contrast_id:optional('ko-contrast'),uniform:$('ko-kind').value==='uniform'});record(d);renderIntervention(d);});
on('training-trace',async()=>{const d=await request('training-trace',{prompt:prompt()});record(d);$('training-result').innerHTML=stamp(d)+`<p>Mean loss ${num(d.mean_loss_nats)} nats · perplexity ${num(d.perplexity)} · no weights updated.</p>`+table(['Position','Input token','Next-token target / ID','Target p','−ln p (nats)','∂ single-position loss / ∂ target logit'],d.rows.map(r=>[r.position,str(r.input_token),`${str(r.target_text)} / ${r.target_id}`,pct(r.target_probability),num(r.loss_nats),num(r.d_loss_d_target_logit_unaveraged)]))+`<p>${esc(d.note)} For the mean-loss gradient, divide each row’s single-position gradient by the number of targets.</p>`;});
on('toy-training',async()=>{const d=await request(`toy-training?steps=${int('training-steps')}&learning_rate=${Number($('learning-rate').value)}`);record(d);$('toy-training-result').innerHTML=`<h3>${esc(d.label)}</h3><p>Input ${esc(JSON.stringify(d.input))}; target ${esc(d.labels[d.target_id])}; learning rate ${d.learning_rate}. Gradient = (p − one_hot(target)) ⊗ input. Update = weights − learning_rate × gradient.</p>`+table(['Step','Loss before','Loss after','Complete calculation'],d.steps.map(s=>[s.step,num(s.loss,8),num(s.next_loss,8),detail('Weights, logits, probabilities, gradients, update',s)]));});
on('lenses-run',async()=>{const d=await request('lenses',{prompt:prompt()});record(d);renderLenses(d);});
on('probe-fit',async()=>{const lines=id=>$(id).value.split(/\r?\n/).filter(s=>s.trim().length>0);const d=await request('probe-fit',{train_prompts:lines('probe-train'),test_prompts:lines('probe-test'),ridge:Number($('probe-ridge').value)});record(d);$('probe-result').innerHTML=stamp(d)+`<p>${esc(d.method)}. ${esc(d.limitations)}</p>`+table(['Raw stage','Held-out raw-lens KL','Held-out probe KL','Constant-mean KL','Relative state error'],d.scores.map(s=>[s.stage,num(s.raw_lens_kl),num(s.probe_kl),num(s.constant_final_mean_kl),num(s.state_relative_l2_error)]))+detail('Exact corpus and per-example held-out scores',d)+'<p>Rerun layer readouts to compare this probe on Prompt A.</p>';});
on('jacobian-run',async()=>{const d=await request('jacobian',{prompt:prompt(),layer:int('jac-layer'),coordinate:int('jac-coordinate'),epsilon:Number($('jac-epsilon').value)});record(d);$('jacobian-result').innerHTML=stamp(d)+checks(d.checks)+`<p>JVP relative difference ${num(d.jvp_relative_error)}; local prediction relative error ${num(d.approximation_relative_error)}. ${esc(d.warning)}</p>`+namedNumericViews(d,['base','jvp','finite_difference','actual_perturbed','local_prediction'])+detail('All coordinate values: anchor, derivative, finite difference, perturbed state, approximation',d);});
on('verify',async()=>{const d=await request('verify',{prompt:prompt()});record(d);state.verification=d;$('verification-result').innerHTML=stamp(d)+`<h3 class="${d.passed?'pass':'fail'}">${d.passed?'PASS':'FAIL'}: ${d.passed_count} / ${d.check_count}</h3><p>${esc(d.scope)}</p>`+checks(d.checks);});
on('export-verification',()=>{if(!state.verification)throw Error('Run model-backed verification first.');download('model_verification.json',state.verification);});
on('save-note',()=>{state.notes.push({created_utc:new Date().toISOString(),prompt_a:prompt(),prompt_b:$('prompt-b').value,temperature:temperature(),hypothesis:$('hypothesis').value,observation:$('observation').value,explanation:$('explanation').value,last_run_id:state.records.at(-1)?.run_id||state.records.at(-1)?.probe_id||null});$('note-status').textContent=`${state.notes.length} notes saved in page memory. Export before reloading.`;});
on('export-notes',()=>download('experiment_notes.json',state.notes));
on('export-session',()=>download('learning_lab_session.json',{app_version:'2.1.0',exported_utc:new Date().toISOString(),current_prompt:prompt(),temperature:temperature(),notes:state.notes,records:state.records,retention:'Most recent 10 experiment records; export regularly. Model weights and fitted probe coefficients are not included.'}));
showTab(0);renderDistribution();
