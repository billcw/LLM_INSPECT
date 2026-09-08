/* Shared browser mathematics. Also tested directly with Node's test runner. */
(function (root) {
  'use strict';
  function softmax(logits, temperature) {
    if (!logits.length || !Number.isFinite(temperature) || temperature < 0 || !Array.from(logits).every(Number.isFinite)) throw Error('Invalid logits or temperature');
    const p = new Float64Array(logits.length);
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    if (temperature === 0) { p[best] = 1; return p; }
    let sum = 0;
    for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - logits[best]) / temperature); sum += p[i]; }
    return p.map(x => x / sum);
  }
  function order(logits) { return Array.from(logits.keys()).sort((a,b) => logits[b]-logits[a] || a-b); }
  function cumulative(probs, ids) {
    let sum = 0;
    const c = ids.map(i => (sum += probs[i]));
    c[c.length-1] = 1; // roundoff at the endpoint; selection uses strict >
    return c;
  }
  function draw(c, r) {
    if (!Number.isFinite(r) || r < 0 || r >= 1 || !c.length) throw Error('Draw must be in [0, 1)');
    let lo = 0, hi = c.length-1;
    while (lo < hi) { const mid = Math.floor((lo+hi)/2); if (c[mid] > r) hi = mid; else lo = mid+1; }
    return lo;
  }
  function aggregate(heads, mode) {
    if (mode !== 'mean' && mode !== 'max') return heads[Number(mode)];
    return heads[0].map((row,i) => row.map((_,j) => mode === 'max'
      ? Math.max(...heads.map(h => h[i][j])) : heads.reduce((s,h) => s+h[i][j],0)/heads.length));
  }
  function escape(text) { return String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function label(text) { return JSON.stringify(text); }
  function entropy(p) { return Array.from(p).reduce((s,x) => x > 0 ? s-x*Math.log(x) : s, 0); }
  const api = {softmax, order, cumulative, draw, aggregate, escape, label, entropy};
  if (typeof module !== 'undefined') module.exports = api;
  root.LabMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
