/* Experiment console — Analysis tab.
 *
 * Ask questions about the experiments in plain language; the model answers by running
 * Python (arabic_eval.analysis, preloaded as `ae`) in a sandboxed worker on the server.
 * Every step's code, output, tables and Plotly figures are shown; numbers of the answer
 * that no step printed are highlighted. Server side: src/arabic_eval/tools/analysis_agent.py
 * and the /api/analysis/* routes of debugger/serve_experiment_console.py.
 *
 * Loaded by experiment_console.html after its inline script: it uses the page's helpers
 * ($, $$, h, esc, api, openModal, mdToHtml) and CSS tokens. Libraries (cdnjs, SRI-pinned,
 * fetched on first open): marked + DOMPurify (answers), Prism's python grammar (code; the
 * page already loads prism-core), plotly.js 4.1.1 = the version plotly.py 7.1 writes for.
 */
'use strict';
(() => {
const LIB = {
  marked: {src: 'https://cdnjs.cloudflare.com/ajax/libs/marked/18.0.13/lib/marked.umd.min.js',
           sri: 'sha512-kHavuYjOa82OKvUxD8j04s+kMIH84FmETnuIgj0yFyY9LWU10sXNwU54Iozpewat8VxAzH4aXiwTEZMRrpwfBw=='},
  purify: {src: 'https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.4.16/purify.min.js',
           sri: 'sha512-flQmhkXNRQ3iUfvtdCobtpMjCb6kE+zLnTiAulAQJ3WykVfy84JvCMOo6vhwWXHTXjR25h9Zj+LyGFRWKjRoag=='},
  prismPy: {src: 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.30.0/components/prism-python.min.js',
            sri: 'sha512-AKaNmg8COK0zEbjTdMHJAPJ0z6VeNqvRvH4/d5M4sHJbQQUToMBtodq4HaV4fa+WV2UTfoperElm66c9/8cKmQ=='},
  plotly: {src: 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/4.1.1/plotly.min.js',
           sri: 'sha512-2SAjXeni/1+3u+HXGF7+CYi8nJ5WYF/2cBFcrgIv+4Ie9W5rpoXVnHGAM0tm0HBg+sKDj8+CnFEQwCWxA4uqig=='},
};
/* dataviz reference palette; the dark steps were validated against the console's dark panel #141D19 */
const PAL_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
const PAL_DARK  = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const SUGGEST = [
  'Which cells exist, and under which eval rules?',
  'Compare the gemma4_31b judge scores of the three v5 arms, paired, with CIs',
  'On arabic_exam, where does AraRooPat v5 lose to BPE-16K v5? Break it down by sub-config',
  'Plot the loop-stop rate against the judge mean for the current free-form cells',
  'How did the Phase 3 eval loss evolve for the v5 arms? Plot it',
  'Show three free-form prompts where AraRooPat v5 and native v5 disagree most',
];
const AN = {inited: false, eps: [], sandbox: null, ep: null, sessions: [], sid: null, sess: null, busy: false,
            abort: null, scope: [], exps: [], codeMode: false, figs: [], live: null, pollT: null, lastTokens: null};

/* ---------------------------------------------------------------- libraries */
const loaded = {};
function loadScript(key){
  if (loaded[key]) return loaded[key];
  const {src, sri} = LIB[key];
  loaded[key] = new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.integrity = sri; s.crossOrigin = 'anonymous'; s.referrerPolicy = 'no-referrer';
    s.onload = () => res(true); s.onerror = () => { loaded[key] = null; rej(new Error('could not load ' + src)); };
    document.head.append(s);
  });
  return loaded[key];
}
async function ensureText(){
  const waitPrism = new Promise(r => { let n = 0; const t = setInterval(() => { if (window.Prism || ++n > 40){ clearInterval(t); r(); } }, 50); });
  await Promise.allSettled([loadScript('marked'), loadScript('purify'), waitPrism.then(() => window.Prism ? loadScript('prismPy') : null)]);
}
const ensurePlotly = () => loadScript('plotly');

/* ---------------------------------------------------------------- rendering helpers */
const PY_BLOCK = /```(?:python|py)[ \t]*\n[\s\S]*?```/gi;
const proseOf = t => String(t ?? '').replace(PY_BLOCK, '').trim();
function md(text){
  let html;
  if (window.marked && window.DOMPurify){
    html = window.DOMPurify.sanitize(window.marked.parse(String(text ?? ''), {gfm: true, breaks: false}));
  } else html = mdToHtml(text);                           // the page's minimal renderer (no tables) as a fallback
  const box = h('div', {class: 'an-md', html});
  box.querySelectorAll('p,li,td,th,h1,h2,h3,h4,blockquote').forEach(e => e.setAttribute('dir', 'auto'));
  box.querySelectorAll('a[href]').forEach(a => { a.target = '_blank'; a.rel = 'noopener noreferrer'; });
  box.querySelectorAll('table').forEach(t => t.classList.add('res'));
  return box;
}
function hlPy(code){
  return (window.Prism && Prism.languages && Prism.languages.python) ? Prism.highlight(code, Prism.languages.python, 'python') : esc(code);
}
const fmtK = n => n == null ? '—' : n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'K' : String(n);
const fmtS = s => s == null ? '' : s < 60 ? s.toFixed(1) + ' s' : Math.floor(s / 60) + ' min ' + Math.round(s % 60) + ' s';
function isDark(){ const t = document.documentElement.dataset.theme; return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; }
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
/** mark the numbers the provenance check could not find in any step output */
function markUnverified(root, unverified){
  const texts = [...new Set((unverified || []).map(u => u.text).filter(Boolean))];
  if (!texts.length) return;
  const rx = new RegExp(texts.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const n of nodes){
    if (n.parentElement.closest('code,pre,mark')) continue;
    const s = n.nodeValue; rx.lastIndex = 0; if (!rx.test(s)) continue;
    const frag = document.createDocumentFragment(); let last = 0; rx.lastIndex = 0; let m;
    while ((m = rx.exec(s))){
      frag.append(s.slice(last, m.index));
      frag.append(h('mark', {class: 'an-unv', title: 'no output of this session printed this number (computed or recalled by the model — check it)'}, m[0]));
      last = m.index + m[0].length;
    }
    frag.append(s.slice(last)); n.replaceWith(frag);
  }
}

/* ---------------------------------------------------------------- figures */
function themed(fig){
  const f = JSON.parse(JSON.stringify(fig));
  const dark = isDark();
  const surf = cssVar('--surface'), ink = cssVar('--ink'), ink2 = cssVar('--ink-2'), ink3 = cssVar('--ink-3'), line = cssVar('--line'), line2 = cssVar('--line-2');
  const map = {'#fcfcfb': surf, '#0b0b0b': ink, '#52514e': ink2, '#898781': ink3, '#e1e0d9': line, '#c3c2b7': line2};
  if (dark){ PAL_LIGHT.forEach((c, i) => { map[c] = PAL_DARK[i]; }); Object.assign(map, {black: ink, '#000': ink, '#000000': ink}); }
  const walk = o => {
    if (Array.isArray(o)) return o.map(walk);
    if (o && typeof o === 'object'){ for (const k of Object.keys(o)) o[k] = walk(o[k]); return o; }
    if (typeof o === 'string'){ const r = map[o.toLowerCase()]; return r === undefined ? o : r; }
    return o;
  };
  f.data = walk(f.data || []); f.layout = walk(f.layout || {});
  const L = f.layout;
  L.paper_bgcolor = surf; L.plot_bgcolor = surf; L.autosize = true;
  L.hoverlabel = Object.assign({}, L.hoverlabel, {bgcolor: surf, bordercolor: line2, font: {color: ink}});
  if (!L.height) L.height = 440;
  return f;
}
async function drawFigure(box){
  const fig = box._fig; if (!fig) return;
  try { await ensurePlotly(); }
  catch(e){ box.replaceChildren(h('div', {class: 'chip bad'}, 'plotly.js could not load: ' + e.message)); return; }
  const t = themed(fig);
  window.Plotly.react(box, t.data, t.layout, {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                                              toImageButtonOptions: {format: 'png', scale: 2, filename: box.dataset.name || 'figure'}});
}
function redrawAll(){ AN.figs = AN.figs.filter(b => b.isConnected); AN.figs.forEach(drawFigure); }

function displayEl(d){
  const wrap = h('div', {class: 'an-disp'});
  const dl = d.url ? h('a', {href: d.url + '&download=1', class: 'an-dl', title: 'download'}, '⤓') : null;
  if (d.kind === 'plotly'){
    // no header row: the figure carries its own title, and its toolbar downloads a PNG
    const box = h('div', {class: 'an-fig', 'data-name': (d.path || 'figure').replace(/\.plotly\.json$/, '')}, h('div', {class: 'empty'}, 'loading figure…'));
    wrap.append(box);
    fetch(d.url).then(r => r.json()).then(fig => { box.replaceChildren(); box._fig = fig; AN.figs.push(box); drawFigure(box); })
      .catch(e => box.replaceChildren(h('div', {class: 'chip bad'}, 'figure unavailable: ' + e.message)));
  } else if (d.kind === 'image'){
    wrap.append(h('div', {class: 'an-disp-h'}, h('span', {}, d.title || 'figure'), dl), h('img', {src: d.url, class: 'an-img', alt: d.title || 'figure'}));
  } else if (d.kind === 'table'){
    const more = d.n_rows > (d.rows || []).length ? ` · first ${(d.rows || []).length} shown` : '';
    wrap.append(h('div', {class: 'an-disp-h'}, h('span', {}, (d.title ? d.title + ' · ' : '') + `${d.n_rows} rows × ${d.n_cols} cols${more}`),
                  d.url ? h('a', {href: d.url + '&download=1', class: 'an-dl', title: 'the whole table as CSV'}, 'CSV ⤓') : null),
                tableEl(d.columns || [], d.rows || []));
  } else if (d.kind === 'markdown'){
    wrap.append(md(d.markdown || ''));
  } else wrap.append(h('pre', {class: 'an-pre'}, d.text || JSON.stringify(d)));
  return wrap;
}
function tableEl(cols, rows){
  const box = h('div', {class: 'an-tbl'});
  let sort = {i: -1, desc: false};
  const numeric = cols.map((_, i) => rows.length > 0 && rows.every(r => r[i] == null || typeof r[i] === 'number'));
  const render = () => {
    const rs = rows.slice();
    if (sort.i >= 0) rs.sort((a, b) => { const x = a[sort.i], y = b[sort.i]; if (x == null) return 1; if (y == null) return -1; const c = x < y ? -1 : x > y ? 1 : 0; return sort.desc ? -c : c; });
    const t = h('table', {class: 'res'},
      h('thead', {}, h('tr', {}, cols.map((c, i) => h('th', {class: numeric[i] ? 'num' : '', title: 'sort', onclick: () => { sort = {i, desc: sort.i === i ? !sort.desc : false}; render(); }},
        c + (sort.i === i ? (sort.desc ? ' ↓' : ' ↑') : ''))))),
      h('tbody', {}, rs.map(r => h('tr', {}, r.map((v, i) => h('td', {class: numeric[i] ? 'num' : '', dir: 'auto'},
        v == null ? '' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : String(+v.toFixed(6))) : String(v)))))));
    box.replaceChildren(t);
  };
  render();
  return box;
}

/* ---------------------------------------------------------------- turns */
function stepEl(st){
  const el = h('div', {class: 'an-step', 'data-step': st.n ?? ''});
  const prose = proseOf(st.reply);
  if (prose) el.append(md(prose));
  if (st.code){
    const det = h('details', {class: 'an-code'});
    det.append(h('summary', {}, h('span', {class: 'an-sn'}, `step ${st.n}`), h('span', {class: 'an-st'}), h('span', {class: 'spacer'}),
      h('button', {class: 'btn sm', title: 'copy the code', onclick: e => { e.preventDefault(); navigator.clipboard && navigator.clipboard.writeText(st.code); }}, 'copy'),
      h('button', {class: 'btn sm', title: 'put this code in the run-code box', onclick: e => { e.preventDefault(); setCodeMode(true, st.code); }}, 'edit & run')));
    det.append(h('pre', {class: 'an-src'}, h('code', {class: 'language-python', html: hlPy(st.code)})));
    el.append(det);
    el.append(h('div', {class: 'an-out'}));
    if (st.exec) fillExec(el, st.exec);
  }
  if (st.draft_answer){
    const n = ((st.provenance || {}).unverified || []).map(u => u.text);
    el.replaceChildren(h('details', {class: 'an-draft'},
      h('summary', {}, `draft answer held back — ${n.length} number${n.length === 1 ? '' : 's'} not printed by any step (${n.slice(0, 6).join(', ')}${n.length > 6 ? ', …' : ''}); the model was asked to compute them`),
      md(st.reply || '')));
  } else if (st.feedback && !st.code) el.append(h('div', {class: 'an-note'}, st.feedback));
  return el;
}
function setStepStatus(el, kind, text){
  const s = el.querySelector('.an-st'); if (!s) return;
  s.className = 'an-st chip ' + kind; s.textContent = text;
}
function fillExec(el, ex){
  const out = el.querySelector('.an-out'); if (!out) return;
  out.replaceChildren();
  if (ex.declined){ setStepStatus(el, 'warn', 'not run — declined'); return; }
  const ok = ex.ok;
  setStepStatus(el, ok ? 'good' : 'bad', (ok ? 'ok' : (ex.error && ex.error.ename) || 'failed') + (ex.elapsed != null ? ' · ' + fmtS(ex.elapsed) : ''));
  if (ex.restarted) out.append(h('div', {class: 'an-note'}, 'the Python worker was restarted before this step — earlier variables were lost'));
  if (ex.stdout) out.append(h('pre', {class: 'an-pre', dir: 'auto'}, ex.stdout.replace(/\n$/, '')));
  if (ex.result) out.append(h('pre', {class: 'an-pre an-val', dir: 'auto'}, ex.result));
  for (const d of ex.displays || []) out.append(displayEl(d));
  if (ex.error){
    const e = ex.error;
    out.append(h('details', {class: 'an-err'}, h('summary', {}, `${e.ename}: ${String(e.evalue || '').slice(0, 300)}`),
      e.traceback ? h('pre', {class: 'an-pre'}, e.traceback) : null));
  }
  if (ex.stderr && !ok) out.append(h('pre', {class: 'an-pre an-stderr'}, ex.stderr.slice(-1500)));
}
function answerEl(text, prov){
  const box = h('div', {class: 'an-answer'});
  const body = md(text);
  markUnverified(body, prov && prov.unverified);
  box.append(body);
  if (prov){
    const n = prov.checked || 0, u = (prov.unverified || []).length;
    box.append(h('div', {class: 'an-prov'}, u
      ? h('span', {class: 'chip warn', title: 'these numbers appear in no output of the session: ' + prov.unverified.map(x => x.text).join(', ')}, `⚠ ${u} of ${n} numbers not printed by any step`)
      : h('span', {class: 'chip good', title: 'every number of the answer appears in an output of the session (step outputs, tables, figure data, code or the prompt)'}, `✓ ${n} number${n === 1 ? '' : 's'} traced to outputs`)));
  }
  return box;
}
function footEl(t){
  const nSteps = (t.steps || []).filter(s => s.code).length;
  const u = t.usage || {};
  const bits = [t.status || '', `${nSteps} step${nSteps === 1 ? '' : 's'}`, t.elapsed_sec != null ? fmtS(t.elapsed_sec) : null,
                u.total_tokens ? `${fmtK(u.prompt_tokens)} in / ${fmtK(u.completion_tokens)} out tokens` : null, t.model || null].filter(Boolean);
  const cls = t.status === 'ok' ? '' : t.status === 'error' ? ' bad' : ' warn';
  return h('div', {class: 'an-foot' + cls}, bits.join(' · ') + (t.error ? ' — ' + t.error : ''));
}
function turnEl(t){
  const el = h('div', {class: 'an-turn'});
  if (t.kind === 'code'){
    const st = {n: (t.exec && t.exec.step) || '·', reply: '', code: t.code, exec: t.exec};
    el.append(h('div', {class: 'an-q code'}, h('span', {class: 'chip'}, 'code run by hand')));
    const s = stepEl(st); s.querySelector('details') && (s.querySelector('details').open = true);
    el.append(h('div', {class: 'an-steps'}, s));
    return el;
  }
  el.append(h('div', {class: 'an-q', dir: 'auto'}, t.user));
  const steps = h('div', {class: 'an-steps'});
  for (const st of t.steps || []) if (st.code || st.draft_answer || (st.reply && st.reply !== '(empty reply)' && st.feedback)) steps.append(stepEl(st));
  el.append(steps);
  if (t.answer) el.append(answerEl(t.answer, t.provenance));
  if (t.status && t.status !== 'running') el.append(footEl(t));
  return el;
}

/* ---------------------------------------------------------------- layout */
function build(){
  const v = $('#view-analysis');
  v.replaceChildren(
    h('div', {class: 'an-grid'},
      h('aside', {class: 'panel an-side'},
        h('div', {class: 'panel-h'}, h('h2', {}, 'sessions'), h('span', {class: 'spacer'}),
          h('button', {class: 'btn sm primary', id: 'an-new', title: 'a new conversation (a fresh Python worker)'}, '+ new')),
        h('div', {class: 'an-sessions', id: 'an-sessions'})),
      h('div', {class: 'panel an-main'},
        h('div', {class: 'panel-h an-head'},
          h('label', {class: 'k'}, 'model'), h('select', {id: 'an-ep'}), h('span', {id: 'an-ep-st', class: 'chip'}, '…'),
          h('span', {id: 'an-opts', class: 'an-inline'}),
          h('span', {id: 'an-key', class: 'an-inline'}),
          h('button', {class: 'btn sm', id: 'an-scope-btn', title: 'which experiments the model is told about (it can still read any cell)'}, 'scope: all'),
          h('span', {class: 'spacer'}),
          h('span', {id: 'an-sbx', class: 'chip'}),
          h('span', {id: 'an-meter', class: 'chip', title: 'size of the last request against the model\'s context budget; session total'}),
          h('button', {class: 'btn sm', id: 'an-ctx', title: 'the exact messages the next question would send'}, 'context'),
          h('button', {class: 'btn sm', id: 'an-exp-nb', title: 'the session as a Jupyter notebook (code cells with their outputs)'}, '⤓ ipynb'),
          h('button', {class: 'btn sm', id: 'an-exp-md', title: 'the session as Markdown'}, '⤓ md'),
          h('button', {class: 'btn sm', id: 'an-restart', title: 'drop this session\'s Python worker (its variables are lost)'}, '↺ python')),
        h('div', {id: 'an-local', class: 'ev-bar', style: 'display:none'}),
        h('div', {id: 'an-scope', class: 'ev-bar an-scope', style: 'display:none'}),
        h('div', {id: 'an-banner'}),
        h('div', {class: 'an-chat', id: 'an-chat'}),
        h('div', {class: 'an-compose'},
          h('div', {class: 'an-suggest', id: 'an-suggest'}),
          h('textarea', {id: 'an-text', rows: 3, placeholder: 'Ask about the experiments — e.g. "compare the v5 arms on the judge, with CIs". Enter sends, Shift+Enter is a new line.'}),
          h('div', {class: 'an-row'},
            h('button', {class: 'btn primary', id: 'an-send'}, 'Ask'),
            h('button', {class: 'btn', id: 'an-stop', style: 'display:none'}, '■ Stop'),
            h('button', {class: 'btn sm', id: 'an-codemode', title: 'run your own Python in this session\'s worker (ae, pd, np, px, go, plt preloaded)'}, '⌨ run code'),
            h('span', {class: 'hint', id: 'an-hint'}, ''))))));
  $('#an-new').onclick = () => newSession();
  $('#an-ep').onchange = e => { AN.ep = e.target.value; store(); renderHead(); syncSession({endpoint: AN.ep}); };
  $('#an-scope-btn').onclick = () => { const s = $('#an-scope'); s.style.display = s.style.display === 'none' ? '' : 'none'; };
  $('#an-ctx').onclick = showContext;
  $('#an-exp-nb').onclick = () => AN.sid && (location.href = `/api/analysis/export?session=${AN.sid}&format=ipynb`);
  $('#an-exp-md').onclick = () => AN.sid && (location.href = `/api/analysis/export?session=${AN.sid}&format=md`);
  $('#an-restart').onclick = async () => { if (!AN.sid) return; await api('POST', '/api/analysis/kernel/restart', {session: AN.sid}); hint('Python worker dropped — the next step starts a fresh one'); };
  $('#an-send').onclick = () => send();
  $('#an-stop').onclick = stop;
  $('#an-codemode').onclick = () => setCodeMode(!AN.codeMode);
  $('#an-text').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !AN.codeMode){ e.preventDefault(); send(); }
                                                   if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && AN.codeMode){ e.preventDefault(); send(); } });
  const sg = $('#an-suggest');
  for (const q of SUGGEST) sg.append(h('button', {title: q, onclick: () => { $('#an-text').value = q; send(); }}, q));
  new MutationObserver(redrawAll).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll);
}
function hint(t, bad){ const e = $('#an-hint'); e.textContent = t || ''; e.style.color = bad ? 'var(--bad)' : ''; }
function store(){ try { localStorage.setItem('xc-analysis', JSON.stringify({ep: AN.ep, sid: AN.sid})); } catch(e){} }
function restore(){ try { const d = JSON.parse(localStorage.getItem('xc-analysis') || 'null'); if (d){ AN.ep = d.ep || null; AN.sid = d.sid || null; } } catch(e){} }
const curEp = () => AN.eps.find(e => e.name === AN.ep || e.id === AN.ep) || AN.eps.find(e => e.ready)
  || AN.eps.find(e => e.kind !== 'local_vllm') || AN.eps[0] || null;

async function loadEndpoints(){
  try { const d = await api('GET', '/api/analysis/endpoints'); AN.eps = d.endpoints || []; AN.sandbox = d.sandbox; }
  catch(e){ AN.eps = []; hint(e.message, true); }
  renderHead();
}
function renderHead(){
  const sel = $('#an-ep'); sel.replaceChildren();
  for (const e of AN.eps) sel.append(h('option', {value: e.name, selected: curEp() === e}, `${e.name} · ${e.model || '?'}`));
  const ep = curEp(); if (ep) AN.ep = ep.name;
  const st = $('#an-ep-st');
  if (!ep){ st.className = 'chip bad'; st.textContent = 'no configs/analysis/*.yaml'; }
  else if (ep.error){ st.className = 'chip bad'; st.textContent = 'invalid config'; st.title = ep.error; }
  else if (ep.ready){ st.className = 'chip good'; st.textContent = ep.kind === 'local_vllm' ? 'local · up' : 'ready'; st.title = ep.description || ep.base_url; }
  else { st.className = 'chip warn'; st.textContent = ep.kind === 'local_vllm' ? 'local · down' : 'not ready'; st.title = ep.why || ''; }
  // options: reasoning effort (per session)
  const opts = $('#an-opts'); opts.replaceChildren();
  if (ep && (ep.reasoning_efforts || []).length){
    const cur = (AN.sess && AN.sess.options && AN.sess.options.reasoning_effort) || ep.reasoning_effort || '';
    const s = h('select', {id: 'an-effort', title: 'reasoning effort (this session)'}, ep.reasoning_efforts.map(x => h('option', {value: x, selected: x === cur}, 'effort: ' + x)));
    s.onchange = e => syncSession({options: Object.assign({}, (AN.sess && AN.sess.options) || {}, {reasoning_effort: e.target.value})});
    opts.append(s);
  }
  // key
  const k = $('#an-key'); k.replaceChildren();
  if (ep && ep.key && ep.key.needed){
    if (ep.key.set){
      k.append(h('span', {class: 'chip', title: `${ep.key.name} from the ${ep.key.source}`}, `key ${ep.key.hint} · ${ep.key.source}`));
      if (ep.key.source === 'tab') k.append(h('a', {href: '#', title: 'forget the key (server memory)', onclick: async e => { e.preventDefault(); await api('POST', '/api/analysis/key', {name: ep.key.name, key: null}); loadEndpoints(); }}, 'forget'));
    } else {
      const inp = h('input', {type: 'password', id: 'an-key-in', placeholder: `${ep.key.name} — kept in the server's memory only`, autocomplete: 'off', style: 'width:300px'});
      const use = async () => { try { await api('POST', '/api/analysis/key', {name: ep.key.name, key: inp.value}); inp.value = ''; loadEndpoints(); } catch(e){ hint(e.message, true); } };
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') use(); });
      k.append(inp, h('button', {class: 'btn sm', onclick: use}, 'use key'));
    }
  }
  // local server (Gemma): status + start / stop
  const loc = $('#an-local');
  if (ep && ep.kind === 'local_vllm'){
    loc.style.display = ''; loc.replaceChildren();
    const s = ep.server || {};
    const up = ['starting', 'ready', 'stopping'].includes(s.state);
    loc.append(h('label', {class: 'k'}, 'local server'),
      h('span', {class: 'chip ' + (s.ready ? 'good' : s.state === 'starting' || s.state === 'stopping' ? 'run' : s.state === 'failed' ? 'bad' : 'warn')},
        (s.state === 'starting' || s.state === 'stopping' ? '● ' : '') + (s.state || (ep.ready ? 'up' : 'down'))),
      h('span', {class: 'hint', style: 'flex:1;min-width:200px'}, s.detail || ep.why || ''),
      h('button', {class: 'btn sm', title: 'the server log (vLLM)', onclick: showLocalLog}, 'log'),
      h('button', {class: 'btn sm primary', disabled: up || s.startable === false, title: 'start vllm serve on the GPU (it takes the whole H100; runs cannot start while it is up)', onclick: () => localCtl('start')}, '▶ start'),
      h('button', {class: 'btn sm', disabled: !up || s.state === 'stopping', onclick: () => localCtl('stop')}, '■ stop'));
    clearTimeout(AN.localT);                                  // follow a start / stop until it settles
    if (s.state === 'starting' || s.state === 'stopping') AN.localT = setTimeout(loadEndpoints, 4000);
  } else { loc.style.display = 'none'; clearTimeout(AN.localT); }
  // sandbox
  const sb = $('#an-sbx');
  if (AN.sandbox){ sb.className = 'chip ' + (AN.sandbox.available ? 'good' : 'warn');
    sb.textContent = AN.sandbox.available ? 'sandbox ✓' : 'no sandbox — steps need approval';
    sb.title = AN.sandbox.available ? 'code runs read-only (except the session folder), without network, in its own PID namespace' : AN.sandbox.reason; }
  renderMeter();
  $('#an-send').disabled = AN.busy || !(ep && ep.ready) && !AN.codeMode;
}
async function localCtl(what, force){
  try { await api('POST', `/api/analysis/local/${what}`, {endpoint: AN.ep, force: !!force}); hint(''); }
  catch(e){
    if (e.status === 409 && what === 'start' && !force){ if (confirm(e.message)) return localCtl('start', true); }
    else hint(e.message, true);
  }
  loadEndpoints();
}
async function showLocalLog(){
  try {
    const d = await api('GET', `/api/analysis/local/log?endpoint=${encodeURIComponent(AN.ep)}&n=300`);
    $('#text-title').textContent = `local model server log — ${AN.ep}`;
    $('#text-body').textContent = d.exists ? (d.lines.join('\n') || '(empty)') : '(no log yet — the server has not been started)';
    openModal('#text-modal');
  } catch(e){ hint(e.message, true); }
}
function renderMeter(){
  const m = $('#an-meter'); const ep = curEp();
  const used = AN.sess && AN.sess.usage ? AN.sess.usage.total_tokens : 0;
  const last = AN.lastTokens;
  m.textContent = (last != null && ep ? `≈ ${fmtK(last)} / ${fmtK(ep.context_tokens)} ctx` : 'ctx —') + (used ? ` · ${fmtK(used)} used` : '');
  m.className = 'chip' + (last != null && ep && last > 0.8 * ep.context_tokens ? ' warn' : '');
}

async function loadScope(){
  try { const d = await api('GET', '/api/analysis/scope'); AN.exps = d.experiments || []; } catch(e){ AN.exps = []; }
  renderScope();
}
function renderScope(){
  const box = $('#an-scope'); box.replaceChildren(h('label', {class: 'k'}, 'experiments'));
  const cur = new Set((AN.sess && AN.sess.scope && AN.sess.scope.experiments) || AN.scope || []);
  for (const e of AN.exps){
    const cb = h('input', {type: 'checkbox', checked: cur.has(e.experiment)});
    cb.onchange = () => { const set = new Set((AN.sess && AN.sess.scope && AN.sess.scope.experiments) || AN.scope || []);
      cb.checked ? set.add(e.experiment) : set.delete(e.experiment); AN.scope = [...set]; syncSession({scope: {experiments: AN.scope}}); renderScopeBtn(); };
    box.append(h('label', {class: 'chip'}, cb, `${e.experiment} (${e.cells})`));
  }
  box.append(h('a', {href: '#', onclick: ev => { ev.preventDefault(); AN.scope = []; syncSession({scope: {experiments: []}}); renderScope(); }}, 'all'));
  renderScopeBtn();
}
function renderScopeBtn(){
  const s = (AN.sess && AN.sess.scope && AN.sess.scope.experiments) || AN.scope || [];
  $('#an-scope-btn').textContent = s.length ? `scope: ${s.length === 1 ? s[0] : s.length + ' experiments'} ▾` : 'scope: all ▾';
}

/* ---------------------------------------------------------------- sessions */
async function loadSessions(){
  try { const d = await api('GET', '/api/analysis/sessions'); AN.sessions = d.sessions || []; } catch(e){ AN.sessions = []; }
  const box = $('#an-sessions'); box.replaceChildren();
  if (!AN.sessions.length) box.append(h('div', {class: 'empty'}, 'no session yet — ask a question'));
  for (const s of AN.sessions){
    box.append(h('div', {class: 'cfg-item' + (s.id === AN.sid ? ' on' : ''), onclick: () => openSession(s.id)},
      h('div', {class: 'name', dir: 'auto'}, s.title || '(untitled)'),
      h('div', {class: 'meta'}, h('span', {}, s.model || s.endpoint), h('span', {}, `${s.turns} question${s.turns === 1 ? '' : 's'}`), h('span', {}, fmtTime(s.updated)))));
  }
}
async function openSession(id){
  if (AN.busy) return;
  try { AN.sess = await api('GET', `/api/analysis/session?id=${encodeURIComponent(id)}`); }
  catch(e){ AN.sid = null; AN.sess = null; store(); renderChat(); return; }
  AN.sid = id; AN.ep = AN.sess.endpoint || AN.ep; AN.lastTokens = null; store();
  renderHead(); renderScope(); renderChat(); loadSessions();
  const rt = AN.sess.runtime || {};
  clearInterval(AN.pollT);
  if (rt.busy){
    $('#an-banner').replaceChildren(h('div', {class: 'status on warn'}, 'a question is still running in this session (started from another page) — this view refreshes when it ends'));
    AN.pollT = setInterval(async () => { const s = await api('GET', `/api/analysis/session?id=${id}`).catch(() => null);
      if (s && !(s.runtime || {}).busy){ clearInterval(AN.pollT); $('#an-banner').replaceChildren(); openSession(id); } }, 3000);
  } else $('#an-banner').replaceChildren();
}
async function newSession(){
  if (AN.busy) return null;
  const ep = curEp(); if (!ep){ hint('no model configured (configs/analysis/*.yaml)', true); return null; }
  const effort = $('#an-effort') ? $('#an-effort').value : null;
  AN.sess = await api('POST', '/api/analysis/sessions', {endpoint: ep.name, scope: {experiments: AN.scope || []},
                                                          options: effort ? {reasoning_effort: effort} : {}});
  AN.sid = AN.sess.id; AN.lastTokens = null; store();
  renderChat(); loadSessions(); renderHead(); renderScope();
  return AN.sess;
}
async function syncSession(patch){
  if (!AN.sid || AN.busy) return;
  try { AN.sess = Object.assign(AN.sess || {}, await api('POST', '/api/analysis/session/update', Object.assign({id: AN.sid}, patch))); }
  catch(e){ hint(e.message, true); }
  renderHead();
}
function renderChat(){
  const chat = $('#an-chat'); chat.replaceChildren(); AN.figs = [];
  const turns = (AN.sess && AN.sess.turns) || [];
  $('#an-suggest').style.display = turns.length ? 'none' : '';
  if (!turns.length) chat.append(h('div', {class: 'empty'}, AN.sess ? 'Ask a question below, or pick a suggestion.' :
    'Ask a question below — a session is created for it. Earlier sessions are on the left.'));
  for (const t of turns) chat.append(turnEl(t));
  chat.scrollTop = chat.scrollHeight;
  renderMeter();
}

/* ---------------------------------------------------------------- asking */
function setBusy(b){
  AN.busy = b;
  $('#an-send').style.display = b ? 'none' : ''; $('#an-stop').style.display = b ? '' : 'none';
  $('#an-new').disabled = b; renderHead();
}
function setCodeMode(on, code){
  AN.codeMode = !!on;
  const ta = $('#an-text');
  ta.classList.toggle('code', AN.codeMode);
  ta.placeholder = AN.codeMode ? 'Python, run in this session\'s worker (ae, pd, np, px, go, plt preloaded). Ctrl+Enter runs.' :
    'Ask about the experiments — e.g. "compare the v5 arms on the judge, with CIs". Enter sends, Shift+Enter is a new line.';
  if (code !== undefined) ta.value = code;
  $('#an-send').textContent = AN.codeMode ? '▶ Run' : 'Ask';
  $('#an-codemode').classList.toggle('on', AN.codeMode);
  $('#an-codemode').textContent = AN.codeMode ? '✎ ask instead' : '⌨ run code';
  renderHead(); ta.focus();
}
async function send(){
  const ta = $('#an-text'); const text = ta.value.trim();
  if (!text || AN.busy) return;
  hint('');
  if (!AN.sid && !(await newSession().catch(e => { hint(e.message, true); return null; }))) return;
  if (AN.codeMode) return runCode(text);
  const ep = curEp(); if (!ep || !ep.ready){ hint(ep ? ep.why : 'no model', true); return; }
  ta.value = '';
  const chat = $('#an-chat'); if (!(AN.sess.turns || []).length) chat.replaceChildren();
  $('#an-suggest').style.display = 'none';
  const tEl = h('div', {class: 'an-turn'}, h('div', {class: 'an-q', dir: 'auto'}, text));
  const steps = h('div', {class: 'an-steps'}); tEl.append(steps); chat.append(tEl);
  const live = {el: null, text: '', step: null, stepEls: {}, tEl, steps, t0: performance.now(), turn: {user: text, steps: []}};
  AN.live = live; setBusy(true);
  const ctrl = new AbortController(); AN.abort = ctrl;
  const scroll = () => { const near = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 240; if (near) chat.scrollTop = chat.scrollHeight; };
  const liveBubble = () => {
    if (!live.el){ live.el = h('div', {class: 'an-live cur'}); steps.append(live.el); }
    live.el.replaceChildren(live.text ? md(live.text) : h('span', {class: 'hint'}, 'thinking…'));
  };
  const handle = ev => {
    switch (ev.type){
      case 'turn': break;
      case 'model_start': live.text = ''; if (ev.approx_tokens != null){ AN.lastTokens = ev.approx_tokens; renderMeter(); } liveBubble(); break;
      case 'token': live.text += ev.text; break;
      case 'reply': {
        if (live.el){ live.el.remove(); live.el = null; }
        const st = {n: ev.step, reply: ev.text, code: ev.code};
        if (ev.draft){ st.draft_answer = true; st.provenance = {unverified: (ev.unverified || []).map(t => ({text: t}))}; }
        if (!ev.code) st.feedback = ev.nudged ? 'the model announced work without code — asked it to send the code' : ev.truncated ? 'the reply was cut off before its code block closed — asked for a shorter step' : null;
        const el = stepEl(st); steps.append(el);
        if (ev.code){ live.stepEls[ev.step] = el; setStepStatus(el, 'run', 'queued'); }
        break;
      }
      case 'approval': {
        const el = live.stepEls[ev.step]; if (!el) break;
        el.querySelector('details').open = true; setStepStatus(el, 'warn', 'waiting for your approval');
        const bar = h('div', {class: 'an-approve'}, h('span', {}, ev.reason || 'approve this step?'),
          h('button', {class: 'btn sm primary', onclick: () => approve(ev.step, true, bar)}, '▶ run it'),
          h('button', {class: 'btn sm', onclick: () => approve(ev.step, false, bar)}, 'skip'));
        el.querySelector('.an-out').append(bar); break;
      }
      case 'exec_start': { const el = live.stepEls[ev.step]; if (el){ setStepStatus(el, 'run', 'running…'); el._t0 = performance.now(); } break; }
      case 'exec': { const el = live.stepEls[ev.step]; if (el) fillExec(el, ev); break; }
      case 'answer': if (live.el){ live.el.remove(); live.el = null; } tEl.append(answerEl(ev.text, ev.provenance)); break;
      case 'error': if (live.el){ live.el.remove(); live.el = null; } tEl.append(h('div', {class: 'status on bad'}, ev.message)); break;
      case 'done':
        live.done = true;
        tEl.append(footEl({status: ev.status, steps: Object.keys(live.stepEls).map(k => ({code: 1})), elapsed_sec: ev.elapsed_sec, usage: ev.usage, model: AN.sess && AN.sess.model}));
        if (AN.sess) AN.sess.usage = ev.session_usage;
        break;
    }
  };
  let last = 0;
  const tick = setInterval(() => { for (const el of Object.values(live.stepEls)){ if (el._t0 && el.querySelector('.an-st').textContent.startsWith('running')) setStepStatus(el, 'run', 'running… ' + fmtS((performance.now() - el._t0) / 1000)); } }, 500);
  try {
    const r = await fetch('/api/analysis/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({session: AN.sid, message: text}), signal: ctrl.signal});
    if (!r.ok){ const d = await r.json().catch(() => ({error: r.statusText})); throw new Error(d.error || r.statusText); }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
    while (true){
      const {value, done} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0){
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of chunk.split('\n')) if (line.startsWith('data:')){ try { handle(JSON.parse(line.slice(5))); } catch(e){ console.warn('bad analysis event', e); } }
      }
      const now = performance.now(); if (live.el && now - last > 90){ last = now; liveBubble(); } scroll();
    }
    if (!live.done) tEl.append(h('div', {class: 'status on warn'}, 'the stream ended before the turn was complete'));
  } catch(e){
    if (live.el){ live.el.remove(); live.el = null; }
    tEl.append(h('div', {class: 'status on ' + (e.name === 'AbortError' ? 'warn' : 'bad')}, e.name === 'AbortError' ? 'stopped' : e.message));
    if (!AN.sess.turns.length) ta.value = text;
  }
  clearInterval(tick);
  AN.live = null; AN.abort = null; setBusy(false);
  try { AN.sess = await api('GET', `/api/analysis/session?id=${AN.sid}`); } catch(e){}
  renderMeter(); loadSessions(); scroll();
}
async function approve(step, ok, bar){
  try { await api('POST', '/api/analysis/approve', {session: AN.sid, step, approve: ok}); bar.remove(); }
  catch(e){ hint(e.message, true); }
}
async function stop(){
  if (!AN.sid) return;
  try { await api('POST', '/api/analysis/stop', {session: AN.sid}); } catch(e){}
  setTimeout(() => { if (AN.busy && AN.abort) AN.abort.abort(); }, 8000);   // the server ends the turn; this is the fallback
}
async function runCode(code){
  setBusy(true);
  const chat = $('#an-chat'); if (!(AN.sess.turns || []).length) chat.replaceChildren();
  const st = {n: '…', reply: '', code};
  const el = stepEl(st); el.querySelector('details').open = true; setStepStatus(el, 'run', 'running…');
  const tEl = h('div', {class: 'an-turn'}, h('div', {class: 'an-q code'}, h('span', {class: 'chip'}, 'code run by hand')), h('div', {class: 'an-steps'}, el));
  chat.append(tEl); chat.scrollTop = chat.scrollHeight;
  try {
    const r = await api('POST', '/api/analysis/exec', {session: AN.sid, code});
    el.querySelector('.an-sn').textContent = `step ${r.step}`; fillExec(el, r); $('#an-text').value = '';
  } catch(e){ setStepStatus(el, 'bad', 'failed'); el.querySelector('.an-out').append(h('div', {class: 'status on bad'}, e.message)); }
  setBusy(false);
  try { AN.sess = await api('GET', `/api/analysis/session?id=${AN.sid}`); } catch(e){}
  loadSessions();
}
async function showContext(){
  if (!AN.sid){ hint('no session yet', true); return; }
  try {
    const d = await api('POST', '/api/analysis/preview', {session: AN.sid, message: AN.codeMode ? '' : $('#an-text').value});
    $('#text-title').textContent = `analysis context — ${d.model} · ${d.messages.length} messages · ${d.chars.toLocaleString()} chars ≈ ${d.approx_tokens.toLocaleString()} of ${d.context_tokens.toLocaleString()} tokens`;
    $('#text-body').textContent = d.messages.map(m => `═══ ${m.role} ═══\n${m.content}`).join('\n\n');
    openModal('#text-modal');
  } catch(e){ hint(e.message, true); }
}

/* ---------------------------------------------------------------- styles */
const CSS = `
.an-grid{display:grid;grid-template-columns:270px minmax(0,1fr);gap:16px;align-items:start}
@media (max-width:1100px){.an-grid{grid-template-columns:1fr}}
.an-side{position:sticky;top:62px}
.an-sessions{max-height:calc(100vh - 150px);overflow:auto}
.an-main{display:flex;flex-direction:column;height:calc(100vh - 92px);min-height:520px}
.an-head select,.an-head input{padding:4px 7px;border:1px solid var(--line-2);border-radius:6px;background:var(--surface)}
.an-head label.k,.an-scope label.k,#an-local label.k{color:var(--ink-3);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
.an-inline{display:inline-flex;gap:6px;align-items:center}
.an-scope .chip input{margin:0}
.an-chat{flex:1;overflow:auto;padding:14px 18px;display:flex;flex-direction:column;gap:18px}
.an-turn{display:flex;flex-direction:column;gap:10px}
.an-q{align-self:flex-end;max-width:80%;background:var(--accent-soft);border-radius:10px;padding:8px 12px;white-space:pre-wrap;font-size:14px}
.an-q.code{background:none;padding:0}
.an-steps{display:flex;flex-direction:column;gap:10px}
.an-step{border-left:3px solid var(--line-2);padding-left:12px;display:flex;flex-direction:column;gap:6px}
.an-md{font-size:14px;line-height:1.55;overflow-wrap:anywhere}
.an-md p{margin:0 0 7px}.an-md p:last-child{margin-bottom:0}
.an-md ul,.an-md ol{margin:4px 0 8px 20px;padding:0}
.an-md h1,.an-md h2,.an-md h3,.an-md h4{font-size:14.5px;margin:8px 0 4px}
.an-md code{font-family:var(--mono);font-size:12.5px;background:var(--sunken);padding:0 4px;border-radius:4px}
.an-md pre{background:var(--sunken);border:1px solid var(--line);border-radius:8px;padding:8px 10px;overflow:auto}
.an-md pre code{background:none;padding:0}
.an-md table.res{width:auto;margin:6px 0}
.an-md table.res td{text-align:right}
.an-md blockquote{margin:6px 0;padding-left:10px;border-left:3px solid var(--line-2);color:var(--ink-2)}
details.an-code{border:1px solid var(--line);border-radius:8px;background:var(--surface)}
details.an-code>summary{display:flex;align-items:center;gap:8px;padding:5px 10px;cursor:pointer;list-style:none;font-size:12.5px;color:var(--ink-2)}
details.an-code>summary::before{content:'▸';color:var(--ink-3)}
details.an-code[open]>summary::before{content:'▾'}
details.an-code>summary .btn{padding:1px 8px;font-size:11.5px}
.an-sn{font-family:var(--mono);color:var(--ink)}
pre.an-src{margin:0;padding:8px 12px;border-top:1px solid var(--line);overflow:auto;font-family:var(--mono);font-size:12.5px;line-height:1.5;max-height:420px;background:var(--sunken)}
.an-out{display:flex;flex-direction:column;gap:8px}
.an-pre{margin:0;padding:7px 10px;background:var(--sunken);border:1px solid var(--line);border-radius:8px;font-family:var(--mono);font-size:12px;line-height:1.45;white-space:pre;overflow:auto;max-height:300px}
.an-val{border-style:dashed}
.an-stderr{color:var(--warn)}
details.an-err{border:1px solid var(--bad);background:var(--bad-soft);color:var(--bad);border-radius:8px;padding:5px 10px;font-size:12.5px}
details.an-err pre{margin:6px 0 0;color:var(--ink)}
.an-note{font-size:12.5px;color:var(--ink-3);font-style:italic}
details.an-draft{border:1px dashed var(--warn);border-radius:8px;padding:5px 10px;font-size:12.5px;color:var(--warn)}
details.an-draft>summary{cursor:pointer}
details.an-draft .an-md{color:var(--ink-3);margin-top:6px;font-size:13px}
.an-disp{border:1px solid var(--line);border-radius:10px;background:var(--surface);overflow:hidden}
.an-disp-h{display:flex;align-items:center;gap:8px;padding:6px 12px;border-bottom:1px solid var(--line);font-size:12.5px;color:var(--ink-2)}
.an-disp-h span{flex:1}
.an-dl{text-decoration:none;font-size:12px}
.an-fig{width:100%;min-height:200px}
.an-img{max-width:100%;display:block;margin:0 auto}
.an-tbl{max-height:380px;overflow:auto}
.an-tbl table.res{width:auto;min-width:100%}
.an-tbl table.res th{position:sticky;top:0;cursor:pointer;white-space:nowrap;z-index:1;text-align:left;font-family:var(--sans)}
.an-tbl table.res td{font-family:var(--sans);text-align:left;white-space:nowrap;max-width:520px;overflow:hidden;text-overflow:ellipsis}
.an-tbl table.res td.num{text-align:right;font-family:var(--mono)}
.an-tbl table.res th.num{text-align:right}
.an-answer{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;padding:12px 16px;box-shadow:var(--shadow)}
.an-prov{margin-top:8px}
mark.an-unv{background:var(--warn-soft);color:var(--warn);border-bottom:2px dotted var(--warn);padding:0 2px;border-radius:3px}
.an-foot{font-size:12px;color:var(--ink-3)}
.an-chat .status,#an-banner .status{border-radius:8px;border:1px solid var(--line)}
.an-foot.warn{color:var(--warn)}.an-foot.bad{color:var(--bad)}
.an-live{font-size:14px;color:var(--ink-2)}
.an-live.cur .an-md>*:last-child::after{content:'▍';color:var(--accent);animation:pulse 1s steps(1) infinite}
.an-approve{display:flex;gap:8px;align-items:center;padding:6px 10px;border:1px solid var(--warn);background:var(--warn-soft);border-radius:8px;font-size:12.5px}
.an-compose{border-top:1px solid var(--line);padding:10px 14px 12px;display:flex;flex-direction:column;gap:8px}
.an-compose textarea{width:100%;resize:vertical;min-height:64px;padding:8px 10px;border:1px solid var(--line-2);border-radius:8px;background:var(--surface);font:inherit;line-height:1.45}
.an-compose textarea.code{font-family:var(--mono);font-size:12.5px;min-height:120px}
.an-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.an-suggest{display:flex;gap:6px;flex-wrap:wrap}
.an-suggest button{padding:3px 10px;border-radius:999px;border:1px solid var(--line);background:var(--surface);cursor:pointer;font-size:12.5px;color:var(--ink-2)}
.an-suggest button:hover{border-color:var(--accent);color:var(--accent)}
#an-codemode.on{background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}
pre.an-src .token.comment{color:var(--ink-3);font-style:italic}
pre.an-src .token.string,pre.an-src .token.triple-quoted-string{color:var(--good)}
pre.an-src .token.keyword,pre.an-src .token.builtin{color:var(--accent)}
pre.an-src .token.number,pre.an-src .token.boolean{color:var(--warn)}
pre.an-src .token.function{color:var(--run)}
pre.an-src .token.operator,pre.an-src .token.punctuation{color:var(--ink-2)}
`;

/* ---------------------------------------------------------------- entry */
async function open(){
  if (!AN.inited){
    AN.inited = true;
    document.head.append(h('style', {}, CSS));
    restore(); build();
    await ensureText();
    await loadEndpoints();
    await loadScope();
    await loadSessions();
    if (AN.sid && AN.sessions.some(s => s.id === AN.sid)) await openSession(AN.sid);
    else { AN.sid = null; renderChat(); }
  } else {
    loadEndpoints(); loadSessions();
  }
}
window.AnalysisTab = {open, state: AN};
})();
