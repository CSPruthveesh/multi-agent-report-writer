/* Shared rendering for both pages.
 *
 * traceRow() builds a row and returns it rather than appending it, and does nothing
 * with cost. That is the seam between the two pages: /dev feeds the same frame into a
 * baseline meter, / prints a token count, and neither concern belongs in the function
 * that draws the row.
 *
 * An earlier version of this had addEvent() append the row AND update the meter, which
 * is why the user page could not reuse it and grew a second copy instead.
 */

const esc = s => String(s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/* One line per event, not a row of boxed fields.
 *
 * The boxes rendered every field the node happened to emit, at equal weight, wrapping
 * to three lines on the busier nodes — so reading the shape of a run meant reading
 * twenty numbers. This picks the few that say what the node did, and marks the ones
 * that mean something is wrong.
 *
 * Fields not named here are deliberately dropped rather than appended. A summary that
 * grows a term every time a node gains a field is not a summary.
 */
function summarise(e, d, caps) {
  const n = v => `<b>${esc(v)}</b>`;
  const bad = v => `<span class="bad">${esc(v)}</span>`;
  const bits = [];

  switch (e.node) {
    case 'researcher': {
      if (e.found != null) bits.push(`${n(e.found)} sources`);
      if (e.queries != null) bits.push(`${n(e.queries)} ${e.queries === 1 ? 'query' : 'queries'}`);
      if (e.mode) bits.push(esc(e.mode));
      const used = d.budgets.searches, cap = caps.searches;
      if (used) bits.push(used >= cap
        ? `<span class="warn">search budget ${used}/${cap} spent</span>`
        : `budget ${n(`${used}/${cap}`)}`);
      break;
    }
    case 'analyst':
      if (e.sections != null) bits.push(`${n(e.sections)} sections`);
      if (e.tensions) bits.push(`${n(e.tensions)} tensions held apart`);
      bits.push(e.gaps_kept ? `${n(e.gaps_kept)} evidence gap to close` : 'no gaps');
      break;

    case 'supervisor':
      if (e.loop) bits.push(`loop ${n(e.loop)}`);
      if (e.count != null) bits.push(`${n(e.count)} gap${e.count === 1 ? '' : 's'} retired`);
      if (e.loops) bits.push(`loops ${n(e.loops)}`);
      if (e.searches) bits.push(`searches ${n(e.searches)}`);
      if (e.attempt) bits.push(`attempt ${n(e.attempt)}`);
      if (e.min_score != null) bits.push(`lowest score ${n(e.min_score)}`);
      break;

    case 'writer':
      if (e.words != null) bits.push(`${n(e.words)} words`);
      if (e.cited != null) bits.push(`${n(e.cited)} cited`);
      if (e.broken_cites) bits.push(bad(`${e.broken_cites} broken citations`));
      if (e.ends_on) bits.push(`closes on ${n(e.ends_on)}`);
      if (e.changed_pct != null) bits.push(`${n(e.changed_pct + '%')} of the text changed`);
      if (e.fallback) bits.push(bad('fell back to a rewrite'));
      break;

    case 'critic':
      if (e.verdict) bits.push(n(e.verdict));
      if (e.mean != null) bits.push(`mean ${n(e.mean)}`);
      if (e.worst) bits.push(`worst ${esc(e.worst)}`);
      if (e.dropped_ungrounded) bits.push(bad(`${e.dropped_ungrounded} ungrounded dropped`));
      break;

    case 'finalize':
      if (e.words != null) bits.push(`${n(e.words)} words`);
      bits.push(e.unclosed_gaps
        ? bad(`${e.unclosed_gaps} gap${e.unclosed_gaps === 1 ? '' : 's'} left unclosed`)
        : 'every gap closed');
      if (e.node_failures) bits.push(bad(`${e.node_failures} node failures`));
      break;
  }

  if (e.degraded) bits.unshift(bad('degraded'));
  return bits.join(' · ');
}

/**
 * Build one trace row.
 * @param d     an SSE "node" frame: {event, cost, budgets, counts}
 * @param caps  budget ceilings from /api/topics, for the searches chip
 * @returns     the row element, for the caller to append
 */
function traceRow(d, caps) {
  const e = d.event;
  const isLoop = (e.why || '').includes('gap') || e.action === 'retire_gaps' ||
                 (d.budgets.revisions > 0 && e.node === 'writer');
  const alarm = ['degraded', 'parse_failed', 'skipped'].includes(e.action);

  const detail = summarise(e, d, caps);

  const queries = Array.isArray(e.sent) && e.sent.length
    ? `<ul class="queries">${e.sent.map(q => `<li>${esc(q)}</li>`).join('')}</ul>` : '';

  const row = document.createElement('div');
  row.className = 'ev';
  row.dataset.loop = isLoop ? '1' : '0';
  row.dataset.alarm = alarm ? '1' : '0';
  row.innerHTML = `<div class="node">${esc(e.node)}<div class="dot"></div></div>
    <div class="body">
      <div class="action">${esc(e.action)}${e.to ? ` → ${esc(e.to)}` : ''}
        ${e.why ? `<span class="why"> — ${esc(e.why)}</span>` : ''}</div>
      ${detail ? `<div class="detail">${detail}</div>` : ''}
      ${queries}
    </div>`;
  return row;
}

/* Enough markdown for what the Writer emits: one H1, H2 sections, bold, rules, and
 * F-number citations. Escaped first, so a model-written report cannot inject markup. */
function renderMarkdown(md) {
  return esc(md)
    .replace(/\[((?:F\d{3})(?:,\s*F\d{3})*)\]/g, '<cite>$1</cite>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^---$/gm, '<hr>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .split(/\n{2,}/)
    .map(b => /^<(h1|h2|hr)/.test(b) ? b : `<p>${b.replace(/\n/g, ' ')}</p>`)
    .join('');
}

/* "Looked for and not found" — or nothing at all when every gap was closed. */
function gapsBlock(gaps) {
  return (gaps || []).length
    ? `<div class="gaps"><h3>Looked for and not found</h3><ul>${
        gaps.map(g => `<li>${esc(g)}</li>`).join('')}</ul></div>`
    : '';
}

/* Read an SSE byte stream, calling handlers[eventName](data) per frame. Both pages
 * drove this loop with their own copy; the framing rules are not page-specific. */
async function readSSE(res, handlers) {
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const frames = buf.split('\n\n');
    buf = frames.pop();
    for (const f of frames) {
      const name = (f.match(/^event: (.+)$/m) || [])[1];
      const dm = f.match(/^data: (.+)$/m);
      if (!dm || !handlers[name]) continue;
      handlers[name](JSON.parse(dm[1]));
    }
  }
}

/* ---------------------------------------------------------------- sequencing
 *
 * The stream only carries COMPLETED events — a node emits its trace entry after its
 * model call returns. So the page sat still for the ten or twenty seconds a node was
 * working, then a finished row appeared from nowhere.
 *
 * A placeholder row is added for the node about to run and replaced when its real
 * event arrives: "searching…" becomes "searched · 36 sources · 3 queries".
 *
 * Which node runs next is not a guess. Most edges in the graph are fixed —
 * researcher → analyst and writer → critic — and the rest go through the supervisor,
 * which announces its choice in the route event's `to` field. The supervisor itself
 * never gets a placeholder: it makes no model call, so it would flash and vanish.
 */
const WORKING = {
  researcher: 'searching', analyst: 'planning the report',
  writer: 'writing', critic: 'reviewing', finalize: 'finishing',
};
const FIXED_EDGE = { researcher: 'analyst', writer: 'critic' };

function pendingRow(node) {
  const row = document.createElement('div');
  row.className = 'ev pending';
  row.innerHTML = `<div class="node">${esc(node)}<div class="dot"></div></div>
    <div class="body"><div class="action">${esc(WORKING[node])}…</div></div>`;
  return row;
}

/** Owns the trace element: appends real rows, and keeps one placeholder ahead. */
function traceView(el, caps) {
  let pending = null;
  const clear = () => { if (pending) { pending.remove(); pending = null; } };
  const expect = node => {
    if (node && WORKING[node]) { pending = pendingRow(node); el.appendChild(pending); }
  };
  return {
    reset() { el.innerHTML = ''; pending = null; },
    start() { clear(); expect('researcher'); },
    push(d) {
      clear();
      el.appendChild(traceRow(d, caps));
      const e = d.event;
      expect(e.node === 'supervisor' ? e.to : FIXED_EDGE[e.node]);
    },
    finish() { clear(); },
  };
}
