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

function chip(label, val, cls = '') {
  return `<span class="chip ${cls}">${label} <b>${esc(val)}</b></span>`;
}

/* budget: traceRow injects its own chip with threshold styling, so the raw field would
 *         draw the same "3/5" twice.
 * sent:   a list of search queries. Rendered below the chips instead — it is the
 *         longest field in the trace and the only one worth reading in full. */
const HIDE = new Set(['node', 'action', 'why', 'to', 'tokens', 'budget', 'sent']);

/* Fields where a non-zero value is a problem rather than a statistic. */
function isHot(k, v) {
  return (k === 'degraded' && v) || (k === 'broken_cites' && v > 0) ||
         (k === 'fallback' && v) || (k === 'dropped_ungrounded' && v > 0);
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

  const chips = Object.entries(e)
    .filter(([k, v]) => !HIDE.has(k) && v !== undefined && v !== '' && v !== false)
    .map(([k, v]) => chip(k.replace(/_/g, ' '), v, isHot(k, v) ? 'hot' : ''));

  if (e.node === 'researcher' && d.budgets.searches) {
    chips.unshift(chip('budget', `${d.budgets.searches}/${caps.searches}`,
      d.budgets.searches >= caps.searches ? 'warn' : ''));
  }

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
      ${chips.length ? `<div class="chips">${chips.join('')}</div>` : ''}
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
