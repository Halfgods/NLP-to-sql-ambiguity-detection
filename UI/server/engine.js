// Legacy dependency-free heuristic engine. The live UI no longer calls this:
// `/api/query` now goes through model_worker.py (the real Qwen pipeline).
// Kept for reference / offline experiments only.
//
// It is deliberately heuristic so the whole app runs offline: it links NL terms
// to schema objects, detects the places where the request could be read in more
// than one way, and asks clarifying questions (with concrete options) before it
// commits to SQL. Answers are fed back with `answers` so the same question is
// never asked twice.

const STOP = new Set(
  ('a an the of in on at to for and or is are was were be been been do does did how what which who whom whose ' +
    'show me list all please there that this these those it its with by per each every any please give give me ' +
    'from into out up down can could would should i you we they them data please tell').split(' '),
)

const AGG = {
  count: ['how many', 'count', 'number of'],
  sum: ['sum', 'total', 'add up', 'overall', 'how much'],
  avg: ['average', 'avg', 'mean'],
  max: ['maximum', 'highest value', 'max'],
  min: ['minimum', 'lowest value', 'min'],
}

const TIME_PHRASES = [
  ['this month', 'thisMonth'],
  ['last month', 'lastMonth'],
  ['this week', 'thisWeek'],
  ['last week', 'lastWeek'],
  ['this year', 'thisYear'],
  ['last year', 'lastYear'],
  ['last 30 days', 'last30'],
  ['last 7 days', 'last7'],
  ['yesterday', 'yesterday'],
  ['today', 'today'],
  ['recently', 'recent'],
  ['recent', 'recent'],
]

const VAGUE = [
  'top', 'best', 'highest', 'biggest', 'largest', 'most', 'popular', 'high', 'big', 'large',
  'expensive', 'pricey', 'low', 'lowest', 'cheapest', 'smallest', 'fewest', 'many', 'good',
]

// Vocabulary → column-name candidates (keys are STEMS, since tokens are
// stemmed before lookup), so "earners" finds `salary` and "sales" finds
// `amount` in whatever table you upload.
const SYNONYMS = {
  earner: ['salary'], earning: ['salary'], earn: ['salary'], paid: ['salary'],
  income: ['salary'], wage: ['salary'], compensation: ['salary'],
  spend: ['amount'], spending: ['amount'], sale: ['amount'], revenue: ['amount'],
  cost: ['price', 'amount'], price: ['price'],
  people: ['full_name', 'name'], person: ['full_name', 'name'],
  staff: ['full_name'], worker: ['full_name'], team: ['department'],
  tenure: ['hire_date'], join: ['hire_date'], hired: ['hire_date'],
  boss: ['manager_name'], manager: ['manager_name'], supervisor: ['manager_name'],
  customer: ['customer', 'full_name'], client: ['customer', 'full_name'],
}

const TIME_OPTS = {
  lastMonth: { label: 'Previous calendar month', note: "date >= date('now','start of month','-1 month') AND date < date('now','start of month')" },
  thisMonth: { label: 'Current calendar month', note: "date >= date('now','start of month') AND date < date('now','start of month','+1 month')" },
  lastWeek: { label: 'Previous calendar week', note: "date >= date('now','weekday 0','-7 days') AND date < date('now','weekday 0')" },
  thisWeek: { label: 'Current week (since Sunday)', note: "date >= date('now','weekday 0','-6 days')" },
  lastYear: { label: 'Previous calendar year', note: "date >= date('now','start of year','-1 year') AND date < date('now','start of year')" },
  thisYear: { label: 'Current calendar year', note: "date >= date('now','start of year')" },
  last30: { label: 'Rolling last 30 days', note: "date >= date('now','-30 days')" },
  last7: { label: 'Rolling last 7 days', note: "date >= date('now','-7 days')" },
  yesterday: { label: "Yesterday (date = date('now','-1 day'))", note: '' },
  today: { label: "Today (date = date('now'))", note: '' },
  recent: { label: 'Last 30 days (my default for “recent”)', note: "date >= date('now','-30 days')" },
}

function esc(s) {
  return String(s).replace(/'/g, "''")
}
function q(id) {
  return `"${String(id).replace(/"/g, '""')}"`
}
function stem(w) {
  const s = String(w).toLowerCase()
  return s.length > 3 && s.endsWith('s') ? s.slice(0, -1) : s
}
function colParts(name) {
  return String(name).toLowerCase().split('_').filter(Boolean).map(stem)
}
function tokensOf(text) {
  return String(text)
    .toLowerCase()
    .replace(/['’]/g, '')
    .replace(/[^a-z0-9_]+/g, ' ')
    .split(/\s+/)
    .filter((t) => t && !STOP.has(t))
}

function matchColScore(token, colName) {
  const t = stem(token)
  const parts = colParts(colName)
  const flat = parts.join('')
  if (parts.includes(t)) return 1
  if (t.length >= 4 && flat.startsWith(t)) return 0.8
  if (flat.length >= 4 && t.includes(flat)) return 0.7
  return 0
}

function isDateCol(c) {
  return (
    c.type === 'date' ||
    /(date|time|day|month|year|created|updated|hire|joined|at)$/.test(c.name)
  )
}
function isNumericCol(c) {
  return c.type === 'number' || c.type === 'integer' || c.type === 'real'
}

/** Build the WHERE fragment for a named time resolution against `col`. */
function timeClause(col, kind) {
  const c = q(col)
  switch (kind) {
    case 'lastMonth':
      return `${c} >= date('now','start of month','-1 month') AND ${c} < date('now','start of month')`
    case 'thisMonth':
      return `${c} >= date('now','start of month') AND ${c} < date('now','start of month','+1 month')`
    case 'lastWeek':
      return `${c} >= date('now','weekday 0','-7 days') AND ${c} < date('now','weekday 0')`
    case 'thisWeek':
      return `${c} >= date('now','weekday 0','-6 days')`
    case 'lastYear':
      return `${c} >= date('now','start of year','-1 year') AND ${c} < date('now','start of year')`
    case 'thisYear':
      return `${c} >= date('now','start of year')`
    case 'last30':
      return `${c} >= date('now','-30 days')`
    case 'last7':
      return `${c} >= date('now','-7 days')`
    case 'yesterday':
      return `${c} = date('now','-1 day')`
    case 'today':
      return `${c} = date('now')`
    case 'recent':
      return `${c} >= date('now','-30 days')`
    default:
      return null
  }
}

function clarify(questions, extra = {}) {
  return {
    status: 'clarify',
    ambiguities: questions,
    sql: null,
    results: null,
    reply:
      questions.length === 1
        ? `One quick check before I write the SQL — ${questions[0].question}`
        : `A couple of things in that request could go more than one way — help me pin them down and I'll generate the SQL right away.`,
    confidence: Math.max(0.25, 0.75 - 0.15 * (questions.length - 1)),
    ...extra,
  }
}

/**
 * @param {string} query     natural-language request
 * @param {Array}  schema    [{ name, columns: [{name,type,values?}], rows, source }]
 * @param {object} answers   { [ambiguityId]: optionIndex }
 */
export function respond(query, schema, answers = {}) {
  const raw = String(query || '').trim()
  if (!raw) {
    return { status: 'clarify', reply: 'Ask me anything about your data — for example “how many employees are in Engineering?”', sql: null, results: null, ambiguities: [], confidence: 0 }
  }
  const lower = ` ${raw.toLowerCase()} `
  const toks = tokensOf(raw)
  if (toks.length === 0) {
    return { status: 'ok', reply: 'I just need a little more detail than that 🙂', sql: null, results: null, ambiguities: [], confidence: 0.3 }
  }

  const tables = schema.filter((t) => t.rows > 0 || t.columns.length)
  if (tables.length === 0) {
    return {
      status: 'ok',
      reply: 'There is no data yet — upload a CSV in the sidebar first (it gets stored in `UI/uploads/`), then ask me anything about it.',
      sql: null, results: null, ambiguities: [], confidence: 1,
    }
  }

  // ---------- Phase A: which table / join / columns? ----------
  const scored = tables.map((t) => {
    let score = 0
    const tbl = stem(t.name)
    if (toks.includes(tbl) || toks.includes(t.name)) score += 4
    for (const tk of toks) {
      if (stem(tk) === tbl) score += 4
      for (const c of t.columns) if (matchColScore(tk, c.name) >= 0.8) score += 0.7
    }
    return { table: t, score }
  }).sort((a, b) => b.score - a.score)

  let table = null
  const joinAmb = []

  if (scored.length > 1 && scored[0].score > 0 && scored[1].score > 0 && scored[0].score === scored[1].score) {
    // tied tables → table ambiguity
  } else if (scored[0] && scored[0].score > 0) {
    table = scored[0].table
  }

  // columns matched anywhere (per table)
  const hitsByTable = new Map()
  for (const t of tables) {
    const hits = []
    for (const tk of toks) {
      const cands = [tk, ...(SYNONYMS[stem(tk)] || [])]
      for (const c of t.columns) {
        let s = 0
        for (const cand of cands) s = Math.max(s, matchColScore(cand, c.name))
        if (s > 0) hits.push({ token: tk, col: c, score: s })
      }
    }
    hitsByTable.set(t.name, hits)
  }
  const involved = tables.filter((t) => hitsByTable.get(t.name).some((h) => h.score >= 1))

  if (!table && involved.length === 1) table = involved[0]

  if (!table) {
    if (involved.length > 1) {
      const qid = 'table'
      if (answers[qid] == null) {
        return clarify([{
          id: qid,
          kind: 'table',
          question: 'Which table should I read this from?',
          options: involved.map((t) => ({ label: t.name, note: `${t.columns.length} columns · ${t.rows} rows` })),
        }])
      }
      table = involved[Number(answers[qid])] || involved[0]
    } else if (tables.length > 1) {
      const qid = 'table'
      if (answers[qid] == null) {
        return clarify([{
          id: qid,
          kind: 'table',
          question: 'Which table do you mean?',
          options: tables.map((t) => ({ label: t.name, note: `${t.columns.length} columns · ${t.rows} rows` })),
        }])
      }
      table = tables[Number(answers[qid])] || tables[0]
    } else {
      table = tables[0]
    }
  }

  // ---------- join detection (columns from a 2nd table referenced) ----------
  const otherTables = tables.filter((t) => t.name !== table.name)
  for (const other of otherTables) {
    const others = hitsByTable.get(other.name) || []
    const strong = others.filter((h) => h.score >= 1)
    if (strong.length > 0) {
      const shared = table.columns.find((c) => other.columns.some((d) => d.name === c.name))
      const qid = 'join'
      if (answers[qid] == null) {
        const opts = [
          { label: `${table.name} only`, note: 'ignore the other table' },
          { label: `${other.name} only`, note: 'ignore the other table' },
        ]
        if (shared) opts.push({ label: `JOIN on ${shared.name}`, note: `${table.name}.${shared.name} = ${other.name}.${shared.name}` })
        joinAmb.push({
          id: qid, kind: 'join',
          question: `Your question touches both \`${table.name}\` and \`${other.name}\` — how should I connect them?`,
          options: opts,
        })
      } else {
        const pick = Number(answers[qid])
        if (pick === 1) table = other
        else if (pick === 2 && shared) joinAmb.push({ _join: { table: other.name, on: shared.name, left: table.name } })
      }
      break
    }
  }
  const pendingJoin = joinAmb.filter((j) => !j._join)
  if (pendingJoin.length) return clarify(pendingJoin)
  const join = joinAmb.find((j) => j._join)?._join || null

  // ---------- column ambiguity (a term that fits several columns) ----------
  // Detect count intent first: COUNT(*) needs no identifier columns, so a
  // vague term like "people" must not force an irrelevant question.
  let earlyAgg = null
  for (const [fn, words] of Object.entries(AGG)) {
    if (words.some((w) => lower.includes(w))) { earlyAgg = fn; break }
  }

  const hits = [...hitsByTable.get(table.name)]
  const ambiguousCols = []
  const usedTokens = new Set()
  const groups = new Map() // token -> [{col,score}]
  for (const h of hits) {
    if (!groups.has(h.token)) groups.set(h.token, [])
    groups.get(h.token).push(h)
  }
  const resolvedCols = {}
  for (const [token, list] of groups) {
    const uniq = [...new Map(list.map((l) => [l.col.name, l])).values()]
    if (uniq.length > 1) {
      if (earlyAgg === 'count') continue // COUNT(*) — identifier columns are irrelevant
      const qid = `col:${token}`
      if (answers[qid] == null) {
        ambiguousCols.push({
          id: qid, kind: 'column',
          question: `“${token}” could mean a few different columns — which one?`,
          options: uniq.map((u) => ({ label: u.col.name, note: `${table.name} · ${u.col.type}` })),
        })
      } else {
        resolvedCols[token] = uniq[Number(answers[qid])]?.col || uniq[0].col
        usedTokens.add(token)
      }
    } else if (uniq[0]) {
      resolvedCols[token] = uniq[0].col
      usedTokens.add(token)
    }
  }
  if (ambiguousCols.length) return clarify(ambiguousCols)

  const selectedCols = [...new Set(Object.values(resolvedCols).map((c) => c.name))]
    .map((n) => table.columns.find((c) => c.name === n))
    .filter(Boolean)
  const dateCol = table.columns.find(isDateCol)
  const numericCols = table.columns.filter(isNumericCol)

  // ---------- Phase B: aggregate / time / threshold / value ----------
  const phaseB = []

  // aggregate intent
  let agg = null
  let aggWord = null
  for (const [fn, words] of Object.entries(AGG)) {
    for (const w of words) {
      if (lower.includes(w)) { agg = fn; aggWord = w; break }
    }
    if (agg) break
  }
  // "what was the revenue in September?" over one resolved numeric column is a
  // scalar total, even without the word "sum".
  if (!agg && /^(what's|what is|what was|whats|tell me|show me the total of)\b/.test(lower.trim())) {
    const numericSelected = selectedCols.filter(isNumericCol)
    if (numericSelected.length === 1) {
      agg = 'sum'
      aggWord = 'the question’s scalar measure'
    }
  }
  const wantsSummary = agg || /\bper\b|\bby\b|\bacross\b|\beach\b/.test(lower)

  let aggCol = null
  if (agg) {
    const numericNamed = selectedCols.filter(isNumericCol)
    if (agg === 'count') {
      aggCol = null // COUNT(*)
    } else {
      if (numericNamed.length === 1) aggCol = numericNamed[0]
      else {
        // is a numeric column named in the sentence?
        const named = numericCols.filter((c) =>
          toks.some((tk) => matchColScore(tk, c.name) >= 1),
        )
        if (named.length === 1) aggCol = named[0]
        else if (numericCols.length > 1) {
          const qid = 'agg'
          if (answers[qid] == null) {
            phaseB.push({
              id: qid, kind: 'aggregate',
              question: `Which measure should I ${agg} for “${aggWord}”?`,
              options: numericCols.slice(0, 8).map((c) => ({ label: c.name, note: `${agg.toUpperCase()}(${c.name})` })),
            })
          } else aggCol = numericCols[Number(answers[qid])] || numericCols[0]
        } else aggCol = numericCols[0] || null
      }
    }
  }

  // time phrase
  let timeKind = null
  let timeCol = dateCol
  for (const [phrase, kind] of TIME_PHRASES) {
    if (lower.includes(` ${phrase}`) || lower.includes(`${phrase} `)) {
      timeKind = kind
      break
    }
  }
  if (timeKind && timeCol) {
    const qid = 'time'
    if (answers[qid] == null) {
      phaseB.push({
        id: qid, kind: 'time',
        question: `How should I interpret the time window against \`${timeCol.name}\`?`,
        options: Object.entries(TIME_OPTS).slice(0, 7).map(([k, v]) => ({ label: v.label, note: v.note, key: k })),
      })
    } else {
      timeKind = Object.keys(TIME_OPTS)[Number(answers[qid])] || timeKind
    }
  } else if (timeKind && !timeCol) {
    timeKind = null // no date column → silently ignore, mention later
  }

  // vague magnitude → threshold question (never for COUNT: "how many" is a
  // count phrase, not a magnitude request)
  const countPhrase = /how many|number of/.test(lower)
  const vagueHit = countPhrase
    ? null
    : VAGUE.find((v) => new RegExp(`\\b${v}\\b`).test(lower))
  const hasNumber = /\b\d+\b/.test(raw)
  const wantsTop = /\btop\s*\d*\b|\bfirst\s*\d*\b|\bbottom\s*\d*\b|\blast\s+\d+\b/.test(lower)
  let threshold = null
  if (vagueHit && earlyAgg !== 'count' && !hasNumber && (agg || wantsTop || /best|top|highest|biggest|cheapest|lowest|most|popular/.test(lower)) && numericCols.length) {
    const qid = 'threshold'
    if (answers[qid] == null) {
      phaseB.push({
        id: qid, kind: 'threshold',
        question: `“${vagueHit}” needs a rule — how strict should I be?`,
        options: [
          { label: 'Top 5', note: 'ORDER BY … DESC LIMIT 5' },
          { label: 'Top 10', note: 'ORDER BY … DESC LIMIT 10' },
          { label: 'Top 25%', note: 'value >= 75th percentile' },
          { label: 'Above average', note: 'value > AVG(value)' },
        ],
      })
    } else threshold = Number(answers[qid])
  }

  // literal values in the sentence (department = Engineering …)
  const eqFilters = []
  const valueAmb = []
  for (const c of table.columns) {
    if (!c.values || c.values.length === 0 || isDateCol(c)) continue
    let best = null
    for (const v of c.values) {
      const lv = String(v).toLowerCase()
      if (lv.length < 2) continue
      const re = new RegExp(`(?<![a-z0-9])${lv.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![a-z0-9])`, 'i')
      if (re.test(raw)) {
        if (!best || lv.length > String(best).length) best = v
      }
    }
    if (best != null) {
      const qid = `value:${c.name}`
      const near = c.values.filter((v) => {
        const lv = String(v).toLowerCase()
        const gl = String(best).toLowerCase()
        return lv !== gl && (lv.startsWith(gl.slice(0, 4)) || gl.startsWith(lv.slice(0, 4)))
      })
      if (near.length && answers[qid] == null) {
        valueAmb.push({
          id: qid, kind: 'value',
          question: `For \`${c.name}\`, “${best}” — did you mean one of these?`,
          options: [best, ...near.slice(0, 4)].map((v) => ({ label: String(v), note: `${c.name} = ${v}` })),
        })
      } else {
        const chosen = answers[qid] != null ? [best, ...near.slice(0, 4)][Number(answers[qid])] : best
        eqFilters.push({ col: c.name, value: chosen })
      }
    }
  }
  if (valueAmb.length) phaseB.push(...valueAmb)

  // numeric comparison (salary over 90000)
  const numFilters = []
  const numRe = /([a-z_][a-z0-9_]*)\s*(over|above|more than|greater than|at least|>=|>|under|below|less than|at most|<=|<)\s*([0-9][0-9,.]*)/i
  const nm = raw.match(numRe)
  if (nm) {
    const col = table.columns.find((c) => matchColScore(nm[1], c.name) >= 1 && isNumericCol(c))
    if (col) {
      const val = Number(nm[3].replace(/,/g, ''))
      const op = /^(over|above|more than|greater than|at least|>=|>)$/i.test(nm[2]) ? '>' : '<'
      numFilters.push({ col: col.name, op, val })
    }
  }

  if (phaseB.length) return clarify(phaseB)

  // ---------- Phase C: build SQL ----------
  const where = []
  for (const f of eqFilters) where.push(`${q(f.col)} = '${esc(f.value)}'`)
  for (const f of numFilters) where.push(`${q(f.col)} ${f.op} ${f.val}`)
  if (timeKind && timeCol) {
    const clause = timeClause(timeCol.name, timeKind)
    if (clause) where.push(clause)
  }

  // group-by: "per X" / "by X"
  const groupBy = []
  for (const tk of toks) {
    const c = resolvedCols[tk]
    if (c && !isNumericCol(c) && !eqFilters.some((f) => f.col === c.name) && (/\bper\b|\bby\b|\bacross\b|\beach\b/.test(lower))) {
      if (!groupBy.includes(c.name)) groupBy.push(c.name)
    }
  }

  const measure =
    aggCol ||
    selectedCols.find(isNumericCol) ||
    numericCols[0] ||
    null

  // ordering
  let orderBy = null
  let limit = null
  const topM = lower.match(/\btop\s+(\d+)\b|\bfirst\s+(\d+)\b/)
  const botM = lower.match(/\bbottom\s+(\d+)\b|\blast\s+(\d+)\b/)
  const ascM = /\blowest\b|\bcheapest\b|\bsmallest\b|\bfewest\b|\bcheapest\b|\basc\b|\bascending\b|\boleast\b/.test(lower)
  const wantsTopish = /\btop\b|\bbest\b|\bhighest\b|\bbiggest\b|\blargest\b|\bmost\b|\bpopular\b|\blatest\b|\bnewest\b|\brecent\b/.test(lower)

  if (topM || botM || wantsTopish) {
    const m = topM ? Number(topM[1] || topM[2]) : botM ? Number(botM[1] || botM[2]) : null
    const dir = botM ? 'ASC' : ascM ? 'ASC' : 'DESC'
    const orderCol =
      (groupBy.length ? null : null) ||
      (wantsTopish || topM ? measure : measure) ||
      (ascM ? measure : measure) ||
      null
    if (orderCol) orderBy = { col: orderCol.name, dir: (botM && !topM) ? 'ASC' : dir }
    if (m) limit = m
    else if (threshold === 0) limit = 5
    else if (threshold === 1) limit = 10
  }
  if (!orderBy && measure && !groupBy.length && /sorted|order|rank|ranked/.test(lower)) {
    orderBy = { col: measure.name, dir: ascM ? 'ASC' : 'DESC' }
  }

  const selectList = []
  for (const g of groupBy) selectList.push(q(g))
  if (agg) {
    const expr = `${agg.toUpperCase()}(${aggCol ? q(aggCol.name) : '*'})`
    const alias = aggCol ? `${agg}_${aggCol.name}` : `${agg}_all`
    selectList.push(`${expr} AS ${q(alias)}`)
  }
  if (!agg && !groupBy.length) {
    let cols = selectedCols
    // "who was hired last month" / "show me the top earners" read better with
    // an identifier column next to the date/measure.
    const nameCol = table.columns.find((c) => c.type === 'text' && /(name|title|label)/.test(c.name))
    const asksWho = /\bwho\b|\bwhose\b|\bwhich (person|people|employee|staff|worker)/.test(lower)
    const topRanking = orderBy && measure && cols.every((c) => isNumericCol(c))
    if (nameCol && !cols.some((c) => c.name === nameCol.name)) {
      if (asksWho) cols = [nameCol, ...cols]
      else if (topRanking) cols = [nameCol, ...cols]
    }
    if (cols.length) for (const c of cols) selectList.push(q(c.name))
    else selectList.push('*')
  }
  if (agg && !groupBy.length && selectList.length === 1 && !eqFilters.length) {
    // plain scalar aggregate still needs a FROM, nothing extra
  }
  if (groupBy.length && !agg && measure) {
    // "employees per department" style → count rows
    selectList.push(`COUNT(*) AS ${q('row_count')}`)
  }

  const from = q(table.name) + (join ? ` JOIN ${q(join.table)} ON ${q(join.table)}.${q(join.on)} = ${q(join.left)}.${q(join.on)}` : '')

  let sql = `SELECT ${selectList.join(', ')} FROM ${from}`
  if (where.length) sql += ` WHERE ${where.join(' AND ')}`

  // percentile / above-average threshold modes
  if (threshold === 2 && measure) {
    const pct = `(SELECT ${q(measure.name)} FROM ${q(table.name)} ORDER BY ${q(measure.name)} ASC LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.75 AS INT) FROM ${q(table.name)}))`
    sql += (where.length ? ' AND ' : ' WHERE ') + `${q(measure.name)} >= ${pct}`
  } else if (threshold === 3 && measure) {
    const avg = `(SELECT AVG(${q(measure.name)}) FROM ${q(table.name)})`
    sql += (where.length ? ' AND ' : ' WHERE ') + `${q(measure.name)} > ${avg}`
  }

  if (groupBy.length) sql += ` GROUP BY ${groupBy.map(q).join(', ')}`
  if (orderBy) sql += ` ORDER BY ${q(orderBy.col)} ${orderBy.dir}`
  if (limit) sql += ` LIMIT ${limit}`
  else if (!groupBy.length && !agg) sql += ' LIMIT 50'
  else if (groupBy.length || agg) sql += ' LIMIT 50'

  // summary line
  const bits = []
  if (agg) bits.push(`${agg.toUpperCase()}(${aggCol ? aggCol.name : '*'})`)
  if (groupBy.length) bits.push(`grouped by ${groupBy.join(', ')}`)
  if (selectedCols.length && !agg) bits.push(`columns ${selectedCols.map((c) => c.name).join(', ')}`)
  if (eqFilters.length) bits.push(eqFilters.map((f) => `${f.col} = ${f.value}`).join(' and '))
  if (numFilters.length) bits.push(numFilters.map((f) => `${f.col} ${f.op} ${f.val}`).join(' and '))
  if (timeKind && timeCol) bits.push(`window: ${TIME_OPTS[timeKind]?.label || timeKind} on ${timeCol.name}`)
  if (orderBy) bits.push(`sorted by ${orderBy.col} ${orderBy.dir}`)
  if (limit) bits.push(`limit ${limit}`)
  if (threshold === 2) bits.push('75th-percentile cut-off')
  if (threshold === 3) bits.push('above-average cut-off')
  const summary = `Reading your request as: ${bits.join(', ') || 'plain SELECT'} — against ${table.name}.`

  return {
    status: 'ok',
    sql,
    summary,
    reply: agg || groupBy.length
      ? `Here's what I ran on \`${table.name}\`:`
      : `Here's the result from \`${table.name}\`:`,
    ambiguities: [],
    confidence: 0.95,
  }
}
