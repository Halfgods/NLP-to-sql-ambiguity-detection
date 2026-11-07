// Tiny dependency-free CSV/TSV parser with quote handling + type inference.

export function detectDelimiter(text) {
  const candidates = [',', '\t', ';', '|']
  const counts = new Map(candidates.map((d) => [d, 0]))
  let quoted = false
  for (let i = 0; i < Math.min(text.length, 10000); i++) {
    const char = text[i]
    if (char === '"') {
      if (quoted && text[i + 1] === '"') i++
      else quoted = !quoted
    } else if (!quoted && (char === '\n' || char === '\r')) {
      break
    } else if (!quoted && counts.has(char)) {
      counts.set(char, counts.get(char) + 1)
    }
  }
  return candidates.reduce((best, d) =>
    counts.get(d) > counts.get(best) ? d : best, ',')
}

export function parseDelimited(text) {
  const delim = detectDelimiter(text)
  const rows = []
  let row = []
  let field = ''
  let inQuotes = false

  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i++
        } else {
          inQuotes = false
        }
      } else {
        field += c
      }
    } else if (c === '"') {
      inQuotes = true
    } else if (c === delim) {
      row.push(field)
      field = ''
    } else if (c === '\n') {
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else if (c !== '\r') {
      field += c
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field)
    rows.push(row)
  }

  if (inQuotes) throw new Error('unterminated quoted field')
  const clean = rows.filter((r) => r.some((v) => v.trim() !== ''))
  if (clean.length === 0) return { columns: [], rows: [] }

  const rawHeaders = clean[0].map((h, i) =>
    (i === 0 ? h.replace(/^\uFEFF/, '') : h).trim() || `column_${i + 1}`)
  if (rawHeaders.length > 500) throw new Error('CSV exceeds 500 columns')
  if (clean.length > 200001) throw new Error('CSV exceeds 200,000 data rows')
  const used = new Map()
  const headers = rawHeaders.map((h) => {
    const base = h.toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'column'
    const n = (used.get(base) ?? 0) + 1
    used.set(base, n)
    return n === 1 ? base : `${base}_${n}`
  })

  const body = clean.slice(1)
  const objects = body.map((r) => {
    if (r.length > headers.length) {
      throw new Error(`row has ${r.length} fields but the header has ${headers.length}`)
    }
    const o = {}
    headers.forEach((h, i) => {
      o[h] = (r[i] ?? '').trim()
    })
    return o
  })

  const columns = headers.map((name) => ({ name, type: inferType(objects.map((o) => o[name])) }))
  return { columns, rows: objects }
}

export function inferType(values) {
  const filled = values.filter((v) => v !== '' && v != null)
  if (filled.length === 0) return 'text'
  const isNumber = filled.every((v) => /^-?\d+(?:\.\d+)?$/.test(v.replace(/,/g, '')))
  if (isNumber) return 'number'
  const isDate = filled.every(
    (v) => /^\d{4}-\d{1,2}-\d{1,2}$/.test(v) || /^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(v),
  )
  if (isDate) return 'date'
  return 'text'
}

/** Coerce a parsed CSV row's values using the inferred column types. */
export function coerceRows(columns, rows) {
  const numeric = new Set(columns.filter((c) => c.type === 'number').map((c) => c.name))
  return rows.map((r) => {
    const o = {}
    for (const [k, v] of Object.entries(r)) {
      o[k] = numeric.has(k) && v !== '' ? Number(String(v).replace(/,/g, '')) : v
    }
    return o
  })
}
