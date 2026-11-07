// The SQLite workspace the Qwen pipeline actually queries.
//
// The demo data lives in two interconnected files owned by the Python project
// (`data/people.db`, `data/academics.db`, built by
// `scripts/build_demo_databases.py`): `academics.enrollments` bridges to
// `people.students` across the two files. When the dev server starts we copy
// their tables into `UI/uploads/clarion.db` and add uploaded CSV/TSV/JSON
// files as real tables, so the same dynamic schema retrieval / validation /
// read-only execution path applies to demo and uploaded data alike.
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { DatabaseSync } from 'node:sqlite'
import { parseDelimited, coerceRows } from './csv.js'
import { PROJECT_ROOT, findPython } from './python.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
export const UPLOAD_DIR = path.resolve(
  process.env.CLARION_UPLOAD_DIR || path.join(__dirname, '..', 'uploads'))
export const WORKSPACE_DB = path.resolve(
  process.env.CLARION_WORKSPACE_DB || path.join(UPLOAD_DIR, 'clarion.db'))
const DEMO_DBS = [
  path.join(PROJECT_ROOT, 'data', 'people.db'),
  path.join(PROJECT_ROOT, 'data', 'academics.db'),
]

/** `scripts/build_demo_databases.py` owns the interlinked demo data. */
function ensureDemoDatabases() {
  if (DEMO_DBS.every((p) => fs.existsSync(p))) return true
  const result = spawnSync(
    findPython(),
    [path.join(PROJECT_ROOT, 'scripts', 'build_demo_databases.py')],
    { cwd: PROJECT_ROOT, stdio: 'ignore' },
  )
  return result.status === 0 && DEMO_DBS.every((p) => fs.existsSync(p))
}

function configureWorkspaceConnection(db, enableWal = false) {
  db.exec('PRAGMA busy_timeout=5000')
  if (enableWal && db.prepare('PRAGMA journal_mode').get().journal_mode !== 'wal') {
    db.exec('PRAGMA journal_mode=WAL')
  }
}

export function ensureWorkspace() {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true })
  if (!ensureDemoDatabases()) {
    if (!fs.existsSync(WORKSPACE_DB)) {
      const db = new DatabaseSync(WORKSPACE_DB)
      try {
        configureWorkspaceConnection(db, true)
      } finally {
        db.close()
      }
    }
    return
  }

  const registryPath = path.join(UPLOAD_DIR, 'registry.json')
  let uploadedTables = new Set()
  try {
    uploadedTables = new Set(JSON.parse(fs.readFileSync(registryPath, 'utf8'))
      .flatMap((entry) => entry.tables?.map((table) => table.name)
        || (entry.table ? [entry.table] : [])))
  } catch { /* a missing registry means there are no registered uploads */ }

  const db = new DatabaseSync(WORKSPACE_DB)
  try {
    configureWorkspaceConnection(db, true)
    DEMO_DBS.forEach((src, i) => {
      db.exec(`ATTACH DATABASE '${src.replaceAll("'", "''")}' AS demo${i}`)
      const tables = db.prepare(
        `SELECT name FROM demo${i}.sqlite_master ` +
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").all()
      for (const { name } of tables) {
        if (uploadedTables.has(name)) continue
        const quoted = `"${name.replaceAll('"', '""')}"`
        // Seed demo tables only when absent. Re-copying on each API/schema
        // refresh erased confirmed INSERT/UPDATE changes from the workspace.
        db.exec(`CREATE TABLE IF NOT EXISTS main.${quoted} AS SELECT * FROM demo${i}.${quoted}`)
      }
    })
  } finally {
    db.close()
  }
}

export function resetWorkspace() {
  fs.rmSync(WORKSPACE_DB, { force: true })
  ensureWorkspace()
}

function open() {
  ensureWorkspace()
  const db = new DatabaseSync(WORKSPACE_DB)
  configureWorkspaceConnection(db)
  return db
}

export function safeTable(base) {
  const t = String(base || '')
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '')
  return (t || 'dataset').slice(0, 40)
}

export function tableNames() {
  const db = open()
  try {
    return db
      .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
      .all()
      .map((r) => r.name)
  } finally {
    db.close()
  }
}

export function uniqueTable(base) {
  const existing = new Set(tableNames())
  let name = safeTable(base)
  let i = 1
  while (existing.has(name)) name = `${safeTable(base).slice(0, 36)}_${i++}`
  return name
}

/** Import every ordinary table from an uploaded SQLite database as workspace copies.
 *
 * `dropTables` are removed inside the import's own transaction — same-name
 * re-uploads use it to replace a previous import atomically (new tables may
 * then reuse the old table names, since they were dropped from within this
 * transaction's snapshot).
 */
export function importSQLiteFile(fullPath, fileBase, dropTables = []) {
  const source = new DatabaseSync(fullPath, { readOnly: true })
  let target
  try {
    target = open()
    const sourceTables = source.prepare(
      "SELECT name, sql FROM sqlite_master WHERE type = 'table' " +
      "AND name NOT LIKE 'sqlite_%' ORDER BY name",
    ).all()
    if (!sourceTables.length) throw new Error('database contains no importable tables')
    if (sourceTables.length > 100) throw new Error('database has more than 100 tables')
    if (sourceTables.some((t) => /^\s*CREATE\s+VIRTUAL\s+TABLE/i.test(t.sql || ''))) {
      throw new Error('virtual tables are not supported')
    }

    target.exec('BEGIN IMMEDIATE')
    for (const name of dropTables) {
      target.exec(`DROP TABLE IF EXISTS ${quote(name)}`)
    }
    const existing = new Set(target.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    ).all().map((r) => r.name.toLowerCase()))
    const imported = []
    for (const sourceTable of sourceTables) {
      const sourceName = sourceTable.name
      const columns = source.prepare(`PRAGMA table_info(${quote(sourceName)})`).all()
      if (!columns.length) continue
      const base = safeTable(`${fileBase}_${sourceName}`).slice(0, 40)
      let table = base
      let suffix = 1
      while (existing.has(table.toLowerCase())) {
        const tail = `_${suffix++}`
        table = `${base.slice(0, 40 - tail.length)}${tail}`
      }
      existing.add(table.toLowerCase())

      const primary = columns.filter((c) => c.pk).sort((a, b) => a.pk - b.pk)
      const defs = columns.map((c) => {
        const bits = [quote(c.name), c.type || '']
        if (primary.length === 1 && c.pk && /^INTEGER$/i.test(c.type || '')) {
          bits.push('PRIMARY KEY')
        } else if (c.notnull) {
          bits.push('NOT NULL')
        }
        return bits.filter(Boolean).join(' ')
      })
      if (primary.length > 1 || (primary.length === 1 && !/^INTEGER$/i.test(primary[0].type || ''))) {
        defs.push(`PRIMARY KEY (${primary.map((c) => quote(c.name)).join(', ')})`)
      }
      target.exec(`CREATE TABLE ${quote(table)} (${defs.join(', ')})`)

      const names = columns.map((c) => c.name)
      const insert = target.prepare(
        `INSERT INTO ${quote(table)} (${names.map(quote).join(', ')}) ` +
        `VALUES (${names.map(() => '?').join(', ')})`,
      )
      let rows = 0
      for (const row of source.prepare(`SELECT ${names.map(quote).join(', ')} FROM ${quote(sourceName)}`).iterate()) {
        insert.run(...names.map((name) => row[name]))
        rows++
      }
      imported.push({
        name: table,
        sourceName,
        rows,
        columns: columns.map((c) => ({ name: c.name, type: uiType(c.type) })),
      })
    }
    if (!imported.length) throw new Error('database has no tables with columns')
    target.exec('COMMIT')
    return imported
  } catch (err) {
    try { target?.exec('ROLLBACK') } catch { /* no transaction left */ }
    throw err
  } finally {
    source.close()
    target?.close()
  }
}

export function readWorkspaceInfo() {
  const tables = tableNames()
  return {
    database: path.relative(PROJECT_ROOT, WORKSPACE_DB),
    tables: tables.length,
    tableNames: tables,
  }
}

export function dropTable(table) {
  const db = open()
  try {
    db.exec(`DROP TABLE IF EXISTS ${quote(table)}`)
  } finally {
    db.close()
  }
}

export function tablesForDataset(entry) {
  return entry.tables?.map((table) => table.name)
    || (entry.table ? [entry.table] : [])
}

/** Read CSV/TSV/JSON into {columns:[{name,type}], rows:[{col:value}]}. */
export function parseUploadFile(fullPath, ext) {
  const text = fs.readFileSync(fullPath, 'utf8')
  if (ext === '.json') return parseJsonExport(text)
  const parsed = parseDelimited(text)
  parsed.rows = coerceRows(parsed.columns, parsed.rows)
  return parsed
}

function parseJsonExport(text) {
  const data = JSON.parse(text)
  const list = Array.isArray(data)
    ? data
    : Array.isArray(data?.rows) ? data.rows
      : Array.isArray(data?.data) ? data.data
        : null
  if (!list) throw new Error('expected a JSON array of objects (or {rows: [...]})')

  const rawKeys = []
  for (const row of list) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) continue
    for (const key of Object.keys(row)) if (!rawKeys.includes(key)) rawKeys.push(key)
  }
  const keys = uniqueHeaders(rawKeys)
  const rows = list.map((row) => {
    const out = {}
    keys.forEach(({ source, name }) => {
      out[name] = row && typeof row === 'object' ? row[source] : undefined
    })
    return out
  }).filter((row) => Object.values(row).some((v) => v !== '' && v != null))

  if (keys.length === 0) return { columns: [], rows: [] }

  const columns = keys.map(({ name }) => ({
    name,
    type: jsonColumnType(rows.map((r) => r[name])),
  }))
  const numeric = new Set(columns.filter((c) => c.type === 'number').map((c) => c.name))
  for (const row of rows) {
    for (const col of numeric) {
      const v = row[col]
      row[col] = v === '' || v == null ? null : Number(String(v).replace(/,/g, ''))
    }
  }
  return { columns, rows }
}

function uniqueHeaders(rawKeys) {
  const used = new Map()
  return rawKeys.map((source) => {
    const base = safeTable(source)
    const n = (used.get(base) || 0) + 1
    used.set(base, n)
    return { source, name: n === 1 ? base : `${base}_${n}` }
  })
}

function jsonColumnType(values) {
  const filled = values.filter((v) => v !== '' && v != null)
  if (filled.length === 0) return 'text'
  if (filled.every((v) => typeof v === 'number'
    || (typeof v === 'string' && /^-?\d+(?:\.\d+)?$/.test(v.replace(/,/g, ''))))) {
    return 'number'
  }
  if (filled.every((v) => typeof v === 'string' && /^\d{4}-\d{1,2}-\d{1,2}$/.test(v))) {
    return 'date'
  }
  return 'text'
}

/** Create/replace `table` with the parsed rows, inside one transaction.
 *
 * `dropTables` are removed in the same transaction before the create — the
 * replacement path (same-name re-upload) relies on this so a failed import
 * rolls back to the previous data intact.
 */
export function importTable(table, parsed, dropTables = []) {
  if (!parsed.columns.length) throw new Error('no columns to import')
  const db = open()
  try {
    db.exec('BEGIN')
    for (const name of dropTables) {
      db.exec(`DROP TABLE IF EXISTS ${quote(name)}`)
    }
    db.exec(`DROP TABLE IF EXISTS ${quote(table)}`)

    const defs = parsed.columns.map((c) => {
      let sqlType = 'TEXT'
      if (c.type === 'number') {
        const allInts = parsed.rows.every((r) => {
          const v = r[c.name]
          return v == null || Number.isInteger(Number(v))
        })
        sqlType = allInts ? 'INTEGER' : 'REAL'
      }
      return `${quote(c.name)} ${sqlType}`
    })
    db.exec(`CREATE TABLE ${quote(table)} (${defs.join(', ')})`)

    const statement = db.prepare(
      `INSERT INTO ${quote(table)} (${parsed.columns.map((c) => quote(c.name)).join(', ')}) `
      + `VALUES (${parsed.columns.map(() => '?').join(', ')})`,
    )
    for (const row of parsed.rows) {
      statement.run(...parsed.columns.map((c) => normalizeValue(row[c.name])))
    }
    db.exec('COMMIT')
  } catch (err) {
    try { db.exec('ROLLBACK') } catch { /* no transaction left */ }
    throw err
  } finally {
    db.close()
  }
}

function normalizeValue(value) {
  if (value === undefined || value === '') return null
  if (typeof value === 'boolean') return value ? 1 : 0
  if (ArrayBuffer.isView(value)) return value
  if (typeof value === 'object') return JSON.stringify(value)
  return value
}

/** Schema exactly as the sidebar displays it, read from SQLite. */
export function listSchema(uploadTables = new Set()) {
  const db = open()
  try {
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
      .all()
      .map((r) => r.name)

    return tables.map((name) => {
      const info = db.prepare(`PRAGMA table_info(${quote(name)})`).all()
      const count = db.prepare(`SELECT COUNT(*) AS n FROM ${quote(name)}`).get()
      return {
        name,
        source: uploadTables.has(name) ? 'upload' : 'demo',
        rows: Number(count?.n || 0),
        columns: info.map((c) => ({
          name: c.name,
          type: uiType(c.type),
        })),
      }
    })
  } finally {
    db.close()
  }
}

function uiType(sqlType) {
  const t = String(sqlType || '').toUpperCase()
  if (/INT|REAL|FLOA|DOUB|NUM|DEC/.test(t)) return 'number'
  if (/DATE|TIME/.test(t) && !/DATETIME/.test(t)) return 'date'
  return 'text'
}

function quote(identifier) {
  return `"${String(identifier).replace(/"/g, '""')}"`
}
