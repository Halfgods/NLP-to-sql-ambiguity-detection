// Verify CSV and SQLite imports persist in an isolated workspace.

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'clarion-upload-smoke-'))
process.env.CLARION_UPLOAD_DIR = temp
process.env.CLARION_WORKSPACE_DB = path.join(temp, 'workspace.db')
const workspace = await import('../UI/server/workspace.js')

try {
  const csvFile = path.join(temp, 'contacts.csv')
  fs.writeFileSync(csvFile, 'id,name,city\n1,Ada Lovelace,London\n')
  const parsed = workspace.parseUploadFile(csvFile, '.csv')
  workspace.importTable('contacts', parsed)

  const sourceDb = path.join(temp, 'school.db')
  const source = new DatabaseSync(sourceDb)
  source.exec('CREATE TABLE courses (id INTEGER PRIMARY KEY, name TEXT, credits INTEGER)')
  source.prepare('INSERT INTO courses VALUES (?, ?, ?)').run(1, 'Databases', 3)
  source.close()
  const imported = workspace.importSQLiteFile(sourceDb, 'school')
  assert.equal(imported.length, 1)
  assert.equal(imported[0].name, 'school_courses')

  const db = new DatabaseSync(process.env.CLARION_WORKSPACE_DB)
  db.prepare('UPDATE contacts SET city = ? WHERE id = ?').run('Oxford', 1)
  db.prepare('UPDATE school_courses SET credits = ? WHERE id = ?').run(4, 1)
  db.close()

  // Schema refreshes must not reseed over confirmed changes.
  workspace.listSchema(new Set(['contacts', 'school_courses']))
  const check = new DatabaseSync(process.env.CLARION_WORKSPACE_DB, { readOnly: true })
  assert.equal(check.prepare('SELECT city FROM contacts WHERE id = 1').get().city, 'Oxford')
  assert.equal(check.prepare('SELECT credits FROM school_courses WHERE id = 1').get().credits, 4)
  check.close()
  assert.match(fs.readFileSync(csvFile, 'utf8'), /London/)
  const original = new DatabaseSync(sourceDb, { readOnly: true })
  assert.equal(original.prepare('SELECT credits FROM courses WHERE id = 1').get().credits, 3)
  original.close()
  console.log('PASS: CSV and SQLite tables import, remain writable in the workspace after schema refresh, and leave source files/databases unchanged.')
} finally {
  fs.rmSync(temp, { recursive: true, force: true })
}
