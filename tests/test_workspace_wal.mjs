import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'clarion-wal-'))
process.env.CLARION_UPLOAD_DIR = tempDir
process.env.CLARION_WORKSPACE_DB = path.join(tempDir, 'workspace.db')
const workspace = await import('../UI/server/workspace.js')

test('workspace enables WAL for imports and reads', (t) => {
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))
  workspace.ensureWorkspace()
  const db = new DatabaseSync(workspace.WORKSPACE_DB)
  try {
    assert.equal(db.prepare('PRAGMA journal_mode').get().journal_mode, 'wal')
    workspace.importTable('stress_write', {
      columns: [{ name: 'value', type: 'text' }],
      rows: [{ value: 'committed' }],
    })
    assert.equal(db.prepare('SELECT value FROM stress_write').get().value, 'committed')
  } finally {
    db.close()
  }
})
