// Locate the Python interpreter that has the project's dependencies installed.
// The React UI is a front end for the Python pipeline, so `.venv` is the
// normal answer; CLARION_PYTHON is the escape hatch for other setups.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
export const PROJECT_ROOT = path.resolve(__dirname, '..', '..')

export function findPython() {
  const candidates = [
    process.env.CLARION_PYTHON,
    path.join(PROJECT_ROOT, '.venv', 'bin', 'python'),
    path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe'),
  ].filter(Boolean)
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate
  }
  return process.env.CLARION_PYTHON || 'python3'
}
