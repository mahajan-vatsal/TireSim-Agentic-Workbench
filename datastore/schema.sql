-- Tables for Phase A: enough to verify "init" works

CREATE TABLE IF NOT EXISTS tasks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  prompt TEXT,
  objective TEXT,
  params_json TEXT,
  status TEXT
);

CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  step TEXT,
  tool TEXT,
  inputs_json TEXT,
  outputs_json TEXT,
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  ended_at TEXT,
  status TEXT,
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS metrics(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  name TEXT,
  value REAL,
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);
