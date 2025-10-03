import os, json, sqlite3
from typing import Dict, Any, List, Optional

class Store:
    def __init__(self, db_path="taw.db", artifacts_dir="artifacts"):
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

    def init(self, schema_path="datastore/schema.sql"):
        with open(schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def ensure_artifacts_dir(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        return self.artifacts_dir

    def new_task(self, prompt:str, objective:str, params:Dict[str,Any], status:str="created")->int:
        cur = self.conn.execute(
            "INSERT INTO tasks(prompt, objective, params_json, status) VALUES(?,?,?,?)",
            (prompt, objective, json.dumps(params, ensure_ascii=False), status)
        )
        self.conn.commit()
        return cur.lastrowid

    def log_run(self, task_id:int, step:str, tool:str, inputs:Dict[str,Any], outputs:Dict[str,Any], status:str="ok")->int:
        cur = self.conn.execute(
            "INSERT INTO runs(task_id, step, tool, inputs_json, outputs_json, status) VALUES(?,?,?,?,?,?)",
            (task_id, step, tool,
             json.dumps(inputs, ensure_ascii=False),
             json.dumps(outputs, ensure_ascii=False),
             status)
        )
        self.conn.commit()
        return cur.lastrowid

    def write_metric(self, task_id:int, name:str, value:float)->int:
        cur = self.conn.execute(
            "INSERT INTO metrics(task_id, name, value) VALUES(?,?,?)",
            (task_id, name, float(value))
        )
        self.conn.commit()
        return cur.lastrowid

    # handy helpers (optional)
    def list_tasks(self)->List[dict]:
        cur = self.conn.execute("SELECT id, created_at, prompt, objective, params_json, status FROM tasks ORDER BY id DESC")
        rows = cur.fetchall()
        return [
            {"id": r[0], "created_at": r[1], "prompt": r[2], "objective": r[3],
             "params": json.loads(r[4] or "{}"), "status": r[5]}
            for r in rows
        ]

    def get_task(self, task_id:int)->Optional[dict]:
        cur = self.conn.execute("SELECT id, created_at, prompt, objective, params_json, status FROM tasks WHERE id=?", (task_id,))
        r = cur.fetchone()
        if not r: return None
        return {"id": r[0], "created_at": r[1], "prompt": r[2], "objective": r[3],
                "params": json.loads(r[4] or "{}"), "status": r[5]}

    def list_runs(self, task_id:int)->List[dict]:
        cur = self.conn.execute("""SELECT id, step, tool, inputs_json, outputs_json, started_at, ended_at, status
                                   FROM runs WHERE task_id=? ORDER BY id ASC""", (task_id,))
        rows = cur.fetchall()
        return [
            {"id": r[0], "step": r[1], "tool": r[2],
             "inputs": json.loads(r[3] or "{}"),
             "outputs": json.loads(r[4] or "{}"),
             "started_at": r[5], "ended_at": r[6], "status": r[7]}
            for r in rows
        ]

    def list_metrics(self, task_id:int)->List[dict]:
        cur = self.conn.execute("SELECT id, name, value FROM metrics WHERE task_id=? ORDER BY id ASC", (task_id,))
        rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "value": r[2]} for r in rows]
