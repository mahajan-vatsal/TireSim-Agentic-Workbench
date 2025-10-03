# datastore/store.py
import sqlite3
from pathlib import Path

class Store:
    def __init__(self, db_path: str = "taw.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")

    def init(self, schema_path: str = "datastore/schema.sql") -> None:
        schema_file = Path(schema_path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        with schema_file.open("r", encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()
