import sqlite3

import numpy as np

from utils import Embedding16


class DB:
    def __init__(self, db_path: str | None = None):
        db_path = db_path or "index.sqlite"
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._set_pragmas()
        self._init_table()
        self.commit()

    def _set_pragmas(self):
        self.cursor.execute("PRAGMA journal_mode=WAL")
        self.cursor.execute("PRAGMA synchronous=NORMAL")
        self.cursor.execute(f"PRAGMA page_size={2**13}")
        self.cursor.execute(f"PRAGMA cache_size=-{2**14}")
        self.cursor.execute(f"PRAGMA mmap_size={2**32-1}")

    def _init_table(self):
        self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anchor TEXT,
                chunk TEXT,
                embedding BLOB,
                file TEXT,
                model TEXT
            )""")

        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_anchor ON embeddings (file, anchor)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings (model)")

    def store(self, anchor: str, chunk: str, embedding: Embedding16, file: str, model: str):
        self.cursor.execute(
            """
            INSERT INTO embeddings (anchor, chunk, embedding, file, model) VALUES (?, ?, ?, ?, ?)
            """,
            (anchor, chunk, embedding.tobytes(), file, model),
        )

    def all(self, model: str) -> list[tuple[str, str, Embedding16, str]]:
        res = self.cursor.execute(
            """
            SELECT anchor, chunk, embedding, file FROM embeddings WHERE model = ?""",
            (model,),
        ).fetchall()

        # change embedding to matrix, don't loop
        return [
            dict(
                anchor=r[0],
                chunk=r[1],
                embedding=np.frombuffer(r[2], dtype=np.float16),
                file=r[3],
            )
            for r in res
        ]

    def exists(self, anchor: str, file: str, model: str, chunk: str) -> bool:
        res = self.cursor.execute(
            """
            SELECT chunk FROM embeddings WHERE anchor = ? AND file = ? AND model = ?""",
            (anchor, file, model),
        ).fetchone()

        if res is None:
            return False

        return chunk == res[0]

    def commit(self):
        self.conn.commit()
