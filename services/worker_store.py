import sqlite3, os, time

class WorkerStore:
    def __init__(self, db_path="db/workers.db"):
        self.db_path = db_path
        os.makedirs("db", exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            name TEXT PRIMARY KEY,
            trades INTEGER,
            wins INTEGER,
            losses INTEGER,
            pnl REAL,
            last_retrain TEXT
        )
        """)
        conn.commit()
        conn.close()

    def update(self, worker):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO workers (name, trades, wins, losses, pnl, last_retrain)
                      VALUES (?, ?, ?, ?, ?, ?)
                   """, (worker.name, worker.state["trades"], worker.state["wins"],
                           worker.state["losses"], worker.state["pnl"], time.ctime()))
        conn.commit()
        conn.close()
