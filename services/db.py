import sqlite3
from datetime import datetime

DB_PATH = "bot_data.db"

class Database:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS worker_stats (
            worker TEXT,
            trades INTEGER,
            wins INTEGER,
            losses INTEGER,
            pnl REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker TEXT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            pnl REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS equity_curve (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equity REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.conn.commit()

    def insert_trade(self, worker, symbol, side, qty, price, pnl):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO trades (worker, symbol, side, qty, price, pnl)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (worker, symbol, side, qty, price, pnl))
        self.conn.commit()

    def update_worker_stats(self, worker, trades, wins, losses, pnl):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO worker_stats (worker, trades, wins, losses, pnl)
            VALUES (?, ?, ?, ?, ?)
        """, (worker, trades, wins, losses, pnl))
        self.conn.commit()

    def insert_equity(self, equity):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO equity_curve (equity) VALUES (?)
        """, (equity,))
        self.conn.commit()
