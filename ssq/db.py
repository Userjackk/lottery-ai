import sqlite3
from contextlib import contextmanager
from .config import DB_PATH, DATA_DIR


@contextmanager
def get_conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS draws (
                draw_no   TEXT PRIMARY KEY,
                date      TEXT NOT NULL,
                red1      INTEGER NOT NULL,
                red2      INTEGER NOT NULL,
                red3      INTEGER NOT NULL,
                red4      INTEGER NOT NULL,
                red5      INTEGER NOT NULL,
                red6      INTEGER NOT NULL,
                blue      INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS models (
                name       TEXT PRIMARY KEY,
                version    INTEGER NOT NULL DEFAULT 1,
                trained_on TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                pickle     BLOB NOT NULL
            );
        """)


def insert_draws(draws: list[dict]) -> int:
    if not draws:
        return 0
    with get_conn() as conn:
        existing = {row[0] for row in conn.execute("SELECT draw_no FROM draws")}
        new = [d for d in draws if d["draw_no"] not in existing]
        if not new:
            return 0
        conn.executemany(
            "INSERT INTO draws (draw_no, date, red1, red2, red3, red4, red5, red6, blue) "
            "VALUES (:draw_no, :date, :red1, :red2, :red3, :red4, :red5, :red6, :blue)",
            new,
        )
        return len(new)


def get_all_draws() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM draws ORDER BY draw_no ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_draws_since(n: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM draws ORDER BY draw_no DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def get_latest_draw_number() -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT draw_no FROM draws ORDER BY draw_no DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None


def get_draw_count() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) FROM draws").fetchone()
        return row[0]


def save_model(name: str, trained_on: str, data: bytes):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO models (name, trained_on, pickle) VALUES (?, ?, ?)",
            (name, trained_on, data),
        )


def load_model(name: str) -> tuple[str, bytes] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT trained_on, pickle FROM models WHERE name = ?", (name,)
        ).fetchone()
        return (row["trained_on"], row["pickle"]) if row else None


def get_next_draw_number() -> str:
    latest = get_latest_draw_number()
    if not latest:
        return ""
    year = int(latest[:4])
    seq = int(latest[4:]) + 1
    return f"{year}{seq:03d}"
