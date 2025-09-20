import json
from pathlib import Path
import pandas as pd

LOGS_PATH = Path("logs/events.jsonl")
LABELED_LOGS_PATH = Path("logs/labeled_events.jsonl")

def load_logs():
    """Read raw event logs into a DataFrame."""
    rows = []
    if not LOGS_PATH.exists():
        print("No raw logs found.")
        return pd.DataFrame()
    with open(LOGS_PATH, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def label_trades(df):
    """
    Match order fills into round trips (buy→sell) and assign PnL labels.
    Simple FIFO method: assumes one open trade at a time per worker/symbol.
    """
    labeled_rows = []
    open_positions = {}

    for _, row in df.iterrows():
        if row["event"] != "order_fill":
            labeled_rows.append(row.to_dict())
            continue

        worker = row.get("worker", "unknown")
        symbol = row["symbol"]
        key = (worker, symbol)

        qty = row["qty"]
        price = row["price"]
        direction = row["direction"]

        # Open position
        if key not in open_positions:
            open_positions[key] = {
                "qty": qty,
                "price": price,
                "direction": direction,
                "row": row.to_dict()
            }
        else:
            # Closing trade → compute PnL
            entry = open_positions.pop(key)
            pnl = (price - entry["price"]) * entry["qty"]
            if entry["direction"] == "Sell":
                pnl = -pnl

            # Add PnL label to both entry and exit rows
            entry["label"] = 1 if pnl > 0 else 0
            row_dict = row.to_dict()
            row_dict["label"] = 1 if pnl > 0 else 0

            labeled_rows.append(entry)
            labeled_rows.append(row_dict)

    # Carry over unmatched trades without labels
    for entry in open_positions.values():
        labeled_rows.append(entry)

    return labeled_rows


def save_labeled(rows):
    LABELED_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELED_LOGS_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Labeled logs written to {LABELED_LOGS_PATH}")


def main():
    df = load_logs()
    if df.empty:
        print("No logs to label.")
        return

    labeled = label_trades(df)
    save_labeled(labeled)


if __name__ == "__main__":
    main()
