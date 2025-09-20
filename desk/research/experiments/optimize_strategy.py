
import itertools
import pandas as pd
from strategies.vwap_revert import VWAPRevertStrategy

def main():
    # Load bars
    df = pd.read_csv("sample_bars.csv")

    thresholds = [0.005, 0.01, 0.02]
    lookbacks = [10, 20, 30]

    best = None
    for t, lb in itertools.product(thresholds, lookbacks):
        strat = VWAPRevertStrategy("BTCUSD", {"lookback": lb, "threshold": t})
        pnl = 0
        for _, row in df.iterrows():
            class Bar: pass
            bar = Bar()
            bar.Close, bar.Open, bar.High, bar.Low, bar.Volume = row["close"], row["open"], row["high"], row["low"], row["volume"]
            if strat.passes_rules(bar):
                pnl += strat.score_edge(bar) * (bar.Close - bar.Open)
        if not best or pnl > best[0]:
            best = (pnl, t, lb)

    print("Best config:", best)

if __name__ == "__main__":
    main()
