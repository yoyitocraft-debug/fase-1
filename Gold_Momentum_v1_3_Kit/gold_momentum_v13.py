#!/usr/bin/env python3
"""Gold Momentum v1.3 research backtest and daily paper-signal helper.

Implements the attached specification's price rules without optimization:
- GLD daily bars
- SMA(100) and 60-day momentum
- Long only; enter when Close > SMA100 and Momentum60 > 0
- Exit when Close < SMA100 OR Momentum60 <= 0
- Signal at close, execute at next session's open (no look-ahead)
- Volatility-aware position size: risk budget / (2 x daily volatility x price),
  capped by available cash. The 2x volatility distance is an explicit proxy,
  because the specification leaves the volatility multiple / risk distance open.
- Slippage per side varies with whether the prior day's range is > 2x its
  trailing 20-session average. Commission is a configurable per-side rate.

Research tool only. Does not send orders or connect to a broker.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "gold_data"
REPORT_DIR = SCRIPT_DIR / "gold_reports"
TRADING_DAYS = 252

@dataclass
class CostSettings:
    commission: float = 0.0       # per side; e.g. 0.001 = 0.10%
    normal_slippage: float = 0.0006  # 0.06% per side, within spec's 0.05-0.08%
    high_slippage: float = 0.0015    # 0.15% per side, within spec's 0.12-0.20%
    risk_fraction: float = 0.005
    vol_multiple: float = 2.0
    initial_cash: float = 10000.0


def drop_incomplete_current_session(data: pd.DataFrame) -> pd.DataFrame:
    """Do not use today's potentially unfinished daily candle as a completed close signal."""
    if data.empty:
        return data
    now_et = pd.Timestamp.now(tz="America/New_York")
    last_date = pd.Timestamp(data.index[-1]).date()
    if last_date == now_et.date() and (now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 15)):
        return data.iloc[:-1].copy()
    return data


def download_data(start: str, end: str | None, refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"GLD_{start}_{end or 'latest'}.csv"
    if cache.exists() and not refresh:
        data = pd.read_csv(cache, index_col=0, parse_dates=True)
        if not data.empty and {"Open", "High", "Low", "Close"}.issubset(data.columns):
            data.index = pd.to_datetime(data.index).tz_localize(None)
            return drop_incomplete_current_session(data.sort_index())

    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Missing dependency yfinance. Run: python -m pip install yfinance pandas numpy")
    print(f"Downloading GLD daily data from Yahoo Finance: {start} to {end or 'latest'}...")
    # auto_adjust=True keeps OHLC consistently adjusted for splits/distributions.
    raw = yf.download("GLD", start=start, end=end, interval="1d", auto_adjust=True,
                      actions=False, progress=False, multi_level_index=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Check your internet connection, dates, or try --refresh.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {missing}. Columns received: {list(raw.columns)}")
    data = raw[required].dropna().copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "Date"
    data = drop_incomplete_current_session(data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache)
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["SMA100"] = d["Close"].rolling(100, min_periods=100).mean()
    d["Momentum60"] = d["Close"] / d["Close"].shift(60) - 1.0
    d["DailyReturn"] = d["Close"].pct_change()
    d["Vol20"] = d["DailyReturn"].rolling(20, min_periods=20).std(ddof=1) * math.sqrt(TRADING_DAYS)
    d["RangePct"] = (d["High"] - d["Low"]) / d["Close"].shift(1)
    d["AvgRange20"] = d["RangePct"].rolling(20, min_periods=20).mean()
    d["HighVolDay"] = d["RangePct"] > 2.0 * d["AvgRange20"]
    ready = d["SMA100"].notna() & d["Momentum60"].notna() & d["Vol20"].notna()
    d["Ready"] = ready
    d["EntrySignal"] = ready & (d["Close"] > d["SMA100"]) & (d["Momentum60"] > 0)
    d["ExitSignal"] = ready & ((d["Close"] < d["SMA100"]) | (d["Momentum60"] <= 0))
    return d


def max_drawdown(equity: pd.Series) -> float:
    clean = equity.dropna()
    if clean.empty:
        return float("nan")
    peak = clean.cummax()
    dd = clean / peak - 1.0
    return float(dd.min())


def longest_underwater_days(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    under = equity < equity.cummax() * (1 - 1e-12)
    groups = (under != under.shift(fill_value=False)).cumsum()
    lengths = under[under].groupby(groups[under]).size()
    return int(lengths.max()) if not lengths.empty else 0


def max_losing_streak(trades: pd.DataFrame) -> int:
    if trades.empty or "NetPnL" not in trades:
        return 0
    best = cur = 0
    for pnl in trades["NetPnL"]:
        if pnl < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def run_backtest(data: pd.DataFrame, costs: CostSettings) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Simulate orders at today's open using only yesterday's completed close signal."""
    d = add_indicators(data)
    if len(d) < 150:
        raise RuntimeError(f"Only {len(d)} rows downloaded; at least 150 daily bars are needed for warm-up and a minimal test.")

    cash = float(costs.initial_cash)
    shares = 0
    entry_date = None
    entry_equity = 0.0
    entry_raw = entry_exec = 0.0
    entry_qty = 0
    entry_commission_slippage = 0.0
    trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    buy_count = sell_count = 0

    # At open[i], act on the signal known after close[i-1].
    for i in range(1, len(d)):
        date = d.index[i]
        signal_date = d.index[i - 1]
        row_sig = d.iloc[i - 1]
        row_exec = d.iloc[i]
        price = float(row_exec["Open"])
        if not np.isfinite(price) or price <= 0:
            equity_rows.append({"Date": date, "Equity": cash + shares * float(row_exec["Close"]), "Cash": cash, "Shares": shares, "InMarket": shares > 0})
            continue

        # Slippage is chosen using information available at prior close only.
        is_high_vol = bool(row_sig.get("HighVolDay", False)) if pd.notna(row_sig.get("HighVolDay", False)) else False
        slip = costs.high_slippage if is_high_vol else costs.normal_slippage
        ready = bool(row_sig["Ready"])
        enter_signal = ready and bool(row_sig["EntrySignal"])
        exit_signal = ready and bool(row_sig["ExitSignal"])

        if shares > 0 and exit_signal:
            proceeds_gross = shares * price * (1.0 - slip)
            commission_paid = proceeds_gross * costs.commission
            proceeds_net = proceeds_gross - commission_paid
            pnl = proceeds_net - (entry_qty * entry_raw + entry_commission_slippage)
            # Entry basis equals total cash paid (raw notional plus entry slippage/commission).
            entry_cost = entry_qty * entry_raw + entry_commission_slippage
            account_return = pnl / entry_equity if entry_equity > 0 else np.nan
            trade_rows.append({
                "EntryDate": entry_date, "ExitSignalDate": signal_date, "ExitDate": date,
                "EntryRawOpen": entry_raw, "EntryExecPrice": entry_exec,
                "ExitRawOpen": price, "ExitExecPrice": price * (1.0 - slip),
                "Shares": entry_qty, "EntryEquity": entry_equity, "GrossPnL": entry_qty * (price - entry_raw),
                "EntryAndExitCosts": entry_cost - entry_qty * entry_raw + (shares * price - proceeds_net),
                "NetPnL": pnl, "ReturnPct": pnl / entry_cost if entry_cost else np.nan,
                "AccountReturnPct": account_return * 100 if np.isfinite(account_return) else np.nan,
                "ExitSlippagePct": slip, "HighVolExecution": is_high_vol,
            })
            cash += proceeds_net
            shares = 0
            entry_date = None
            entry_equity = 0.0
            entry_raw = entry_exec = 0.0
            entry_qty = 0
            entry_commission_slippage = 0.0
            sell_count += 1

        elif shares == 0 and enter_signal:
            equity_before = cash
            vol_ann = float(row_sig["Vol20"]) if pd.notna(row_sig["Vol20"]) else np.nan
            daily_vol = vol_ann / math.sqrt(TRADING_DAYS) if np.isfinite(vol_ann) else np.nan
            if not np.isfinite(daily_vol) or daily_vol <= 0:
                qty = 0
            else:
                risk_budget = equity_before * costs.risk_fraction
                risk_distance_per_share = price * daily_vol * costs.vol_multiple
                qty_by_risk = int(math.floor(risk_budget / risk_distance_per_share)) if risk_distance_per_share > 0 else 0
                unit_cost = price * (1.0 + slip) * (1.0 + costs.commission)
                qty_by_cash = int(math.floor(cash / unit_cost)) if unit_cost > 0 else 0
                qty = max(0, min(qty_by_risk, qty_by_cash))
            if qty > 0:
                entry_exec = price * (1.0 + slip)
                purchase_gross = qty * entry_exec
                commission_paid = purchase_gross * costs.commission
                total_purchase = purchase_gross + commission_paid
                # Fees/slippage are charged to cash, not omitted from equity.
                cash -= total_purchase
                shares = qty
                entry_date = date
                entry_equity = equity_before
                entry_raw = price
                entry_qty = qty
                entry_commission_slippage = total_purchase - qty * entry_raw
                buy_count += 1

        mark_price = float(row_exec["Open"])
        equity_rows.append({"Date": date, "Equity": cash + shares * mark_price, "Cash": cash, "Shares": shares, "InMarket": shares > 0})

    equity = pd.DataFrame(equity_rows).set_index("Date")
    # Mark any still-open position at the last available close; do not invent an exit.
    if len(equity) and shares > 0:
        last_close = float(d.iloc[-1]["Close"])
        equity.iloc[-1, equity.columns.get_loc("Equity")] = cash + shares * last_close

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["EntryDate", "ExitSignalDate", "ExitDate", "EntryRawOpen", "EntryExecPrice", "ExitRawOpen", "ExitExecPrice", "Shares", "EntryEquity", "GrossPnL", "EntryAndExitCosts", "NetPnL", "ReturnPct", "AccountReturnPct", "ExitSlippagePct", "HighVolExecution"])

    eq = equity["Equity"].astype(float)
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / TRADING_DAYS) if len(equity) else np.nan
    initial = costs.initial_cash
    final = float(eq.iloc[-1]) if len(eq) else initial
    cagr = (final / initial) ** (1 / years) - 1 if final > 0 and years > 0 else np.nan
    sharpe = float((ret.mean() / ret.std(ddof=1)) * math.sqrt(TRADING_DAYS)) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    winners = trades.loc[trades["NetPnL"] > 0, "NetPnL"] if not trades.empty else pd.Series(dtype=float)
    losers = trades.loc[trades["NetPnL"] < 0, "NetPnL"] if not trades.empty else pd.Series(dtype=float)
    gross_profit = float(winners.sum()) if not winners.empty else 0.0
    gross_loss_abs = float(abs(losers.sum())) if not losers.empty else 0.0
    pf = gross_profit / gross_loss_abs if gross_loss_abs > 0 else (float("inf") if gross_profit > 0 else np.nan)
    metrics: dict[str, Any] = {
        "Start": str(d.index.min().date()), "End": str(d.index.max().date()), "Bars": len(d),
        "InitialCapital": initial, "FinalEquity": final, "NetReturnPct": (final / initial - 1) * 100,
        "CAGR_Pct": cagr * 100, "Sharpe": sharpe, "MaxDrawdownPct": max_drawdown(eq) * 100,
        "ClosedTrades": len(trades), "WinRatePct": float((trades["NetPnL"] > 0).mean() * 100) if len(trades) else np.nan,
        "ProfitFactor": pf, "AvgWinner": float(winners.mean()) if len(winners) else np.nan,
        "AvgLoser": float(losers.mean()) if len(losers) else np.nan,
        "MaxLosingStreak": max_losing_streak(trades), "MaxUnderwaterDays": longest_underwater_days(eq),
        "ExposurePct": float(equity["InMarket"].mean() * 100) if len(equity) else np.nan,
        "Buys": buy_count, "Sells": sell_count, "OpenPositionAtEnd": bool(shares > 0),
        "CommissionPerSidePct": costs.commission * 100,
        "NormalSlippagePerSidePct": costs.normal_slippage * 100,
        "HighVolSlippagePerSidePct": costs.high_slippage * 100,
        "RiskFractionPct": costs.risk_fraction * 100, "VolMultiple": costs.vol_multiple,
    }
    return d, trades, {**metrics, "_equity": equity}


def money(v: Any) -> str:
    return "N/A" if not np.isfinite(v) else f"${v:,.2f}"


def print_metrics(label: str, m: dict[str, Any]) -> None:
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
    rows = [
        ("Period", f"{m['Start']} → {m['End']} ({m['Bars']} daily bars)"),
        ("Initial capital", money(m["InitialCapital"])),
        ("Final equity", money(m["FinalEquity"])),
        ("Net return", f"{m['NetReturnPct']:.2f}%"),
        ("Annualized return (CAGR)", f"{m['CAGR_Pct']:.2f}%"),
        ("Sharpe ratio", f"{m['Sharpe']:.3f}" if np.isfinite(m["Sharpe"]) else "N/A"),
        ("Maximum drawdown", f"{m['MaxDrawdownPct']:.2f}%"),
        ("Closed trades", str(m["ClosedTrades"])),
        ("Win rate", f"{m['WinRatePct']:.2f}%" if np.isfinite(m["WinRatePct"]) else "N/A"),
        ("Profit factor", f"{m['ProfitFactor']:.3f}" if np.isfinite(m["ProfitFactor"]) else "N/A"),
        ("Average winner", money(m["AvgWinner"])),
        ("Average loser", money(m["AvgLoser"])),
        ("Max consecutive losing trades", str(m["MaxLosingStreak"])),
        ("Max underwater duration", f"{m['MaxUnderwaterDays']} trading sessions"),
        ("Exposure", f"{m['ExposurePct']:.2f}%"),
        ("Commission, per side", f"{m['CommissionPerSidePct']:.3f}%"),
        ("Slippage normal / high vol, per side", f"{m['NormalSlippagePerSidePct']:.3f}% / {m['HighVolSlippagePerSidePct']:.3f}%"),
        ("Open position at end", str(m["OpenPositionAtEnd"])),
    ]
    for k, v in rows:
        print(f"{k:<35} {v}")


def walk_forward_report(data: pd.DataFrame, costs: CostSettings, first_test_years: int = 4) -> pd.DataFrame:
    """Fixed-parameter, calendar-year OOS check; no parameter optimization is performed."""
    d, trades, result = run_backtest(data, costs)
    equity = result["_equity"]["Equity"].copy()
    rows = []
    start_year = int(d.index.min().year + first_test_years + 1)
    end_year = int(d.index.max().year)
    for year in range(start_year, end_year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        segment = equity[(equity.index >= start) & (equity.index < end)]
        if len(segment) < 20:
            continue
        # Use equity immediately before segment as comparison base when available.
        before = equity[equity.index < start]
        base = float(before.iloc[-1]) if len(before) else float(segment.iloc[0])
        final = float(segment.iloc[-1])
        period_return = (final / base - 1.0) * 100 if base else np.nan
        rets = segment.pct_change().dropna()
        sharpe = (rets.mean() / rets.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan
        rows.append({"OOS_Year": year, "StartEquity": base, "EndEquity": final, "OOS_ReturnPct": period_return,
                     "OOS_Sharpe": sharpe, "OOS_MaxDDPct": max_drawdown(segment) * 100, "Sessions": len(segment)})
    return pd.DataFrame(rows)


def crisis_report(data: pd.DataFrame, costs: CostSettings) -> pd.DataFrame:
    d, _, result = run_backtest(data, costs)
    eq = result["_equity"]["Equity"]
    exposed = result["_equity"]["InMarket"]
    windows = [
        ("2008 financial crisis", "2008-01-01", "2009-01-01"),
        ("Gold top / correction", "2011-01-01", "2014-01-01"),
        ("2018", "2018-01-01", "2019-01-01"),
        ("COVID liquidity shock", "2020-03-01", "2020-04-01"),
        ("2022 rate hikes", "2022-01-01", "2023-01-01"),
    ]
    out=[]
    for name, a, b in windows:
        seg = eq[(eq.index >= a) & (eq.index < b)]
        exp = exposed[(exposed.index >= a) & (exposed.index < b)]
        if len(seg) < 2:
            out.append({"Window":name,"Sessions":len(seg),"ReturnPct":np.nan,"MaxDDPct":np.nan,"ExposurePct":np.nan})
            continue
        prior = eq[eq.index < a]
        base = float(prior.iloc[-1]) if len(prior) else float(seg.iloc[0])
        out.append({"Window":name,"Sessions":len(seg),"ReturnPct":(float(seg.iloc[-1])/base-1)*100,
                    "MaxDDPct":max_drawdown(pd.concat([pd.Series([base]),seg]))*100,
                    "ExposurePct":float(exp.mean()*100) if len(exp) else np.nan})
    return pd.DataFrame(out)


def monte_carlo(trades: pd.DataFrame, initial_cash: float, n_sims: int = 1000, seed: int = 42) -> dict[str, Any]:
    col = "AccountReturnPct" if not trades.empty and "AccountReturnPct" in trades.columns else "ReturnPct"
    returns = trades[col].dropna().to_numpy(dtype=float) / (100.0 if col == "AccountReturnPct" else 1.0) if not trades.empty else np.array([])
    if len(returns) < 5:
        return {"trades":len(returns),"message":"Fewer than 5 closed trades; Monte Carlo percentiles would be too unstable to interpret."}
    rng = np.random.default_rng(seed)
    final_values=[]; drawdowns=[]
    n=len(returns)
    # Bootstrap net per-trade returns with replacement; no signal or sizing changes implied.
    for _ in range(n_sims):
        sample = rng.choice(returns, size=n, replace=True)
        eq = initial_cash * np.cumprod(1.0 + sample)
        path = np.concatenate([[initial_cash], eq])
        final_values.append(float(path[-1]))
        drawdowns.append(float(np.min(path / np.maximum.accumulate(path) - 1.0) * 100))
    return {"trades":n,"simulations":n_sims,
            "final_p05":float(np.percentile(final_values,5)),"final_p50":float(np.percentile(final_values,50)),"final_p95":float(np.percentile(final_values,95)),
            "dd_p05":float(np.percentile(drawdowns,5)),"dd_p50":float(np.percentile(drawdowns,50)),"dd_p95":float(np.percentile(drawdowns,95))}


def paper_signal(data: pd.DataFrame) -> pd.DataFrame:
    """Create a daily close signal table. This logs signals only; it sends no orders."""
    d = add_indicators(data)
    cols=["Open","High","Low","Close","Volume","SMA100","Momentum60","Vol20","EntrySignal","ExitSignal"]
    signals=d[cols].copy()
    actions=[]
    planned_long=False
    for _, row in signals.iterrows():
        if pd.isna(row["SMA100"]) or pd.isna(row["Momentum60"]):
            action="WARMUP"
        elif not planned_long and bool(row["EntrySignal"]):
            action="BUY_NEXT_OPEN"
            planned_long=True
        elif planned_long and bool(row["ExitSignal"]):
            action="EXIT_NEXT_OPEN"
            planned_long=False
        else:
            action="HOLD_LONG" if planned_long else "CASH"
        actions.append(action)
    signals["ActionAtClose"] = actions
    signals.index.name="Date"
    return signals


def append_latest_paper_log(signals: pd.DataFrame) -> Path:
    """Upsert the latest completed daily signal into a persistent manual paper log."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "paper_trading_log.csv"
    latest = signals.tail(1).copy()
    latest.index.name = "Date"
    if path.exists():
        previous = pd.read_csv(path, index_col=0, parse_dates=True)
        previous.index = pd.to_datetime(previous.index).tz_localize(None)
        previous = previous[~previous.index.isin(latest.index)]
        combined = pd.concat([previous, latest], axis=0).sort_index()
    else:
        combined = latest
    combined.to_csv(path, index_label="Date")
    return path


def main() -> int:
    parser=argparse.ArgumentParser(description="Gold Momentum v1.3 historical backtest / robustness checks / paper signal log")
    parser.add_argument("--start",default="2004-11-18",help="Start date (default: GLD inception era, 2004-11-18)")
    parser.add_argument("--end",default=None,help="End date exclusive, e.g. 2026-01-01; default: latest available")
    parser.add_argument("--capital",type=float,default=10000.0,help="Initial simulated capital (default 10000)")
    parser.add_argument("--commission",type=float,default=0.0,help="Commission rate per side as decimal; 0.001 = 0.10%%")
    parser.add_argument("--risk",type=float,default=0.005,help="Risk budget fraction per trade; default 0.005 = 0.50%%")
    parser.add_argument("--vol-multiple",type=float,default=2.0,help="Volatility distance multiple; default 2.0 (explicit provisional assumption)")
    parser.add_argument("--normal-slip",type=float,default=0.0006,help="Normal-day slippage per side; default 0.0006 = 0.06%%")
    parser.add_argument("--high-slip",type=float,default=0.0015,help="High-vol-day slippage per side; default 0.0015 = 0.15%%")
    parser.add_argument("--refresh",action="store_true",help="Redownload data instead of using local cache")
    parser.add_argument("--no-walk-forward",action="store_true",help="Skip calendar-year fixed-parameter OOS report")
    parser.add_argument("--no-crisis",action="store_true",help="Skip crisis window report")
    parser.add_argument("--no-monte-carlo",action="store_true",help="Skip Monte Carlo bootstrap")
    parser.add_argument("--mc-sims",type=int,default=1000,help="Monte Carlo simulations, default 1000")
    args=parser.parse_args()
    if args.capital<=0 or not (0<args.risk<=0.02) or args.commission<0 or args.normal_slip<0 or args.high_slip<0 or args.vol_multiple<=0:
        parser.error("Capital and volatility multiple must be positive; risk must be in (0, 0.02]; costs cannot be negative.")

    DATA_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    raw=download_data(args.start,args.end,args.refresh)
    costs=CostSettings(commission=args.commission,normal_slippage=args.normal_slip,high_slippage=args.high_slip,
                       risk_fraction=args.risk,vol_multiple=args.vol_multiple,initial_cash=args.capital)
    data,trades,result=run_backtest(raw,costs)
    equity=result.pop("_equity")
    print_metrics("GOLD MOMENTUM v1.3 — BASE BACKTEST",result)
    print("\nNOTE: signals are calculated at close and orders simulated at the following session's open.")
    print("NOTE: risk sizing uses a provisional 2 x daily volatility distance, capped by available cash.")
    print("NOTE: no broker commissions are assumed unless you pass --commission; slippage is included.")

    data.to_csv(REPORT_DIR/"gld_indicators.csv",index_label="Date")
    trades.to_csv(REPORT_DIR/"closed_trades.csv",index=False)
    equity.to_csv(REPORT_DIR/"equity_curve.csv",index_label="Date")
    pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,(pd.Series,pd.DataFrame))}]).to_csv(REPORT_DIR/"summary_metrics.csv",index=False)

    if not args.no_walk_forward:
        wf=walk_forward_report(raw,costs)
        wf.to_csv(REPORT_DIR/"walk_forward_yearly_oos.csv",index=False)
        print("\nFIXED-PARAMETER OOS YEAR CHECK (first four years used as warm-up/training history; no optimization):")
        if wf.empty: print("Not enough historical years to produce OOS yearly rows.")
        else:
            print(wf.to_string(index=False,formatters={"OOS_ReturnPct":lambda x:f"{x:.2f}%","OOS_Sharpe":lambda x:f"{x:.3f}","OOS_MaxDDPct":lambda x:f"{x:.2f}%"}))
            print("Mean yearly OOS return: %.2f%%" % wf["OOS_ReturnPct"].mean())
    if not args.no_crisis:
        cr=crisis_report(raw,costs)
        cr.to_csv(REPORT_DIR/"crisis_windows.csv",index=False)
        print("\nCRISIS WINDOWS (diagnostic only; windows may have limited observations):")
        print(cr.to_string(index=False,formatters={"ReturnPct":lambda x:f"{x:.2f}%" if pd.notna(x) else "N/A","MaxDDPct":lambda x:f"{x:.2f}%" if pd.notna(x) else "N/A","ExposurePct":lambda x:f"{x:.2f}%" if pd.notna(x) else "N/A"}))
    if not args.no_monte_carlo:
        mc=monte_carlo(trades,args.capital,args.mc_sims)
        pd.DataFrame([mc]).to_csv(REPORT_DIR/"monte_carlo.csv",index=False)
        print("\nMONTE CARLO BOOTSTRAP OF CLOSED-TRADE NET RETURNS:")
        if "message" in mc: print(mc["message"])
        else:
            print(f"Simulations: {mc['simulations']} | closed trades sampled: {mc['trades']}")
            print(f"Final equity p05 / p50 / p95: {money(mc['final_p05'])} / {money(mc['final_p50'])} / {money(mc['final_p95'])}")
            print(f"Max drawdown p05 / p50 / p95: {mc['dd_p05']:.2f}% / {mc['dd_p50']:.2f}% / {mc['dd_p95']:.2f}%")

    # Cost sensitivity: base slippage versus the specification's conservative high end.
    stress_costs = CostSettings(commission=args.commission, normal_slippage=0.0008, high_slippage=0.0020,
                                risk_fraction=args.risk, vol_multiple=args.vol_multiple, initial_cash=args.capital)
    _, stress_trades, stress_result = run_backtest(raw, stress_costs)
    cost_sensitivity = pd.DataFrame([
        {"Scenario":"Base slippage", "CommissionPerSidePct":args.commission*100,
         "NormalSlippagePct":costs.normal_slippage*100,"HighVolSlippagePct":costs.high_slippage*100,
         "NetReturnPct":result["NetReturnPct"],"CAGR_Pct":result["CAGR_Pct"],"Sharpe":result["Sharpe"],
         "MaxDrawdownPct":result["MaxDrawdownPct"],"ClosedTrades":result["ClosedTrades"],"ProfitFactor":result["ProfitFactor"]},
        {"Scenario":"Stress slippage (upper end)", "CommissionPerSidePct":args.commission*100,
         "NormalSlippagePct":stress_costs.normal_slippage*100,"HighVolSlippagePct":stress_costs.high_slippage*100,
         "NetReturnPct":stress_result["NetReturnPct"],"CAGR_Pct":stress_result["CAGR_Pct"],"Sharpe":stress_result["Sharpe"],
         "MaxDrawdownPct":stress_result["MaxDrawdownPct"],"ClosedTrades":stress_result["ClosedTrades"],"ProfitFactor":stress_result["ProfitFactor"]}
    ])
    cost_sensitivity.to_csv(REPORT_DIR/"cost_sensitivity.csv",index=False)
    print("\nCOST SENSITIVITY (base vs. upper-end slippage; commission is the configured assumption):")
    print(cost_sensitivity.to_string(index=False, formatters={"NetReturnPct":lambda x:f"{x:.2f}%", "CAGR_Pct":lambda x:f"{x:.2f}%", "Sharpe":lambda x:f"{x:.3f}" if pd.notna(x) else "N/A", "MaxDrawdownPct":lambda x:f"{x:.2f}%", "ProfitFactor":lambda x:f"{x:.3f}" if pd.notna(x) else "N/A"}))

    # Generate a current signal file using most recent completed bar.
    signals=paper_signal(raw)
    signals.to_csv(REPORT_DIR/"daily_paper_signals.csv",index_label="Date")
    paper_log_path = append_latest_paper_log(signals)
    latest=signals.iloc[-1]
    print("\nLATEST COMPLETED DAILY BAR — PAPER SIGNAL ONLY")
    print(f"Date: {signals.index[-1].date()} | Adjusted close: {latest['Close']:.2f} | SMA100: {latest['SMA100']:.2f} | Momentum60: {latest['Momentum60']:.2%}")
    print(f"Signal at close: {latest['ActionAtClose']} (if actionable, execution is next session open).")
    print(f"Manual paper log updated: {paper_log_path}")
    print(f"\nReports saved in: {REPORT_DIR}")
    print("Nothing was sent to a broker; this script does not place orders.")
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.",file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}",file=sys.stderr)
        raise
