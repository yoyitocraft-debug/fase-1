#!/usr/bin/env python3
"""Audited GLD benchmark comparison for Gold Momentum v1.3.

Uses the same dates and daily OPEN mark-to-market convention as the current
Gold Momentum v1.3 equity_curve.csv. Compares:
  1) Gold Momentum equity curve as saved by the backtest;
  2) full-capital GLD buy-and-hold;
  3) GLD buy-and-hold with an initial allocation equal to the strategy's
     average daily notional exposure (shares * adjusted Open / equity).

This is a diagnostic comparator, not a broker simulation. It uses cached data
from gold_data/ and reports from gold_reports*/; it never places orders.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "gold_data"
OUT = ROOT / "comparacion_vs_buy_hold_v2.csv"
INITIAL_CAPITAL = 10_000.0
BASE_SLIP = 0.0006
STRESS_SLIP = 0.0008
TRADING_DAYS = 252


def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna().astype(float)
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min() * 100.0)


def metrics(eq: pd.Series, initial: float = INITIAL_CAPITAL) -> dict[str, float]:
    eq = eq.dropna().astype(float)
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1 / TRADING_DAYS)
    final = float(eq.iloc[-1])
    std = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    sharpe = float(rets.mean() / std * math.sqrt(TRADING_DAYS)) if np.isfinite(std) and std > 0 else float("nan")
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 else float("nan")
    return {
        "FinalEquity": final,
        "NetReturnPct": (final / initial - 1) * 100,
        "CAGR_Pct": cagr,
        "Sharpe": sharpe,
        "MaxDrawdownPct": max_dd(eq),
    }


def load_data() -> tuple[pd.DataFrame, Path]:
    preferred = DATA_DIR / "GLD_2004-11-18_latest.csv"
    candidates = [preferred] if preferred.exists() else sorted(DATA_DIR.glob("GLD_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No encuentro {DATA_DIR}\\GLD_*.csv. Ejecuta primero el backtest con --refresh.")
    p = candidates[0]
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if not {"Open", "Low", "Close"}.issubset(d.columns):
        raise ValueError(f"El CSV {p} necesita columnas Open, Low y Close; encontré {list(d.columns)}")
    return d.dropna(subset=["Open", "Low", "Close"]), p


def infer_commission(folder: Path) -> float:
    p = folder / "summary_metrics.csv"
    if p.exists():
        try:
            r = pd.read_csv(p).iloc[0]
            if "CommissionPerSidePct" in r:
                return float(r["CommissionPerSidePct"]) / 100.0
        except Exception:
            pass
    nm = folder.name.lower()
    if nm.endswith("_005"):
        return 0.0005
    if nm.endswith("_010"):
        return 0.001
    return 0.0


def load_strategy_curve(folder: Path, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float, float]:
    p = folder / "equity_curve.csv"
    if not p.exists():
        raise FileNotFoundError(f"Falta {p}")
    e = pd.read_csv(p, parse_dates=["Date"])
    if not {"Date", "Equity"}.issubset(e.columns):
        raise ValueError(f"{p} necesita columnas Date y Equity")
    e = e.dropna(subset=["Date", "Equity"]).drop_duplicates("Date", keep="last").sort_values("Date").set_index("Date")
    e.index = pd.to_datetime(e.index)
    if getattr(e.index, "tz", None) is not None:
        e.index = e.index.tz_localize(None)
    e = e.join(prices[["Open"]].rename(columns={"Open": "PriceOpen"}), how="inner")
    if e.empty:
        raise ValueError(f"No hay fechas comunes entre {p} y el CSV de precios")
    eq = e["Equity"].astype(float)
    # Average *notional capital exposure*, distinct from % of days in market.
    if "Shares" in e.columns:
        notional = e["Shares"].astype(float) * e["PriceOpen"].astype(float)
        exposure_frac = (notional / e["Equity"].astype(float).replace(0, np.nan)).clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
        avg_exposure = float(exposure_frac.mean()) if not exposure_frac.empty else 0.0
    elif "InMarket" in e.columns:
        avg_exposure = float(e["InMarket"].astype(bool).mean())
    else:
        raise ValueError(f"{p} no tiene Shares ni InMarket para estimar exposición")
    market_days = float(e["InMarket"].astype(bool).mean() * 100) if "InMarket" in e.columns else float("nan")
    return e, eq, avg_exposure, market_days


def buy_hold_open_curve(prices: pd.DataFrame, dates: pd.DatetimeIndex, commission: float, slip: float, allocation: float) -> pd.Series:
    """Hold an initial fraction of capital in GLD, remaining cash earns 0%.

    All marks and the final liquidation use adjusted OPEN prices so the series
    aligns with the strategy equity curve's daily-open marking. Assumes
    fractional shares and entry/exit friction on the GLD allocation only.
    """
    px = prices.reindex(dates).dropna(subset=["Open"])
    if len(px) < 2:
        return pd.Series(dtype=float)
    allocation = float(np.clip(allocation, 0.0, 1.0))
    first_open = float(px["Open"].iloc[0])
    last_open = float(px["Open"].iloc[-1])
    if first_open <= 0 or last_open <= 0:
        return pd.Series(dtype=float)
    shares = (INITIAL_CAPITAL * allocation) / (first_open * (1 + slip) * (1 + commission))
    cash = INITIAL_CAPITAL - shares * first_open * (1 + slip) * (1 + commission)
    eq = cash + shares * px["Open"].astype(float)
    # Liquidate at the last observed open and pay exit slippage + commission.
    eq.iloc[-1] = cash + shares * last_open * (1 - slip) * (1 - commission)
    return eq


def main() -> int:
    prices, data_path = load_data()
    report_dirs = sorted(p for p in ROOT.glob("gold_reports*") if p.is_dir() and (p / "equity_curve.csv").exists())
    if not report_dirs:
        raise FileNotFoundError(f"No encuentro carpetas gold_reports* con equity_curve.csv en {ROOT}")
    rows: list[dict] = []
    for folder in report_dirs:
        try:
            curve_df, strat_eq, avg_exposure, market_days = load_strategy_curve(folder, prices)
        except Exception as exc:
            print(f"Omitiendo {folder.name}: {exc}")
            continue
        commission = infer_commission(folder)
        common_dates = curve_df.index
        strat = metrics(strat_eq)
        rows.append({
            "ReportFolder": folder.name,
            "CommissionPerSidePct": commission * 100,
            "Asset": "Gold Momentum v1.3 (saved equity; OPEN marks)",
            "AvgNotionalExposurePct": avg_exposure * 100,
            "DaysInMarketPct": market_days,
            "SlippagePerSidePct": "0.06%/0.15% variable in strategy",
            **strat,
        })
        for allocation, alloc_label in ((1.0, "GLD buy-and-hold 100%"), (avg_exposure, "GLD buy-and-hold / avg-exposure-matched allocation")):
            for slip, slip_label in ((BASE_SLIP, "0.06% per side"), (STRESS_SLIP, "0.08% per side")):
                bh = buy_hold_open_curve(prices, common_dates, commission, slip, allocation)
                if bh.empty:
                    continue
                rows.append({
                    "ReportFolder": folder.name,
                    "CommissionPerSidePct": commission * 100,
                    "Asset": alloc_label,
                    "AvgNotionalExposurePct": allocation * 100,
                    "DaysInMarketPct": 100.0 if allocation > 0 else 0.0,
                    "SlippagePerSidePct": slip_label,
                    **metrics(bh),
                })
    if not rows:
        raise RuntimeError("No se pudieron calcular las comparaciones.")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("=" * 105)
    print("COMPARACIÓN AUDITADA: GOLD MOMENTUM v1.3 VS GLD BUY-AND-HOLD")
    print("=" * 105)
    print(f"Precios: {data_path}")
    print("Convención: misma fecha y valoración diaria al OPEN para estrategia y benchmarks.")
    print("Benchmark de exposición igualada: compra pasiva con asignación inicial equivalente a la exposición nocional media de la estrategia; el resto queda en efectivo sin rendimiento.")
    print("Comisiones/slippage son supuestos; acciones fraccionarias teóricas; no hay broker ni órdenes reales.")
    cols = ["ReportFolder", "CommissionPerSidePct", "Asset", "AvgNotionalExposurePct", "FinalEquity", "NetReturnPct", "CAGR_Pct", "Sharpe", "MaxDrawdownPct"]
    print(out[cols].to_string(index=False, formatters={
        "CommissionPerSidePct": lambda x: f"{x:.3f}%",
        "AvgNotionalExposurePct": lambda x: f"{x:.2f}%",
        "FinalEquity": lambda x: f"${x:,.2f}",
        "NetReturnPct": lambda x: f"{x:.2f}%",
        "CAGR_Pct": lambda x: f"{x:.2f}%",
        "Sharpe": lambda x: f"{x:.3f}",
        "MaxDrawdownPct": lambda x: f"{x:.2f}%",
    }))
    print(f"\nCSV guardado en: {OUT}")
    print("Nota: la curva sigue siendo de valoración al OPEN; no mide pérdidas intradía extremas con High/Low.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
