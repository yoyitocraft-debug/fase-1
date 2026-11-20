#!/usr/bin/env python3
"""Gold Momentum v1.3 benchmark v3.

Reads local GLD data + saved strategy equity_curve.csv files. Compares the
strategy to (a) full buy-and-hold, (b) fixed initial-allocation buy-and-hold,
and (c) a daily rebalanced GLD allocation whose target notional weight equals
the strategy's average daily notional exposure. Costs apply to each trade side.
This is a research comparator only; it does not connect to a broker.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "gold_data"
OUT = ROOT / "comparacion_vs_buy_hold_v3.csv"
INITIAL_CAPITAL = 10_000.0
BASE_SLIP = 0.0006
STRESS_SLIP = 0.0008
TRADING_DAYS = 252


def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna().astype(float)
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min() * 100)


def metrics(eq: pd.Series) -> dict[str, float]:
    eq = eq.dropna().astype(float)
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / TRADING_DAYS)
    final = float(eq.iloc[-1])
    std = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    sharpe = float(rets.mean() / std * math.sqrt(TRADING_DAYS)) if np.isfinite(std) and std > 0 else float("nan")
    cagr = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if final > 0 else float("nan")
    return {
        "FinalEquity": final,
        "NetReturnPct": (final / INITIAL_CAPITAL - 1) * 100,
        "CAGR_Pct": cagr,
        "Sharpe": sharpe,
        "MaxDrawdownPct": max_dd(eq),
    }


def load_prices() -> tuple[pd.DataFrame, Path]:
    preferred = DATA_DIR / "GLD_2004-11-18_latest.csv"
    candidates = [preferred] if preferred.exists() else sorted(DATA_DIR.glob("GLD_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No encuentro {DATA_DIR}\\GLD_*.csv. Ejecuta primero el backtest con --refresh.")
    path = candidates[0]
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    ix = pd.to_datetime(d.index)
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_localize(None)
    d.index = ix
    d = d[~d.index.duplicated(keep="last")].sort_index()
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(d.columns):
        raise ValueError(f"{path} necesita columnas {sorted(required)}; encontré {list(d.columns)}")
    return d.dropna(subset=["Open", "High", "Low", "Close"]), path


def commission_for(folder: Path) -> float:
    summary = folder / "summary_metrics.csv"
    if summary.exists():
        try:
            row = pd.read_csv(summary).iloc[0]
            if "CommissionPerSidePct" in row:
                return float(row["CommissionPerSidePct"]) / 100
        except Exception:
            pass
    name = folder.name.lower()
    if name.endswith("_005"):
        return 0.0005
    if name.endswith("_010"):
        return 0.001
    return 0.0


def load_curve(folder: Path, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float, float]:
    p = folder / "equity_curve.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    e = pd.read_csv(p, parse_dates=["Date"])
    required = {"Date", "Equity", "Cash", "Shares", "InMarket"}
    if not required.issubset(e.columns):
        raise ValueError(f"{p} requiere columnas {sorted(required)}; encontré {list(e.columns)}")
    e = e.dropna(subset=["Date", "Equity"]).drop_duplicates("Date", keep="last").sort_values("Date").set_index("Date")
    e.index = pd.to_datetime(e.index)
    if getattr(e.index, "tz", None) is not None:
        e.index = e.index.tz_localize(None)
    e = e.join(prices[["Open", "High", "Low", "Close"]].rename(columns={"Open":"PriceOpen", "High":"PriceHigh", "Low":"PriceLow", "Close":"PriceClose"}), how="inner")
    if e.empty:
        raise ValueError(f"No hay fechas comunes entre {p} y los precios")
    eq = e["Equity"].astype(float)
    notional = e["Shares"].astype(float) * e["PriceOpen"].astype(float)
    frac = (notional / eq.replace(0, np.nan)).clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
    avg_exposure = float(frac.mean()) if len(frac) else 0.0
    days_market = float(e["InMarket"].astype(bool).mean() * 100)
    return e, eq, avg_exposure, days_market


def friction_buy(slip: float, commission: float) -> float:
    return (1 + slip) * (1 + commission) - 1


def friction_sell(slip: float, commission: float) -> float:
    return 1 - (1 - slip) * (1 - commission)


def buy_hold_curve(px: pd.Series, allocation: float, commission: float, slip: float) -> pd.Series:
    """Fixed initial allocation; never rebalanced. Remaining capital earns zero."""
    px = px.dropna().astype(float)
    if len(px) < 2:
        return pd.Series(dtype=float)
    alloc = float(np.clip(allocation, 0, 1))
    p0 = float(px.iloc[0])
    p1 = float(px.iloc[-1])
    if p0 <= 0 or p1 <= 0:
        return pd.Series(dtype=float)
    bcost = friction_buy(slip, commission)
    scost = friction_sell(slip, commission)
    invested_market_value = INITIAL_CAPITAL * alloc / (1 + bcost)
    shares = invested_market_value / p0
    cash = INITIAL_CAPITAL - invested_market_value - invested_market_value * bcost
    equity = cash + shares * px
    equity.iloc[-1] = cash + shares * p1 * (1 - scost)
    return equity


def target_weight_rebalanced_curve(px: pd.Series, weight: float, commission: float, slip: float) -> tuple[pd.Series, float, int]:
    """Rebalance at each daily OPEN to constant stock-weight; costs charged on turnover."""
    px = px.dropna().astype(float)
    if len(px) < 2:
        return pd.Series(dtype=float), 0.0, 0
    w = float(np.clip(weight, 0, 1))
    bcost = friction_buy(slip, commission)
    scost = friction_sell(slip, commission)
    cash = INITIAL_CAPITAL
    shares = 0.0
    values: list[float] = []
    turnovers: list[float] = []
    ntrades = 0
    for i, (date, price) in enumerate(px.items()):
        p = float(price)
        equity_before = cash + shares * p
        if i == len(px) - 1:
            # Liquidate at the final OPEN, paying sell-side costs.
            liquidation_cost = shares * p * scost
            final_equity = equity_before - liquidation_cost
            values.append(final_equity)
            turnovers.append(shares * p)
            if shares > 1e-12:
                ntrades += 1
            break
        current_notional = shares * p
        target_notional = w * equity_before
        delta = target_notional - current_notional
        if abs(delta) > max(1e-8, equity_before * 1e-8):
            cost_rate = bcost if delta > 0 else scost
            trading_cost = abs(delta) * cost_rate
            ntrades += 1
        else:
            trading_cost = 0.0
        shares = target_notional / p
        cash = equity_before - target_notional - trading_cost
        equity_after = cash + shares * p
        values.append(equity_after)
        turnovers.append(abs(delta))
    return pd.Series(values, index=px.index, dtype=float), float(sum(turnovers)), ntrades


def intraday_low_dd(curve: pd.DataFrame) -> float:
    """Intraday valuation at Low versus recorded equity high-water mark.

    Not an exact pathwise max drawdown because OHLC bars do not reveal whether
    each day's high occurred before its low. It can understate a peak-to-trough
    drawdown if an intraday high occurred before the low.
    """
    if curve.empty:
        return float("nan")
    shares = curve["Shares"].astype(float)
    cash = curve["Cash"].astype(float)
    lows = cash + shares * curve["PriceLow"].astype(float)
    # Include previous marked equity peaks as possible prior intraday peaks.
    peaks = curve["Equity"].astype(float).cummax()
    ratio = lows / peaks.replace(0, np.nan) - 1
    return float(ratio.min() * 100)


def main() -> int:
    prices, data_path = load_prices()
    report_dirs = sorted(p for p in ROOT.glob("gold_reports*") if p.is_dir() and (p / "equity_curve.csv").exists())
    if not report_dirs:
        raise FileNotFoundError(f"No encuentro carpetas gold_reports* con equity_curve.csv en {ROOT}")
    results: list[dict] = []
    for folder in report_dirs:
        try:
            curve, strat_eq, weight, days = load_curve(folder, prices)
        except Exception as exc:
            print(f"Omitiendo {folder.name}: {exc}")
            continue
        commission = commission_for(folder)
        eqm = metrics(strat_eq)
        results.append({
            "ReportFolder": folder.name,
            "CommissionPerSidePct": commission * 100,
            "Asset": "Gold Momentum v1.3 (saved equity)",
            "TargetOrAvgExposurePct": weight * 100,
            "DaysInMarketPct": days,
            "Transactions": np.nan,
            "IntradayLowDD_EstPct": intraday_low_dd(curve),
            **eqm,
        })
        px = curve["PriceOpen"].astype(float)
        cases = [
            ("GLD buy-and-hold 100%", 1.0, "fixed"),
            ("GLD buy-and-hold fixed initial allocation", weight, "fixed"),
            ("GLD target-weight rebalanced daily", weight, "rebalanced"),
        ]
        for label, alloc, mode in cases:
            for slip, slip_name in ((BASE_SLIP, "base 0.06%"), (STRESS_SLIP, "stress 0.08%")):
                if mode == "fixed":
                    eq = buy_hold_curve(px, alloc, commission, slip)
                    trans = 2 if alloc > 0 else 0
                else:
                    eq, turnover, trans = target_weight_rebalanced_curve(px, alloc, commission, slip)
                results.append({
                    "ReportFolder": folder.name,
                    "CommissionPerSidePct": commission * 100,
                    "Asset": f"{label} / {slip_name}",
                    "TargetOrAvgExposurePct": alloc * 100,
                    "DaysInMarketPct": 100.0 if alloc > 0 else 0.0,
                    "Transactions": trans,
                    "IntradayLowDD_EstPct": np.nan,
                    **metrics(eq),
                })
    if not results:
        raise RuntimeError("No se pudieron calcular las comparaciones")
    out = pd.DataFrame(results)
    out.to_csv(OUT, index=False)
    print("=" * 108)
    print("COMPARACIÓN GLD v3 — INCLUYE BENCHMARK CON PESO OBJETIVO REBALANCEADO DIARIAMENTE")
    print("=" * 108)
    print(f"Datos: {data_path}")
    print("Todas las curvas se valoran al OPEN; costes aplicados a cada lado de las operaciones.")
    print("El benchmark rebalanced daily mantiene un peso objetivo igual a la exposición nocional media de la estrategia.")
    print("IntradayLowDD_EstPct compara el valor de cartera en Low con el máximo registrado de equity; puede subestimar el drawdown intradía real porque OHLC no revela si High ocurrió antes de Low.")
    cols = ["ReportFolder", "CommissionPerSidePct", "Asset", "TargetOrAvgExposurePct", "FinalEquity", "NetReturnPct", "CAGR_Pct", "Sharpe", "MaxDrawdownPct", "IntradayLowDD_EstPct", "Transactions"]
    print(out[cols].to_string(index=False, formatters={
        "CommissionPerSidePct": lambda x: f"{x:.3f}%",
        "TargetOrAvgExposurePct": lambda x: f"{x:.2f}%",
        "FinalEquity": lambda x: f"${x:,.2f}",
        "NetReturnPct": lambda x: f"{x:.2f}%",
        "CAGR_Pct": lambda x: f"{x:.2f}%",
        "Sharpe": lambda x: f"{x:.3f}",
        "MaxDrawdownPct": lambda x: f"{x:.2f}%",
        "IntradayLowDD_EstPct": lambda x: "" if pd.isna(x) else f"{x:.2f}%",
    }))
    print(f"\nCSV guardado en: {OUT}")
    print("Nada se envió a un bróker; este script no coloca órdenes.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
