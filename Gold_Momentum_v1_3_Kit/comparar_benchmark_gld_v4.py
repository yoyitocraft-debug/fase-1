#!/usr/bin/env python3
"""Gold Momentum v1.3 — benchmark v4 with matched variable slippage.

Purpose:
- Re-run the existing Gold Momentum v1.3 rules at the folder's commission rate
  under base slippage (0.06% normal / 0.15% high-vol) and stress slippage
  (0.08% normal / 0.20% high-vol).
- Compare each version with a daily rebalanced GLD benchmark whose *average
  notional weight* matches the corresponding strategy run.
- Apply the same prior-session volatility classifier and per-side cost model
  to all simulated benchmark trades.

This is diagnostic research only, not an order system. The target weight is
calibrated using the full sample's average strategy exposure, so it is an
ex-post benchmark and not a deployable trading rule.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from gold_momentum_v13 import CostSettings, add_indicators, run_backtest

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "gold_data"
OUT = ROOT / "comparacion_vs_buy_hold_v4.csv"
INITIAL_CAPITAL = 10_000.0
TRADING_DAYS = 252
BASE_NORMAL, BASE_HIGH = 0.0006, 0.0015
STRESS_NORMAL, STRESS_HIGH = 0.0008, 0.0020


def clean_index(index):
    ix = pd.to_datetime(index)
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_localize(None)
    return ix


def metrics(eq: pd.Series) -> dict[str, float]:
    eq = eq.dropna().astype(float)
    if len(eq) < 2:
        return {}
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / TRADING_DAYS)
    final = float(eq.iloc[-1])
    std = float(ret.std(ddof=1)) if len(ret) > 1 else float("nan")
    sharpe = float(ret.mean() / std * math.sqrt(TRADING_DAYS)) if np.isfinite(std) and std > 0 else float("nan")
    return {
        "FinalEquity": final,
        "NetReturnPct": (final / INITIAL_CAPITAL - 1) * 100,
        "CAGR_Pct": ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if final > 0 else float("nan"),
        "Sharpe": sharpe,
        "MaxDrawdownPct": float((eq / eq.cummax() - 1).min() * 100),
    }


def load_prices() -> tuple[pd.DataFrame, Path]:
    candidates = [DATA_DIR / "GLD_2004-11-18_latest.csv"]
    candidates += sorted(DATA_DIR.glob("GLD_*.csv"))
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No encuentro {DATA_DIR}\\GLD_*.csv; ejecuta primero el backtest con --refresh.")
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    d.index = clean_index(d.index)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    req = {"Open", "High", "Low", "Close", "Volume"}
    if not req.issubset(d.columns):
        raise ValueError(f"Datos incompletos en {path}; faltan {sorted(req - set(d.columns))}")
    return d.dropna(subset=["Open", "High", "Low", "Close"]), path


def commission_for(folder: Path) -> float:
    p = folder / "summary_metrics.csv"
    if p.exists():
        try:
            row = pd.read_csv(p).iloc[0]
            for key in ("CommissionPerSidePct",):
                if key in row and pd.notna(row[key]):
                    return float(row[key]) / 100
        except Exception:
            pass
    n = folder.name.lower()
    if n.endswith("_005"):
        return 0.0005
    if n.endswith("_010"):
        return 0.001
    return 0.0


def exposure_pct(eq_df: pd.DataFrame, raw: pd.DataFrame) -> float:
    e = eq_df.copy()
    e.index = clean_index(e.index)
    op = raw["Open"].reindex(e.index)
    notional = e["Shares"].astype(float) * op.astype(float)
    frac = (notional / e["Equity"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(frac.mean()) if len(frac) else 0.0


def slip_series(raw: pd.DataFrame, index: pd.Index, normal: float, high: float) -> pd.Series:
    d = add_indicators(raw)
    # A trade at today's OPEN may only use the prior session's completed range.
    prior_high_vol = d["HighVolDay"].shift(1).reindex(index).fillna(False).astype(bool)
    return pd.Series(np.where(prior_high_vol, high, normal), index=index, dtype=float)


def friction_buy(slip: float, commission: float) -> float:
    return (1 + slip) * (1 + commission) - 1


def friction_sell(slip: float, commission: float) -> float:
    return 1 - (1 - slip) * (1 - commission)


def rebalanced_curve(px: pd.Series, slips: pd.Series, weight: float, commission: float) -> tuple[pd.Series, int]:
    px = px.dropna().astype(float)
    slips = slips.reindex(px.index).fillna(BASE_NORMAL).astype(float)
    if len(px) < 2:
        return pd.Series(dtype=float), 0
    w = float(np.clip(weight, 0, 1))
    cash, shares = INITIAL_CAPITAL, 0.0
    values, ntrades = [], 0
    for i, date in enumerate(px.index):
        p = float(px.loc[date])
        slip = float(slips.loc[date])
        equity_before = cash + shares * p
        if i == len(px) - 1:
            final = equity_before - shares * p * friction_sell(slip, commission)
            values.append(final)
            if shares > 1e-12:
                ntrades += 1
            break
        current_notional = shares * p
        target_notional = w * equity_before
        delta = target_notional - current_notional
        trading_cost = 0.0
        if abs(delta) > max(1e-8, equity_before * 1e-8):
            trading_cost = abs(delta) * (friction_buy(slip, commission) if delta > 0 else friction_sell(slip, commission))
            ntrades += 1
        shares = target_notional / p
        cash = equity_before - target_notional - trading_cost
        values.append(cash + shares * p)
    return pd.Series(values, index=px.index, dtype=float), ntrades


def intraday_low_dd(eq_df: pd.DataFrame, raw: pd.DataFrame) -> float:
    e = eq_df.copy()
    e.index = clean_index(e.index)
    lows = raw["Low"].reindex(e.index).astype(float)
    opens = raw["Open"].reindex(e.index).astype(float)
    # Diagnostic only: values held at each OPEN marked against that session's Low.
    low_equity = e["Cash"].astype(float) + e["Shares"].astype(float) * lows
    open_peaks = e["Equity"].astype(float).cummax()
    return float((low_equity / open_peaks.replace(0, np.nan) - 1).min() * 100)


def main() -> int:
    raw, data_path = load_prices()
    folders = sorted(p for p in ROOT.glob("gold_reports*") if p.is_dir() and (p / "equity_curve.csv").exists())
    if not folders:
        raise FileNotFoundError(f"No encuentro carpetas gold_reports* con equity_curve.csv en {ROOT}")
    rows = []
    for folder in folders:
        commission = commission_for(folder)
        for scenario, normal, high in (("BASE", BASE_NORMAL, BASE_HIGH), ("STRESS", STRESS_NORMAL, STRESS_HIGH)):
            costs = CostSettings(commission=commission, normal_slippage=normal, high_slippage=high,
                                 risk_fraction=0.005, vol_multiple=2.0, initial_cash=INITIAL_CAPITAL)
            indicators, trades, result = run_backtest(raw, costs)
            eq_df = result["_equity"].copy()
            eq = eq_df["Equity"].astype(float)
            weight = exposure_pct(eq_df, raw)
            strat_metrics = metrics(eq)
            rows.append({
                "ReportFolder": folder.name,
                "CommissionPerSidePct": commission * 100,
                "Scenario": scenario,
                "Asset": "Gold Momentum v1.3",
                "AvgNotionalExposurePct": weight * 100,
                "DaysInMarketPct": float(eq_df["InMarket"].astype(bool).mean() * 100),
                "SlippageNormalPerSidePct": normal * 100,
                "SlippageHighVolPerSidePct": high * 100,
                "Transactions": int(result["Buys"] + result["Sells"]),
                "IntradayLowDD_EstPct": intraday_low_dd(eq_df, raw),
                **strat_metrics,
            })
            index = eq.index
            prices_open = raw["Open"].reindex(index).dropna().astype(float)
            slips = slip_series(raw, prices_open.index, normal, high)
            bench_eq, trans = rebalanced_curve(prices_open, slips, weight, commission)
            rows.append({
                "ReportFolder": folder.name,
                "CommissionPerSidePct": commission * 100,
                "Scenario": scenario,
                "Asset": "GLD daily-rebalanced target weight (matched avg exposure)",
                "AvgNotionalExposurePct": weight * 100,
                "DaysInMarketPct": 100.0 if weight > 0 else 0.0,
                "SlippageNormalPerSidePct": normal * 100,
                "SlippageHighVolPerSidePct": high * 100,
                "Transactions": trans,
                "IntradayLowDD_EstPct": np.nan,
                **metrics(bench_eq),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("=" * 112)
    print("COMPARACIÓN GLD v4 — COSTOS VARIABLES EMPAREJADOS PARA ESTRATEGIA Y BENCHMARK")
    print("=" * 112)
    print(f"Datos: {data_path}")
    print("Base: 0.06% normal / 0.15% alta volatilidad; stress: 0.08% normal / 0.20% alta volatilidad, por lado.")
    print("La volatilidad de costos usa solo el rango de la sesión previa; las órdenes se modelan en la apertura.")
    print("El benchmark iguala la exposición nocional media observada de cada ejecución; comparación ex-post, no señal operable.")
    print("IntradayLowDD_EstPct solo se calcula para Gold Momentum y es una aproximación, no un drawdown intradía exacto.")
    cols = ["ReportFolder", "CommissionPerSidePct", "Scenario", "Asset", "AvgNotionalExposurePct", "FinalEquity", "NetReturnPct", "CAGR_Pct", "Sharpe", "MaxDrawdownPct", "IntradayLowDD_EstPct", "Transactions"]
    print(out[cols].to_string(index=False, formatters={
        "CommissionPerSidePct": lambda x: f"{x:.3f}%",
        "AvgNotionalExposurePct": lambda x: f"{x:.2f}%",
        "FinalEquity": lambda x: f"${x:,.2f}",
        "NetReturnPct": lambda x: f"{x:.2f}%",
        "CAGR_Pct": lambda x: f"{x:.2f}%",
        "Sharpe": lambda x: f"{x:.3f}",
        "MaxDrawdownPct": lambda x: f"{x:.2f}%",
        "IntradayLowDD_EstPct": lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A",
    }))
    print(f"\nCSV guardado en: {OUT}")
    print("Solo simulación; nada se envió a un bróker.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
