#!/usr/bin/env python3
"""Compare Gold Momentum report folders against GLD buy-and-hold.

Place this file in the same directory as gold_momentum_v13.py and gold_data/.
Reads only cached GLD data and existing gold_reports*/equity_curve.csv files;
no trades or broker connections are made.
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "gold_data"
OUT = ROOT / "comparacion_vs_buy_hold.csv"
INITIAL_CAPITAL = 10_000.0
BASE_SLIP = 0.0006   # 0.06% per side, matching configured base normal slippage
STRESS_SLIP = 0.0008 # 0.08% per side, upper-end normal-day slippage
TRADING_DAYS = 252


def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna().astype(float)
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min() * 100.0)


def calc_metrics(eq: pd.Series, start_capital: float) -> dict:
    eq = eq.dropna().astype(float)
    if len(eq) < 2:
        return {}
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / TRADING_DAYS)
    final = float(eq.iloc[-1])
    cagr = ((final / start_capital) ** (1 / years) - 1) * 100 if final > 0 else float("nan")
    std = ret.std(ddof=1)
    sharpe = float(ret.mean() / std * math.sqrt(TRADING_DAYS)) if len(ret) > 1 and std > 0 else float("nan")
    return {
        "FinalEquity": final,
        "NetReturnPct": (final / start_capital - 1) * 100,
        "CAGR_Pct": cagr,
        "Sharpe": sharpe,
        "MaxDrawdownPct": max_dd(eq),
    }


def load_price_data() -> pd.DataFrame:
    preferred = DATA_DIR / "GLD_2004-11-18_latest.csv"
    candidates = [preferred] if preferred.exists() else sorted(DATA_DIR.glob("GLD_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No encuentro gold_data\\GLD_*.csv en {ROOT}. Ejecuta primero el backtest con --refresh."
        )
    data = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    if not {"Open", "Close"}.issubset(data.columns):
        raise ValueError(f"El CSV de precios no contiene Open y Close: {candidates[0]}")
    return data.dropna(subset=["Open", "Close"]), candidates[0]


def infer_commission(folder: Path, metrics_path: Path) -> float:
    if metrics_path.exists():
        try:
            row = pd.read_csv(metrics_path).iloc[0]
            return float(row.get("CommissionPerSidePct", 0.0)) / 100.0
        except Exception:
            pass
    name = folder.name.lower()
    if name.endswith("_005"):
        return 0.0005
    if name.endswith("_010"):
        return 0.001
    return 0.0


def buy_hold_equity(data: pd.DataFrame, dates: pd.DatetimeIndex, commission: float, slip: float) -> pd.Series:
    aligned = data.reindex(dates).dropna(subset=["Open", "Close"])
    if len(aligned) < 2:
        return pd.Series(dtype=float)
    first_open = float(aligned["Open"].iloc[0])
    last_close = float(aligned["Close"].iloc[-1])
    # Fractional shares for a clean theoretical benchmark; capital is fully invested.
    entry_unit_cost = first_open * (1.0 + slip) * (1.0 + commission)
    shares = INITIAL_CAPITAL / entry_unit_cost
    eq = shares * aligned["Close"].astype(float)
    # Assume the benchmark liquidates at the end, paying exit slippage and commission.
    eq.iloc[-1] = shares * last_close * (1.0 - slip) * (1.0 - commission)
    return eq


def main() -> int:
    data, data_path = load_price_data()
    report_dirs = sorted(p for p in ROOT.glob("gold_reports*") if p.is_dir())
    rows = []
    if not report_dirs:
        raise FileNotFoundError(f"No encuentro carpetas gold_reports* en {ROOT}.")

    for folder in report_dirs:
        eq_path = folder / "equity_curve.csv"
        metrics_path = folder / "summary_metrics.csv"
        if not eq_path.exists():
            continue
        eq_df = pd.read_csv(eq_path, parse_dates=["Date"])
        if not {"Date", "Equity"}.issubset(eq_df.columns):
            print(f"Omitiendo {folder.name}: equity_curve.csv no tiene Date y Equity.")
            continue
        eq_df = eq_df.dropna(subset=["Date", "Equity"]).drop_duplicates("Date", keep="last").sort_values("Date")
        strategy_eq = pd.Series(eq_df["Equity"].astype(float).to_numpy(), index=pd.DatetimeIndex(eq_df["Date"]))
        strategy_eq = strategy_eq[~strategy_eq.index.duplicated(keep="last")]
        if strategy_eq.empty:
            continue
        commission = infer_commission(folder, metrics_path)
        strat_m = calc_metrics(strategy_eq, INITIAL_CAPITAL)
        rows.append({
            "ReportFolder": folder.name,
            "CommissionPerSidePct": commission * 100,
            "Asset": "Gold Momentum v1.3 / base slippage",
            "AssumedSlippagePerSidePct": "0.06% / 0.15% variable",
            **strat_m,
        })
        # Include the strategy's own upper-end slippage row when the saved report exists.
        sensitivity_path = folder / "cost_sensitivity.csv"
        if sensitivity_path.exists():
            try:
                sens = pd.read_csv(sensitivity_path)
                stress = sens[sens["Scenario"].astype(str).str.contains("Stress slippage", case=False)].head(1)
                if not stress.empty:
                    sr = stress.iloc[0]
                    rows.append({
                        "ReportFolder": folder.name,
                        "CommissionPerSidePct": commission * 100,
                        "Asset": "Gold Momentum v1.3 / stress slippage",
                        "AssumedSlippagePerSidePct": "0.08% / 0.20% variable",
                        "FinalEquity": INITIAL_CAPITAL * (1 + float(sr["NetReturnPct"]) / 100),
                        "NetReturnPct": float(sr["NetReturnPct"]),
                        "CAGR_Pct": float(sr["CAGR_Pct"]),
                        "Sharpe": float(sr["Sharpe"]),
                        "MaxDrawdownPct": float(sr["MaxDrawdownPct"]),
                    })
            except Exception as exc:
                print(f"Aviso: no pude leer {sensitivity_path.name} en {folder.name}: {exc}")
        for label, slip in (("GLD buy-and-hold / base costs", BASE_SLIP), ("GLD buy-and-hold / stress normal slippage", STRESS_SLIP)):
            bh = buy_hold_equity(data, strategy_eq.index, commission, slip)
            if bh.empty:
                continue
            bh_m = calc_metrics(bh, INITIAL_CAPITAL)
            rows.append({
                "ReportFolder": folder.name,
                "CommissionPerSidePct": commission * 100,
                "Asset": label,
                "AssumedSlippagePerSidePct": slip * 100,
                **bh_m,
            })

    if not rows:
        raise RuntimeError("No pude calcular comparaciones. Confirma que las carpetas contengan equity_curve.csv.")
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print("=" * 86)
    print("COMPARACIÓN GLD BUY-AND-HOLD VS GOLD MOMENTUM v1.3")
    print("=" * 86)
    print(f"Precios usados: {data_path}")
    print("Periodo comparable: fechas que aparecen en cada equity_curve.csv")
    print("Capital inicial: $10,000 | Buy-and-hold: fracciones teóricas, costos de entrada y salida incluidos")
    print("Comisión: la asociada a cada carpeta; slippage buy-and-hold: 0.06% y 0.08% por lado")
    cols = ["ReportFolder", "CommissionPerSidePct", "Asset", "FinalEquity", "NetReturnPct", "CAGR_Pct", "Sharpe", "MaxDrawdownPct"]
    print(result[cols].to_string(index=False, formatters={
        "CommissionPerSidePct": lambda x: f"{x:.3f}%",
        "FinalEquity": lambda x: f"${x:,.2f}",
        "NetReturnPct": lambda x: f"{x:.2f}%",
        "CAGR_Pct": lambda x: f"{x:.2f}%",
        "Sharpe": lambda x: f"{x:.3f}",
        "MaxDrawdownPct": lambda x: f"{x:.2f}%",
    }))
    print(f"\nCSV guardado en: {OUT}")
    print("Nota: benchmark teórico con acciones fraccionarias; no es una simulación de broker.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
