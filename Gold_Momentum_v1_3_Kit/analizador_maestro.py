#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analizador maestro de resultados Gold Momentum. Solo lee archivos; no opera."""
from pathlib import Path
import argparse, zipfile, csv, json, math, sys
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("Falta pandas. Ejecuta INSTALAR_Y_ANALIZAR.bat o: python -m pip install pandas")
    raise SystemExit(1)

METRIC_FILES = {
    "summary_metrics.csv": "metricas",
    "closed_trades.csv": "operaciones",
    "equity_curve.csv": "capital",
    "walk_forward_yearly_oos.csv": "walk_forward",
    "crisis_windows.csv": "crisis",
    "monte_carlo.csv": "monte_carlo",
    "cost_sensitivity.csv": "costes",
    "daily_paper_signals.csv": "senales",
    "paper_trading_log.csv": "bitacora",
    "comparacion_vs_buy_hold.csv": "benchmark",
}

def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"AVISO: no pude leer {path}: {e}")
        return pd.DataFrame()

def fmt(x, kind=""):
    if pd.isna(x): return "N/D"
    try:
        v=float(x)
        if kind=="money": return f"${v:,.2f}"
        if kind=="pct": return f"{v:.2f}%"
        if kind=="sharpe": return f"{v:.3f}"
        if kind=="int": return f"{int(v):,}"
        return f"{v:,.3f}" if abs(v)<100 else f"{v:,.2f}"
    except (ValueError, TypeError):
        return str(x)

def find_report_dirs(root):
    found=set()
    for p in root.rglob("summary_metrics.csv"):
        found.add(p.parent)
    for p in root.rglob("equity_curve.csv"):
        found.add(p.parent)
    # Permit folders with only other detailed outputs, but avoid treating root as a report.
    return sorted(found, key=lambda p: str(p).lower())

def metric_row(folder):
    p=folder/"summary_metrics.csv"
    df=read_csv(p) if p.exists() else pd.DataFrame()
    if df.empty: return {}
    return df.iloc[0].to_dict()

def main():
    ap=argparse.ArgumentParser(description="Reúne resultados de varias estrategias Gold Momentum en un informe.")
    ap.add_argument("carpeta", nargs="?", default=".", help="Carpeta que contiene ZIPs y/o carpetas gold_reports*")
    args=ap.parse_args()
    root=Path(args.carpeta).resolve()
    if not root.exists(): print(f"No existe la carpeta: {root}"); return 2

    # Extrae ZIPs de resultados de forma segura, en una subcarpeta propia.
    extracted=root/"_extraido_para_analisis"
    for zp in root.glob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as z:
                safe=[]
                for member in z.infolist():
                    dest=(extracted/member.filename).resolve()
                    if dest == extracted.resolve() or extracted.resolve() in dest.parents:
                        safe.append(member)
                if safe:
                    extracted.mkdir(exist_ok=True)
                    z.extractall(extracted, members=safe)
                    print(f"ZIP leído/descomprimido: {zp.name}")
        except zipfile.BadZipFile:
            print(f"AVISO: ZIP inválido: {zp.name}")

    dirs=find_report_dirs(root)
    if not dirs:
        print("\nNO ENCONTRÉ RESULTADOS DE BACKTEST.")
        print("Busqué summary_metrics.csv y equity_curve.csv dentro de la carpeta y subcarpetas.")
        print("El ZIP original Gold Momentum v1.3 contiene el código y la guía, pero no trae resultados históricos ya calculados.")
        print("Primero ejecuta el backtest para que se generen gold_reports*/. Después coloca las carpetas de las 4 variantes aquí y vuelve a ejecutar este analizador.")
        return 1

    out=root/"resultado_maestro"
    out.mkdir(exist_ok=True)
    all_metrics=[]; all_trades=[]; all_wf=[]; all_crisis=[]; all_mc=[]; all_cost=[]; all_bench=[]; all_signals=[]
    report=[]
    report += ["INFORME MAESTRO — GOLD MOMENTUM", "="*72,
               f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               f"Carpeta analizada: {root}", f"Variantes/carpetas de resultados encontradas: {len(dirs)}",
               "", "IMPORTANTE: este informe resume los CSV existentes. No vuelve a ejecutar ni altera backtests.",
               "No es una garantía de resultados futuros ni una orden de compra/venta.", ""]
    for idx, folder in enumerate(dirs,1):
        label=folder.name
        m=metric_row(folder)
        m["Estrategia"] = label
        m["Carpeta"] = str(folder.relative_to(root)) if folder.is_relative_to(root) else str(folder)
        all_metrics.append(m)
        report += [f"{idx}. {label}", "-"*72, f"Ruta: {m['Carpeta']}"]
        if m:
            aliases=[
                (["Start"],"Inicio"),(["End"],"Fin"),(["Bars"],"Velas", "int"),
                (["InitialCapital"],"Capital inicial","money"),(["FinalEquity"],"Capital final","money"),
                (["NetReturnPct"],"Rentabilidad neta","pct"),(["CAGR_Pct"],"Rentabilidad anualizada CAGR","pct"),
                (["Sharpe"],"Sharpe","sharpe"),(["MaxDrawdownPct"],"Drawdown máximo","pct"),
                (["ClosedTrades"],"Operaciones cerradas","int"),(["WinRatePct"],"Tasa de acierto","pct"),
                (["ProfitFactor"],"Profit factor"),(["MaxLosingStreak"],"Racha máxima de pérdidas","int"),
                (["MaxUnderwaterDays"],"Máximo tiempo bajo máximo (sesiones)","int"),
                (["ExposurePct"],"Tiempo en mercado","pct"),(["CommissionPerSidePct"],"Comisión por lado","pct"),
                (["NormalSlippagePerSidePct"],"Slippage normal por lado","pct"),
                (["HighVolSlippagePerSidePct"],"Slippage alta volatilidad por lado","pct"),
                (["OpenPositionAtEnd"],"Posición abierta al final")]
            for item in aliases:
                keys,title=item[0],item[1]; kind=item[2] if len(item)>2 else ""
                val=next((m[k] for k in keys if k in m),None)
                if val is not None: report.append(f"  {title}: {fmt(val,kind)}")
        else:
            report.append("  No hay summary_metrics.csv legible; se reportará lo disponible.")
        for filename, kind in METRIC_FILES.items():
            p=folder/filename
            if not p.exists(): continue
            df=read_csv(p)
            if df.empty: continue
            df.insert(0,"Estrategia",label)
            if kind=="metricas": continue
            target={"operaciones":all_trades,"walk_forward":all_wf,"crisis":all_crisis,
                    "monte_carlo":all_mc,"costes":all_cost,"benchmark":all_bench,
                    "senales":all_signals,"bitacora":all_signals}.get(kind)
            if target is not None: target.append(df)
            if kind=="operaciones" and "NetPnL" in df.columns:
                pnl=pd.to_numeric(df["NetPnL"],errors="coerce").dropna()
                report += [f"  Operaciones con PnL legible: {len(pnl)}",
                           f"  PnL neto acumulado de operaciones cerradas: {fmt(pnl.sum(),'money')}",
                           f"  Ganadoras / perdedoras: {(pnl>0).sum()} / {(pnl<0).sum()}"]
            if kind=="walk_forward" and "OOS_ReturnPct" in df.columns:
                r=pd.to_numeric(df["OOS_ReturnPct"],errors="coerce").dropna()
                if len(r): report.append(f"  Walk-forward OOS: {len(r)} años; años positivos {int((r>0).sum())}/{len(r)}; retorno anual mediano {fmt(r.median(),'pct')}")
            if kind=="crisis" and "ReturnPct" in df.columns:
                r=pd.to_numeric(df["ReturnPct"],errors="coerce").dropna()
                if len(r): report.append(f"  Ventanas de crisis legibles: {len(r)}; ventanas positivas {int((r>0).sum())}/{len(r)}")
            if kind=="monte_carlo":
                report.append("  Monte Carlo: resultados guardados en la tabla consolidada; revisar percentiles y número de operaciones.")
            if kind=="benchmark":
                report.append("  Comparación GLD buy-and-hold disponible; ver comparacion_vs_buy_hold_consolidada.csv.")
        report.append("")

    # Save consolidated tables.
    tables=[
        ("metricas_consolidadas.csv",all_metrics),
        ("operaciones_consolidadas.csv",all_trades),
        ("walk_forward_consolidado.csv",all_wf),
        ("crisis_consolidadas.csv",all_crisis),
        ("monte_carlo_consolidado.csv",all_mc),
        ("sensibilidad_costes_consolidada.csv",all_cost),
        ("comparacion_vs_buy_hold_consolidada.csv",all_bench),
        ("senales_y_bitacoras_consolidadas.csv",all_signals),
    ]
    for filename, frames in tables:
        if filename=="metricas_consolidadas.csv":
            df=pd.DataFrame(frames)
        else:
            df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        if not df.empty: df.to_csv(out/filename,index=False,encoding="utf-8-sig")

    # Ranking: prioritise net return and drawdown as descriptive ranking, not a promise.
    mdf=pd.DataFrame(all_metrics)
    if not mdf.empty:
        for c in ["NetReturnPct","CAGR_Pct","Sharpe","MaxDrawdownPct","ProfitFactor","WinRatePct"]:
            if c in mdf: mdf[c]=pd.to_numeric(mdf[c],errors="coerce")
        sort_cols=[c for c in ["CAGR_Pct","NetReturnPct"] if c in mdf.columns]
        if sort_cols:
            ranked=mdf.sort_values(sort_cols,ascending=False,na_position="last")
            report += ["RESUMEN COMPARATIVO", "="*72]
            for _,r in ranked.iterrows():
                report.append(f"- {r.get('Estrategia','?')}: retorno {fmt(r.get('NetReturnPct'))}%, CAGR {fmt(r.get('CAGR_Pct'))}%, Sharpe {fmt(r.get('Sharpe'),'sharpe')}, drawdown {fmt(r.get('MaxDrawdownPct'))}%, operaciones {fmt(r.get('ClosedTrades'),'int')}")
            report += ["", "LECTURA PRUDENTE", "="*72,
                "• No elijas solo por rentabilidad: contrasta drawdown, Sharpe, Profit Factor y robustez por años.",
                "• Comprueba que las cuatro variantes usan el mismo periodo, capital, comisiones y slippage antes de compararlas.",
                "• Un backtest con pocas operaciones o un único régimen de mercado puede ser poco representativo.",
                "• Si hay posición abierta al final, el resultado puede incluir valoración no realizada.",
                "• Los datos faltantes se muestran como N/D; no se han inventado ni rellenado."]
    report += ["", "ARCHIVOS GENERADOS", "="*72, f"Todos los resultados se guardaron en: {out}"]
    (out/"INFORME_MAESTRO.txt").write_text("\n".join(report),encoding="utf-8")
    (out/"INFORME_MAESTRO.html").write_text("<html><meta charset='utf-8'><title>Informe Maestro Gold Momentum</title><body><pre style='white-space:pre-wrap;font:14px Arial'>"+ "\n".join(report).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") +"</pre></body></html>",encoding="utf-8")
    print("\n".join(report))
    print(f"\nListo. Abre: {out/'INFORME_MAESTRO.html'}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
