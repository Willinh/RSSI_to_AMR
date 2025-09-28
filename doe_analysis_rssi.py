# -*- coding: utf-8 -*-
"""
DOE Analysis for RSSI Prediction Experiments
--------------------------------------------
Single-file script to:
  1) Load 1..N result files (CSV or Excel)
  2) Build a standardized Z-Composite per (dataset_id, seconds)
  3) Fit OLS with dataset as block + main effects + selected 2-way interactions
  4) Export CSV summaries and a PDF with figures (each page: chart + "how to read")

Usage:
  python doe_analysis_rssi.py \
      --inputs "results/*.csv,/path/to/other.xlsx" \
      --output_dir "./out" \
      --pdf "report_doe.pdf"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

CANON_MAP = {
    "model": ["model", "modelo", "MODEL", "MODEL_TYPE", "model_type"],
    "resample_ms": ["group_msec", "group_ms", "resample (ms)", "resample_ms", "resample"],
    "seconds": ["seconds_ahead", "seconds to predict (s)", "seconds_to_predict", "sec_to_predict"],
    "forecast_steps": ["forecast steps", "forecast_steps"],
    "timesteps": ["timesteps", "timesteps_orig", "window", "lookback", "Timesteps"],
    "epochs": ["epochs", "Epochs"],
    "units_str": ["units_range", "units (min-max:step)", "units", "Units"],
    "batch_size": ["batch_size", "batchsize", "batch", "Batchsize"],
    "train_dataset": ["train_dataset", "train dataset"],
    "exp_dataset": ["exp_dataset", "exp dataset", "test_dataset", "test dataset"],
    "z_composite": ["z-score_composite", "z-score composite", "ranking final", "Z_Composite_std"],
}

METRIC_MIN_BETTER = [
    "multi_mae","multi_rmse","multi_mse",
    "step1_mae","step1_rmse","stepN_mae","stepN_rmse",
    "direct_mae","direct_rmse","direct_mse"
]
METRIC_MAX_BETTER = [
    "multi_r2","step1_r2","stepN_r2","direct_r2"
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    low2orig = {str(c).lower(): c for c in df.columns}
    canon = {}
    for k, syns in CANON_MAP.items():
        for s in syns:
            if s.lower() in low2orig:
                canon[k] = low2orig[s.lower()]
                break
    df.attrs["canon"] = canon
    return df

def getcol(df, name, default=None):
    return df.attrs.get("canon", {}).get(name, default)

def parse_units(u: str):
    if not isinstance(u, str):
        return (np.nan, np.nan, np.nan)
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(\d+)\s*$", u)
    if not m:
        return (np.nan, np.nan, np.nan)
    lo, hi, st = map(int, m.groups())
    avg = 0.5*(lo+hi)
    span = hi-lo
    return (avg, span, st)

def derive_dataset_id(df: pd.DataFrame) -> pd.Series:
    tr = getcol(df, "train_dataset")
    ex = getcol(df, "exp_dataset")
    if tr is not None and ex is not None:
        A = df[tr].astype(str).str.replace(r".*/", "", regex=True)
        B = df[ex].astype(str).str.replace(r".*/", "", regex=True)
        return (A + "→" + B).str.replace(r"\.parquet$|\.csv$|\.xlsx$", "", regex=True)
    if tr is not None:
        return df[tr].astype(str)
    if ex is not None:
        return df[ex].astype(str)
    return pd.Series(["DATASET"]*len(df), index=df.index)

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return normalize_columns(df)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
            return normalize_columns(df)
        except Exception:
            raw = pd.read_excel(path, header=None)
            header = raw.iloc[1].tolist()
            df = raw.iloc[2:].copy()
            df.columns = header
            return normalize_columns(df)
    raise RuntimeError(f"Unsupported file: {path}")

def load_many(patterns: str):
    frames = []
    for token in patterns.split(","):
        token = token.strip()
        if not token:
            continue
        p = Path(token)
        if p.exists():
            df = load_any(p)
            df["__source__"] = str(p)
            frames.append(df)
        else:
            for q in Path(".").glob(token):
                df = load_any(q)
                df["__source__"] = str(q)
                frames.append(df)
    if not frames:
        raise RuntimeError("No inputs found.")
    df = pd.concat(frames, ignore_index=True)
    return normalize_columns(df)

def build_composite(df: pd.DataFrame) -> pd.DataFrame:
    sec_col = getcol(df, "seconds")
    if sec_col is None:
        fs = getcol(df, "forecast_steps")
        rs = getcol(df, "resample_ms")
        if fs and rs:
            df["_seconds_auto_"] = pd.to_numeric(df[fs], errors="coerce") * (pd.to_numeric(df[rs], errors="coerce")/1000.0)
            sec_col = "_seconds_auto_"
        else:
            df["_seconds_auto_"] = np.nan
            sec_col = "_seconds_auto_"

    df["_dataset_id_"] = derive_dataset_id(df)

    metrics_min = [c for c in METRIC_MIN_BETTER if c in df.columns]
    metrics_max = [c for c in METRIC_MAX_BETTER if c in df.columns]

    if not metrics_min and not metrics_max and getcol(df, "z_composite"):
        base = pd.to_numeric(df[getcol(df, "z_composite")], errors="coerce")
        z = base.groupby([df["_dataset_id_"], df[sec_col]]).transform(
            lambda s: (s - s.mean())/ (s.std(ddof=0) if s.std(ddof=0)!=0 else 1.0)
        )
        df["Z_Composite_std"] = z
        return df

    def zscore(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / (sd if sd!=0 else 1.0)

    z_list = []
    for (_, g) in df.groupby([df["_dataset_id_"], df[sec_col]]):
        g = g.copy()
        comps = []
        for c in metrics_min:
            s = pd.to_numeric(g[c], errors="coerce")
            comps.append(-zscore(s))
        for c in metrics_max:
            s = pd.to_numeric(g[c], errors="coerce")
            comps.append( zscore(s))
        if comps:
            arr = np.vstack([np.nan_to_num(x, nan=np.nanmean(x)) for x in comps])
            z_list.append(pd.Series(np.nanmean(arr, axis=0), index=g.index))
        else:
            z_list.append(pd.Series(np.nan, index=g.index))
    df["Z_Composite_std"] = pd.concat(z_list).sort_index()
    return df

def feature_matrix(df: pd.DataFrame):
    y = pd.to_numeric(df["Z_Composite_std"], errors="coerce")

    # block
    D = pd.get_dummies(df["_dataset_id_"].astype(str), prefix="ds", drop_first=True)

    # model
    model_col = getcol(df, "model") or "model"
    if model_col not in df.columns:
        df[model_col] = "MODEL"
    M = pd.get_dummies(df[model_col].astype(str), prefix="model", drop_first=True)

    # resample scaled [-1,1]
    rs_col = getcol(df, "resample_ms")
    resample = pd.to_numeric(df[rs_col], errors="coerce") if rs_col else pd.Series(np.nan, index=df.index)
    if resample.notna().any():
        lo, hi = resample.min(), resample.max()
        rescale = (2*(resample - lo)/(hi-lo) - 1.0) if lo!=hi else pd.Series(0.0, index=df.index)
    else:
        rescale = pd.Series(np.nan, index=df.index)

    # others
    def num(colname):
        c = getcol(df, colname) or colname
        return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)

    seconds  = num("seconds")
    timesteps= num("timesteps")
    epochs   = num("epochs")

    u_col = getcol(df, "units_str")
    if u_col:
        parsed = df[u_col].astype(str).apply(parse_units)
        units_avg  = parsed.apply(lambda t: t[0])
        units_span = parsed.apply(lambda t: t[1])
        units_step = parsed.apply(lambda t: t[2])
    else:
        units_avg=units_span=units_step = pd.Series(np.nan, index=df.index)

    b_col = getcol(df, "batch_size")
    batch  = pd.to_numeric(df[b_col], errors="coerce") if b_col else pd.Series(np.nan, index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        batch_log2 = np.log2(batch)

    X = pd.DataFrame({"Intercept": 1.0}, index=df.index)
    X["resample_scaled"] = rescale
    X["seconds"]   = seconds
    X["timesteps"] = timesteps
    X["epochs"]    = epochs
    X["units_avg"]  = units_avg
    X["units_span"] = units_span
    X["units_step"] = units_step
    X["batch"]      = batch
    X["batch_log2"] = batch_log2
    X = pd.concat([X, M, D], axis=1)

    # interactions among selected predictors (exclude dataset dummies)
    Ds = list(D.columns)
    base = [c for c in X.columns if c not in ["Intercept"] + Ds]
    keep = [c for c in base if c.startswith("model_") or c in ["resample_scaled","seconds","timesteps","epochs","units_avg","batch_log2"]]
    for i in range(len(keep)):
        for j in range(i+1, len(keep)):
            a, b = keep[i], keep[j]
            X[f"{a}:{b}"] = X[a]*X[b]

    valid = (~X.isna().any(axis=1)) & (~y.isna())
    return X.loc[valid].copy(), y.loc[valid].copy()

def fit_ols(X: pd.DataFrame, y: pd.Series):
    Xmat = X.values.astype(float)
    yvec = y.values.astype(float)
    XtX = Xmat.T @ Xmat
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ Xmat.T @ yvec
    yhat = Xmat @ beta
    resid = yvec - yhat
    n, p = Xmat.shape
    dof = max(1, n - p)
    s2 = float((resid @ resid)/dof)
    se = np.sqrt(np.diag(s2*XtX_inv))
    tstat = beta/se
    if HAVE_SCIPY:
        pval = 2*(1 - stats.t.cdf(np.abs(tstat), df=dof))
    else:
        pval = np.full_like(tstat, np.nan, dtype=float)
    effects = pd.DataFrame({"term": X.columns, "coef": beta, "std_err": se, "t_stat": tstat, "p_value": pval})
    r2 = 1 - (resid@resid)/np.sum((yvec - yvec.mean())**2)
    return effects, yhat, resid, dof, float(r2)

def page_caption(fig, text):
    wrapped = "\\n".join(textwrap.wrap(text, width=100))
    fig.text(0.1, 0.04, wrapped, ha="left", va="bottom", fontsize=9)

def fig_pareto(effects: pd.DataFrame, pdf: PdfPages, title: str):
    terms = effects[effects["term"]!="Intercept"].copy()
    terms["abs_coef"] = terms["coef"].abs()
    top = terms.sort_values("abs_coef", ascending=False).head(15)
    fig = plt.figure(figsize=(9,6))
    plt.bar(top["term"], top["abs_coef"])
    plt.xticks(rotation=60, ha="right")
    plt.title(title); plt.ylabel("|coef| (escala Z-Composite)")
    plt.tight_layout(rect=[0,0.1,1,1])
    page_caption(fig, "Como ler: barras maiores indicam termos (efeitos principais ou interações) com maior influência no modelo linear ajustado. Use para priorizar investigação.")
    pdf.savefig(fig); plt.close(fig)

def fig_means_by_levels(df: pd.DataFrame, col: str, ycol: str, title: str, pdf: PdfPages):
    means = df.groupby(col, dropna=True)[ycol].mean().reset_index()
    fig = plt.figure(figsize=(7,5))
    plt.bar(means[col].astype(str), means[ycol])
    plt.title(title); plt.xlabel(col); plt.ylabel(f"Média de {ycol}")
    plt.tight_layout(rect=[0,0.12,1,1])
    page_caption(fig, f"Como ler: cada barra mostra a média de {ycol} para um nível de {col}. Útil para perceber tendências ou curvaturas (níveis intermediários vs extremos).")
    pdf.savefig(fig); plt.close(fig)

def fig_interaction(df: pd.DataFrame, a: str, b: str, ycol: str, title: str, pdf: PdfPages, levels_a=None, levels_b=None):
    sA, sB = df[a], df[b]
    uA = np.sort(pd.unique(sA.dropna())); uB = np.sort(pd.unique(sB.dropna()))
    if levels_a is not None: uA = [x for x in uA if x in levels_a]
    if levels_b is not None: uB = [x for x in uB if x in levels_b]
    if len(uA)<2 or len(uB)<2: return
    Bs = [uB[0], uB[-1]]
    means = {bval: [df.loc[(sA==uA[0])&(sB==bval), ycol].mean(),
                    df.loc[(sA==uA[-1])&(sB==bval), ycol].mean()] for bval in Bs}
    fig = plt.figure(figsize=(6,4))
    xs = [str(uA[0]), str(uA[-1])]
    for bval in Bs:
        plt.plot(xs, means[bval], marker="o", label=f"{b}={bval}")
    plt.xlabel(f"{a} (níveis)"); plt.ylabel(f"Média de {ycol}")
    plt.title(title); plt.legend()
    plt.tight_layout(rect=[0,0.1,1,1])
    page_caption(fig, "Como ler: linhas não paralelas indicam interação entre fatores. A diferença entre inclinações sugere força da interação.")
    pdf.savefig(fig); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Comma-separated globs or paths (CSV/XLSX)")
    ap.add_argument("--output_dir", default="./out", help="Output directory")
    ap.add_argument("--pdf", default="analysis_report.pdf", help="PDF filename (inside output_dir)")
    args = ap.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_many(args.inputs)
    df = build_composite(df)
    df.to_csv(outdir/"combined_with_composite.csv", index=False)

    X, y = feature_matrix(df)
    effects, yhat, resid, dof, r2 = fit_ols(X, y)
    effects.to_csv(outdir/"ols_effects.csv", index=False)

    # simple main-effect diffs for 2-level columns (if applicable)
    def mean_diff_2level(col):
        if col not in df.columns: return np.nan
        s = pd.to_numeric(df[col], errors="coerce")
        u = np.sort(s.dropna().unique())
        if len(u)>=2:
            low, high = u[0], u[-1]
            m_low = df.loc[s==low, "Z_Composite_std"].mean()
            m_high= df.loc[s==high,"Z_Composite_std"].mean()
            return float(m_high - m_low)
        return np.nan

    rs_col = getcol(df, "resample_ms") or "resample_ms"
    sec_col= getcol(df, "seconds") or "seconds"
    ts_col = getcol(df, "timesteps") or "timesteps"
    ep_col = getcol(df, "epochs") or "epochs"
    u_col  = getcol(df, "units_str") or "units_str"
    b_col  = getcol(df, "batch_size") or "batch_size"
    model_col = getcol(df, "model") or "model"

    rows = []
    for col in [rs_col, sec_col, ts_col, ep_col, b_col]:
        if col in df.columns:
            rows.append({"factor": col, "mean_diff_(hi - lo)": mean_diff_2level(col)})
    if model_col in df.columns:
        means = df.groupby(model_col, dropna=True)["Z_Composite_std"].mean()
        if len(means)>=2:
            rows.append({"factor": model_col, "mean_diff_(hi - lo)": float(means.max()-means.min())})
    pd.DataFrame(rows).to_csv(outdir/"main_effects_simple.csv", index=False)

    # PDF
    with PdfPages(outdir/args.pdf) as pdf:
        fig = plt.figure(figsize=(8.5, 11)); plt.axis("off")
        txt = [
            "DOE Analysis Report – RSSI Prediction (AMR)",
            "",
            f"Rows used in OLS: n={len(y)}, p={effects.shape[0]+1}, dof≈{dof}, R²≈{r2:.3f}",
            "Este relatório traz gráficos com instruções de interpretação (não conclusões).",
        ]
        fig.text(0.1, 0.8, "\n".join(txt), ha="left", va="top", fontsize=12); pdf.savefig(fig); plt.close(fig)

        fig_pareto(effects, pdf, "Pareto – |coeficientes| (OLS)")

        if rs_col in df.columns: fig_means_by_levels(df, rs_col, "Z_Composite_std", "Médias por nível – Resample (ms)", pdf)
        if sec_col in df.columns: fig_means_by_levels(df, sec_col,"Z_Composite_std","Médias por nível – Seconds to predict (s)", pdf)
        if ts_col  in df.columns: fig_means_by_levels(df, ts_col, "Z_Composite_std", "Médias por nível – Timesteps", pdf)
        if ep_col  in df.columns: fig_means_by_levels(df, ep_col, "Z_Composite_std", "Médias por nível – Epochs", pdf)
        if u_col   in df.columns: fig_means_by_levels(df, u_col,  "Z_Composite_std", "Médias por nível – Units (min-max:step)", pdf)
        if b_col   in df.columns: fig_means_by_levels(df, b_col,  "Z_Composite_std", "Médias por nível – Batchsize", pdf)

        if ts_col in df.columns and u_col in df.columns:
            fig_interaction(df, ts_col, u_col, "Z_Composite_std", "Interação – Timesteps × Units", pdf)
        if rs_col in df.columns and sec_col in df.columns:
            fig_interaction(df, rs_col, sec_col, "Z_Composite_std", "Interação – Resample × Seconds", pdf)

    print("Arquivos salvos em:", outdir)

if __name__ == "__main__":
    main()
