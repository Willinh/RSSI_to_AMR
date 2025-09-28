# -*- coding: utf-8 -*-
"""
DOE Analysis for RSSI Prediction – V3 (config block + robust parsing)
--------------------------------------------------------------------
Novidades da V3:
- Bloco de configuração no início do arquivo (DEFAULT_*). Se você rodar no PyCharm com "Run",
  o script usa esses DEFAULTS. Os parâmetros de linha de comando continuam funcionando e
  *sobrescrevem* os defaults.
- Parsing numérico robusto (vírgula decimal / separador de milhar).
- Colunas 100% NaN são descartadas antes de construir X; evita perder todas as linhas.
- PDF com páginas de "como ler" para cada gráfico.

Edite SOMENTE esta seção logo abaixo antes de rodar:
    DEFAULT_INPUTS, DEFAULT_OUTPUT_DIR, DEFAULT_PDF_NAME,
    DEFAULT_COMPOSITE_MODE, DEFAULT_COMPOSITE_COL, DEFAULT_AUDIT
"""

# ===================== CONFIGURAÇÃO DO USUÁRIO =====================
# Caminhos/Globs para arquivos de resultados (CSV e/ou XLSX). Pode ser múltiplo, separados por vírgula.
# Exemplo Windows (use string raw r"..."): r"C:\...\Resultados.xlsx"
DEFAULT_INPUTS = r"C:\Users\Micro\Documents\RSSI_to_AMR\experiments_archive\Exp_2DS_DS11_288Tests_autorun\Resultados_Testes_2DS_DoE_full_288.xlsx"  # <<-- PREENCHER: ex.: r"C:\Users\...\Resultados_Testes_2DS_DoE_full_288.xlsx"
# Pasta de saída
DEFAULT_OUTPUT_DIR = r"C:\Users\Micro\Documents\RSSI_to_AMR\experiments_archive\Exp_2DS_DS11_288Tests_autorun\out_DoE"  # <<-- PREENCHER: ex.: r"C:\Users\...\out_DoE"

# Nome do PDF dentro da pasta de saída
DEFAULT_PDF_NAME = "relatorio_DOE_2DS_DS11_288Tests.pdf"

# Como montar o composto: "auto" | "column" | "metrics"
DEFAULT_COMPOSITE_MODE = "column"

# Nome da coluna do composto caso DEFAULT_COMPOSITE_MODE == "column"
DEFAULT_COMPOSITE_COL = "Z-Score_Composite"

# Salvar arquivos de auditoria? (True/False)
DEFAULT_AUDIT = True
# ===================================================================

import argparse, textwrap, re
from pathlib import Path
import numpy as np
import pandas as pd

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
    "train_dataset": ["train_dataset", "train dataset", "Datasets usados", "datasets usados"],
    "exp_dataset": ["exp_dataset", "exp dataset", "test_dataset", "test dataset"],
    "z_composite": ["z-score_composite", "z-score composite", "ranking final", "Z_Composite_std", "Z-Score_Composite"],
}

METRIC_MIN_BETTER = ["multi_mae","multi_rmse","multi_mse","step1_mae","step1_rmse","stepN_mae","stepN_rmse","direct_mae","direct_rmse","direct_mse"]
METRIC_MAX_BETTER = ["multi_r2","step1_r2","stepN_r2","direct_r2"]

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

def smart_to_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    n1 = s.notna().sum()
    if n1 >= max(1, int(0.5*len(series))):
        return s
    s2 = pd.to_numeric(series.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    n2 = s2.notna().sum()
    return s2 if n2 > n1 else s

def parse_units(u: str):
    if not isinstance(u, str): return (np.nan, np.nan, np.nan)
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(\d+)\s*$", u)
    if not m: return (np.nan, np.nan, np.nan)
    lo, hi, st = map(int, m.groups())
    return (0.5*(lo+hi), hi-lo, st)

def derive_dataset_id(df: pd.DataFrame) -> pd.Series:
    tr = getcol(df, "train_dataset"); ex = getcol(df, "exp_dataset")
    if tr is not None and ex is not None:
        A = df[tr].astype(str).str.replace(r".*/", "", regex=True)
        B = df[ex].astype(str).str.replace(r".*/", "", regex=True)
        return (A + "→" + B).str.replace(r"\.parquet$|\.csv$|\.xlsx$", "", regex=True)
    if tr is not None: return df[tr].astype(str)
    if ex is not None: return df[ex].astype(str)
    return pd.Series(["DATASET"]*len(df), index=df.index)

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return normalize_columns(pd.read_csv(path))

    if path.suffix.lower() in {".xlsx", ".xls"}:
        # 1) tenta ler normalmente
        df0 = pd.read_excel(path)
        # Heurística: muitos "Unnamed" OU presença de grupos típicos => cabeçalho em 2 linhas
        cols = [str(c) for c in df0.columns]
        has_many_unnamed = sum(c.startswith("Unnamed:") for c in cols) >= max(3, int(0.3*len(cols)))
        has_group_headers = any(g in cols for g in ["Hiperparametros", "Tempos", "Métricas - Resultados"])
        if has_many_unnamed or has_group_headers:
            raw = pd.read_excel(path, header=None)
            # usa a linha 2 (índice 1) como header real, e os dados a partir da linha 3 (índice 2)
            header = raw.iloc[1].tolist()
            df = raw.iloc[2:].copy()
            df.columns = header
            return normalize_columns(df)
        else:
            return normalize_columns(df0)

    raise RuntimeError(f"Unsupported file: {path}")


def load_many(patterns: str):
    frames = []
    for token in patterns.split(","):
        token = token.strip()
        if not token: continue
        p = Path(token)
        if p.exists():
            df = load_any(p); df["__source__"] = str(p); frames.append(df)
        else:
            for q in Path(".").glob(token):
                df = load_any(q); df["__source__"] = str(q); frames.append(df)
    if not frames: raise RuntimeError("No inputs found.")
    df = pd.concat(frames, ignore_index=True)
    return normalize_columns(df)

def build_composite(df: pd.DataFrame, mode: str, column_name: str=None) -> pd.DataFrame:
    sec_col = getcol(df, "seconds")
    if sec_col is None:
        fs = getcol(df, "forecast_steps"); rs = getcol(df, "resample_ms")
        if fs and rs:
            df["_seconds_auto_"] = smart_to_numeric(df[fs]) * (smart_to_numeric(df[rs])/1000.0)
            sec_col = "_seconds_auto_"
        else:
            df["_seconds_auto_"] = np.nan; sec_col = "_seconds_auto_"
    df["_dataset_id_"] = derive_dataset_id(df)

    def z_by_group(base):
        return base.groupby([df["_dataset_id_"], df[sec_col]]).transform(
            lambda s: (s - s.mean())/(s.std(ddof=0) if s.std(ddof=0)!=0 else 1.0)
        )

    if mode == "column":
        col = column_name if (column_name and column_name in df.columns) else getcol(df, "z_composite")
        if col is None:
            raise RuntimeError("--composite=column mas a coluna não existe. Use --composite-col NomeExato")
        base = smart_to_numeric(df[col])
        df["Z_Composite_std"] = z_by_group(base)
        return df

    metrics_min = [c for c in METRIC_MIN_BETTER if c in df.columns]
    metrics_max = [c for c in METRIC_MAX_BETTER if c in df.columns]

    if mode in ("auto","metrics") and (metrics_min or metrics_max):
        def zscore(s):
            sd = s.std(ddof=0); sd = sd if (sd and not np.isnan(sd)) else 1.0
            return (s - s.mean())/sd
        z_list = []
        for (_, g) in df.groupby([df["_dataset_id_"], df[sec_col]]):
            comps = []
            for c in metrics_min: comps.append(-zscore(smart_to_numeric(g[c])))
            for c in metrics_max: comps.append( zscore(smart_to_numeric(g[c])))
            if comps:
                arr = np.vstack([np.nan_to_num(x, nan=np.nanmean(x)) for x in comps])
                z_list.append(pd.Series(np.nanmean(arr, axis=0), index=g.index))
            else:
                z_list.append(pd.Series(np.nan, index=g.index))
        df["Z_Composite_std"] = pd.concat(z_list).sort_index()
        if mode == "auto" and df["Z_Composite_std"].isna().all():
            col = getcol(df, "z_composite")
            if col is not None:
                df["Z_Composite_std"] = z_by_group(smart_to_numeric(df[col]))
        return df

    col = getcol(df, "z_composite")
    if col is not None:
        df["Z_Composite_std"] = z_by_group(smart_to_numeric(df[col]))
        return df

    raise RuntimeError("Não foi possível construir o Z-Composite.")

def feature_matrix(df: pd.DataFrame):
    y = smart_to_numeric(df["Z_Composite_std"])

    D = pd.get_dummies(df["_dataset_id_"].astype(str), prefix="ds", drop_first=True)
    model_col = getcol(df, "model") or "model"
    if model_col not in df.columns: df[model_col] = "MODEL"
    M = pd.get_dummies(df[model_col].astype(str), prefix="model", drop_first=True)

    def num(colname):
        c = getcol(df, colname) or colname
        return smart_to_numeric(df[c]) if c in df.columns else pd.Series(np.nan, index=df.index)

    rs = num("resample_ms")
    sec= num("seconds")
    if sec.isna().all():
        fs_col = getcol(df, "forecast_steps")
        rs_col = getcol(df, "resample_ms")
        if fs_col and rs_col and (fs_col in df.columns) and (rs_col in df.columns):
            sec = smart_to_numeric(df[fs_col]) * (smart_to_numeric(df[rs_col]) / 1000.0)

    ts = num("timesteps")
    ep = num("epochs")

    u_col = getcol(df, "units_str")
    if u_col:
        parsed = df[u_col].astype(str).apply(parse_units)
        uavg = parsed.apply(lambda t: t[0]); uspan=parsed.apply(lambda t: t[1]); ustep=parsed.apply(lambda t: t[2])
    else:
        uavg=uspan=ustep=pd.Series(np.nan, index=df.index)

    b_col = getcol(df, "batch_size")
    batch = smart_to_numeric(df[b_col]) if b_col else pd.Series(np.nan, index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        blog2 = np.log2(batch)

    if rs.notna().any() and rs.min()!=rs.max():
        rscaled = (2*(rs - rs.min())/(rs.max()-rs.min()))-1.0
    else:
        rscaled = pd.Series(0.0, index=df.index)

    Xbase = pd.DataFrame({
        "resample_scaled": rscaled,
        "seconds": sec, "timesteps": ts, "epochs": ep,
        "units_avg": uavg, "units_span": uspan, "units_step": ustep,
        "batch": batch, "batch_log2": blog2
    }, index=df.index).dropna(axis=1, how="all")

    X = pd.concat([pd.DataFrame({"Intercept":1.0}, index=df.index), Xbase, M, D], axis=1)

    # em feature_matrix(), troque a geração de interações por algo mais focado:
    inter_vars = list(M.columns) + ["resample_scaled", "seconds", "timesteps", "units_avg", "batch_log2"]
    pairs = [
        ("timesteps", "units_avg"),
        ("resample_scaled", "seconds"),
        # model x (timesteps, batch_log2) — se existir a dummy de model
    ]
    for a, b in pairs:
        if a in X.columns and b in X.columns:
            X[f"{a}:{b}"] = X[a] * X[b]

    X = X.dropna(axis=1, how="all")
    valid = (~X.isna().any(axis=1)) & (~y.isna())
    return X.loc[valid].copy(), y.loc[valid].copy(), valid

def fit_ols(X: pd.DataFrame, y: pd.Series):
    Xmat = X.values.astype(float); yvec = y.values.astype(float)
    n, p = Xmat.shape
    if n==0:
        return None, None, None, 0, np.nan
    XtX = Xmat.T @ Xmat
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ Xmat.T @ yvec
    yhat = Xmat @ beta
    resid = yvec - yhat
    dof = max(1, n - p)
    s2 = float((resid @ resid) / dof)
    n, p = Xmat.shape
    dof = max(1, n - p)  # se n <= p, não há GL; protege de divisão por zero
    var_beta = s2 * XtX_inv
    diag_var = np.diag(var_beta)
    # corrige “resíduos” numéricos negativos muito pequenos
    diag_var = np.where(diag_var >= 0.0, diag_var, 0.0)
    se = np.sqrt(diag_var) if dof > 0 else np.full(p, np.nan)

    # t-stat: evita divisão por zero/NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = beta / se

    # p-valor
    if HAVE_SCIPY and np.isfinite(dof):
        try:
            from scipy import stats
            pval = 2 * (1 - stats.t.cdf(np.abs(tstat), df=dof))
        except Exception:
            pval = np.full_like(tstat, np.nan, dtype=float)
    else:
        pval = np.full_like(tstat, np.nan, dtype=float)

    # R² com proteção
    denom = np.sum((yvec - yvec.mean()) ** 2)
    r2 = float(1 - (resid @ resid) / denom) if denom > 0 else float("nan")

    effects = pd.DataFrame({
        "term": X.columns,
        "coef": beta,
        "std_err": se,
        "t_stat": tstat,
        "p_value": pval
    })
    return effects, yhat, resid, dof, r2


def audit_dump(outdir: Path, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, valid_mask: pd.Series):
    nan_df = df.isna().sum().sort_values(ascending=False).to_frame("n_nan")
    nan_df.to_csv(outdir/"audit_nan_report.csv")
    lev_cols = ["model","Modelo","group_msec","Resample (ms)","seconds_ahead","Seconds to predict (s)","timesteps","timesteps_orig","Epochs","units_range","Units (min-max:step)","batch_size","Batchsize"]
    lev = []
    for c in [col for col in lev_cols if col in df.columns]:
        u = pd.unique(df[c])
        lev.append({"column":c,"n_levels":int(len([x for x in u if str(x)!="nan"])), "sample_levels":", ".join(map(str, list(u)[:6]))})
    pd.DataFrame(lev).to_csv(outdir/"audit_levels.csv", index=False)
    if X is not None and y is not None and len(y)>0:
        X.to_csv(outdir/"audit_X_matrix.csv", index=False)
        y.to_frame("Z_Composite_std").to_csv(outdir/"audit_y_vector.csv", index=False)
    with open(outdir/"audit.txt","w",encoding="utf-8") as f:
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Valid rows used in OLS: {0 if y is None else len(y)}\n")
        f.write(f"Valid ratio: {0.0 if y is None else (len(y)/len(df) if len(df)>0 else 0.0):.3f}\n")
        f.write("\nFirst columns:\n")
        f.write(", ".join(list(df.columns)[:40]))

def page_caption(fig, text):
    fig.text(0.1, 0.04, "\n".join(textwrap.wrap(text, width=100)), ha="left", va="bottom", fontsize=9)

def fig_pareto(effects: pd.DataFrame, pdf: PdfPages, title: str):
    terms = effects[effects["term"]!="Intercept"].copy()
    terms["abs_coef"] = terms["coef"].abs()
    top = terms.sort_values("abs_coef", ascending=False).head(15)
    fig = plt.figure(figsize=(9,6))
    plt.bar(top["term"], top["abs_coef"])
    plt.xticks(rotation=60, ha="right")
    plt.title(title); plt.ylabel("|coef| (escala Z-Composite)")
    plt.tight_layout(rect=[0,0.1,1,1])
    page_caption(fig, "Como ler: barras maiores indicam termos mais influentes no modelo linear.")
    pdf.savefig(fig); plt.close(fig)

def fig_means_by_levels(df: pd.DataFrame, col: str, ycol: str, title: str, pdf: PdfPages):
    means = df.groupby(col, dropna=True)[ycol].mean().reset_index()
    fig = plt.figure(figsize=(7,5))
    plt.bar(means[col].astype(str), means[ycol])
    plt.title(title); plt.xlabel(col); plt.ylabel(f"Média de {ycol}")
    plt.tight_layout(rect=[0,0.12,1,1])
    page_caption(fig, f"Como ler: cada barra mostra a média de {ycol} para um nível de {col}. Níveis intermediários maiores podem sugerir curvatura.")
    pdf.savefig(fig); plt.close(fig)

def fig_interaction(df: pd.DataFrame, a: str, b: str, ycol: str, title: str, pdf: PdfPages):
    sA, sB = df[a], df[b]
    uA = np.sort(pd.unique(sA.dropna())); uB = np.sort(pd.unique(sB.dropna()))
    if len(uA)<2 or len(uB)<2: return
    Bs = [uB[0], uB[-1]]
    means = {bval: [df.loc[(sA==uA[0])&(sB==bval), ycol].mean(), df.loc[(sA==uA[-1])&(sB==bval), ycol].mean()] for bval in Bs}
    fig = plt.figure(figsize=(6,4))
    xs = [str(uA[0]), str(uA[-1])]
    for bval in Bs:
        plt.plot(xs, means[bval], marker="o", label=f"{b}={bval}")
    plt.xlabel(f"{a} (níveis)"); plt.ylabel(f"Média de {ycol}")
    plt.title(title); plt.legend()
    plt.tight_layout(rect=[0,0.1,1,1])
    page_caption(fig, "Como ler: linhas não paralelas indicam interação entre fatores.")
    pdf.savefig(fig); plt.close(fig)

def parse_args_or_defaults():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=None, help="Comma-separated globs/paths (CSV/XLSX)")
    ap.add_argument("--output_dir", default=None, help="Output dir")
    ap.add_argument("--pdf", default=None, help="PDF filename (inside output_dir)")
    ap.add_argument("--audit", action="store_true", help="Save audit files (NaN report, levels, X,y)")
    ap.add_argument("--composite", choices=["auto","column","metrics"], default=None, help="How to build composite")
    ap.add_argument("--composite-col", default=None, help="Column name to use when --composite=column")
    args = ap.parse_args()

    # Fallback para DEFAULTS quando argumento não for passado
    inputs = args.inputs or DEFAULT_INPUTS
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    pdf_name = args.pdf or DEFAULT_PDF_NAME
    composite = args.composite or DEFAULT_COMPOSITE_MODE
    composite_col = args.composite_col or DEFAULT_COMPOSITE_COL
    audit = args.audit or DEFAULT_AUDIT

    if not inputs:
        raise SystemExit("Defina DEFAULT_INPUTS no topo do arquivo (ou passe --inputs).")
    if not output_dir:
        raise SystemExit("Defina DEFAULT_OUTPUT_DIR no topo do arquivo (ou passe --output_dir).")

    return inputs, output_dir, pdf_name, composite, composite_col, audit

def main():
    inputs, output_dir, pdf_name, composite, composite_col, audit = parse_args_or_defaults()

    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_many(inputs)
    df = build_composite(df, mode=composite, column_name=composite_col if composite=="column" else None)
    df.to_csv(outdir/"combined_with_composite.csv", index=False)

    X, y, valid_mask = feature_matrix(df)
    effects, yhat, resid, dof, r2 = fit_ols(X, y)

    if audit:
        audit_dump(outdir, df, X if effects is not None else None, y if effects is not None else None, valid_mask)

    if effects is None:
        with open(outdir/"README_NO_ROWS.txt","w",encoding="utf-8") as f:
            f.write("Nenhuma linha válida para OLS.\n")
            f.write("1) Abra combined_with_composite.csv e verifique se Z_Composite_std NÃO está todo NaN.\n")
            f.write("2) Veja audit_nan_report.csv para identificar colunas com muitos NaNs.\n")
            f.write("3) Confirme nomes de colunas (model, resample, seconds, timesteps, epochs, units, batch).\n")
            f.write("4) Se sua planilha usa vírgula decimal, esta versão já tenta converter automaticamente.\n")
            f.write("5) Você pode também forçar column + nome da coluna no bloco DEFAULT_*.\n")
        print("ATENÇÃO: Nenhuma linha válida para OLS. Veja arquivos de auditoria.")
        return

    effects.to_csv(outdir/"ols_effects.csv", index=False)

    # Main-effect diffs (2-level convenience)
    def mean_diff(col):
        if col not in df.columns: return np.nan
        s = smart_to_numeric(df[col])
        u = np.sort(s.dropna().unique())
        if len(u)>=2:
            low, high = u[0], u[-1]
            return float(df.loc[s==high, "Z_Composite_std"].mean() - df.loc[s==low, "Z_Composite_std"].mean())
        return np.nan

    rs_col = getcol(df,"resample_ms") or "resample_ms"
    sec_col= getcol(df,"seconds") or "seconds"
    ts_col = getcol(df,"timesteps") or "timesteps"
    ep_col = getcol(df,"epochs") or "epochs"
    u_col  = getcol(df,"units_str") or "units_str"
    b_col  = getcol(df,"batch_size") or "batch_size"
    model_col = getcol(df,"model") or "model"

    rows = []
    for col in [rs_col, sec_col, ts_col, ep_col, b_col]:
        if col in df.columns:
            rows.append({"factor": col, "mean_diff_(hi - lo)": mean_diff(col)})
    if model_col in df.columns:
        means = df.groupby(model_col, dropna=True)["Z_Composite_std"].mean()
        if len(means)>=2: rows.append({"factor": model_col, "mean_diff_(hi - lo)": float(means.max()-means.min())})
    pd.DataFrame(rows).to_csv(outdir/"main_effects_simple.csv", index=False)

   # antes de montar txt:
    r2_str = "nan"
    if (r2 is not None) and np.isfinite(r2):
        r2_str = f"{r2:.3f}"

    # PDF
    with PdfPages(outdir/pdf_name) as pdf:
        fig = plt.figure(figsize=(8.5,11)); plt.axis("off")
        txt = [
            "DOE Analysis Report – RSSI Prediction (AMR) – V3",
            f"Rows used in OLS: n={len(y)}, p={effects.shape[0] + 1}, dof≈{dof}, R²≈{r2_str}",
            "Este relatório traz gráficos com instruções de interpretação (não conclusões).",
        ]
        fig.text(0.1, 0.8, "\n".join(txt), ha="left", va="top", fontsize=12); pdf.savefig(fig); plt.close(fig)

        # Pareto
        fig_pareto(effects, pdf, "Pareto – |coeficientes| (OLS)")

        # Means by level (only if present)
        if rs_col in df.columns: fig_means_by_levels(df, rs_col, "Z_Composite_std", "Médias por nível – Resample (ms)", pdf)
        if sec_col in df.columns: fig_means_by_levels(df, sec_col,"Z_Composite_std","Médias por nível – Seconds to predict (s)", pdf)
        if ts_col in df.columns: fig_means_by_levels(df, ts_col,"Z_Composite_std","Médias por nível – Timesteps", pdf)
        if ep_col in df.columns: fig_means_by_levels(df, ep_col,"Z_Composite_std","Médias por nível – Epochs", pdf)
        if u_col in df.columns:  fig_means_by_levels(df, u_col, "Z_Composite_std","Médias por nível – Units (min-max:step)", pdf)
        if b_col in df.columns:  fig_means_by_levels(df, b_col, "Z_Composite_std","Médias por nível – Batchsize", pdf)

        # Interactions
        if ts_col in df.columns and u_col in df.columns:
            fig_interaction(df, ts_col, u_col, "Z_Composite_std", "Interação – Timesteps × Units", pdf)
        if rs_col in df.columns and sec_col in df.columns:
            fig_interaction(df, rs_col, sec_col, "Z_Composite_std", "Interação – Resample × Seconds", pdf)

    print("OK. Arquivos salvos em:", outdir)

if __name__ == "__main__":
    main()
