# -*- coding: utf-8 -*-
r"""
doe_analysis_rssi_id_v2.py
Análise IN-DISTRIBUTION (ID) a partir de um ÚNICO arquivo (CSV/XLSX).

Melhorias v2:
- Gráficos com ANOTAÇÕES numéricas: barras (contagens/medias), boxplots (mediana e n), Pareto (|coef|).
- Contagem por horizonte com eixo categórico (sem “barras coladas”).
- Textos “O que é / Como interpretar” expandidos.
- Saídas para seção 5:
   * ols_effects_top10.csv  (Top-10 |coef|) + Pareto anotado
   * main_effects_simple.csv (efeitos por fator: binário=alto−baixo; ≥3 níveis=máx−mín) + figura

Exemplo:
  python .\doe_analysis_rssi_id_v2.py `
    --input r"C:\...\Resultado_experimentos_RSSI_Z-Score_BigDS.xlsx" `
    --output_dir r"C:\...\out_id_v2" `
    --pdf "id_report_v2.pdf" `
    --composite-col "Z-Score_Composite" `
    --robust-z 1
"""

import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

# ===================== CONFIG (ajuste se quiser rodar sem CLI) =====================
DEFAULT_INPUT = r"C:\Users\Micro\Documents\RSSI_to_AMR\DoE\Data\Resultado_experimentos_RSSI_Z-Score_BigDS.xlsx"  # ex.: r"C:\...\Resultado_experimentos_RSSI_Z-Score_BigDS.xlsx"
DEFAULT_OUTPUT_DIR = r"C:\Users\Micro\Documents\RSSI_to_AMR\DoE\out_id"
DEFAULT_PDF_NAME = "id_report.pdf"
DEFAULT_COMPOSITE_COL = "Z-Score_Composite"  # coluna com "maior=melhor"
DEFAULT_ROBUST_Z = 0  # 1=usar Z robusto (mediana/MAD), 0=Z clássico (média/DP)

# ===================== MAPEAMENTO DE COLUNAS =====================
CANON_MAP = {
    "model": ["model","modelo","MODEL","MODEL_TYPE","model_type","model type","Modelo"],
    "resample_ms": ["group_msec","group_ms","resample (ms)","resample_ms","resample","Resample (ms)"],
    "seconds": ["seconds_ahead","seconds to predict (s)","seconds_to_predict","sec_to_predict","Seconds to predict (s)","Seconds"],
    "forecast_steps": ["forecast steps","forecast_steps","Forecast steps"],
    "timesteps": ["timesteps","timesteps_orig","window","lookback","Timesteps"],
    "epochs": ["epochs","Epochs"],
    "units_str": ["units_range","units (min-max:step)","units","Units (min-max:step)","Units"],
    "batch_size": ["batch_size","batchsize","batch","Batchsize","Batch"],
    "z_composite": ["z-score_composite","z-score composite","ranking final","Z_Composite_std","Z-Score_Composite","Z-Score_Composite "],
}

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

def smart_to_numeric(s: pd.Series) -> pd.Series:
    s1 = pd.to_numeric(s, errors="coerce")
    s2 = pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    return s2 if s2.notna().sum() > s1.notna().sum() else s1

def parse_units(u: str):
    if not isinstance(u, str): return (np.nan, np.nan, np.nan)
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(\d+)\s*$", u)
    if not m: return (np.nan, np.nan, np.nan)
    lo, hi, st = map(int, m.groups()); return (0.5*(lo+hi), hi-lo, st)

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".csv":
        df = pd.read_csv(path)
        df["__source__"]=str(path); df["__file__"]=path.name
        return normalize_columns(df)
    if path.suffix.lower() in (".xlsx",".xls"):
        try:
            df0 = pd.read_excel(path)
        except Exception:
            df0 = None
        if df0 is not None:
            cols = [str(c) for c in df0.columns]
            many_unnamed = sum(c.startswith("Unnamed:") for c in cols) >= max(3, int(0.3*len(cols)))
            has_groups = any(g in cols for g in ["Hiperparametros","Tempos","Métricas - Resultados"])
            if not (many_unnamed or has_groups):
                df0["__source__"]=str(path); df0["__file__"]=path.name
                return normalize_columns(df0)
        raw = pd.read_excel(path, header=None)
        header = raw.iloc[1].tolist()
        df = raw.iloc[2:].copy(); df.columns = header
        df["__source__"]=str(path); df["__file__"]=path.name
        return normalize_columns(df)
    raise RuntimeError(f"Unsupported file: {path}")

def compute_forecast_steps_derived(df: pd.DataFrame) -> pd.DataFrame:
    rs_col = getcol(df,"resample_ms") or "resample_ms"
    sc_col = getcol(df,"seconds") or "seconds"
    rs = smart_to_numeric(df[rs_col]) if rs_col in df.columns else pd.Series(np.nan,index=df.index)
    sc = smart_to_numeric(df[sc_col]) if sc_col in df.columns else pd.Series(np.nan,index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        der = np.rint(sc * 1000.0 / rs)
    df["_forecast_steps_derived_"] = pd.Series(der, index=df.index).astype("Int64")
    fs_col = getcol(df,"forecast_steps")
    if fs_col and fs_col in df.columns:
        orig = smart_to_numeric(df[fs_col]).astype("Int64")
        df.attrs["forecast_mismatch_count"] = int(orig.ne(df["_forecast_steps_derived_"]).sum())
    else:
        df.attrs["forecast_mismatch_count"] = 0
    return df

def build_composite(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    sec_col = getcol(df, "seconds")
    if sec_col is None:
        fs = getcol(df,"forecast_steps"); rs = getcol(df,"resample_ms")
        if fs and rs and fs in df.columns and rs in df.columns:
            df["_seconds_auto_"] = smart_to_numeric(df[fs]) * (smart_to_numeric(df[rs])/1000.0)
            sec_col = "_seconds_auto_"
        else:
            df["_seconds_auto_"] = np.nan; sec_col = "_seconds_auto_"
    df["_seconds_"] = smart_to_numeric(df[sec_col])
    if column_name not in df.columns:
        col = getcol(df,"z_composite")
        if not col or col not in df.columns:
            raise RuntimeError(f"Coluna '{column_name}' não encontrada e não foi possível inferir.")
        column_name = col
    df["_composite_base_"] = smart_to_numeric(df[column_name])  # maior=melhor
    return df

def z_by_seconds(series: pd.Series, seconds: pd.Series, robust=True) -> pd.Series:
    def z_std(v):  mu, sd = np.nanmean(v), np.nanstd(v, ddof=0); return (v-mu)/(sd if sd>0 else 1.0)
    def z_rob(v):  med = np.nanmedian(v); mad = np.nanmedian(np.abs(v-med)); sc = 1.4826*mad if mad>0 else 1.0; return (v-med)/sc
    fn = z_rob if robust else z_std
    return series.groupby(seconds).transform(lambda v: fn(np.asarray(v, dtype=float)))

def make_key(df: pd.DataFrame):
    m = getcol(df,"model") or "model"
    rs = getcol(df,"resample_ms") or "resample_ms"
    sc = getcol(df,"seconds") or "seconds"
    ts = getcol(df,"timesteps") or "timesteps"
    ep = getcol(df,"epochs") or "epochs"
    un = getcol(df,"units_str") or "units_str"
    bs = getcol(df,"batch_size") or "batch_size"
    cols = [c for c in [m,rs,sc,ts,ep,un,bs] if c in df.columns]
    return df[cols].astype(str).agg("|".join, axis=1)

# ---------- helpers de plot ----------
def _annotate_bars(ax, fmt="{:.0f}", inside=True, ypos="auto"):
    for p in ax.patches:
        h = p.get_height()
        if not np.isfinite(h): continue
        x = p.get_x() + p.get_width()/2.0
        y = (p.get_y() + h*0.5) if inside and h>0 else (p.get_y() + h + (0.01*ax.get_ylim()[1]) )
        if ypos!="auto": y = ypos
        ax.text(x, y, fmt.format(h), ha="center", va="center" if inside and h>0 else "bottom", fontsize=9)

def page_title_explain(pdf, title, what_is, how_to_read=None,
                       width_chars=110, line_height=0.036, fontsize=11):
    """Página de texto com quebra de linha estável (A4 horizontal)."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    y = 0.92
    ax.text(0.06, y, title, fontsize=18, fontweight="bold", ha="left", va="top")
    y -= 0.06

    def add_block(prefix, text, ycur):
        if not text:
            return ycur
        # quebra manual para caber no quadro
        wrapped = textwrap.fill(str(text), width=width_chars,
                                break_long_words=False, break_on_hyphens=False)
        ax.text(0.06, ycur, prefix + wrapped, fontsize=fontsize, ha="left", va="top")
        nlines = wrapped.count("\n") + 1
        return ycur - (nlines * line_height) - 0.02

    y = add_block("O que é: ", what_is, y)
    if how_to_read:
        y = add_block("Como interpretar: ", how_to_read, y)

    pdf.savefig(fig); plt.close(fig)


# ---------- Figuras ----------
def fig_counts_by_seconds(pdf, df):
    cnt = df.groupby("_seconds_").size().rename("n").reset_index()
    cnt = cnt.sort_values("_seconds_")
    # eixo categórico (sem colar barras)
    labels = [f"{int(s)} s" if float(s).is_integer() else f"{s:.1f} s" for s in cnt["_seconds_"].values]
    pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2,3.4))
    bars = ax.bar(pos, cnt["n"].values, width=0.6)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    ax.set_xlabel("Horizonte (s)"); ax.set_ylabel("n"); ax.set_title("Contagem por horizonte")
    _annotate_bars(ax, fmt="{:.0f}", inside=True)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_box_by_seconds(pdf, z, seconds):
    secs = sorted(pd.unique(seconds.dropna()))
    data = [z[seconds==s].values for s in secs]
    fig, ax = plt.subplots(figsize=(6.8,4.2))
    bp = ax.boxplot(data, tick_labels=[f"{int(s)} s" for s in secs])
    ax.axhline(0, color="k", lw=1, alpha=0.4)  # linha de referência no 0
    ax.set_ylabel("Z por horizonte")
    ax.set_title("Distribuição do desempenho por horizonte (mediana=0 por construção)")
    # anota IQR e n (em vez da mediana)
    for i, s in enumerate(secs, start=1):
        vals = z[seconds==s].dropna().values
        if len(vals)==0: continue
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        iqr = q3 - q1
        ax.text(i, q3, f"IQR={iqr:.2f}\n(n={len(vals)})", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_mean_ci_by_factor(pdf, df, zcol, factor, title):
    if factor not in df.columns: return
    s = df[[factor, zcol]].copy()
    s = s[s[zcol].notna()]
    if s.empty: return
    grp = s.groupby(factor, dropna=True)[zcol].agg(['mean','std','count']).reset_index()
    grp = grp[grp['count']>0]
    if grp.empty: return
    grp['se'] = grp['std'] / np.sqrt(grp['count'].clip(lower=1))
    grp['ci95'] = 1.96*grp['se']
    grp['__label__'] = grp[factor].astype(str)
    grp = grp.sort_values('mean', ascending=False).reset_index(drop=True)
    x = np.arange(len(grp)); y = grp['mean'].values; e = grp['ci95'].values; labels = grp['__label__'].tolist()
    fig, ax = plt.subplots(figsize=(max(6,0.6*len(labels)),4.2))
    ax.bar(x, y, yerr=e, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("média Z"); ax.set_title(title)
    ax.axhline(0, color="k", lw=1, alpha=0.4)  # baseline Z=0

    # anota média e n
    for xi, yi, ni in zip(x, y, grp['count'].values):
        ax.text(xi, yi, f"{yi:.2f}\n(n={int(ni)})", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_heatmap_two(pdf, df, zcol, f1, f2, title):
    piv = df.pivot_table(index=f1, columns=f2, values=zcol, aggfunc='mean')
    fig = plt.figure(figsize=(max(5,0.5*len(piv.columns)), max(4,0.5*len(piv.index))))
    im = plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
    plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns], rotation=45, ha='right')
    plt.title(title); plt.tight_layout(); PdfPages
    plt.savefig
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return piv

# ---------- OLS ----------
def prepare_Xy(df):
    y = df["Z_by_seconds"].values.astype(float)
    mcol = getcol(df,"model") or "model"
    if mcol not in df.columns: df[mcol]="MODEL"
    M = pd.get_dummies(df[mcol].astype(str), prefix="model", drop_first=False)
    if "model_LSTM" not in M.columns and "model_GRU" in M.columns:
        # garante coluna explícita
        pass
    def num(name):
        c = getcol(df,name) or name
        return smart_to_numeric(df[c]) if c in df.columns else pd.Series(np.nan,index=df.index)
    rs = num("resample_ms"); sec = num("seconds"); ts=num("timesteps"); ep=num("epochs")
    u_col = getcol(df,"units_str")
    if u_col and u_col in df.columns:
        parsed = df[u_col].astype(str).apply(parse_units); uavg = parsed.apply(lambda t: t[0])
    else:
        uavg = pd.Series(np.nan,index=df.index)
    b_col = getcol(df, "batch_size")
    batch = smart_to_numeric(df[b_col]) if (b_col and b_col in df.columns) else pd.Series(np.nan, index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        blog2 = np.where(batch>0, np.log2(batch), np.nan)
    blog2 = pd.Series(blog2, index=df.index)
    if rs.notna().any() and rs.min()!=rs.max():
        rscaled = (2*(rs-rs.min())/(rs.max()-rs.min()))-1.0
    else:
        rscaled = pd.Series(0.0,index=df.index)
    X = pd.concat([pd.DataFrame({
        "Intercept":1.0,"resample_scaled":rscaled,"seconds":sec,"timesteps":ts,"epochs":ep,
        "units_avg":uavg,"batch_log2":blog2
    }), M], axis=1).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    for a,b in [("timesteps","units_avg"),("resample_scaled","seconds"),("batch_log2","model_GRU")]:
        if a in X.columns and b in X.columns: X[f"{a}:{b}"] = X[a]*X[b]
    return X.values.astype(float), y.astype(float), X.columns.tolist()

def fit_ols_simple(X, y):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xc, yc = X[mask], y[mask]
    if len(yc) < 8 or np.nanstd(yc) == 0:
        beta = np.zeros(X.shape[1]); r2 = np.nan
        return beta.astype(float), float(r2), mask
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    r = yc - (Xc @ beta)
    denom = np.sum((yc - yc.mean())**2)
    r2 = 1 - (r @ r) / denom if denom>0 else np.nan
    return beta.astype(float), float(r2), mask

# def fig_pareto_coefs(pdf, coefs, terms, title):
#     abscoef = np.abs(coefs); order = np.argsort(-abscoef)[:15]
#     t = np.array(terms)[order]; v = abscoef[order]
#     fig, ax = plt.subplots(figsize=(8.8,5))
#     ax.bar(t, v)
#     ax.set_xticklabels(t, rotation=60, ha="right")
#     ax.set_ylabel("|coef| (Z_by_seconds)"); ax.set_title(title)
#     for xi, yi in zip(range(len(v)), v):
#         ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
#     plt.subplots_adjust(bottom=0.3)
#     plt.tight_layout();
#     pdf.savefig(fig);
#     plt.close(fig)
def fig_pareto_coefs(pdf, coefs, terms, title):
    abscoef = np.abs(coefs)
    order = np.argsort(-abscoef)[:15]
    t = np.array(terms)[order]; v = abscoef[order]; s = coefs[order]
    colors = np.where(s >= 0, "#2ca02c", "#d62728")  # verde/ vermelho

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(t, v, color=colors)
    ax.set_xticklabels(t, rotation=60, ha="right")
    ax.set_ylabel("|coef| (Z_by_seconds)"); ax.set_title(title)

    for xi, (vi, si) in enumerate(zip(v, s)):
        ax.text(xi, vi, f"{vi:.2f}\n({ ' +' if si>=0 else ' -'} )", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

# ---------- Main effects simples (para seção 5) ----------
def main_effects_simple(df, zcol, factor):
    """Para fator binário: alto−baixo (ordenado). Para ≥3 níveis: max(mean)−min(mean)."""
    if factor not in df.columns: return None
    s = df[[factor, zcol]].dropna()
    if s.empty: return None
    means = s.groupby(factor)[zcol].mean().sort_index()
    levels = means.index.tolist()
    if len(levels)==1: return None
    try:
        # se níveis numéricos: alto - baixo
        lev_num = pd.to_numeric(pd.Series(levels), errors="coerce")
    except Exception:
        lev_num = pd.Series([np.nan]*len(levels))
    if lev_num.notna().all() and len(levels)==2:
        diff = means.iloc[1] - means.iloc[0]  # hi - lo
        return {"factor": factor, "type": "binary_hi_lo", "levels": str(levels), "effect": float(diff)}
    # caso geral: range
    diff = means.max() - means.min()
    return {"factor": factor, "type": "range_max_min", "levels": str(levels), "effect": float(diff)}

# ---------- ECDF e LOESS ----------

def _ecdf_vals(a: np.ndarray):
    """Retorna (xs, ys) da ECDF para um vetor 1D (dropna)."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(a)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys

def fig_ecdf_by_model(pdf, df, zcol, title, by_seconds=False):
    """ECDF de Z_by_seconds por modelo; se by_seconds=True, plota uma figura por horizonte."""
    mcol = getcol(df, "model") or "model"
    if mcol not in df.columns:
        return
    models = sorted(df[mcol].astype(str).dropna().unique())

    if not by_seconds:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for m in models:
            xs, ys = _ecdf_vals(df.loc[df[mcol].astype(str) == m, zcol].values.astype(float))
            if xs.size:
                ax.step(xs, ys, where="post", label=str(m))
        ax.set_xlabel(zcol); ax.set_ylabel("ECDF"); ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
        return

    # por horizonte
    secs = sorted(df["_seconds_"].dropna().unique())
    for s in secs:
        sub = df[df["_seconds_"] == s]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for m in models:
            xs, ys = _ecdf_vals(sub.loc[sub[mcol].astype(str) == m, zcol].values.astype(float))
            if xs.size:
                ax.step(xs, ys, where="post", label=str(m))
        ax.set_xlabel(zcol); ax.set_ylabel("ECDF")
        ax.set_title(f"{title} — {int(s)} s")
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def _loess_1d(x, y, frac=0.4):
    """
    LOESS (local linear, kernel tricúbico) minimalista.
    Usa apenas os pontos (x,y) fornecidos — idealmente, já agregados por nível.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 3:
        # poucos pontos: devolve y ordenado
        ord_idx = np.argsort(x)
        return x[ord_idx], y[ord_idx]

    ord_idx = np.argsort(x); x = x[ord_idx]; y = y[ord_idx]
    n = x.size
    r = max(2, int(np.ceil(frac * n)))  # vizinhos-alvo
    yhat = np.zeros_like(y)

    for i in range(n):
        # distância e banda (h = dist do r-ésimo vizinho)
        d = np.abs(x - x[i])
        h = np.partition(d, r-1)[r-1]
        if h <= 0:  # pontos repetidos
            yhat[i] = y[i]
            continue
        w = (1 - (d / h) ** 3) ** 3
        w[d > h] = 0.0
        # regressão local linear em torno de x[i]
        X = np.column_stack([np.ones(n), x - x[i]])
        W = w[:, None]
        # solução ponderada via mínimos quadrados
        beta, *_ = np.linalg.lstsq((W * X), (W[:, 0] * y), rcond=None)
        yhat[i] = beta[0]  # valor no ponto x[i] (por construção)

    return x, yhat

def fig_loess_by_factor(pdf, df, zcol, xcol, xlabel, title, frac=0.5):
    """
    Curva LOESS do Z_by_seconds em função de um hiperparâmetro numérico.
    Para estabilidade, agregamos primeiro por nível de x (média de Z).
    """
    if xcol not in df.columns:
        return
    tmp = df[[xcol, zcol]].copy()
    tmp = tmp.dropna()
    if tmp.empty:
        return

    # agrega por nível (evita LOESS com valores duplicados)
    grp = tmp.groupby(xcol)[zcol].mean().reset_index()
    x = grp[xcol].astype(float).values
    y = grp[zcol].astype(float).values
    if x.size < 3:
        # poucos níveis — plota linha conectando médias
        ord_idx = np.argsort(x); x_, y_ = x[ord_idx], y[ord_idx]
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.plot(x_, y_, marker="o")
        ax.set_xlabel(xlabel); ax.set_ylabel("média Z"); ax.set_title(title)
        # marca máximo
        imax = np.argmax(y_); ax.scatter([x_[imax]], [y_[imax]], s=40)
        ax.text(x_[imax], y_[imax], f" max={y_[imax]:.2f}", va="bottom", ha="left", fontsize=9)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
        return

    xs, ys = _loess_1d(x, y, frac=frac)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.scatter(x, y, s=18, alpha=0.7, label="média por nível")
    ax.plot(xs, ys, lw=2, label="LOESS")
    ax.set_xlabel(xlabel); ax.set_ylabel("média Z"); ax.set_title(title)
    # marca máximo da curva LOESS
    imax = np.argmax(ys); ax.scatter([xs[imax]], [ys[imax]], s=40)
    ax.text(xs[imax], ys[imax], f" pico≈{ys[imax]:.2f}", va="bottom", ha="left", fontsize=9)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_loess_by_factor_per_horizon(pdf, df, zcol, xcol, xlabel, base_title, frac=0.5):
    """Mesma ideia da LOESS, mas uma figura por horizon (seconds)."""
    secs = sorted(df["_seconds_"].dropna().unique())
    for s in secs:
        sub = df[df["_seconds_"] == s].copy()
        title = f"{base_title} — {int(s)} s"
        fig_loess_by_factor(pdf, sub, zcol, xcol, xlabel, title, frac=frac)


# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Caminho para 1 arquivo (CSV/XLSX).")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Pasta de saída.")
    ap.add_argument("--pdf", default=DEFAULT_PDF_NAME, help="Nome do PDF de saída.")
    ap.add_argument("--composite-col", default=DEFAULT_COMPOSITE_COL, help="Coluna do composto (maior=melhor).")
    ap.add_argument("--robust-z", type=int, default=DEFAULT_ROBUST_Z, help="1=Z robusto (mediana/MAD), 0=Z clássico.")
    args = ap.parse_args()

    if not args.input:
        raise SystemExit("Forneça --input ou edite DEFAULT_INPUT no topo do arquivo.")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / args.pdf

    # --------- Carrega/prepara
    df = load_any(Path(args.input))
    df = build_composite(df, column_name=args.composite_col)
    df = compute_forecast_steps_derived(df)
    df["_key_"] = make_key(df)
    df["Z_by_seconds"] = z_by_seconds(df["_composite_base_"], df["_seconds_"], robust=bool(args.robust_z))

    # ---- colunas numéricas auxiliares para LOESS ----
    # units_avg (média do intervalo de units_str), se não existir
    u_col = getcol(df, "units_str") or "units_str"
    if u_col in df.columns and "_units_avg_" not in df.columns:
        parsed = df[u_col].astype(str).apply(parse_units)
        df["_units_avg_"] = parsed.apply(lambda t: t[0])

    # batch_log2
    b_col = getcol(df, "batch_size") or "batch_size"
    if b_col in df.columns:
        bnum = smart_to_numeric(df[b_col])
        with np.errstate(divide="ignore", invalid="ignore"):
            df["_batch_log2_"] = np.where(bnum > 0, np.log2(bnum), np.nan)

    # salva dataset “limpo”
    cols_keep = ["_key_","_seconds_","Z_by_seconds","_composite_base_","__file__",
                 getcol(df,"model") or "model",
                 getcol(df,"resample_ms") or "resample_ms",
                 getcol(df,"seconds") or "seconds",
                 getcol(df,"timesteps") or "timesteps",
                 getcol(df,"epochs") or "epochs",
                 getcol(df,"units_str") or "units_str",
                 getcol(df,"batch_size") or "batch_size",
                 "_forecast_steps_derived_"]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df[cols_keep].to_csv(out/"id_dataset_clean.csv", index=False)

    pdf = PdfPages(pdf_path)

    # CAPA
    page_title_explain(
        pdf,
        "Relatório ID (in-distribution) — visão geral",
        "Padronização por horizonte (Z_by_seconds), estatística descritiva, efeitos principais e interações; "
        "saídas auxiliares para a Seção 5 (tabelas/figuras).",
        "Z_by_seconds alto indica melhor desempenho relativo dentro de cada horizonte. Compare fatores e interações para localizar 'sweet spots'."
    )
    page_title_explain(
        pdf,
        "Nota metodológica: 'Forecast steps' é fator derivado",
        "forecast_steps = round(seconds × 1000 / resample_ms). Por ser determinístico, não entra como fator independente.",
        "Usamos Z robusto (mediana/MAD) por padrão para reduzir a influência de outliers; pode alternar com --robust-z 0."
    )

    # Contagem por horizonte
    page_title_explain(
        pdf, "Cobertura por horizonte (com contagens anotadas)",
        "Mostra a distribuição de linhas por horizonte (seconds).",
        "Cobertura desequilibrada pode enviesar médias e OLS; avalie necessidade de ponderação/estratificação."
    )
    fig_counts_by_seconds(pdf, df)

    # Boxplot por horizonte com mediana e n
    page_title_explain(
        pdf, "Distribuição do desempenho por horizonte (boxplots anotados)",
        "Boxplots de Z_by_seconds por seconds (robusto: centrado na mediana do horizonte)",
        "Medianas altas e caixas compactas indicam horizontes mais favoráveis/estáveis; caudas longas sinalizam variância alta. Por construção, a mediana=0 em cada caixa; compare dispersão (IQR), caudas e outliers. Para ver medianas deslocadas, rode com --robust-z 0 (média/DP)."
    )
    fig_box_by_seconds(pdf, df["Z_by_seconds"], df["_seconds_"])

    # Médias por fator (barras com IC95 e anotações)
    factors = [
        (getcol(df,"model") or "model", "Modelo"),
        (getcol(df,"resample_ms") or "resample_ms", "Resample (ms)"),
        (getcol(df,"seconds") or "seconds", "Seconds (horizonte)"),
        (getcol(df,"timesteps") or "timesteps", "Timesteps"),
        (getcol(df,"epochs") or "epochs", "Epochs"),
        (getcol(df,"units_str") or "units_str", "Units (min-max:step)"),
        (getcol(df,"batch_size") or "batch_size", "Batch size"),
    ]
    for fac, lab in factors:
        if fac in df.columns:
            page_title_explain(
                pdf, f"Médias por nível — {lab}",
                f"Média e IC95% de Z_by_seconds por níveis de {lab}.",
                "Barras mais altas sugerem níveis mais vantajosos; ICs sobrepostos indicam diferenças pouco conclusivas. Barras acima de 0 indicam média acima da mediana do horizonte; abaixo de 0 = abaixo da mediana. ICs sobrepostos sugerem diferenças menos conclusivas."
            )
            fig_mean_ci_by_factor(pdf, df, "Z_by_seconds", fac, f"{lab}: média±IC95% (anotado)")

    # Heatmaps (interações)
    if (getcol(df,"resample_ms") or "resample_ms") in df.columns and (getcol(df,"seconds") or "seconds") in df.columns:
        page_title_explain(
            pdf, "Interação — Resample × Seconds",
            "Heatmap da média de Z_by_seconds para cada par (resample, seconds).",
            "Regiões 'quentes' indicam combinações-alvo de taxa de amostragem e horizonte."
        )
        fig_heatmap_two(pdf, df, "Z_by_seconds", getcol(df,"resample_ms") or "resample_ms",
                        getcol(df,"seconds") or "seconds", "Média(Z) por Resample×Seconds")

    u_col = getcol(df,"units_str") or "units_str"
    if u_col in df.columns and (getcol(df,"timesteps") or "timesteps") in df.columns:
        parsed = df[u_col].astype(str).apply(parse_units); df["_units_avg_"] = parsed.apply(lambda t: t[0])
        page_title_explain(
            pdf, "Interação — Timesteps × Units_avg",
            "Heatmap da média de Z_by_seconds em função de (timesteps, média de units).",
            "Procure cristas/vales que indiquem 'sweet spots' de janela vs capacidade do modelo."
        )
        fig_heatmap_two(pdf, df, "Z_by_seconds", getcol(df,"timesteps") or "timesteps",
                        "_units_avg_", "Média(Z) por Timesteps×Units_avg")

    if (getcol(df,"batch_size") or "batch_size") in df.columns and (getcol(df,"model") or "model") in df.columns:
        page_title_explain(
            pdf, "Interação — Batch × Modelo",
            "Heatmap da média de Z_by_seconds para (batch, modelo).",
            "Tendências diagonais podem indicar maior benefício de batch em um dos modelos."
        )
        fig_heatmap_two(pdf, df, "Z_by_seconds", getcol(df,"batch_size") or "batch_size",
                        getcol(df,"model") or "model", "Média(Z) por Batch×Modelo")

    # ================== ECDF por modelo ==================
    page_title_explain(
        pdf, "ECDF do desempenho por modelo (ID)",
        "A ECDF (função de distribuição cumulativa empírica) compara a distribuição de Z_by_seconds entre modelos.",
        "Curva mais à direita indica maior probabilidade de valores altos de Z (melhor desempenho). Diferenças grandes sugerem vantagem consistente."
    )
    fig_ecdf_by_model(pdf, df, "Z_by_seconds", "ECDF — Z_by_seconds (global)", by_seconds=False)

    page_title_explain(
        pdf, "ECDF por horizonte e por modelo (ID)",
        "Mesma ideia da ECDF, estratificada por horizonte (seconds).",
        "Permite verificar se a vantagem de um modelo é consistente entre horizontes ou específica de certas janelas temporais."
    )
    fig_ecdf_by_model(pdf, df, "Z_by_seconds", "ECDF — Z_by_seconds por horizonte", by_seconds=True)

    # ================== LOESS por hiperparâmetro ==================
    # Defina a lista de eixos numéricos que existem na sua planilha
    loess_axes = []
    if (getcol(df, "resample_ms") or "resample_ms") in df.columns:
        loess_axes.append((getcol(df, "resample_ms") or "resample_ms", "Resample (ms)"))
    if (getcol(df, "seconds") or "seconds") in df.columns:
        loess_axes.append((getcol(df, "seconds") or "seconds", "Seconds (s)"))
    if (getcol(df, "timesteps") or "timesteps") in df.columns:
        loess_axes.append((getcol(df, "timesteps") or "timesteps", "Timesteps"))
    if (getcol(df, "epochs") or "epochs") in df.columns:
        loess_axes.append((getcol(df, "epochs") or "epochs", "Epochs"))
    if "_units_avg_" in df.columns:
        loess_axes.append(("_units_avg_", "Units (média)"))
    if "_batch_log2_" in df.columns:
        loess_axes.append(("_batch_log2_", "Batch (log2)"))

    page_title_explain(
        pdf, "Curvas LOESS (global)",
        "Curvas LOESS (localmente ponderadas) da média de Z_by_seconds vs hiperparâmetros numéricos.",
        "Crestas/picos da LOESS sugerem faixas 'ótimas' aproximadas; quedas indicam regimes a evitar."
    )
    for xcol, xlabel in loess_axes:
        fig_loess_by_factor(pdf, df, "Z_by_seconds", xcol, xlabel, f"LOESS — {xlabel} (global)", frac=0.5)

    page_title_explain(
        pdf, "Curvas LOESS por horizonte",
        "Mesma análise, agora separada por seconds (uma figura por horizon).",
        "Ajuda a identificar se a 'faixa ótima' muda com o horizonte de previsão."
    )
    for xcol, xlabel in loess_axes:
        fig_loess_by_factor_per_horizon(pdf, df, "Z_by_seconds", xcol, xlabel, f"LOESS — {xlabel}", frac=0.5)

    # OLS global e por-horizonte (Pareto anotado) + CSV top10
    # page_title_explain(
    #     pdf, "Efeitos principais (OLS) — Global.",
    #     "Regressão de Z_by_seconds nos fatores independentes (sem forecast_steps); figura com |coef| anotado. "
    #     " Mínimos Quadrados Ordinários (OLS) é uma regressão linear que estima como cada hiperparâmetro contribui para a métrica alvo. Aqui, ajustamos"
    #     " Z by_seconds ≈𝛽0 + ∑𝛽𝑖𝑥𝑖 +βij + xixj + ε,"
    #     "onde 𝑥𝑖  são os fatores independentes (modelo, resample, seconds, timesteps, epochs, média de units, batch_log2) e algumas interações"
    #     "(p.ex., timesteps×units, resample×seconds, batch×modelo). A coluna forecast_steps não entra (é derivada). Categóricos entram via dummies (comparação a uma categoria-base)."
    #     "O gráfico mostra um Pareto de ∣𝛽∣(módulo do coeficiente) para ranquear a influência relativa; o R² no título indica a fração de variância explicada."
    #     " "
    #     " "
    #     " "
    #     "\cr ",
    #     "Barras maiores (|coef|) indicam maior influência relativa; o sinal do coeficiente orienta a direção do efeito."
    #     "Barras mais altas (maior ∣𝛽∣) ⇒ fator mais influente na variação de Zby_seconds neste ajuste linear"
    #     "O sinal de 𝛽 informa a direção (positivo melhora, negativo piora em média) e está na tabela CSV de coeficientes; no Pareto mostramos o módulo para facilitar o ranking."
    #     "Coeficientes com IC que cruza 0 (ou módulo muito pequeno) sugerem efeito fraco/instável. "
    #     "R² mais alto ⇒ o modelo linear explica melhor os dados; valores modestos são comuns porque a relação real pode ser não linear e com interações. "
    #     "Observações: (i) parte dos contínuos é reescalada (ex.: batch_log2, resample_scaled), então a comparação de magnitudes entre variáveis de escalas "
    #     "muito diferentes deve ser cautelosa; (ii) resultados por horizonte podem mudar—use também os Paretos “por-horizonte” para recomendações específicas; (iii) "
    #     "forecast_steps é excluído por ser determinístico dado seconds e resample."
    # )

    what_ols = (
        "Regressão linear por Mínimos Quadrados Ordinários (OLS) para explicar o índice Z_by_seconds "
        "a partir dos fatores (modelo, resample, seconds, timesteps, epochs, units_avg, batch_log2) e de algumas "
        "interações (p.ex., timesteps×units_avg, resample×seconds, batch_log2×modelo). "
        "'forecast_steps' fica de fora por ser derivado. O gráfico mostra um Pareto do módulo dos coeficientes (|beta|) "
        "e o R^2 do ajuste aparece no título."
    )

    how_ols = (
        "Barras maiores (|beta|) ⇒ maior influência relativa no Z_by_seconds. "
        "O sinal de beta indica a direção (positivo melhora, negativo piora em média) — veja a tabela CSV de coeficientes. "
        "Coeficientes ~0 ou com IC cruzando 0 sugerem efeito fraco/instável. "
        "R^2 maior ⇒ o ajuste linear explica mais variação; valores moderados são esperados porque há não linearidades e interações."
    )

    page_title_explain(pdf, "Efeitos principais (OLS) — Global", what_ols, how_ols)

    X,y,terms = prepare_Xy(df); b,r2,_ = fit_ols_simple(X,y)
    fig_pareto_coefs(pdf, b, terms, f"Pareto |coef| — Global (R²≈{np.nan if r2 is None else r2:.3f})")
    top10_idx = np.argsort(-np.abs(b))[:10]
    pd.DataFrame({"term":np.array(terms)[top10_idx], "coef":b[top10_idx], "abscoef":np.abs(b[top10_idx])})\
      .to_csv(out/"ols_effects_top10.csv", index=False)

    secs = sorted(df["_seconds_"].dropna().unique())
    for s in secs:
        sub = df[df["_seconds_"]==s].copy()
        if len(sub) < 10: continue
        page_title_explain(
            pdf, f"Efeitos principais (OLS) — {int(s)} s",
            "Ajuste restrito ao horizonte; Pareto com |coef| anotado.",
            "Evidencia fatores mais relevantes para cada janela temporal."
        )
        Xs,ys,ts = prepare_Xy(sub); bs,r2s,_ = fit_ols_simple(Xs,ys)
        fig_pareto_coefs(pdf, bs, ts, f"Pareto |coef| — {int(s)} s (R²≈{np.nan if r2s is None else r2s:.3f})")
        pd.DataFrame({"term":ts,"coef":bs,"abscoef":np.abs(bs)}).sort_values("abscoef",ascending=False)\
          .head(10).to_csv(out/f"ols_effects_top10_{int(s)}s.csv", index=False)

    # Main effects simples (binário: hi-lo; >=3 níveis: range)
    factors_for_effects = [c for c,_ in factors if c in df.columns]
    rows=[]
    for fac in factors_for_effects:
        r = main_effects_simple(df, "Z_by_seconds", fac)
        if r: rows.append(r)
    if rows:
        me = pd.DataFrame(rows).sort_values("effect", ascending=False)
        me.to_csv(out/"main_effects_simple.csv", index=False)

        page_title_explain(
            pdf, "Efeitos principais simples — Diferença entre níveis",
            "Para fatores binários: (alto − baixo). Para ≥3 níveis: (média do melhor nível − média do pior nível).",
            "Valores altos indicam fator com forte influência marginal; útil como 'mapa' de priorização de ajustes."
        )
        fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.35*len(me))))
        ax.barh(me["factor"], me["effect"])
        for yi, val in enumerate(me["effect"]):
            ax.text(val, yi, f"{val:.2f}", va="center", ha="left", fontsize=8)
        ax.set_xlabel("Δ efeito (Z)"); ax.set_title("Main effects simples (ID)")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Top-N por horizonte
    page_title_explain(
        pdf, "Top-N por horizonte (ID)",
        "Lista as melhores configurações por seconds segundo Z_by_seconds (maior=melhor).",
        "Útil para seleção de candidatos em produção no mesmo domínio; combine com Pareto e efeitos simples."
    )
    tops=[]
    for s in secs:
        sub = df[df["_seconds_"]==s].copy().sort_values("Z_by_seconds", ascending=False).head(10)
        tops.append(sub[["_key_","Z_by_seconds"]])
    if tops:
        top_tbl = pd.concat(tops, keys=[f"{int(s)}s" for s in secs])
        top_tbl.to_csv(out/"topN_id_por_horizonte.csv")
        # snippet na página
        fig = plt.figure(figsize=(11, 3.2)); plt.axis("off")
        disp = top_tbl.reset_index(level=0).rename(columns={"level_0":"Horizon"}).head(12).copy()
        disp["_key_"] = disp["_key_"].astype(str).str.slice(0, 70)
        disp["Z_by_seconds"] = disp["Z_by_seconds"].round(3)
        tbl = plt.table(cellText=disp.astype(str).values, colLabels=disp.columns.tolist(),
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1,1.15)
        plt.title("Top-N por horizonte (amostra) — ver CSV completo")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    pdf.close()

    # log
    print("OK. Saídas em:", out.as_posix())
    print("PDF:", pdf_path.as_posix())
    for f in ["id_dataset_clean.csv","ols_effects_top10.csv","main_effects_simple.csv",
              "topN_id_por_horizonte.csv"]:
        p = out/f; print(f"{f}: {p.exists()}")

if __name__ == "__main__":
    main()
