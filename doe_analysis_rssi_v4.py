# -*- coding: utf-8 -*-
r"""
doe_analysis_rssi_v4.py
Relatório DOE cross-domain (2 planilhas espelho: 2DS1112 e 2DS1211).
- Lê 2 arquivos (CSV/XLSX), padroniza colunas, calcula Z composto "pooled" por horizonte (seconds).
- Pareia ensaios 1-a-1 pela chave de hiperparâmetros.
- Gera PDF com: explicação, scatter pareado, Bland–Altman, Jaccard@k, correlação de ranking,
  Pareto de coeficientes OLS por direção + consistência de sinais, Top-N robustos.
Dependências: pandas, numpy, matplotlib, openpyxl (para .xlsx).

Exemplo (Windows/PowerShell):
  python doe_analysis_rssi_v4.py --inputs "C:\...\Resultados_Testes_2DS1112_DoE_full_288.xlsx,C:\...\Resultados_Testes_2DS1211_DoE_full_288.xlsx" ^
        --output_dir "C:\...\out_v4" --composite-col "Z-Score_Composite"
"""
import os, re, textwrap, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------- mapeamento de colunas ----------
CANON_MAP = {
    "model": ["model","modelo","MODEL","MODEL_TYPE","model_type","model type","Modelo","Model"],
    "resample_ms": ["group_msec","group_ms","resample (ms)","resample_ms","resample","Resample (ms)"],
    "seconds": ["seconds_ahead","seconds to predict (s)","seconds_to_predict","sec_to_predict","Seconds to predict (s)"],
    "forecast_steps": ["forecast steps","forecast_steps","Forecast steps"],
    "timesteps": ["timesteps","timesteps_orig","window","lookback","Timesteps"],
    "epochs": ["epochs","Epochs"],
    "units_str": ["units_range","units (min-max:step)","units","Units (min-max:step)","Units"],
    "batch_size": ["batch_size","batchsize","batch","Batchsize"],
    "train_dataset": ["train_dataset","train dataset","Datasets usados","datasets usados"],
    "exp_dataset": ["exp_dataset","exp dataset","test_dataset","test dataset"],
    "z_composite": ["z-score_composite","z-score composite","ranking final","Z_Composite_std","Z-Score_Composite","Z-Score_Composite"],
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

def smart_to_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    n1 = s.notna().sum()
    s2 = pd.to_numeric(series.astype(str).str.replace(".", "", regex=False)
                                  .str.replace(",", ".", regex=False), errors="coerce")
    return s2 if s2.notna().sum() > n1 else s

def parse_units(u: str):
    if not isinstance(u, str): return (np.nan, np.nan, np.nan)
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(\d+)\s*$", u)
    if not m: return (np.nan, np.nan, np.nan)
    lo, hi, st = map(int, m.groups()); return (0.5*(lo+hi), hi-lo, st)

def compute_forecast_steps_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna derivada _forecast_steps_derived_ = round(seconds*1000/resample_ms)
       e audita divergências em relação à coluna original (se existir)."""
    rs_col = getcol(df, "resample_ms") or "resample_ms"
    sc_col = getcol(df, "seconds") or "seconds"
    rs = smart_to_numeric(df[rs_col]) if rs_col in df.columns else pd.Series(np.nan, index=df.index)
    sc = smart_to_numeric(df[sc_col]) if sc_col in df.columns else pd.Series(np.nan, index=df.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        derived = np.rint(sc * 1000.0 / rs)
    df["_forecast_steps_derived_"] = pd.Series(derived, index=df.index).astype("Int64")

    # auditoria (se existir 'forecast_steps' original)
    fs_col = getcol(df, "forecast_steps")
    if fs_col and fs_col in df.columns:
        orig = smart_to_numeric(df[fs_col]).astype("Int64")
        mismatch_mask = orig.ne(df["_forecast_steps_derived_"])
        df.attrs["forecast_mismatch_count"] = int(mismatch_mask.sum())
        # salva linhas divergentes (se houver)
        if mismatch_mask.any():
            df.loc[mismatch_mask].to_csv(
                Path(df["__source__"].iloc[0]).with_suffix("").as_posix() + "_forecast_mismatch.csv",
                index=False
            )
    else:
        df.attrs["forecast_mismatch_count"] = 0
    return df

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".csv":
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]  # <<< ADICIONE ESTA LINHA
        df["__source__"] = str(path);
        df["__file__"] = path.name
        return normalize_columns(df)

    if path.suffix.lower() in (".xlsx",".xls"):
        try:
            df0 = pd.read_excel(path)
            df0.columns = [str(c).strip() for c in df0.columns]
        except Exception:
            df0 = None
        if df0 is not None:
            cols = [str(c) for c in df0.columns]
            has_many_unnamed = sum(c.startswith("Unnamed:") for c in cols) >= max(3, int(0.3*len(cols)))
            has_group_headers = any(g in cols for g in ["Hiperparametros","Tempos","Métricas - Resultados"])
            if not (has_many_unnamed or has_group_headers):
                df0["__source__"]=str(path); df0["__file__"]=path.name
                return normalize_columns(df0)
        raw = pd.read_excel(path, header=None)
        header = [str(x).strip() for x in raw.iloc[1].tolist()]
        df = raw.iloc[2:].copy()  # <-- cria o df a partir da linha 2
        df.columns = header  # <-- aplica o cabeçalho da linha 1
        df.columns = [str(c).strip() for c in df.columns]  # strip final
        df["__source__"] = str(path)
        df["__file__"] = path.name
        return normalize_columns(df)

    raise RuntimeError(f"Unsupported file: {path}")

def build_composite(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # seconds
    sec_col = getcol(df, "seconds")
    if sec_col is None:
        fs = getcol(df,"forecast_steps"); rs = getcol(df,"resample_ms")
        if fs and rs and fs in df.columns and rs in df.columns:
            df["_seconds_auto_"] = smart_to_numeric(df[fs]) * (smart_to_numeric(df[rs])/1000.0)
            sec_col = "_seconds_auto_"
        else:
            df["_seconds_auto_"] = np.nan; sec_col = "_seconds_auto_"
    df["_seconds_"] = smart_to_numeric(df[sec_col])

    # direção (train→exp)
    tr = getcol(df,"train_dataset"); ex = getcol(df,"exp_dataset")
    if tr and ex and tr in df.columns and ex in df.columns:
        A = df[tr].astype(str).str.replace(r".*/","",regex=True)
        B = df[ex].astype(str).str.replace(r".*/","",regex=True)
        df["_dataset_id_"] = (A+"→"+B).str.replace(r"\.parquet$|\.csv$|\.xlsx$","",regex=True)
    else:
        f = df["__file__"].astype(str)
        df["_dataset_id_"] = np.where(f.str.contains("AB"),"A→B",
                               np.where(f.str.contains("BA"),"B→A","DATASET"))

    if column_name not in df.columns:
        # tenta o mapeamento canônico
        col = getcol(df,"z_composite")
        if not col or col not in df.columns:
            raise RuntimeError(f"Coluna '{column_name}' não encontrada e não foi possível inferir o Z composto.")
        column_name = col

    base = smart_to_numeric(df[column_name])  # assume MAIOR = melhor (como combinamos)
    df["_composite_base_"] = base

    # # Z "pooled" por seconds (junta os dois datasets para comparabilidade cross-domain)
    # df["Z_pool_std"] = base.groupby(df["_seconds_"]).transform(
    #     lambda v: (v - v.mean())/(v.std(ddof=0) if v.std(ddof=0)!=0 else 1.0)
    # )
    return df


def make_key(df: pd.DataFrame):
    m = getcol(df,"model") or "model"
    rs = getcol(df,"resample_ms") or "resample_ms"
    sc = getcol(df,"seconds") or "seconds"
    # fs = getcol(df,"forecast_steps") or "forecast_steps"
    ts = getcol(df,"timesteps") or "timesteps"
    ep = getcol(df,"epochs") or "epochs"
    un = getcol(df,"units_str") or "units_str"
    bs = getcol(df,"batch_size") or "batch_size"
    cols = [c for c in [m,rs,sc,ts,ep,un,bs] if c in df.columns]
    return df[cols].astype(str).agg("|".join, axis=1)

# ---------- figuras ----------
def page_title_explain(pdf, title, what_is, how_to_read=None):
    fig = plt.figure(figsize=(11.69, 8.27)); plt.axis("off")
    y = 0.9
    plt.text(0.06,y,title,fontsize=18,fontweight="bold",ha="left",va="top")
    y -= 0.08; plt.text(0.06,y,"O que é: "+what_is,fontsize=11,ha="left",va="top",wrap=True)
    if how_to_read:
        y -= 0.14; plt.text(0.06,y,"Como interpretar: "+how_to_read,fontsize=11,ha="left",va="top",wrap=True)
    pdf.savefig(fig); plt.close(fig)

def fig_scatter_pairs(pdf, pairs_sec, seconds_val):
    fig = plt.figure(figsize=(7,6))
    x = pairs_sec["Z_AB"].values; y = pairs_sec["Z_BA"].values
    plt.scatter(x,y,alpha=0.7)
    if len(x) and len(y):
        lo = float(np.nanmin([x.min(),y.min()])); hi = float(np.nanmax([x.max(),y.max()]))
    else:
        lo,hi = -3,3
    plt.plot([lo,hi],[lo,hi],"--"); plt.xlabel("Z (A→B)"); plt.ylabel("Z (B→A)")
    plt.title(f"Scatter pareado por configuração – {seconds_val:.0f} s")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_bland_altman(pdf, pairs_sec, seconds_val):
    a = pairs_sec["Z_AB"].values; b = pairs_sec["Z_BA"].values
    mean = (a+b)/2.0; diff = a-b
    mu = float(np.nanmean(diff)) if len(diff) else np.nan
    sd = float(np.nanstd(diff, ddof=1)) if len(diff)>1 else np.nan
    loa1,loa2 = (mu-1.96*sd, mu+1.96*sd) if np.isfinite(sd) else (np.nan,np.nan)
    fig = plt.figure(figsize=(7,6))
    plt.scatter(mean,diff,alpha=0.7)
    if np.isfinite(mu): plt.axhline(mu)
    if np.isfinite(loa1): plt.axhline(loa1,ls="--")
    if np.isfinite(loa2): plt.axhline(loa2,ls="--")
    plt.xlabel("Média (Z)"); plt.ylabel("Diferença (ZA→B − ZB→A)")
    plt.title(f"Bland–Altman – {seconds_val:.0f} s")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_jaccard_topk(pdf, pairs_all, seconds_list, out_csv, ks=(5,10,20)):
    data=[]
    for s in seconds_list:
        sub = pairs_all[pairs_all["_seconds_"]==s].copy()
        idx1 = sub.sort_values("Z_AB",ascending=False).index.tolist()
        idx2 = sub.sort_values("Z_BA",ascending=False).index.tolist()
        for k in ks:
            set1, set2 = set(idx1[:k]), set(idx2[:k])
            denom = len(set1|set2); j = len(set1&set2)/denom if denom else np.nan
            data.append((s,k,j))
    if not data: return
    dfj = pd.DataFrame(data,columns=["seconds","k","jaccard"])
    fig = plt.figure(figsize=(8,5))
    plt.bar(np.arange(len(dfj)), dfj["jaccard"].values)
    plt.xticks(np.arange(len(dfj)), [f"{int(s)}s@k={k}" for s,k in zip(dfj["seconds"], dfj["k"])], rotation=40, ha="right")
    plt.ylabel("Jaccard"); plt.title("Sobreposição de Top-k entre direções (maior é melhor)")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
    dfj.to_csv(out_csv,index=False)

def fig_spearman_by_seconds(pdf, pairs_all, out_csv):
    rows=[]
    for s in sorted(pairs_all["_seconds_"].dropna().unique()):
        sub = pairs_all[pairs_all["_seconds_"]==s].copy()
        if len(sub)>1:
            r1 = sub["Z_AB"].rank(ascending=False).values
            r2 = sub["Z_BA"].rank(ascending=False).values
            corr = float(np.corrcoef(r1,r2)[0,1])
        else:
            corr = np.nan
        rows.append((s,corr))
    dfr = pd.DataFrame(rows,columns=["seconds","spearman_like"])
    fig = plt.figure(figsize=(6,4))
    plt.plot(dfr["seconds"], dfr["spearman_like"], marker="o")
    plt.xlabel("Horizon (s)"); plt.ylabel("Correlação de ranking (ρ)")
    plt.title("Concordância de ranking entre direções")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
    dfr.to_csv(out_csv,index=False)

# ---------- OLS ----------
def prepare_Xy(df):
    """
    Monta X,y para OLS:
      - y = Z_pool_std
      - contínuos centrados e escalados (z-score)
      - categóricos: dummies (drop_first=True)
      - interações: numéricas + batch_log2_z × cada dummy de modelo
    """
    # alvo
    y = smart_to_numeric(df["Z_pool_std"]).astype(float).values

    # modelo (dummy)
    mcol = getcol(df, "model") or "model"
    if mcol not in df.columns:
        df[mcol] = "MODEL"
    M = pd.get_dummies(
        df[mcol].astype(str).str.strip().fillna("nan"),
        prefix="model", drop_first=True
    )

    # helper: pegar coluna numérica (ou NaN)
    def num(name):
        c = getcol(df, name) or name
        return smart_to_numeric(df[c]) if c in df.columns else pd.Series(np.nan, index=df.index)

    # contínuos brutos
    rs  = num("resample_ms")
    sec = num("seconds")
    ts  = num("timesteps")
    ep  = num("epochs")

    u_col = getcol(df, "units_str")
    if u_col and u_col in df.columns:
        parsed = df[u_col].astype(str).apply(parse_units)
        uavg = parsed.apply(lambda t: t[0])  # média do intervalo
    else:
        uavg = pd.Series(np.nan, index=df.index)

    b_col = getcol(df, "batch_size")
    if b_col and b_col in df.columns:
        batch = smart_to_numeric(df[b_col])
        with np.errstate(divide="ignore", invalid="ignore"):
            blog2 = np.where(batch > 0, np.log2(batch), np.nan)
        blog2 = pd.Series(blog2, index=df.index)
    else:
        blog2 = pd.Series(np.nan, index=df.index)

    # z-score (centrado e escalado)
    def _z(s):
        s = pd.to_numeric(s, errors="coerce")
        mu, sd = np.nanmean(s), np.nanstd(s)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=df.index)
        return (s - mu) / sd

    rs_z  = _z(rs)
    sec_z = _z(sec)
    ts_z  = _z(ts)
    ep_z  = _z(ep)
    ua_z  = _z(uavg)
    bl_z  = _z(blog2)

    # matriz base + dummies
    X = pd.concat([
        pd.DataFrame({
            "Intercept": 1.0,
            "resample_z": rs_z,
            "seconds_z": sec_z,
            "timesteps_z": ts_z,
            "epochs_z": ep_z,
            "units_avg_z": ua_z,
            "batch_log2_z": bl_z
        }, index=df.index),
        M
    ], axis=1)

    # interações numéricas
    if {"timesteps_z","units_avg_z"}.issubset(X.columns):
        X["timesteps_z:units_avg_z"] = X["timesteps_z"] * X["units_avg_z"]
    if {"resample_z","seconds_z"}.issubset(X.columns):
        X["resample_z:seconds_z"] = X["resample_z"] * X["seconds_z"]

    # interações batch × modelo (uma coluna por dummy presente)
    model_cols = [c for c in X.columns if c.startswith("model_")]
    for mc in model_cols:
        X[f"batch_log2_z:{mc}"] = X["batch_log2_z"] * X[mc]

    # saneamento final
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X.values.astype(float), y.astype(float), X.columns.tolist()


def fit_ols_simple(X, y):
    # mascara de linhas válidas
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xc, yc = X[mask], y[mask]
    if len(yc) < 8 or np.nanstd(yc) == 0:
        # pouco dado ou y sem variância => não ajusta
        beta = np.zeros(X.shape[1], dtype=float)
        yhat = np.full_like(y, np.nan, dtype=float)
        resid = np.full_like(y, np.nan, dtype=float)
        r2 = np.nan
        return beta, yhat, resid, r2
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    yhat_c = Xc @ beta
    r = yc - yhat_c
    denom = np.sum((yc - yc.mean())**2)
    r2 = 1 - (r @ r) / denom if denom > 0 else np.nan
    # re-injeta no tamanho original (opcional)
    yhat = np.full_like(y, np.nan, dtype=float); yhat[mask] = yhat_c
    resid = np.full_like(y, np.nan, dtype=float); resid[mask] = r
    return beta, yhat, resid, float(r2)


def fig_pareto_coefs(pdf, coefs, terms, title):
    # remove Intercept do ranking
    terms_arr = np.array(terms)
    coefs_arr = np.array(coefs, dtype=float)
    mask = np.char.lower(terms_arr) != "intercept"
    terms_arr, coefs_arr = terms_arr[mask], coefs_arr[mask]

    abscoef = np.abs(coefs_arr)
    order = np.argsort(-abscoef)[:15]
    t = terms_arr[order]; v = abscoef[order]; s = coefs_arr[order]
    colors = np.where(s >= 0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(t, v, color=colors)
    ax.set_xticklabels(t, rotation=60, ha="right")
    ax.set_ylabel("|coef| (Z_pool_std)"); ax.set_title(title)
    for xi, (vi, si) in enumerate(zip(v, s)):
        ax.text(xi, vi, f"{vi:.2f}\n({'+' if si>=0 else '-'})", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)



def fig_sign_consistency(pdf, beta1, terms1, beta2, terms2, title):
    d1 = {t:b for t,b in zip(terms1,beta1) if t!="Intercept"}
    d2 = {t:b for t,b in zip(terms2,beta2) if t!="Intercept"}
    common = sorted(set(d1)&set(d2))
    vals = [np.sign(d1[t])*np.sign(d2[t]) for t in common]
    fig = plt.figure(figsize=(min(10,max(6,0.4*len(common))),4))
    plt.bar(common, vals); plt.ylim(-1.1,1.1); plt.yticks([-1,0,1],["Flip","Zero","Igual"])
    plt.xticks(rotation=60, ha="right"); plt.title(title)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def desirability_from_z(z, kappa=1.5):
    # mapeia Z (−∞..+∞) → (0,1) de forma suave e simétrica
    # d = 1 / (1 + exp(−z/kappa))
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)/float(kappa)))

def fig_box_by_seconds_pooled(pdf, z, seconds):
    secs = sorted(pd.unique(seconds.dropna()))
    data = [z[seconds==s].values for s in secs]
    fig, ax = plt.subplots(figsize=(7.2,4.2))
    ax.boxplot(data, tick_labels=[f"{int(s)} s" for s in secs])
    ax.axhline(0, color="k", lw=1, alpha=0.4)
    ax.set_ylabel("Z (pooled por horizonte)")
    ax.set_title("Distribuição do desempenho por horizonte (pooled)")
    # anota IQR e n
    for i, s in enumerate(secs, start=1):
        vals = z[seconds==s].dropna().values
        if len(vals)==0: continue
        q1, q3 = np.quantile(vals, [0.25, 0.75]); iqr = q3-q1
        ax.text(i, q3, f"IQR={iqr:.2f}\n(n={len(vals)})", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _ecdf_vals(a):
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    if a.size==0: return np.array([]), np.array([])
    xs = np.sort(a); ys = np.arange(1, xs.size+1)/xs.size
    return xs, ys

def fig_ecdf_by_dir(pdf, df_pairs, title, by_seconds=False):
    # espera colunas: Z_1112, Z_1211, _seconds_
    if not {"Z_AB","Z_BA","_seconds_"}.issubset(df_pairs.columns): return
    if not by_seconds:
        fig, ax = plt.subplots(figsize=(7.2,4.0))
        for col, lbl in [("Z_AB","A→B"), ("Z_BA","B→A")]:
            xs, ys = _ecdf_vals(df_pairs[col].values)
            if xs.size: ax.step(xs, ys, where="post", label=lbl)
        ax.set_xlabel("Z"); ax.set_ylabel("ECDF")
        ax.set_title(title); ax.legend(loc="lower right", frameon=False)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig); return

    secs = sorted(df_pairs["_seconds_"].dropna().unique())
    for s in secs:
        sub = df_pairs[df_pairs["_seconds_"]==s]
        fig, ax = plt.subplots(figsize=(7.2,4.0))
        for col, lbl in [("Z_AB","A→B"), ("Z_BA","B→A")]:
            xs, ys = _ecdf_vals(sub[col].values)
            if xs.size: ax.step(xs, ys, where="post", label=lbl)
        ax.set_xlabel("Z"); ax.set_ylabel("ECDF")
        ax.set_title(f"{title} — {int(s)} s")
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def fig_coef_parity(pdf, termsA, coefsA, termsB, coefsB, title):
    dA = dict(zip(termsA, coefsA)); dB = dict(zip(termsB, coefsB))
    common = [t for t in dA if t in dB]
    if not common: return
    x = np.array([dA[t] for t in common], float)
    y = np.array([dB[t] for t in common], float)

    fig, ax = plt.subplots(figsize=(6.2,6.0))
    ax.scatter(x, y, alpha=0.8)
    lim = np.nanmax(np.abs(np.r_[x,y])); lim = float(np.ceil((lim+1e-6)*10)/10)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("coef (A→B)"); ax.set_ylabel("coef (B→A)")
    ax.set_title(title)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

from scipy.stats import wilcoxon

def table_paired_bias(pdf, df_pairs, out_csv):
    rows=[]
    for s in sorted(df_pairs["_seconds_"].dropna().unique()):
        sub = df_pairs[df_pairs["_seconds_"]==s]
        # diferenças pareadas Z_1112 - Z_1211
        d = (sub["Z_AB"] - sub["Z_BA"]).dropna().values
        if d.size < 5: continue
        try:
            stat, p = wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan
        bias = float(np.nanmean(d)); mad = float(np.nanmedian(np.abs(d-np.nanmedian(d))))
        rows.append({"seconds": int(s), "n": int(d.size), "mean_diff": bias, "MAD_diff": mad, "p_wilcoxon": p})

    if not rows: return
    tab = pd.DataFrame(rows).sort_values("seconds")
    tab.to_csv(out_csv, index=False)

    # desenha mini tabela no PDF (amostra)
    fig = plt.figure(figsize=(8.5, 2.8)); plt.axis("off")
    disp = tab.copy(); disp["mean_diff"] = disp["mean_diff"].map(lambda x: f"{x:.3f}")
    disp["MAD_diff"]  = disp["MAD_diff"].map(lambda x: f"{x:.3f}")
    disp["p_wilcoxon"]= disp["p_wilcoxon"].map(lambda x: f"{x:.3g}" if pd.notna(x) else "na")
    tbl = plt.table(cellText=disp.astype(str).values, colLabels=disp.columns.tolist(),
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.1)
    plt.title("Viés pareado (Z_AB − Z_BA) por horizonte — Wilcoxon")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fmt_num_cell(x, nd=3):
    """Formata número: inteiro sem .0; senão até nd casas, sem zeros à direita."""
    if pd.isna(x):
        return ""
    try:
        f = float(x)
    except Exception:
        return str(x)
    if np.isfinite(f) and float(int(round(f))) == f:
        return str(int(round(f)))
    return f"{f:.{nd}f}".rstrip("0").rstrip(".")

def tidy_key_str(s: str) -> str:
    """Limpa a coluna _key_: '100.0|3.0|60.0|...' -> '100|3|60|...'
       Mantém tokens não numéricos (ex.: '16-64:16') intactos."""
    parts = str(s).split("|")
    out = []
    for p in parts:
        try:
            f = float(p)
            if float(int(round(f))) == f:
                out.append(str(int(round(f))))
            else:
                out.append(f"{f}".rstrip("0").rstrip("."))
        except Exception:
            out.append(p)
    return "|".join(out)

def _as_int_str(x):
    try:
        f = float(x)
        return str(int(round(f))) if float(int(round(f))) == f else str(f).rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def split_key_row(key):
    """
    Converte _key_ = 'MODEL|resample|seconds|timesteps|epochs|units_str|batch'
    em colunas legíveis para o PDF.
    """
    toks = str(key).split("|")
    # preenche com "" se faltar algo
    toks = toks + [""] * (7 - len(toks))
    return {
        "Model": toks[0],
        "Resample (ms)": _as_int_str(toks[1]),
        "Seconds (s)": _as_int_str(toks[2]),
        "Timesteps": _as_int_str(toks[3]),
        "Epochs": _as_int_str(toks[4]),
        "Units": toks[5],
        "Batchsize": _as_int_str(toks[6]),
    }

def attach_key_columns_for_display(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Para o PDF: expande _key_ em colunas, mantém as demais colunas da tabela.
    Não altera o DataFrame usado para CSV.
    """
    if "_key_" not in df_in.columns:
        return df_in

    keydf = pd.DataFrame([split_key_row(k) for k in df_in["_key_"].values], index=df_in.index)

    disp = pd.concat([df_in.drop(columns=["_key_"]), keydf], axis=1)

    # cria/normaliza coluna 'Horizon' se fizer sentido
    if "Horizon" not in disp.columns and "Seconds (s)" in disp.columns:
        disp["Horizon"] = disp["Seconds (s)"].apply(lambda v: f"{v}s" if v != "" else "")

    # ordem sugerida
    front = ["Horizon", "Model", "Resample (ms)", "Seconds (s)", "Timesteps", "Epochs", "Units", "Batchsize"]
    ordered = [c for c in front if c in disp.columns] + [c for c in disp.columns if c not in front]
    return disp[ordered]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Dois caminhos (CSV/XLSX) separados por vírgula.")
    ap.add_argument("--output_dir", default="./out_v4", help="Pasta de saída.")
    ap.add_argument("--composite-col", default="Z-Score_Composite", help="Coluna do composto (maior=melhor).")
    ap.add_argument("--robust-lambda", type=float, default=0.3,
                    help="Peso de penalização da assimetria entre direções (default=0.3)")
    ap.add_argument("--robust-kappa", type=float, default=1.5,
                    help="Abertura da sigmoide para o Z→[0,1] (default=1.5)")
    args = ap.parse_args()

    lam = args.robust_lambda
    kap = args.robust_kappa

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
    if len(paths) < 2:
        raise SystemExit("Forneça dois arquivos em --inputs (ex.: 2DS1112.xlsx,2DS1211.xlsx).")

    dfA = load_any(Path(paths[0])); dfB = load_any(Path(paths[1]))
    dfA = build_composite(dfA, column_name=args.composite_col)
    dfB = build_composite(dfB, column_name=args.composite_col)

    # Marca 'Forecast steps' como derivado e audita
    dfA = compute_forecast_steps_derived(dfA)
    dfB = compute_forecast_steps_derived(dfB)

    # ---- Pooled Z por seconds (A+B combinados) ----
    comb = pd.concat(
        [dfA[["_seconds_", "_composite_base_"]].assign(_src_="A"),
         dfB[["_seconds_", "_composite_base_"]].assign(_src_="B")],
        ignore_index=True
    )
    comb["Z_pool_std"] = comb["_composite_base_"].groupby(comb["_seconds_"]).transform(
        lambda v: (v - v.mean()) / (v.std(ddof=0) if v.std(ddof=0) != 0 else 1.0)
    )
    # escreve de volta para cada fonte
    dfA["Z_pool_std"] = comb.loc[comb["_src_"] == "A", "Z_pool_std"].values
    dfB["Z_pool_std"] = comb.loc[comb["_src_"] == "B", "Z_pool_std"].values

    # def normalize_units_string(s):
    #     if not isinstance(s, str): return s
    #     m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(\d+)\s*$", s)
    #     return f"{m.group(1)}-{m.group(2)}:{m.group(3)}" if m else s
    #
    # u_col = getcol(dfA, "units_str") or "units_str"
    # if u_col in dfA.columns: dfA[u_col] = dfA[u_col].astype(str).map(normalize_units_string)
    # u_col = getcol(dfB, "units_str") or "units_str"
    # if u_col in dfB.columns: dfB[u_col] = dfB[u_col].astype(str).map(normalize_units_string)

    dfA["_key_"] = make_key(dfA); dfB["_key_"] = make_key(dfB)

    base_keep = ["_key_", "_dataset_id_", "_seconds_", "Z_pool_std", "__file__", "_forecast_steps_derived_"]

    keepA = base_keep.copy()
    keepB = base_keep.copy()

    for k in ["model", "resample_ms", "seconds", "timesteps", "epochs", "units_str", "batch_size"]:
        cA = getcol(dfA, k) or k
        cB = getcol(dfB, k) or k
        if cA in dfA.columns and cA not in keepA: keepA.append(cA)
        if cB in dfB.columns and cB not in keepB: keepB.append(cB)

    g1 = dfA[keepA].copy()
    g2 = dfB[keepB].copy()

    pairs = pd.merge(
        g1.rename(columns={"Z_pool_std":"Z_AB"}),
        g2[["_key_","Z_pool_std","_seconds_"]].rename(columns={"Z_pool_std":"Z_BA","_seconds_":"_seconds_2"}),
        on="_key_", how="inner"
    )
    pairs = pairs[pairs["_seconds_"]==pairs["_seconds_2"]].copy()
    pairs.drop(columns=["_seconds_2"], inplace=True)
    pairs["DeltaZ"] = pairs["Z_AB"] - pairs["Z_BA"]
    pairs.to_csv(out_dir/"paired_cross_domain.csv", index=False)

    # ---------- Índice robusto S(λ, κ) a partir de Z pooled ----------
    dAB = desirability_from_z(pairs["Z_AB"].values, kap)
    dBA = desirability_from_z(pairs["Z_BA"].values, kap)
    Smean = 0.5 * (dAB + dBA)
    Spene = lam * np.abs(dAB - dBA)
    pairs["Score_robusto"] = Smean - Spene
    pairs["dAB"] = dAB
    pairs["dBA"] = dBA
    pairs.to_csv(out_dir / "paired_with_score_robusto.csv", index=False)

    pdf_path = out_dir/"cross_domain_DOE_report_v4.pdf"
    pdf = PdfPages(pdf_path)

    page_title_explain(
        pdf,
        f"Índice robusto S(λ={lam:.2f}, κ={kap:.2f})",
        "Mapeia o Z padronizado (por horizonte, pooled entre domínios) para desejabilidade d∈(0,1) via sigmoide, "
        "combina os dois domínios pela média e penaliza a assimetria entre direções.",
        "Interpretação: S alto exige bom desempenho em ambos os domínios e pequena diferença entre eles. "
        "λ controla o quanto penalizamos discrepância; κ controla a 'abertura' da sigmoide (quão agressivo é o mapeamento Z→[0,1])."
    )

    page_title_explain(pdf,
        "Relatório DOE Cross-Domain (v4)",
        "Comparação pareada entre direções (ex.: A→B e B→A) com Z-score composto padronizado por horizonte (pooled).",
        "Este relatório não ajusta métricas; os gráficos de tempo nos relatórios de treino podem usar alinhamento visual."
    )

    page_title_explain(
        pdf,
        "Nota metodológica: 'Forecast steps' é fator derivado",
        "Forecast steps = round(seconds × 1000 / resample_ms). Por ser determinístico dado seconds e resample_ms, "
        "não é tratado como fator independente nas regressões/efeitos (OLS). Mantemos a coluna derivada apenas para rastreabilidade.",
        "O script audita eventuais divergências entre a coluna original e a derivada e salva um CSV de inconsistências, se houver."
    )

    # ---- Diagnóstico de pareamento ----
    keysA = set(dfA["_key_"]);
    keysB = set(dfB["_key_"])
    onlyA = keysA - keysB
    onlyB = keysB - keysA

    # contagens por seconds em cada arquivo
    sumA = dfA.groupby("_seconds_").size().rename("n_A")
    sumB = dfB.groupby("_seconds_").size().rename("n_B")
    sumP = pairs.groupby("_seconds_").size().rename("n_pairs")
    diag = pd.concat([sumA, sumB, sumP], axis=1).fillna(0).astype(int).reset_index()

    # salva CSVs com as linhas não pareadas (útil para auditar chave)
    dfA[dfA["_key_"].isin(list(onlyA))].to_csv(out_dir / "unmatched_only_in_AVB.csv", index=False)
    dfB[dfB["_key_"].isin(list(onlyB))].to_csv(out_dir / "unmatched_only_in_BA.csv", index=False)
    diag.to_csv(out_dir / "pairing_diagnostic_by_seconds.csv", index=False)

    # página no PDF
    page_title_explain(
        pdf,
        "Diagnóstico de pareamento",
        "Mostra n por horizonte em cada planilha e quantos foram pareados (merge 1-a-1 por hiperparâmetros).",
        "Diferenças entre n_A e n_B vs n_pairs indicam configurações sem par exato (linhas apagadas, formatação diferente de unidades/lotes etc.)."
    )
    fig = plt.figure(figsize=(8.5, 2.6));
    plt.axis("off")
    tbl = plt.table(cellText=diag.values, colLabels=diag.columns.tolist(),
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False);
    tbl.set_fontsize(9);
    tbl.scale(1, 1.2)
    plt.title("Contagens por horizonte (A=2DS1112, B=2DS1211)");
    plt.tight_layout();
    pdf.savefig(fig);
    plt.close(fig)

    page_title_explain(pdf,
                       "Distribuição por horizonte (pooled)",
                       "Boxplots do Z padronizado por horizonte (pool entre direções).",
                       "Mediana tende a 0; compare dispersão (IQR), caudas e outliers para identificar horizontes mais estáveis."
                       )
    fig_box_by_seconds_pooled(pdf, comb["Z_pool_std"], comb["_seconds_"])

    # resumo por seconds
    pairs_sum = pairs.groupby("_seconds_").agg(n=("DeltaZ","size"),
                                               mean_delta=("DeltaZ","mean"),
                                               std_delta=("DeltaZ","std")).reset_index()
    pairs_sum.to_csv(out_dir/"paired_summary_by_seconds.csv", index=False)
    fig = plt.figure(figsize=(9,2.4)); plt.axis("off")
    tbl = plt.table(cellText=np.round(pairs_sum.values,3),
                    colLabels=pairs_sum.columns.tolist(),
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.2)
    plt.title("Resumo pareado por horizonte (ΔZ = Z(A→B) − Z(B→A))")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    seconds_vals = sorted(pairs["_seconds_"].dropna().unique())
    for s in seconds_vals:
        sub = pairs[pairs["_seconds_"]==s].copy()
        page_title_explain(pdf,
            f"Scatter pareado – {int(s)} s",
            "Cada ponto é uma configuração. Eixo X: Z em A→B, Eixo Y: Z em B→A; linha tracejada é y=x.",
            "Pontos próximos da linha indicam robustez entre domínios; dispersão indica sensibilidade ao domínio."
        )
        fig_scatter_pairs(pdf, sub, s)
        page_title_explain(pdf,
            f"Bland–Altman – {int(s)} s",
            "Diferença (ZA→B − ZB→A) em função da média entre eles; linhas mostram viés e limites de concordância.",
            "Faixas estreitas e viés próximo de zero sugerem boa concordância. Tendências com a média indicam heterocedasticidade."
        )
        fig_bland_altman(pdf, sub, s)

    page_title_explain(pdf,
                       "ECDF por direção (global)",
                       "ECDF compara as distribuições de Z entre as duas direções.",
                       "Curva mais à direita indica maior probabilidade de Z altos; distâncias grandes sugerem vantagem consistente de uma direção."
                       )
    fig_ecdf_by_dir(pdf, pairs, "ECDF — Z por direção (global)", by_seconds=False)

    page_title_explain(pdf,
                       "ECDF por direção e por horizonte",
                       "Mesma análise, estratificada por seconds.",
                       "Ajuda a ver se a assimetria entre direções é constante ou depende do horizonte."
                       )
    fig_ecdf_by_dir(pdf, pairs, "ECDF — Z por direção", by_seconds=True)

    page_title_explain(pdf,
                       "Viés pareado entre direções (teste não-paramétrico)",
                       "Teste de Wilcoxon sobre as diferenças Z_AB−Z_BA por horizonte; tabela com média, MAD e p-valor.",
                       "p pequeno sugere viés sistemático; use em conjunto com Bland–Altman e ECDF."
                       )
    table_paired_bias(pdf, pairs, out_dir / "paired_bias_wilcoxon.csv")

    page_title_explain(pdf,
        "Sobreposição de Top-k",
        "Índice de Jaccard entre os Top-k de cada direção, por horizonte.",
        "Valores mais altos indicam maior estabilidade do ranking dos melhores entre domínios."
    )
    fig_jaccard_topk(pdf, pairs, seconds_vals, out_csv=out_dir/"jaccard_topk.csv", ks=(5,10,20))

    page_title_explain(pdf,
        "Concordância de ranking (Spearman-like)",
        "Correlação entre os ranks (ordem) dos Z em cada direção, por horizonte.",
        "ρ perto de 1 indica rankings similares; perto de 0 indica pouca concordância."
    )
    fig_spearman_by_seconds(pdf, pairs, out_csv=out_dir/"spearman_by_seconds.csv")

    # OLS por direção
    mask_AB = dfA["_dataset_id_"].astype(str).str.contains("A→B")
    mask_BA = dfB["_dataset_id_"].astype(str).str.contains("B→A")
    df_AB = dfA.loc[mask_AB].copy() if mask_AB.any() else dfA.copy()
    df_BA = dfB.loc[mask_BA].copy() if mask_BA.any() else dfB.copy()

    page_title_explain(pdf,
        "Efeitos principais (OLS) – A→B vs B→A",
        "Regressão linear de Z_pool_std nos fatores principais por direção.",
        "Coeficientes de maior módulo sugerem fatores mais influentes. Sinais iguais entre direções indicam efeito consistente."
    )
    X1,y1,terms1 = prepare_Xy(df_AB); b1,_,_,r2_1 = fit_ols_simple(X1,y1)
    fig_pareto_coefs(pdf, b1, terms1, f"Pareto |coef| – A→B (R²≈{r2_1:.3f})")
    X2,y2,terms2 = prepare_Xy(df_BA); b2,_,_,r2_2 = fit_ols_simple(X2,y2)
    fig_pareto_coefs(pdf, b2, terms2, f"Pareto |coef| – B→A (R²≈{r2_2:.3f})")
    fig_sign_consistency(pdf, b1, terms1, b2, terms2, "Consistência de sinais dos coeficientes (A→B vs B→A)")

    page_title_explain(pdf,
                       "Paridade de coeficientes OLS",
                       "Dispersão entre coeficientes ajustados por direção; linha y=x indica igualdade.",
                       "Pontos próximos da diagonal sugerem efeitos consistentes; quadrantes opostos indicam inversões de sinal."
                       )
    fig_coef_parity(pdf, terms1, b1, terms2, b2, "Paridade de coeficientes (OLS)")

    # Top-N robustos
    page_title_explain(pdf,
        "Top-N robustos (critério min-Z)",
        "Score robusto = min(ZA→B, ZB→A) por horizonte; listamos os melhores.",
        "Scores altos em ambas as direções indicam estabilidade; grandes diferenças sugerem sensibilidade ao domínio."
    )
    top_rows=[]
    for s in seconds_vals:
        sub = pairs[pairs["_seconds_"]==s].copy()
        sub["robust_score"] = np.minimum(sub["Z_AB"], sub["Z_BA"])
        top = sub.sort_values("robust_score", ascending=False).head(10)
        top_rows.append(pd.DataFrame(top[["_key_","robust_score","Z_AB","Z_BA"]]))
    if top_rows:
        top_table = pd.concat(top_rows, keys=[f"{int(s)}s" for s in seconds_vals])
        top_table.to_csv(out_dir / "topN_robustos_por_horizonte.csv")
        fig = plt.figure(figsize=(11, 3.2));
        plt.axis("off")
        disp = top_table.reset_index(level=0).rename(columns={"level_0": "Horizon"}).head(12).copy()
        # encurta a chave para caber na página
        disp["_key_"] = disp["_key_"].astype(str).str.slice(0, 60)

        # Para PDF: quebra _key_ em colunas legíveis
        disp_pdf = attach_key_columns_for_display(disp)

        # Formatação: números bonitos e _key_ (se ainda existir) higienizado
        disp_fmt = disp_pdf.copy()
        for c in disp_fmt.columns:
            if c == "_key_":
                disp_fmt[c] = disp_fmt[c].astype(str).map(tidy_key_str)
            elif pd.api.types.is_numeric_dtype(disp_fmt[c]):
                disp_fmt[c] = disp_fmt[c].apply(lambda v: fmt_num_cell(v, nd=3))
            else:
                disp_fmt[c] = disp_fmt[c].astype(str)

        # Render da tabela
        tbl = plt.table(
            cellText=disp_fmt.values,
            colLabels=list(disp_pdf.columns),
            loc="center",
            cellLoc="center"
        )
        tbl.auto_set_font_size(False);
        tbl.set_fontsize(8);
        tbl.scale(1, 1.1)
        plt.title("Top-N robustos (amostra) – ver CSV completo para a lista completa")
        plt.tight_layout();
        pdf.savefig(fig);
        plt.close(fig)

    # ---------- Top-N pelo ÍNDICE ROBUSTO ----------
    page_title_explain(
        pdf,
        "Top-N por Índice robusto S(λ, κ)",
        "Lista, por horizonte, as melhores configurações pelo score S (maior=melhor), que combina desejabilidade média "
        "entre direções com penalização de assimetria.",
        "Útil para selecionar candidatos estáveis cross-domain. Consulte também a sensibilidade a λ e κ."
    )

    top_rows_idx = []
    for s in seconds_vals:
        sub = pairs[pairs["_seconds_"] == s].copy()
        top = sub.sort_values("Score_robusto", ascending=False).head(10)
        top_rows_idx.append(top[["_key_", "Score_robusto", "dAB", "dBA", "Z_AB", "Z_BA"]])

    if top_rows_idx:
        top_table_idx = pd.concat(top_rows_idx, keys=[f"{int(s)}s" for s in seconds_vals])
        top_table_idx.to_csv(out_dir / "topN_indice_robusto_por_horizonte.csv")

        fig = plt.figure(figsize=(11, 3.2));
        plt.axis("off")
        disp = top_table_idx.reset_index(level=0).rename(columns={"level_0": "Horizon"}).head(12).copy()
        disp["_key_"] = disp["_key_"].astype(str).str.slice(0, 60)
        # Para PDF: quebra _key_ em colunas legíveis
        disp_pdf = attach_key_columns_for_display(disp)

        # Formatação: números bonitos e _key_ (se ainda existir) higienizado
        disp_fmt = disp_pdf.copy()
        for c in disp_fmt.columns:
            if c == "_key_":
                disp_fmt[c] = disp_fmt[c].astype(str).map(tidy_key_str)
            elif pd.api.types.is_numeric_dtype(disp_fmt[c]):
                disp_fmt[c] = disp_fmt[c].apply(lambda v: fmt_num_cell(v, nd=3))
            else:
                disp_fmt[c] = disp_fmt[c].astype(str)

        # Render da tabela
        tbl = plt.table(
            cellText=disp_fmt.values,
            colLabels=list(disp_pdf.columns),
            loc="center",
            cellLoc="center"
        )
        tbl.auto_set_font_size(False);
        tbl.set_fontsize(8);
        tbl.scale(1, 1.1)
        plt.title("Top-N por Índice robusto (amostra) – ver CSV completo")
        plt.tight_layout();
        pdf.savefig(fig);
        plt.close(fig)

    # ---------- Sensibilidade do Índice (varrendo λ e κ) ----------
    page_title_explain(
        pdf,
        "Sensibilidade do Índice robusto a λ e κ",
        "Compara o Top-10 (por horizonte) obtido com o par (λ,κ) corrente versus alternativas em uma malha.",
        "Valores Jaccard altos indicam estabilidade do ranking S sob variação de λ e κ; valores baixos sugerem sensibilidade."
    )

    lam_grid = [0.0, 0.2, 0.3, 0.4, 0.6]
    kap_grid = [1.0, 1.5, 2.0, 3.0]

    # Top-10 baseline por horizonte
    baseline_tops = {}
    for s in seconds_vals:
        sub = pairs[pairs["_seconds_"] == s].copy()
        baseline_tops[s] = set(sub.sort_values("Score_robusto", ascending=False).head(10).index.tolist())

    # varre λ, κ
    heat = np.zeros((len(lam_grid), len(kap_grid)))
    sens_rows = []
    for i, lam2 in enumerate(lam_grid):
        for j, kap2 in enumerate(kap_grid):
            # recalcula S com novos parâmetros (em cima dos mesmos Z_**)
            d1 = desirability_from_z(pairs["Z_AB"].values, kap2)
            d2 = desirability_from_z(pairs["Z_BA"].values, kap2)
            S2 = 0.5 * (d1 + d2) - lam2 * np.abs(d1 - d2)

            # Jaccard médio dos Top-10 vs baseline, agregando sobre os horizontes
            js = []
            for s in seconds_vals:
                sub = pairs[pairs["_seconds_"] == s].copy()
                idx_alt = set(sub.assign(S=S2[sub.index]).sort_values("S", ascending=False).head(10).index.tolist())
                idx_base = baseline_tops[s]
                denom = len(idx_alt | idx_base)
                jacc = len(idx_alt & idx_base) / denom if denom else np.nan
                js.append(jacc)
            heat[i, j] = float(np.nanmean(js)) if len(js) else np.nan
            sens_rows.append((lam2, kap2, heat[i, j]))

    # salva CSV de sensibilidade
    pd.DataFrame(sens_rows, columns=["lambda", "kappa", "mean_Jaccard_Top10"]).to_csv(
        out_dir / "sensibilidade_indice_lambda_kappa.csv", index=False
    )

    # plota heatmap
    fig = plt.figure(figsize=(6.8, 4.6))
    im = plt.imshow(heat, aspect="auto", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(kap_grid)), [f"{k:.1f}" for k in kap_grid])
    plt.yticks(range(len(lam_grid)), [f"{l:.1f}" for l in lam_grid])
    plt.xlabel("κ");
    plt.ylabel("λ")
    plt.title("Estabilidade do Top-10 (Jaccard médio vs baseline)")
    plt.tight_layout();
    pdf.savefig(fig);
    plt.close(fig)


    pdf.close()
    print("OK. Arquivos salvos em:", out_dir)
    print("PDF:", (out_dir/'cross_domain_DOE_report_v4.pdf').as_posix())
    # for aux in ["paired_cross_domain.csv","paired_summary_by_seconds.csv","jaccard_topk.csv","spearman_by_seconds.csv","topN_robustos_por_horizonte.csv"]:
    #     p = (out_dir/aux); print(aux, "=>", p.exists())

    for aux in [
        "paired_cross_domain.csv",
        "paired_with_score_robusto.csv",
        "paired_summary_by_seconds.csv",
        "jaccard_topk.csv",
        "spearman_by_seconds.csv",
        "topN_robustos_por_horizonte.csv",
        "topN_indice_robusto_por_horizonte.csv",
        "sensibilidade_indice_lambda_kappa.csv",
    ]:
        p = (out_dir / aux)
        print(aux, "=>", p.exists())


if __name__ == "__main__":
    main()
