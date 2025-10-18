r"""
doe_analysis_rssi_id.py
Análise IN-DISTRIBUTION (ID) a partir de um ÚNICO arquivo (CSV/XLSX).
- Lê 1 arquivo, padroniza colunas, marca forecast_steps como DERIVADO (seconds,resample_ms),
- Constrói um composto base (coluna "maior=melhor", ex.: Z-Score_Composite),
- Padroniza por horizonte (seconds) -> Z por seconds (média/DP OU robusto mediana/MAD),
- Gera PDF com: diagnóstico (contagens), distribuição por horizonte, médias por fator, interações 2D,
  OLS/Pareto (efeitos principais e por-horizonte), Top-N por horizonte.
Dependências: pandas, numpy, matplotlib, openpyxl (p/ .xlsx).

Exemplos de uso (Windows/PowerShell):
    python .\doe_analysis_rssi_id.py `
      --input r"C:\Users\Micro\Documents\RSSI_to_AMR\DoE\Data\Resultados_experimentos_RSSI_Z-Score_BigDS.xlsx" `
      --output_dir r"C:\User\Micro\Documents\RSSI_to_AMR\DoE\outV1" `
      --pdf "id_report.pdf" `
      --composite-col "Z-Score_Composite" `
      --robust-z 1

Se preferir, ajuste os DEFAULT_* abaixo e rode sem argumentos (F5/Run).


"""

import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ===================== CONFIGURAÇÃO DO USUÁRIO (opcional) =====================
DEFAULT_INPUT = r"C:\Users\Micro\Documents\RSSI_to_AMR\DoE\Data\Resultado_experimentos_RSSI_Z-Score_BigDS.xlsx"  # ex.: r"C:\...\Resultado_experimentos_RSSI_Z-Score_BigDS.xlsx"
DEFAULT_OUTPUT_DIR = r"C:\Users\Micro\Documents\RSSI_to_AMR\DoE\out_id"
DEFAULT_PDF_NAME = "id_report.pdf"
DEFAULT_COMPOSITE_COL = "Z-Score_Composite"  # coluna com "maior=melhor"
DEFAULT_ROBUST_Z = 1  # 1=usar Z robusto (mediana/MAD), 0=Z clássico (média/DP)

# ===================== MAPEAMENTO DE COLUNAS (sinônimos) ======================
CANON_MAP = {
    "model": ["model","modelo","MODEL","MODEL_TYPE","model_type","model type","Modelo"],
    "resample_ms": ["group_msec","group_ms","resample (ms)","resample_ms","resample","Resample (ms)"],
    "seconds": ["seconds_ahead","seconds to predict (s)","seconds_to_predict","sec_to_predict","Seconds to predict (s)"],
    "forecast_steps": ["forecast steps","forecast_steps","Forecast steps"],
    "timesteps": ["timesteps","timesteps_orig","window","lookback","Timesteps"],
    "epochs": ["epochs","Epochs"],
    "units_str": ["units_range","units (min-max:step)","units","Units (min-max:step)","Units"],
    "batch_size": ["batch_size","batchsize","batch","Batchsize"],
    "z_composite": ["z-score_composite","z-score composite","ranking final","Z_Composite_std","Z-Score_Composite","Z-Score_Composite "],
}

# ===================== HELPERS =====================
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
            has_many_unnamed = sum(c.startswith("Unnamed:") for c in cols) >= max(3, int(0.3*len(cols)))
            has_group_headers = any(g in cols for g in ["Hiperparametros","Tempos","Métricas - Resultados"])
            if not (has_many_unnamed or has_group_headers):
                df0["__source__"]=str(path); df0["__file__"]=path.name
                return normalize_columns(df0)
        # fallback: header em 2ª linha (padrão das tuas planilhas consolidadas)
        raw = pd.read_excel(path, header=None)
        header = raw.iloc[1].tolist()
        df = raw.iloc[2:].copy(); df.columns = header
        df["__source__"]=str(path); df["__file__"]=path.name
        return normalize_columns(df)
    raise RuntimeError(f"Unsupported file: {path}")

def compute_forecast_steps_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Cria _forecast_steps_derived_ = round(seconds*1000/resample_ms) e audita divergências da coluna original."""
    rs_col = getcol(df, "resample_ms") or "resample_ms"
    sc_col = getcol(df, "seconds") or "seconds"
    rs = smart_to_numeric(df[rs_col]) if rs_col in df.columns else pd.Series(np.nan, index=df.index)
    sc = smart_to_numeric(df[sc_col]) if sc_col in df.columns else pd.Series(np.nan, index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        derived = np.rint(sc * 1000.0 / rs)
    df["_forecast_steps_derived_"] = pd.Series(derived, index=df.index).astype("Int64")
    fs_col = getcol(df, "forecast_steps")
    if fs_col and fs_col in df.columns:
        orig = smart_to_numeric(df[fs_col]).astype("Int64")
        mismatch = orig.ne(df["_forecast_steps_derived_"])
        df.attrs["forecast_mismatch_count"] = int(mismatch.sum())
        if mismatch.any():
            Path(df["__source__"].iloc[0]).with_suffix("")
    else:
        df.attrs["forecast_mismatch_count"] = 0
    return df

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

    # composto base
    if column_name not in df.columns:
        col = getcol(df,"z_composite")
        if not col or col not in df.columns:
            raise RuntimeError(f"Coluna '{column_name}' não encontrada e não foi possível inferir o Z composto.")
        column_name = col
    base = smart_to_numeric(df[column_name])  # assume MAIOR = melhor
    df["_composite_base_"] = base
    return df

def z_by_seconds(series: pd.Series, seconds: pd.Series, robust: bool=True) -> pd.Series:
    """Padroniza por horizon (seconds). robust=True usa mediana/MAD; senão, média/DP."""
    def z_std(v):
        v = np.asarray(v, dtype=float); mu = np.nanmean(v); sd = np.nanstd(v, ddof=0)
        return (v - mu) / (sd if sd>0 else 1.0)
    def z_robust(v):
        v = np.asarray(v, dtype=float); med = np.nanmedian(v); mad = np.nanmedian(np.abs(v - med))
        scale = 1.4826*mad if mad>0 else 1.0
        return (v - med)/scale
    fn = z_robust if robust else z_std
    return series.groupby(seconds).transform(fn)

def desirability_from_z(z, kappa=1.5):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)/float(kappa)))

def make_key(df: pd.DataFrame):
    """Chave textual para identificar configuração (sem forecast_steps, que é DERIVADO)."""
    m = getcol(df,"model") or "model"
    rs = getcol(df,"resample_ms") or "resample_ms"
    sc = getcol(df,"seconds") or "seconds"
    ts = getcol(df,"timesteps") or "timesteps"
    ep = getcol(df,"epochs") or "epochs"
    un = getcol(df,"units_str") or "units_str"
    bs = getcol(df,"batch_size") or "batch_size"
    cols = [c for c in [m,rs,sc,ts,ep,un,bs] if c in df.columns]
    return df[cols].astype(str).agg("|".join, axis=1)

# ===================== FIGURAS =====================
def page_title_explain(pdf, title, what_is, how_to_read=None):
    fig = plt.figure(figsize=(11.69, 8.27)); plt.axis("off")
    y = 0.9
    plt.text(0.06,y,title,fontsize=18,fontweight="bold",ha="left",va="top")
    y -= 0.08; plt.text(0.06,y,"O que é: "+what_is,fontsize=11,ha="left",va="top",wrap=True)
    if how_to_read:
        y -= 0.14; plt.text(0.06,y,"Como interpretar: "+how_to_read,fontsize=11,ha="left",va="top",wrap=True)
    pdf.savefig(fig); plt.close(fig)

def fig_counts_by_seconds(pdf, df):
    cnt = df.groupby("_seconds_").size().rename("n").reset_index()
    fig = plt.figure(figsize=(6,3.2))
    plt.bar(cnt["_seconds_"].astype(int), cnt["n"].values, width=1.4)
    plt.xlabel("Horizon (s)"); plt.ylabel("n"); plt.title("Contagem por horizonte")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_box_by_seconds(pdf, z, seconds):
    fig = plt.figure(figsize=(6.4,4))
    data = [z[seconds==s].values for s in sorted(pd.unique(seconds.dropna()))]
    plt.boxplot(data, tick_labels=[str(int(s))+" s" for s in sorted(pd.unique(seconds.dropna()))])
    plt.ylabel("Z por horizon"); plt.title("Distribuição do desempenho por horizonte")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

def fig_mean_ci_by_factor(pdf, df, zcol, factor, title):
    # Seleciona somente as colunas necessárias e elimina NaN no zcol
    if factor not in df.columns:
        return
    s = df[[factor, zcol]].copy()
    s = s[ s[zcol].notna() ]
    if s.empty:
        return

    # Agrega: média, desvio, n; filtra grupos sem dados
    grp = s.groupby(factor, dropna=True)[zcol].agg(['mean','std','count']).reset_index()
    grp = grp[ grp['count'] > 0 ]
    if grp.empty:
        return

    # IC95% (aprox normal)
    grp['se'] = grp['std'] / np.sqrt(grp['count'].clip(lower=1))
    grp['ci95'] = 1.96 * grp['se']

    # Labels sempre como string para os ticks
    grp['__label__'] = grp[factor].astype(str)

    # Ordena por média desc e monta vetores diretamente do dataframe ordenado
    grp = grp.sort_values('mean', ascending=False).reset_index(drop=True)

    # Se sobrar apenas 1 nível, ainda desenha uma barra
    x = np.arange(len(grp))
    y = grp['mean'].values
    e = grp['ci95'].values
    labels = grp['__label__'].tolist()

    # Figura
    fig = plt.figure(figsize=(max(6, 0.55*len(labels)), 4))
    plt.bar(x, y, yerr=e, capsize=3)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("média Z")
    plt.title(title)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fig_heatmap_two(pdf, df, zcol, f1, f2, title):
    # cross tab mean
    piv = df.pivot_table(index=f1, columns=f2, values=zcol, aggfunc='mean')
    fig = plt.figure(figsize=(max(5,0.45*len(piv.columns)), max(4,0.45*len(piv.index))))
    im = plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
    plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns], rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return piv

# ===================== OLS =====================
def prepare_Xy(df):
    y = df["Z_by_seconds"].values.astype(float)
    mcol = getcol(df,"model") or "model"
    if mcol not in df.columns: df[mcol]="MODEL"
    M = pd.get_dummies(df[mcol].astype(str), prefix="model", drop_first=True)
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
        blog2 = np.where(batch > 0, np.log2(batch), np.nan)
    blog2 = pd.Series(blog2, index=df.index)
    if rs.notna().any() and rs.min()!=rs.max():
        rscaled = (2*(rs-rs.min())/(rs.max()-rs.min()))-1.0
    else:
        rscaled = pd.Series(0.0,index=df.index)
    X = pd.concat([pd.DataFrame({
        "Intercept":1.0,"resample_scaled":rscaled,"seconds":sec,"timesteps":ts,"epochs":ep,
        "units_avg":uavg,"batch_log2":blog2
    }), M], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # interações mais relevantes
    for a,b in [("timesteps","units_avg"),("resample_scaled","seconds"),("batch_log2","model_GRU")]:
        if a in X.columns and b in X.columns: X[f"{a}:{b}"] = X[a]*X[b]
    return X.values.astype(float), y.astype(float), X.columns.tolist()

def fit_ols_simple(X, y):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xc, yc = X[mask], y[mask]
    if len(yc) < 8 or np.nanstd(yc) == 0:
        beta = np.zeros(X.shape[1], dtype=float)
        r2 = np.nan
        return beta, r2, mask
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    r = yc - (Xc @ beta)
    denom = np.sum((yc - yc.mean())**2)
    r2 = 1 - (r @ r) / denom if denom > 0 else np.nan
    return beta.astype(float), float(r2), mask

def fig_pareto_coefs(pdf, coefs, terms, title):
    abscoef = np.abs(coefs); order = np.argsort(-abscoef)[:15]
    fig = plt.figure(figsize=(8,5))
    plt.bar(np.array(terms)[order], abscoef[order])
    plt.xticks(rotation=60, ha="right"); plt.ylabel("|coef| (Z_by_seconds)"); plt.title(title)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Caminho para 1 arquivo (CSV/XLSX).", default=DEFAULT_INPUT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Pasta de saída.")
    ap.add_argument("--pdf", default=DEFAULT_PDF_NAME, help="Nome do PDF de saída.")
    ap.add_argument("--composite-col", default=DEFAULT_COMPOSITE_COL, help="Coluna do composto (maior=melhor).")
    ap.add_argument("--robust-z", type=int, default=DEFAULT_ROBUST_Z, help="1=Z robusto (mediana/MAD), 0=Z clássico.")
    args = ap.parse_args()

    if not args.input:
        raise SystemExit("Forneça --input ou preencha DEFAULT_INPUT no topo do arquivo.")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / args.pdf

    # ----------------- Carrega e prepara -----------------
    df = load_any(Path(args.input))
    df = build_composite(df, column_name=args.composite_col)
    df = compute_forecast_steps_derived(df)
    df["_key_"] = make_key(df)

    # Z por seconds (robusto ou clássico)
    df["Z_by_seconds"] = z_by_seconds(df["_composite_base_"], df["_seconds_"], robust=bool(args.robust_z))

    # Exporta base limpa
    df_out = df.copy()
    keep_cols = ["_key_","_seconds_","Z_by_seconds","_composite_base_","__file__",
                 getcol(df,"model") or "model",
                 getcol(df,"resample_ms") or "resample_ms",
                 getcol(df,"seconds") or "seconds",
                 getcol(df,"timesteps") or "timesteps",
                 getcol(df,"epochs") or "epochs",
                 getcol(df,"units_str") or "units_str",
                 getcol(df,"batch_size") or "batch_size",
                 "_forecast_steps_derived_"]
    keep_cols = [c for c in keep_cols if c in df_out.columns]
    df_out[keep_cols].to_csv(out_dir/"id_dataset_clean.csv", index=False)

    # ----------------- PDF -----------------
    pdf = PdfPages(pdf_path)

    page_title_explain(
        pdf,
        "Relatório ID (in-distribution) – visão geral",
        "Uma única planilha: padronização por horizonte (Z_by_seconds), estatística descritiva, efeitos principais e interações.",
        "Quanto mais alto o Z_by_seconds, melhor. As figuras seguintes mostram padrões por fator e por pares de fatores."
    )

    page_title_explain(
        pdf,
        "Nota metodológica: 'Forecast steps' é fator derivado",
        "Forecast steps = round(seconds × 1000 / resample_ms). Por ser determinístico dado seconds e resample_ms, "
        "não é tratado como fator independente em OLS. Mantemos a coluna derivada para rastreabilidade.",
        "A padronização Z_by_seconds pode ser robusta (mediana/MAD) ou clássica (média/DP), conforme a flag --robust-z."
    )

    # Diagnóstico de contagens
    page_title_explain(
        pdf, "Diagnóstico de cobertura por horizonte",
        "Mostra quantas linhas cada horizonte possui no arquivo ID.",
        "Distribuição muito desigual por horizons pode enviesar efeitos; considere ponderações se necessário."
    )
    fig_counts_by_seconds(pdf, df)

    # Distribuição por horizonte
    page_title_explain(
        pdf, "Distribuição do desempenho por horizonte",
        "Boxplots de Z_by_seconds por 'seconds'.",
        "Caixas altas e compactas favorecem horizons específicos; caudas longas sugerem variância alta em certas configurações."
    )
    fig_box_by_seconds(pdf, df["Z_by_seconds"], df["_seconds_"])

    # Médias por fator (barras com IC95)
    for fac, fac_label in [
        (getcol(df,"model") or "model", "Modelo"),
        (getcol(df,"resample_ms") or "resample_ms", "Resample (ms)"),
        (getcol(df,"seconds") or "seconds", "Seconds (horizon)"),
        (getcol(df,"timesteps") or "timesteps", "Timesteps"),
        (getcol(df,"epochs") or "epochs", "Epochs"),
        (getcol(df,"units_str") or "units_str", "Units range"),
        (getcol(df,"batch_size") or "batch_size", "Batch size"),
    ]:
        if fac in df.columns:
            page_title_explain(
                pdf, f"Médias por fator – {fac_label}",
                f"Média e IC95% de Z_by_seconds por níveis de {fac_label}.",
                "Barras mais altas indicam níveis mais favoráveis; ICs que se sobrepõem fortemente sugerem diferenças menos conclusivas."
            )
            fig_mean_ci_by_factor(pdf, df, "Z_by_seconds", fac, f"{fac_label}: média±IC95%")

    # Interações 2D (heatmaps de média Z)
    # resample x seconds
    if (getcol(df,"resample_ms") or "resample_ms") in df.columns and (getcol(df,"seconds") or "seconds") in df.columns:
        page_title_explain(
            pdf, "Interação – Resample × Seconds",
            "Heatmap da média de Z_by_seconds para cada par (resample, seconds).",
            "Plataformas/quadrantes com média mais alta são combinações-alvo; desalinhamentos sugerem sensibilidade de horizonte."
        )
        piv = fig_heatmap_two(pdf, df, "Z_by_seconds", getcol(df,"resample_ms") or "resample_ms",
                              getcol(df,"seconds") or "seconds",
                              "Média(Z) por Resample×Seconds")
        piv.to_csv(out_dir/"heatmap_resample_seconds.csv")

    # timesteps x units_avg (binariza units_str em média)
    u_col = getcol(df,"units_str") or "units_str"
    if u_col in df.columns and (getcol(df,"timesteps") or "timesteps") in df.columns:
        parsed = df[u_col].astype(str).apply(parse_units); df["_units_avg_"] = parsed.apply(lambda t: t[0])
        page_title_explain(
            pdf, "Interação – Timesteps × Units_avg",
            "Heatmap da média de Z_by_seconds em função de (timesteps, média de units).",
            "Procure cristas/vales que indiquem 'sweet spots' de janela vs. capacidade do modelo."
        )
        piv2 = fig_heatmap_two(pdf, df, "Z_by_seconds", getcol(df,"timesteps") or "timesteps",
                               "_units_avg_", "Média(Z) por Timesteps×Units_avg")
        piv2.to_csv(out_dir/"heatmap_timesteps_unitsavg.csv")

    # batch x modelo
    if (getcol(df,"batch_size") or "batch_size") in df.columns and (getcol(df,"model") or "model") in df.columns:
        page_title_explain(
            pdf, "Interação – Batch × Modelo",
            "Heatmap da média de Z_by_seconds para (batch, modelo).",
            "Tendências diagonais sugerem ganhos com lotes maiores em um modelo específico."
        )
        piv3 = fig_heatmap_two(pdf, df, "Z_by_seconds", getcol(df,"batch_size") or "batch_size",
                               getcol(df,"model") or "model", "Média(Z) por Batch×Modelo")
        piv3.to_csv(out_dir/"heatmap_batch_model.csv")

    # OLS (global e por-horizonte)
    page_title_explain(
        pdf, "Efeitos principais (OLS) – global",
        "Regressão de Z_by_seconds nos fatores independentes (sem forecast_steps).",
        "Barras maiores (|coef|) sugerem fatores mais influentes; sinal indica direção do efeito."
    )
    X,y,terms = prepare_Xy(df); b,r2,_ = fit_ols_simple(X,y)
    fig_pareto_coefs(pdf, b, terms, f"Pareto |coef| – Global (R²≈{np.nan if r2 is None else r2:.3f})")
    pd.DataFrame({"term":terms,"coef":b}).to_csv(out_dir/"ols_coeffs_global.csv", index=False)

    # OLS por seconds
    secs = sorted(df["_seconds_"].dropna().unique())
    for s in secs:
        sub = df[df["_seconds_"]==s].copy()
        if len(sub) < 10: continue
        page_title_explain(
            pdf, f"Efeitos principais (OLS) – {int(s)} s",
            "Ajuste restrito ao horizonte indicado.",
            "Útil para capturar particularidades de cada janela temporal."
        )
        Xs,ys,ts = prepare_Xy(sub); bs,r2s,_ = fit_ols_simple(Xs,ys)
        fig_pareto_coefs(pdf, bs, ts, f"Pareto |coef| – {int(s)} s (R²≈{np.nan if r2s is None else r2s:.3f})")
        pd.DataFrame({"term":ts,"coef":bs}).to_csv(out_dir/f"ols_coeffs_{int(s)}s.csv", index=False)

    # Top-N por horizonte (pelo Z_by_seconds)
    page_title_explain(
        pdf, "Top-N por horizonte (ID)",
        "Lista as melhores configurações por 'seconds' segundo Z_by_seconds (maior=melhor).",
        "Útil para selecionar candidatos de produção no mesmo domínio."
    )
    tops=[]
    for s in secs:
        sub = df[df["_seconds_"]==s].copy().sort_values("Z_by_seconds", ascending=False).head(10)
        tops.append(sub[["_key_","Z_by_seconds"]])
    if tops:
        top_tbl = pd.concat(tops, keys=[f"{int(s)}s" for s in secs])
        top_tbl.to_csv(out_dir/"topN_id_por_horizonte.csv")
        # mostra amostra na página
        fig = plt.figure(figsize=(11, 3.0)); plt.axis("off")
        disp = top_tbl.reset_index(level=0).rename(columns={"level_0":"Horizon"}).head(12).copy()
        disp["_key_"] = disp["_key_"].astype(str).str.slice(0, 60)
        disp_fmt = disp.copy()
        for c in disp_fmt.columns:
            if pd.api.types.is_numeric_dtype(disp_fmt[c]): disp_fmt[c] = disp_fmt[c].round(3)
        disp_fmt = disp_fmt.astype(str)
        tbl = plt.table(cellText=disp_fmt.values, colLabels=disp_fmt.columns.tolist(),
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1,1.15)
        plt.title("Top-N por horizonte (amostra) — ver CSV completo")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    pdf.close()

    print("OK. Arquivos salvos em:", out_dir.as_posix())
    print("PDF:", pdf_path.as_posix())
    for aux in [
        "id_dataset_clean.csv",
        "heatmap_resample_seconds.csv",
        "heatmap_timesteps_unitsavg.csv",
        "heatmap_batch_model.csv",
        "ols_coeffs_global.csv",
        "topN_id_por_horizonte.csv",
    ]:
        p = (out_dir / aux); print(aux, "=>", p.exists())

if __name__ == "__main__":
    main()
