# tools/backfill_results.py
# -*- coding: utf-8 -*-
import os, sys, json, argparse, shutil, glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# importa util para alinhar GROUP_MSEC e carregar dataset/escala
import utils as U
from utils import import_dataset

CSV_DEFAULT = os.path.join("experiments", "results_summary_2DS.csv")
EXP_DIR_DEFAULT = "experiments"

def inverse_scale_from_scaler(scaler, arr, feature_name="serving_cell_rssi_1"):
    # tenta localizar índice da feature no scaler (robusto a versões)
    try:
        if hasattr(scaler, "feature_names_in_"):
            idx = list(scaler.feature_names_in_).index(feature_name)
        else:
            idx = 0
        mn = float(scaler.data_min_[idx])
        mx = float(scaler.data_max_[idx])
    except Exception:
        # fallback: assume [0,1] -> [0,1] (não deve acontecer)
        mn, mx = 0.0, 1.0
    return arr * (mx - mn) + mn

def compute_denorm_metrics(experiment_folder: str):
    # carrega JSONs
    with open(os.path.join(experiment_folder, 'training_config.json'), 'r') as f:
        training_config = json.load(f)
    with open(os.path.join(experiment_folder, 'training_results.json'), 'r') as f:
        training_results = json.load(f)

    params = (training_config.get("params") or {})
    group_ms = int(params.get("group_msec", 100))
    timesteps = params.get("timesteps")
    if timesteps is None:
        timesteps = (training_config.get("Sequencing") or {}).get("timesteps")
    if timesteps is None:
        raise RuntimeError("timesteps não encontrado no training_config.json")

    # alinhar GROUP_MSEC para reimportar dataset com o mesmo downsample usado no treino
    if hasattr(U, "set_group_msec"):
        U.set_group_msec(group_ms)
    else:
        U.GROUP_MSEC = group_ms

    # nome do dataset de experimento
    ds_info = training_results.get("dataset_name", {})
    if isinstance(ds_info, dict):
        exp_name = ds_info.get("experiment")
    else:
        exp_name = ds_info
    if not exp_name:
        raise RuntimeError("Nome do dataset de experimento ausente no training_results.json")

    # carregar apenas para obter scaler e test_df
    scaler, _, _, test_df, _, _ = import_dataset(os.path.join("data", exp_name), split_train_val_test=False)

    # matrizes salvas (não refaz previsões)
    perf_rec = training_results.get("performance_recursive", {}) or {}
    preds_full = np.array(perf_rec.get("multi_step_predictions", []))
    y_test    = np.array(perf_rec.get("multi_step_true", []))
    if preds_full.size == 0 or y_test.size == 0:
        raise RuntimeError("multi_step_predictions/multi_step_true ausentes — pular ou recomputar.")

    # desscalar
    y_test_den = inverse_scale_from_scaler(scaler, y_test)
    preds_den  = inverse_scale_from_scaler(scaler, preds_full)

    # métricas agregadas (todos os passos)
    mae_den  = mean_absolute_error(y_test_den.ravel(), preds_den.ravel())
    mse_den  = mean_squared_error(y_test_den.ravel(), preds_den.ravel())
    rmse_den = np.sqrt(mse_den)
    r2_den   = r2_score(y_test_den.ravel(), preds_den.ravel())

    # passo 1 e último passo
    mae1  = mean_absolute_error(y_test_den[:, 0], preds_den[:, 0])
    mse1  = mean_squared_error(y_test_den[:, 0], preds_den[:, 0])
    rmse1 = np.sqrt(mse1)
    r21   = r2_score(y_test_den[:, 0], preds_den[:, 0])

    maeN  = mean_absolute_error(y_test_den[:, -1], preds_den[:, -1])
    mseN  = mean_squared_error(y_test_den[:, -1], preds_den[:, -1])
    rmseN = np.sqrt(mseN)
    r2N   = r2_score(y_test_den[:, -1], preds_den[:, -1])

    return {
        "multi_mae_den": mae_den, "multi_mse_den": mse_den, "multi_rmse_den": rmse_den, "multi_r2_den": r2_den,
        "step1_mae_den": mae1, "step1_mse_den": mse1, "step1_rmse_den": rmse1, "step1_r2_den": r21,
        "stepN_mae_den": maeN, "stepN_mse_den": mseN, "stepN_rmse_den": rmseN, "stepN_r2_den": r2N,
    }

def update_csv_inplace(csv_path: str, exp_dir: str, dry_run=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    if "experiment_folder" not in df.columns:
        raise RuntimeError("CSV não tem coluna 'experiment_folder' — não consigo fazer o match de forma confiável.")

    # backup
    bak = csv_path + ".bak"
    shutil.copyfile(csv_path, bak)
    print(f"[INFO] Backup criado: {bak}")

    # índice por pasta
    df["__key__"] = df["experiment_folder"].astype(str).str.replace("\\", "/").str.strip()
    updated = 0; skipped = 0; failed = 0

    # percorre pastas válidas
    folders = [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]
    for f in sorted(folders):
        key = f.replace("\\", "/")
        if key not in set(df["__key__"]):
            # linha não está no CSV — pula (ou poderíamos anexar)
            skipped += 1
            continue
        try:
            metrics = compute_denorm_metrics(f)
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")
            failed += 1
            continue

        # atualiza colunas
        for k, v in metrics.items():
            df.loc[df["__key__"] == key, k] = v
        updated += 1

    df = df.drop(columns=["__key__"])
    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV atualizado: {csv_path} | pastas atualizadas: {updated} | pulares: {skipped} | falhas: {failed}")
    else:
        print(f"[DRY-RUN] atualizaria {updated} linhas; pulares {skipped}; falhas {failed}")

def rebuild_csv_from_experiments(csv_out: str, exp_dir: str):
    """
    Alternativa: reconstruir o CSV do zero lendo cada experimento e
    recompondo todas as colunas essenciais + campos denorm.
    (Útil se o CSV atual estiver inconsistente.)
    """
    rows = []
    folders = [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]
    for f in sorted(folders):
        try:
            with open(os.path.join(f, 'training_config.json'), 'r') as c:
                cfg = json.load(c)
            with open(os.path.join(f, 'best_hyperparameters.json'), 'r') as b:
                best = json.load(b)
            with open(os.path.join(f, 'training_results.json'), 'r') as r:
                res = json.load(r)
            # denorm metrics
            den = compute_denorm_metrics(f)
        except Exception as e:
            print(f"[WARN] {os.path.basename(f)}: {e}")
            continue

        params = (cfg.get("params") or {})
        ds_names = res.get("dataset_name", {})
        if isinstance(ds_names, dict):
            trn = ds_names.get("training"); exp = ds_names.get("experiment")
        else:
            trn, exp = None, ds_names

        row = {
            "experiment_folder": f.replace("\\", "/"),
            "model": res.get("model_type"),
            "train_dataset": trn, "exp_dataset": exp,
            "start_training": res.get("start_datetime_training"),
            "end_training": res.get("end_datetime_training"),
            "train_time_s": res.get("training_elapsed_time_seconds"),
            "start_pred": res.get("start_datetime_predictions"),
            "end_pred": res.get("end_datetime_predictions"),
            "pred_time_s": res.get("predictions_elapsed_time_seconds"),
            "group_msec": params.get("group_msec"),
            "seconds_ahead": params.get("forecast_horizon_sec"),
            "forecast_steps": res.get("forecast_steps") or params.get("forecast_steps"),
            "timesteps_orig": params.get("timesteps_orig"),
            "batch_size": params.get("batch_size"),
            "units_range": f"{params.get('min_units')}-{params.get('max_units')}:{params.get('units_step')}",
            # melhores HPs
            "best_layers": best.get("num_layers"),
            "best_activation": best.get("activation"),
            "best_optimizer": best.get("optimizer"),
            "best_dropout": best.get("dropout_rate"),
            "best_units_l1": best.get("units_layer_1"),
            "best_units_l2": best.get("units_layer_2"),
            # direct
            "direct_mae": (res.get("performance_normal") or {}).get("mae"),
            "direct_mse": (res.get("performance_normal") or {}).get("mse"),
            "direct_rmse": (res.get("performance_normal") or {}).get("rmse"),
            "direct_r2":  (res.get("performance_normal") or {}).get("r2"),
            # multi (normalizado) se existir
            "multi_mae":  (res.get("performance_recursive") or {}).get("mae"),
            "multi_mse":  (res.get("performance_recursive") or {}).get("mse"),
            "multi_rmse": (res.get("performance_recursive") or {}).get("rmse"),
            "multi_r2":   (res.get("performance_recursive") or {}).get("r2"),
            # multi (dBm) — denorm
            **den
        }
        rows.append(row)

    if not rows:
        print("[ERRO] Nenhum experimento válido encontrado para reconstruir CSV.")
        sys.exit(2)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[INFO] CSV reconstruído: {csv_out} ({len(df)} linhas)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT, help="Caminho do CSV a atualizar (default: experiments/results_summary_2DS.csv)")
    ap.add_argument("--expdir", default=EXP_DIR_DEFAULT, help="Pasta com experimentos (default: experiments/)")
    ap.add_argument("--mode", choices=["update", "rebuild"], default="update",
                    help="update=atualiza colunas denorm no CSV existente; rebuild=gera novo CSV do zero.")
    ap.add_argument("--dry-run", action="store_true", help="Não salva alterações; só reporta.")
    args = ap.parse_args()

    if args.mode == "update":
        update_csv_inplace(args.csv, args.expdir, dry_run=args.dry_run)
    else:
        # cria novo CSV ao lado do original
        out = args.csv
        if os.path.exists(out):
            base, ext = os.path.splitext(out)
            out = base + "_rebuild" + ext
        rebuild_csv_from_experiments(out, args.expdir)

if __name__ == "__main__":
    main()
