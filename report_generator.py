import os, sys, json, argparse
import numpy as np
import pandas as pd
import utils as U

# backend sem GUI (seguro em servidor/IDE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from tensorflow.keras.models import load_model  # (não usado nesta versão, mas deixei)
from utils import import_dataset

# Opcional: seletor de pasta via GUI
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_OK = True
except Exception:
    TK_OK = False


REQUIRED_JSONS = ("training_config.json", "best_hyperparameters.json", "training_results.json")

def has_required_jsons(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, f)) for f in REQUIRED_JSONS)


ALIGN_TO_TARGET_TIME = True  # se False, volta ao alinhamento pelo fim da janela

# === Alinhamento opcional só para visualização (gráfico temporal) ===
VISUAL_SHIFT = True                 # habilita/desabilita o ajuste visual
SHIFT_METHOD = "auto"               # "auto" (xcorr) ou "fixed"
FIXED_SHIFT_STEPS = 0               # usado se SHIFT_METHOD == "fixed" (ex.: -6 adianta o predito 6 amostras)
MAX_LAG_SEC = float(HSEC) if 'HSEC' in globals() else 5.0  # procura até o horizonte previsto


def _best_lag_by_xcorr(y_true: np.ndarray, y_pred: np.ndarray, max_lag_samples: int) -> int:
    """Retorna lag em amostras que maximiza a correlação (lag>0: predito atrasado; lag<0: adiantado)."""
    y1 = np.asarray(y_true, float); y2 = np.asarray(y_pred, float)
    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag_samples, max_lag_samples+1):
        if lag < 0:   # pred adiantado -> cortar fim do pred
            a = y1[-lag:]              # corta começo do true
            b = y2[:len(a)]
        elif lag > 0: # pred atrasado -> cortar começo do pred
            a = y1[:-lag]
            b = y2[lag:]
        else:
            a, b = y1, y2
        n = min(len(a), len(b))
        if n < 16: continue
        a0 = a[:n] - a[:n].mean(); b0 = b[:n] - b[:n].mean()
        den = (a0.std() * b0.std());
        if den == 0: continue
        corr = float((a0 @ b0) / (n * den))
        if np.isfinite(corr) and corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag

def _apply_lag_for_plot(y_true, y_pred, t_axis, lag):
    """Aplica o lag no par (true, pred) e corta t para manter comprimentos iguais."""
    if lag < 0:   # pred adiantado -> atrasar no plot (cortar fim do pred)
        y_pred_plot = y_pred[:len(y_pred)+lag]     # lag negativo
        y_true_plot = y_true[-lag:][:len(y_pred_plot)]
        t_plot      = t_axis[-lag:][:len(y_pred_plot)]
    elif lag > 0: # pred atrasado -> adiantar no plot (cortar início do pred)
        y_pred_plot = y_pred[lag:]
        y_true_plot = y_true[:len(y_pred_plot)]
        t_plot      = t_axis[lag:][:len(y_pred_plot)]
    else:
        y_true_plot, y_pred_plot, t_plot = y_true, y_pred, t_axis
    m = min(len(y_true_plot), len(y_pred_plot), len(t_plot))
    return y_true_plot[:m], y_pred_plot[:m], t_plot[:m]

def _compute_lag_table(y_true_steps: np.ndarray, y_pred_steps: np.ndarray, sps: float, max_lag_s: float = "auto"):
    """
    y_*_steps: shape [N, n_steps] (cada coluna = passo j)
    Retorna DataFrame com Horizon(s), Step, Lag(samples/s), Corr@lag (só p/ referência), MAE raw/alinhado.
    """
    assert y_true_steps.shape == y_pred_steps.shape and y_true_steps.ndim == 2
    N, n_steps = y_true_steps.shape
    if max_lag_s == "auto":
        max_lag_samples = n_steps - 1
    else:
        max_lag_samples = int(round(float(max_lag_s) * sps))
    horizons_s = np.arange(n_steps, dtype=float) / float(sps)

    rows = []
    for j in range(n_steps):
        yt = np.asarray(y_true_steps[:, j], float)
        yp = np.asarray(y_pred_steps[:, j], float)
        lag = _best_lag_by_xcorr(yt, yp, max_lag_samples)
        # métricas "cruas" (sem alinhamento) – NÃO usamos alinhadas para avaliação
        mae_raw = float(np.mean(np.abs(yt - yp)))
        # só para análise de sensibilidade/visual:
        if lag < 0:
            yp2 = yp[:len(yp)+lag]; yt2 = yt[-lag:][:len(yp2)]
        elif lag > 0:
            yp2 = yp[lag:]; yt2 = yt[:len(yp2)]
        else:
            yt2, yp2 = yt, yp
        m = min(len(yt2), len(yp2))
        mae_al = float(np.mean(np.abs(yt2[:m] - yp2[:m]))) if m else np.nan
        corr   = np.corrcoef(yt2[:m] - np.mean(yt2[:m]), yp2[:m] - np.mean(yp2[:m]))[0,1] if m>1 else np.nan
        rows.append({
            "Step": j+1,
            "Horizon (s)": horizons_s[j],
            "Lag (samples)": lag,
            "Lag (s)": lag/float(sps),
            "Corr@lag": corr,
            "MAE (raw)": mae_raw,
            "MAE (aligned)": mae_al,
            "ΔMAE (%)": 100.0*(mae_al - mae_raw)/mae_raw if mae_raw else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("Horizon (s)").reset_index(drop=True)
    return df


def print_progress_bar(*args, **kwargs):
    # Mantido só para compat (não usamos)
    pass

def generate_report_pdf(experiment_folder: str) -> str:
    """
    Gera o PDF do experimento informado, reutilizando resultados já salvos em JSON.
    Retorna o caminho do PDF gerado.
    """
    if not experiment_folder or not os.path.isdir(experiment_folder):
        raise ValueError(f"Pasta de experimento inválida: {experiment_folder}")

    # === CARREGAR JSONs ===
    with open(os.path.join(experiment_folder, 'training_config.json'), 'r') as f:
        training_config = json.load(f)
    with open(os.path.join(experiment_folder, 'best_hyperparameters.json'), 'r') as f:
        best_hps = json.load(f)
    with open(os.path.join(experiment_folder, 'training_results.json'), 'r') as f:
        training_results = json.load(f)

    params = training_config.get("params", {}) or {}
    group_ms = int(params.get("group_msec", 100))  # fallback 100


    # >>> importante: alinhar o GROUP_MSEC do utils ao do experimento <<<
    if hasattr(U, "set_group_msec"):
        U.set_group_msec(group_ms)
    else:
        U.GROUP_MSEC = group_ms

    # === DADOS BÁSICOS ===
    model_type = training_results.get('model_type', 'Unknown')
    params = training_config.get("params", {}) or {}

    # samples_per_sec / horizonte (segundos)
    SPS = params.get("samples_per_sec", 1) or 1
    HSEC = params.get("forecast_horizon_sec")
    if HSEC is None:
        # fallback: steps / sps
        fs = training_results.get('forecast_steps') or params.get("forecast_steps") or 1
        HSEC = float(fs) / float(SPS)

    # timesteps de treino (para alinhar timestamps/posições do test_df)
    timesteps = params.get("timesteps")
    if timesteps is None:
        # outro local possível
        timesteps = training_config.get("Sequencing", {}).get("timesteps")
    if timesteps is None:
        raise ValueError("timesteps não encontrado em training_config/params.")

    forecast_steps = training_results.get('forecast_steps') or params.get("forecast_steps")
    if forecast_steps is None:
        # deduz pelo shape das matrizes
        ms_preds = training_results.get("performance_recursive", {}).get("multi_step_predictions")
        forecast_steps = len(ms_preds[0]) if ms_preds else 1

    # nomes dos datasets
    dataset_info = training_results.get("dataset_name", {})
    if isinstance(dataset_info, dict):
        train_name = dataset_info.get("training", "Unknown training dataset")
        exp_name   = dataset_info.get("experiment", "Unknown experiment dataset")
        dataset_name_str = f"Training: {train_name} | Experiment: {exp_name}"
        experiment_dataset_name = exp_name
    else:
        train_name = None
        exp_name = dataset_info
        dataset_name_str = str(dataset_info)
        experiment_dataset_name = exp_name

    if not experiment_dataset_name:
        raise ValueError("Nome do dataset de experimento não encontrado no training_results.json")

    # === CARREGAR APENAS O DATASET DE EXPERIMENTO (para scaler/timestamps/posições) ===
    experiment_dataset_path = os.path.join("data", experiment_dataset_name)
    scaler, _, _, test_df, _, ds_meta_from_import = import_dataset(experiment_dataset_path, split_train_val_test=False)

    # === PEGAR PREVISÕES/JANELAS JÁ SALVAS (NÃO REFAZ PREVISÕES) ===
    perf_rec = training_results.get("performance_recursive", {}) or {}
    preds_full = np.array(perf_rec.get("multi_step_predictions", []))  # shape (N, forecast_steps)
    y_test     = np.array(perf_rec.get("multi_step_true", []))         # shape (N, forecast_steps)

    if preds_full.size == 0 or y_test.size == 0:
        raise ValueError("multi_step_predictions / multi_step_true não encontrados no training_results.json")

    # === DESSCALAR (robusto a versões do sklearn) ===
    if hasattr(scaler, "feature_names_in_"):
        try:
            rssi_idx = list(scaler.feature_names_in_).index("serving_cell_rssi_1")
        except ValueError:
            rssi_idx = 0
    else:
        rssi_idx = 0

    min_rssi = float(scaler.data_min_[rssi_idx])
    max_rssi = float(scaler.data_max_[rssi_idx])
    inverse_scale = lambda arr: arr * (max_rssi - min_rssi) + min_rssi

    # First-step e last-step (desscalados)
    y_true_first = inverse_scale(y_test[:, 0])
    y_pred_first = inverse_scale(preds_full[:, 0])

    y_true_last  = inverse_scale(y_test[:, -1])
    y_pred_last  = inverse_scale(preds_full[:, -1])

    # >>> MÉTRICAS DESSCALADAS (NOVIDADE) <<<
    # Todos os passos combinados (flatten)
    y_true_all_den = inverse_scale(y_test).ravel()
    y_pred_all_den = inverse_scale(preds_full).ravel()

    mae_den = mean_absolute_error(y_true_all_den, y_pred_all_den)
    mse_den = mean_squared_error(y_true_all_den, y_pred_all_den)
    rmse_den = np.sqrt(mse_den)
    r2_den = r2_score(y_true_all_den, y_pred_all_den)

    # Step 1 (comparável ao one-step-ahead)
    mae_first_den = mean_absolute_error(y_true_first, y_pred_first)
    mse_first_den = mean_squared_error(y_true_first, y_pred_first)
    rmse_first_den = np.sqrt(mse_first_den)
    r2_first_den = r2_score(y_true_first, y_pred_first)

    # Último passo (pior caso dentro da janela multi-step)
    mae_last_den = mean_absolute_error(y_true_last, y_pred_last)
    mse_last_den = mean_squared_error(y_true_last, y_pred_last)
    rmse_last_den = np.sqrt(mse_last_den)
    r2_last_den = r2_score(y_true_last, y_pred_last)

    # N de janelas (igual ao número de linhas em y_test/preds_full)
    N = len(y_true_first)

    # Slices do test_df alinhados ao início (i + timesteps)
    t_all = test_df['time[s]'].to_numpy()
    posx_all = test_df['position_x'].to_numpy()
    posy_all = test_df['position_y'].to_numpy()

    # CORRETO:
    first_shift = 0 if ALIGN_TO_TARGET_TIME else 0
    # last_shift = (forecast_steps - 1) if ALIGN_TO_TARGET_TIME else (forecast_steps - 1)
    last_shift = 0

    # # First step: alinhar ao instante do ALVO (t_{k+1})
    # t_first = t_all[timesteps + first_shift : timesteps + first_shift + N]
    # x_first = x_all[timesteps + first_shift : timesteps + first_shift + N]
    # y_first = y_all[timesteps + first_shift : timesteps + first_shift + N]
    # FIRST STEP — alvo em t[ timesteps ]
    first_idx0 = timesteps
    N1 = min(len(y_true_first), len(y_pred_first), len(t_all) - first_idx0)

    t_first = t_all[first_idx0 + first_shift : first_idx0 + first_shift  + N1]
    pos_x_first = posx_all[first_idx0 + first_shift: first_idx0 + first_shift + N1]
    pos_y_first = posy_all[first_idx0 + first_shift: first_idx0 + first_shift + N1]
    y_pred_1step = y_pred_first[:N1]  # predito (1º passo)

    # Clip defensivo para manter os comprimentos iguais
    L = min(N1, len(t_first), len(pos_x_first), len(pos_y_first))
    y_true_first = y_true_first[:L]
    y_pred_1step = y_pred_1step[:L]
    t_first = t_first[:L]
    pos_x_first = pos_x_first[:L]
    pos_y_first = pos_y_first[:L]

    df_first = pd.DataFrame({
        'time[s]': t_first,
        'serving_cell_rssi_1': y_true_first,
        'predicted_rssi': y_pred_1step,
        'position_x': pos_x_first,
        'position_y': pos_y_first
    })

    # # Last step: alinhar ao instante do ALVO no último passo (t_{k + forecast_steps})
    # offset = forecast_steps
    #
    # t_last = t_all[timesteps + last_shift : timesteps + last_shift + len(y_true_last)]
    # x_last = x_all[timesteps + last_shift : timesteps + last_shift + len(y_true_last)]
    # y_last = y_all[timesteps + last_shift : timesteps + last_shift + len(y_true_last)]
    # LAST STEP — alvo em t[ timesteps + forecast_steps - 1 ]
    last_idx0 = timesteps + forecast_steps - 1
    N2 = min(len(y_true_last), len(y_pred_last), len(t_all) - last_idx0)
    t_last = t_all[last_idx0: last_idx0 + N2]
    pos_x_last = posx_all[last_idx0: last_idx0 + N2]
    pos_y_last = posy_all[last_idx0: last_idx0 + N2]
    y_pred_end = y_pred_last[:N2]


    # metas do dataset (no teu main, ficou aninhado dentro de performance_recursive)
    ds_meta = (training_results.get("performance_recursive", {}) or {}).get("dataset_meta", {})

    # === CRIAR PDF ===
    report_filename = os.path.join(
        experiment_folder, f"{model_type.lower()}_training_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    )
    with PdfPages(report_filename) as pdf:
        # Página 1 — Resumo
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')

        perf_train = training_results.get("performance_training", {}) or {}
        perf_val   = training_results.get("performance_validation", {}) or {}
        perf_norm  = training_results.get("performance_normal", {}) or {}
        rec_errors = (training_results.get("performance_recursive", {}) or {}).get("recursive_errors_per_step", {}) or {}

        text = []
        text.append("Training Configuration and Summary:\n")
        text.append(f"Dataset used: {dataset_name_str}\n\n")

        text.append("Params:\n")
        for k, v in (params.items() if isinstance(params, dict) else []):
            text.append(f"  - {k}: {v}\n")

        # Tuning info
        tuning = training_results.get("tuning_info", {}) or {}
        text.append("\nTuning:\n")
        text.append(f"  Trials executed: {tuning.get('executed_trials')}/{tuning.get('max_trials')}\n")

        # Perf
        text.append("\nTraining Performance:\n")
        text.append(f"  Loss: {perf_train.get('loss', np.nan):.4f} | MAE: {perf_train.get('mae', np.nan):.4f}\n")
        text.append("Validation Performance:\n")
        text.append(f"  Loss: {perf_val.get('loss', np.nan):.4f} | MAE: {perf_val.get('mae', np.nan):.4f}\n")

        # Direct (multi-output)
        text.append("\nDirect (multi-output) Performance:\n")
        text.append(
            f"  MAE: {perf_norm.get('mae', np.nan):.4f} | "
            f"MSE: {perf_norm.get('mse', np.nan):.4f} | "
            f"RMSE: {perf_norm.get('rmse', np.nan):.4f} | "
            f"R2: {perf_norm.get('r2', np.nan):.4f}\n"
        )

        # Multi-step (aggregate)
        rec = training_results.get("performance_recursive", {}) or {}
        text.append("\nMulti-Step Performance (aggregate):\n")
        text.append(
            f"  Steps: {forecast_steps} (~{HSEC:.1f} s) | "
            f"MAE: {rec.get('mae', np.nan):.4f} | "
            f"MSE: {rec.get('mse', np.nan):.4f} | "
            f"RMSE: {rec.get('rmse', np.nan):.4f} | "
            f"R2: {rec.get('r2', np.nan):.4f}\n"
        )
        text.append("\nMulti-Step (denormalized, all steps combined):\n")
        text.append(
            f"  MAE: {mae_den:.2f} | MSE: {mse_den:.2f} | RMSE: {rmse_den:.2f} | R2: {r2_den:.3f}\n"
        )

        text.append("Per-step (denormalized):\n")
        text.append(
            f"  Step 1 → MAE: {mae_first_den:.2f} | MSE: {mse_first_den:.2f} | "
            f"RMSE: {rmse_first_den:.2f} | R2: {r2_first_den:.3f}\n"
        )
        text.append(
            f"  Step {forecast_steps} → MAE: {mae_last_den:.2f} | MSE: {mse_last_den:.2f} | "
            f"RMSE: {rmse_last_den:.2f} | R2: {r2_last_den:.3f}\n"
        )

        text.append("\nBest Hyperparameters:\n")
        text.append(f"  Layers: {best_hps.get('num_layers')} | "
                    f"Activation: {best_hps.get('activation')} | Optimizer: {best_hps.get('optimizer')} | "
                    f"Dropout: {best_hps.get('dropout_rate')}\n")

        text.append("\nTiming:\n")
        text.append(f"  Train: {training_results.get('start_datetime_training')} → "
                    f"{training_results.get('end_datetime_training')} "
                    f"({(training_results.get('training_elapsed_time_seconds', 0))/60:.2f} min)\n")
        text.append(f"  Predict: {training_results.get('start_datetime_predictions')} → "
                    f"{training_results.get('end_datetime_predictions')} "
                    f"({(training_results.get('predictions_elapsed_time_seconds', 0))/60:.2f} min)\n")

        plt.text(0.01, 0.99, "".join(text), ha='left', va='top', wrap=True, fontsize=9)
        pdf.savefig(); plt.close()

        # Página 2 — Downsample meta (se houver)
        if ds_meta:
            tm = ds_meta.get("training", {})
            em = ds_meta.get("experiment", {})
            txt = []
            if isinstance(tm, dict) and tm:
                txt.append("Training dataset down-sample:\n")
                txt.append(f"  Rows before: {tm.get('orig_len')} | after: {tm.get('down_len')}\n\n")
            if isinstance(em, dict) and em:
                txt.append("Experiment dataset down-sample:\n")
                txt.append(f"  Rows before: {em.get('orig_len')} | after: {em.get('down_len')}\n")
            plt.figure(figsize=(8.5, 11)); plt.axis('off')
            plt.text(0.01, 0.99, "".join(txt), va='top', ha='left', family='monospace', wrap=True, fontsize=9)
            pdf.savefig(); plt.close()

        # Página 3 — First step no tempo
        plt.figure(figsize=(12, 6))
        plt.plot(df_first['time[s]'], df_first['serving_cell_rssi_1'], label="Actual RSSI", alpha=0.8)
        plt.plot(df_first['time[s]'], df_first['predicted_rssi'], label="Predicted RSSI (First Step)", alpha=0.8)
        plt.title("Actual vs Predicted RSSI (First Step)")
        plt.xlabel("Time [s]"); plt.ylabel("RSSI"); plt.legend(); plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # --- ajuste visual opcional por lag (não altera métricas) ---
        if VISUAL_SHIFT:
            # amostras por segundo (tente tirar do config; se não tiver, use 1/Δt)
            sps = float(SPS)  # já veio dos params (25 Hz)
            max_lag_samples = int(round(MAX_LAG_SEC * sps))
            best_lag = _best_lag_by_xcorr(y_true_last, y_pred_end, max_lag_samples)
            y_true_last_plot, y_pred_end_plot, t_last_plot = _apply_lag_for_plot(y_true_last, y_pred_end, t_last,
                                                                                 best_lag)
            # Tempo relativo para o gráfico (só visual)
            t_last_rel = t_last_plot - t_last_plot[0]

        else:
            y_true_last_plot, y_pred_end_plot, t_last_plot = y_true_last, y_pred_end, t_last

        # Página 4 — Last step no tempo (alinha início com offset)
        plt.figure(figsize=(12, 6))
        plt.plot(t_last_rel, y_true_last_plot, label="Actual RSSI (Last Step)", alpha=0.8)
        plt.plot(t_last_rel, y_pred_end_plot, label="Predicted RSSI (Last Step)", alpha=0.8)
        plt.title(f"Actual vs Predicted RSSI (Last Step – step {forecast_steps} ≈ {HSEC:.1f}s) "
                  f"[lag={best_lag} samples ≈ {best_lag / sps:.2f}s]")
        plt.xlabel("Time [s]");
        plt.ylabel("RSSI");
        plt.legend();
        plt.grid(True, ls='--', alpha=0.5)

        plt.tight_layout();
        pdf.savefig();
        plt.close()

        # Página 5 — Scatter First Step
        plt.figure(figsize=(7.5, 7.5))
        sns.scatterplot(x=y_true_first, y=y_pred_1step, alpha=0.6)
        lim = [min(y_true_first.min(), y_pred_1step.min()), max(y_true_first.max(), y_pred_1step.max())]
        plt.plot(lim, lim, 'r--'); plt.title("Actual vs Predicted (First Step)")
        plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Página 6 — Scatter Last Step
        plt.figure(figsize=(7.5, 7.5))
        sns.scatterplot(x=y_true_last, y=y_pred_end, alpha=0.6)
        lim = [min(y_true_last.min(), y_pred_end.min()), max(y_true_last.max(), y_pred_end.max())]
        plt.plot(lim, lim, 'r--'); plt.title(f"Actual vs Predicted (Last Step – step {forecast_steps})")
        plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Página 7 — Erros por passo (MAE/RMSE) vs tempo
        mae_per_step = np.array(rec_errors.get("mae", []))
        rmse_per_step = np.array(rec_errors.get("rmse", []))
        if mae_per_step.size and rmse_per_step.size:
            interval = 1.0 / float(SPS)
            time_axis = np.arange(1, forecast_steps + 1) * interval
            plt.figure(figsize=(10, 6))
            plt.plot(time_axis, mae_per_step, marker='o', label='MAE')
            plt.plot(time_axis, rmse_per_step, marker='x', label='RMSE')
            plt.title('Multi-Step Forecast Error by Time Ahead')
            plt.xlabel('Time Ahead [s]'); plt.ylabel('Error'); plt.legend(); plt.grid(True, ls='--', alpha=0.7)
            plt.tight_layout(); pdf.savefig(); plt.close()

        # Página 8 — Mapas espaciais (First e Last/multi)
        shared_vmin = min(y_pred_1step.min(), y_pred_end.min())
        shared_vmax = max(y_pred_1step.max(), y_pred_end.max())

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(df_first['position_x'], df_first['position_y'],
                         c=df_first['predicted_rssi'], cmap='viridis', s=50, alpha=0.8,
                         vmin=shared_vmin, vmax=shared_vmax)
        plt.colorbar(sc, label="Predicted RSSI (First Step)")
        plt.title("Predicted RSSI in Spatial Context (First Step)")
        plt.xlabel("Position X"); plt.ylabel("Position Y"); plt.grid(True, ls='--', alpha=0.4)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # multi-step (último passo)
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(pos_x_last, pos_y_last,
                         c=y_pred_end, cmap='viridis', s=50, alpha=0.8,
                         vmin=shared_vmin, vmax=shared_vmax)
        plt.colorbar(sc, label=f"Predicted RSSI (Multi-Step – step {forecast_steps})")
        plt.title(f"Predicted RSSI in Spatial Context (Multi-Step – {forecast_steps} steps ≈ {HSEC:.1f} s)")
        plt.xlabel("Position X"); plt.ylabel("Position Y"); plt.grid(True, ls='--', alpha=0.4)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Página 9 — Histogramas de erro
        errors_first = df_first['serving_cell_rssi_1'] - df_first['predicted_rssi']
        errors_last = y_true_last - y_pred_end
        plt.figure(figsize=(10, 6))
        sns.histplot(errors_first, bins=30, kde=True, alpha=0.7)
        plt.title("Distribution of Prediction Errors (First Step)")
        plt.xlabel("Error (Actual - Predicted)"); plt.ylabel("Frequency"); plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout(); pdf.savefig(); plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(errors_last, bins=30, kde=True, alpha=0.7)
        plt.title(f"Distribution of Prediction Errors (Last Step – step {forecast_steps})")
        plt.xlabel("Error (Actual - Predicted)"); plt.ylabel("Frequency"); plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout(); pdf.savefig(); plt.close()

    print(f"[INFO] Relatório gerado: {report_filename}")
    return report_filename


# def pick_experiment_folder():
#     base = "experiments"
#     options = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
#     options.sort()
#     if not options:
#         print("Nenhuma pasta em 'experiments/'.")
#         sys.exit(2)
#     print("\nEscolha um experimento:")
#     for i, d in enumerate(options, 1):
#         print(f"  {i:2d}: {d}")
#     sel = input("\nDigite o número: ").strip()
#     idx = int(sel) - 1
#     return os.path.join(base, options[idx])

def pick_folder_gui(start_dir="experiments"):
    """Abre um diálogo de pasta. Retorna caminho escolhido ou None."""
    if not TK_OK:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(
        initialdir=start_dir,
        title="Escolha a pasta do experimento (que contém os JSONs)"
    )
    try:
        root.destroy()
    except Exception:
        pass
    return path if path else None

def pick_experiment_folder(base="experiments"):
    """Fallback textual (lista subpastas de base)."""
    options = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    options.sort()
    if not options:
        print(f"Nenhuma pasta em '{base}/'.")
        sys.exit(2)
    print("\nEscolha um experimento:")
    for i, d in enumerate(options, 1):
        print(f"  {i:2d}: {d}")
    sel = input("\nDigite o número: ").strip()
    idx = int(sel) - 1
    return os.path.join(base, options[idx])

def find_latest_experiment(base="experiments"):
    latest_path, latest_mtime = None, -1
    for d in os.listdir(base):
        p = os.path.join(base, d)
        tr = os.path.join(p, "training_results.json")
        if os.path.isdir(p) and os.path.isfile(tr):
            mtime = os.path.getmtime(tr)
            if mtime > latest_mtime:
                latest_mtime, latest_path = mtime, p
    if not latest_path:
        raise RuntimeError(f"Nenhum training_results.json encontrado em '{base}'")
    return latest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", dest="exp", type=str, default="",
                        help="Caminho da pasta do experimento (experiments/...).")
    parser.add_argument("--all", action="store_true",
                        help="Gerar PDF para todas as pastas que ainda não têm PDF.")
    parser.add_argument("--gui", action="store_true",
                        help="Abrir seletor gráfico de pasta (tkinter).")
    parser.add_argument("--root", type=str, default="experiments",
                        help="Pasta base para procurar experiments (default: 'experiments').")
    parser.add_argument("--latest", action="store_true",
                        help="Autorun: pegar automaticamente o experimento mais recente (com JSONs).")
    parser.add_argument("--pick", action="store_true",
                        help="Abrir um seletor de pasta (GUI) para escolher o experimento.")
    parser.add_argument("--base", type=str, default="experiments",
                        help="Pasta base para --pick e menu textual (default: experiments)")
    args = parser.parse_args()
    if not any([args.pick, args.latest, args.all, args.exp]):
        args.pick = True

    if args.all:
        # processamento em lote (já existia)
        from glob import glob

        base = args.base
        folders = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        folders.sort()


        def has_pdf(path):
            return any(f.endswith(".pdf") for f in os.listdir(path))


        for f in folders:
            if not has_pdf(f):
                try:
                    print(f"[BATCH] Gerando PDF para: {f}")
                    generate_report_pdf(f)
                except Exception as e:
                    print(f"[BATCH] Falhou em {f}: {e}")
        sys.exit(0)

    # Novo: --latest escolhe automaticamente o mais recente
    if args.latest:
        exp_folder = find_latest_experiment(args.base)
        generate_report_pdf(exp_folder)
        sys.exit(0)

    # Novo: --pick abre GUI; se GUI indisponível, cai no menu textual
    if args.pick and not args.exp:
        chosen = pick_folder_gui(start_dir=args.base)
        if not chosen:
            chosen = pick_experiment_folder(base=args.base)
        generate_report_pdf(chosen)
        sys.exit(0)

    # Comportamento padrão (compat): usa --exp se veio, senão menu textual
    exp_folder = args.exp or pick_experiment_folder(base=args.base)
    generate_report_pdf(exp_folder)


