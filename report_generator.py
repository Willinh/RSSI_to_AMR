import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import import_dataset
import sys
import time
import tkinter as tk
from tkinter import filedialog

def print_progress_bar(iteration, total, prefix='', suffix='', length=30):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

def create_multistep_dataset(X, y, timesteps, forecast_steps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps - forecast_steps + 1):
        Xs.append(X[i:(i + timesteps)])
        ys.append(y[(i + timesteps):(i + timesteps + forecast_steps)])
    return np.array(Xs), np.array(ys)

# === SELECIONAR A PASTA DO EXPERIMENTO MANUALMENTE ===
root = tk.Tk()
root.withdraw()  # Oculta a janela principal

print("[INFO] Selecione a pasta do experimento para gerar o relatório:")
latest_experiment = filedialog.askdirectory(initialdir='experiments', title='Selecione a pasta do experimento')

if not latest_experiment:
    raise ValueError("Nenhuma pasta selecionada!")

print(f"[INFO] Carregando dados da pasta: {latest_experiment}")

# === CARREGAR CONFIGURAÇÕES ===
with open(os.path.join(latest_experiment, 'training_config.json'), 'r') as f:
    training_config = json.load(f)

with open(os.path.join(latest_experiment, 'best_hyperparameters.json'), 'r') as f:
    best_hps = json.load(f)

with open(os.path.join(latest_experiment, 'training_results.json'), 'r') as f:
    training_results = json.load(f)

# training_results['start_datetime_training'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# training_results['end_datetime_training'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# training_results['training_elapsed_time_seconds'] = 0
# training_results['start_datetime_predictions'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# training_results['end_datetime_predictions'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# training_results['predictions_elapsed_time_seconds'] = 0

dataset_info = training_results.get("dataset_name", {})
dataset_meta  = training_results.get("dataset_meta", {})

train_meta_txt = ""
if "training" in dataset_meta:
    tm = dataset_meta["training"]
    train_meta_txt += f"\nTraining dataset: {dataset_info.get('training')}\n"
    train_meta_txt += f"  Rows before: {tm.get('orig_len')}, after: {tm.get('down_len')}\n"

if "experiment" in dataset_meta:
    em = dataset_meta["experiment"]
    train_meta_txt += f"\nExperiment dataset: {dataset_info.get('experiment')}\n"
    train_meta_txt += f"  Rows before: {em.get('orig_len')}, after: {em.get('down_len')}\n"

dataset_params = training_results.get("dataset_params", {})
params_txt = "\nDataset Parameters:\n"
for phase, vals in dataset_params.items():
    params_txt += f"  {phase.capitalize()}: Δt≈{vals['deltaT_MS']:.1f} ms | samples/window={vals['samples_per_win']:.2f} | timesteps={vals['timesteps']}\n"

tuning_info = training_results.get("tuning_info", {})
tuning_txt = f"\nTuning:\n  Trials executed: {tuning_info.get('executed_trials')}/{tuning_info.get('max_trials')}\n"

perf_train = training_results.get("performance_training", {})
perf_val   = training_results.get("performance_validation", {})

perf_txt = f"""
Training Performance:
  Loss: {perf_train.get('loss', 'n/a'):.4f} | MAE: {perf_train.get('mae', 'n/a'):.4f}
Validation Performance:
  Loss: {perf_val.get('loss', 'n/a'):.4f} | MAE: {perf_val.get('mae', 'n/a'):.4f}
"""


model_type = training_results.get('model_type', 'Unknown')
dataset_name = training_config.get("dataset_name", "Unknown file")
# Ajusta para novo formato com dois datasets
if isinstance(dataset_name, dict):
    train_name = dataset_name.get("training", "Unknown training dataset")
    exp_name   = dataset_name.get("experiment", "Unknown experiment dataset")
    dataset_name_str = f"Training: {train_name} | Experiment: {exp_name}"
else:
    dataset_name_str = dataset_name
# === PREPARAR TESTE ===
# --- obter timesteps, seja em params ou em Sequencing -----------------
timesteps = (
    training_config.get('params', {}).get('timesteps') or
    training_config.get('Sequencing', {}).get('timesteps')
)
if timesteps is None:
    raise ValueError("timesteps não encontrado em training_config!")
forecast_steps = (
    training_config.get('forecast_steps') or
    training_results.get('forecast_steps')
)

ds_meta = training_results.get("dataset_meta", {})

# Se ainda veio None, infere pelo shape da matriz multi-step
if forecast_steps is None:
    ms_preds = training_results.get("performance_recursive", {})     \
                                .get("multi_step_predictions")
    if ms_preds is not None and len(ms_preds) > 0:
        forecast_steps = len(ms_preds[0])
    else:
        forecast_steps = 1     # fallback de segurança

# === CARREGAR DATASET ===
# scaler, train_df, val_df, test_df, DATASET_PATH, ds_meta = import_dataset()
# === DEFINIR CAMINHO DO DATASET DE EXPERIMENTO ===
dataset_info = training_results.get("dataset_name", {})
if isinstance(dataset_info, dict):
    train_name = dataset_info.get("training", "Unknown training dataset")
    experiment_dataset_name = dataset_info.get("experiment", None)
    dataset_name_str = f"Training: {train_name} | Experiment: {experiment_dataset_name or 'Unknown experiment dataset'}"
else:
    experiment_dataset_name = dataset_info
    dataset_name_str = dataset_info

if not experiment_dataset_name:
    raise ValueError("Nome do dataset de experimento não encontrado no training_results.json")

experiment_dataset_path = os.path.join("data", experiment_dataset_name)
# === CARREGAR APENAS O DATASET DE EXPERIMENTO ===
scaler, _, _, test_df, _, ds_meta = import_dataset(experiment_dataset_path, split_train_val_test=False)


# === CARREGAR MODELO ===
model = load_model(os.path.join(latest_experiment, f'{model_type.lower()}_model.keras'))

# === Refazer a previsão multi-step para o relatório ===
print("[INFO] Gerando previsões multi-step para o relatório...")

X_test, y_test = create_multistep_dataset(
    test_df.drop(columns=['time[s]']).values,
    test_df['serving_cell_rssi_1'].values,
    timesteps, forecast_steps
)

multi_step_predictions = []

total_steps = len(X_test)

for i in range(total_steps):
    input_sequence = X_test[i]
    input_sequence = input_sequence.reshape((1, input_sequence.shape[0], input_sequence.shape[1]))

    pred = model.predict(input_sequence, verbose=0)
    multi_step_predictions.append(pred.flatten())

    # Barra de progresso
    print_progress_bar(i + 1, total_steps, prefix='Progress', suffix='Complete')


multi_step_predictions = np.array(multi_step_predictions)

# transformar em matriz (N, forecast_steps)
multi_step_predictions = np.vstack(multi_step_predictions)

# === desnormalização --------------------------------------------
# 1. Localiza a coluna do RSSI
if hasattr(scaler, "feature_names_in_"):
    try:
        RSSI_IDX = list(scaler.feature_names_in_).index("serving_cell_rssi_1")
    except ValueError:
        RSSI_IDX = 0          # fallback seguro
else:
    RSSI_IDX = 0              # versões antigas do sklearn

# 2. usa min / max corretos
min_rssi = scaler.data_min_[RSSI_IDX]
max_rssi = scaler.data_max_[RSSI_IDX]
inverse_scale = lambda arr: arr * (max_rssi - min_rssi) + min_rssi

# decide qual passo quer traçar:
multi_step_predictions_descaled = inverse_scale(multi_step_predictions[:, -1])   # 1.º passo
#  ou  multi_step_predictions[:, -1]  → último passo

# === RSSI do ÚLTIMO passo (step = forecast_steps) ===
y_true_last  = inverse_scale(y_test[:, -1])          # shape (N,)
y_pred_last  = multi_step_predictions_descaled       # mesmo vetor (N,)

errors_last  = y_true_last - y_pred_last             # guardar para hist

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

predictions = model.predict(test_dataset)

# True RSSI
true_rssi = y_test[:,0]  # já está corretamente alinhado com predictions

predictions = predictions[:true_rssi.shape[0]]

# === DESESCALAR ===
# min_rssi = scaler.data_min_[0]
# max_rssi = scaler.data_max_[0]
#
# def inverse_scale(x):
#     return x * (max_rssi - min_rssi) + min_rssi

true_rssi_descaled = inverse_scale(true_rssi)
predicted_rssi_descaled = inverse_scale(predictions)

# === USAR APENAS O PRIMEIRO PASSO ===
y_true_first = true_rssi_descaled
y_pred_first = predicted_rssi_descaled[:, 0]

# OBTER TIMESTAMPS E POSIÇÕES
timestamps = test_df['time[s]'].values[timesteps:]
positions_x = test_df['position_x'].values[timesteps:]
positions_y = test_df['position_y'].values[timesteps:]

n_samples = y_true_first.shape[0]
timestamps = timestamps[:n_samples]
positions_x = positions_x[:n_samples]
positions_y = positions_y[:n_samples]

# CRIAR DATAFRAME
df_results = pd.DataFrame({
    'time[s]': timestamps,
    'serving_cell_rssi_1': y_true_first,
    'predicted_rssi': y_pred_first,
    'position_x': positions_x,
    'position_y': positions_y
})

errors = df_results['serving_cell_rssi_1'] - df_results['predicted_rssi']


# === GERAR RELATÓRIO PDF ===
report_filename = os.path.join(latest_experiment, f"{model_type.lower()}_training_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")

with PdfPages(report_filename) as pdf:
    # Resumo
    plt.figure(figsize=(8, 11))
    plt.axis('off')
    text = "Training Configuration and Summary:\n\n"

    text += f"Dataset used: {dataset_name_str}\n\n"
    # for section, params in training_config.items():
    #     text += f"{section}:\n"
    #     for key, value in params.items():
    #         text += f"  - {key}: {value}\n"
    #     text += "\n"
    params = training_config.get("params", {})

    SPS = params.get("samples_per_sec", 1)  # fallback 1
    HSEC = params.get("forecast_horizon_sec",
                      params.get("forecast_steps", 1) / SPS)

    # se vier como string, tenta converter de JSON
    if isinstance(params, str):
        try:
            import json
            params = json.loads(params)
        except Exception:
            params = {"note": params}  # guarda como texto simples

    # agora é seguro iterar
    for key, value in params.items():
        text += f"{key}: {value}\n"

    text += train_meta_txt
    text += params_txt
    text += tuning_txt
    text += perf_txt

    text += f"""
    # {model_type} Multi-Step Training Report

    Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Training Start: {training_results['start_datetime_training']}
    Training End: {training_results['end_datetime_training']}
    Training Duration: {training_results['training_elapsed_time_seconds'] / 60:.2f} minutes
    
    Predictions Start: {training_results['start_datetime_predictions']}
    Predictions End: {training_results['end_datetime_predictions']}
    Predictions Duration: {training_results['predictions_elapsed_time_seconds'] / 60:.2f} minutes

    Best Hyperparameters:
    Number of Layers: {best_hps.get('num_layers')}
    Activation: {best_hps.get('activation')}
    Optimizer: {best_hps.get('optimizer')}
    Dropout Rate: {best_hps.get('dropout_rate'):.2f}

    MAE: {training_results['performance_normal']['mae']:.4f}
    MSE: {training_results['performance_normal']['mse']:.4f}
    RMSE: {training_results['performance_normal']['rmse']:.4f}
    R2 Score: {training_results['performance_normal']['r2']:.4f}
    
    Multi-Step Forecast:
    Forecast Steps: {training_results['forecast_steps']}
    MAE: {training_results['performance_recursive']['mae']:.4f}
    MSE: {training_results['performance_recursive']['mse']:.4f}
    RMSE: {training_results['performance_recursive']['rmse']:.4f}
    R2 Score: {training_results['performance_recursive']['r2']:.4f}
    """

    plt.text(0.01, 0.99, text, ha='left', va='top', wrap=True, fontsize=9)
    pdf.savefig()
    plt.close()

    # -------- Página 2: resumo do down-sample -----------------
    if ds_meta:
        txt = (
                f"Dataset size before down-sample: {ds_meta['orig_len']:,} rows\n"
                f"Dataset size after  down-sample: {ds_meta['down_len']:,} rows\n\n"
                "First 5 timestamps BEFORE:\n"
                + "\n".join(str(t) for t in ds_meta['first5_before'])
                + "\n...\n"
                  "Last 5 timestamps  BEFORE:\n"
                + "\n".join(str(t) for t in ds_meta['last5_before'])
                + "\n\n"
                  "First 5 timestamps AFTER:\n"
                + "\n".join(str(t) for t in ds_meta['first5_after'])
                + "\n...\n"
                  "Last 5 timestamps  AFTER:\n"
                + "\n".join(str(t) for t in ds_meta['last5_after'])
        )

        plt.figure(figsize=(8, 11))
        plt.axis('off')
        plt.text(0.01, 0.99, txt, va='top', ha='left', family='monospace', wrap=True, fontsize=9)
        pdf.savefig()
        plt.close()

    # Gráficos
    plt.figure(figsize=(12, 6))
    plt.plot(df_results['time[s]'], df_results['serving_cell_rssi_1'], label="Actual RSSI", color='blue', alpha=0.7)
    plt.plot(df_results['time[s]'], df_results['predicted_rssi'], label="Predicted RSSI", color='orange', alpha=0.7)
    plt.title("Actual vs Predicted RSSI (First Step)")
    plt.xlabel("Time [s]")
    plt.ylabel("RSSI")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # === Comparação Previsão Actual vs Multi-Step ===
    plt.figure(figsize=(12, 6))

    # RSSI real (do primeiro passo)
    plt.plot(timestamps[:len(y_true_first)], y_true_first, label="Actual RSSI", color='blue', alpha=0.7)
    # Previsão com o modelo multi-step (usando várias janelas)
    multi_step_timestamps = timestamps[:len(multi_step_predictions_descaled)]
    plt.plot(multi_step_timestamps, multi_step_predictions_descaled,
             label="Predicted RSSI (Multi-Step)", color='green', alpha=0.7)

    # # Previsão tradicional (first step)
    # plt.plot(timestamps[:len(y_pred_first)], y_pred_first, label="Predicted RSSI (First Step)", color='red',
    #          alpha=0.7)

    plt.title(f"Actual vs Predicted RSSI (Multi-Step (step {forecast_steps}" f"≈ {HSEC:.1f} s)")
    plt.xlabel("Time [s]")
    plt.ylabel("RSSI")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=df_results['serving_cell_rssi_1'], y=df_results['predicted_rssi'], alpha=0.6)
    plt.plot([
        df_results['serving_cell_rssi_1'].min(), df_results['serving_cell_rssi_1'].max()
    ], [
        df_results['serving_cell_rssi_1'].min(), df_results['serving_cell_rssi_1'].max()
    ], color='red', linestyle='--')
    plt.title("Actual vs Predicted RSSI (First Step)")
    plt.xlabel("Actual RSSI")
    plt.ylabel("Predicted RSSI")
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # === Scatter Last-Step ==================================
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true_last, y=y_pred_last, alpha=0.6)
    plt.plot([y_true_last.min(), y_true_last.max()],
             [y_true_last.min(), y_true_last.max()],
             color='red', linestyle='--')
    plt.title(f"Actual vs Predicted RSSI (Last Step – step {forecast_steps}" f"≈ {HSEC:.1f} s)")
    plt.xlabel("Actual RSSI")
    plt.ylabel("Predicted RSSI")
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # === Página: Actual vs Predicted RSSI (Multi-Step) ===
    if 'multi_step_predictions' in training_results:
        y_true_multi = np.array(training_results['multi_step_true'])
        y_pred_multi = np.array(training_results['multi_step_predictions'])

    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple', alpha=0.7)
    plt.title("Distribution of Prediction Errors (First Step)")
    plt.xlabel("Error (Actual RSSI - Predicted RSSI)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # === Histograma de erro – último passo ==================
    plt.figure(figsize=(10, 6))
    sns.histplot(errors_last, bins=30, kde=True, color='teal', alpha=0.7)
    plt.title(f"Distribution of Prediction Errors (Last Step – step {forecast_steps}" f"≈ {HSEC:.1f} s)")
    plt.xlabel("Error (Actual RSSI - Predicted RSSI)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# === Predicted RSSI in Spatial Context (First Step) block ===
    multi_step_preds = training_results.get("performance_recursive", {}).get("multi_step_predictions", None)
    multi_mat = np.asarray(multi_step_preds)
    y_pred_multi = inverse_scale(multi_mat[:, -1])
    # y_pred_multi = multi_mat[:, 0]           # (ou use esta linha p/ 1.º passo)
    # y_pred_multi = multi_mat.mean(axis=1)    # (ou média)
    shared_vmin = min(y_pred_first.min(), y_pred_multi.min())
    shared_vmax = max(y_pred_first.max(), y_pred_multi.max())

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df_results['position_x'], df_results['position_y'], c=df_results['predicted_rssi'],
                     cmap='viridis', s=50, alpha=0.8,vmin=shared_vmin, vmax=shared_vmax)
    plt.colorbar(sc, label="Predicted RSSI")
    plt.title("Predicted RSSI in Spatial Context (First Step)")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    # === Página: Predicted RSSI in Spatial Context (Multi-Step) ===
    if multi_step_preds is not None:
        y_pred_multi = y_pred_multi[:len(df_results)]  # GARANTIR MESMO COMPRIMENTO DOS EIXOS
        position_x = df_results['position_x'][:len(y_pred_multi)]
        position_y = df_results['position_y'][:len(y_pred_multi)]
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(position_x, position_y,
                         c=y_pred_multi, cmap='viridis', s=50, alpha=0.8, vmin=shared_vmin, vmax=shared_vmax)
        plt.colorbar(sc, label="Predicted RSSI (Multi-Step)")
        chosen_step = forecast_steps
        plt.title(f"Predicted RSSI in Spatial Context (Multi-Step - step {chosen_step}" f"≈ {HSEC:.1f} s)")
        plt.xlabel("Position X")
        plt.ylabel("Position Y")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # === Erro por passo de previsão multi-step ===
    forecast_steps = training_results['forecast_steps']
    recursive_errors = training_results['performance_recursive']['recursive_errors_per_step']

    plt.figure(figsize=(10, 6))
    steps = list(range(1, forecast_steps + 1))
    SPS = training_config.get("params", {}).get("samples_per_sec", 1)
    interval = 1.0 / SPS
    time_axis = np.arange(1, forecast_steps + 1) * interval
    plt.plot(time_axis, recursive_errors['mae'], marker='o', label='MAE')
    plt.plot(time_axis, recursive_errors['rmse'], marker='x', label='RMSE')
    plt.title('Multi-Step Forecast Error by Time Ahead')
    plt.xlabel('Time Ahead [S]')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # === Comparação Previsão 1-Step vs Multi-Step ===
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps[:len(true_rssi)], true_rssi, label="Actual RSSI", color='blue', alpha=0.7)
    plt.plot(timestamps[:len(y_pred_first)], y_pred_first, label="Predicted RSSI (First Step)", color='orange',
             alpha=0.7)
    # plt.plot(timestamps[-len(multi_step_predictions_descaled):], multi_step_predictions_descaled,
    #          label="Predicted RSSI (Multi-Step)", color='green', alpha=0.7)
    offset = forecast_steps - 1

    # Limitar comprimento para evitar erro
    max_len = min(len(timestamps) - offset, len(multi_step_predictions_descaled))

    plt.plot(
        timestamps[offset:offset + max_len],
        multi_step_predictions_descaled[:max_len],
        label="Predicted RSSI (Multi-Step)",
        color='green', alpha=0.7
    )


    # === Novo resumo automático 2.0 =======================================
    recursive_errors   = training_results['performance_recursive']['recursive_errors_per_step']
    mae_per_step       = np.array(recursive_errors['mae'])
    rmse_per_step      = np.array(recursive_errors['rmse'])

    step_1_mae,  step_1_rmse  = mae_per_step[0],  rmse_per_step[0]
    step_m_mae,  step_m_rmse  = np.median(mae_per_step), np.median(rmse_per_step)
    step_N_mae,  step_N_rmse  = mae_per_step[-1], rmse_per_step[-1]

    # crescimento percentual do 1.º ao último passo
    growth_pct = 100 * (step_N_mae - step_1_mae) / max(step_1_mae, 1e-6)

    # flag de estabilidade
    if growth_pct < 20:
        stability_flag = "Estável (erro cresce < 20 %)"
    elif growth_pct < 50:
        stability_flag = "Moderado (erro cresce 20-50 %)"
    else:
        stability_flag = "Degrada Rápido (> 50 %)"

    # 3 piores horizontes
    worst_idx = mae_per_step.argsort()[-3:][::-1] + 1   # +1 p/ passo humano
    worst_str = ", ".join(f"{i} ({mae_per_step[i-1]:.2f})" for i in worst_idx)

    # recomendação simples
    if growth_pct < 20 and step_N_mae < 2:
        reco = "O modelo é adequado para decisões de hand-over até o horizonte máximo configurado."
    elif growth_pct < 50:
        reco = "Use as previsões até a metade do horizonte; após isso a incerteza cresce consideravelmente."
    else:
        reco = "Considere aumentar o histórico de entrada ou usar um modelo mais profundo para melhorar multi-step."

    auto_summary = f"""
        Resumo Automático
        -----------------
        Dataset:        {dataset_name_str}
        Modelo:         {model_type}
        Forecast steps: {forecast_steps}
    
        MAE / RMSE (dB)
            • 1º passo : {step_1_mae:.2f} / {step_1_rmse:.2f}
            • Mediana  : {step_m_mae:.2f} / {step_m_rmse:.2f}
            • {forecast_steps}º passo : {step_N_mae:.2f} / {step_N_rmse:.2f}
    
        Crescimento do erro: {growth_pct:.1f}%   →   {stability_flag}
    
        3 piores passos (MAE): {worst_str}
    
        Recomendação:
        {reco}
    """
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.01, 0.99, auto_summary, ha='left', va='top',
             wrap=True, fontsize=11, family="monospace")
    pdf.savefig()
    plt.close()
    # ======================================================================

print(f"[INFO] Relatório gerado: {report_filename}")