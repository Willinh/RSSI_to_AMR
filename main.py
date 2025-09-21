import gc
import json
import os
import shutil
import time
from datetime import datetime
from typing import Any

import keras_tuner as kt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras_tuner.src.backend import keras
from numpy import ndarray, dtype
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import import_dataset, calc_dataset_params, print_progress_bar, GROUP_MSEC, TRAINING_DATASET_PATH, \
    EXPERIMENT_DATASET_PATH

# ========================== CONFIGURAÇÕES ==========================

MODEL_TYPE = 'GRU'   # 'LSTM' ou 'GRU' ou 'BiLSTM' ou 'BiGRU' ou 'SimpleRNN'
# Sequenciamento dos dados
TIMESTEPS_ORIG = 60   # Número de passos de tempo para olhar para trás (default = 60)
BATCH_SIZE = 8192      # Tamanho do batch durante o treino

# Hiperparâmetros do Modelo
MIN_UNITS = 16      # Número mínimo de unidades por camada
MAX_UNITS = 256     # Número máximo de unidades por camada
UNITS_STEP = 16       # Passo de incremento de unidades testadas
DROPOUT_MIN = 0.2     # Dropout mínimo considerado
DROPOUT_MAX = 0.3     # Dropout máximo considerado
DROPOUT_STEP = 0.1    # Incremento do dropout considerado

# Funções e Otimizadores
ACTIVATION_FUNCTIONS = ['tanh', 'relu', 'elu']  # Funções de ativação possíveis: ['tanh', 'relu', 'elu']
OPTIMIZERS = ['adam', 'rmsprop', 'Nadam']   # Otimizadores possíveis: ['adam', 'rmsprop', 'Nadam']

# Estrutura da Rede
MIN_LAYERS = 1        # Número mínimo de camadas
MAX_LAYERS = 2        # Número máximo de camadas [default = 2]

# Tuning
MAX_EPOCHS = 50   # Número máximo de épocas no treinamento (20, 50, 100, 150)
HYPERBAND_ITERATIONS = 3  # Número de iterações no Hyperband Tuner

# Previsão Multi-Step
FORECAST_HORIZON_SEC = 4          # quero prever X segundos à frente

# ------------------ derivar parâmetros dependentes -------------------------
#   amostras por segundo depois do down-sample
SAMPLES_PER_SEC = max(1, 1000 // GROUP_MSEC)      # int; p.ex. 1000/100 = 10 Hz → 10

FORECAST_STEPS = max(1, int(FORECAST_HORIZON_SEC * SAMPLES_PER_SEC))
TIMESTEPS      = max(1, TIMESTEPS_ORIG // (GROUP_MSEC or 1))

# Organização de Pastas
FORCE_RESTART = True  # Se True, apaga diretório do tuner antes de novo teste
TUNER_DIRECTORY = 'my_dir'
PROJECT_NAME = f'{MODEL_TYPE.lower()}_hyperparameter_tuning'
full_path = os.path.join(TUNER_DIRECTORY, PROJECT_NAME)

# ====================================================================

# deltaT_MS = estimate_dt_ms(DATASET_PATH)           # e.g. 95.4 ms
# samples_per_win = max(1, GROUP_MSEC / deltaT_MS)
# TIMESTEPS = max(1, int(TIMESTEPS_ORIG / samples_per_win))
# print(f"Δt≈{deltaT_MS:.1f} ms  ·  samples/window≈{samples_per_win:.2f}  "
#       f"→  TIMESTEPS={TIMESTEPS}")

# Calcula parâmetros para o dataset de treino
TIMESTEPS_train: int
deltaT_MS_train, samples_per_win_train, TIMESTEPS_train = calc_dataset_params(
    TRAINING_DATASET_PATH, TIMESTEPS_ORIG, GROUP_MSEC
)

# Calcula parâmetros para o dataset de teste
deltaT_MS_test, samples_per_win_test, TIMESTEPS_test = calc_dataset_params(
    EXPERIMENT_DATASET_PATH, TIMESTEPS_ORIG, GROUP_MSEC
)

# Para o treinamento e tuning, usamos o TIMESTEPS do dataset de treino
TIMESTEPS = TIMESTEPS_train
SAMPLES_PER_SEC = max(1, 1000 // GROUP_MSEC)
FORECAST_STEPS = max(1, int(FORECAST_HORIZON_SEC * SAMPLES_PER_SEC))

PARAMS = {
    "model_type": MODEL_TYPE,
    "group_sec":  GROUP_MSEC,
    # "roll_window": ROLL_WINDOW,
    "timesteps_orig": TIMESTEPS_ORIG,
    "batch_size": BATCH_SIZE,
    "min_units": MIN_UNITS,
    "max_units": MAX_UNITS,
    "units_step": UNITS_STEP,
    "dropout_min": DROPOUT_MIN,
    "dropout_max": DROPOUT_MAX,
    "dropout_step": DROPOUT_STEP,
    "activation_functions": ACTIVATION_FUNCTIONS,
    "optimizers": OPTIMIZERS,
    "min_layers": MIN_LAYERS,
    "max_layers": MAX_LAYERS,
    "max_epochs": MAX_EPOCHS,
    "hyperband_iterations": HYPERBAND_ITERATIONS,
    "group_msec": GROUP_MSEC,
    "samples_per_sec": SAMPLES_PER_SEC,
    "forecast_horizon_sec": FORECAST_HORIZON_SEC,
    "forecast_steps": FORECAST_STEPS,        # continua gravando também em steps
    "timesteps": TIMESTEPS
}


# === Função para criar dataset multi-step ===
def create_multistep_dataset(X, y, timesteps, forecast_steps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps - forecast_steps + 1):
        Xs.append(X[i:(i + timesteps)])
        ys.append(y[(i + timesteps):(i + timesteps + forecast_steps)])
    return np.array(Xs), np.array(ys)

# === IMPORTAR DATASET ===
# scaler, train_df, val_df, test_df, DATASET_NAME, DS_META = import_dataset()

# alteração para importar datasets para treino e teste separados
# from utils import
# Importa dataset para treino
scaler, train_df, val_df, _, TRAIN_DATASET_NAME, TRAIN_META = import_dataset(TRAINING_DATASET_PATH, split_train_val_test=True)
# Importa dataset para experimento (somente teste)
_, _, _, test_df, EXP_DATASET_NAME, TEST_META = import_dataset(EXPERIMENT_DATASET_PATH, split_train_val_test=False)

print(f"Linhas de treino: {len(train_df)}  |  preciso de ≥ {TIMESTEPS + FORECAST_STEPS}")

# === CRIAR CONFIG PARA SALVAR JSON===
config = {
    "training_dataset": TRAIN_DATASET_NAME,
    "experiment_dataset": EXP_DATASET_NAME,
    "forecast_steps": FORECAST_STEPS,
    "params": PARAMS
}


# === APAGAR DIRETÓRIO DO TUNER SE NECESSÁRIO ===
if FORCE_RESTART and os.path.exists(full_path):
    print(f"[INFO] Apagando a pasta '{full_path}' para novo tuning...")
    shutil.rmtree(full_path, ignore_errors=True)


# === CRIAR DATASETS ===
# -- Dataset de treino --
X_train, y_train = create_multistep_dataset(
    train_df.drop(columns=['time[s]']).values,
    train_df['serving_cell_rssi_1'].values,
    TIMESTEPS, FORECAST_STEPS
)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)

# -- Dataset de validação --
X_val, y_val = create_multistep_dataset(
    val_df.drop(columns=['time[s]']).values,
    val_df['serving_cell_rssi_1'].values,
    TIMESTEPS, FORECAST_STEPS
)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

# === DEFINIR MODELO ===
def build_model(hp):
    if MODEL_TYPE == 'LSTM':
        RNNLayer = keras.layers.LSTM
    elif MODEL_TYPE == 'GRU':
        RNNLayer = keras.layers.GRU
    elif MODEL_TYPE == 'SimpleRNN':
        RNNLayer = keras.layers.SimpleRNN
    elif MODEL_TYPE == 'BiLSTM':
        def RNNLayer(*args, **kwargs):
            return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(*args, **kwargs))
    elif MODEL_TYPE == 'BiGRU':
        def RNNLayer(*args, **kwargs):
            return tf.keras.layers.Bidirectional(tf.keras.layers.GRU(*args, **kwargs))
    else:
        raise ValueError("Modelo desconhecido")

    model = tf.keras.models.Sequential()
    for i in range(hp.Int('num_layers', MIN_LAYERS, MAX_LAYERS)):
        model.add(RNNLayer(
            units=hp.Int(f'units_{i}', min_value=MIN_UNITS, max_value=MAX_UNITS, step=UNITS_STEP),
            activation=hp.Choice('activation', values=ACTIVATION_FUNCTIONS),
            return_sequences=True if i < hp.get('num_layers') - 1 else False,
            input_shape=(TIMESTEPS, X_train.shape[2]) if i == 0 else None
        ))
        model.add(tf.keras.layers.Dropout(
            hp.Float('dropout_rate', DROPOUT_MIN, DROPOUT_MAX, step=DROPOUT_STEP)
        ))

    model.add(tf.keras.layers.Dense(FORECAST_STEPS))

    model.compile(
        optimizer=hp.Choice('optimizer', values=OPTIMIZERS),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model

# === INICIAR TUNER ===
# tuner = kt.Hyperband(
#     build_model,
#     objective='val_loss',
#     max_epochs=MAX_EPOCHS,
#     hyperband_iterations=HYPERBAND_ITERATIONS,
#     directory=TUNER_DIRECTORY,
#     project_name=f'{MODEL_TYPE.lower()}_hyperparameter_tuning'
# )
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # máximo de modelos diferentes
    executions_per_trial=1,
    directory=TUNER_DIRECTORY,
    project_name=PROJECT_NAME
)


# === INICIAR CONTAGEM DE TEMPO ===
start_time_training = time.time()
start_datetime_training = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# === EXECUTAR TUNING ===
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

tuner.search(
    train_dataset,
    epochs=MAX_EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stop]
    )

executed_trials = len(tuner.oracle.trials)
max_trials = tuner.oracle.max_trials or executed_trials  # se max_trials for None, usa trials executados
progress = (executed_trials / max_trials) * 100

print(f"[INFO] Tuning concluído: {executed_trials}/{max_trials} trials executados ({progress:.1f}%)")



# === PEGAR MELHOR MODELO ===
K.clear_session()
gc.collect()

# Após o tuning com o Keras Tuner
# best_trials = tuner.get_best_trials(num_trials=10)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, epochs=MAX_EPOCHS, validation_data=val_dataset)

# Para obter as métricas (scores), você precisa avaliar esses modelos ou acessar pelo histórico do tuner:
# best_trials  = tuner.get_best_trials(num_trials=10)
best_trials  = tuner.oracle.get_best_trials(num_trials=10)
best_models = []
for t in best_trials:
    # Reconstrói o modelo a partir dos hiperparâmetros do trial,
    # do zero (sem carregar pesos).
    m = tuner.hypermodel.build(t.hyperparameters)
    best_models.append(m)



# === PREPARAR TESTE ===
y_test: ndarray[Any, dtype[Any]]
X_test, y_test = create_multistep_dataset(
    test_df.drop(columns=['time[s]']).values,
    test_df['serving_cell_rssi_1'].values,
    TIMESTEPS, FORECAST_STEPS
)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# === PREVISÃO ===
predictions = model.predict(test_dataset)

# === DESESCALAR ===

# === DESESCALAR (versão robusta) ==================================
if hasattr(scaler, "feature_names_in_"):
    RSSI_IDX = list(scaler.feature_names_in_).index("serving_cell_rssi_1")
else:                                 # fallback para versões antigas
    RSSI_IDX = 0                      # ajuste aqui se a posição for outra

min_rssi = scaler.data_min_[RSSI_IDX]
max_rssi = scaler.data_max_[RSSI_IDX]

inverse_scale = lambda x: x * (max_rssi - min_rssi) + min_rssi

true_rssi_descaled = inverse_scale(y_test)
predicted_rssi_descaled = inverse_scale(predictions)


# === AVALIAR ===
mae = mean_absolute_error(true_rssi_descaled, predicted_rssi_descaled)
mse = mean_squared_error(true_rssi_descaled, predicted_rssi_descaled)
rmse = np.sqrt(mse)
r2 = r2_score(true_rssi_descaled, predicted_rssi_descaled)

end_time_training = time.time()
end_datetime_training = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
elapsed_time_training = end_time_training - start_time_training



# === CRIAR PASTA DO EXPERIMENTO ===
report_base_name = f"{MODEL_TYPE.lower()}_training_report_{start_datetime_training}"
experiment_folder = os.path.join('experiments', report_base_name)
os.makedirs(experiment_folder, exist_ok=True)

print(f"[INFO] Pasta do experimento criada: {experiment_folder}")

# === SALVAR TUDO ===
model.save(os.path.join(experiment_folder, f'{MODEL_TYPE.lower()}_model.keras'), include_optimizer=False)

with open(os.path.join(experiment_folder, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=4)

best_hps_dict = {
    'num_layers': best_hps.get('num_layers'),
    'activation': best_hps.get('activation'),
    'optimizer': best_hps.get('optimizer'),
    'dropout_rate': best_hps.get('dropout_rate')
}
for i in range(best_hps.get('num_layers')):
    best_hps_dict[f'units_layer_{i+1}'] = best_hps.get(f'units_{i}')

with open(os.path.join(experiment_folder, 'best_hyperparameters.json'), 'w') as f:
    json.dump(best_hps_dict, f, indent=4)

# === PREVISÃO MULTI-STEP ===

start_time_predictions = time.time()
start_datetime_predictions = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print(f"[INFO] Realizando previsão multi-step ({FORECAST_STEPS} steps) ...")

recursive_predictions = []
true_future_values = []
total_steps = len(X_test) - FORECAST_STEPS
start_multi_step_time = time.time()

for i in range(total_steps):
    input_sequence = X_test[i]
    input_sequence = input_sequence.reshape((1, TIMESTEPS, input_sequence.shape[1]))

    pred = model.predict(input_sequence, verbose=0)
    recursive_predictions.append(pred.flatten()[0])

    true_value = y_test[i][0]
    true_future_values.append(true_value)

    # Atualizar barra de progresso
    print_progress_bar(i + 1, total_steps, start_multi_step_time)

# Previsão vetorial e Avaliação Direta
preds_full = model.predict(X_test, verbose=0)

mae_full = np.mean(np.abs(preds_full - y_test))
mse_full = np.mean(np.square(preds_full - y_test))
rmse_full = np.sqrt(mse_full)

# erros por horizonte (0-39)
errors_per_step = {
    "mae": np.mean(np.abs(preds_full - y_test), axis=0).tolist(),
    "mse": np.mean(np.square(preds_full - y_test), axis=0).tolist(),
    "rmse": np.sqrt(np.mean(np.square(preds_full - y_test), axis=0)).tolist(),
    "r2": [r2_score(y_test[:,k], preds_full[:,k]) for k in range(FORECAST_STEPS)],
    }

print()  # pular linha após finalização da barra


recursive_predictions = np.array(recursive_predictions)
true_future_values = np.array(true_future_values)

# === CALCULAR MÉTRICAS PARA MULTI-STEP ===
recursive_mae = mean_absolute_error(true_future_values, recursive_predictions)
recursive_mse = mean_squared_error(true_future_values, recursive_predictions)
recursive_rmse = np.sqrt(recursive_mse)
recursive_r2 = r2_score(true_future_values, recursive_predictions)

print(f"[INFO] Multi-step MAE: {recursive_mae:.4f}, RMSE: {recursive_rmse:.4f}")

# === ERROS POR STEP (média) ===
recursive_errors_per_step = {
    "mae": [],
    "mse": [],
    "rmse": [],
    "r2": []
}

# Calcula o erro MÉDIO de cada passo à frente
for step in range(FORECAST_STEPS):
    y_true_step = y_test[:, step]
    y_pred_step = predictions[:, step]

    mae_step = mean_absolute_error(y_true_step, y_pred_step)
    mse_step = mean_squared_error(y_true_step, y_pred_step)
    rmse_step = np.sqrt(mse_step)
    r2_step = r2_score(y_true_step, y_pred_step)

    recursive_errors_per_step["mae"].append(mae_step)
    recursive_errors_per_step["mse"].append(mse_step)
    recursive_errors_per_step["rmse"].append(rmse_step)
    recursive_errors_per_step["r2"].append(r2_step)



end_time_predictions = time.time()
end_datetime_predictions = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
elapsed_time_predictions = end_time_predictions - start_time_predictions

results = {
    "model_type": MODEL_TYPE,
    "start_datetime_training": start_datetime_training,
    "end_datetime_training": end_datetime_training,
    "training_elapsed_time_seconds": elapsed_time_training,
    "start_datetime_predictions": start_datetime_predictions,
    "end_datetime_predictions": end_datetime_predictions,
    "predictions_elapsed_time_seconds": elapsed_time_predictions,
    "best_hyperparameters": {
        "num_layers": best_hps.get('num_layers'),
        "activation": best_hps.get('activation'),
        "optimizer": best_hps.get('optimizer'),
        "dropout_rate": best_hps.get('dropout_rate'),
    },
    "dataset_params":{
        "training": {
            "deltaT_MS": deltaT_MS_train,
            "samples_per_win": samples_per_win_train,
            "timesteps": TIMESTEPS_train
        },
        "experiment": {
            "deltaT_MS": deltaT_MS_test,
            "samples_per_win": samples_per_win_test,
            "timesteps": TIMESTEPS_test
        }
    },
    "forecast_steps": FORECAST_STEPS,
    "performance_normal": {  # <<<
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    },
    "performance_recursive": {
        "mae": recursive_mae,
        "mse": recursive_mse,
        "rmse": recursive_rmse,
        "r2": recursive_r2,
        "recursive_errors_per_step": recursive_errors_per_step,
        "dataset_meta":{
            "training": TRAIN_META,
            "experiment": TEST_META
        }
    }
}


results["tuning_info"] = {
    "executed_trials": executed_trials,
    "max_trials": max_trials,
    "stop_reason": tuner.oracle.get_best_trials()[0].status if tuner.oracle.get_best_trials() else None
}

results["best_hyperparameters_full"] = best_hps.values

train_loss = history.history['loss'][-1]
train_mae  = history.history['mae'][-1]

val_loss = history.history['val_loss'][-1]
val_mae  = history.history['val_mae'][-1]

results["performance_training"] = {
    "loss": train_loss,
    "mae": train_mae
}
results["performance_validation"] = {
    "loss": val_loss,
    "mae": val_mae
}

# === SALVAR PREDIÇÕES MULTI-STEP PARA O RELATÓRIO ===
results["performance_recursive"]["multi_step_predictions"] = preds_full.tolist()
results["performance_recursive"]["multi_step_true"] = y_test.tolist()
results["performance_recursive"]["recursive_errors_per_step"] = errors_per_step

results["dataset_name"] = {
    "training": TRAIN_DATASET_NAME,
    "experiment": EXP_DATASET_NAME
}
with open(os.path.join(experiment_folder, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print(f"[INFO] Treinamento finalizado e arquivos salvos em {experiment_folder}")

# Envio de notificação
from pushbullet import Pushbullet
import traceback

try:
    # Notificação Push bullet
    # PUSHBULLET_TOKEN = 'o.5COmkZvYUTzgSp9KBSNrj2PvC1cQFMV9'
    API_KEY = os.getenv('PUSHBULLET_TOKEN')
    if not API_KEY:
        raise ValueError("Token do Pushbullet não encontrado. Defina a variável de ambiente 'PUSHBULLET_TOKEN'.")

    pb = Pushbullet(API_KEY)

    training_duration = round(elapsed_time_training / 60, 2)
    experiment_duration = round(elapsed_time_predictions / 60, 2)
    mensagem = f"Treinamento finalizados em {training_duration} minutos e Experimentos finalizados em {experiment_duration} minutos."
    pb.push_note("✅ Experimento finalizado", mensagem)

except Exception as e:
    print(f"❌ Erro ao enviar notificação: {e}")
traceback.print_exc()
