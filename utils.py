from tkinter.constants import NUMERIC
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys

TRAINING_DATASET_PATH = r'C:\\Users\\admpdi\\OneDrive - ISI SIM\\Documents\\GitHub\\RSSI-to-AMR\\data\\rssi_odom_dataset_11apr25_isolado.parquet'
EXPERIMENT_DATASET_PATH = r'C:\\Users\\admpdi\\OneDrive - ISI SIM\\Documents\\GitHub\\RSSI-to-AMR\\data\\rssi_odom_dataset_12apr25_isolado.parquet'
# DATASET_PATH = r'C:\\Users\\admpdi\\OneDrive - ISI SIM\\Documents\\GitHub\\RSSI-to-AMR\\data\\rssi_odom_dataset_12apr25_ajust.parquet'
GROUP_MSEC = 80               # agrupa timesteps por valor de milissegundo informado
# TX_POS = (-1.0, 1.0, 0.0)          # <<< coordenadas do AP em metros  (ajuste!)

def import_raw_data(file_path: str, debug: bool = False):
    df = pd.read_parquet(file_path, engine="fastparquet")

    meta = {
        "orig_len": len(df),
        "first5_before": df['time[s]'].head(5).tolist(),
        "last5_before": df['time[s]'].tail(5).tolist()
    }

    # — limpeza numérica mínima —
    for c in ['serving_cell_rssi_1', 'serving_cell_snr_1',
              'position_x', 'position_y', 'position_z', 'time[s]']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = (df.dropna(subset=['time[s]', 'serving_cell_rssi_1'])
             .sort_values('time[s]'))

    # — agrupar por segundos inteiros —
    df['time_ms'] = df['time[s]']
    df['sec'] = (df['time_ms'] // GROUP_MSEC).astype(int)

    agg = {
        'time[s]': 'first',                 # início do segundo
        'serving_cell_rssi_1': 'mean',      # média no segundo
        'serving_cell_snr_1':  'mean',
        'position_x':          'mean',
        'position_y':          'mean',
        'position_z':          'mean'
    }
    df_down = (df.groupby('sec', as_index=False)
                 .agg(agg)
                 .reset_index(drop=True))

    meta.update({
        "down_len": len(df_down),
        "first5_after": df_down['time[s]'].head(5).tolist(),
        "last5_after": df_down['time[s]'].tail(5).tolist()
    })

    df_down[['time[s]', 'serving_cell_rssi_1', 'serving_cell_snr_1',
                    'position_x', 'position_y', 'position_z']]

    print("Antes:", meta['orig_len'], "→ Depois:", meta['down_len'])

    if debug:
        return df_down, meta
    else:
        return df_down

# def add_distance_cols(df: pd.DataFrame,
#                       tx_pos: tuple[float, float, float] = TX_POS):
#     """Inclui distância euclidiana e seu log10 (para modelo log-shadow)."""
#     dx = df['position_x'] - tx_pos[0]
#     dy = df['position_y'] - tx_pos[1]
#     dz = df['position_z'] - tx_pos[2]
#     df['distance_m'] = np.sqrt(dx**2 + dy**2 + dz**2).clip(lower=1e-3)
#     df['log10_d']   = np.log10(df['distance_m'])
#     return df

def import_dataset(file_path: str, split_train_val_test=True):
    df, meta = import_raw_data(file_path=file_path, debug=True)

    # Converter a coluna de RSSI para numérico, caso haja valores como strings
    df['serving_cell_rssi_1'] = pd.to_numeric(df['serving_cell_rssi_1'], errors='coerce')

    # Verificar se há valores infinitos e substituí-los
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    # Normalizar os dados (exceto a coluna de tempo)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['time[s]']))

    # Adicionar a coluna de tempo de volta ao DataFrame escalado
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns[1:])
    scaled_df['time[s]'] = df['time[s]'].values

    # Verificar se há valores NaN ou infinitos após a normalização
    print("Dados contêm NaN:", scaled_df.isna().values.any())
    numeric_df = scaled_df.select_dtypes(include=[np.number])
    print("Dados contêm infinitos:", np.isinf(numeric_df.values).any())

    # Dividir os dados em treino, validação e teste
    if split_train_val_test:
        train_size = int(len(scaled_df) * 0.75)
        val_size = int(len(scaled_df) * 0.15)
        train_df = scaled_df[:train_size]
        val_df = scaled_df[train_size: train_size + val_size]
        test_df = scaled_df[train_size + val_size:]
    else:
        train_df = val_df = None
        test_df = scaled_df

    dataset_name = os.path.basename(file_path)
    return scaler, train_df, val_df, test_df, dataset_name, meta

def estimate_dt_ms(parquet_path, n_rows=2000):
    df_tmp = pd.read_parquet(parquet_path, columns=['time[s]'], engine='fastparquet')
    dt = np.diff(df_tmp['time[s]'].head(n_rows))
    return float(np.nanmedian(dt)) * 1000


def format_elapsed_time(start_time):
    elapsed_seconds = int(time.time() - start_time)

    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# === Função para imprimir progress bar ===
def print_progress_bar(current, total, start_time, length=30, every=100):
    if current % every and current != total:
        return
    percent = current / total
    filled = int(length * percent)
    bar = '=' * filled + '-' * (length - filled)

    elapsed = format_elapsed_time(start_time)
    sys.stdout.write(
        f'\rProgress: |{bar}| {percent:.0%} | Elapsed Time: {elapsed}'
    )
    sys.stdout.flush()

def calc_dataset_params(file_path, timesteps_orig, group_msec):
    """Calcula parâmetros derivados do dataset após downsampling."""
    deltaT_MS = estimate_dt_ms(file_path)  # intervalo médio em ms
    samples_per_win = max(1, group_msec / deltaT_MS)
    timesteps = max(1, int(timesteps_orig / samples_per_win))

    print(f"[INFO] Dataset: {os.path.basename(file_path)}")
    print(f"Δt≈{deltaT_MS:.1f} ms  ·  samples/window≈{samples_per_win:.2f}  →  TIMESTEPS={timesteps}")

    return deltaT_MS, samples_per_win, timesteps
