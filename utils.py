import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def import_raw_data():
    file_path = 'C:\\Github\\iv2lp_lstm\\data\\iV2Ip.parquet'
    df = pd.read_parquet(file_path, engine='fastparquet')
    selected_columns = [
        'time[s]', 'serving_cell_rssi_1', 'serving_cell_snr_1',
        'position_x', 'position_y', 'position_z'
    ]

    return df[selected_columns]


def import_dataset():
    df = import_raw_data()

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
    print("Dados contêm NaN:", np.any(np.isnan(scaled_df)))
    print("Dados contêm infinitos:", np.any(np.isinf(scaled_df)))

    # Dividir os dados em treino, validação e teste
    train_size = int(len(scaled_df) * 0.7)
    val_size = int(len(scaled_df) * 0.15)
    test_size = len(scaled_df) - train_size - val_size

    train_df = scaled_df[:train_size]
    val_df = scaled_df[train_size:train_size + val_size]
    test_df = scaled_df[train_size + val_size:]

    return scaler, train_df, val_df, test_df
