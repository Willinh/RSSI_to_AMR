import pandas as pd, numpy as np



def quick_scan(csv_path):
    df = pd.read_csv(csv_path)

    print("\n==== SCAN", csv_path.split('/')[-1], "====")
    print("Rows:", len(df))

    # 1. Colunas presentes
    print("Cols:", df.columns.tolist())

    # 2. Estatísticas rápidas da coluna RSSI original
    rssi_col = [c for c in df.columns if 'rssi' in c.lower()][0]
    print("RSSI unique:", df[rssi_col].nunique())
    print(df[rssi_col].describe())

    # 3. Converter timestamp (segundos desde 0) para Timedelta
    if 'timestamp' in df.columns:
        # ① forçar numérico; strings inválidas → NaN
        secs = pd.to_numeric(df['timestamp'], errors='coerce')

        # ② usar unit='s' só se agora é numérico
        df['timestamp'] = pd.to_timedelta(secs, unit='s')
    else:
        secs = pd.to_numeric(df['time[s]'], errors='coerce')
        df['timestamp'] = pd.to_timedelta(secs, unit='s')

    # 4. Tamanho de cada janela 100 ms
    counts = (df.set_index('timestamp')
                .resample('100ms')
                .size())
    print("Samples per 100 ms window – top 5:\n", counts.value_counts().head())

quick_scan(r'C:\\Users\\admpdi\\OneDrive - ISI SIM\\Documents\\GitHub\\RSSI-to-AMR\\data\\rssi_odom_dataset_11apr25.csv')
quick_scan(r'C:\\Users\\admpdi\\OneDrive - ISI SIM\\Documents\\GitHub\\RSSI-to-AMR\\data\\rssi_odom_dataset_12apr25.csv')
