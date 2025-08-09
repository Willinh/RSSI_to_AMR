import pandas as pd
import matplotlib.pyplot as plt, numpy as np

df = pd.read_parquet(r'C:\\Users\\admpdi\\OneDrive - ISI SIM\\Documents\\GitHub\\RSSI-to-AMR\\data\\rssi_odom_dataset_11apr25_ajust.parquet')
df['timestamp'] = pd.to_timedelta(df['time[s]'], unit='s')

empty_ratio = (df.set_index('timestamp')
                 .resample('100ms')        # mesma janela do pipeline
                 .size()
                 .eq(0)
                 .mean())
print("Janelas vazias de 100 ms:", f"{empty_ratio:.0%}")
secs = pd.to_numeric(df['time[s]'], errors='coerce')
if secs.max() > 1e10:
    secs = secs / 1000          # ms → s

plt.plot(secs, df['serving_cell_rssi_1'], '.', ms=1)
plt.xlabel("time[s]"); plt.ylabel("RSSI")