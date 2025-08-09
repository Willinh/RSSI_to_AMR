"""
conversor_csv_parquet.py  –  v4
--------------------------------
• Lê CSV gerado pelo logger (agora com separador ';' e cabeçalhos
  repetidos no meio do arquivo).
• Descarta linhas-cabeçalho extras e linhas vazias.
• Converte RSSI e coordenadas para float.
• Gera Parquet com colunas:
  ['time[s]', 'serving_cell_rssi_1', 'serving_cell_snr_1',
   'position_x', 'position_y', 'position_z']
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tkinter import Tk, filedialog

# ---------- 1. Selecionar CSV --------------------------------------------
Tk().withdraw()
csv_path = filedialog.askopenfilename(
    title="Selecione o CSV bruto",
    filetypes=[("CSV ; separated", "*.csv")],
)
if not csv_path:
    print("Nada selecionado – saindo.")
    raise SystemExit

# ---------- 2. Ler, aceitando linhas ruins -------------------------------
#   • sep=';'   • on_bad_lines='skip'  (pandas >=1.3)
df_raw = pd.read_csv(
    csv_path,
    sep=';',
    dtype=str,
    low_memory=False,
    on_bad_lines='skip'          # ignora linhas com nº colunas diferente
)

# ---------- 3. Remover cabeçalhos embutidos ------------------------------
# linhas cujo primeiro campo literalmente seja 'timestamp'
df = df_raw[df_raw['timestamp'].str.lower() != 'timestamp'].copy()

# ---------- 4. Renomear para nomes internos ------------------------------
rename = {
    'timestamp': 'time[s]',
    'rssi':      'serving_cell_rssi_1',
    'x':         'position_x',
    'y':         'position_y',
}
df.rename(columns=rename, inplace=True)

# ---------- 5. Converter colunas numéricas -------------------------------
# 5.1 tempo em segundos (int); se ainda contém ponto decimal, remover ponto
# 1) troca vírgula por ponto, preserva a parte decimal

if 'time[s]' in df.columns:
    print("Processando a coluna 'time[s]'...")
    # 1. Garante que a coluna é numérica, transformando inválidos em 'Não um Número' (NaN)
    df['time[s]'] = pd.to_numeric(df['time[s]'], errors='coerce')

    # 2. Remove linhas onde a conversão do tempo falhou
    df.dropna(subset=['time[s]'], inplace=True)

    # 3. Converte de segundos (float) para milissegundos (inteiro de 64 bits)
    #    Exemplo: 1712926785.451 -> 1712926785451
    df['time[s]'] = (df['time[s]'] * 1000).astype('int64')
    print("Coluna 'time[s]' convertida para milissegundos (inteiro).")

# secs_float = pd.to_numeric(df['time[s]'].str.replace(',', '.'), errors='coerce')
#
# # 2) se você realmente quer segundos inteiros: arredonde/astype
# #    senão, mantenha float com fração de segundo
# df['time[s]'] = secs_float

# 5.2 RSSI inteiro (já sem ponto)
df['serving_cell_rssi_1'] = pd.to_numeric(
    df['serving_cell_rssi_1'].str.extract(r'(-?\d+)')[0],
    errors='coerce'
)

# 5.3 posições podem vir em notação expoencial
for col in ('position_x', 'position_y'):
    df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

# 5.4 colunas faltantes
df['serving_cell_snr_1'] = 0
df['position_z'] = 0

# ---------- 6. Limpeza final ---------------------------------------------
df.dropna(subset=['time[s]', 'serving_cell_rssi_1'], inplace=True)
df.sort_values('time[s]', inplace=True)

# ---------- 7. Guardar somente as colunas exigidas -----------------------
ordered = [
    'time[s]',
    'serving_cell_rssi_1',
    'serving_cell_snr_1',
    'position_x',
    'position_y',
    'position_z'
]
df_final = df[ordered]

# ---------- 8. Salvar em Parquet -----------------------------------------
parquet_path = Path(csv_path).with_suffix('.parquet')
df_final.to_parquet(parquet_path, index=False)
print(f"✅  Parquet salvo em {parquet_path}  ({len(df_final):,} linhas)")
