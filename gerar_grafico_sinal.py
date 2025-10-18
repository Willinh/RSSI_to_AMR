import numpy as np
import matplotlib.pyplot as plt

# --- Configurações Iniciais ---
# Ajuste estes parâmetros para modificar a aparência do gráfico
distancia_max = 15 # Distância máxima em metros
n_pontos = 150      # Número de pontos para a simulação
path_loss_exponent = 2.5  # Expoente de perda de percurso (n)
shadowing_std_dev = 4.0   # Desvio padrão do sombreamento em dB
frequencia_shadowing = 0.1 # Frequência das variações lentas (sombreamento)
# Fator para aumentar a "agressividade" dos picos e vales
fator_escala_rayleigh = 1.2  # Experimente valores entre 1.2 e 2.5

# --- 1. Geração dos Dados ---

# Eixo de distância
distancia = np.linspace(0.2, distancia_max, n_pontos)
# Evitar log(0) usando uma distância mínima de 1 metro

# a) Modelo de Perda de Percurso (Path Loss)
# P_r(d) = P_0 - 10 * n * log10(d/d_0)
# Assumindo P_0 (potência a 1m) = 0 dB para simplificar a visualização relativa
perda_percurso = -10 * path_loss_exponent * np.log10(distancia)

# b) Modelo de Sombreamento (Log-Normal Shadowing)
# Gerar um ruído gaussiano "lento"
ruido_lento_bruto = np.random.normal(0, shadowing_std_dev, n_pontos)
# Suavizar o ruído para criar variações lentas usando uma média móvel simples
tamanho_janela = int(n_pontos * frequencia_shadowing)
janela = np.ones(tamanho_janela) / tamanho_janela
sombreamento_suavizado = np.convolve(ruido_lento_bruto, janela, 'same')
sinal_com_sombreamento = perda_percurso + sombreamento_suavizado

# c) Modelo de Desvanecimento por Multipercursos (Rayleigh Fading)
# A distribuição Rayleigh modela a amplitude do sinal.
# A potência é o quadrado da amplitude. Em dB, isso se torna um fator 20.
# O desvanecimento Rayleigh tem média não nula, então subtraímos a média para que as flutuações ocorram em torno de 0 dB.
rayleigh_amplitude = np.random.rayleigh(1, n_pontos)
desvanecimento_rayleigh_db = 20 * np.log10(rayleigh_amplitude)
# Centralizar as flutuações em torno da média
desvanecimento_rayleigh_db -= np.mean(desvanecimento_rayleigh_db)

# d) Sinal Combinado (Sinal Instantâneo)
# sinal_final = sinal_com_sombreamento + desvanecimento_rayleigh_db

# d) Sinal Combinado (Sinal Instantâneo)
sinal_final = sinal_com_sombreamento + (desvanecimento_rayleigh_db * fator_escala_rayleigh)
# --- 2. Plotagem com Matplotlib ---

# Configurações de estilo para uma aparência mais profissional
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Plotar cada componente
ax.plot(distancia, sinal_final, label='Sinal Instantâneo (com multipercursos)', color='black', linewidth=1.0)
ax.plot(distancia, sinal_com_sombreamento, label='Sombreamento (média local)', color='gray', linestyle='--', linewidth=1.5)
ax.plot(distancia, perda_percurso, label='Perda no Percurso (média em área)', color='dimgray', linestyle=':', linewidth=2.0)

# # Anotações no gráfico (similar à imagem original)
# ax.annotate('Desvanecimento\npor multipercursos\n(instantâneo)',
#             xy=(distancia[15], sinal_final[15]),
#             xytext=(distancia[15] + 1.5, sinal_final[15] + 1),
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#             fontsize=12, ha='center')
#
# ax.annotate('Sombreamento\n(média local)',
#             xy=(distancia[60], sinal_com_sombreamento[60]),
#             xytext=(distancia[60] + 2, sinal_com_sombreamento[60] + 1),
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#             fontsize=12, ha='center')
#
# ax.annotate('Perda no percurso\n(média em área)',
#             xy=(distancia[80], perda_percurso[80]),
#             xytext=(distancia[80], perda_percurso[80] - 1),
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#             fontsize=12, ha='center')

# Configuração dos eixos e título
ax.set_xlabel('Distância[m]', fontsize=14)
ax.set_ylabel('Amplitude relativa [dB]', fontsize=14)
ax.set_title('Modelo Combinado de Atenuação de Sinal Sem Fio', fontsize=16)
ax.legend(fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)

# Ajustar os limites para melhorar a visualização
ax.set_ylim(np.min(sinal_final) - 5, np.max(sinal_final) + 15)
ax.set_xlim(0, distancia_max)

# Salvar a figura em alta resolução
plt.savefig('modelo_sinal_combinado.png', dpi=300, bbox_inches='tight')

# Mostrar o gráfico
plt.show()