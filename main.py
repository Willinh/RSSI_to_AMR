from utils import import_dataset
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import keras_tuner as kt

# Importar dataset e escalador
scaler, train_df, val_df, test_df = import_dataset()

# Preparar sequências para LSTM
timesteps = 10

# Gerador de sequências para treino
train_generator = TimeseriesGenerator(
    train_df.drop(columns=['time[s]']).values,
    train_df['serving_cell_rssi_1'].values,
    length=timesteps, batch_size=32
)

# Gerador de sequências para validação
val_generator = TimeseriesGenerator(
    val_df.drop(columns=['time[s]']).values,
    val_df['serving_cell_rssi_1'].values,
    length=timesteps, batch_size=32
)

# Gerador de sequências para teste
test_generator = TimeseriesGenerator(
    test_df.drop(columns=['time[s]']).values,
    test_df['serving_cell_rssi_1'].values,
    length=timesteps, batch_size=32
)


# Definir o modelo para o Keras Tuner
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 1, 3)):  # otimizando o número de camadas de 1 a 3
        model.add(LSTM(
            units=hp.Int('units_' + str(i), min_value=16, max_value=1024, step=16),
            activation=hp.Choice('activation', values=['relu', 'tanh','sigmoid','linear','softmax','elu','exponential','softsign']),
            return_sequences=True if i < hp.get('num_layers') - 1 else False,
            input_shape=(timesteps, train_df.shape[1] - 1) if i == 0 else None
        ))
        model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd','Adadelta','Nadam','Adamax']),
        loss='mse'
    )
    return model


# Configurar o Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=30,
    hyperband_iterations=2,
    directory='my_dir',
    project_name='lstm_hyperparameter_tuning'
)

# Executar o grid search
tuner.search(train_generator, epochs=30, validation_data=val_generator)

# Obter os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Melhor número de camadas ocultas: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"Melhor número de neurônios na camada {i + 1}: {best_hps.get('units_' + str(i))}")
print(f"Melhor função de ativação: {best_hps.get('activation')}")
print(f"Melhor otimizador: {best_hps.get('optimizer')}")

# Treinar o modelo com os melhores hiperparâmetros
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_generator, epochs=30, validation_data=val_generator)

# Avaliar o modelo no conjunto de teste
test_loss = model.evaluate(test_generator)
print("Test Loss:", test_loss)

# Salvar o modelo treinado
model.save('lstm_model.keras')

# Fazer previsões
predictions = model.predict(test_generator)
predicted_rssi = predictions.flatten()

# Obter os valores reais de RSSI
true_rssi = test_df['serving_cell_rssi_1'].values[timesteps:]

# Desnormalizar os valores previstos e reais
predicted_rssi_descaled = scaler.inverse_transform(
    np.hstack((predicted_rssi.reshape(-1, 1), np.zeros((predicted_rssi.shape[0], train_df.shape[1] - 2)))))[:, 0]

true_rssi_descaled = scaler.inverse_transform(
    np.hstack((true_rssi.reshape(-1, 1), np.zeros((true_rssi.shape[0], train_df.shape[1] - 2)))))[:, 0]

# Obter os timestamps correspondentes
timestamps = test_df['time[s]'].values[timesteps:]

# Obter as posições correspondentes
positions_x = test_df['position_x'].values[timesteps:]
positions_y = test_df['position_y'].values[timesteps:]

# Criar um novo DataFrame com os dados de timestamp, valores reais e previstos, e posições
results_df = pd.DataFrame({
    'time[s]': timestamps,
    'serving_cell_rssi_1': true_rssi_descaled,
    'predicted_rssi': predicted_rssi_descaled,
    'position_x': positions_x,
    'position_y': positions_y
})

# Calcular métricas de erro
mae = mean_absolute_error(true_rssi_descaled, predicted_rssi_descaled)
mse = mean_squared_error(true_rssi_descaled, predicted_rssi_descaled)
rmse = np.sqrt(mse)
r2 = r2_score(true_rssi_descaled, predicted_rssi_descaled)

# Exibir as métricas de erro
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
