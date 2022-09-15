# -*- coding: utf-8 -*-

from kerastuner.tuners import Hyperband
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

def preprocess(pixel_map, variables):
    
    pixel_map = pixel_map.astype("float32") / 255
    x_train, x_test, y_train, y_test = train_test_split(pixel_map, variables, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    variable_scaler = MinMaxScaler()
    y_train = variable_scaler.fit_transform(y_train)
    y_test = variable_scaler.transform(y_test)
    
    return x_train, x_test, y_train, y_test
    
def tune_model(hp):
    model = keras.Sequential(
        [
            keras.Input(shape=((64,64, 1))),
            
            layers.Conv2D(filters = hp.Choice(
                'num_filters_1',
                values=[8, 16, 32, 64, 128],
                default=32), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(filters = hp.Choice(
                'num_filters_2',
                values=[8, 16, 32, 64, 128],
                default=32), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(filters = hp.Choice(
                'num_filters_3',
                values=[8, 16, 32, 64, 128],
                default=32), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Dropout(rate = hp.Float(
                'dropout_1',
                min_value=0.0, max_value=0.5, default=0.4, step=0.05)),
            
            layers.Flatten(),
            
            layers.Dense(units = hp.Int(
                'units_1',
                min_value=0, max_value=128, default=16, step=16),
                activation="relu"),
          
            layers.Dense(3, activation="linear"),
        ]
    )   
    
    model.compile(optimizer=keras.optimizers.Adam(1e-4),loss='mse')
    return model

#%%

tuner = Hyperband(tune_model,objective='val_loss', project_name="best_regression_model_64Pixels")
x_train, x_test, y_train, y_test = preprocess(noisy_combined_pixel_map_64, multiplied_variables)
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))    
best_model = tuner.get_best_models()[0]

#%%

for trial in tuner.oracle.trials:
    trial.metrics.get_history('loss')
