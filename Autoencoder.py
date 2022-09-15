# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:42:49 2022

@author: user 
"""
#%% Recover Dataset
import os 
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import glob
import pickle 
from Data_Synthesis import Image
#import MedianFilter as Filter
import Generate_Training_Data as data
import Machine_Learning as CNN
import time

#%%


# X = noise-free images
# y = noisy images
def Train_model(pixel_map, noisy_pixel_map, epochs, store_model = False, model_name = ""):
    train, test, noisy_train, noisy_test = train_test_split(pixel_map, noisy_pixel_map, test_size=0.2, random_state=42)
    
    train = np.expand_dims(train, -1)
    test = np.expand_dims(test, -1)
    noisy_train = np.expand_dims(noisy_train, -1)
    noisy_test = np.expand_dims(noisy_test, -1)
    
    train = train.astype("float32") / 255
    test = test.astype("float32") / 255
    noisy_train = noisy_train.astype("float32") / 255
    noisy_test = noisy_test.astype("float32") / 255
    
    print("train shape:", train.shape)
    print(train.shape[0], "train samples")
    print(test.shape[0], "test samples")
    
    callback = keras.callbacks.EarlyStopping(patience=3)
    model = keras.Sequential(
        [
            keras.Input(shape=train[0].shape),
            # encoder
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # decoder
            layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, activation="relu", padding="same"),
            layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, activation="relu", padding="same"),
            layers.Conv2D(1, kernel_size=(3, 3), activation="sigmoid", padding="same"),
        ]
    )
    
    model.summary()

    batch_size = 128
    
    model.compile(loss="binary_crossentropy", optimizer="Adam")
    history = model.fit(noisy_train, train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, callbacks=[callback])
    
    score = model.evaluate(test, noisy_test, verbose=0)
    print("Test binary_crossentropy (loss):", score) # find a mean using the test

    if store_model:
        model.save(model_name+".h5")
        with open(f"{model_name}.obj","wb") as f0:
            pickle.dump(history.history, f0)
            
    return model, history, score

def getModel(model_name):
    new_model = keras.models.load_model(model_name+".h5")
    with open(f"{model_name}.obj","rb") as f0:
        history = pickle.load(f0)
    return new_model, history

def plot_loss(history):
    Epoch = np.arange(1, len(history['loss']) + 1, 1)
    plt.plot(Epoch, history['loss'], label='loss')
    plt.plot(Epoch, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error ["mse"]')
    plt.legend()
    plt.grid(True)

def predict(model, noisy_pixel_map):
    t = time.time()
    pixel_map = np.expand_dims(noisy_pixel_map, -1)
    pixel_map = pixel_map.astype("float32") / 255
    filtered_pixel_map = model.predict(pixel_map) * 255
    filtered_pixel_map = np.squeeze(filtered_pixel_map, -1)
    elapsed_time = time.time() - t
    print(f"Prediction took: {elapsed_time}")
    
    return filtered_pixel_map 

def evaluate(model, pixel_map, noisy_pixel_map):
    pixel_map = np.expand_dims(pixel_map, -1)
    pixel_map = pixel_map.astype("float32") / 255
    noisy_pixel_map = np.expand_dims(noisy_pixel_map, -1)
    noisy_pixel_map = noisy_pixel_map.astype("float32") / 255
    return model.evaluate(pixel_map, noisy_pixel_map)
    
#%% Get DATA

proton_pixel_map_32, combined_pixel_map_32, proton_pixel_map_64, combined_pixel_map_64, variables = data.getMultiTrackData("Dataset3",["Thermal energy of beam (MeV)","Maximum energy of beam (MeV)","Number of accepted particles in beam"])

x_range_mm = [0, 35]
y_range_mm = [0, 35]

weight = 0.1
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 4

thresh_proton_pixel_map_32 = data.add_noise(proton_pixel_map_32, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
noisy_combined_pixel_map_32 = data.add_noise(combined_pixel_map_32, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
multiplied_variables = np.repeat(variables, multiplier, axis=0)
#%%

epochs = 100

model, history, score = Train_model(thresholded_combined_pixel_map_64, noisy_combined_pixel_map_64, epochs, False, "")
plot_loss(history.history)

#%% predict

filtered_proton_pixel_map_32 = predict(model, noisy_combined_pixel_map_32)
Filter.plot_maps(noisy_combined_pixel_map_32[0], filtered_proton_pixel_map_32[0], thresh_proton_pixel_map_32[0], "100 Epochs")
Filter.evaluate(filtered_proton_pixel_map_32[0], thresh_proton_pixel_map_32[0], detector_threshold, True)

#%% CNN train

CNNmodel, CNNhistory, CNNscore, variable_scaler = CNN.CNNTrain(noisy_combined_pixel_map_64, multiplied_variables, epochs)
plot_loss(CNNhistory.history)

#%%

CNNPredictions = CNN.CNNPredict(CNNmodel, filtered_proton_pixel_map_32, variable_scaler)

#%%

CNNEvaluations = CNN.CNNEvaluate(CNNPredictions, multiplied_variables)

#%%
from matplotlib.colors import LogNorm
fig, axs = plt.subplots(1,2,figsize=(10,5))
img0 = axs[0].imshow(combined_pixel_map_64[0].T, cmap='hot', origin='lower', norm=LogNorm(vmax=255))
axs[0].set_title('Clean image')
plt.colorbar(img0, ax=axs[0])
img1 = axs[1].imshow(noisy_combined_pixel_map_64[0].T, cmap='hot', origin='lower')
plt.colorbar(img1, ax=axs[1])
axs[1].set_title('Noisy image')
axs[0].set_xlabel(r"Position $x$ (pixel)")
axs[1].set_xlabel(r"Position $x$ (pixel)")
axs[0].set_ylabel(r"Position $y$ (pixel)")
#%%

noisy_combined_pixel_map_64 = data.add_noise(combined_pixel_map_64, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
thresholded_combined_pixel_map_64 = data.add_noise(combined_pixel_map_64, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
