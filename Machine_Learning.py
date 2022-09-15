# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:42:49 2022

@author: user
"""
#%% 

import os 
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import glob
import pickle 
from Data_Synthesis import Image
import Generate_Training_Data as data
from sklearn.preprocessing import MinMaxScaler
import time

#%%

def CNNTrain(pixel_map, variables, epochs):
    pixel_map = pixel_map.astype("float32") / 255
    x_train, x_test, y_train, y_test = train_test_split(pixel_map, variables, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    variable_scaler = MinMaxScaler()
    y_train = variable_scaler.fit_transform(y_train)
    y_test = variable_scaler.transform(y_test)
    callback = keras.callbacks.EarlyStopping(patience=3)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    
    #callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = keras.Sequential(
        [
            keras.Input(shape=x_train[0].shape),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(96, activation="relu"),
            layers.Dense(3, activation="linear"),
        ]
    )
    
    model.summary()

    batch_size = 128
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', metrics=['mean_absolute_percentage_error', "mean_absolute_error"],optimizer=optimizer)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test mean_squared_error (loss):", score)
            
    return model, history, score, variable_scaler

def saveModel(model, history, model_name):
    model.save(model_name+".h5")
    with open(f"{model_name}.obj","wb") as f0:
        pickle.dump(history.history, f0)
    print(f"Model {model_name} saved")

def getModel(model_name):
    new_model = keras.models.load_model(model_name+".h5")
    with open(f"{model_name}.obj","rb") as f0:
        history = pickle.load(f0)
    
    return new_model, history

def plot_loss(history):
    Epoch = np.arange(1, len(history['loss']) + 1, 1)
    plt.plot(Epoch, history['loss'], label='Training loss')
    plt.plot(Epoch, history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss [mean squared error]')
    plt.legend()
    plt.grid(True)

def CNNPredict(model, pixel_map, variable_scaler):
    t = time.time()
    pixel_map = np.expand_dims(pixel_map, -1)
    pixel_map = pixel_map.astype("float32") / 255
    predictions = model.predict(pixel_map)
    predictions = variable_scaler.inverse_transform(predictions)
    elapsed_time = time.time() - t
    print(f"Prediction took: {elapsed_time}")
    return predictions

def CNNEvaluate (predicted_variables, true_variables):
    absolute_percentage_errorList = []
    for i in range(len(predicted_variables[0])):
        absolute_percentage_error = abs((np.array(predicted_variables[:,i]) - np.array(true_variables[:,i]))) / np.array(true_variables[:,i]) * 100
        absolute_percentage_errorList.append(absolute_percentage_error)
    res = plt.boxplot(absolute_percentage_errorList, sym="")
    plt.xlabel("Variable")
    plt.ylabel("Absolute percentage error (%)")
    plt.legend()
        
    return absolute_percentage_errorList, res

def saveResults(predicted_variables, true_variables):
    Results = dict()
    Results["Predicted_variables"] = predicted_variables
    Results["True_variables"] = true_variables
    INFO = "64_128_64CNN_64Pixels_CLEAN"
    foldername = "CNNResults"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    with open(f"{foldername}/{INFO}.obj","wb") as f0:
        pickle.dump(Results, f0)

#%%

proton_pixel_map_32, combined_pixel_map_32, proton_pixel_map_64, combined_pixel_map_64, variables = data.getMultiTrackData("Dataset4",["Thermal energy of beam (MeV)","Maximum energy of beam (MeV)","Number of accepted particles in beam"])

x_range_mm = [0, 35]
y_range_mm = [0, 35]

weight = 0.1
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 2

epochs = 50

#thresh_proton_pixel_map_64 = data.add_noise(proton_pixel_map_64, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
#thresh_combined_pixel_map_64 = data.add_noise(combined_pixel_map_64, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
noisy_combined_pixel_map_64 = data.add_noise(combined_pixel_map_64, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
#noisy_combined_pixel_map_32 = data.add_noise(combined_pixel_map_32, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
multiplied_variables = np.repeat(variables, multiplier, axis=0)

#%%

noisy_model, noisy_history = getModel("64_128_64CNN_64Pixels_NOISY_Overfit")

noisy_model, noisy_history, noisy_score, variable_scaler, noisy_x_test, noisy_y_test = CNNTrain(noisy_combined_pixel_map_64, multiplied_variables, epochs)
plot_loss(noisy_history.history)

noisy_x_train, noisy_x_test, noisy_y_train, noisy_y_test = train_test_split(noisy_combined_pixel_map_64, multiplied_variables, test_size=0.2, random_state=42)

noisy_variable_scaler = MinMaxScaler()

noisy_y_train = noisy_variable_scaler.fit_transform(noisy_y_train)

noisy_predictions= CNNPredict(noisy_model, noisy_x_test, noisy_variable_scaler)
#%%
noisy_evaluations, noisy64res  = CNNEvaluate(noisy_predictions, noisy_y_test)
data = [item.get_ydata() for item in protonres['medians']]
#%%
noisy_plot = plt.boxplot(noisy_evaluations, patch_artist=True)

colors1 = ['lightpink','lightpink','lightpink'] 

for patch, color in zip(noisy_plot['boxes'], colors1): 
    patch.set_facecolor(color) 

for median in noisy_plot['medians']:
    median.set_color('red')
    
plt.legend([noisy_plot["boxes"][0]], ['Machine learning'])
plt.grid(True, axis='y')

ticks = [r'Effective temperature $T$', r'Maximum energy $E$', r'Particle number $N$']

plt.xticks([1,2,3], ticks)
 
plt.xlabel("Variable")
plt.ylabel("Absolute percentage error (%)")
    
#saveModel(noisy_model, noisy_history, "64_128_64CNN_64Pixels_NOISY_Overfit")

#saveResults(noisy_predictions, variable_scaler.inverse_transform(noisy_y_test))

#%%

clean_model, clean_history = getModel("64_128_64CNN_64Pixels_CLEAN_Overfit")

clean_model, clean_history, clean_score, clean_variable_scaler, clean_x_test, clean_y_test = CNNTrain(thresh_combined_pixel_map_64, multiplied_variables, epochs)
plot_loss(clean_history.history)

clean_variable_scaler = MinMaxScaler()

clean_y_train = clean_variable_scaler.fit_transform(clean_y_train)

clean_y_test = clean_variable_scaler.transform(clean_y_test)


clean_predictions = CNNPredict(clean_model, clean_x_test , clean_variable_scaler)

clean_evaluations, cleanres = CNNEvaluate(clean_predictions, clean_variable_scaler.inverse_transform(clean_y_test))

saveModel(clean_model, clean_history, "64_128_64CNN_64Pixels_CLEAN_Overfit")

saveResults(clean_predictions, clean_variable_scaler.inverse_transform(clean_y_test))

#%%

predictions = [noisy_predictions, clean_predictions]
#%%
true_variables = clean_variable_scaler.inverse_transform(clean_y_test)
#plt.figure(figsize=(8, 6))
ticks = [r'Effective temperature $T$', r'Maximum energy $E$', r'Particle number $N$']
 
# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
noisy_plot = plt.boxplot(noisy_evaluations ,positions=np.array(np.arange(3))*2.0-0.35,widths=0.6, sym="", patch_artist=True)
clean_plot = plt.boxplot(clean_evaluations ,positions=np.array(np.arange(3))*2.0+0.35,widths=0.6, sym="", patch_artist=True)

colors1 = ['lightpink','lightpink','lightpink'] 

for patch, color in zip(noisy_plot['boxes'], colors1): 
    patch.set_facecolor(color) 

for median in noisy_plot['medians']:
    median.set_color('red')
    
for median in clean_plot['medians']:
    median.set_color('red')    
    
colors2 = ['#8bb5d9','#8bb5d9','#8bb5d9'] 

for patch, color in zip(clean_plot['boxes'], colors2): 
    patch.set_facecolor(color) 
    
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k))
         
    # use plot function to draw a small line to name the legend.
    #plt.plot([], c=color_code, label=label)
    #plt.legend()

plt.legend([noisy_plot["boxes"][0], clean_plot["boxes"][0]], ['Noisy test data', 'Clean test data'])
plt.grid(True, axis='y')
# setting colors for each groups
define_box_properties(noisy_plot, 'red', 'Convolutional neural network')
define_box_properties(clean_plot, 'blue', 'Traditional analysis')
 
# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
 
plt.xlabel("Variable")
plt.ylabel("Absolute percentage error (%)")


# set the limit for x axis
 
# set the title

#%%

fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].scatter(noisy_y_test[:,0],noisy_evaluations[0],marker='x')
axs[0].set_xlabel(r'Effective temperature $T$ (MeV)')
axs[0].set_ylabel("Absolute percentage error (%)")
axs[1].scatter(noisy_y_test[:,1],noisy_evaluations[1],marker='x')
axs[1].set_xlabel( r'Maximum energy $E$ (MeV)')
axs[1].set_ylabel("Absolute percentage error (%)")
axs[2].scatter(noisy_y_test[:,2],noisy_evaluations[2],marker='x')
axs[2].set_xlabel(r'Particle number $N$')
axs[2].set_ylabel("Absolute percentage error (%)")
#plt.scatter(variable_scaler.inverse_transform(noisy_y_test)[:,0],noisy_evaluations[0],marker='x')

#%%
from matplotlib import colors
from matplotlib.colors import LogNorm
fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].hist2d(noisy_y_test[:,0],noisy_evaluations[0],50, [[min(noisy_y_test[:,0]), max(noisy_y_test[:,0])], [0, 300]],cmap='hot',norm=LogNorm())
axs[0].set_xlabel(r'Effective temperature $T$ (MeV)',fontsize=18)
axs[0].set_ylabel("Absolute percentage error (%)",fontsize=18)
img1=axs[1].hist2d(noisy_y_test[:,1],noisy_evaluations[1],50, [[min(noisy_y_test[:,1]), max(noisy_y_test[:,1])], [0, 120]],cmap='hot',norm=LogNorm())
axs[1].set_xlabel( r'Maximum energy $E$ (MeV)',fontsize=18)
axs[1].set_ylabel("Absolute percentage error (%)",fontsize=18)
img2=axs[2].hist2d(noisy_y_test[:,2],noisy_evaluations[2],50, [[min(noisy_y_test[:,2]), max(noisy_y_test[:,2])], [0, 25]],cmap='hot',norm=LogNorm())
axs[2].set_xlabel(r'Particle number $N$',fontsize=18)
axs[2].set_ylabel("Absolute percentage error (%)",fontsize=18)
fig.colorbar(img2[3])
#plt.scatter(variable_scaler.inverse_transform(noisy_y_test)[:,0],noisy_evaluations[0],marker='x')
#plt.hist2d(noisy_y_test[:,0],noisy_evaluations[0],20, [[min(noisy_y_test[:,0]), max(noisy_y_test[:,0])], [0, 200]])
#%%

plt.scatter(variable_scaler.inverse_transform(noisy_y_test)[:,0],noisy_predictions[:,0],marker='x')

#%%

clean_x_train, clean_x_test, clean_y_train, clean_y_test = train_test_split(thresh_combined_pixel_map_64, multiplied_variables, test_size=0.2, random_state=42)

clean_variable_scaler = MinMaxScaler()

clean_y_train = clean_variable_scaler.fit_transform(clean_y_train)

clean_y_test = clean_variable_scaler.transform(clean_y_test)

clean_predictions = CNNPredict(clean_model, clean_x_test , clean_variable_scaler)

clean_evaluations, cleanres = CNNEvaluate(clean_predictions, clean_variable_scaler.inverse_transform(clean_y_test))

clean_evaluations, clean_res = CNNEvaluate(np.array(manual_variablesList), clean_variable_scaler.inverse_transform(clean_y_test))

manual_noisy_evaluations = CNNEvaluate(np.array(manual_variablesList), clean_variable_scaler.inverse_transform(clean_y_test))
#%%
real_count = []
for i in median_proton_pixel_map_64:
    count = sum(sum(i))
    real_count.append(count)
real_count = np.array(real_count)
real_y_test = []
for j in noisy_y_test:
    listt = []
    for i in range(3):
        listt.append(j[i])
    real_y_test.append(listt)
real_y_test = np.array(real_y_test)
#%%
proton_evaluations, protonres = CNNEvaluate(np.array(manual_proton_variablesList), real_y_test)

#%%

fig, axs = plt.subplots(3,1, figsize=(6,11))
axs[0].scatter(clean_variable_scaler.inverse_transform(clean_y_test)[:,0],manual_noisy_evaluations[0],marker='x')
axs[0].set_xlabel(r'Effective temperature $T$ (MeV)')
axs[0].set_ylabel("Absolute percentage error (%)")
axs[0].set_ylim(0,900)
axs[1].scatter(clean_variable_scaler.inverse_transform(clean_y_test)[:,1],manual_noisy_evaluations[1],marker='x')
axs[1].set_xlabel( r'Maximum energy $E$ (MeV)')
axs[1].set_ylabel("Absolute percentage error (%)")
axs[2].scatter(clean_variable_scaler.inverse_transform(clean_y_test)[:,2],manual_noisy_evaluations[2],marker='x')
axs[2].set_xlabel(r'Particle number $N$')
axs[2].set_ylabel("Absolute percentage error (%)")

#%%

plt.figure(figsize=(7, 5))
ticks = [r'Effective temperature $T$', r'Maximum energy $E$', r'Particle number $N$']
 
# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
noisy_plot = plt.boxplot(manual_noisy_evaluations ,positions=np.array(np.arange(3))*2.0-0.35,widths=0.6, sym="", patch_artist=True)
clean_plot = plt.boxplot(proton_evaluations ,positions=np.array(np.arange(3))*2.0+0.35,widths=0.6, sym="", patch_artist=True)

colors1 = ['lightpink','lightpink','lightpink'] 

for patch, color in zip(noisy_plot['boxes'], colors1): 
    patch.set_facecolor(color) 

for median in noisy_plot['medians']:
    median.set_color('red')
    
for median in clean_plot['medians']:
    median.set_color('red')    
    
colors2 = ['#8bb5d9','#8bb5d9','#8bb5d9'] 

for patch, color in zip(clean_plot['boxes'], colors2): 
    patch.set_facecolor(color) 
    
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k))
         
    # use plot function to draw a small line to name the legend.
    #plt.plot([], c=color_code, label=label)
    #plt.legend()

plt.legend([noisy_plot["boxes"][0], clean_plot["boxes"][0]], ['Noisy test data', 'Clean test data'])
plt.grid(True, axis='y')
# setting colors for each groups
define_box_properties(noisy_plot, 'red', 'Noisy test data')
define_box_properties(clean_plot, 'blue', 'Clean test data')
 
# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
 
plt.xlabel("Variable")
plt.ylabel("Absolute percentage error (%)")

#%%

noisy_model32, noisy_history32, noisy_score32, noisy_variable_scaler32 = CNNTrain(noisy_combined_pixel_map_32, multiplied_variables, epochs)
plot_loss(noisy_history32.history)

noisy_x_train32, noisy_x_test32, noisy_y_train32, noisy_y_test32 = train_test_split(noisy_combined_pixel_map_32, multiplied_variables, test_size=0.2, random_state=42)

saveModel(noisy_model32, noisy_history32, "128_32_8CNN_32Pixels_NOISY")

noisy_predictions32 = CNNPredict(noisy_model32, noisy_x_test32, noisy_variable_scaler32)

noisy_evaluations32 = CNNEvaluate(noisy_predictions32, noisy_y_test32)
