# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:23:03 2022

@author: user
"""
#%%
from scipy.ndimage import median_filter
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import os
import glob 
import pickle
import scipy.constants 
from scipy.optimize import curve_fit
import Generate_Training_Data as data

#%%

def remove_noise(noisy_map):
    
    def median():
        f = median_filter(noisy_map, size=(3,3))
        return f
    
    def remove_background(f):
        most_common = stat.mode(f.flatten())
        f -= most_common
        return f # subtract most common element
    
    
    #t = time.time()
    filtered_map = median()
    filtered_map = remove_background(filtered_map)
    #elapsed_time = time.time() - t
    #print(f"Manual denoising process took {elapsed_time} s")
    
    return filtered_map

def plot_maps(noisy_map, filtered_map, noisefree_map, title="Median filtered 3x3"):
    fig, axs = plt.subplots(3,1)
    axs[0].imshow(noisy_map.T, cmap='hot', origin='lower')
    axs[0].set_title('Noisy')
    axs[1].imshow(filtered_map.T, cmap='hot', origin='lower')
    axs[1].set_title('Filtered')
    axs[2].imshow(noisefree_map.T, cmap='hot', origin='lower') # plot the thresholded version
    axs[2].set_title('Noise-free')
    fig.suptitle(title)
    
def evaluate(filtered_map, noisefree_map, threshold, track_only=False):
    
    def mean_squared_error():
        total = 0
        pixels = 0 # total number of pixels in the image
        map_size = filtered_map.shape[0]
        for i in range(map_size):
            for j in range(map_size):
                diff = filtered_map[i][j] - noisefree_map[i][j]
                
                if track_only:
                    #  unless  both are zero, otherwise count
                    if (filtered_map[i][j] != 0 and noisefree_map[i][j] == 0) or (noisefree_map[i][j] != 0 and filtered_map[i][j] == 0): 
                        total += diff ** 2
                        pixels += 1
                else:
                    total += diff ** 2
                    pixels += 1
                    
        error = total / pixels
        
        return error
                
    def signal_noise():
        mse = mean_squared_error()
        ratio = 10 * np.log10(threshold ** 2 / mse)
        
        return ratio
    
    signal_to_noise_ratio = signal_noise()
    SSIM = ssim(noisefree_map, filtered_map, data_range=filtered_map.max() - filtered_map.min()) # otherwise treat as -1 to 1 
    
    print(f"Signal-to-noise ratio (SNR): {signal_to_noise_ratio}")
    print(f"Structural similarity index measure (SSIM): {SSIM}")
    
    # function is checked by comparing two noise free images
    # SSIM = 1 as expected

# warning: this function uses the parabolic equation - assumed uniform E and B field
def construct_thermal_distribution(pixel_map, energy_cut, x_range_mm, y_range_mm, charge_e, B_field_mT, E_field_kV, field_length_mm, distance_mm, mass_MeV, title="Energy distribution"):
    x_range = np.array(x_range_mm) * 1e-3
    y_range = np.array(y_range_mm) * 1e-3
    q = charge_e * 1.6e-19
    B = B_field_mT * 1e-3
    l = field_length_mm * 1e-3
    E = E_field_kV * 1e3 / l
    D = distance_mm * 1e-3 # distance from centre of Thomson spec to detector
    m = mass_MeV * (1e6*scipy.constants.e/scipy.constants.c**2)
    
    def get_energy_x(x): # find energy using x
        if x != 0:
            E_kin = (q * E * l * D) / (2*x)
        else: 
            E_kin = 10*1e6*scipy.constants.e # supposed to be  -> arbitraily assigned now
        return E_kin / (1e6*scipy.constants.e)
    
    def get_energy_y(y): # find energy using x
        if y != 0:
            E_kin = (q * B * l * D) ** 2 / (2 * m * y**2) 
        else: 
            E_kin = 10*1e6*scipy.constants.e
        return E_kin / (1e6*scipy.constants.e)
    
    map_size = pixel_map.shape[0]
    pixel_size = (x_range.max()-x_range.min()) / map_size
  
    #E_x_binedges = [] # energy range
    E_y_binedges = []
    for i in range(map_size + 1):
        #E_x_binedges.append(get_energy_x(i*pixel_size))
        E_y_binedges.append(get_energy_y(i*pixel_size))
    #E_x_binedges = E_x_binedges[::-1] # reorder so smallest energy at the start
    E_y_binedges = E_y_binedges[::-1]
    
    #vertical_count = [] 
    #for i in range(map_size):
        #vcount = sum(pixel_map[i])
        #vertical_count.append(vcount)
    horizontal_count = [] 
    for i in range(map_size):
        hcount = sum(pixel_map.T[i])
        horizontal_count.append(hcount)
    #vertical_count = vertical_count[::-1]
    horizontal_count = horizontal_count[::-1]
    """
    def getbinheight(bin_edges, count):
        heights = []
        widths = []
        N = sum(sum(pixel_map))
        first_count = 0
        widths.append(bin_edges[-(data_points+1)]-bin_edges[0])
        for i in range(len(count)+1-data_points):
            first_count += count[i]
        heights.append(first_count / (widths[0] * N))
        for i in range(len(count)-data_points,len(count)-1):
            width = bin_edges[i+1] - bin_edges[i]
            heights.append(count[i] / (width * N))
            widths.append(width)
            
        return np.array(heights),np.array(widths) # normalised probability density and width
    """
    def getbinheight(bin_edges, count):
        heights = []
        widths = []
        N = sum(sum(pixel_map))
        for i in range(len(count)):
            width = bin_edges[i+1] - bin_edges[i]
            heights.append(count[i] / (width * N))
            widths.append(width) 
        widths[-1] = 0.01
    
        return np.array(heights),np.array(widths) # normalised probability density and width
    
    #x_heights, x_widths = getbinheight(E_x_binedges, vertical_count)
    
    #E_x_bincentres = np.array(E_x_binedges[:-1]) + x_widths / 2
    """
    E_x_bincentres = []
    E_x_bincentres.append(E_x_binedges[0] + x_widths[0] / 2)
    for i in range(1,data_points):
        E_x_bincentres.append(E_x_binedges[i+len(vertical_count)-1-data_points] + x_widths[i] / 2)
    """
    y_heights, y_widths = getbinheight(E_y_binedges, horizontal_count)
    
    E_y_bincentres = np.array(E_y_binedges[:-1]) + y_widths / 2
    print(E_y_bincentres)
    
    """
    E_y_bincentres = []
    E_y_bincentres.append(E_y_binedges[0] + y_widths[0] / 2)
    for i in range(1,data_points):
        E_y_bincentres.append(E_y_binedges[i+len(horizontal_count)-1-data_points] + y_widths[i] / 2)
    """
    # rewrite the variables to remove data points that have no count
    
    y_widths = y_widths[E_y_bincentres > energy_cut]
    y_heights = y_heights[E_y_bincentres > energy_cut]
    E_y_bincentres = E_y_bincentres[E_y_bincentres > energy_cut]
    
    def getMaxEnergy():
        
        maxE  = E_y_bincentres[np.argwhere(y_heights>0.01)[-1][0]]
        
        return maxE
    
    def getCount():
        N = sum(sum(pixel_map))
        return N / 0.1 # weight
    
    maxE = getMaxEnergy()
    
    count = getCount()
    
    return E_y_bincentres, y_heights, y_widths, maxE, count # errorbars

def BoltzmannPDF(E, A, T):
    pdf = A * np.exp(-E / T)
            
    return pdf

def linear(x, m, c):
    y = m*x + c
            
    return y
    
def Boltzmannfit(x,y):
    try:
        popt, pcov = curve_fit(BoltzmannPDF,x,y)
        predictedA = popt[0]
        predictedT = popt[1]
        #err = np.sqrt(pcov[1][1])
        #print(f"Predicted T: {predictedT} /pm {err} MeV")
    except:
        predictedA, predictedT = 0,0
    
    return predictedA, predictedT

def Linearfit(x,y):
    popt, pcov = curve_fit(linear,x,y)
    m = popt[0]
    c = popt[1]
    
    return m,c

def plot_thermal_distribution(x,y,predictedA,predictedT,widths):
    Range = np.linspace(min(x),x[:-2],100)
    plt.plot(Range,BoltzmannPDF(Range,predictedA,predictedT), "--", color="black", label="Fit")
    plt.errorbar(x[:-2],y[:-2],xerr=widths[:-2]/2,fmt="x",capsize=3, label="Bin centre") 
    plt.legend()
    plt.xlabel(r"Energy $E$ (MeV)", fontsize=14)
    plt.ylabel(r"Probability density $p(E)$", fontsize=14)
    plt.yscale("log")
    

#%%

x_range_mm = [0, 35]
y_range_mm = [0, 35]

weight = 0.1
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 2

proton_pixel_map_64, combined_pixel_map_64, variables = data.getMultiTrackData("Dataset4",["Thermal energy of beam (MeV)","Maximum energy of beam (MeV)","Number of accepted particles in beam"])
thresh_combined_pixel_map_64 = data.add_noise(combined_pixel_map_64, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
thresh_proton_pixel_map_64 = data.add_noise(proton_pixel_map_64, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
noisy_combined_pixel_map_64 = data.add_noise(combined_pixel_map_64, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
noisy_proton_pixel_map_32 = data.add_noise(proton_pixel_map_32, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
multiplied_variables = np.repeat(variables, multiplier, axis=0)

#%%

B_field_mT = 100
E_field_kV = 2
mass_MeV = 938.28
charge_e = 1
field_length_mm = 50
distance_mm = 125

E_y_bincentres, y_heights, y_widths, maxE, count = construct_thermal_distribution(thresh_proton_pixel_map_64[0], 0.1, x_range_mm, y_range_mm, charge_e, B_field_mT, E_field_kV, field_length_mm, distance_mm, mass_MeV, title="Energy distribution")

predictedA, predictedT = Boltzmannfit(E_y_bincentres,y_heights)

plot_thermal_distribution(E_y_bincentres,y_heights,predictedA,predictedT,y_widths)

#%%
clean_x_train, clean_x_test, clean_y_train, clean_y_test = train_test_split(thresh_combined_pixel_map_64, multiplied_variables, test_size=0.2, random_state=42)
noisy_x_train, noisy_x_test, noisy_y_train, noisy_y_test = train_test_split(noisy_proton_pixel_map_32, multiplied_variables, test_size=0.2, random_state=42)
ppppproton_pixel_map_64 = []
median_combined_pixel_map_64 = []
median_proton_pixel_map_64 = []
median_proton_pixel_map_64 = []
median_proton_pixel_map_32 = []
#squeezed_noisy_x_test = np.squeeze(noisy_x_test, -1)
#%%
t = time.time()

for noisy_map in noisy_x_test:
    filtered_map = remove_noise(noisy_map)
    median_proton_pixel_map_32.append(filtered_map)
elapsed_time = time.time() - t
print("Denoising took ", elapsed_time)

#%%

t = time.time()

manual_proton_variablesList = []

for filtered_map in median_proton_pixel_map_32:
    E_y_bincentres, y_heights, y_widths, maxE, count = construct_thermal_distribution(filtered_map, 0.08, x_range_mm, y_range_mm, charge_e, B_field_mT, E_field_kV, field_length_mm, distance_mm, mass_MeV, title="Energy distribution")
    predictedA, predictedT = Boltzmannfit(E_y_bincentres[:-1],y_heights[:-1])
    #predictedT = - np.log(np.e) / m
    variables = np.array([predictedT,maxE,count])
    manual_proton_variablesList.append(variables)
elapsed_time = time.time() - t
print("Denoising took ", elapsed_time)

#%%
from sklearn.model_selection import train_test_split
proton_x_train, proton_x_test, proton_y_train, proton_y_test = train_test_split(thresh_proton_pixel_map_64, multiplied_variables, test_size=0.2, random_state=42)

#squeezed_proton_x_test = np.squeeze(proton_x_test, -1)

#%%

t = time.time()

proton_variablesList = []

for mapp in proton_x_test:
    E_y_bincentres, y_heights, y_widths, maxE, count = construct_thermal_distribution(mapp, 0.08, x_range_mm, y_range_mm, charge_e, B_field_mT, E_field_kV, field_length_mm, distance_mm, mass_MeV, title="Energy distribution")
    predictedA, predictedT = Boltzmannfit(E_y_bincentres[:-1],y_heights[:-1])
    #predictedT = - np.log(np.e) / m
    variables = np.array([predictedT,maxE,count])
    proton_variablesList.append(variables)
elapsed_time = time.time() - t
print("Denoising took ", elapsed_time)

#%%
from matplotlib.colors import LogNorm

fig, axs = plt.subplots(1,2)
img1 = axs[0].imshow(median_combined_pixel_map_64[2].T, cmap='hot', origin='lower', norm=LogNorm(vmin=0.01, vmax=255))
axs[0].set_xlabel(r'Position $x$ (pixel)')
axs[0].set_title('Filtered image')
fig.colorbar(img1, ax=axs[0])
img2 = axs[1].imshow(thresh_proton_pixel_map_64[2].T, cmap='hot', origin='lower', norm=LogNorm(vmin=0.01, vmax=255))
axs[1].set_xlabel(r'Position $x$ (pixel)')  
axs[0].set_ylabel(r'Position $y$ (pixel)')    
axs[1].set_title('Clean proton track')
fig.colorbar(img2, ax=axs[1])
#fig.colorbar(img2)

#%%

a = np.array([14.315392,14.287662,13.876450,14.382810,13.529870])

b = np.array([18.187136,17.976140,20.253621,18.921891,18.626251])

c = a+b