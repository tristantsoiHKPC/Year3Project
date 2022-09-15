# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 03:00:05 2022

@author: user
"""
#%%
from Data_Synthesis import ThomsonParabolaSpec, OutputPlane, Beam, Run, Image
import matplotlib.pyplot as pl
import random
import pickle
import os
import time
import numpy as np
import glob
from matplotlib.colors import LogNorm

def GenerateDataset(Beam, Thomson, Detector, thermal_E_MeV, Num_particles, E_max_MeV, dataset_size, foldername, x_range_mm, y_range_mm, Propagation_step_size = 9 * 10 ** -11):
    config = dict()
    config["Number of particles range"] = Num_particles
    config["Thermal energy range (MeV)"] = thermal_E_MeV
    config["Maximium energy range (MeV)"] = E_max_MeV
    config["Image x Range (mm)"] = x_range_mm
    config["Image y Range (mm)"] = y_range_mm
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    with open(f"{foldername}/Simulation Configurations.obj","wb") as f0:
        pickle.dump(config, f0)
        
    for i in range(dataset_size):
        dataset = dict()
        start = time.time()
        
        E_MeV = random.uniform(thermal_E_MeV[0], thermal_E_MeV[1])
        E_max = random.uniform(E_max_MeV[0], E_max_MeV[1])
        Num = random.randint(Num_particles[0], Num_particles[1])
        
        dataset["Thermal energy of beam (MeV)"] = E_MeV
        dataset["Maximum energy of beam (MeV)"] = E_max
        dataset["Number of particles in beam"] = Num 
        
        Particles = Beam.generate_beam(Num, E_MeV, E_max)
    
        res = Run(Particles, Thomson, Detector, Propagation_step_size)
        
        data = res.copy()
        dataset["Number of accepted particles in beam"] = len(data[0])

            
        h = Image.generate_image(data[0], data[1], data[2], [32, 32], x_range_mm, y_range_mm)
        j = Image.generate_image(data[0], data[1], data[2], [64, 64], x_range_mm, y_range_mm)
        k = Image.generate_image(data[0], data[1], data[2], [128, 128], x_range_mm, y_range_mm)
        dataset["Pixels map 28 x 28"] = h.copy()
        dataset["Pixels map 64 x 64"] = j.copy()
        dataset["Pixels map 128 x 128"] = k.copy()
      
        print(f"Image {i} completed")
        end = time.time()
        dataset["Run time(s)"] = end - start

        with open(f"{foldername}/{time.time()}.obj","wb") as f0:
            pickle.dump(dataset, f0)

def GenerateMultiTracks(ProtonBeam, CarbonIonBeamList, Thomson, Detector, thermal_E_MeV, Num_particles, E_max_MeV, dataset_size, foldername, x_range_mm, y_range_mm, Propagation_step_size = 9 * 10 ** -11):
    # storing only configuration of protons
    config = dict()
    config["README"] = "Config of proton beam only"
    config["Number of particles range"] = Num_particles
    config["Thermal energy range (MeV)"] = thermal_E_MeV
    config["Maximium energy range (MeV)"] = E_max_MeV
    config["Image x Range (mm)"] = x_range_mm
    config["Image y Range (mm)"] = y_range_mm
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    with open(f"{foldername}/Simulation Configurations.obj","wb") as f0:
        pickle.dump(config, f0)
        
    for i in range(dataset_size):
        dataset = dict()
        dataset["Protons"] = dict()
        dataset["Carbon ions"] = dict()
        start = time.time()
        
        E_MeV = random.uniform(thermal_E_MeV[0], thermal_E_MeV[1])
        E_max = random.uniform(E_max_MeV[0], E_max_MeV[1])
        Num = random.randint(Num_particles[0], Num_particles[1])
        
        dataset["Protons"]["Thermal energy of beam (MeV)"] = E_MeV
        dataset["Protons"]["Maximum energy of beam (MeV)"] = E_max
        dataset["Protons"]["Number of particles in beam"] = Num 
        
        # same as protons'
        dataset["Carbon ions"]["Thermal energy of beam (MeV)"] = E_MeV
        # same as protons'
        dataset["Carbon ions"]["Maximum energy of beam (MeV)"] = E_max
        # carbon ions are 1:100 to protons
        CarbonNum = int(Num/100)
        dataset["Carbon ions"]["Number of particles in beam"] = CarbonNum 
        
        # store list of particles
        ProtonParticles = ProtonBeam.generate_beam(Num, E_MeV, E_max)
        ProtonResults = Run(ProtonParticles, Thomson, Detector, Propagation_step_size)
        CarbonParticles = []
        for beam in CarbonIonBeamList:
            CarbonParticles += beam.generate_beam(CarbonNum, E_MeV, E_max)
        CarbonResults = Run(CarbonParticles, Thomson, Detector, Propagation_step_size)
        
        ProtonData = ProtonResults.copy()
        CarbonData = CarbonResults.copy()
        
        # storing for protons only
        dataset["Protons"]["Number of accepted particles in beam"] = len(ProtonData[0])
        
        proton_map_32 = Image.generate_image(ProtonData[0], ProtonData[1], ProtonData[2], [32, 32], x_range_mm, y_range_mm)
        carbon_map_32 = Image.generate_image(CarbonData[0], CarbonData[1], CarbonData[2], [32, 32], x_range_mm, y_range_mm)
        combined_map_32 = np.array(proton_map_32) + np.array(carbon_map_32)
        proton_map_64 = Image.generate_image(ProtonData[0], ProtonData[1], ProtonData[2], [64, 64], x_range_mm, y_range_mm)
        carbon_map_64 = Image.generate_image(CarbonData[0], CarbonData[1], CarbonData[2], [64, 64], x_range_mm, y_range_mm)
        combined_map_64 = np.array(proton_map_64) + np.array(carbon_map_64)
        
        dataset["Proton pixel map 32 x 32"] = proton_map_32
        dataset["Combined pixel map 32 x 32"] = combined_map_32
        dataset["Proton pixel map 64 x 64"] = proton_map_64
        dataset["Combined pixel map 64 x 64"] = combined_map_64

        print(f"Proton image {i} and combined image {i} completed")
        end = time.time()
        dataset["Run time(s)"] = end - start

        with open(f"{foldername}/{time.time()}.obj","wb") as f0:
            pickle.dump(dataset, f0)
            
def getData(foldername):
    pixel_map_32 = []
    pixel_map_64 = []
    T = []
    path = f'{foldername}'
    files = glob.glob(os.path.join(path, '*.obj'))
    print(f"Total number of files {len(files)}")
    init = 0
    for filename in files:
        with open(filename,"rb") as f0:
            if filename != path + "\Simulation Configurations.obj":
                dataset = pickle.load(f0)   
                try:
                    pixel_map_32.append(dataset["Pixels map 32 x 32"])
                    pixel_map_64.append(dataset["Pixels map 64 x 64"])
                    T.append(dataset["Thermal energy of beam (MeV)"])
                except:
                    print(f"Corrupt data file {filename}")
                init += 1 
                print(f"{init} / {len(files)}")
    pixel_map_32 = np.array(pixel_map_32)
    pixel_map_64 = np.array(pixel_map_64)
    T = np.array(T)

    return pixel_map_32, pixel_map_64, T

def getMultiTrackData(foldername, VariableList):
    #proton_pixel_map_32 = []
    #combined_pixel_map_32 = []
    proton_pixel_map_64 = []
    combined_pixel_map_64 = []
    variables = []
    path = f'{foldername}'
    files = glob.glob(os.path.join(path, '*.obj'))
    print(f"Total number of files {len(files)}")
    init = 0
    for filename in files:
        with open(filename,"rb") as f0:
            if filename != path + "\Simulation Configurations.obj":
                dataset = pickle.load(f0)   
                try:
                    #proton_pixel_map_32.append(dataset["Proton pixel map 32 x 32"])
                    #combined_pixel_map_32.append(dataset["Combined pixel map 32 x 32"])
                    proton_pixel_map_64.append(dataset["Proton pixel map 64 x 64"])
                    combined_pixel_map_64.append(dataset["Combined pixel map 64 x 64"])
                    variables_single_map = []
                    for variablename in VariableList:
                        variables_single_map.append(dataset["Protons"][variablename])
                    variables.append(variables_single_map)
                except:
                    print(f"Corrupt data file {filename}")
                init += 1 
                print(f"{init} / {len(files)}")
    proton_pixel_map_64, combined_pixel_map_64 = np.array(proton_pixel_map_64), np.array(combined_pixel_map_64)
    #proton_pixel_map_64, combined_pixel_map_64 = np.array(proton_pixel_map_64), np.array(combined_pixel_map_64)

    variables = np.array(variables)

    #return proton_pixel_map_32, combined_pixel_map_32, proton_pixel_map_64, combined_pixel_map_64, variables
    return proton_pixel_map_64, combined_pixel_map_64, variables

def add_noise(pixel_map, dataset_multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, Collimator_diameter_micron, x_range, y_range):
    noisy_pixel_map = []
    print(f"Total Dataset Size: {len(pixel_map)}")
    print(f"Mutiplier: {dataset_multiplier}")
    Total_size = len(pixel_map) * dataset_multiplier
    print(f"Noisy Dataset Size: {Total_size}")
    init = 0
    for h in range(len(pixel_map)):
        for i in range(dataset_multiplier):
            init += 1 
            print(f"{init} / {Total_size}")
            res = Image.add_noise(pixel_map[h], weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, Collimator_diameter_micron, x_range, y_range)
            noisy_pixel_map.append(res.copy())

    return np.array(noisy_pixel_map)

#Define System Here

Collimator_mm = 120
Collimator_diameter_micron = 200

E_strength_kV = 2
B_strength_mT = r'circular_magnet.mat'
Thomson_Plane_mm = 170
dimensions_mm = [10, 50, 50]

Detector_Plane_mm = 320
Detector_dimensions_mm = [100, 100]

#Define Beam Here
ProtonMass_MeV = 938.28
ProtonCharge_e = 1

CarbonMass_MeV = 12 * ProtonMass_MeV

# Define Image parameters here
x_range_mm = [0, 35]
y_range_mm = [0, 35]

# learning parameters    (Need to input a range)
thermal_E_MeV = [0.1, 6]    #Teff
Num_particles = [18000, 24000]
E_max_MeV = [4, 10]

dataset_size = 1
foldername = "Test"
"""
Thomson = ThomsonParabolaSpec(E_strength_kV, B_strength_mT, Thomson_Plane_mm, dimensions_mm)
Detector = OutputPlane(Detector_Plane_mm, Detector_dimensions_mm)
ProtonBeam = Beam(Collimator_mm, Collimator_diameter_micron, ProtonMass_MeV, ProtonCharge_e)

CarbonIonBeamList = []
CarbonChargeList = [1,2,3,4]
for charge_e in CarbonChargeList:
    CarbonIonBeamList.append(Beam(Collimator_mm, Collimator_diameter_micron, CarbonMass_MeV, charge_e))

#dataset = GenerateMultiTracks(ProtonBeam, CarbonIonBeamList, Thomson, Detector, thermal_E_MeV, Num_particles, E_max_MeV, dataset_size, foldername, x_range_mm, y_range_mm)

proton_pixel_map_32, combined_pixel_map_32, proton_pixel_map_64, combined_pixel_map_64, variables = getMultiTrackData(foldername, None)

pl.imshow(combined_pixel_map_64.T, cmap='hot', origin='lower', norm=LogNorm(vmin=0.01, vmax=255))
pl.colorbar()
#%%
pl.imshow(proton_pixel_map_64.T, cmap='hot', origin='lower', norm=LogNorm(vmin=0.01, vmax=255))
pl.colorbar()
"""

