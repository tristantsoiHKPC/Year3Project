
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:29:20 2022

@author: user
"""

import numpy as np
from scipy import constants
import matplotlib.pyplot as pl
import scipy.io
from scipy import interpolate
import time
from scipy.ndimage import gaussian_filter

pl.rcParams.update({'font.size': 14})

class Particle:
    
    def convertKEtovel(self, mass_MeV, energy_MeV):
        return (2 * energy_MeV / mass_MeV) ** 0.5 * constants.c
    
    def __init__(self, mass_MeV, charge_e, energy_MeV, position_mm = [0,0,0], vel_unit = [0,0,1]):
        self._mass = mass_MeV * 10 ** 6 * constants.e / constants.c ** 2
        self._charge = charge_e * constants.e
        self._energy = energy_MeV
        self._position = [np.array(position_mm) * 10 ** -3]
        self._velocity = [self.convertKEtovel(mass_MeV, energy_MeV) * np.array(vel_unit)]
        self._valid = True
        
    def mass_kg(self):
        return self._mass
    
    def charge_C(self):
        return self._charge
    
    def pos_m(self):
        return self._position[-1]
    
    def vel_m(self):
        return self._velocity[-1]
    
    def update(self, new_pos, new_vel):
        self._position.append(new_pos)
        self._velocity.append(new_vel)
    
    def vertices(self):
        return self._position
    
    def velocities(self):
        return self._velocity
    
    def invalidparticle(self):
        self._valid = False
        
    def valid(self):
        return self._valid
    
class OutputPlane:
    
    def __init__(self, z0_mm, dimensions_mm):
        self._z0 = z0_mm * 10 ** -3 # Convert from mm to m 
        self._dimensions = np.array(dimensions_mm) * 10 ** -3
        
    def normalise(self, vector):
        norm = vector / np.linalg.norm(vector)
        return norm
        
    def intercept (self, particle):
        """Find intercept of a particle with an Output Plane at z0
        
        Keyword arguments:
        Particle -- Class Particle
        
        Returns: 
            output: 3-d ndarray - returns the point of intercept of the ray.
                    None - if ray does not intercect with the element. 
        """
        v = particle.vel_m()
        pos = particle.pos_m()
        
        #Finds the interept with linear extrapolation
        k_hat = self.normalise(v)
        l = -(pos[2] - self._z0) / k_hat[2]            
        IntersectionPoint = pos + l * k_hat
           #Checks ray is not going backwards and intersects with the plane
        if l > 0 and abs(IntersectionPoint[0]) < self._dimensions[0] / 2 and abs(IntersectionPoint[1]) < self._dimensions[1] / 2:
            return IntersectionPoint
        else: 
            particle.invalidparticle()
            print("Particle is not valid")
        
        
    def propagate(self, particle):
        if particle.valid():
            particle.update(self.intercept(particle), particle.vel_m())
        
class ThomsonParabolaSpec:
    def __init__(self, E_strength_kV, B_strength_mT, z0_mm, dimensions_mm, field_dir = [1,0,0]):
        
        """
        The B_strength_mT here could be a number or a string.
        
        If number is enter, it would represent a uniform magnetic field.
        
        Otherwise, the string name of the matfile could also be entered.
        
        """
        self._dimensions = np.array(dimensions_mm) * 10 ** -3
        self._E_strength = E_strength_kV * 10 ** 3 / self._dimensions[0]
        
        if type(B_strength_mT) == str:
            self._B_strength = B_strength_mT
            
            magnet_map=scipy.io.loadmat(self._B_strength)
            z_mag=magnet_map['x_mag'][0]
            y_mag=magnet_map['y_mag'][0]
            B=magnet_map['B_field_map'] * 10 ** -3
            f = interpolate.interp2d(z_mag, y_mag, B, kind='cubic')
            self._B_func = f
            self._B_y_min = y_mag[0]
            self._B_y_max = y_mag[-1]
            self._B_z_min = z_mag[0]
            self._B_z_max = z_mag[-1]
        else:
            self._B_strength = B_strength_mT * 10 ** - 3 
        self._z0 = z0_mm * 10 ** -3  

        self._B_radius = self._dimensions[2] / 2
        self._field_dir = np.array(field_dir)
        if self._z0 >= 0:
            self._center = self._z0 + self._B_radius
        else: 
            self._center = self._z0 - self._B_radius
    
    def B_field(self, particle):
        x,y,z = particle.pos_m()
        
        z_centre = z - self._center  # Defines the position of particle relative to the centre of the ThomsonSpec
        
        if type(self._B_strength) == str:
                return self._B_func(z_centre, y) * self._field_dir
        else:
            if (z_centre) **2 + y ** 2 <= self._B_radius ** 2:
                return self._B_strength * self._field_dir
            else:
                return np.array([0,0,0])
        
    def E_field(self, particle):
        x,y,z = particle.pos_m()
        
        if z >= self._z0 and z <= self._z0 + self._dimensions[2]:
            return self._E_strength  * self._field_dir
        else:
            return np.array([0,0,0])
        
    def Lorentzacc(self, particle):
        q = particle.charge_C()
        m = particle.mass_kg()
        v = particle.vel_m()
        
        return q / m * (self.E_field(particle) + np.cross(v, self.B_field(particle)))
    
    def EulerPusher(self, particle, step):
        x = particle.pos_m() 
        v = particle.vel_m() 
        
        x_new = x + v * step
        v_new = v + self.Lorentzacc(particle) * step
        
        particle.update(x_new, v_new)
        
    def BorisPusher(self, particle, step):
        """
        Implementation reference from:
            
        https://github.com/iwhoppock/boris-algorithm/blob/master/boris%20in%20C/boris.c
        https://www.particleincell.com/2011/vxb-rotation/
        """
        x = particle.pos_m() 
        v = particle.vel_m()
        
        
        B = self.B_field(particle)
        E = self.E_field(particle)
        m = particle.mass_kg()
        q = particle.charge_C()
        
        t = B * q * step / (2 * m) 
        tsqr = np.dot(t, t)
        
        s = 2 * t / (1 + tsqr)
        
        vminus = v + q * E * step / (2 * m)
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)
        
        v_new = vplus +  q * E * step / (2 * m)
        x_new = x + v_new * step
        
        particle.update(x_new, v_new)

    def propagate(self, particle, step = 10 ** -12, method = "Boris"):
        if particle.valid():
            # Find Intercept Point with the Apparatus
            Intercept_Plane = OutputPlane(self._z0 * 10 ** 3, [self._dimensions[0] * 10 ** 3, self._dimensions[1] * 10 ** 3]) #Output plane takes in mm as in the input
            Intercept_Plane.propagate(particle) 
            #Valid Entering Particle
                
            while particle.pos_m()[2] <= self._z0 + self._dimensions[2] and particle.valid():
                
                if method == "Boris":
                    self.BorisPusher(particle, step)
                else: 
                    self.EulerPusher(particle, step)
                
                # Particle hits the side of the Thomson spec
                if abs(particle.pos_m()[0]) >= self._dimensions[0] / 2 or abs(particle.pos_m()[1]) >= self._dimensions[1] / 2:
                    particle.invalidparticle()
                    #print("Particle hit the side")
                
    
            
class Beam:
    def __init__(self, z0_mm, diameter_micron, mass_MeV, charge_e):# check
        self._z0 = z0_mm 
        self._radius = diameter_micron / 2 * 1e-3
        self._m = mass_MeV * 1e6 * constants.e / constants.c ** 2
        self._charge_e = charge_e
        self._mass_MeV = mass_MeV


    def sample_Boltzmann(self, particle_num, thermal_E_MeV, E_max_MeV, E_min_MeV= 0):
        thermal_E = thermal_E_MeV * 1e6 * constants.e
        E_min = E_min_MeV * 1e6 * constants.e
        E_max = E_max_MeV * 1e6 * constants.e
        E_list = []
        comparison = 1 / (thermal_E) 
        while len(E_list) < particle_num:
            E = np.random.uniform(E_min, E_max)
            pdf = 1 / (thermal_E) * np.exp(-E / thermal_E)
            if pdf > np.random.uniform(0, comparison): # comparison function defined by most probable energy which is E = kT
                E_list.append(E) # append accepted energy
        return E_list
    """
    Sample uniformly inside the circular collimator 
    """
    def normalise(self, vector):
        norm = vector / np.linalg.norm(vector)
        return norm
    
    def sample_collimator(self, particle_num):
        pos_list = []
        vel_unit_list = []
        while len(pos_list) < particle_num:
            x = np.random.uniform(-self._radius, self._radius)
            y = np.random.uniform(-self._radius, self._radius)
            if x ** 2 + y ** 2 < self._radius ** 2:
                pos = np.array([x, y, self._z0])
                vel_unit = self.normalise(pos)
                pos_list.append(pos)
                vel_unit_list.append(vel_unit)
        return pos_list, vel_unit_list
                
    """
    Generate the positions and velocities of the particles (currently last updated at the collimator)
    """
    def generate_beam(self, particle_num, thermal_E_MeV, E_max_MeV, E_min_MeV= 0):  # of one particle type
        Beam = []
        E_list = self.sample_Boltzmann(particle_num, thermal_E_MeV, E_max_MeV, E_min_MeV= 0)
        pos_list, vel_unit_list = self.sample_collimator(particle_num)
        E_MeV = np.array(E_list) * 10 ** -6 / constants.e 
        for i in range(particle_num):
            particle = Particle(self._mass_MeV, self._charge_e, E_MeV[i], pos_list[i], vel_unit_list[i])
            Beam.append(particle)
        return Beam
    

def Run(beam, Thomson, Detector, step):
    start = time.time()
    x_list, y_list = [], []
    x_init_list, y_init_list = [], []
    E_list = []

    for i in range(len(beam)):

        Thomson.propagate(beam[i], step)
        Detector.propagate(beam[i])

        if beam[i].valid():
            x_init = beam[i].vertices()[0][0]
            y_init = beam[i].vertices()[0][1]
            E_list.append(beam[i]._energy)
            x = beam[i].pos_m()[0]
            y = beam[i].pos_m()[1]
                    
            x_list.append(x)
            y_list.append(y)
            x_init_list.append(x_init)
            y_init_list.append(y_init)
    end = time.time()
    print("Total number of accepted particles:" + f"{len(x_list)}", "Execution time:" + f"{end - start}")
    return [x_list, y_list, E_list, x_init_list, y_init_list]        

class Image:
    def generate_image(x_pos, y_pos, E_list, pixels, x_range_mm, y_range_mm):
        def weighting(E_list):
            """
            Flat weighting is applied here, alternatively energy dependent pixel weighting could 
            also be applied. 
            """
            return [1] * len(E_list)
        x_range = np.array(x_range_mm) * 10 ** -3
        y_range = np.array(y_range_mm) * 10 ** -3
        
        res = pl.hist2d(x_pos, y_pos, pixels, [x_range, y_range], cmap="hot", weights = weighting(E_list))
        return res[0]
    
    def add_noise(pixels_map, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, Collimator_diameter_micron, x_range, y_range):
        h = pixels_map.copy()
    
        def uniform_background():
            for i in range(len(h)):
                for j in range(len(h[i])):
                    h[i][j] += background_noise
                    
        def convolution():
            return gaussian_filter(h, sigma = abberations)
        
        def hard_hit():
            x = np.random.randint(0, len(h))
            y = np.random.randint(0, len(h[0]))
            amp = np.random.randint(0, detector_threshold)
            h[x][y] += amp
            
        def zero_point():
            """
            Adds in saturation particles according to the collmator size
            """
            bin_edges_x = np.linspace(x_range[0], x_range[1], len(h), endpoint = False)
            bin_edges_y = np.linspace(y_range[0], y_range[1], len(h[0]), endpoint = False)
            
            for i in range(len(bin_edges_x)):
                for j in range(len(bin_edges_y)):
                    if bin_edges_x[i] + bin_edges_y[j] ** 2 < Collimator_diameter_micron * 10 ** -3:
                        h[i][j] += detector_threshold
              
        h = pixels_map.copy() * weight
        zero_point()
        """
        Applies a flat weighting on the image to avoid saturations. 
        If energy dependent weighting is needed, change in generate_image
        function. 
        """
        uniform_background()
        h = convolution()
        for i in range(int(hard_hit_noise_fraction * len(h) * len(h[0]))):
            hard_hit()
            
        for i in range(len(h)):
            for j in range (len(h[0])):
                h[i][j] = int(h[i][j])
                if h[i][j] > detector_threshold:
                    h[i][j] = detector_threshold    
        return h 
# General Investigation tools

def validatedeflection(mass_MeV, charge_e, energy_min_MeV, energy_max_MeV, step, Thomson, Detector, method, num_step = 10 ** -12):
    """
        Plots a graph of percentage difference in numerically calculated
        deflection (x and y direction) and analytical solution against energy
        
        NB: Uniform B field is required for the Thomson Spectrometer
    """
        
    Energy = np.arange(energy_min_MeV, energy_max_MeV, step)
    x_diff_list = []
    y_diff_list = []
    
    for i in range(len(Energy)):
        particle = Particle(mass_MeV, charge_e, Energy[i])
        Thomson.propagate(particle, num_step, method)
        Detector.propagate(particle)
        
        x_numerical = particle.pos_m()[0]
        y_numerical = particle.pos_m()[1]
        
        q = particle.charge_C()
        E = Thomson._E_strength
        B = Thomson._B_strength
        l = Thomson._dimensions[2]
        D = Detector._z0 - (Thomson._z0 + l) + l / 2 
        E_kin = particle._energy * constants.e * 10 ** 6
        m = particle.mass_kg()
        
        x_analytical = q * E * l * D / (2 * E_kin)
        y_analytical = q * B * l * D / (2 * E_kin * m) ** 0.5
        
        x_diff = abs(x_analytical - x_numerical) * 100 / x_analytical
        y_diff = abs(y_analytical - y_numerical) * 100 / y_analytical
        
        x_diff_list.append(x_diff)
        y_diff_list.append(y_diff)
        
    pl.plot(Energy, np.array(x_diff_list), label = "x-axis")
    #pl.plot(Energy, y_diff_list, label = "y_diff")
    pl.xlabel("Energy (MeV)")
    pl.ylabel("Absolute percentage error (%)")
    pl.legend()
    pl.show()
    
    x_average_diff = sum(x_diff_list) / len(x_diff_list)
   
    y_average_diff = sum(y_diff_list) / len(y_diff_list)
 
    
    return x_average_diff, y_average_diff

#%%
"""
E_strength_kV = 2
B_strength_mT = 100
#B_strength_mT = r'circular_magnet.mat'
Thomson_Plane_mm = 170
dimensions_mm = [10, 50, 50]

Detector_Plane_mm = 320
Detector_dimensions_mm = [100, 100]
mass_MeV = 938.28
charge_e = 1
energy_min_MeV = 0.1
energy_max_MeV = 10
step = 0.1
Thomson = ThomsonParabolaSpec(E_strength_kV, B_strength_mT, Thomson_Plane_mm, dimensions_mm)
Detector = OutputPlane(Detector_Plane_mm, Detector_dimensions_mm)
method = "Euler"
num_step = 9 * 10 ** -11

Collimator_mm = 120
Collimator_diameter_micron = 200

#%%

ProtonBeam = Beam(Collimator_mm, Collimator_diameter_micron, mass_MeV, charge_e)
Particles = ProtonBeam.generate_beam(10000, 5, 10)
res = Run(Particles, Thomson, Detector, num_step)

#%% validation

x_av, y_av = validatedeflection(mass_MeV, charge_e, energy_min_MeV, energy_max_MeV, step, Thomson, Detector, method)

#%%

q = charge_e * 1.6e-19
B = 100 * 1e-3
l = 50 * 1e-3
E = E_strength_kV * 1e3 / l
D = 125 * 1e-3 # distance from centre of Thomson spec to detector
m = mass_MeV * (1e6*scipy.constants.e/scipy.constants.c**2)
#%%
pl.scatter(res[2],np.array(res[1])*1000, marker="x", label="Experimental magnetic field")
RANGE = np.linspace(0.005,10,10000)
pl.scatter(RANGE, q * B * l * D / (2 * RANGE * m) ** 0.5 * 1000000000, marker="x", label="Uniform magnetic field")
pl.xlabel(r"Energy $E$ (MeV)")
pl.ylabel(r"Position $y$ (mm)")
pl.xlim(0,3)
pl.legend()
"""