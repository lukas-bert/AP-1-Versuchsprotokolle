import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp 
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs


n1 = np.genfromtxt("content/data/data1.txt", unpack = True)
n2 = np.genfromtxt("content/data/data2.txt", unpack = True)

u = 1/5.046         # Untersetzungsverhältnis
d = 5e-3            # m Verschiebung des Spiegels 
laser = 635e-9      # m Wellenlänge des Lasers
b = 50e-3           # m Schichtdicke der Messzelle

T_0 = 273.15        # K
T   = 295.14        # K, Raumtemperatur (22°C)
p_0 = 101325        # Pascal
d_p =  60000        # Pascal Druch nach Vakkumpumpen

z1 = ufloat(np.mean(n1), np.std(n1))
print("Mittelwert der Zählraten: ", z1)

laser_exp = 2*d*u/z1
d_laser = np.abs(noms(laser_exp)- laser)/laser

print("Experimentelle Wellenlänge: ", '{0:.5e}'.format(laser_exp))
print("Abweíchung:                 ", d_laser)


