import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


# Messaufgabe 2: Brechung

a, b = np.genfromtxt("content/data/Brechung.txt", unpack = True)

a = unp.uarray(a, np.ones(len(a)))*np.pi/180 # Eintrittswinkel alpha in rad, Fehler jeweils 1°
b = unp.uarray(b, np.ones(len(a)))*np.pi/180 # Austrittsinkel beta in rad, "

n = unp.sin(a)/unp.sin(b)

n_mean = 0
for i in range(len(n)):
    n_mean = n_mean + n[i]

n_mean = n_mean/len(n)

print("-------------------------------------------------------------------------------")
print("Brechungsindizes Plexiglas:")
print(n)
print("Mittelwert:", n_mean)
print("-------------------------------------------------------------------------------")

# Strahlenversatz (3)

d= 5.85 #cm

s_m1=d*(unp.sin(a-b))/(unp.cos(b))  # Strahlenversatz in cm

beta_calc=unp.arcsin(unp.sin(a)/n_mean)      # Winkel beta über Brechungsindex n_mean
s_m2=d*(unp.sin(a-beta_calc))/(unp.cos(beta_calc)) # Strahlenversatz in cm

print("-------------------------------------------------------------------------------")
print(f'Strahlenversatz 1.Methode: {s_m1}')
print("-------------------------------------------------------------------------------")
print(f'Strahlenversatz 2.Methode: {s_m2}')
print("-------------------------------------------------------------------------------")

# Messaufgabe 3 (4): Prisma

n_kron = 1.5067         # Brechungsindex Kronglas
gamma = 60 * np.pi/180  # Prismenwinkel in rad

a1, a2_g, a2_r = np.genfromtxt("content/data/Prisma.txt", unpack = True)

a1 = unp.uarray(a1, np.ones(len(a1)))*np.pi/180
a2_g = unp.uarray(a2_g, np.ones(len(a2_g)))*np.pi/180
a2_r = unp.uarray(a2_r, np.ones(len(a2_r)))*np.pi/180

beta_1= unp.arcsin((unp.sin(a1))/n_kron)
beta_2= gamma - beta_1

print("-------------------------------------------------------------------------------")
print("beta_1:")
print(beta_1)
print("-------------------------------------------------------------------------------")

delta_rot = (a1 + a2_r) - (beta_1 + beta_2)
delta_grün = (a1 + a2_g) - (beta_1 + beta_2)

print("-------------------------------------------------------------------------------")
print(f'Ablenkung rot: {delta_rot}')
print("-------------------------------------------------------------------------------")
print(f'Ablenkung grün: {delta_grün}')
print("-------------------------------------------------------------------------------")

# Messaufgabe 5: Beugung am Gitter

k100, psi_100_g, psi_100_r = np.genfromtxt("content/data/100.txt", unpack = True)
k300, psi_300_g, psi_300_r = np.genfromtxt("content/data/300.txt", unpack = True)
k600, psi_600_g, psi_600_r = np.genfromtxt("content/data/600.txt", unpack = True)

psi_100_g = psi_100_g *np.pi/180
psi_100_r = psi_100_r *np.pi/180
psi_300_g = psi_300_g *np.pi/180
psi_300_r = psi_300_r *np.pi/180
psi_600_g = psi_600_g *np.pi/180
psi_600_r = psi_600_r *np.pi/180

def wave(d, psi, k):
    return d*np.sin(psi)/k

d100 = (1/100)*10**(-3)
d300 = (1/300)*10**(-3)
d600 = (1/600)*10**(-3)

lam100g = wave(d100, psi_100_g, k100)
lam100r = wave(d100, psi_100_r, k100)

lam300g = wave(d300, psi_300_g, k300)
lam300r = wave(d300, psi_300_r, k300)

lam600g = wave(d600, psi_600_g, k600)
lam600r = wave(d600, psi_600_r, k600)

print("Wellenlängen aus Beugungsmessung")
print("-------------------------------------------------------------------------------")
print("100/mm:")
print("grün:")
print(lam100g)
print("rot:")
print(lam100r)
print("-------------------------------------------------------------------------------")
print("300/mm:")
print("grün:")
print(lam300g)
print("rot:")
print(lam300r)
print("-------------------------------------------------------------------------------")
print("600/mm:")
print("grün:")
print(lam600g)
print("rot:")
print(lam600r)
print("-------------------------------------------------------------------------------")

# Mittelwert: 
lam_rot = np.concatenate((lam100r, lam300r, lam600r))
lam_rot = ufloat(np.mean(lam_rot), np.std(lam_rot))

lam_grün = np.concatenate((lam100g, lam300g, lam600g))
lam_grün = ufloat(np.mean(lam_grün), np.std(lam_grün))

print("-------------------------------------------------------------------------------")
print("Mittelwerte der Wellenlängen")
print("rot  :", '{:.4e}'.format(lam_rot))
print("grün :", '{:.4e}'.format(lam_grün))
print("-------------------------------------------------------------------------------")
