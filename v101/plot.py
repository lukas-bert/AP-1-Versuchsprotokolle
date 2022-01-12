import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat

# Berechnung der Winkelrichtgröße D
phi, F = np.genfromtxt("content/data1.txt", unpack = True)
phi = phi * np.pi / 360     # Umrechnung in Bogenmaß
a = 0.2                     # Länge des Kraftarms in m

D_ = a* F/phi
D = D_.mean()               # Mittelwert
std_D = D_.std(ddof = 1)    # Mittelwertfehler
D = ufloat(D, std_D)
print("Winkelrichtgröße:    ", D)

# Bestimmung des Eigenträgheitsmoments 

a, T_3 = np.genfromtxt("content/data2.txt", unpack = True)
T = T_3/3       # Mitteln auf eine Periodendauer
a = a*10**(-2)  # Umrechnen in m

def linear(x, m, b):
    return m*x + b

params, pcov = op.curve_fit(linear, a**2, T**2)     # Lineare Regression
err = np.sqrt(np.diag(pcov))                       # Fehler aus Kovarianz-Matrix
x = np.linspace(0, 0.1, 100)

plt.plot(a**2, T**2, 'rx', label = "Messdaten")
plt.plot(x, linear(x, params[0], params[1]), label = "Lineare Regression")
#plt.xlabel(r'$a^2 \mathbin{/} \symup{m^2}$')
#plt.ylabel(r'$T^2 \mathbin{/} \symup{s^2}$')
plt.xlim(0, 0.1)
plt.ylim(0, 80)
plt.legend()

# Kleiner Plot im Plot
plt.axes([0.6, 0.25, 0.3, 0.25])
plt.plot(a**2, T**2, 'rx', label = "Messdaten")
plt.plot(x, linear(x, params[0], params[1]), label = "Lineare Regression")
plt.xlim(0, 0.02)
plt.ylim(0, 20)

plt.tight_layout()
plt.savefig('build/plot.pdf')
plt.close()

print(params, err)
b = ufloat(params[1], err[1])

# Berechnung des Eigenträgeheitmoments I_D

m = 0.2612  # Masse eines Gewichts in kg
h = 0.02    # Höhe in m
r = 0.0225    # Radius in m

I_D = b*D/(4*np.pi**2) - m*(r**2/2 + h**2/6)
print(I_D)
