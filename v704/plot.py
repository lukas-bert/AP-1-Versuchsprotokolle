import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp

# Einlesen der Daten
d_pb, N_pb, t_pb = np.genfromtxt("content/data/gamma_Pb.txt", unpack = True)    # Daten des Gamma-Strahler zu Blei (Pb)
d_zn, N_zn, t_zn = np.genfromtxt("content/data/gamma_Zn.txt", unpack = True)    # Daten des Gamma-Strahler zu Zink (Zn)

N_pb = N_pb/t_pb
N_zn = N_zn/t_zn
d_pb = d_pb*10**(-3)
d_zn = d_zn*10**(-3)

d_b, err_d_b, N_b, t_b = np.genfromtxt("content/data/beta.txt", unpack = True)  # Daten des Beta-Strahlers (Al)
d_b = unp.uarray(d_b, err_d_b)*10**(-9)
N_b = N_b/t_b

x = np.linspace(0, 10, 1000)
y = x ** np.sin(x)

plt.subplot(1, 2, 1)
plt.plot(d_pb, N_pb, marker = 'x', color = 'firebrick', label = 'Messwerte', linewidth = 0)
plt.xlabel(r'$d / m$')
plt.ylabel(r'$N$')
plt.yscale('log')
plt.grid()
plt.title("Plumbum")
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(d_zn, N_zn, marker = 'x', color = 'firebrick', label = 'Messwerte', linewidth = 0)
plt.xlabel(r'$d / m$')
plt.ylabel(r'$N$')
plt.title("Zink")
plt.yscale('log')
plt.grid()
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
