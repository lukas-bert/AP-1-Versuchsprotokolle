import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.constants as const
import scipy.optimize as op

U, Z, I = np.genfromtxt("content/data/data1.txt", unpack = True)

# Fehler der Messgroßen
Z = unp.uarray(Z, np.sqrt(Z))
N = Z/120                       # 120 Sekunden gemessen
I = unp.uarray(I, 0.1)*10**(-6)

# Charakteristische Kurve des Zählrohres
down, up = [3, 31]

# Linearer Fit
def f(x, a, b):
    return a*x + b

params, pcov = op.curve_fit(f, U[down:up], noms(N[down:up]))

# Parameter
err = np.sqrt(np.diag(pcov))
a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])

print("------------------------------------------------------------------------")
print("Parameter des Fits")
print("a:               ", a, "[1/V] = ", 100*a, "[1/100V]")
print("b:               ", b, "[V]")
print("Plateaulänge:    ", U[up-1] - U[down], "[V]")
print("------------------------------------------------------------------------")

# Plot

x = np.linspace(U[down], U[up-1], 100)

plt.errorbar(U, noms(N), yerr = stds(N), marker = ".", linestyle = None, color = "firebrick", capsize = 3, linewidth = 0, elinewidth = 1)
plt.errorbar(U[down:up], noms(N[down:up]), yerr = stds(N[down:up]), marker = ".", linestyle = None, color = "forestgreen", capsize = 3,
                linewidth = 0, elinewidth = 1, label = "Plateaubereich")
plt.plot(x, f(x, *params), color = "cornflowerblue", label = "Regression")                

plt.xlabel(r"$U \mathbin{/} \unit{\volt}$")
plt.ylabel(r"$N \mathbin{/} \unit{\second^{-1}}$")

plt.grid()
plt.legend()
#plt.show()

plt.savefig('build/plot.pdf')
plt.close()

# Totzeit über 2 Quellen Methode

N1, N12, N2 = np.genfromtxt("content/data/data2.txt", unpack = True)

# Fehler der Messgrößen
N1 = ufloat(N1, np.sqrt(N1))
N2 = ufloat(N2, np.sqrt(N2))
N12 = ufloat(N12, np.sqrt(N12))

T = (N1 + N2 - N12)/(2*N1*N2)

print("------------------------------------------------------------------------")
print("Totzeit über 2-Quellen-Methode:", '{0:.3e}'.format(T))
print("------------------------------------------------------------------------")

# Pro Teilchen freiwerdende Ladung
dt = 120 # s

dQ = I*dt/(Z*const.e)
print("------------------------------------------------------------------------")
print("Ladungen in eV")
for i in range(len(dQ)):
    print('{0:.3e}'.format(dQ[i]), "eV")
print("------------------------------------------------------------------------")