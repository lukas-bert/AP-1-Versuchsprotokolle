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
Steigung = a*100/f(500, *params)
print(f(500, *params))

print("------------------------------------------------------------------------")
print("Parameter des Fits")
print("a:               ", a, "[1/V]")
print("Plateausteigung: ", Steigung, "[1/V]")
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
N1 = ufloat(N1, np.sqrt(N1))/120
N2 = ufloat(N2, np.sqrt(N2))/120
N12 = ufloat(N12, np.sqrt(N12))/120

T = (N1 + N2 - N12)/(2*N1*N2)

#print(np.sqrt((noms((N2-N12)/(2*N2*N1**2)) *stds(N1) )**2 + (noms((N12-N1)/(2*N1*N2**2)) *stds(N2) )**2 + (stds(N12)/noms(2*N1*N2))**2)) überprüfung weil fehler sehr groß lul

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
#plot der freigesetzten Ladung

params, pcov = op.curve_fit(f, noms(U), noms(dQ)) # p0 = (noms(dQ[5]-dQ[0])/noms(I[5]-I[0]), 0)

#x = np.linspace(0, 9, 100)*10**(-7)
x = np.linspace(300, 720, 100)

plt.plot(x, f(x, *params), color = "cornflowerblue", label = "Ausgleichsgerade")
plt.errorbar(noms(U), noms(dQ), yerr = stds(dQ), marker = ".", linestyle = None,
     label = "freigesetzte Ladung", color = "firebrick", capsize = 3, linewidth = 0, elinewidth = 1)

plt.xlabel(r"$U \mathbin{/} \unit{\volt}$")
plt.ylabel(r"$dQ \mathbin{/} e_0")

#plt.xlim(0, 9*10**(-7))
plt.ylim(0, 5*10**10)

plt.grid()
plt.legend()
plt.show()

plt.savefig('build/plot1.pdf')
plt.close()
