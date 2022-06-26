
import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp 
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
import scipy.optimize as op

background = ufloat(340, np.sqrt(340))/600

print("Nulleffekt: ", background)

print("----------------------------------------------------------------------")
print("Messdaten mit Fehlern")
t_v, N_v = np.genfromtxt("content/data/Vanadium.txt", unpack = True)
print(N_v, np.around(np.sqrt(N_v)))
N_v = unp.uarray(N_v, np.sqrt(N_v)) - background*30     # dt = 30s
print("----------------------------------------------------------------------")
t_r, N_r = np.genfromtxt("content/data/Rhodium.txt", unpack = True)
print(N_r, np.around(np.sqrt(N_r)))
N_r = unp.uarray(N_r, np.sqrt(N_r)) - background*15     # dt = 15s
print("----------------------------------------------------------------------")
#print(N_v[N_v < 0], N_r[N_r < 0])

def f(x, a, b):
    return a*x + b

######################################################################################
# Vanadium

log_N_v = unp.log(N_v[noms(N_v) > 0])   # nur positive Zählraten betrachten
t_v = t_v[noms(N_v) > 0]

# Regression

params_, pcov = op.curve_fit(f, t_v, noms(log_N_v))
params = unp.uarray(params_, np.sqrt(np.diag(pcov)))

T = -np.log(2)/params[0]

print("----------------------------------------------------------------------")
print("Parameter des Fits:")
print("a = -lambda =", params[0])
print("b =", params[1])
print("Halbwertszeit: T =", T)
print("----------------------------------------------------------------------")
# Plot

x = np.linspace(-20, 900, 100)

plt.plot(x, f(x, *params_), color = "cornflowerblue", label = "Regression")
plt.errorbar(t_v, noms(log_N_v), yerr = devs(log_N_v), ms = 7, capsize = 2.5, fmt = ".", color = "firebrick", label = "Messwerte")    

plt.xlim(-20, 900)
plt.ylabel(r'$\symup{log}(N_{\symup{\Delta}t})$')
plt.xlabel(r'$t \mathbin{/} \unit{\second}$')
plt.legend()
plt.grid()
#plt.show()

plt.savefig("build/Vanadium.pdf")
plt.close()

####################################################################################
# Rhodium
t_cut2 = 330
t_cut1 = 270

log_N_r = unp.log(N_r)

# Maske für Ausreißer
maske_ausreißer = np.ones(len(t_r), dtype = bool)
maske_ausreißer[15], maske_ausreißer[28], maske_ausreißer[41], maske_ausreißer[42] = [False, False, False, False]

# Masken für verschiedene Bereiche
mask0 = np.zeros(len(t_r), dtype = bool)
mask0[t_r <= t_cut1] = True
mask0 = mask0 * maske_ausreißer

mask1 = np.zeros(len(t_r), dtype = bool)
mask1[t_r <= t_cut2] = True
mask1x = np.zeros(len(t_r), dtype = bool)
mask1x[t_r > t_cut1] = True
mask1 = mask1 * maske_ausreißer * mask1x

mask2 = np.zeros(len(t_r), dtype = bool)
mask2[t_r > t_cut2] = True
mask2 = mask2 * maske_ausreißer

log_N_r0 = log_N_r[mask0]
t_r0 = t_r[mask0]

log_N_r1 = log_N_r[mask1]
t_r1 = t_r[mask1]

t_r2 = t_r[mask2]
log_N_r2 = log_N_r[mask2]

# Regression 1:

params2_, pcov2 = op.curve_fit(f, t_r2, noms(log_N_r2))
params2 = unp.uarray(params2_, np.sqrt(np.diag(pcov2)))

# Regression 2:

params0_, pcov0 = op.curve_fit(f, t_r0, noms(unp.log(N_r[mask0] - unp.exp(f(t_r0, *params2_)))))
params0 = unp.uarray(params0_, np.sqrt(np.diag(pcov0)))

# Summe der beiden Kurven:

def Summe(t):
    
    return np.log(np.exp(f(t, *params0_)) + np.exp(f(t, *params2_)))     

x = np.linspace(0, 740)
x0 = np.linspace(0, t_cut1, 100)
x2 = np.linspace(t_cut2, 740, 100)

plt.plot(x, Summe(x), color = "blueviolet", label = "Summenkurve", lw = 1.5)
plt.plot(x0, f(x0, *params0_), color = "mediumseagreen", label = "Regression Kurzzeit", lw = 1.5)
plt.plot(x2, f(x2, *params2_), color = "mediumblue", label = "Regression Langzeit", lw = 1.5)
plt.vlines(t_cut2, 0, 7.5, 'darkslategray', 'dashed',label='$t^*$')

plt.errorbar(t_r[mask0 + mask1], noms(log_N_r[mask0 + mask1]), yerr = devs(log_N_r[mask0 + mask1]), capsize = 2.5, fmt = ".", color = "firebrick", label = "Messwerte")
plt.errorbar(t_r2, noms(log_N_r2), yerr = devs(log_N_r2), capsize = 2.5, fmt = ".", color = "royalblue")  
plt.errorbar(t_r[maske_ausreißer == False], noms(log_N_r[maske_ausreißer == False]), yerr = devs(log_N_r[maske_ausreißer == False]), capsize = 2.5, fmt = ".",
     color = "gray", label = "Ausreißer")

plt.ylabel(r'$\symup{log}(N_{\symup{\Delta}t})$')
plt.xlabel(r'$t \mathbin{/} \unit{\second}$')
plt.ylim(0, 6.5)
plt.xlim(-20, 740)
plt.legend()
plt.grid()

#plt.show()
plt.savefig("build/Rhodium.pdf")
plt.close()

# Berechnung der Halbwertszeiten

T_0 = -np.log(2)/params0[0]
T_2 = -np.log(2)/params2[0]

print("----------------------------------------------------------------------")
print("Parameter des Fits Langzeit:")
print("a = -lambda =", params2[0])
print("b =", np.e**params2[1])
print("Halbwertszeit: T2 =", T_2)

print("Parameter des Fits Kurzzeit:")
print("a = -lambda =", params0[0])
print("b =", np.e**params0[1])
print("Halbwertszeit: T1 =", T_0)
print("----------------------------------------------------------------------")

#abw
deltat2 = np.abs(224 - T)/224
print(deltat2)
deltarhi = np.abs(42.3 - T_0)/42.3
deltarh = np.abs(260 - T_2)/260
print("RH",deltarh)
print("RHI",deltarhi)