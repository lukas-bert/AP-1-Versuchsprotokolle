import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
import uncertainties as unc
from uncertainties import ufloat
import scipy.constants as const
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs

# Einlesen der Daten T_1, p_b: warm     T_2, p_a: kalt
t, T_2, T_1, p_a, p_b, P = np.genfromtxt("content/data.txt", unpack = True)

# Umrechnen in SI-Einheiten:
p_a = p_a*10**5             # bar in Pascal
p_b = p_b*10**5
t = 60*t                    # Minuten in Sekunden
T_2 = T_2 + 273.15          # °C in K
T_1 = T_1 + 273.15 

# Aufgabe a),b)

# Funktionen des Fits
def f1(t, A, B, C):
    return A*t**2 + B*t + C

# Ableitungsfunktion
def f1dt(t, A, B):
    return 2*A*t + B

# Quadratischer Fit
params2, pcov2 = op.curve_fit(f1, t, T_2)
params1, pcov1 = op.curve_fit(f1, t, T_1)

# Fehler der Parameter A, B und C
params2_err = np.sqrt(np.diag(pcov2))
params1_err = np.sqrt(np.diag(pcov1))

A2 = ufloat(params2[0], params2_err[0])
B2 = ufloat(params2[1], params2_err[1])
C2 = ufloat(params2[2], params2_err[2])
A1 = ufloat(params1[0], params1_err[0])
B1 = ufloat(params1[1], params1_err[1])
C1 = ufloat(params1[2], params1_err[2])

print("-------------------------------------------------------------------------")
print("Parameter des Fits: (kalt)      ", '{0:.8f}'.format(A2), '{0:.5f}'.format(B2), '{0:.2f}'.format(C2))
print("Parameter des Fits: (warm)      ", '{0:.8f}'.format(A1), '{0:.5f}'.format(B1), '{0:.2f}'.format(C1))
print("-------------------------------------------------------------------------")

# Plot 1
x = np.linspace(0, 1300, 1000)

plt.plot(x, f1(x, params2[0], params2[1], params2[2]), 'c', label = "Fit zum kalten Gefäß")
plt.plot(x, f1(x, params1[0], params1[1], params1[2]), 'tab:orange', label = "Fit zum warmen Gefäß")
plt.errorbar(t, T_2, xerr = 0, yerr = 0.1, fmt = 'bx', label = 'Messwerte kaltes Gefäß')
plt.errorbar(t, T_1, xerr = 0, yerr = 0.1, fmt = 'rx', label = 'Messwerte warmes Gefäß')
plt.xlabel(r'$t\,/\,$s')
plt.ylabel(r'$T\,/\,$K')
plt.xlim(0, 1250)
plt.legend(loc='best')
plt.savefig('build/plot1.pdf')
plt.close()

# Aufgabe c)

# Ohne Fehler gerechnet
#dt = np.array([3, 8, 13, 18])*60                # Array mit 4 Zeitpunkten im Messintervall       
#dT2 = f1dt(dt, params2[0], params2[1])
#dT1 = f1dt(dt, params1[0], params1[1])
#print(dT1, dT2)

# Mit Fehlerrechnung
u_f1dt = unc.wrap(f1dt)                         # Funktion zur Fehlerrechnung

dT21 = u_f1dt(3*60, A2, B2)                     # Es wurden die Zeipunkte 3, 8, 13, 18 Minuten ausgewählt 
dT22 = u_f1dt(8*60, A2, B2)
dT23 = u_f1dt(13*60, A2, B2)
dT24 = u_f1dt(18*60, A2, B2)

dT11 = u_f1dt(3*60, A1, B1)
dT12 = u_f1dt(8*60, A1, B1)
dT13 = u_f1dt(13*60, A1, B1)
dT14 = u_f1dt(18*60, A1, B1)

dT2 = unp.uarray([noms(dT21), noms(dT22), noms(dT23), noms(dT24)], [devs(dT21), devs(dT22), devs(dT23), devs(dT24)])
dT1 = unp.uarray([noms(dT11), noms(dT12), noms(dT13), noms(dT14)], [devs(dT11), devs(dT12), devs(dT13), devs(dT14)])

print("-------------------------------------------------------------------------")
print("dT2:          ", dT2)
print("dT1:          ", dT1)
print("-------------------------------------------------------------------------")
# c_wasser = 4.1818 kJ/(kg*K)
c_w = 4181.8

dQ1 = (3*c_w + 750)*dT1
dQ2 = (3*c_w + 750)*dT2

# Aufgabe d)

# Berechnen der Güteziffer
v1 = dQ1[0]/P[2]
v2 = dQ1[1]/P[7]
v3 = dQ1[2]/P[12]
v4 = dQ1[3]/P[17]

# Theoriewert
v1t = T_1[2]/(T_1[2]-T_2[2])
v2t = T_1[7]/(T_1[7]-T_2[7])
v3t = T_1[12]/(T_1[12]-T_2[12])
v4t = T_1[17]/(T_1[17]-T_2[17])

print("Güteziffern:")
print("Experiment:              ", v1, v2, v3, v4)
print("Theorie                  ", v1t, v2t, v3t, v4t)

# Aufgabe e)

def fl(x, A, B):
    return A*x + B

# Linearer Fit zur Bestimmung der Verdampfungswärme L mit den Werten für das warme Gefäß
params, pcov = op.curve_fit(fl, 1/T_1, np.log(p_b/100000))   
params_err = np.sqrt(np.diag(pcov))
A = ufloat(params[0], params_err[0])
B = ufloat(params[1], params_err[1])
print("Parameter des linearen Fits:      ", '{0:.2f}'.format(A), '{0:.3f}'.format(B))

# Plot 2
px = np.linspace(3.1*10**-3, 3.4*10**-3, 1000)

plt.plot(1/T_1, np.log(p_b/100000), 'rx', label = 'Messwerte des warmen Reservoirs')
plt.plot(px, fl(px, params[0], params[1]), label = 'Lineare Regression')
plt.xlabel(r'$T^{-1}\,/\,10^{-3}$K')
plt.ylabel(r'log($p_b\,/\,p_0$)')
plt.xticks([0.0031, 0.00315, 0.0032, 0.00325, 0.0033, 0.00335],
           [r"$3.1$", r"$3.15$", r"$3.2$", r"$3.25$", r"$3.3$", r"$3.35$"])
plt.xlim(3.1*10**-3, 3.4*10**-3)
plt.legend()
plt.savefig('build/plot2.pdf')
plt.close()

# Berechnen der Verdampfungswärme
L = - A * const.R                   # R: Allgemeine Gaskonstante
M = 0.12091                          # Molare Masse (g/mol) von Dichlordifluormethan 
L = L/M

print("Allgemeine Gaskonstante, Verdampfungswärme:      ", const.R, '{0:.2f}'.format(L))

# Berechnen des Massendruchsatzes 
dm = dQ2/L
print("-------------------------------------------------------------------------")
print("Massendurchsatz:         ", dm)
print("-------------------------------------------------------------------------")

roh_0 = 5.51 
k = 1.14

# Berechnen der Dichten das gases zu verschiedenen Temperaturen
roh1 = roh_0*237.15*p_b[2]/(100000*T_2[2])
roh2 = roh_0*237.15*p_b[7]/(100000*T_2[7])
roh3 = roh_0*237.15*p_b[12]/(100000*T_2[12])
roh4 = roh_0*237.15*p_b[17]/(100000*T_2[17])
print("roh:                      ", roh1, roh2, roh3, roh4)

# Berechnen der mechanischen Leistung aus Messwerten
N1 = 1/(k-1) * (p_b[2]*(p_a[2]/p_b[2])**(1/k)-p_a[2]) * 1/roh1 *dm[0]
N2 = 1/(k-1) * (p_b[7]*(p_a[7]/p_b[7])**(1/k)-p_a[7]) * 1/roh2 *dm[1] 
N3 = 1/(k-1) * (p_b[12]*(p_a[12]/p_b[12])**(1/k)-p_a[12]) * 1/roh3 *dm[2] 
N4 = 1/(k-1) * (p_b[17]*(p_a[17]/p_b[17])**(1/k)-p_a[17]) * 1/roh4 *dm[3] 

print("Mechanische Leistung:    ", N1, N2, N3, N4)
