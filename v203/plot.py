import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

T_1, p_1 = np.genfromtxt("content/data1.txt", unpack = True)
p_2, T_2 = np.genfromtxt("content/data2.txt", unpack = True)
p_2 = p_2*100000            # Umrechnung in Pascal
p_0 = 1010                  # Außendruck in mbar
T_2 = T_2 + 273.15          # °C in K
T_1 = T_1 + 273.15

# Ausgelichsrechnung
def f(x,m,b):
    return m*x+b

params, pcov = op.curve_fit(f, (1/T_1), np.log(p_1/p_0)) 
err = np.sqrt(np.diag(pcov))                                
m = ufloat(params[0], err[0])

print(params, err)

# Berechnung der Verdampfungswärme in b
L = -m*const.R
print("-------------------------------------------------------------------------")
print("Verdampfungswärme:      ", '{0:.2f}'.format(unp.nominal_values(L)), "+-",'{0:.2f}'.format(unp.std_devs(L)), " J*mol^-1")
print("-------------------------------------------------------------------------")

# Berechnung der äußeren Verdampfungswärme L_a in c
L_a = const.R*373
print("-------------------------------------------------------------------------")
print("L_a:      ",L_a, " J*mol^-1")

# Berechnung von L_i in c
L_i = L - L_a
print("-------------------------------------------------------------------------")
print("L_i:      ",'{0:.2f}'.format(L_i), " J*mol^-1")

# Umrechnung in eV
L_i = L_i/(const.N_A*const.e)
print("-------------------------------------------------------------------------")
print("L_i:      ", '{0:.4f}'.format(L_i), " eV")

# Plot der Daten und der linearen Ausgleichsfunktion

x = np.linspace(0.0025, 0.0035, 1000)

plt.plot(x, f(x, *params), label = "Ausgleichsgerade", color = "cornflowerblue")
plt.plot(1/np.abs(T_1), np.log(p_1/p_0), linestyle = "none", marker = "1", label='Messdaten', color = "firebrick", markersize = "4.5")
plt.xlabel(r'$\frac{1}{\symup{T}} \mathbin{/} \unit{\per\kelvin}$')
plt.ylabel(r'ln$(p/p_0)$')
plt.legend(loc='best')
plt.xlim(0.00267, 0.00342)
plt.ylim(-4, 0.5)
plt.grid()
plt.tight_layout()
plt.savefig('build/plot1.pdf')
plt.close()

# Zweite Messreihe, Bestimmung der Temperaturabhängigkeit der Verdampfungswärme
# Ausgleichsrechnung

def f3(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

params, pcov = op.curve_fit(f3, T_2, p_2)
err = np.sqrt(np.diag(pcov))

print(params, err)

# Plot des Fits

#x = np.linspace(108.5+273.15,196.5+273.15,10000)
x = np.linspace(380, 470, 1000)

plt.plot(x, f3(x, *params), label = "Fit", color = "cornflowerblue")
plt.plot(T_2, p_2, linestyle = "none", marker = "x", color = "firebrick", label = "Messwerte")
plt.ylabel(r'$p \mathbin{/} \unit{\pascal}$')
plt.xlabel(r'$T \mathbin{/} \unit{\kelvin}$')
plt.grid()
plt.xlim(380, 470)
plt.ylim(0, 16*10**5)
plt.legend(loc='best')
plt.savefig('build/plot2.pdf')
plt.close()

# Plots zu den Funktionen L(T)

def df3(x,a,b,c):               # Ableitung des Polynoms
    return 3*a*x**2+2*b*x+c

def L_m(T,a,b,c,d):
    return ((const.R*T)/(2*f3(T,a,b,c,d)) - np.sqrt(((const.R*T)/(2*f3(T,a,b,c,d)))**2 - 0.9/f3(T,a,b,c,d)))*df3(T,a,b,c)*T

def L_p(T,a,b,c,d):
    return ((const.R*T)/(2*f3(T,a,b,c,d)) + np.sqrt(((const.R*T)/(2*f3(T,a,b,c,d)))**2 - 0.9/f3(T,a,b,c,d)))*df3(T,a,b,c)*T


# L_+

plt.plot(x, L_p(x, *params), label=r'$L_+(T)$', color = "chocolate")
plt.ylabel(r'$L_+ \mathbin{/} \unit{\joule\mol^-1}$')
plt.xlabel(r'$T \mathbin{/} \unit{\kelvin}$')
plt.grid()
plt.legend(loc='best')
plt.xlim(380, 470)
plt.ylim(36000, 50000)
plt.savefig('build/plot3.pdf')
plt.close()

# Plot von L_-

plt.plot(x, L_m(x, *params), label=r'$L_-(T)$', color = "chocolate")
plt.ylabel(r'$L_- \mathbin{/} \unit{\joule\mol^-1}$')
plt.xlabel(r'$T \mathbin{/} \unit{\kelvin}$')
plt.grid()
plt.legend(loc='best')
plt.xlim(380, 470)
plt.ylim(400, 4000)
plt.savefig('build/plot4.pdf')
plt.close()
