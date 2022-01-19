import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

T_1, p_1 = np.genfromtxt("content/data1.txt", unpack = True)
p_2, T_2 = np.genfromtxt("content/data2.txt", unpack = True)
P_2 = p_2*100000
p_0 = 1010
T_2 = T_2 + 273.15          # °C in K
T_1 = T_1 + 273.15

# Ausgelichsrechnung
def f(x,m,b):
    return m*x+b

params, pcov = op.curve_fit(f, (1/T_1), np.log(p_1/p_0)) 
print(params)

# Berechnung der Verdampfungswärme in b
L = -params[0]*const.R
print("-------------------------------------------------------------------------")
print("Verdampfungswärme:      ",L, " J*mol^-1")
print("-------------------------------------------------------------------------")

# Berechnung der Verdampfungswärme L_a in c
L_a = const.R*373
print("-------------------------------------------------------------------------")
print("L_a:      ",L_a, " J*mol^-1")

#Berechnung von L_i in c
L_i = L - L_a
print("-------------------------------------------------------------------------")
print("L_i:      ",L_i, " J*mol^-1")
# umrechnung in eV
L_i = L_i/(const.N_A*const.e)
print("-------------------------------------------------------------------------")
print("L_i:      ",L_i, " eV")

plt.plot(1/np.abs(T_1), np.log(p_1/p_0), label='Kurve')
plt.xlabel(r'$1/T\,/\,K**-1$')
plt.ylabel("ln"r'$(p/p_0)$')
plt.legend(loc='best')
plt.savefig('build/plot1.pdf')
plt.close()
# Zweite Messreihe 
# Ausgelichsrechnung
def f3(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

params, pcov = op.curve_fit(f3, T_2, p_2) 
print(params)
x = np.linspace(300,500,1000)

def df3(x,a,b,c):
    return 3*a*x**2+2*b*x+c

def L_m(t, a,b,c,d):
    return ((const.R*t)/(2*f3(t,a,b,c,d)) - np.sqrt(((const.R*t)/(2*f3(t,a,b,c,d)))**2-0.9/f3(t,a,b,c,d)))*df3(t,a,b,c)*t
def L_p(t, a,b,c,d):
    return ((const.R*t)/(2*f3(t,a,b,c,d)) + np.sqrt(((const.R*t)/(2*f3(t,a,b,c,d)))**2-0.9/f3(t,a,b,c,d)))*df3(t,a,b,c)*t


x = np.linspace(108.5+273.15,196.5+273.15,10000)
plt.plot(x, L_p(x,params[0], params[1], params[2], params[3]), label='L_+')
plt.ylabel(r'$L_+\,/\,\unit{\joule\mol^-1}$')
plt.xlabel(r'$T / \symup{K}$')
plt.legend(loc='best')
plt.plot(x, L_m(x,params[0], params[1], params[2], params[3]), label='L_-')
plt.ylabel(r'$L_+\,/\,\unit{\joule\mol^-1}$')
plt.xlabel(r'$T / \symup{K}$')
plt.legend(loc='best')
plt.savefig('build/plot2.pdf')
plt.close()