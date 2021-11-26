import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const


r_m, I = np.genfromtxt("content/Messung1.txt",unpack = True)
I_1 = unp.uarray(I,0.05)
r_m = unp.uarray(r_m*(10**(-2)),0.00001)
r_z = ufloat((0.995*(10**(-2)))/2, 0.00001)
r_ks = ufloat(1.25*(10**(-2)), 0.00001)
r_k = ufloat((5.4*(10**(-2)))/2,0.00001)
r_ges = (r_m +  r_z + r_ks + r_k) #in m
N = 195
R_Spule = 0.109
d = 0.138
B = 195*((const.mu_0)*I_1*(R_Spule**2))/(((R_Spule**2)+(d/2)**2)**(3/2))

def f(x, m, b):
    return m*x + b
parameters, pcov = op.curve_fit(f, unp.nominal_values(r_ges), unp.nominal_values(B), sigma=unp.std_devs(B))
err = np.sqrt(np.diag(pcov))
x = np.linspace(0,0.15,100)
b = ufloat(parameters[0],err[0])
mu = (0.00139*const.g)/b
print(b,mu)
plt.rcParams.update({'font.size': 22})
plt.subplot(1, 2, 1)
plt.errorbar(unp.nominal_values(r_ges), unp.nominal_values(B),xerr = unp.std_devs(r_ges),yerr = unp.std_devs(B), fmt='r.',label='Messdaten')
plt.plot(x,f(x,parameters[0],parameters[1]), label='lineare Regression')
#plt.xlim(3, 12)
plt.xlabel("$r\, / \, \mathrm{cm}$ ")
plt.ylabel("$B\, / \, \mathrm{T}$")
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')
plt.show()
plt.savefig('build/plot.pdf')
plt.close()

I1,T = np.genfromtxt("content/Messung2_T.txt", unpack = True)
T_f11 = unp.uarray(T,0.025) #Fehler ist die eigene Reaktionszeit
I_f = unp.uarray(I1,0.05)
T_f = T_f11**2
B1 = 1/(195*((const.mu_0)*I_f*(R_Spule**2))/(((R_Spule**2)+(d/2)**2)**(3/2)))
params, pcov1 = op.curve_fit(f, unp.nominal_values(B1), unp.nominal_values(T_f), sigma=unp.std_devs(T_f))
h = np.linspace(0,350000,100000)
m = 0.14176
a = ufloat(params[0],np.sqrt(np.diag(pcov1))[0])
J_k = (2/5)*m*(r_k**2)
mu1 = (4*(np.pi**2)*J_k)/a



plt.subplot(1, 2, 2)
plt.errorbar(unp.nominal_values(B1), unp.nominal_values(T_f),xerr = unp.std_devs(B1),yerr = unp.std_devs(T_f), fmt='r.',label='Messdaten')
plt.plot(h,f(h,params[0],params[1]), label='lineare Regression')
plt.xlabel(r'$B^{-1} \,/\, \mathrm{T}^{-1}\cdot 10^{-4}$')
plt.ylabel(r'$T^{2} \,/\, \mathrm{s}^{2}$')
plt.xlim(0, 3200)
plt.ylim(0, 6.8)
plt.grid(True, which="both", ls="-")
#plt.xticks(ticks = [5*(10**2),10*(10**2),15*(10**2),20*(10**2),25*(10**2),30*(10**2)],labels=["5","10","15","20","25","30"])
plt.legend(loc='best')
plt.savefig('build/plot.pdf')
plt.close()