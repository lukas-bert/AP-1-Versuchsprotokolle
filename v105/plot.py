import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const


r_m, I = np.genfromtxt("content/Messung1.txt",unpack = True)
I_1 = unp.uarray(I,0.05)
r_m = unp.uarray(r_m,0.001)
r_z = ufloat(0.995/2, 0.001)
r_ks = ufloat(1.25, 0.001)
r_k = ufloat(5.4/2,0.001)
r_ges = (r_m +  r_z + r_ks + r_k) #in cm
N = 195
R_Spule = 0.109
d = 0.138
B = ((const.mu_0)*I_1*(R_Spule**2))/(((R_Spule**2)+(d/2)**2)**(3/2))

def f(x, m, b):
    return m*x + b
parameters, pcov = op.curve_fit(f, unp.nominal_values(r_ges), unp.nominal_values(B), sigma=unp.std_devs(B))
err = np.sqrt(np.diag(pcov))
x = np.linspace(0,15,100)

plt.subplot(1, 2, 1)
plt.errorbar(unp.nominal_values(r_ges), unp.nominal_values(B),xerr = unp.std_devs(r_ges),yerr = unp.std_devs(B), fmt='r.',label='Messdaten')
plt.plot(x,f(x,parameters[0],parameters[1]), label='lineare Regression')
plt.xlim(3, 12)
plt.xlabel("$r\, / \, \mathrm{cm}$ ")
plt.ylabel("$B\, / \, \mathrm{T}$")
plt.legend(loc='best')
plt.savefig('build/plot.pdf')
plt.close()

#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')

#plt.savefig('build/plot.pdf')
#plt.close()