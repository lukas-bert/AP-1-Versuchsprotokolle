import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

r_m, I = np.genfromtxt("content/Messung1.txt",unpack = True)
N = 195
R_Spule = 0.109
d = 0.138
B = ((const.mu_0)*I*(R_Spule**2))/(((R_Spule**2)+(d/2)**2)**(3/2))

plt.subplot(1, 2, 1)
plt.plot(r_m, B, label='Bestimmung des magnetischen Moments einer Vollkugel durch Gleichgewichtisbetrachtung')
plt.xlabel("$t\, / \, m$ ")
plt.ylabel("$B\, / \, mT$")
#plt.legend(loc='best')

#plt.savefig('build/plot.pdf')
#plt.close()

#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')

plt.savefig('build/plot.pdf')
#plt.close()