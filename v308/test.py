import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op

#----------------------------------------------------------------------------------------
# Hysterse Kurve
I, B = np.genfromtxt("content/dataHysterese.txt", unpack = True)

# Grid und Achsen durch (0,0)
plt.grid()                                                 
plt.axhline(y = 0, color='k', linestyle='--', lw = 0.5)
plt.axvline(x = 0, color='k', linestyle='--', lw = 0.5)
# Plot der Messwerte
plt.plot(I[:11], B[:11], 'b.', label = 'Messwerte Neukurve')
plt.plot(I[11:], B[11:], 'r.', label = 'Messwerte')
# Marker an der Stelle der Remenanz 
plt.plot(0, 124.5, 'k_', markersize = 7)
plt.text(-1.2, 124.5, r"$B_r$")
# Sättigungswert
plt.hlines(y = 696.8, xmin = 0, xmax = 10, color='b', linestyle='--', lw = 0.8)
plt.vlines(x = 10, ymin = 0, ymax = 696.8, color='b', linestyle='--', lw = 0.8)
plt.text(-1.2, 696.8, r"$B_s$")
# Achseneinstellungen
#plt.xlabel(r'$I \mathbin{/} \unit{\ampere}$')
#plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')
plt.legend(loc='lower right')
plt.savefig('build/plotHysterese.pdf')
plt.close()

# Bestimmung der Koerzitivkraft

# Geradengleichung
def p1(x, a, b):
    return a*x + b   

# Bestimmung der Parameter     
params0, pcov0 = op.curve_fit(p1, I[19:22], B[19:22])
params01, pcov01 = op.curve_fit(p1, I[39:42], B[39:42])    

print(params0, params01)

Ix1 = np.linspace(-2, 2, 100)

# Plot der Messwerte
plt.plot(I[:11], B[:11], 'bx', label = 'Messwerte Neukurve')
plt.plot(I[11:], B[11:], 'rx', label = 'Messwerte')

# Plot des Fits
plt.plot(Ix1, p1(Ix1, params0[0], params0[1]), linestyle = "-", color = "forestgreen", label = "Ausgleichsgeraden")
plt.plot(Ix1, p1(Ix1, params01[0], params01[1]), linestyle = "-", color = "forestgreen")

# Achseneinstellungen
plt.grid()
plt.xlim(-1.5, 1.5)
plt.ylim(-400, 400)
#plt.xlabel(r'$I \mathbin{/} \unit{\ampere}$')
#plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')
plt.axhline(y = 0, color= 'k', linestyle= '--', lw = 1.2)
plt.axvline(x = 0, color= 'k', linestyle= '--', lw = 1.2)
plt.legend(loc = 'best')

plt.savefig('build/plotHystereseFit.pdf')
plt.close()
