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
#Koerzitivkraft
plt.plot(-0.638, 0, 'k|', markersize = 7)
plt.text(-1.7, 30, r"$H_c$")
plt.plot(0.651, 0, 'k|', markersize = 7)
plt.text(0.8, -80, r"$H_c$")
# Achseneinstellungen
plt.xlabel(r'$I \mathbin{/} \unit{\ampere}$')
plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')
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

Ix1 = np.linspace(-2, 2, 100)

# Plot der Messwerte
plt.plot(I[:11], B[:11], 'bx', label = 'Messwerte Neukurve')
plt.plot(I[11:], B[11:], 'rx', label = 'Messwerte')

# Plot des Fits
plt.plot(Ix1, p1(Ix1, params0[0], params0[1]), linestyle = "-", color = "forestgreen", label = "Ausgleichsgeraden")
plt.plot(Ix1, p1(Ix1, params01[0], params01[1]), linestyle = "-", color = "forestgreen")

#Koerzitivkraft
plt.plot(-0.638, 0, 'k|', markersize = 7)
plt.text(-0.6, -50, r"$H_c$")
plt.plot(0.651, 0, 'k|', markersize = 7)
plt.text(0.7, -50, r"$H_c$")

# Achseneinstellungen
plt.grid()
plt.xlim(-1.5, 1.5)
plt.ylim(-400, 400)
plt.xlabel(r'$I \mathbin{/} \unit{\ampere}$')
plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')
plt.axhline(y = 0, color= 'k', linestyle= '--', lw = 1.2)
plt.axvline(x = 0, color= 'k', linestyle= '--', lw = 1.2)
plt.legend(loc = 'best')

plt.savefig('build/plotHystereseFit.pdf')
plt.close()

#----------------------------------------------------------------------------------------

def B_H(x, d, I, R, N):
    return const.mu_0*I*R**2*N/2 * (((x-d/2)**2+R**2)**(-3/2) + ((x+d/2)**2+R**2)**(-3/2))

def B_H_theo(x, I, R, N):
    return const.mu_0*I*R**2*N/(R**2 + x**2)**(3/2)   

xl = np.linspace(-15, 25, 1000)

R = 0.0625          # Radius der Spulen in m
N = 100             # Windungen der Spulen
I = 4               # Eingestellte Stromstärke
d1 = 0.1
d2 = 0.15
d3 = 0.2            # Abstände der Spulen

print("Theoriewerte:        ", B_H_theo(d1/2, I, R, N)*10**3, B_H_theo(d2/2, I, R, N)*10**3, B_H_theo(d3/2, I, R, N)*10**3)  

# Helmholtzspule Länge = 10cm

x1, B1 = np.genfromtxt("content/dataHelmholtz1.txt", unpack = True)

plt.plot(xl + d1/2*10**2 - 2.3, B_H(xl*10**(-2), d1, I, R, N)*10**3, "b--", label = "Theoriekurve")

plt.plot(x1, B1, 'rx', label = 'Messwerte')
plt.plot(d1/2*10**2 -2.3, B_H_theo(d1/2, I, R, N)*10**3, 'g+', marker = 6, color = 'deepskyblue', markersize = 7, label = "Theoriewert im Mittelpunkt")               # d1/2 --> Mittelpunkt,  "-2.3" , da Skalen nicht identisch an der Messsaparatur

plt.grid()
plt.xlabel(r'$x \mathbin{/} \unit{\centi\metre}$')
plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')
plt.xlim(-5, 20)
plt.legend(loc='best')
plt.savefig('build/plotHelmHoltz1.pdf')
plt.close()

# Helmholtzspule Länge = 15cm
x2, B2 = np.genfromtxt("content/dataHelmholtz2.txt", unpack = True)

plt.plot(xl + d2/2*10**2 - 2.3, B_H(xl*10**(-2), d2, I, R, N)*10**3, "b--", label = "Theoriekurve") 

plt.plot(x2, B2, 'rx', label = 'Messwerte')
plt.plot(d2/2*10**2 -2.3, B_H_theo(d2/2, I, R, N)*10**3, 'g+', marker = 6, color = 'deepskyblue', markersize = 7, label = "Theoriewert im Mittelpunkt")       

plt.grid()
plt.xlabel(r'$x \mathbin{/} \unit{\centi\metre}$')
plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')
plt.xlim(-5, 24)
plt.legend(loc='best')
plt.savefig('build/plotHelmHoltz2.pdf')
plt.close()

#----------------------------------------------------------------------------------------

# Helmholtzspule Länge = 20cm
x3, B3 = np.genfromtxt("content/dataHelmholtz3.txt", unpack = True)

plt.plot(xl + d3/2*10**2 - 2.3, B_H(xl*10**(-2), d3, I, R, N)*10**3, "b--", label = "Theoriekurve")

plt.plot(x3, B3, 'rx', label = 'Messwerte')
plt.plot(d3/2*10**2 -2.3, B_H_theo(d3/2, I, R, N)*10**3, 'g+', marker = 6, color = 'deepskyblue', markersize = 7, label = "Theoriewert im Mittelpunkt")

plt.grid()
plt.xlabel(r'$x \mathbin{/} \unit{\centi\metre}$')
plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')
plt.xlim(-5, 27)
plt.legend(loc='best')
plt.savefig('build/plotHelmHoltz3.pdf')
plt.close()

#----------------------------------------------------------------------------------------
def B_lang(x, N, I, R, L):
    return const.mu_0*N*I/2 * ((x+L/2)/np.sqrt(R**2+ (x+L/2)**2) - (x-L/2)/np.sqrt(R**2 + (x-L/2)**2))
xs = np.linspace(0, 12, 1000)

N = 300
R = 0.0205  
I = 1
L = 0.3

# Lange Spule
x4, B4 = np.genfromtxt("content/dataLangeSpule.txt", unpack = True)
x4 = np.flip(x4) - 17

l = 0.164
B_theorie = const.mu_0 * 300/l

#plt.plot(xs, B_lang(xs*10**(-2), N, I, R, L)*10**(3))      # klappt nicht amk
plt.plot(x4, B4, 'rx', label = 'Messwerte')
plt.grid()
plt.xlabel(r'$x \mathbin{/} \unit{\centi\metre}$')
plt.ylabel(r'$B \mathbin{/} \unit{\milli\tesla}$')

plt.hlines(y = B_theorie*10**3, xmin = 0, xmax = 6, color='b', linestyle='--', label = "Theoriewert")
plt.xlim(-6.5, 6)

plt.legend(loc='best')
plt.savefig('build/plotLangeSpule.pdf')
plt.close()
