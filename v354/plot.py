import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

t_0, U_0 = np.genfromtxt("content/data_a.txt", unpack = True)
t = t_0 * 0.05
errt = 0.025
errU_0 = 0.05
p0 = (3, 6)
def ExpFit(x, m, tt):
    return m * np.exp(-tt * x)
a,b = scipy.optimize.curve_fit(ExpFit, t, U_0,p0)
x = np.linspace(-0.01,0.36,100)

#Plot Zu A
plt.subplot(1, 2, 1)
plt.plot(x,ExpFit(x,a[0],a[1]), label = "exponetielle Fitkurve")
plt.errorbar(t, U_0,xerr = errt, yerr = errU_0, fmt='ro', label = "zeitlicher Amplitudenverlauf")
plt.xlabel("$t\, /$ " "$\mathrm{\mu}$" r'$\mathrm{s}$')
plt.ylabel("$U_0\, /$ "  r'$\mathrm{V}$')
plt.grid(True, which="both", ls="-")
plt.legend(loc='best',prop={"size":6})

plt.subplot(1, 2, 2)
plt.plot(x,ExpFit(x,a[0],a[1]), label = "lineare Fitkurve")
plt.errorbar(t, U_0,xerr = errt, yerr = errU_0, fmt='ro', label = "logarithmierter zeitlicher Amplitudenverlauf")
plt.xlabel("$t\, /$ " "$\mathrm{\mu}$" r'$\mathrm{s}$')
plt.ylabel("$U_0\, /$ "  r'$\mathrm{V}$')
plt.grid(True, which="both", ls="-")
plt.yscale('log')
plt.legend(loc='best',prop={"size":5})

plt.savefig('PlotZuA.pdf')
plt.close

f, U_C, U, a_d, b_d = np.genfromtxt("content/data_c_d.txt", unpack = True)
phi, nix = np.genfromtxt("content/Messwerte_cd.txt", unpack = True)

#Plot zu D
plt.subplot(2,2,1)
#plt.plot(f, phi, "rx")
def theory(w, R, L, C):
    return np.arctan(((-2*np.pi*w)*R*C)/(1-(L*C*(4*np.pi**2*w**2))))
w = np.linspace(0,100,100)
R = 67.2
L = 16.87*(10**(-3))
C = 2.060*(10**(-9))
plt.plot(w,theory(w,R,L,C))
