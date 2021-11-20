import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

t_0, U_0 = np.genfromtxt("content/data_a.txt", unpack = True)
t = t_0 * 0.05
errt = 0.025
errU_0 = 0.05
p0 = (0, 3.4)
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
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(x,ExpFit(x,a[0],a[1]), label = "lineare Fitkurve")
plt.errorbar(t, U_0,xerr = errt, yerr = errU_0, fmt='ro', label = "logarithmierter zeitlicher Amplitudenverlauf")
plt.xlabel("$t\, /$ " "$\mathrm{\mu}$" r'$\mathrm{s}$')
plt.ylabel("$U_0\, /$ "  r'$\mathrm{V}$')
plt.grid(True, which="both", ls="-")
plt.yscale('log')
plt.legend(loc='best')

plt.savefig('PlotZuA.pdf')
plt.close

