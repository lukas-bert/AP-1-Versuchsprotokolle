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
plt.plot(x,ExpFit(x,a[0],a[1]), label = "Fit")
plt.errorbar(t, U_0,xerr = errt, yerr = errU_0, fmt='ro', label = "Messdaten")
plt.xlabel("$t\, /$ " "$\mathrm{\mu}$" r'$\mathrm{µs}$')
plt.ylabel("$U_0\, /$ "  r'$\mathrm{V}$')
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')
plt.legend(loc='best',prop={"size":6})

plt.subplot(1, 2, 2)
plt.plot(x,ExpFit(x,a[0],a[1]), label = "Fit (Logarithmische Skala)")
plt.errorbar(t, U_0,xerr = errt, yerr = errU_0, fmt='ro', label = "Messdaten")
plt.xlabel("$t\, /$ " "$\mathrm{\mu}$" r'$\mathrm{µs}$')
plt.ylabel("$U_0\, /$ "  r'$\mathrm{V}$')
plt.grid(True, which="both", ls="-")
plt.yscale('log')
plt.legend(loc='best')
plt.legend(loc='best',prop={"size":6})

plt.tight_layout()

plt.savefig('build/PlotZuA.pdf')
plt.close



#Plot zu D
def theory(w, R, L, C):
    return np.arctan((((-w)*R*C)/(1-(L*C*(w**2)))))
w = np.linspace(0,80,100)
R = 732
L = 16.87*(10**(-3))
C = 2.060*(10**(-9))
errf = 0
errphi = 0.005
f, U_C, U, a_d, b_d = np.genfromtxt("content/data_c_d.txt", unpack = True)
phi, nix = np.genfromtxt("content/Messwerte_cd.txt", unpack = True)

plt.subplot(2,2,1)
plt.plot(f, phi, "rx")
# funktioniert einfach nicht plt.plot(w,theory(2000*np.pi*w,R,L,C))
plt.errorbar(f, phi,xerr = errf, yerr = errphi, fmt='r.', label = "Freuquenzabhänigkeit der Phase")
plt.xlabel("$f\,/$"r'$\,\mathrm{Hz}$')
plt.ylabel("$\phi\,/$"r'$\,\mathrm{rad}$')
plt.grid(True, which="both", ls="-")
plt.legend(loc='best',prop={"size":6})
plt.yticks([0,(1/4)*np.pi,(1/2)*np.pi,(3/4)*np.pi,np.pi],["0","$1/4\pi$","$1/2\pi$","$3/4\pi$","$\pi$"])
plt.savefig("PlotZuD.pdf")
plt.close