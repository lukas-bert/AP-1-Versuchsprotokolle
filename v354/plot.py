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
plt.xlabel("$t\, /$ " r'$\mathrm{\mu}\mathrm{µs}$')
plt.ylabel("$U_0\, /$ "  r'$\mathrm{V}$')
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')
plt.legend(loc='best',prop={"size":6})

plt.subplot(1, 2, 2)
plt.plot(x,ExpFit(x,a[0],a[1]), label = "Fit (Logarithmische Skala)")
plt.errorbar(t, U_0,xerr = errt, yerr = errU_0, fmt='ro', label = "Messdaten")
plt.xlabel("$t\, /$ " r'$\mathrm{\mu}\mathrm{µs}$')
plt.ylabel("$U_0\, /$ "  r'$\mathrm{V}$')
plt.grid(True, which="both", ls="-")
plt.yscale('log')
plt.legend(loc='best')
plt.legend(loc='best',prop={"size":6})

plt.tight_layout()

plt.savefig('build/PlotZuA.pdf')
plt.close()


#Plot Zu C
f, U_C, U, a, b = np.genfromtxt("content/data_c_d.txt", unpack = True)
phi, U_x = np.genfromtxt("content/Messwerte_cd.txt", unpack = True)
R = 732
L = 16.87*10**-3
C = 2.060 *10**-9

p0 = (L*C, R**2*C**2)
def Theorie_Fit(w, a, b):
    return (1/np.sqrt((1-a*(2000*np.pi*w)**2)**2+(2000*np.pi*w)**2*b))
params, pacov = scipy.optimize.curve_fit(Theorie_Fit, f, U_x, p0)

def Theorie_c(w, R, L, C):
    return (1/np.sqrt((1-L*C*w**2)**2+w**2*R**2*C**2))

#print(R/(2*L))
#print(np.sqrt(L*C)*1/(R*C)) 
#print(np.sqrt(1/(L*C)-R**2/(2*L**2))/(2*np.pi*1000))  

w = np.linspace(10, 70, 1000)
# Erster Plot zu C
plt.plot(w, Theorie_c(2*np.pi*1000*w, R, L, C), label = "Theoriekurve")
plt.plot(f, U_x, 'rx', label = "Messwerte")
plt.plot(26.55, 3.92013543016825 , 'go', label = "Maximum Theorie")

plt.ylabel("$U_C / U$")
plt.xlabel(r'$f\,/\,$kHz')
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')
plt.xscale('log')
plt.xlim(10, 70)
plt.ylim(0, 5)

plt.savefig("build/PlotZuC1.pdf")
plt.close()

#Zweiter Plot Zu C
def Breite_theo(x):
    return ((1/np.sqrt(2))*3.92013543016825)*x/x
x = np.linspace(22.74, 29.79, 10)

def Breite_exp(x):
    return (3.39/np.sqrt(2))*x/x
x_1 = np.linspace(22.27, 29.37, 10)

plt.plot(w, Theorie_c(2*np.pi*1000*w, R, L, C), label = "Theoriekurve")
plt.plot(f, U_x, 'r', label = "Messwertkurve")
#plt.plot(w, Theorie_Fit(w, params[0], params[1]), label = "Theorie Fit")
#plt.plot(26.55,3.92013543016825 , 'go', label = "Maximum Theorie")
plt.plot(x, Breite_theo(x), '--', label = "Breite der Theoriekurve")
plt.plot(x_1, Breite_exp(x),'--', label = "Breite der Messwertkurve")

plt.ylabel("$U_C / U$")
plt.xlabel(r'$f\,/\,$kHz')
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')
plt.ylim(0, 5)
plt.xlim(15, 35)

plt.savefig("build/PlotZuC2.pdf")
plt.close()

#Plot zu D

import uncertainties as unc
import uncertainties.unumpy as unp

flinspace2 = np.linspace(0, 100, 1000)

plt.plot(f, phi, 'rx', label='Messwerte')
plt.plot(flinspace2, np.pi/2+unp.nominal_values(unp.arctan((-2*np.pi*flinspace2*1e3*(R)*C/(1-L*C*(2*np.pi*flinspace2*1e3)**2))**(-1))), label='Theoriekurve')
plt.xscale('log')
plt.xlabel(r'$f\,/\,$kHz')
plt.ylabel(r'$\phi/$rad')
plt.axis((10, 70, -0.2, 4))  # 1e-3 weil f in kHZ ist
# Slicing: -3 ist drittletztes Element, : bedeutet bis zum Ende
plt.plot(f[-1:], phi[-1:], 'ok', markersize=8, markeredgewidth=1, markerfacecolor='None' )

# Macht die y-Achse schön
x = np.linspace(0, 2 * np.pi)

plt.ylim(0, 1.25 * np.pi)
# erste Liste: Tick-Positionen, zweite Liste: Tick-Beschriftung
plt.yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi/4, np.pi],
           [r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])

plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/PlotZuD1.pdf')

#Zweiter Plot zu D
w_1 = -R/(2*L) + np.sqrt(R**2/(4*L**2)+ 1/(L*C))
w_2 = R/(2*L) + np.sqrt(R**2/(4*L**2)+ 1/(L*C))
w_res = np.sqrt(1/(L*C)-R**2/(2*L**2))/(2*np.pi*1000)

plt.xscale('linear')
plt.plot(f, phi, 'r-', label='Messwertkurve')
plt.axis((10, 60, 0, np.pi))
plt.axvline(22.9, color='tab:orange', linestyle=':', label=r'$f_1 = 22.9$kHz und $f_2 = 29.8$kHz')
plt.axvline(29.8, color='tab:orange', linestyle=':')
plt.axvline(w_res, color='g', linestyle=':', label='Resonanzfrequenz')
plt.legend()
plt.savefig('build/PlotZuD2.pdf')
