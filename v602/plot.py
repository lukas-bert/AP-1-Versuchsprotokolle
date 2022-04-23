import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op

d_lif = 201.4*10**-12 # in m
R = 13.6 # in eV
alpha = 7.297*10**-3

# Überprüfung der braggbedingung
theta2, imp = np.genfromtxt("content/data/braggbedingung.txt", unpack = True)

# Abweichung vom Theoriewinkel
abw1 = np.abs(27.3 - 28)/28
print("Abweichung des maximums der Braggbedingung: ", abw1)

plt.plot(theta2/2, imp, color = "firebrick", label = "Messwerte", marker = "x", markersize = 5)
plt.plot(27.3/2, 221.0, "o", markersize = 6 ,color = "cornflowerblue", label = "Maximum")
plt.grid()
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
plt.savefig('build/plotbragg.pdf')
#plt.show()
plt.close()

# Emissionsspektrum der Kupferanode

theta2, imp = np.genfromtxt("content/data/emissionspektrum.txt", unpack = True)
#lambda1 = 2*d_lif*np.sin(theta2/2*np.pi/180)
plt.plot(theta2/2, imp, color = "firebrick", label = "Messwerte")
plt.plot(40.8/2, 1544.0, "v", markersize = 7 ,color = "chocolate", label = r"$K_\alpha\text{-Linie}$")
plt.plot(45.5/2, 5129.0, "v", markersize = 7 ,color = "forestgreen", label = r"$K_\beta\text{-Linie}$")
plt.plot(theta2[46]/2, imp[46], "o", markersize = 6, color = "cornflowerblue", label = "Bremsberg")
#plt.plot(theta2[40:48],imp[40:48], "o", markersize = 6, color = "cornflowerblue", label = "Bremsberg")
plt.grid()
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
plt.savefig('build/plotemission.pdf')
#plt.show()
plt.close()

# Detailspektrum

theta2, imp = np.genfromtxt("content/data/detailmessung.txt", unpack = True)
plt.plot(theta2/2, imp, color = "firebrick", marker = "x", markersize = 5, label = "Messwerte")
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
kalpha = np.linspace(40.41154/2, 41.308/2, 1000)                                                        #Intervall der Halbwertsbreite zu K_alpha
plt.plot(kalpha, 760 + kalpha*0, label = r"$\text{Halbwertsbreite des } K_{\beta}\text{-Peaks}$", linestyle = "--", linewidth = 1.5)
kbeta = np.linspace(44.880769/2, 45.83878469/2, 1000)                                                   #Intervall der Halbwertsbreite zu K_beta
plt.plot(kbeta , 2641.5 + kbeta*0, label = r"$\text{Halbwertsbreite des } K_{\alpha}\text{-Peaks}$", linestyle = "--", linewidth = 1.5)
plt.grid()
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('build/detailspektrum.pdf')
plt.close()

# Berechnung des Auflösungsvermögens

# experimentell
E_Kbetaexp = const.h*const.c/(2*d_lif*np.sin(40.8*np.pi/(2*180))*const.e*1000)
E_Kalphaexp = const.h*const.c/(2*d_lif*np.sin(45.4*np.pi/(2*180))*const.e*1000)

DeltaE_Kbetaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(40.41154*np.pi/(2*180))*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(41.308*np.pi/(2*180))*const.e*1000))
DeltaE_Kalphaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(44.880769*np.pi/(2*180))*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(45.83878469*np.pi/(2*180))*const.e*1000))
print(E_Kalphaexp, E_Kbetaexp, DeltaE_Kalphaexp,DeltaE_Kbetaexp)

# Auflösungsvermögen

A_Kalpha = E_Kalphaexp/DeltaE_Kalphaexp
A_Kbeta = E_Kbetaexp/DeltaE_Kbetaexp
print(A_Kalpha, A_Kbeta)

# Abschirmkonstanten
sigma1 = 29 - np.sqrt(8.988*1000/R)
print("sigma_1 von Kupfer: ", sigma1)
sigma2 = 29 - np.sqrt(4*(29 - sigma1)**2 - 4*(E_Kalphaexp*1000/R))
print("sigma_2 von Kupfer: ", sigma2)
sigma3 = 29 - np.sqrt(9*(29 - sigma1)**2 - 9*(E_Kbetaexp*1000/R))
print("sigma_3 von Kupfer: ", sigma3)


# Theorie
sigma1t = 29 - np.sqrt(8.988*1000/R)
print("sigma_1t von Kupfer: ", sigma1t)
sigma2t = 29 - np.sqrt(4*(29 - sigma1)**2 - 4*(8*1000/R))
print("sigma_2t von Kupfer: ", sigma2t)
sigma3t = 29 - np.sqrt(9*(29 - sigma1)**2 - 9*(8.95*1000/R))
print("sigma_3t von Kupfer: ", sigma3t)

# Abweichungen
ds1 = np.abs(sigma1-sigma1t)/sigma1t
ds2 = np.abs(sigma2-sigma2t)/sigma2t
ds3 = np.abs(sigma3-sigma3t)/sigma3t
print("Abweichungen zu sigma1,2,3: ", ds1, ds2, ds3)
# Abschirmkonstanten
E_abs = 8.988
sigma1 = 29 - np.sqrt(E_abs*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
sigma2 = 29 - np.sqrt(E_Kalphaexp*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
sigma3 = 29 - np.sqrt(E_Kbetaexp*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
#print(sigma1, sigma2, sigma3)

# Absorptionsspektrum

h = const.h 
e = const.e
c = const.c

def sigmak(Z, E):                                                   # Funktion zur Berechnung der Abschirmkonstante
    return Z - np.sqrt((E*1000/R) - (((alpha**2)*(Z**4))/4))

def E_K(theta):                                                     # Gibt zu theta zugehörige Energie in keV an
    wave_length = 2*d_lif* np.sin(theta*np.pi/180)
    return h * c / (wave_length*e*1000) 

# Zink-30
Z = 30
theta2, imp = np.genfromtxt("content/data/Zn30.txt", unpack = True)
theta = theta2/2
#plt.subplot(3, 2, 1)

plt.plot(theta, imp, color = "firebrick", label = "Messwerte", linewidth = 0, marker = "x")   
#plt.plot(96-46, imp_mittel, marker = "x", linewidth = 0, color = "cornflowerblue", label = r"$E_{\text{abs}}$")

# Grafische Ermittlung der K-Kante

mean = (96+46)/2        # Mittelwert aus Maximum und Minimum
theta_K = 18.76      # Grafisch austesten: Schnittpunkt mit Messkurve

plt.axhline(mean, color = "cornflowerblue", linestyle = "dotted", label = "Mittelwert")
plt.plot(theta[10:12], imp[10:12], color = "firebrick", linewidth = 1, linestyle = "-")      # Gerade zwischen zwei Messwerten
plt.axvline(theta_K, color = "forestgreen", linestyle = "dotted", label = r"$\theta_K = 18.76°$")

# Plot schöner machen :)

plt.grid()
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
plt.savefig('build/Zn30.pdf')
#plt.show()
plt.close()

# Berechnung der entsprechenden Energie:

E_absorbzn = E_K(theta_K)
sigma_kzn = sigmak(Z, E_absorbzn)

# Abweichung zur Theorie
dE = (E_absorbzn - 9.65)/9.65
dS = (sigma_kzn - 3.56)/3.56
#E_absorbzn = const.h*const.c/(2*d_lif*np.sin(37.56*np.pi/(2*180))*const.e*1000)

# Print der Ergebnisse
print("-----------------------------------------------------------")
print("sigma_k für Zn: ", sigma_kzn, dS)
print("E_abs für Zn : ", E_absorbzn, dE)
print("-----------------------------------------------------------")

# Gallium-31
Z = 31
theta2, imp = np.genfromtxt("content/data/Ga31.txt", unpack = True)
theta = theta2/2

plt.plot(theta, imp, color = "firebrick", label = "Messwerte", linewidth = 0, marker = "x")   

# Grafische Ermittlung der K-Kante

mean = (90+44)/2        # Mittelwert aus Maximum und Minimum
theta_K = 17.525     # Grafisch austesten: Schnittpunkt mit Messkurve

plt.axhline(mean, color = "cornflowerblue", linestyle = "dotted", label = "Mittelwert")
plt.plot(theta[10:12], imp[10:12], color = "firebrick", linewidth = 1, linestyle = "-")      # Gerade zwischen zwei Messwerten
plt.axvline(theta_K, color = "forestgreen", linestyle = "dotted", label = r"$\theta_K = 17.53°$")

# Plot schöner machen :)

plt.grid()
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()

plt.savefig('build/Ga31.pdf')
plt.close()

# Berechnung der entsprechenden Energie:

E_absorbga = E_K(theta_K)
sigma_kga = sigmak(Z, E_absorbga)

# Abweichung zur Theorie
dE = (E_absorbga - 10.37)/10.37
dS = (sigma_kga - 3.61)/3.61

# Print der Ergebnisse
print("-----------------------------------------------------------")
print("sigma_k für Ga : ", sigma_kga, dS)
print("E_abs für Ga : ", E_absorbga, dE)
print("-----------------------------------------------------------")

# Brom-35
Z = 35
theta2, imp = np.genfromtxt("content/data/Br35.txt", unpack = True)
theta = theta2/2

plt.plot(theta, imp, color = "firebrick", label = "Messwerte", linewidth = 0, marker = "x")   

# Grafische Ermittlung der K-Kante

mean = (19+9)/2        # Mittelwert aus Maximum und Minimum
theta_K = 13.5      # Grafisch austesten: Schnittpunkt mit Messkurve

plt.axhline(mean, color = "cornflowerblue", linestyle = "dotted", label = "Mittelwert")
#plt.plot(theta[11:13], imp[11:13], color = "firebrick", linewidth = 1, linestyle = "-")      # Gerade zwischen zwei Messwerten
plt.axvline(theta_K, color = "forestgreen", linestyle = "dotted", label = r"$\theta_K = 13.5°$")

# Plot schöner machen :)

plt.grid()
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()

plt.savefig('build/Br35.pdf')
plt.close()

# Berechnung der entsprechenden Energie:

E_absorbbr = E_K(theta_K)
sigma_kbr = sigmak(Z, E_absorbbr)

# Abweichung zur Theorie
dE = (E_absorbbr - 13.47)/13.47
dS = (sigma_kbr - 3.85)/3.85

# Print der Ergebnisse
print("-----------------------------------------------------------")
print("sigma_k für Br : ", sigma_kbr, dS)
print("E_abs für Br : ", E_absorbbr, dE)
print("-----------------------------------------------------------")

# Strontium-38
Z = 38
theta2, imp = np.genfromtxt("content/data/Sr38.txt", unpack = True)
theta = theta2/2

plt.plot(theta, imp, color = "firebrick", label = "Messwerte", linewidth = 0, marker = "x")   

# Grafische Ermittlung der K-Kante

mean = (18+62)/2        # Mittelwert aus Maximum und Minimum
theta_K = 11.36      # Grafisch austesten: Schnittpunkt mit Messkurve

plt.axhline(mean, color = "cornflowerblue", linestyle = "dotted", label = "Mittelwert")
plt.plot(theta[13:15], imp[13:15], color = "firebrick", linewidth = 1, linestyle = "-")      # Gerade zwischen zwei Messwerten
plt.axvline(theta_K, color = "forestgreen", linestyle = "dotted", label = r"$\theta_K = 11.36°$")

# Plot schöner machen :)

plt.grid()
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()

plt.savefig('build/Sr38.pdf')
plt.close()

# Berechnung der entsprechenden Energie:

E_absorbsr = E_K(theta_K)
sigma_ksr = sigmak(Z, E_absorbsr)

# Abweichung zur Theorie
dE = (E_absorbsr - 16.1)/16.1
dS = (sigma_ksr - 4)/4

# Print der Ergebnisse
print("-----------------------------------------------------------")
print("sigma_k für Sr : ", sigma_ksr, dS)
print("E_abs für Sr : ", E_absorbsr, dE)
print("-----------------------------------------------------------")

# Zirconium-40
Z = 40
theta2, imp = np.genfromtxt("content/data/Zr40.txt", unpack = True)
theta = theta2/2

plt.plot(theta, imp, color = "firebrick", label = "Messwerte", linewidth = 0, marker = "x")   

# Grafische Ermittlung der K-Kante

mean = (37+82)/2        # Mittelwert aus Maximum und Minimum
theta_K = 10.26     # Grafisch austesten: Schnittpunkt mit Messkurve

plt.axhline(mean, color = "cornflowerblue", linestyle = "dotted", label = "Mittelwert")
plt.plot(theta[7:9], imp[7:9], color = "firebrick", linewidth = 1, linestyle = "-")      # Gerade zwischen zwei Messwerten
plt.axvline(theta_K, color = "forestgreen", linestyle = "dotted", label = r"$\theta_K = 10.26°$")

# Plot schöner machen :)

plt.grid()
plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()

plt.savefig('build/Zr40.pdf')
plt.close()

# Berechnung der entsprechenden Energie:

E_absorbzr = E_K(theta_K)
sigma_kzr = sigmak(Z, E_absorbzr)

# Abweichung zur Theorie
dE = (E_absorbzr - 17.99)/17.99
dS = (sigma_kzr - 4.1)/4.1

# Print der Ergebnisse
print("-----------------------------------------------------------")
print("sigma_k für Zr : ", sigma_kzr, dS)
print("E_abs für Zr : ", E_absorbzr, dE)
print("-----------------------------------------------------------")

# Moseley

EHS = [np.sqrt(E_absorbzn*1000), np.sqrt(E_absorbga*1000), np.sqrt(E_absorbbr*1000),  np.sqrt(E_absorbsr*1000),  np.sqrt(E_absorbzr*1000)]
ZETS = [30-sigma_kzn, 31-sigma_kga, 35- sigma_kbr, 38-sigma_ksr, 40-sigma_kzr]

# Fit
def linfit(x,m,b):
    return m*x+b

params, pcov = op.curve_fit(linfit, ZETS, EHS)

print("Rydbergkonstante nach dem Fit: ", params[0]**2)
print(*params)
x = np.linspace(26, 36, 100)

plt.plot(x, linfit(x, *params), color = "cornflowerblue", label = "Fit")
plt.plot(ZETS, EHS,"x", color = "firebrick", label = "experimentelle Werte")
plt.grid()
plt.xlabel(r'$z_\text{eff}$')
plt.ylabel(r'$\sqrt{E_\text{abs}}\mathbin{/}\sqrt{\symup{eV}} $')
plt.xlim(26, 36)
#plt.ylim(95, 135)
plt.legend()

plt.savefig('build/Rydberg.pdf')
#plt.show()
plt.close()

#Abweichungen

dEa = np.abs(E_Kalphaexp-8)/8
dEb = np.abs(E_Kbetaexp-8.95)/8.95
print(dEa, dEb)

dR = np.abs(params[0]**2-13.6)/13.6
print("Abweichung der Rybergkonstante: ", dR)
