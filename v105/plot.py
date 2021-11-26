import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const


#plt.rcParams.update({'font.size': 22})          # Stellt Fontsize ein

#Materialkonstanten
r_z = ufloat(0.995/2, 0.001)*(10**(-2))         # Radius der Masse am Aluminiumstab
l_ks = ufloat(1.25, 0.001)*(10**(-2))           # Länge des Kugelstiels
r_k = ufloat(5.4/2,0.001)*(10**(-2))            # Radius der Kugel
m_k = 0.14176                                   # Masse der Kugel in kg
m_z = 0.00139                                   # Masse der Zylinderförmigen Masse am Alustab in kg

N = 195                                         # Anzahl der Windungen der Spulen
R_Spule = 0.109                                 # Radius der Spulen
d = 0.138                                       # Abstand der Spulen zueinander

def B(I):
    return N * ((const.mu_0)*I*(R_Spule**2))/(((R_Spule**2)+(d/2)**2)**(3/2))       # Definition des B-Feldes

def f(x, m, b): 
    return m*x + b                                                                  # Definition einer linearen Funktion (Ausgleichsgerade)    



# Messung A: Ausnutzung der Gravitation

r_m, I_1 = np.genfromtxt("content/Messung1.txt", unpack = True)         # Importieren der Messwerte: r_m ist nur das Stück zwischen Kugelstiel und dem unteren Ende der Masse
I_1 = unp.uarray(I_1,0.05)                                              # I_1 hat den Fehler 0.05 A
r_m = unp.uarray(r_m,0.001)*(10**(-2))                                  # r_m hat den Fehler 0.001 cm

r_ges = (r_m +  r_z + l_ks + r_k)                                       # Gesamte Hebellänge in m

B_1 = B(I_1)                                                            # Werte für B-Feld aus Messwerten

# Linearer Fit zu Messung A
params1, pcov1 = op.curve_fit(f, unp.nominal_values(r_ges), unp.nominal_values(B_1), sigma=unp.std_devs(B_1))
err = np.sqrt(np.diag(pcov1))       # Fehler der Variablen aus dem Fit
x1 = np.linspace(0,0.15,100)
a1 = ufloat(params1[0], err[0])
mu1 = (m_z*const.g)/a1              # Berechnung des Dipolmoments

print(a1, mu1)


# Plot zu Messung A
plt.errorbar(unp.nominal_values(r_ges), unp.nominal_values(B_1), xerr = unp.std_devs(r_ges), yerr = unp.std_devs(B_1), fmt='r.', label= 'Messdaten')
plt.plot(x1, f(x1, params1[0], params1[1]), label='lineare Regression')

plt.xlim(0.04, 0.11)
plt.ylim(0, 0.005)
plt.xlabel("$r\, / \, \mathrm{cm}$ ")
plt.ylabel("$B\, / \, \mathrm{mT}$")
plt.yticks(ticks = [0.001, 0.002, 0.003, 0.004], labels = ["1", "2", "3", "4"])
plt.xticks(ticks = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], labels = ["4", "5", "6", "7", "8", "9", "10"])
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')

plt.savefig('build/plot1.pdf')
plt.close()


# Messung B: Schwingungsdauer
I_2, T = np.genfromtxt("content/Messung2_T.txt", unpack = True)
T = unp.uarray(T,0.025)                                             # Fehler ist die eigene Reaktionszeit (in s)
I_2 = unp.uarray(I_2,0.05)                                          # Fehler von I_2 ist 0.05 A
T_squared = T**2                                                                                                            
B_2 = 1/B(I_2)                                                      # Werte zum B-Feld 2 bzw 1/B       

# Linearer Fit zu Messung B
params2, pcov2 = op.curve_fit(f, unp.nominal_values(B_2), unp.nominal_values(T_squared), sigma=unp.std_devs(T_squared))
x2 = np.linspace(0,350000,100000)
a2 = ufloat(params2[0],np.sqrt(np.diag(pcov2))[0])
J_k = (2/5)*m_k*(r_k**2)
mu1 = (4*(np.pi**2)*J_k)/a2

print(a2, mu1)

# Plot zu Messung B
plt.errorbar(unp.nominal_values(B_2), unp.nominal_values(T_squared), xerr = unp.std_devs(B_2), yerr = unp.std_devs(T_squared), fmt='r.', label='Messdaten')
plt.plot(x2, f(x2, params2[0], params2[1]), label='lineare Regression')
#plt.xlabel(r'$B^{-1} \,/\, \mathrm{T}^{-1}\cdot 10^{-4}$')
plt.ylabel(r'$T^{2} \,/\, \mathrm{s}^{2}$')
plt.xlabel(r'$B^{-1}   \,/\, \mathrm{T}^{-1}$')
plt.xlim(0, 1750)
plt.ylim(0, 6.5)
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')
plt.savefig('build/plot2.pdf')
plt.close()