import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

# Bestimmung der Filterkurve

f, U = np.genfromtxt("content/data/Filterkurve.txt", unpack=True)

U = U/8.5

def Gauss(x, a, b):
    return np.exp(-(x-a)**2*b)

params, pcov = op.curve_fit(Gauss, f, U, p0 = [21.6, 1])            # p0, da Gaussglocke ukm 21.6 nach rechts verschoben

#plt.axvline(21.6, ymin=0, ymax=0.8333, color="forestgreen", linestyle="dotted")
plt.axhline(1/np.sqrt(2), color="gray", linestyle="dotted", label = r"$1 / \sqrt{2}$")
plt.plot(21.6, 1, marker="o", markeredgecolor="firebrick", markersize=8, linewidth=0, label="Maximum d. Messwerte")

x = np.linspace(15, 31, 10000)

plt.plot(x, Gauss(x, *params), color = "cornflowerblue")
plt.plot(21.3641, 1/np.sqrt(2), marker = "*", markersize = 8, color = "hotpink", markeredgecolor = "k", markeredgewidth = 0.65, linewidth=0, label = r"$\nu_{-} \mathbin{/} \nu_{+}$")
plt.plot(22.1435, 1/np.sqrt(2), marker = "*", markersize = 8, color = "hotpink", markeredgecolor = "k", markeredgewidth = 0.65, linewidth=0)
plt.plot(f, U, color="firebrick", marker="x", label="Messwerte", linewidth=0)

#plt.text(22, 8.5, r"$\qty{8.5}{\volt}$")
plt.xlabel(r'$f \mathbin{/} \unit{\kilo\hertz}$')
#plt.yticks(ticks = [0, 0.2, 0.4, 0.6, 1/np.sqrt(2), 0.8, 1], labels = [0, 0.2, 0.4, 0.6, r"$\frac{1}{\sqrt{2}}$", 0.8, 1])
plt.ylabel(r'$U \mathbin{/} U_\text{max}$')
plt.ylim(0, 1.2)
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig("build/plot.pdf")
#plt.show()
plt.close()

G = params[0]/(22.14-21.36)
print("------------------------------------------------")
print("nu_-: ", 21.36)
print("nu_0: ", params[0])
print("nu_+: ", 22.14)
print("Güte der Glockenkurve: ", G)
print("------------------------------------------------")

# Ermittlung der Suszeptibilitäten

# Konstanten der Spule / Schaltung

n = 250  # Windungen
A_s = 86.6*1e-6  # Fläche in m^2
l_s = 0.135  # Länge in m
R_s = 0.7  # Widerstand der Spule in Ohm
R_3 = 998  # Widerstand 3 in Ohm

##################################################
# Indexkonvention:
# d: Dy_2 O_3 (Dysprosium(3)oxid)
# g: Gd_2 O_3 (Gadolinium(3)oxid)
# c: C_6 O_12 Pr_2 (Praseodymium oxalate)
#
# b: Brückenspannung
# 0: Vor der Messung
# 2: Nach der Justierung
##################################################

# Berechnung der realen Querschnitte

m_d = 14.38*1e-3
l_d = 0.153
rho_d = 7800
Q_d = m_d/(l_d*rho_d)     # effektiver Querschnitt der Probe in m^2

m_g = 10.2*1e-3
l_g = 0.155
rho_g = 7400
Q_g = m_g/(l_g*rho_g)

Q_c = (np.pi * (0.0085 / 2) ** 2)           # keine Literaturwerte zur Dichte, also Querschnitt der Probe
#print(Q_d, Q_g, Q_c)

# Messdaten

R_0d, R_2d, U_bd, U_0d, U_2d = np.genfromtxt("content/data/Probe1.txt", unpack=True) * 10 **(-3)    # Widerstände noch mit 5 multiplizieren
R_0g, R_2g, U_bg, U_0g, U_2g = np.genfromtxt("content/data/Probe1.txt", unpack=True) * 10 **(-3)
R_0c, R_2c, U_bc, U_0c, U_2c = np.genfromtxt("content/data/Probe1.txt", unpack=True) * 10 **(-3)

R_0d = R_0d * 5
R_2d = R_2d * 5
R_0g = R_0d * 5
R_2g = R_2d * 5
R_0c = R_0d * 5
R_2c = R_2d * 5

# Funktionen zur Berechnung der Suszeptibilität

def chi_R(
    delta_R, Q
):  # delta_R: Differenz der Widerstände vor und nach Abgleich, Q: Querschnitt der Probe
    return 2 * A_s * delta_R / (R_3 * Q)

def chi_U(
    U_0, U_b, Q
):  # Über Spannungsmessung: U_0 Generatorspannung, U_b: Brückenspannung
    return 4 * A_s * U_b / (Q * U_0)


chi_Rd = chi_R(R_0d - R_2d, Q_d)
chi_Rg = chi_R(R_0g - R_2g, Q_g)
chi_Rc = chi_R(R_0c - R_2c, Q_c)

chi_Ud = chi_U(1, np.abs(U_bd - U_0d), Q_d)
chi_Ug = chi_U(1, np.abs(U_bg - U_0g), Q_g)
chi_Uc = chi_U(1, np.abs(U_bc - U_0c), Q_c)

print("------------------------------------------------")
print("Suszeptibilität aus Widerständen:")
print(chi_Rd, chi_Rg, chi_Rc)
print("------------------------------------------------")
print("Suszeptibilität aus Spannungen:")
print(chi_Ud, chi_Ug, chi_Uc)
print("------------------------------------------------")

# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)
#
# plt.subplot(1, 2, 1)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
# plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
# plt.legend(loc='best')
#
# plt.subplot(1, 2, 2)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
# plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
# plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/plot.pdf')
#
