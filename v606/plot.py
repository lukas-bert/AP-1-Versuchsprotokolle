import matplotlib.pyplot as plt
import numpy as np

# Bestimmung der Filterkurve

f, U = np.genfromtxt("content/data/Filterkurve.txt", unpack = True)


plt.axvline(21.6, ymin = 0, ymax = 0.85, color = "cornflowerblue", linestyle = "dotted")
plt.plot(21.6, 8.5, marker = "o", markeredgecolor = "cornflowerblue", markersize = 8, linewidth = 0, label = "Maximum")
plt.plot(f, U, color = "firebrick", marker = "x", label = "Messwerte", linewidth = 0)

#plt.text(22, 8.5, r"$\qty{8.5}{\volt}$")
#plt.xlabel(r'$f \mathbin{/} \unit{\kilo\hertz}$')
#plt.ylabel(r'$U \mathbin{/} \unit{\volt}$')
plt.ylim(0, 10)
plt.legend()
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot.pdf')
#plt.show()
plt.close()

# Ermittlung der Suszeptibilitäten

# Konstanten der Spule / Schaltung

n = 250                     # Windungen 
A_s = 0.000866              # Fläche in m^2
l_s = 0.135                 # Länge in m 
R_s = 0.7                   # Widerstand der Spule in Ohm     
R_3 = 998                   # Widerstand 3 in Ohm        

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

Q_d = 14.38/(15.3 * 7.8) * 10**-4       # effektiver Querschnitt der Probe in Kubikmetern (masse[gramm]/(länge[cm]*Dichte[gram/cm^3]))
Q_g = 10.2/(15.5 * 7.4) * 10**-4
Q_c = np.pi * (0.0085/2)**2             # keine Literaturwerte zur Dichte, also Querschnitt der Probe
print(Q_d, Q_g, Q_c)

# Messdaten

R_0d, R_2d, U_bd, U_0d, U_2d = np.genfromtxt("content/data/Probe1.txt", unpack = True) * 10**(-3)     # Widerstände noch mit 5 multiplizieren
R_0g, R_2g, U_bg, U_0g, U_2g = np.genfromtxt("content/data/Probe1.txt", unpack = True) * 10**(-3)
R_0c, R_2c, U_bc, U_0c, U_2c = np.genfromtxt("content/data/Probe1.txt", unpack = True) * 10**(-3)

R_0d = R_0d *5
R_2d = R_2d *5 
R_0g = R_0d *5
R_2g = R_2d *5 
R_0c = R_0d *5
R_2c = R_2d *5 

# Funktionen zur Berechnung der Suszeptibilität

def chi_R(delta_R, Q):              # delta_R: Differenz der Widerstände vor und nach Abgleich, Q: Querschnitt der Probe
    return 2*A_s*delta_R/(R_3*Q)

def chi_U(U_0, U_b, Q):             # Über Spannungsmessung: U_0 Generatorspannung, U_b: Brückenspannung
    return 4*A_s*U_b/(Q*U_0)    

chi_Rd = chi_R(R_0d - R_2d, Q_d)
chi_Rg = chi_R(R_0g - R_2g, Q_g)
chi_Rc = chi_R(R_0c - R_2c, Q_c)

chi_Ud = chi_U(1, np.abs(U_bd-U_0d), Q_d)
chi_Ug = chi_U(1, np.abs(U_bg-U_0g), Q_g)
chi_Uc = chi_U(1, np.abs(U_bc-U_0c), Q_c)

print(chi_Rd, chi_Rg, chi_Rc)
print(chi_Ud, chi_Ug, chi_Uc)

#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#
#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')
#