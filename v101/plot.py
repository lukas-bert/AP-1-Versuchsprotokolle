import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat

# Berechnung der Winkelrichtgröße D
phi, F = np.genfromtxt("content/data1.txt", unpack = True)
phi = phi * np.pi / 360     # Umrechnung in Bogenmaß
a = 0.2                     # Länge des Kraftarms in m

D_ = a* F/phi
D = D_.mean()               # Mittelwert
std_D = D_.std(ddof = 1)    # Mittelwertfehler
D = ufloat(D, std_D)
print("Winkelrichtgröße:    ", D)

# Bestimmung des Eigenträgheitsmoments 

a, T_3 = np.genfromtxt("content/data2.txt", unpack = True)
T = T_3/3       # Mitteln auf eine Periodendauer
a = a*10**(-2)  # Umrechnen in m

def linear(x, m, b):
    return m*x + b

params, pcov = op.curve_fit(linear, a**2, T**2)     # Lineare Regression
err = np.sqrt(np.diag(pcov))                       # Fehler aus Kovarianz-Matrix
x = np.linspace(0, 0.1, 100)

plt.plot(a**2, T**2, 'rx', label = "Messdaten")
plt.plot(x, linear(x, params[0], params[1]), label = "Lineare Regression")
plt.xlabel(r'$a^2 \mathbin{/} \symup{m^2}$')
plt.ylabel(r'$T^2 \mathbin{/} \symup{s^2}$')
plt.xlim(0, 0.1)
plt.ylim(0, 80)
plt.legend()

# Kleiner Plot im Plot
plt.axes([0.6, 0.25, 0.3, 0.25])
plt.plot(a**2, T**2, 'rx', label = "Messdaten")
plt.plot(x, linear(x, params[0], params[1]), label = "Lineare Regression")
plt.xlim(0, 0.02)
plt.ylim(0, 20)

plt.tight_layout()
plt.savefig('build/plot.pdf')
plt.close()

print("Parameter der Geraden, Fehler:   ", params, err)
b = ufloat(params[1], err[1])

# Berechnung des Eigenträgeheitmoments I_D

m = 0.2612  # Masse eines Gewichts in kg
h = 0.02    # Höhe in m
r = 0.0225    # Radius in m

I_D = b*D/(4*np.pi**2) - m*(r**2/2 + h**2/6)
print("Eigenträgheitsmoment:    ", I_D)

# Bestimmung der Trägheitsmomente von Zylinder und Kugel
# Funktionen der Trägeheitsmomente

def zylinder_p(m, R, a):                    # Zylinder parallel zur Achse
    return m*R**2/2 + m*a**2

def zylinder_o(m, R, h, a):                 # Zylinder orthogonal "  "
    return m*(R**2/4 + h**2/12) + m*a**2    

# Theoriewerte
m_k = 1.1703    # Masse der Kugel in kg
r_k = 0.1472/2  # Radius in m

I_kt = 2/5*m_k*r_k**2

m_z = 0.3677    # Masse des Zylinders in kg
r_z = 0.0983/2  # Radius in m
h_z = 0.1009    # Höhe in m

I_zt = zylinder_p(m_z, r_z, 0)

# Experimentelle Werte

T_z_5, T_k_5 = np.genfromtxt("content/data3.txt", unpack = True)    # je 5 Periodenlängen gemessen
T_z = T_z_5/5
T_k = T_k_5/5

# Mittelwerte
T_z = ufloat(T_z.mean(), T_z.std(ddof = 1))
T_k = ufloat(T_k.mean(), T_k.std(ddof = 1))

# Trägheitsmomente
I_ke = T_k**2/(4*np.pi**2)*D
I_ka = unp.nominal_values(np.abs(I_kt-I_ke)/I_kt)

print("------------------------------------------------------------------------------")
print("Trägheitsmoment der Kugel:")
print("Theoriewert: ", I_kt, "Experimentell: ", I_ke, "Abweichung: ", I_ka)

I_ze = T_z**2/(4*np.pi**2)*D
I_za = unp.nominal_values(np.abs(I_zt-I_ze)/I_zt)

print("Trägheitsmoment des Zylinders:")
print("Theoriewert: ", I_zt, "Experimentell: ", I_ze, "Abweichung: ", I_za)
print("------------------------------------------------------------------------------")

# Trägheitsmoment der Puppe
# Maße

m = 0.1672          # Masse in kg
h_kopf = 0.0414     # Längen in cm
h_arm = 0.1291
h_rumpf = 0.0984
h_bein = 0.1242

kopf, arm, rumpf, bein = np.genfromtxt("content/data4.txt", unpack = True)*10**(-2)/2     # Messwerte der Radien der Gliedmaßen in m !!!!Teilweise mit Nullen aufgefüllt!!!!

r_kopf = ufloat(kopf[0:6].mean(), kopf[0:6].std(ddof = 1))
r_arm = ufloat(arm.mean(), arm.std(ddof = 1))
r_rumpf = ufloat(rumpf[0:8].mean(), rumpf[0:8].std(ddof = 1))
r_bein = ufloat(bein.mean(), bein.std(ddof = 1))

print("----Trägheitsmoment Puppe----")
print("Mittelwerte der Radien:")
print("Kopf: ", r_kopf, "Arm: ", r_arm, "Rumpf: ", r_rumpf, "Bein: ", r_bein)

# Massen der einzelnen Körperteile

def V_z(r, h):
    return h* np.pi* r**2

# Volumina der Gliedmaßen
V_kopf = V_z(r_kopf, h_kopf)   
V_arm = V_z(r_arm, h_arm)         
V_rumpf = V_z(r_rumpf, h_rumpf)   
V_bein = V_z(r_bein, h_bein)      
V_ges = V_kopf + 2*V_arm + V_rumpf + 2*V_bein

m_kopf = m*V_kopf/V_ges
m_arm = m*V_arm/V_ges
m_rumpf = m*V_rumpf/V_ges
m_bein = m*V_bein/V_ges

print("Einzelmassen:")
print("Kopf: ", m_kopf,"Arm: ", m_arm, "Rumpf: ", m_rumpf, "Bein: ", m_bein)

# Theoriewert des Trägheitsmoments

# T-Pose
I_kopf = zylinder_p(m_kopf, r_kopf, 0)
I_arm = zylinder_o(m_arm, r_arm, h_arm, (h_arm/2 + r_rumpf))
I_rumpf = zylinder_p(m_rumpf, r_rumpf, 0)
I_bein = zylinder_p(m_bein, r_bein, (r_rumpf - r_bein))

print("------------------------------------------------------------------------------")
print("T-pose:")
print("Einzelne Trägeheitsmomente:")
print("Kopf: ", I_kopf,"Arm: ", I_arm, "Rumpf: ", I_rumpf, "Bein: ", I_bein)

I_Tpose = I_kopf + 2* I_arm + I_rumpf + 2* I_bein

# Ballerina

I_arm2 = zylinder_p(m_arm, r_arm, (r_rumpf + r_arm))
I_bein2 = zylinder_o(m_bein, r_bein, h_bein, unp.sqrt((h_bein/2)**2 + (r_rumpf - r_bein)**2))
print("------------------------------------------------------------------------------")
print("Ballerina:")
print("Einzelne Trägeheitsmomente:")
print("Arm: ", I_arm2, "Bein: ", I_bein2)

I_Ballerina = I_kopf + 2* I_arm2 + I_rumpf + 2* I_bein2

# Experimentelle Werte

T_Tpose, T_Ballerina = np.genfromtxt("content/data5.txt", unpack = True)/5      # Messwerte von je 5 Perioden

# Mittelwerte
T_Tpose = ufloat(T_Tpose.mean(), T_Tpose.std(ddof = 1))
T_Ballerina = ufloat(T_Ballerina.mean(), T_Ballerina.std(ddof = 1))

I_Tpose_e = T_Tpose**2/(4*np.pi**2)*D
I_Ballerina_e = T_Ballerina**2/(4*np.pi**2)*D

I_Tpose_a = unp.nominal_values(np.abs(I_Tpose-I_Tpose_e)/I_Tpose)
I_Ballerina_a = unp.nominal_values(np.abs(I_Ballerina-I_Ballerina_e)/I_Ballerina)

# Ergebnisse
print("------------------------------------------------------------------------------")
print("Theorie:")
print("I_T-pose: ", '{0:.8f}'.format(I_Tpose))
print("I_Ballerina:", '{0:.8f}'.format(I_Ballerina))
print("Experiment:")
print("I_T-pose: ", '{0:.8f}'.format(I_Tpose_e))
print("I_Ballerina:", '{0:.8f}'.format(I_Ballerina_e))
print("Abweichung:")
print("I_T-pose: ", '{0:.8f}'.format(I_Tpose_a))
print("I_Ballerina:", '{0:.8f}'.format(I_Ballerina_a))

#I_S =  0.18*0.625**2/12 # Test weil Trägeheitsmoment der Stange vernachlässigt wird
#print(I_S, I_D - I_S)
