import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.constants as const
import scipy.optimize as op

N_00g = ufloat(1295, np.sqrt(1295))/900     # Signale ohne Strahlungsquelle bei Gamma-Apparatur
N_00b = ufloat(553, np.sqrt(553))/900       # Signale ohne Strahlungsquelle bei Beta-Apparatur

# Einlesen der Daten
d_pb, N_pb, t_pb = np.genfromtxt("content/data/gamma_Pb.txt", unpack = True)    # Daten des Gamma-Strahler zu Blei (Pb)
d_zn, N_zn, t_zn = np.genfromtxt("content/data/gamma_Zn.txt", unpack = True)    # Daten des Gamma-Strahler zu Zink (Zn)

# Fehler des Messwertes N ist sqrt(N) wegene 
N_pb = unp.uarray(N_pb, np.sqrt(N_pb))
N_zn = unp.uarray(N_zn, np.sqrt(N_zn))

# Umrechnungen usw.
N_pb = N_pb/t_pb - N_00g
N_zn = N_zn/t_zn - N_00g
# d_pb = d_pb *10**(-3) # m     
# d_zn = d_zn *10**(-3) # m 

d_b, err_d_b, N_b, t_b = np.genfromtxt("content/data/beta.txt", unpack = True)  # Daten des Beta-Strahlers (Al)
d_b = unp.uarray(d_b, err_d_b)#*10**(-6) # m
N_b = unp.uarray(N_b, np.sqrt(N_b))

N_b = N_b/t_b - N_00b

# Ausgabe der Nullmessungen
print("--------------------------------------------------------------------------------")
print("Nullmessungen:")
print("Gamma:       , ohne Strahler: ", N_00g, " Messwert ohne Absorber: ", N_pb[0])
print("Beta: ", N_00b)
print("--------------------------------------------------------------------------------")

# Konstanten

Z_pb = 82
Z_zn = 30
rho_pb = 11300 # kg/m^3
rho_zn =  7140 # kg/m^3
M_pb =  0.2072 # kg/mol
M_zn = 0.06539 # kg/mol

# Regression der Messwerte

N_pb_log = unp.log(N_pb)
N_zn_log = unp.log(N_zn)

def f(x, a, b):
    return -a*x + b

def exp(x, N_0, mu):
    return N_0*np.exp(-mu*x)    

params_pb, pcov_pb = op.curve_fit(f, d_pb, noms(N_pb_log))
err_pb = np.sqrt(np.diag(pcov_pb))
params_zn, pcov_zn = op.curve_fit(f, d_zn, noms(N_zn_log))
err_zn = np.sqrt(np.diag(pcov_zn))

# Bestimmung des Absorptionskoeffizienten

mu_pb = ufloat(params_pb[0], err_pb[0])*10**3   # Umrechnen pro meter
mu_zn = ufloat(params_zn[0], err_zn[0])*10**3

b_pb = ufloat(params_pb[1], err_pb[1])
b_zn = ufloat(params_zn[1], err_zn[1])
N_0_pb = unp.exp(b_pb)
N_0_zn = unp.exp(b_zn)

print("--------------------------------------------------------------------------------")
print("Parameter des Fits:")
print("Blei:    a*10^3 = mu = ", mu_pb, "    b = log(N_0) = ", b_pb, "      N_0 = ", N_0_pb)
print("Zink:    a*10^3 = mu = ", mu_zn, "    b = log(N_0) = ", b_zn, "      N_0 = ", N_0_zn)
print("--------------------------------------------------------------------------------")

# Theoriewerte
epsilon = 1.295
r_e = 2.82e-15 # m (klassischer Elektronenradius)

sigma_com = 2*np.pi*r_e**2*((1+ epsilon)/epsilon**2 * (2*(1+epsilon)/(1+2*epsilon) - 1/epsilon*np.log(1+2*epsilon)) + 1/(2*epsilon)*np.log(1+2*epsilon) - (1+ 3*epsilon)/(1+2*epsilon)**2)

n_pb = Z_pb*const.N_A*rho_pb/M_pb
n_zn = Z_zn*const.N_A*rho_zn/M_zn

mu_pb_theo = sigma_com*n_pb
mu_zn_theo = sigma_com*n_zn
print("--------------------------------------------------------------------------------")
print("Theoriewerte:")
print("Sigma_com: ", sigma_com, "[m^2]")
print("Blei:    mu_pb = ", mu_pb_theo, "[m^{-1}]")
print("Zink:    mu_zn = ", mu_zn_theo, "[m^{-1}]")
print("--------------------------------------------------------------------------------")

# Plots
# Blei

x = np.linspace(-2, 45, 100)

plt.subplot(1, 2, 1)
plt.plot(x, f(x, *params_pb), color = "cornflowerblue", label = "Linearer Fit")
plt.errorbar(d_pb, noms(N_pb_log), yerr = stds(N_pb_log), linestyle = None, color = "firebrick", fmt = ".", label = "Messwerte", capsize = 3)
plt.xlabel(r'$d \mathbin{/} \unit{\milli\metre}$')
plt.ylabel(r'$\mathrm{log}(N \mathrm{s})$')
plt.title("Blei")

plt.xlim(0, 45)
plt.grid()
plt.legend(loc='best')

# Zink
x2 = np.linspace(-2, 25, 100)

plt.subplot(1, 2, 2)
plt.plot(x2, f(x2, *params_zn), color = "cornflowerblue", label = "Linearer Fit")
plt.errorbar(d_zn, noms(N_zn_log), yerr = stds(N_zn_log), linestyle = None, color = "firebrick", fmt = ".", label = "Messwerte", capsize = 3)
plt.xlabel(r'$d \mathbin{/} \unit{\milli\metre}$')
plt.ylabel(r'$\mathrm{log}(N \mathrm{s})$')
plt.title("Zink")

plt.xlim(0, 25)
plt.ylim(3.8, 4.8)
plt.grid()
plt.tight_layout()
plt.legend(loc='best')

#plt.show()
plt.savefig('build/gamma.pdf')
plt.close()

# Plot zur beta-Strahlung

N_b_log = unp.log(N_b)

divider = 7 # Zur Unterteilung der Messwerte (rote und blaue Punkte)

params, pcov = op.curve_fit(f, noms(d_b[:divider]), noms(N_b_log[:divider]))
err = np.sqrt(np.diag(pcov))
a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])

const_ = np.mean(noms(N_b_log[divider:]))
const_err = np.std(noms(N_b_log[divider:]))
D_max = -(const_ - b)/a # micro metre

rho_Al = 2.7 # g/cm^3
R_max = rho_Al*D_max*10**(-4)   # g/cm^2 

E_max = 1.92*unp.sqrt(R_max**2 + 0.22*R_max) # MeV
wave = const.c*const.h/(E_max*10**6*const.e)

print("--------------------------------------------------------------------------------")
print("Parameter des zweiten Fits:")
print("a = ", a)
print("b = ", b)
print("const = ", const_,"+-", const_err)
print("D_max = ", D_max, "[??m]")
print("R_max = ", R_max, "[g/cm^2]")
print("E_max = ", E_max, "[MeV]")
print("lambda_max = ", wave, "[m]")
print("--------------------------------------------------------------------------------")

# Plot
x = np.linspace(50, 500, 10)


plt.vlines(noms(D_max), ymin = -10, ymax = 5, color = "forestgreen", label = r"$D_\text{max}$", ls = "dashed")
plt.plot(x, f(x, *params), color = "cornflowerblue", label = "Absorptionskurve")
plt.hlines(y = const_, xmin=50, xmax=500, colors='chocolate', linestyles='-', label = "Hintergrund")
#plt.text(noms(D_max) - 100, const_ -1, r"$D_\text{max} = 341 \unit{\micro\metre}$",)
plt.errorbar(noms(d_b[:divider]), noms(N_b_log[:divider]), xerr = stds(d_b[:divider]), yerr = stds(N_b_log[:divider]), linestyle = None, color = "mediumblue", fmt = ".", label = "Messwerte", capsize=3)
plt.errorbar(noms(d_b[divider:]), noms(N_b_log[divider:]), xerr = stds(d_b[divider:]), yerr = stds(N_b_log[divider:]), linestyle = None, color = "firebrick", fmt = ".", label = "Messwerte", capsize=3)

plt.grid()

plt.xlabel(r'$d \mathbin{/} \unit{\micro\metre}$')
plt.ylabel(r'$\mathrm{log}(N \mathrm{s})$')
plt.ylim(-10, 5)
plt.xlim(50, 500)
plt.legend()
plt.tight_layout()

#plt.show()
plt.savefig('build/beta.pdf')
plt.close()

#####Abweichungen#######
deltaEmax = np.abs(0.294 - E_max)/(0.294)
print("Abweichung der Energie der Betateilchen",deltaEmax)
N_0_exp = ufloat(102.7,1.9)
deltan0pb = np.abs(N_0_exp - N_0_pb)/N_0_exp
deltan0zn = np.abs(N_0_exp - N_0_zn)/N_0_exp
print("Abw. der Nullrate zu Pb:", deltan0pb)
print("Abw. der Nullrate zu Zn:", deltan0zn)
