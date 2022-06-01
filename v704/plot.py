import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.constants as const
import scipy.optimize as op

# Einlesen der Daten
d_pb, N_pb, t_pb = np.genfromtxt("content/data/gamma_Pb.txt", unpack = True)    # Daten des Gamma-Strahler zu Blei (Pb)
d_zn, N_zn, t_zn = np.genfromtxt("content/data/gamma_Zn.txt", unpack = True)    # Daten des Gamma-Strahler zu Zink (Zn)

# Fehler des Messwertes N ist sqrt(N) wegene 
N_pb = unp.uarray(N_pb, np.sqrt(N_pb))
N_zn = unp.uarray(N_zn, np.sqrt(N_zn))


# Umrechnungen usw.
N_pb = N_pb/t_pb
N_zn = N_zn/t_zn
# d_pb = d_pb *10**(-3) # m     
# d_zn = d_zn *10**(-3) # m 

d_b, err_d_b, N_b, t_b = np.genfromtxt("content/data/beta.txt", unpack = True)  # Daten des Beta-Strahlers (Al)
d_b = unp.uarray(d_b, err_d_b)#*10**(-6) # m
N_b = unp.uarray(N_b, np.sqrt(N_b))
N_b = N_b/t_b

# Konstanten

Z_pb = 82
Z_zn = 30
rho_pb = 11400 # kg/m^3
rho_zn =  7100 # kg/m^3
M_pb =  0.2072 # kg/mol
M_zn = 0.06538 # kg/mol

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

N_0_pb = ufloat(params_pb[1], err_pb[1])
N_0_zn = ufloat(params_zn[1], err_zn[1])

print("--------------------------------------------------------------------------------")
print("Parameter des Fits:")
print("Blei:    a*10^3 = mu = ", mu_pb, "    b = N_0 = ", N_0_pb)
print("Zink:    a*10^3 = mu = ", mu_zn, "    b = N_0 = ", N_0_zn)
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
print("Sigma_com: ", sigma_com)
print("Blei:    mu_pb = ", mu_pb_theo)
print("Zink:    mu_zn = ", mu_zn_theo)
print("--------------------------------------------------------------------------------")

# Plots
# Blei

x = np.linspace(-2, 45, 100)

plt.subplot(1, 2, 1)
plt.plot(x, f(x, *params_pb), color = "cornflowerblue", label = "Linearer Fit")
plt.errorbar(d_pb, noms(N_pb_log), yerr = stds(N_pb_log), linestyle = None, color = "firebrick", fmt = ".", label = "Messwerte")
plt.xlabel(r'$d \mathbin{/} \unit{\milli\metre}$')
plt.ylabel(r'$\mathrm{log}(N)$')
plt.title("Plumbum")

plt.xlim(0, 45)
plt.grid()
plt.legend(loc='best')

# Zink
x2 = np.linspace(-2, 25, 100)

plt.subplot(1, 2, 2)
plt.plot(x2, f(x2, *params_zn), color = "cornflowerblue", label = "Linearer Fit")
plt.errorbar(d_zn, noms(N_zn_log), yerr = stds(N_zn_log), linestyle = None, color = "firebrick", fmt = ".", label = "Messwerte")
plt.xlabel(r'$d \mathbin{/} \unit{\milli\metre}$')
plt.ylabel(r'$\mathrm{log}(N)$')
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

params, pcov = op.curve_fit(f, noms(d_b[:6]), noms(N_b_log[:6]))
err = np.sqrt(np.diag(pcov))
a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])

const = np.mean(noms(N_b_log[6:]))
D_max = -(const - b)/a # micro metre

rho_Al = 2.7 # g/cm^3
R_max = rho_Al*D_max*10**(-4)   # g/cm^2 

E_max = 1.92*unp.sqrt(R_max**2 + 0.22*R_max) # MeV

print("--------------------------------------------------------------------------------")
print("Parameter des zweiten Fits:")
print("a = ", a)
print("b = ", b)
print("const = ", const)
print("D_max = ", D_max, "[µm]")
print("R_max = ", R_max, "[g/cm^2]")
print("E_max = ", E_max, "[MeV]")
print("--------------------------------------------------------------------------------")

# Plot
x = np.linspace(50, 300, 10)

plt.plot(x, f(x, *params), color = "cornflowerblue")
plt.hlines(y = const, xmin=50, xmax=500, colors='chocolate', linestyles='-')
plt.plot(noms(D_max), const, marker = "o", color = "forestgreen", label = "Schnittpunkt", lw = 0)
plt.text(noms(D_max) - 20, const -0.3, r"$D_\text{max} = 257 \unit{\micro\metre}$",)
plt.errorbar(noms(d_b[:6]), noms(N_b_log[:6]), xerr = stds(d_b[:6]), yerr = stds(N_b_log[:6]), linestyle = None, color = "mediumblue", fmt = "x", label = "Messwerte")
plt.errorbar(noms(d_b[6:]), noms(N_b_log[6:]), xerr = stds(d_b[6:]), yerr = stds(N_b_log[6:]), linestyle = None, color = "firebrick", fmt = "x", label = "Messwerte")
plt.grid()

plt.xlabel(r'$d \mathbin{/} \unit{\micro\metre}$')
plt.ylabel(r'$\mathrm{log}(N)$')
plt.ylim(-1, 4)
plt.xlim(50, 500)
plt.legend()

plt.show()
plt.savefig('build/beta.pdf')
#plt.close()
