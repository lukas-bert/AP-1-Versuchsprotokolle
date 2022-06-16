import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat

U_20, I_20 = np.genfromtxt("content/data/I_20.txt", unpack = True)
U_21, I_21 = np.genfromtxt("content/data/I_21.txt", unpack = True)
U_22, I_22 = np.genfromtxt("content/data/I_22.txt", unpack = True)
U_23, I_23 = np.genfromtxt("content/data/I_23.txt", unpack = True)
U_24, I_24 = np.genfromtxt("content/data/I_24.txt", unpack = True)

I_20 = I_20/4
I_21 = I_21/4
I_22 = I_22/4
I_23 = I_23/2
I_24 = I_24/2

def prettyPlot():
    plt.ylabel(r'$I \mathbin{/} \unit{\milli\ampere}$')
    plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
    plt.grid()
    plt.legend()
    plt.tight_layout()

# Erster Plot zu den Kennlinien

plt.plot(U_20, I_20, "." , linewidth = 0, label = r"$I = \qty{2.0}{\ampere} \,\,\text{mit}\,\, I_s = \qty{0.072}{\milli\ampere}$", color = "mediumblue")
plt.hlines(0.072, 180, 240, linestyle = "dashed", color = "mediumblue")  #label = "Sättigungsstrom",

plt.plot(U_21, I_21, "." , linewidth = 0, label = r"$I = \qty{2.1}{\ampere} \,\,\text{mit}\,\, I_s = \qty{0.154}{\milli\ampere}$", color = "firebrick")
plt.hlines(0.154, 210, 250, linestyle = "dashed", color = "firebrick")

plt.plot(U_22, I_22, "." , linewidth = 0, label = r"$I = \qty{2.2}{\ampere} \,\,\text{mit}\,\, I_s = \qty{0.35}{\milli\ampere}$", color = "chocolate")
plt.hlines(0.35, 210, 250, linestyle = "dashed", color = "chocolate")

plt.plot(U_23, I_23, "." , linewidth = 0, label = r"$I = \qty{2.3}{\ampere} \,\,\text{mit}\,\, I_s = \qty{0.6305}{\milli\ampere}$", color = "forestgreen")
plt.hlines(0.6305, 240, 250, linestyle = "dashed", color = "forestgreen")

prettyPlot()

#plt.show()
plt.savefig("build/plot1.pdf")
plt.close()

# Zweiter Plot zur missratenen Kennlinie

plt.plot(U_24, I_24, "." , linewidth = 0, label = r"$I = \qty{2.4}{\ampere} \,\,\text{mit}\,\, I_s = \qty{1.08}{\milli\ampere}$", color = "royalblue")
prettyPlot()

#plt.show()
plt.savefig("build/plot2.pdf")
plt.close()

# Berechnung der Temperaturen der Anoden

f = 0.32        # cm^2 Drahtfläche
sigma = 5.7e-12 # W/cm^2K Stefan-Boltzman
n = 0.28        # Emsissionsgrad 
P_WL = 0.95     # W Wärmeleistung der Diode

def Temp(I, U):
    return ((I*U - P_WL)/(f*n*sigma))**(1/4)

Temperaturen = Temp(np.array([2.0, 2.1, 2.2, 2.3, 2.4]), np.array([3.5, 4.0, 4.3, 4.7, 5]))   # np.array damit Multiplikation klappt
print("--------------------------------------------------------------------------")
print("Temperaturen der Diode bei verschiedenen Stromstärken:")
print(Temperaturen)
print("--------------------------------------------------------------------------")

# Berechnung der Austrittsarbeit

e = const.e 
m = const.m_e
k_B = const.k
h = const.h
e_0 = const.epsilon_0

I_S = np.array([0.072, 0.154, 0.35, 0.6305, 1.08])*10**(-3) # Sättigungsströme in A

def phi(T, I):
    return - (k_B * T / e)*np.log( I*h**3 / (f*10**(-4)*4*np.pi*e*m*k_B**2*T**2))

phi_W = phi(Temperaturen, I_S)
print("Austrittsarbeiten in eV:", phi_W)   

# Mittelwert

W_A = ufloat(np.mean(phi_W), np.std(phi_W))
print("Mittelwert: ", W_A)

# Raumladungsdichte (Langmuir-Schottkysch)

a = 0.03 # m Plattenabstand?
##I_25 = I_25*10**(-3)
#
#def I(U, exp, a):
#    return a* U**(exp)
#
#m, mcov = op.curve_fit(I, U_25, I_25)
#m_err = np.sqrt(np.diag(mcov))
#print(m, m_err)
#
#x = np.linspace(0, 150, 10000)
#
#plt.subplot(121)
#plt.plot(U_25, I_25**(2/3), "x" , linewidth = 0, label = "Messwerte", color = "firebrick")
#plt.xlabel("I^(2/3)")
#plt.grid()
#
#plt.subplot(122)
#
#plt.plot(x, I(x, m), color = "cornflowerblue", label = "Fit")                           # da muss noch nen Fehler sein amk
#plt.plot(np.log(U_25), np.log(I_25), "x" , linewidth = 0, label = "Messwerte", color = "firebrick")
#
#plt.tight_layout()
#plt.grid()
#plt.xlim(0, 150)
#plt.ylim(0)
#
#plt.show()
#plt.savefig("build/Raumladung.pdf")
#plt.close()

# Try über lineare Regression:

I_log = np.log(I_24[1:])
U_log = np.log(U_24[1:])

def f(x, m, b):
    return (m*x + b) 

params, pcov = op.curve_fit(f, U_log, I_log)
errors = np.sqrt(np.diag(pcov))

print("Parameter Raumladung-Fit:", params, errors)

x = np.linspace(0, 5.5, 1000)

plt.plot(x, f(x, *params), color = "cornflowerblue", label = "Lineare Regression")                           # da muss noch nen Fehler sein amk
plt.plot(U_log, I_log, "x" , linewidth = 0, label = "Messwerte", color = "firebrick")

plt.grid()
plt.xlim(0.5, 5.2)
plt.ylim(-6, 1)

plt.ylabel(r'$\mathrm{log}(I \mathbin{/} \unit{\milli\ampere}$)')
plt.xlabel(r'$\mathrm{log}(U \mathbin{/} \unit{\volt}$)')
plt.legend()
plt.tight_layout()

#plt.show()
plt.savefig("build/Raumladung.pdf")
plt.close()

# Anlaufstromgebiet

R_I = 1e6 # Ohm

U_G, I_A = np.genfromtxt("content/data/Anlaufkurve.txt", unpack = True)
I_A = I_A*10**(-9)  # Ampere

U_G = U_G + I_A*R_I # Korrektur der gemessenen Spannung

# I ~ c* exp(a*U),  a ~ 1/T

params, pcov = op.curve_fit(f, U_G, np.log(I_A))   
err = np.sqrt(np.diag(pcov))

m = ufloat(params[0], err[0])

print("Parameter des Fits (m, b): ", params, err)

# Plot

x = np.linspace(0, 1, 100)

plt.plot(x, f(x, *params), color = "cornflowerblue", label = "Linearer Fit")
plt.plot(U_G, np.log(I_A), "x", color = "firebrick", linewidth = 0, label = "Messwerte")
plt.xlim(0, 1)
plt.ylim(-24, -17.5)


plt.ylabel(r'$\mathrm{log}(I \mathbin{/} \unit{\ampere}$)')
plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
plt.grid()
plt.legend()
plt.tight_layout()

#plt.show()
plt.savefig("build/Anlaufstrom.pdf")
plt.close()

T = -e/(k_B*m)
print("Aus Anlaufstrom bestimmte Temperatur:", '{:.4f}%'.format(T))

#abweichungen:
a1 = ufloat(4.68, 0.10)
deltam = np.abs(a1 - 4.55)/(4.55)
print(deltam)