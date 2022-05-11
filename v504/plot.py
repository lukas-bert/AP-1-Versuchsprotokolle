import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat

U_19, I_19 = np.genfromtxt("content/data/I_19.txt", unpack = True)
U_20, I_20 = np.genfromtxt("content/data/I_20.txt", unpack = True)
U_21, I_21 = np.genfromtxt("content/data/I_21.txt", unpack = True)
U_22, I_22 = np.genfromtxt("content/data/I_22.txt", unpack = True)
U_25, I_25 = np.genfromtxt("content/data/I_25.txt", unpack = True)

def prettyPlot():
    #plt.ylabel(r'$I \mathbin{/} \unit{\milli\ampere}$')
    #plt.xlabel(r'$U \mathbin{/} \unit{\volt}$')
    plt.grid()
    plt.legend()
    plt.tight_layout()

# Erster Plot zu den Kennlinien

plt.plot(U_19, I_19, "." , linewidth = 0, label = "Kennlinie zu: I = 1.9A", color = "mediumblue")
plt.hlines(0.042, 45, 60, label = "Sättigungsstrom", linestyle = "dashed", color = "mediumblue")

plt.plot(U_20, I_20, "." , linewidth = 0, label = "I = 2.0A", color = "firebrick")
plt.hlines(0.116, 90, 130, linestyle = "dashed", color = "firebrick")

plt.plot(U_21, I_21, "." , linewidth = 0, label = "I = 2.1A", color = "chocolate")
plt.hlines(0.243, 120, 150, linestyle = "dashed", color = "chocolate")

plt.plot(U_22, I_22, "." , linewidth = 0, label = "I = 2.2A", color = "forestgreen")
plt.hlines(0.562, 160, 170, linestyle = "dashed", color = "forestgreen")

prettyPlot()

#plt.show()
plt.savefig("build/plot1.pdf")
plt.close()

# Zweiter Plot zur missratenen Kennlinie

plt.plot(U_25, I_25, "." , linewidth = 0, label = "Kennlinie zu I = 2.5A")
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

Temperaturen = Temp(np.array([1.9, 2.0, 2.1, 2.2, 2.5]), np.array([3.2, 3.5, 4.0, 4.3, 5.5]))   # np.array damit Multiplikation klappt
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

I_S = np.array([0.042, 0.116, 0.243, 0.562, 1.391])*10**(-3) # Sättigungsströme in A

def phi(T, I):
    return - (k_B * T / e)*np.log( I*h**3 / (f*10**(-4)*4*np.pi*e*m*k_B**2*T**2))

phi_W = phi(Temperaturen, I_S)
print("Austrittsarbeiten in eV:", phi_W)   

# Mittelwert

W_A = ufloat(np.mean(phi_W), np.std(phi_W))
print("Mittelwert: ", W_A)

# Raumladungsdichte (Langmuir-Schottkysch)

a = 0.03 # m Plattenabstand?

def I(U, exp):
    return 4/9*f*e_0*np.sqrt(2*e/m)* U**(exp)/(a**2)

m, mcov = op.curve_fit(I, U_25, I_25)
m_err = np.sqrt(np.diag(mcov))
print(m, m_err)

x = np.linspace(0, 150, 10000)

plt.subplot(121)
plt.plot(U_25, I_25**(2/3), "x" , linewidth = 0, label = "Messwerte", color = "firebrick")
plt.xlabel("I^(2/3)")
plt.grid()

plt.subplot(122)

plt.plot(x, I(x, m), color = "cornflowerblue", label = "Fit")                           # da muss noch nen Fehler sein amk
plt.plot(U_25, I_25, "x" , linewidth = 0, label = "Messwerte", color = "firebrick")

plt.grid()
plt.xlim(0, 150)
plt.ylim(0)

#plt.show()
plt.savefig("build/Raumladung.pdf")
plt.close()

# Anlaufstromgebiet

R_I = 1e6 # Ohm

U_G, I_A = np.genfromtxt("content/data/Anlaufkurve.txt", unpack = True)
I_A = I_A*10**(-9)  # Ampere

U_G = U_G + I_A*R_I # Korrektur der gemessenen Spannung

# I ~ c* exp(a*U),  a ~ 1/T

def f(x, m, b):
    return (m*x + b) 

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
plt.grid()
plt.legend()

plt.show()
plt.savefig("build/Anlaufstrom.pdf")
plt.close()

T = -e/(k_B*m)
print("Aus Anlaufstrom bestimmte Temperatur:", T)

