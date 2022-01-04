import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
L = 32.351*10**(-3)     # H
C = 0.8015*10**(-9)     # F
C_sp = 0.037*10**(-9)   # F
n_gem = [14, 12, 10.5, 8.5, 7, 6, 4.5, 3]   # Messwerte zu a)

C_k_, nu_pg, nu_mg =  np.genfromtxt("content/data_b.txt", unpack=True)      # Messwerte zu b)
nu_pg = nu_pg*10**(3)   # Umrechnen in Hertz
nu_mg = nu_mg*10**(3)      

# uarray der Kondensatorwerte mit Fehlern
temp = [9.99*0.003, 8.00*0.003, 6.47*0.003, 5.02*0.003, 4.00*0.003, 3.00*0.003, 2.03*0.003, 1.01*0.003]
C_k = unp.uarray([9.99, 8.00, 6.47, 5.02, 4.00, 3.00, 2.03, 1.01],temp)
C_k = C_k*10**(-9)

# Berechnung der Theoriewerte von nu+, nu- und der Schwebungsfrequenz (nu_s)
nu_p = 1/(2*np.pi*unp.sqrt(L*(C+C_sp)))
nu_m = 1/(2*np.pi*unp.sqrt(L*(((1/C)+(2/C_k))**(-1)+C_sp)))
nu_s = (nu_m-nu_p)          # dient nur zur Berechnung des Verhältnisses n

print("-------------------------------------------------------------")
print("Theoriewerte:")
print("nu+:     ", nu_p)
print("nu-:     ", nu_m)

# relaitive Abweichung zur Resonanzfrequenz
res = 30700 # Hertz

print("-------------------------------------------------------------")
print("Relative Abweichung zur Resonanzfrequenz:")
print((np.abs(nu_p-res))/nu_p)

# a) Berechnung des Amplitudenverhältnisses
n = (nu_p + nu_m)/(2*(nu_m - nu_p))

print("-------------------------------------------------------------")
print("Amplitudenverhältnis:")
print(n)

# relative Abweichung
n_abw = np.abs(n-n_gem)/n

print("")
print("Relative Abweichungen:")
print(unp.nominal_values(n_abw))

# Berechnung der relativen Abweichungen zu Messwerten aus b)
nu_pabw = np.abs(nu_p-nu_pg)/nu_p
nu_mabw = np.abs(nu_m-nu_mg)/nu_m

print("-------------------------------------------------------------")
print("Relative Abweichungen zu Messwerten aus b):")
print("nu+:     ", unp.nominal_values(nu_pabw))
print("nu-:     ", unp.nominal_values(nu_mabw))

# c) Umrechnen der Zeitpunkte von t+, t- in Frequenzen
def linear(x,a,b):
    return a*x+b

a = (49.4-19.7)/(11)        # Parameter der Geraden
b = 19.7

n_p = linear(4.4,a,b)*10**(3)                       # t+ war konstant 4.4µs
t_m = np.array([5.2,5.6,5.8,6.0,6.2,7.0,8.0])       # Messwerte zu t-
n_m = linear(t_m,a,b)*10**(3)                       # Dazugehörige Frequenzen

print("-------------------------------------------------------------")
print("Messwerte aus c):")
print("nu+:     ", n_p)
print("nu-:     ", n_m)

# Relative Abweichungen
n_pabw = np.abs(n_p - nu_p)/nu_p
n_mabw = np.abs(n_m - nu_m[0:7])/nu_m[0:7]

print("")
print("Relative Abweichungen der Messwerte aus c):")
print("nu-:     ", unp.nominal_values(n_mabw))
print("nu+:     ", n_pabw)

print("")
print("Relative Abweichungen:")
print("nu+:     ", unp.nominal_values(n_pabw))
print("nu-:     ", unp.nominal_values(n_mabw))
print("-------------------------------------------------------------")
