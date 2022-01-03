import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
L = 32.351*10**(-3)  #H
C = 0.8015*10**(-9)  #F
C_sp = 0.037*10**(-9) #F
n_gem = [14, 12, 10.5, 8.5, 7, 6, 4.5, 3]

C_kg, nu_pg, nu_mg =  np.genfromtxt("content/data_b.txt", unpack=True)
nu_pg = nu_pg*10**(3)
nu_mg = nu_mg*10**(3)
temp = [9.99*0.003, 8.00*0.003, 6.47*0.003, 5.02*0.003, 4.00*0.003, 3.00*0.003, 2.03*0.003, 1.01*0.003]
C_k = unp.uarray([9.99, 8.00, 6.47, 5.02, 4.00, 3.00, 2.03, 1.01],temp)
C_k = C_k*10**(-9)
#Berechnung nu+
nu_p = 1/(2*np.pi*unp.sqrt(L*(C+C_sp)))
nu_m = 1/(2*np.pi*unp.sqrt(L*(((1/C)+(2/C_k))**(-1)+C_sp)))
nu_s = (nu_m-nu_p)
n = (nu_p + nu_m)/(2*(nu_m - nu_p))

nu_pabw = np.abs(nu_p-nu_pg)/nu_p
nu_mabw = np.abs(nu_m-nu_mg)/nu_m
n_abw = np.abs(n-n_gem)/n

def linear(x,a,b):
    return a*x+b

a = (49.4-19.7)/(11)
b = 19.7
n_p = linear(4.4,a,b)*10**(3)
t_m = np.array([5.2,5.6,5.8,6.0,6.2,7.0,8.0])
n_m = linear(t_m,a,b)*10**(3)

n_pabw = np.abs(n_p - nu_p)/nu_p
n_mabw = np.abs(n_m - nu_m[0:7])/nu_m[0:7]
print(n_mabw, n_pabw)