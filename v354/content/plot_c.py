import matplotlib.pyplot as plt
import numpy as np

f, U_C, U, a, b = np.genfromtxt("data_c_d.txt", unpack = True)
phi, U_x = np.genfromtxt("Messwerte_cd.txt", unpack = True)
R = 732
L = 16.87*10**-3
C = 2.060 *10**-9

def Theorie_c(w, R, L, C):
    return (1/np.sqrt((1-L*C*w**2)**2+w**2*R**2*C**2))

print(R/(2*L))
print(np.sqrt(L*C)*1/(R*C)) 
print(np.sqrt(1/(L*C)-R**2/(2*L**2))/(2*np.pi*1000))  

w = np.linspace(10, 70, 1000)
plt.plot(w, Theorie_c(2*np.pi*1000*w, R, L, C))
plt.plot(f, U_x)

plt.plot(26.55,3.92013543016825 , 'rx')

plt.xscale('log')
plt.ylim(0, 7)
plt.show()