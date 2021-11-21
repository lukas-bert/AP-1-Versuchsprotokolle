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
# Erster Subplot
plt.subplot(1, 2, 1)
plt.plot(w, Theorie_c(2*np.pi*1000*w, R, L, C))
plt.plot(f, U_x, 'rx')

plt.xlabel("$U_C \mathbin{/} U$")
plt.ylabel("\nu \mathbin{/} \unit{\kilo\herz}")
plt.grid(True, which="both", ls="-")
plt.legend(loc='best')
plt.xscale('log')
plt.ylim(0, 5)
plt.plot(26.55,3.92013543016825 , 'go')

#Zweiter Subplot
plt.subplot(1, 2, 2)
plt.plot(w, Theorie_c(2*np.pi*1000*w, R, L, C))
plt.plot(f, U_x, 'rx')

plt.xlabel("$U_C \mathbin{/} U$")
plt.ylabel("\nu \mathbin{/} \unit{\kilo\herz}")
plt.grid(True, which="both", ls="-")
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='best')
plt.ylim(0, 5)
plt.plot(26.55,3.92013543016825 , 'go')

plt.tight_layout()

plt.savefig(build/PlotZuC.pdf)


