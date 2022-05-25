import matplotlib.pyplot as plt
import numpy as np
I_0 = 180*10**(-6)
I_dunkel = 62*10**(-9)
n_real = 3.35268

print("Theorie: " , n_real)

alpha_s, I_s = np.genfromtxt("content/data/s_pol.txt", unpack = True)
alpha_p, I_p = np.genfromtxt("content/data/p_pol.txt", unpack = True)

I_s = I_s*10**(-6) - I_dunkel*10**(-9)
I_p = I_p*10**(-6) - I_dunkel*10**(-9)

n_s = np.sqrt(1 + ((4*np.sqrt(I_s/I_0)*(np.cos(alpha_s*np.pi/180))**2)/((np.sqrt(I_s/I_0) - 1)**2)))

#print((n_s))
print("s-polarisiert: ", np.mean(n_s))

I_Pa = I_p[0:37]
I_Pb = I_p[37:]

alpha_pa = alpha_p[0:37]
alpha_pb = alpha_p[37:]

# Brechungsindex n aus Brewster Winkel

# Brewster-Winkel bei Minimum von I_p --> 75°

n_brewster = np.tan(75*np.pi/180)
print("Brewster: ", n_brewster)

# Klappt nicht amk
def n_spol(a, E):                                                                       # E = E_r/E_e = sqrt(I_r/I_0)
    return np.sqrt(((E-1)/(E+1))**2*np.cos(a)**2 + np.sin(a)**2)

#n_s = n_spol(alpha_s*np.pi/180, np.sqrt(I_s/I_0))
#print(n_s)

def n_ppol(a, E):
    b = ((E+1)/(E-1))**2
    return np.sqrt(b/(2*np.cos(a)**2) + np.sqrt(b**2/(4*np.cos(a)**4) - b*np.tan(a)**2))

n_p = n_ppol(alpha_p*np.pi/180, np.sqrt(I_p/I_0))
#print(n_p[:-3])  
print("s-polarisiert: ", np.mean(n_p[:-3]))  

############################ Plots ###################################

n_s = np.mean(n_s)
n_p = np.mean(n_p)

def KurveS(a, n):
    return -(np.cos(a*np.pi/180) - np.sqrt(n**2-np.sin(a*np.pi/180)**2)) / (np.cos(a*np.pi/180) + np.sqrt(n**2-np.sin(a*np.pi/180)**2))

def KurveP(a, n):
    return (n**2*np.cos(a*np.pi/180) - np.sqrt(n**2-np.sin(a*np.pi/180)**2)) / (n**2*np.cos(a*np.pi/180) + np.sqrt(n**2-np.sin(a*np.pi/180)**2))   

alpha_B = 75 # 73.390
alpha = np.linspace(0, 90, 1000)
alpha_1 = np.linspace(0, alpha_B, 1000)
alpha_2 = np.linspace(alpha_B, 90, 1000)

plt.plot(alpha_s, np.sqrt(I_s/I_0), marker = "x", color = "firebrick", linewidth = 0, label = "Messwerte s-polarisiert")
plt.plot(alpha_p, np.sqrt(I_p/I_0), marker = "x", color = "coral", linewidth = 0, label = "Messwerte p-polarisiert")

plt.plot(alpha, KurveS(alpha, 3.7), color = "cornflowerblue", label = "Theoriekurve s-polarisiert")
plt.plot(alpha_1, KurveP(alpha_1, 3.7), color = "forestgreen", label = "Theoriekurve p-polarisiert")
plt.plot(alpha_2, -KurveP(alpha_2, 3.7), color = "forestgreen")

plt.grid()
plt.xlabel(r"$alpha / °$")
plt.ylabel(r"$I / I_0$")
plt.xlim(0, 90)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("build/plot.pdf")
plt.show()      
