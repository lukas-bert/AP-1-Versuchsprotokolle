import matplotlib.pyplot as plt
import numpy as np
I_0 = 180*10**(-6)
I_dunkel = 62*10**(-9)
alpha_s, I_s = np.genfromtxt("content/data/s_pol.txt", unpack = True)
alpha_p, I_p = np.genfromtxt("content/data/p_pol.txt", unpack = True)
I_s = I_s*10**(-6)
I_p = I_p*10**(-6)
n_s = np.sqrt(1 + ((4*np.sqrt(I_s/I_0)*(np.cos(alpha_s))**2)/((np.sqrt(I_s/I_0) - 1)**2)))
print((n_s))
plt.plot(alpha_s,np.sqrt(I_s/I_0))
plt.show()
I_Pa = I_p[0:37]
I_Pb = I_p[37:]
alpha_pa = alpha_p[0:37]
alpha_pb = alpha_p[37:]
